import os
import numpy as np
import scipy.ndimage as ndimage
import torch
from datetime import datetime
import logging
import torch.distributed as dist
from monai.data import DataLoader
from monai.utils import first


def init_distributed_mode(args):
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.world_size = 1
    args.rank = 0

    if args.distributed:
        dist.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.device = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.device)
        num_gpus = torch.cuda.device_count()
        print(f"Setting up distributed training with {num_gpus} GPUs available")
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        args.device = torch.device("cuda:0")
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0


def create_logger(log_dir, distributed):
    """
    Create a logger that writes to a log file and stdout.
    """
    today_date = datetime.today().strftime('%Y.%m.%d')
    if distributed:
        if dist.get_rank() == 0:  # real logger
            logging.basicConfig(filename=log_dir + f"{today_date}.log",
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
            logger = logging.getLogger(__name__)
        else:  # dummy logger (does nothing)
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
    else:
        logging.basicConfig(filename=log_dir + f"{today_date}.log",
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
        logger = logging.getLogger(__name__)

    return logger


def load_pretrained_from_MeDUET(model, ckpt_path, load_pos_embed=True):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    src = ckpt.get('model', ckpt)
    dst = model.state_dict()
    new_sd = {}

    for k in ['weight', 'bias']:
        sk = f'embed_layer.proj.{k}'
        dk = f'x_embedder.proj.{k}'
        if sk in src and dk in dst and src[sk].shape == dst[dk].shape:
            new_sd[dk] = src[sk]

    if load_pos_embed and ('pos_embed' in src) and ('pos_embed' in dst):
        pos = src['pos_embed']
        pos_tokens = pos[:, 1:, :]
        D = pos_tokens.shape[-1]
        L = pos_tokens.shape[1]

        g = int(round(L ** (1 / 3)))
        assert g * g * g == L

        pos_tokens = pos_tokens.reshape(1, g, g, g, D).permute(0, 4, 1, 2, 3)

        p = model.patch_size
        gx, gy, gz = model.input_size // p, model.input_size // p, model.input_depth // p

        if (g, g, g) != (gx, gy, gz):
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(gx, gy, gz), mode='trilinear', align_corners=False)

        pos_tokens = pos_tokens.permute(0, 2, 3, 4, 1).reshape(1, gx * gy * gz, D)

        if pos_tokens.shape == dst['pos_embed'].shape:
            new_sd['pos_embed'] = pos_tokens

    def want(k: str):
        if not k.startswith('blocks.'):
            return False
        if '.norm' in k:
            return False
        if k.startswith('decoder') or 'decoder_' in k or 'mask_token' in k:
            return False
        return ('.attn.' in k) or ('.mlp.' in k)

    for k, v in src.items():
        if want(k) and (k in dst) and (dst[k].shape == v.shape):
            new_sd[k] = v

    loadable = {k: v for k, v in new_sd.items() if (k in dst and v.shape == dst[k].shape)}

    msg = model.load_state_dict(loadable, strict=False)

    total_params = sum(p.numel() for p in dst.values())
    loaded_params = sum(dst[k].numel() for k in loadable.keys())
    tensors_loaded = len(loadable)
    cov = 100.0 * loaded_params / max(total_params, 1)

    print('Pretrained MeDUET → SiT: loaded tensors:', tensors_loaded)
    print(f'Params loaded: {loaded_params / 1e6:.3f}M / {total_params / 1e6:.3f}M ({cov:.2f}%)')
    print('Missing keys (head/adaLN/decoders/norm,etc):', len(msg.missing_keys))
    print('Unexpected keys:', len(msg.unexpected_keys))
    return model


def calculate_scale_factor(train_loader: DataLoader, device: torch.device, logger: logging.Logger) -> torch.Tensor:
    """
    Calculate the scaling factor for the dataset.

    Args:
        train_loader (DataLoader): Data loader for training.
        device (torch.device): Device to use for calculation.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.Tensor: Calculated scaling factor.
    """
    check_data = first(train_loader)
    z = check_data["latent"].to(device)
    scale_factor = 1 / torch.std(z)
    logger.info(f"Scaling factor set to {scale_factor}.")

    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    return scale_factor


