# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import numpy as np
import scipy.ndimage as ndimage
import torch
from datetime import datetime
from time import time
import logging
import torch.distributed as dist



def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


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


def load_UNETR(model, ckpt_path, load_pos_embed=True):
    model_dict = torch.load(ckpt_path)['model']
    dst = model.state_dict()
    pretrained_dict = {}
    pretrained_dict["vit.patch_embedding.patch_embeddings.weight"] = model_dict["embed_layer.proj.weight"]
    pretrained_dict["vit.patch_embedding.patch_embeddings.bias"] = model_dict["embed_layer.proj.bias"]

    if load_pos_embed and ('pos_embed' in model_dict) and ('pos_embed' in dst):
        pos = model_dict['pos_embed']  # [1, 1+L, D]
        pos_tokens = pos[:, 1:, :]  # drop cls
        D = pos_tokens.shape[-1]
        L = pos_tokens.shape[1]

        g = int(round(L ** (1 / 3)))
        assert g * g * g == L

        # [1, D, g, g, g]
        pos_tokens = pos_tokens.reshape(1, g, g, g, D).permute(0, 4, 1, 2, 3)

        # p = model.patch_size
        # gx, gy, gz = model.input_size // p, model.input_size // p, model.input_depth // p
        gx, gy, gz = model.feat_size

        if (g, g, g) != (gx, gy, gz):
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(gx, gy, gz), mode='trilinear', align_corners=False)

        # [1, gx*gy*gz, D]
        pos_tokens = pos_tokens.permute(0, 2, 3, 4, 1).reshape(1, gx * gy * gz, D)

        if pos_tokens.shape == dst['pos_embed'].shape:
            pretrained_dict['vit.patch_embedding.position_embeddings'] = pos_tokens

    rename = {}
    for k in list(model_dict.keys()):
        if k.startswith(("decoder_", "decoder.", "mask_token", "decoder_pos_embed", "cls_token")):
            continue

        new_k = None
        # Transformer blocks
        if k.startswith("blocks."):
            parts = k.split(".")
            blk_id = parts[1]
            tail = ".".join(parts[2:])
            tail = tail.replace("attn.proj.", "attn.out_proj.")
            tail = tail.replace("mlp.fc1.", "mlp.linear1.")
            tail = tail.replace("mlp.fc2.", "mlp.linear2.")
            new_k = f"vit.blocks.{blk_id}.{tail}"

        # final norm
        elif k == "norm.weight":
            new_k = "vit.norm.weight"
        elif k == "norm.bias":
            new_k = "vit.norm.bias"

        if new_k is not None:
            rename[k] = new_k

    mapped = {rename[k]: model_dict[k] for k in rename.keys()}
    unetr_sd = model.state_dict()

    loaded_keys = set()
    shape_mismatch = []
    missing_in_unetr = []
    validated_initial_skipped = []

    for k in list(pretrained_dict.keys()):
        if (k not in unetr_sd) or (pretrained_dict[k].shape != unetr_sd[k].shape):
            validated_initial_skipped.append((k,
                                              tuple(pretrained_dict[k].shape),
                                              tuple(unetr_sd[k].shape) if k in unetr_sd else None))
            pretrained_dict.pop(k)
        else:
            loaded_keys.add(k)

    for k, v in mapped.items():
        if k in unetr_sd:
            if unetr_sd[k].shape == v.shape:
                pretrained_dict[k] = v
                loaded_keys.add(k)
            else:
                shape_mismatch.append((k, tuple(v.shape), tuple(unetr_sd[k].shape)))
        else:
            missing_in_unetr.append(k)

    def _numel_of(keys, ref_sd):
        return sum(ref_sd[k].numel() for k in keys if k in ref_sd)

    vit_total_params = sum(p.numel() for name, p in unetr_sd.items() if name.startswith("vit."))

    params_loaded = _numel_of(loaded_keys, unetr_sd)
    tensors_loaded = len(loaded_keys)
    coverage = (params_loaded / vit_total_params) if vit_total_params > 0 else 0.0

    msg = model.load_state_dict(pretrained_dict, strict=False)
    print('Pretrained Medusa weights loaded!')

    def _fmt(n):
        return f"{n / 1e6:.3f}M" if n >= 1e6 else str(n)

    print(
        f"[LOAD] tensors: {tensors_loaded}, params: {_fmt(params_loaded)} / vit total: {_fmt(vit_total_params)} ({coverage * 100:.2f}%)")
    print(
        f"[SKIP] shape_mismatch: {len(shape_mismatch) + len(validated_initial_skipped)}, missing_in_unetr: {len(missing_in_unetr)}")

    return model
