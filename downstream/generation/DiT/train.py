# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
from timm.scheduler.cosine_lr import CosineLRScheduler
from models import DiT_models
from data_utils_factor import *
from utils import *
from diffusion import create_diffusion
from tqdm import tqdm



#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    init_distributed_mode(args)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    logger = create_logger(args.log_dir, args.distributed)

    # Create model:
    model = DiT_models[args.model](
        input_size=args.latent_size[0],
        input_depth=args.latent_size[-1],
    )

    if args.finetune:
        pretrained_path = args.pretrain_dir + '/MeDUET.pth'
        print("Load pre-trained checkpoint from: %s" % pretrained_path)
        model = load_pretrained_from_MeDUET(model, pretrained_path)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(args.device)  # Create an EMA of the model for use after training

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        print("Load resume checkpoint from: %s" % args.ckpt)
        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model, strict=True)
        ema.load_state_dict(checkpoint["ema"], strict=True)
        print("Succesfully load EMA model from: %s" % args.ckpt)

    requires_grad(ema, False)

    model = DDP(model.to(args.device), device_ids=[args.rank], find_unused_parameters=True)

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineLRScheduler(opt, warmup_t=args.warmup_epochs, warmup_lr_init=1e-6, t_initial=args.epochs,
                                  lr_min=args.min_lr, cycle_limit=1)

    if args.ckpt:
        opt.load_state_dict(checkpoint["opt"])
        print("Succesfully load opt from: %s" % args.ckpt)

    # Setup data:
    train_loader, train_sampler = get_loader(args)

    scale_factor = calculate_scale_factor(train_loader, args.device, logger)

    # Prepare models for training:
    if args.distributed:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    else:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0 if not args.ckpt else int(args.ckpt.split('/')[-1].split('.')[0]) # xxx/0300000.pt
    log_steps = 0
    running_loss = 0
    start_time = time()

    scaler = GradScaler()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
        progress_bar.set_description(f"Epoch {epoch}")

        for i, batch in progress_bar:
            x = batch["latent"].to(args.device)
            c = batch["content"].to(args.device)
            s = batch["style"].to(args.device)
            x = x * scale_factor

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=args.device)
            model_kwargs = dict(content=c, style=s)

            with autocast(enabled=args.amp):
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            opt.zero_grad(set_to_none=True)

            if args.distributed:
                update_ema(ema, model.module)
            else:
                update_ema(ema, model)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=args.device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(Step={train_steps:08d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Lr: {opt.param_groups[0]['lr']:.6f}")

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if (train_steps % args.ckpt_every == 0 or train_steps in [50000]) and train_steps > 0:
                checkpoint_path = f"{args.model_dir}/{train_steps:07d}.pt"
                if args.rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

        lr_scheduler.step(epoch)

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DiT training', add_help=False)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/4")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU')
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--global_seed", type=int, default=2025)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=20000)

    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a custom DiT checkpoint")

    # distributed training parameters
    parser.add_argument('--distributed', default=True, action='store_true', help='distributed training')
    parser.add_argument("--gpu_ids", default=[0, 1, 2, 3], help="local rank")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")

    # enable amp
    parser.add_argument('--amp', default=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='epochs to warmup LR')

    # Model parameters
    parser.add_argument('--compression_ratio', default=4, type=int, help='compression ratio of pretrained VAE')
    parser.add_argument("--patch_size", default=[256, 256, 128])
    parser.add_argument('--latent_size', default=[64, 64, 32], help='images input size')
    parser.add_argument("--s_ratio", default=0.25, type=float)


    # Dataset parameters
    parser.add_argument('--cache', default=1.0, type=float)
    parser.add_argument('--replace_rate', default=0.2, type=float)
    parser.add_argument("--smartcache_dataset", default=False, help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", default=False, help="use monai cache Dataset")
    parser.add_argument('--num_workers', default=8, type=int)
    
    # Finetune
    parser.add_argument('--finetune', default='True', help='finetune from checkpoint')

    args = parser.parse_args()

    args = parser.parse_args()

    args.log_dir = '../result/downstream/log/generation/DiT/'
    args.latent_dir = '../data/latent/'
    args.factor_dir = '../data/factor/'
    args.cache_dir = '../data/cache/downstream/generation/'
    args.pretrain_dir = '../model_save/pretrain/MeDUET/'
    args.ae_dict = '../AutoEncoder/autoencoder.pt'
    args.model_dir = '../model_save/downstream/generation/DiT/'
    args.sample_dir = '../result/downstream/generation/samples/DiT/'

    main(args)
