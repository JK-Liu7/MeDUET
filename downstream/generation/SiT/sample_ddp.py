# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from models import SiT_models
from transport import create_transport, Sampler
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from utils import *
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import nibabel as nib
import math
import argparse
import sys
from datetime import timedelta, datetime
from concurrent.futures import ThreadPoolExecutor


def create_npz_from_sample_folder(sample_dir, num=1000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], samples.shape[3])
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def sample_cf_guidance(args, content_files, style_files, n,
                       dtype=torch.float32, num_workers=0, replace=True):

    rng = np.random.default_rng()

    c_sel = rng.choice(content_files, size=n, replace=replace)
    s_sel = rng.choice(style_files,  size=n, replace=replace)

    c_paths = [Path(args.factor_dir) / 'content' / f for f in c_sel]
    s_paths = [Path(args.factor_dir) / 'style'   / f for f in s_sel]

    def load_and_reduce(p):
        arr = np.load(p)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return torch.from_numpy(np.ascontiguousarray(arr))

    if num_workers and num_workers > 0:
        with ThreadPoolExecutor(max_workers=min(num_workers, n*2)) as ex:
            c_list = list(ex.map(load_and_reduce, c_paths))
            s_list = list(ex.map(load_and_reduce, s_paths))
    else:
        c_list = [load_and_reduce(p) for p in c_paths]
        s_list = [load_and_reduce(p) for p in s_paths]

    c = torch.stack(c_list, dim=0).to(device=args.device, dtype=dtype)
    s = torch.stack(s_list, dim=0).to(device=args.device, dtype=dtype)

    return c, s



def main(args):
    """
    Run sampling.
    """

    init_distributed_mode(args)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = args.tf32

    # define MAISI_VAE
    autoencoder = AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels = 4,
        num_channels = (64, 128, 256),
        num_res_blocks = (2, 2, 2),
        norm_num_groups = 32,
        norm_eps=1e-06,
        attention_levels=(False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
        norm_float16=True,
        num_splits=8,
        dim_split=1
    )
    state_dict = torch.load(str(args.ae_dict))
    autoencoder.load_state_dict(state_dict)
    if args.rank == 0:
        print('MAISI VAE weighted loaded!')

    autoencoder = autoencoder.to(args.device)
    autoencoder.eval()
    autoencoder.requires_grad_(False)

    # Create SiT model:
    model = SiT_models[args.model](
        input_size=args.latent_size[0],
        input_depth=args.latent_size[-1],
    )

    ckpt_path = args.model_ckpt

    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict["ema"])
    model = model.to(args.device)
    model.eval()  # important!
    model.requires_grad_(False)
    if args.rank == 0:
        print('SiT weighted loaded!')
        print(f"Inference from checkpoint: {ckpt_path}")

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    if args.mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )
    elif args.mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )

    assert args.cfg_scale_c >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale_c > 1.0

    content_files = [f for f in os.listdir(args.factor_dir + 'content/') if f.endswith('.npy')]
    style_files = [f for f in os.listdir(args.factor_dir + 'style/') if f.endswith('.npy')]

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if args.rank == 0:
        print(f"Total number of volumes that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)

    pbar = range(iterations)
    pbar = tqdm(pbar, ncols=100) if args.rank == 0 else pbar
    total = 0
    
    for item in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, args.latent_size[0], args.latent_size[0], args.latent_size[-1], device=args.device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z, z], 0)
            c, s = sample_cf_guidance(args, content_files, style_files, n, dtype=torch.float32, num_workers=args.num_workers)
            model_kwargs = dict(content=c, style=s, cfg_scale_c=args.cfg_scale_c, cfg_scale_s=args.cfg_scale_s)
            model_fn = model.forward_with_cfg
        else:
            c, s = None, None
            model_kwargs = dict(content=c, style=s)
            model_fn = model.forward

        samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            
        if using_cfg:
            samples, _, _ = samples.chunk(3, dim=0)  # Remove null class samples

        with autocast(enabled=args.amp):
            samples = autoencoder.decode_stage_2_outputs(samples / args.scale_factor).detach().cpu().numpy()

        # Save samples to disk as individual .nii.gz files
        for j in range(n):
            index = j * dist.get_world_size() + args.rank + total
            out_img = nib.Nifti1Image(np.float32(samples[j].squeeze(0)), np.eye(4))
            out_filename = args.sample_dir + str(index) + ".nii.gz"
            nib.save(out_img, out_filename)

        total += global_batch_size
        dist.barrier()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="SDE", choices=["ODE", "SDE"])
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/4")
    parser.add_argument("--per-proc-batch-size", type=int, default=4)
    parser.add_argument("--num-fid-samples", type=int, default=1000)
    parser.add_argument("--image-size", default=[256, 256, 128])
    parser.add_argument('--latent_size', default=[64, 64, 32], help='images input size')

    parser.add_argument("--scale_factor", type=float, default=1.0658544301986694)
    parser.add_argument("--cfg-scale_c",  type=float, default=3.0)
    parser.add_argument("--cfg-scale_s", type=float, default=3.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=2025)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--model_ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")

    # distributed training parameters
    parser.add_argument('--distributed', default=True, action='store_true', help='distributed training')
    parser.add_argument("--gpu_ids", default=[0, 1, 2, 3], help="local rank")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument('--num_workers', default=8, type=int)

    # enable amp
    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=True)

    parse_transport_args(parser)

    mode = "SDE"
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE

    args = parser.parse_args()

    args.log_dir = '../result/downstream/log/generation/SiT/'
    args.latent_dir = '../data/latent/'
    args.factor_dir = '../factor/'
    args.cache_dir = '../data/cache/downstream/generation/'
    args.ae_dict = '../AutoEncoder/autoencoder.pt'
    args.sample_dir = '../result/downstream/generation/samples/SiT/'

    main(args)
