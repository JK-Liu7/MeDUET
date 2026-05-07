# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import logging
import os
import nibabel as nib
import numpy as np
import torch
import warnings

import monai
from monai.transforms import Compose
from monai.utils import set_determinism
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from tqdm import tqdm

from util.utils import *
from util.misc import *
from util.data_utils_latent import get_loader


# Set the random seed for reproducibility
set_determinism(seed=0)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*qfac.*")
warnings.filterwarnings("ignore", message=".*pixdim.*")
logging.getLogger("nibabel").setLevel(logging.ERROR)



def process_file(
    i: int,
    filepath: str,
    dataset: str,
    args: argparse.Namespace,
    autoencoder: torch.nn.Module,
    device: torch.device,
    plain_transforms: Compose,
    new_transforms: Compose,
    logger: logging.Logger,
) -> None:
    """
    Process a single file to create training data.

    Args:
        filepath (str): Path to the file to be processed.
        args (argparse.Namespace): Configuration arguments.
        autoencoder (torch.nn.Module): Autoencoder model.
        device (torch.device): Device to process the file on.
        plain_transforms (Compose): Plain transforms.
        new_transforms (Compose): New transforms.
        logger (logging.Logger): Logger for logging information.
    """
    # out_filename_base = filepath.replace(".gz", "").replace(".nii", "")
    out_filename_base = os.path.join(args.latent_dir, dataset + "_" + str(i))
    out_filename = out_filename_base + "_latent.nii.gz"

    if os.path.isfile(out_filename):
        return

    test_data = {"image": os.path.join(args.data_dir, filepath)}
    transformed_data = plain_transforms(test_data)
    nda = transformed_data["image"]

    dim = [int(nda.meta["dim"][_i]) for _i in range(1, 4)]
    spacing = [float(nda.meta["pixdim"][_i]) for _i in range(1, 4)]

    logger.info(f"old dim: {dim}, old spacing: {spacing}")

    new_data = new_transforms(test_data)
    nda_image = new_data["image"]

    new_affine = nda_image.meta["affine"].numpy()
    nda_image = nda_image.numpy().squeeze()

    logger.info(f"new dim: {nda_image.shape}")

    try:
        out_path = Path(out_filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"out_filename: {out_filename}")

        with torch.amp.autocast("cuda"):
            pt_nda = torch.from_numpy(nda_image).float().to(device).unsqueeze(0).unsqueeze(0)
            z = autoencoder.encode_stage_2_inputs(pt_nda)
            logger.info(f"z: {z.size()}, {z.dtype}")

            out_nda = z.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)
            out_img = nib.Nifti1Image(np.float32(out_nda), affine=new_affine)
            nib.save(out_img, out_filename)
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")


@torch.inference_mode()
def diff_model_create_training_data(args) -> None:
    """
    Create training data for pretraining.
    """

    init_distributed_mode(args)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    logger = create_logger(args.log_dir, args.distributed)

    loader, sampler = get_loader(args)

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

    Path(args.latent_dir).mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(enumerate(loader), total=len(loader), ncols=100)
    sampler.set_epoch(0)

    for i, batch in progress_bar:

        x = batch["image"].to(args.device)
        domain = batch["domain"].cpu().numpy()
        voxel = batch["orig_spacing"].cpu().numpy()
        body_part = batch["body_part_idx"].cpu().numpy()

        dataset = batch["dataset"][0]

        if args.rank == 0:
            print(x.shape)

        i_global = i * args.world_size + args.rank
        out_filename_base = os.path.join(args.latent_dir, f"{dataset}_{i_global}")
        out_filename = out_filename_base + "_latent.nii.gz"

        if os.path.isfile(out_filename):
            continue

        try:
            with torch.amp.autocast("cuda"):
                z = autoencoder.encode_stage_2_inputs(x)
                logger.info(f"[rank {args.rank}], z: {z.size()}, {z.dtype}")

                out_nda = z.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)
                out_img = nib.Nifti1Image(np.float32(out_nda), affine=np.eye(4))
                nib.save(out_img, out_filename)

                np.save(out_filename_base + "_domain", domain)
                np.save(out_filename_base + "_voxel", voxel)
                np.save(out_filename_base + "_bp", body_part)

        except Exception as e:
            logger.error(f"Error processing {out_filename}: {e}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training Latent Data Creation")

    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU')
    parser.add_argument('--distributed', default=True, action='store_true', help='distributed training')
    parser.add_argument("--gpu_ids", default=[0, 1, 2, 3], help="local rank")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")

    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--spacing", default=[1.5, 1.5, 1.5])
    parser.add_argument("--roi_size", default=[512, 512, 768])
    parser.add_argument("--patch_size", default=[96, 96, 96])

    # Dataset parameters
    parser.add_argument('--cache', default=0.0, type=float)
    parser.add_argument('--replace_rate', default=0.2, type=float)
    parser.add_argument("--smartcache_dataset", default=False, help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", default=True, help="use monai cache Dataset")
    parser.add_argument('--num_workers', default=8, type=int)

    args = parser.parse_args()

    args.log_dir = '../result/pretrain/log/latent_creation/'
    args.data_dir = '../data/'
    args.latent_dir = '../data/latent_pretrain/'
    args.ae_dict = '../AutoEncoder/autoencoder.pt'

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.latent_dir, exist_ok=True)

    diff_model_create_training_data(args)
