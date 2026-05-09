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
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import pickle
import warnings

import monai
from monai.transforms import Compose
from monai.data import *
from monai.utils import set_determinism
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from tqdm import tqdm

from utils.utils import *


# Set the random seed for reproducibility
set_determinism(seed=0)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*qfac.*")
warnings.filterwarnings("ignore", message=".*pixdim.*")
logging.getLogger("nibabel").setLevel(logging.ERROR)


def _tag_dataset(items, name):
    tagged = []
    for it in items:
        if isinstance(it, dict):
            tagged.append({"image": it["image"], "dataset": name})
        else:
            tagged.append({"image": it, "dataset": name})
    return tagged


def get_datalist(args):

    path = './'

    splits1 = "/btcv.json"
    splits2 = "/MM-WHS.json"
    splits3 = "/spleen.json"
    splits4 = "/dataset_TCIAcovid19_0.json"
    splits5 = "/dataset_LUNA16_0.json"
    splits6 = "/stoic21.json"
    splits7 = "/flare23.json"
    splits8 = "/LIDC.json"
    splits9 = "/HNSCC.json"
    splits10 = "/Totalsegmentator_dataset.json"

    list_dir = path + "jsons/"
    jsonlist1 = list_dir + splits1
    jsonlist2 = list_dir + splits2
    jsonlist3 = list_dir + splits3
    jsonlist4 = list_dir + splits4
    jsonlist5 = list_dir + splits5
    jsonlist6 = list_dir + splits6
    jsonlist7 = list_dir + splits7
    jsonlist8 = list_dir + splits8
    jsonlist9 = list_dir + splits9
    jsonlist10 = list_dir + splits10

    datadir1 = path + "data/BTCV"
    datadir2 = path + "data/MM-WHS"
    datadir3 = path + "data/Dataset009_Spleen"
    datadir4 = path + "data/TCIAcovid19"
    datadir5 = path + "data/Luna16-jx"
    datadir6 = path + "data/stoic21"
    datadir7 = path + "data/Flare23"
    datadir8 = path + "data/LIDC"
    datadir9 = path + "data/HNSCC_convert_v1"
    datadir10 = path + "data/Totalsegmentator_dataset"

    # num_workers = args.num_workers
    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    print("Dataset 1 BTCV: number of data: {}".format(len(datalist1)))
    new_datalist1 = []
    for item in datalist1:
        item_dict = {"image": item["image"],
                     "dataset": "BTCV"}
        new_datalist1.append(item_dict)

    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    print("Dataset 2 MM-WHS: number of data: {}".format(len(datalist2)))
    for item in datalist2:
        item["dataset"] = "MM-WHS"

    datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
    print("Dataset 3 Spleen: number of data: {}".format(len(datalist3)))
    for item in datalist3:
        item["dataset"] = "Spleen"

    datalist4 = load_decathlon_datalist(jsonlist4, False, "training", base_dir=datadir4)
    print("Dataset 4 Covid 19: number of data: {}".format(len(datalist4)))
    for item in datalist4:
        item["dataset"] = "Covid 19"

    datalist5 = load_decathlon_datalist(jsonlist5, False, "training", base_dir=datadir5)
    print("Dataset 5 Luna: number of data: {}".format(len(datalist5)))
    new_datalist5 = []
    for item in datalist5:
        item_dict = {"image": item["image"],
                     "dataset": "Luna"}
        new_datalist5.append(item_dict)

    datalist6 = load_decathlon_datalist(jsonlist6, False, "training", base_dir=datadir6)
    print("Dataset 6 Stoic: number of data: {}".format(len(datalist6)))
    for item in datalist6:
        item["dataset"] = "Stoic"

    datalist7 = load_decathlon_datalist(jsonlist7, False, "training", base_dir=datadir7)
    print("Dataset 7 Flare23: number of data: {}".format(len(datalist7)))
    for item in datalist7:
        item["dataset"] = "Flare23"

    datalist8 = load_decathlon_datalist(jsonlist8, False, "training", base_dir=datadir8)
    print("Dataset 8 LIDC: number of data: {}".format(len(datalist8)))
    for item in datalist8:
        item["dataset"] = "LIDC"

    datalist9 = load_decathlon_datalist(jsonlist9, False, "training", base_dir=datadir9)
    print("Dataset 9 HNSCC: number of data: {}".format(len(datalist9)))
    for item in datalist9:
        item["dataset"] = "HNSCC"

    datalist10 = load_decathlon_datalist(jsonlist10, False, "training", base_dir=datadir10)
    print("Dataset 10 Totalsegmentator: number of data: {}".format(len(datalist10)))
    for item in datalist10:
        item["dataset"] = "Totalsegmentator"

    vallist1 = load_decathlon_datalist(jsonlist1, False, "validation", base_dir=datadir1)
    vallist2 = load_decathlon_datalist(jsonlist2, False, "validation", base_dir=datadir2)
    vallist3 = load_decathlon_datalist(jsonlist3, False, "validation", base_dir=datadir3)
    vallist4 = load_decathlon_datalist(jsonlist4, False, "validation", base_dir=datadir4)
    vallist5 = load_decathlon_datalist(jsonlist5, False, "validation", base_dir=datadir5)


    datalist = new_datalist1 + datalist2 + datalist3 + datalist4 + new_datalist5 + datalist6 + datalist7 + datalist8 + datalist9 + datalist10
    val_files = vallist1 + vallist2 + vallist3 + vallist4 + vallist5

    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(val_files)))

    return datalist


def create_transforms(args, dim: tuple = None) -> Compose:
    if dim:
        return Compose(
            [
                monai.transforms.LoadImaged(keys="image"),
                monai.transforms.EnsureChannelFirstd(keys="image"),
                monai.transforms.Orientationd(keys="image", axcodes="RAS"),
                monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
                monai.transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                monai.transforms.Spacingd(keys="image", pixdim=args.spacing, mode="bilinear"),
                monai.transforms.DivisiblePadd(keys="image", k=4)
            ]
        )
    else:
        return Compose(
            [
                monai.transforms.LoadImaged(keys="image"),
                monai.transforms.EnsureChannelFirstd(keys="image"),
                monai.transforms.Orientationd(keys="image", axcodes="RAS"),
            ]
        )


def round_number(number: int, base_number: int = 128) -> int:
    new_number = max(round(float(number) / float(base_number)), 1.0) * float(base_number)
    return int(new_number)


def load_filenames(data_list_path: str) -> list:
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    filenames_raw = json_data["training"]
    return [_item["image"] for _item in filenames_raw]


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
def diff_model_create_training_data(args, datalist) -> None:

    init_distributed_mode(args)

    logger = create_logger(args.log_dir, args.distributed)

    class _RankFilter(logging.Filter):
        def __init__(self, rank: int):
            super().__init__()
            self._rank = rank

        def filter(self, record: logging.LogRecord) -> bool:
            record.rank = self._rank
            return True

    def _prefix_rank_on_handlers(logger: logging.Logger, rank: int):
        for h in logger.handlers:
            h.addFilter(_RankFilter(rank))
            h.setFormatter(logging.Formatter(
                "[rank %(rank)s] %(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))

    _prefix_rank_on_handlers(logger, args.rank)

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

    filenames_raw = [it["image"] if isinstance(it, dict) else it for it in datalist]
    dataset_raw = [it["dataset"] if isinstance(it, dict) else it for it in datalist]

    plain_transforms = create_transforms(args, dim=None)

    progress_bar = tqdm(enumerate(filenames_raw), total=len(filenames_raw), ncols=100)

    for _iter, _ in progress_bar:
        if _iter % args.world_size != args.rank:
            continue

        filepath = filenames_raw[_iter]
        dataset = dataset_raw[_iter]
        new_dim = args.volume_dim
        new_transforms = create_transforms(args, new_dim)

        process_file(_iter, filepath, dataset, args, autoencoder, args.device, plain_transforms, new_transforms, logger)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training Latent Data Creation")

    parser.add_argument('--distributed', default=True, action='store_true', help='distributed training')
    parser.add_argument("--gpu_ids", default=[0, 1, 2, 3], help="local rank")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")

    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--spacing", default=[1.5, 1.5, 1.5])
    parser.add_argument("--volume_dim", default=[512, 512, 128])

    args = parser.parse_args()

    args.log_dir = '../result/latent_creation/log/'
    args.data_dir = '../data/'
    args.latent_dir = '../data/latent/'
    args.ae_dict = '../AutoEncoder/autoencoder.pt'

    datalist = get_datalist(args)
    diff_model_create_training_data(args, datalist)
