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

import argparse
import os
import warnings
from functools import partial
import torch.nn.parallel
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel
from timm.scheduler.cosine_lr import CosineLRScheduler
from trainer import run_training

from utils.data_utils import get_loader
from utils.utils import *
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose, Lambda, EnsureType
from monai.utils.enums import MetricReduction
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from unetr import UNETR
from pretrain import models_MeDUET


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*qfac.*")
warnings.filterwarnings("ignore", message=".*pixdim.*")
logging.getLogger("nibabel").setLevel(logging.ERROR)


parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument("--dataset", default="Amos2022", help="dataset")
parser.add_argument("--data_ratio", default=1.0, help="training data ratio")
parser.add_argument("--model", default="MeDUET", help="model")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--json_list", default="dataset_CT.json", type=str, help="dataset json file")
parser.add_argument("--seed", default=2025, type=int, help="random seed")

parser.add_argument("--save_checkpoint", default=True, help="save checkpoint during training")
parser.add_argument("--max_epochs", default=1000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=32, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=2e-4, type=float, help="optimization learning rate")
parser.add_argument('--min_lr', type=float, default=2e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--weight_decay", default=2e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--amp", default=True, help="use amp for training")
parser.add_argument("--val_every", default=10, type=int, help="validation frequency")

parser.add_argument("--ckpt_interval", default=500, type=int, help="checkpoint saving frequency")
parser.add_argument('--distributed', default=False, action='store_true', help='distributed training')
parser.add_argument("--gpu_ids", default=[0, 1, 2, 3], help="local rank")
parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")


parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--feature_size", default=32, type=int, help="feature size")
parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--num_classes", default=16, type=int)
parser.add_argument("--use_normal_dataset", default=True, help="use monai Dataset class")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.75, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", default=False, action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", default=False, help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", default=True, help="use self-supervised pretrained weights")
parser.add_argument("--squared_dice", default=True, help="use squared Dice")

# MeDUET parameters
parser.add_argument('--MeDUET', default='MeDUET_vit_base', type=str, metavar='MODEL', help='Name of model to train')
parser.add_argument('--compression_ratio', default=4, type=int, help='compression ratio of pretrained VAE')
parser.add_argument('--latent_size', default=24, type=int, help='images input size')
parser.add_argument('--mask_ratio', default=0.5, type=float, help='Masking ratio')
parser.add_argument('--embed_dim', default=768, type=int, help='embedding dimension of MAE encoder')
parser.add_argument("--s_ratio", default=0.25, type=float)
parser.add_argument("--num_domain", default=10, type=int)
parser.add_argument("--num_roi", default=5, type=int)
parser.add_argument("--tau_c", default=0.1, type=float, help='temperature coefficient of infoNCE loss')
parser.add_argument("--tau_s", default=0.1, type=float, help='temperature coefficient of infoNCE loss')
parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
parser.set_defaults(norm_pix_loss=False)

# Hyperparameters
parser.add_argument('--k', type=int, default=32)
parser.add_argument('--p_aug', type=float, default=0.3)
parser.add_argument('--alpha', default=(0.2, 0.5))
parser.add_argument('--lambda_aug', type=float, default=0.1)
parser.add_argument('--ema_momentum', default=0.90, type=float)



def main():
    args = parser.parse_args()

    args.log_dir = '../result/downstream/log/' + args.dataset + '/' + args.model + '/'
    args.data_dir = '../data/' + args.dataset + '/'
    args.latent_dir = '../latent_downstream/' + args.dataset + '/'
    args.pretrained_dir = '../model_save/pretrain/' + args.model + '/'
    args.cache_dir = '../data/cache/downstream/' + args.dataset + '/'
    args.ae_dict = '../AutoEncoder/autoencoder.pt'

    main_worker(args)


def main_worker(args):
    init_distributed_mode(args)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    args.test_mode = False

    loader, train_dataset = get_loader(args)

    args.inf_size = [args.roi_x // 4, args.roi_y // 4, args.roi_z // 4]

    logger = create_logger(args.log_dir, args.distributed)

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

    model = UNETR(
        in_channels=args.in_channels,
        out_channels=args.num_classes,
        img_size=args.inf_size,
        feature_size=args.feature_size,
        qkv_bias=True,
    ).to(args.device)

    # Load pretrained MeDUET weights
    ckpt_path = args.pretrained_dir + '/MeDUET.pth'
    if args.use_ssl_pretrained:
        model = load_UNETR(model, ckpt_path)

    # define MeDUET model
    MeDUET = models_MeDUET.__dict__[args.MeDUET](args, norm_pix_loss=args.norm_pix_loss,
                                                 img_size=args.latent_size)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    src = ckpt.get('model', ckpt)
    dst = MeDUET.state_dict()
    MeDUET.load_state_dict(src, strict=False)

    total_params = sum(p.numel() for p in dst.values())
    loaded_params = sum(dst[k].numel() for k in src.keys())
    cov = 100.0 * loaded_params / max(total_params, 1)

    if args.rank == 0:
        print('Pretrained MeDUET weighted loaded!')
        print(f'Params loaded: {loaded_params / 1e6:.3f}M / {total_params / 1e6:.3f}M ({cov:.2f}%)')

    MeDUET = MeDUET.to(args.device)
    MeDUET.eval()
    MeDUET.requires_grad_(False)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.rank], find_unused_parameters=True)

    if args.squared_dice:
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr)
    else:
        dice_loss = DiceCELoss(include_background=False, to_onehot_y=False, softmax=True)

    post_label = AsDiscrete(to_onehot=args.num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes)

    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=(args.roi_x, args.roi_y, args.roi_z),
        sw_batch_size=args.sw_batch_size,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.weight_decay)

    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.weight_decay)

    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True,
                                    weight_decay=args.weight_decay)
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = CosineLRScheduler(optimizer, warmup_t=args.warmup_epochs, warmup_lr_init=1e-6,
                                      t_initial=args.max_epochs,
                                      lr_min=args.min_lr, cycle_limit=1)
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    accuracy = run_training(
        args=args,
        model=model,
        autoencoder=autoencoder,
        MeDUET=MeDUET,
        train_dataset=train_dataset,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_dice=dice_loss,
        acc_func=dice_acc,
        logger=logger,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
