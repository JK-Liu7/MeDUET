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
import warnings

import monai

from monai.utils import set_determinism
from tqdm import tqdm

from utils.utils import *
from data_utils_factor import *

from pretrain import models_MeDUET

# Set the random seed for reproducibility
set_determinism(seed=0)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*qfac.*")
warnings.filterwarnings("ignore", message=".*pixdim.*")
logging.getLogger("nibabel").setLevel(logging.ERROR)



@torch.inference_mode()
def diff_model_create_training_factor(args) -> None:
    """
    Create training factor data (content/style) for the diffusion model.
    """
    init_distributed_mode(args)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

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

    loader, sampler = get_loader(args)

    model = models_MeDUET.__dict__[args.model](args, norm_pix_loss=args.norm_pix_loss,
                                            img_size=args.input_size)

    pretrained_path = args.pretrained_dir + '/checkpoint.pth'

    ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    src = ckpt.get('model', ckpt)
    dst = model.state_dict()
    model.load_state_dict(src, strict=False)

    total_params = sum(p.numel() for p in dst.values())
    loaded_params = sum(dst[k].numel() for k in src.keys())
    cov = 100.0 * loaded_params / max(total_params, 1)

    if args.rank == 0:
        print('Pretrained MeDUET weighted loaded!')
        print(f'Params loaded: {loaded_params / 1e6:.3f}M / {total_params / 1e6:.3f}M ({cov:.2f}%)')

    model = model.to(args.device)
    model.eval()
    model.requires_grad_(False)

    progress_bar = tqdm(enumerate(loader), total=len(loader), ncols=100)

    for i, batch in progress_bar:
        z_c_total, z_s_total = [], []
        x = batch["latent"].to(args.device)
        name = batch["name"][0]
        try:
            with torch.amp.autocast("cuda"):
                for j in range(x.shape[1]):
                    x_ = x[:, j, ...]
                    z_c, z_s, _ = model.encode_full(x_)
                    z_c = z_c.mean(dim=1)
                    z_s = z_s.mean(dim=1)
                    if j == 0:
                        logger.info(f"latent_{i}: z_c: {z_c.size()}, z_s: {z_s.size()}")
                    z_c_total.append(z_c)
                    z_s_total.append(z_s)

                z_c_total = torch.cat(z_c_total).mean(dim=0).unsqueeze(dim=0)
                z_s_total = torch.cat(z_s_total).mean(dim=0).unsqueeze(dim=0)
                logger.info(f"latent_{i}: z_c_total: {z_c_total.size()}, z_s_total: {z_s_total.size()}")
                z_c_total = z_c_total.squeeze().cpu().detach().numpy()
                z_s_total = z_s_total.squeeze().cpu().detach().numpy()

                out_filename_c = os.path.join(args.factor_dir, 'content/' + name)
                out_filename_s = os.path.join(args.factor_dir, 'style/' + name)
                out_path_c = Path(out_filename_c)
                out_path_c.parent.mkdir(parents=True, exist_ok=True)
                out_path_s = Path(out_filename_s)
                out_path_s.parent.mkdir(parents=True, exist_ok=True)

                np.save(out_filename_c + "_content", z_c_total)
                np.save(out_filename_s + "_style", z_s_total)

        except Exception as e:
            logger.error(f"Error processing {args.factor_dir}: {e}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training Content/Style Data Creation")

    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU')
    parser.add_argument('--distributed', default=True, action='store_true', help='distributed training')
    parser.add_argument("--gpu_ids", default=[0, 1, 2, 3], help="local rank")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")

    # Model parameters
    parser.add_argument('--model', default='MeDUET_vit_base', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--compression_ratio', default=4, type=int, help='compression ratio of pretrained VAE')
    parser.add_argument('--input_size', default=24, type=int, help='images input size')
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='Masking ratio')
    parser.add_argument('--embed_dim', default=768, type=int, help='embedding dimension of MAE encoder')
    parser.add_argument("--s_ratio", default=0.25, type=float)
    parser.add_argument("--num_domain", default=10, type=int)
    parser.add_argument("--num_roi", default=5, type=int)
    parser.add_argument("--tau_c", default=0.1, type=float, help='temperature coefficient of infoNCE loss')
    parser.add_argument("--tau_s", default=0.1, type=float, help='temperature coefficient of infoNCE loss')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--overlap", default=0.25, type=float)
    parser.add_argument("--spacing", default=[1.5, 1.5, 1.5])
    parser.add_argument("--volume_dim", default=[512, 512, 128])
    parser.add_argument("--patch_size", default=[96, 96, 96])
    parser.add_argument("--latent_size", default=[24, 24, 24])

    # Dataset parameters
    parser.add_argument('--cache', default=0.0, type=float)
    parser.add_argument('--replace_rate', default=0.2, type=float)
    parser.add_argument("--smartcache_dataset", default=False, help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", default=True, help="use monai cache Dataset")
    parser.add_argument('--num_workers', default=8, type=int)

    args = parser.parse_args()

    args.log_dir = '../result/downstream/log/factor_creation/log/'
    args.latent_dir = '../data/latent/'
    args.factor_dir = '../data/factor/'
    args.ae_dict = '../AutoEncoder/autoencoder.pt'
    args.pretrained_dir = '../model_save/pretrain/MeDUET/'

    diff_model_create_training_factor(args)
