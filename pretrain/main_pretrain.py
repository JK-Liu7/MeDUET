import argparse
import os
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel
from timm.scheduler.cosine_lr import CosineLRScheduler
import timm.optim.optim_factory as optim_factory
from util.data_utils_cache import *
import util.misc as misc
import models_MeDUET
from SiQC import *
from MFTD import *
from engine_pretrain import *
from util.misc import *
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*qfac.*")
warnings.filterwarnings("ignore", message=".*pixdim.*")
logging.getLogger("nibabel").setLevel(logging.ERROR)


def get_args_parser():
    parser = argparse.ArgumentParser('MeDUET pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument("--ckpt_interval", default=20, type=int)

    # distributed training parameters
    parser.add_argument('--distributed', default=True, action='store_true', help='distributed training')
    parser.add_argument("--gpu_ids", default=[0, 1, 2, 3], help="local rank")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")

    # enable amp
    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=True)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=2e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=50, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--betas', type=float, default=(0.9, 0.95))
    parser.add_argument('--momentum_start', default=0.997, type=float)
    parser.add_argument('--momentum_final', default=0.9997, type=float)

    # Model parameters
    parser.add_argument('--model', default='MeDUET_vit_base', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--compression_ratio', default=4, type=int, help='compression ratio of pretrained VAE')
    parser.add_argument('--latent_size', default=24, type=int, help='images input size')
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='Masking ratio')
    parser.add_argument('--embed_dim', default=768, type=int, help='embedding dimension of MAE encoder')
    parser.add_argument("--s_ratio", default=0.25, type=float)
    parser.add_argument("--num_domain", default=10, type=int)
    parser.add_argument("--num_roi", default=5, type=int)
    parser.add_argument("--tau_c", default=0.1, type=float, help='temperature coefficient of infoNCE loss')
    parser.add_argument("--tau_s", default=0.1, type=float, help='temperature coefficient of infoNCE loss')
    parser.add_argument('--norm_pix_loss', default=False, help='Use (per-patch) normalized pixels as targets for computing loss')

    # Hyperparameters of loss function
    parser.add_argument('--MFTD_epoch', type=int, default=1000)
    parser.add_argument('--lambda_c', type=float, default=0.5)
    parser.add_argument('--lambda1', type=float, default=0.2)
    parser.add_argument('--lambda2', type=float, default=0.5)
    parser.add_argument('--lambda3', type=float, default=0.3)

    # Dataset parameters
    parser.add_argument("--random_aug", default=True)
    parser.add_argument("--patch_size", default=[96, 96, 96])
    parser.add_argument("--spacing_type", default="fixed")
    parser.add_argument("--spacing", default=[1.5, 1.5, 1.5])
    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument('--cache', default=0.0, type=float)
    parser.add_argument('--replace_rate', default=0.2, type=float)
    parser.add_argument("--smartcache_dataset", default=False, help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", default=False, help="use monai cache Dataset")
    parser.add_argument('--num_workers', default=8, type=int)

    return parser


def main(args):
    misc.init_distributed_mode(args)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    logger = create_logger(args.log_dir, args.distributed)

    train_loader, train_sampler = get_loader(args)

    # define models
    student = models_MeDUET.__dict__[args.model](args, norm_pix_loss=args.norm_pix_loss,
                                            img_size=args.latent_size).to(args.device)
    teacher = deepcopy(student).to(args.device)
    for p in teacher.parameters():
        p.requires_grad = False

    SiQC_module = SiQC(args).to(args.device)
    MFTD_module = MFTD(args).to(args.device)

    student.train()
    teacher.eval()

    if args.distributed:
        student = DistributedDataParallel(student, device_ids=[args.rank], find_unused_parameters=True)
        model_without_ddp = student.module

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=args.betas)

    lr_scheduler = CosineLRScheduler(optimizer, warmup_t=args.warmup_epochs, warmup_lr_init=1e-6, t_initial=args.epochs,
                                  lr_min=args.min_lr, cycle_limit=1)

    momentum_schedule = cosine_scheduler(args.momentum_start, args.momentum_final, args.epochs, len(train_loader))

    scaler = GradScaler()

    print(f"Start training for {args.epochs} epochs")
    logger.info("Start training for {} epochs".format(args.epochs))
    print(("Parameters {:.2f}M".format(sum([x.numel() for x in student.parameters() if x.requires_grad])/1e6)))
    logger.info("Parameters {:.2f}M".format(sum([x.numel() for x in student.parameters() if x.requires_grad])/1e6))

    start_time = timeit.default_timer()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(args, student, teacher, MFTD_module, SiQC_module, train_loader,
            optimizer, momentum_schedule, epoch, scaler, logger)

        lr_scheduler.step(epoch)

        if args.model_dir and (epoch % args.ckpt_interval == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=student, model_without_ddp=model_without_ddp, optimizer=optimizer, epoch=epoch)

    end = timeit.default_timer()
    total_time = end - start_time
    logger.info('Training time {}'.format(total_time))



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    args.log_dir = '../result/pretrain/log/'
    args.latent_dir = '../data/latent_pretrain/'
    args.model_dir = '../model_save/pretrain/MeDUET/'
    args.cache_dir = '../data/cache/pretrain_MeDUET/'
    args.ae_dict = '../AutoEncoder/autoencoder.pt'

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    main(args)