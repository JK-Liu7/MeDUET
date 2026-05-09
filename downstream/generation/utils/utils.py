import os

import numpy as np
import scipy.ndimage as ndimage
import torch
from datetime import datetime
from time import time
import logging
import torch.distributed as dist



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
