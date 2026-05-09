import math
import os
import pickle
import numpy as np
import torch
import itertools as it
from pathlib import Path
from monai import data, transforms
from monai.data import *
from monai.transforms import MapTransform



class AddNameFromPathd(MapTransform):
    def __init__(self, key="latent"):
        super().__init__(keys=[key])
        self.key = key

    def __call__(self, data):
        d = dict(data)
        p = Path(d[self.key])
        fname = p.name

        suffix_exact = "_latent.nii.gz"
        if fname.endswith(suffix_exact):
            name = fname[: -len(suffix_exact)]
        else:
            s = fname
            for sfx in p.suffixes:   # e.g. ['.nii', '.gz']
                if s.endswith(sfx):
                    s = s[: -len(sfx)]
            if s.endswith("_latent"):
                s = s[: -len("_latent")]
            name = s

        d["name"] = name
        return d


def _base_from_latent_path(p: Path) -> str:
    fname = p.name
    suffix_exact = "_latent.nii.gz"
    if fname.endswith(suffix_exact):
        return fname[: -len(suffix_exact)]
    s = fname
    for sfx in p.suffixes:   # e.g. ['.nii', '.gz']
        if s.endswith(sfx):
            s = s[: -len(sfx)]
    if s.endswith("_latent"):
        s = s[: -len("_latent")]
    return s


def get_loader(args):
    data_dir = args.latent_dir
    factor_dir = Path(args.factor_dir)
    root = Path(data_dir)
    files = sorted(root.glob("*.nii.gz"))

    data = []
    missing = []

    for p in files:
        base = _base_from_latent_path(p)
        c = factor_dir / f"content/{base}_content.npy"
        s = factor_dir / f"style/{base}_style.npy"
        if not c.exists() or not s.exists():
            missing.append(base)
            continue
        data.append({
            "latent": str(p),
            "content": str(c),
            "style":  str(s),
        })
    if missing:
        print(f"[Warning] content/style not found for {len(missing)} cases. Examples: {missing[:5]}")

    if args.rank == 0:
        print(len(data))

    train_transforms = transforms.Compose(
        [
            AddNameFromPathd(key="latent"),
            transforms.LoadImaged(keys=["latent", "content", "style"]),
            transforms.EnsureChannelFirstd(keys=["latent"], channel_dim=-1),
            transforms.SpatialPadd(keys=["latent"], spatial_size=args.latent_size),
            transforms.RandSpatialCropd(keys=["latent"], roi_size=args.latent_size, random_size=False, random_center=True),
            transforms.ToTensord(keys=["latent", "content", "style"]),
            transforms.EnsureTyped(keys=["latent", "content", "style"], dtype=torch.float32),
        ]
    )

    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=data, transform=train_transforms, cache_rate=args.cache, num_workers=args.num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=data,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    else:
        print("Using Persistent dataset")
        train_ds = PersistentDataset(data=data,
                                     transform=train_transforms,
                                     pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     cache_dir=args.cache_dir)

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True, rank=args.rank, num_replicas=args.world_size)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, shuffle=(train_sampler is None), drop_last=True,
        pin_memory=True, persistent_workers=True, prefetch_factor=4)

    return train_loader, train_sampler