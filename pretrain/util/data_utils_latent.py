import torch
import numpy as np
from monai.data import *
import pickle
from monai.transforms import *
from monai.transforms import MapTransform


class ExtractSpacingd(MapTransform):
    """
    Extract voxel spacing from metadata and store it under an output key.
    Priority: 'spacing' -> 'pixdim' -> derived from 'affine'.
    """
    def __init__(self, keys, out_key="spacing", meta_key_postfix="meta_dict"):
        super().__init__(keys)
        self.out_key = out_key
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            meta_key = f"{key}_{self.meta_key_postfix}"
            meta = d.get(meta_key, {})
            spacing = None
            # Try common fields from MONAI/ITK/NIfTI readers
            if "spacing" in meta and meta["spacing"] is not None:
                spacing = meta["spacing"]
            elif "pixdim" in meta and meta["pixdim"] is not None:
                pd = meta["pixdim"]
                # Many readers store [dx, dy, dz] (len>=3)
                spacing = pd[:3] if isinstance(pd, (list, tuple, np.ndarray)) else pd
            elif "affine" in meta and meta["affine"] is not None:
                A = np.asarray(meta["affine"])
                # spacing from affine column norms
                spacing = np.sqrt((A[:3, :3] ** 2).sum(axis=0))
            if spacing is not None:
                d[self.out_key] = np.asarray(spacing, dtype=np.float32)
        return d


def get_bp_and_idx(ds_name, body_parts, body_parts_dict):
    UNKNOWN_PART = "Unknown"
    UNKNOWN_IDX = -1
    bp = body_parts.get(ds_name, UNKNOWN_PART)
    idx = body_parts_dict.get(bp, UNKNOWN_IDX)
    return bp, idx


def get_loader(args):

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

    body_parts = {
        "BTCV": "Abdomen",
        "MM-WHS": "Chest",
        "Spleen": "Spleen",
        "Covid 19": "Chest",
        "Luna": "Chest",
        "Stoic": "Chest",
        "Flare23": "Abdomen",
        "LIDC": "Chest",
        "HNSCC": "Head-Neck",
        "Totalsegmentator": "Whole_body"
    }

    body_parts_dict = {
        'Abdomen': 0,
        'Chest': 1,
        'Spleen': 2,
        'Head-Neck': 3,
        'Whole_body': 4
    }

    num_workers = args.num_workers
    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    print("Dataset 1 BTCV: number of data: {}".format(len(datalist1)))
    bp, idx = get_bp_and_idx("BTCV", body_parts, body_parts_dict)
    new_datalist1 = []
    for item in datalist1:
        item_dict = {"image": item["image"],
                     "domain": 0,
                     "dataset": "BTCV",
                     "body_part": bp,
                     "body_part_idx": idx}
        new_datalist1.append(item_dict)

    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    bp, idx = get_bp_and_idx("MM-WHS", body_parts, body_parts_dict)
    print("Dataset 2 MM-WHS: number of data: {}".format(len(datalist2)))
    for item in datalist2:
        item["domain"] = 1
        item["dataset"] = "MM-WHS"
        item["body_part"] = bp
        item["body_part_idx"] = idx

    datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
    bp, idx = get_bp_and_idx("Spleen", body_parts, body_parts_dict)
    print("Dataset 3 Spleen: number of data: {}".format(len(datalist3)))
    for item in datalist3:
        item["domain"] = 2
        item["dataset"] = "Spleen"
        item["body_part"] = bp
        item["body_part_idx"] = idx

    datalist4 = load_decathlon_datalist(jsonlist4, False, "training", base_dir=datadir4)
    bp, idx = get_bp_and_idx("Covid 19", body_parts, body_parts_dict)
    print("Dataset 4 Covid 19: number of data: {}".format(len(datalist4)))
    for item in datalist4:
        item["domain"] = 3
        item["dataset"] = "Covid 19"
        item["body_part"] = bp
        item["body_part_idx"] = idx

    datalist5 = load_decathlon_datalist(jsonlist5, False, "training", base_dir=datadir5)
    print("Dataset 5 Luna: number of data: {}".format(len(datalist5)))
    bp, idx = get_bp_and_idx("Luna", body_parts, body_parts_dict)
    new_datalist5 = []
    for item in datalist5:
        item_dict = {"image": item["image"],
                     "domain": 4,
                     "dataset": "Luna",
                     "body_part": bp,
                     "body_part_idx": idx}
        new_datalist5.append(item_dict)

    datalist6 = load_decathlon_datalist(jsonlist6, False, "training", base_dir=datadir6)
    bp, idx = get_bp_and_idx("Stoic", body_parts, body_parts_dict)
    print("Dataset 6 Stoic: number of data: {}".format(len(datalist6)))
    for item in datalist6:
        item["domain"] = 5
        item["dataset"] = "Stoic"
        item["body_part"] = bp
        item["body_part_idx"] = idx

    datalist7 = load_decathlon_datalist(jsonlist7, False, "training", base_dir=datadir7)
    bp, idx = get_bp_and_idx("Flare23", body_parts, body_parts_dict)
    print("Dataset 7 Flare23: number of data: {}".format(len(datalist7)))
    for item in datalist7:
        item["domain"] = 6
        item["dataset"] = "Flare23"
        item["body_part"] = bp
        item["body_part_idx"] = idx

    datalist8 = load_decathlon_datalist(jsonlist8, False, "training", base_dir=datadir8)
    bp, idx = get_bp_and_idx("LIDC", body_parts, body_parts_dict)
    print("Dataset 8 LIDC: number of data: {}".format(len(datalist8)))
    for item in datalist8:
        item["domain"] = 7
        item["dataset"] = "LIDC"
        item["body_part"] = bp
        item["body_part_idx"] = idx

    datalist9 = load_decathlon_datalist(jsonlist9, False, "training", base_dir=datadir9)
    bp, idx = get_bp_and_idx("HNSCC", body_parts, body_parts_dict)
    print("Dataset 9 HNSCC: number of data: {}".format(len(datalist9)))
    for item in datalist9:
        item["domain"] = 8
        item["dataset"] = "HNSCC"
        item["body_part"] = bp
        item["body_part_idx"] = idx

    datalist10 = load_decathlon_datalist(jsonlist10, False, "training", base_dir=datadir10)
    bp, idx = get_bp_and_idx("Totalsegmentator", body_parts, body_parts_dict)
    print("Dataset 10 Totalsegmentator: number of data: {}".format(len(datalist10)))
    for item in datalist10:
        item["domain"] = 9
        item["dataset"] = "Totalsegmentator"
        item["body_part"] = bp
        item["body_part_idx"] = idx

    vallist1 = load_decathlon_datalist(jsonlist1, False, "validation", base_dir=datadir1)
    vallist2 = load_decathlon_datalist(jsonlist2, False, "validation", base_dir=datadir2)
    vallist3 = load_decathlon_datalist(jsonlist3, False, "validation", base_dir=datadir3)
    vallist4 = load_decathlon_datalist(jsonlist4, False, "validation", base_dir=datadir4)
    vallist5 = load_decathlon_datalist(jsonlist5, False, "validation", base_dir=datadir5)


    datalist = new_datalist1 + datalist2 + datalist3 + datalist4 + new_datalist5 + datalist6 + datalist7 + datalist8 + datalist9 + datalist10
    val_files = vallist1 + vallist2 + vallist3 + vallist4 + vallist5

    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(val_files)))


    train_transforms = Compose([LoadImaged(keys=["image"], image_only=False, dtype=np.int16),
                                EnsureChannelFirstd(keys=["image"]),
                                Orientationd(keys=["image"], axcodes="RAS"),
                                ExtractSpacingd(keys=["image"], out_key="orig_spacing"),

                                ScaleIntensityRanged(
                                    keys=["image"], a_min=args.a_min, a_max=args.a_max,
                                    b_min=args.b_min, b_max=args.b_max, clip=True),
                                Spacingd(keys="image", allow_missing_keys=True, pixdim=args.spacing, mode="bilinear"),
                                ExtractSpacingd(keys=["image"], out_key="spacing"),
                                CropForegroundd(keys=["image"], source_key="image"),
                                CenterSpatialCropd(keys="image", roi_size=args.roi_size),
                                SpatialPadd(keys="image", spatial_size=args.patch_size),
                                DivisiblePadd(keys="image", k=4),

                                SelectItemsd(["image", "domain", "dataset", "orig_spacing", "spacing", "body_part_idx"], allow_missing_keys=True),
                                ToTensord(keys=["image", "domain", "orig_spacing", "spacing", "body_part_idx"], allow_missing_keys=True),
                                EnsureTyped(keys=["image"], allow_missing_keys=True, dtype=torch.float32),
                                EnsureTyped(keys=["domain", "body_part_idx"], allow_missing_keys=True, dtype=torch.long)
                                ])


    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=datalist, transform=train_transforms, cache_rate=args.cache, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=datalist,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    else:
        print("Using Persistent dataset")
        train_ds = PersistentDataset(data=datalist,
                                     transform=train_transforms,
                                     pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     cache_dir=args.cache_dir)

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=False, rank=args.rank, num_replicas=args.world_size)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, shuffle=False, drop_last=True,
        pin_memory=True, persistent_workers=True, prefetch_factor=4
    )

    return train_loader, train_sampler



