from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data.distributed
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from monai.data import DataLoader, decollate_batch
from torch.utils.data import Subset
import torch.distributed as dist

from utils.utils import distributed_all_gather


@torch.no_grad()
def build_style_bank_ddp_uniform(
    args,
    target_dataset,
    MeDUET,
    autoencoder,
    K_total: int,
    device: torch.device,
    epoch: int,
    base_seed: int = 0,
    batch_size: int = 2,
    num_workers: int = 2,
    ema_prev: torch.Tensor | None = None,
    ema_beta: float = 0.0,
):
    assert dist.is_initialized(), "DDP requires initialized torch.distributed."
    world = dist.get_world_size()
    rank = dist.get_rank()
    N = len(target_dataset)

    if rank == 0:
        g = torch.Generator()
        g.manual_seed(base_seed + epoch)
        K_eff = min(K_total, N)
        idx_all = torch.randperm(N, generator=g)[:K_eff].clone().to(device)
    else:
        idx_all = torch.empty(0, dtype=torch.long, device=device)

    length_tensor = (
        torch.tensor([idx_all.numel()], dtype=torch.long, device=device)
        if rank == 0
        else torch.zeros(1, dtype=torch.long, device=device)
    )
    dist.broadcast(length_tensor, src=0)
    Llen = int(length_tensor.item())
    if rank != 0:
        idx_all = torch.empty(Llen, dtype=torch.long, device=device)
    dist.broadcast(idx_all, src=0)
    idx_all = idx_all.cpu()

    if Llen == 0:
        raise RuntimeError("No indices sampled for style bank (K_total=0?).")

    per_rank_idx = idx_all[rank::world]
    assert per_rank_idx.numel() > 0, "Ensure K_total >= world_size."

    subset = Subset(target_dataset, per_rank_idx.tolist())
    loader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    S_part = None
    count_part = torch.tensor(0.0, device=device)
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with autocast(enabled=getattr(args, "amp", False)):
            z = autoencoder.encode_stage_2_inputs(x)
            _, z_s, _ = MeDUET.encode_full(z)
        if S_part is None:
            L, Ds = int(z_s.shape[1]), int(z_s.shape[2])
            S_part = torch.zeros(L, Ds, device=device, dtype=torch.float32)
        S_part += z_s.sum(dim=0)
        count_part += float(z_s.shape[0])

    dist.all_reduce(S_part, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_part, op=dist.ReduceOp.SUM)
    Es = S_part / (count_part + 1e-6)
    if ema_prev is not None and ema_beta > 0.0:
        Es = ema_beta * ema_prev.to(Es.device, Es.dtype) + (1.0 - ema_beta) * Es
    return Es


@torch.no_grad()
def build_style_bank_single_uniform(
    args,
    target_dataset,
    MeDUET,
    autoencoder,
    K_total: int,
    device: torch.device,
    epoch: int,
    base_seed: int = 0,
    batch_size: int = 2,
    num_workers: int = 2,
    ema_prev: torch.Tensor | None = None,
    ema_beta: float = 0.0,
):
    N = len(target_dataset)
    if K_total <= 0 or N == 0:
        raise RuntimeError("No indices sampled for style bank (K_total<=0 or dataset empty).")

    g = torch.Generator()
    g.manual_seed(base_seed + epoch)
    K_eff = min(K_total, N)
    idx_all = torch.randperm(N, generator=g)[:K_eff].tolist()

    subset = Subset(target_dataset, idx_all)
    pin_mem = device.type == "cuda"
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        shuffle=False,
        drop_last=False,
    )

    S_part = None
    count_part = 0.0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with autocast(enabled=getattr(args, "amp", False)):
            z = autoencoder.encode_stage_2_inputs(x)
            _, z_s, _ = MeDUET.encode_full(z)
        if S_part is None:
            L, Ds = int(z_s.shape[1]), int(z_s.shape[2])
            S_part = torch.zeros(L, Ds, device=device, dtype=torch.float32)
        S_part += z_s.sum(dim=0).to(S_part.dtype)
        count_part += float(z_s.shape[0])

    Es = S_part / (count_part + 1e-6)
    if ema_prev is not None and ema_beta > 0.0:
        Es = ema_beta * ema_prev.to(Es.device, Es.dtype) + (1.0 - ema_beta) * Es
    return Es


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


class CompositePredictor(nn.Module):
    def __init__(self, autoencoder, model):
        super().__init__()
        self.autoencoder = autoencoder
        self.model = model

    def forward(self, x):
        z = self.autoencoder.encode_stage_2_inputs(x)
        logits = self.model(z)
        logits = F.interpolate(logits, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return logits


def style_augmentation(z, MeDUET, style_token, alpha):
    z_c, z_s, cls = MeDUET.encode_full(z)
    a, b = alpha
    alpha = torch.empty(1, device=z.device).uniform_(a, b).item()
    z_s_mix = (1 - alpha) * z_s + alpha * style_token
    z_aug = MeDUET.decode_full(z_c, z_s_mix, cls)
    return z_aug


def _split_batch(x: torch.Tensor, max_chunks: int = 2):
    n = x.shape[0]
    chunks = min(max_chunks, n)
    return list(torch.chunk(x, chunks=chunks, dim=0))


def train_epoch(args, model, autoencoder, MeDUET, ema_Es, dataset, loader, optimizer, scaler, epoch, loss_dice, logger):
    model.train()
    run_loss = AverageMeter()

    if args.distributed:
        Es = build_style_bank_ddp_uniform(
            args=args,
            target_dataset=dataset,
            MeDUET=MeDUET,
            autoencoder=autoencoder,
            K_total=args.k,
            device=args.device,
            epoch=epoch,
            base_seed=2025,
            batch_size=2,
            num_workers=args.workers,
            ema_prev=ema_Es,
            ema_beta=args.ema_momentum,
        )
    else:
        Es = build_style_bank_single_uniform(
            args=args,
            target_dataset=dataset,
            MeDUET=MeDUET,
            autoencoder=autoencoder,
            K_total=args.k,
            device=args.device,
            epoch=epoch,
            base_seed=2025,
            batch_size=2,
            num_workers=args.workers,
            ema_prev=ema_Es,
            ema_beta=args.ema_momentum,
        )
    ema_Es = Es.detach()

    progress_bar = tqdm(enumerate(loader), total=len(loader), ncols=100)
    progress_bar.set_description(f"Epoch {epoch}")

    for idx, batch_data in progress_bar:
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data = data.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True).long()

        with torch.no_grad():
            with autocast(enabled=args.amp):
                z = autoencoder.encode_stage_2_inputs(data)

        z_chunks = _split_batch(z, max_chunks=2)
        tgt_chunks = _split_batch(target, max_chunks=2)
        num_mb = len(z_chunks)
        do_aug = torch.rand(1, device=args.device).item() < args.p_aug

        optimizer.zero_grad(set_to_none=True)

        loss_seg_total = 0.0
        loss_seg_aug_total = 0.0

        for z_mb, tgt_mb in zip(z_chunks, tgt_chunks):
            with autocast(enabled=args.amp):
                logits_mb = model(z_mb)
                logits_mb = F.interpolate(logits_mb, size=tgt_mb.shape[-3:], mode="trilinear", align_corners=False)
                loss_seg_mb = loss_dice(logits_mb, tgt_mb) / num_mb

                if do_aug:
                    with torch.no_grad():
                        z_aug_mb = style_augmentation(z_mb, MeDUET, ema_Es, args.alpha)
                    logits_aug_mb = model(z_aug_mb)
                    logits_aug_mb = F.interpolate(logits_aug_mb, size=tgt_mb.shape[-3:], mode="trilinear", align_corners=False)
                    loss_seg_aug_mb = loss_dice(logits_aug_mb, tgt_mb) / num_mb
                    loss_mb = loss_seg_mb + args.lambda_aug * loss_seg_aug_mb
                    loss_seg_aug_total += float(loss_seg_aug_mb.detach())
                else:
                    loss_mb = loss_seg_mb

                loss_seg_total += float(loss_seg_mb.detach())

            if args.amp:
                scaler.scale(loss_mb).backward()
            else:
                loss_mb.backward()

        if args.amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        loss_total = loss_seg_total + (args.lambda_aug * loss_seg_aug_total if do_aug else 0.0)
        loss = torch.as_tensor(loss_total, device=args.device)

        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=data.shape[0])

        steps = len(loader)
        interval = 1 if steps == 1 else max(1, steps // 2)
        log_now = (idx % interval == 0) and (steps == 1 or idx != 0)
        if log_now:
            msg = (
                f"Epoch:{epoch}, Seg_Loss:{loss_seg_total:.4f}, "
                f"Seg_Loss_Aug:{loss_seg_aug_total:.4f}, "
                f"Lr:{optimizer.param_groups[0]['lr']:.6f}"
            )
            if args.rank == 0:
                print(msg)
            logger.info(msg)

    return run_loss.avg, ema_Es


def val_epoch(args, model, autoencoder, loader, epoch, acc_func, model_inferer, post_label, post_pred, logger):
    model.eval()
    run_acc = AverageMeter()
    acc_func.reset()

    predictor = CompositePredictor(autoencoder, model)

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data = data.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True).long()

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    try:
                        logits = model_inferer(inputs=data, predictor=predictor)
                    except TypeError:
                        logits = model_inferer(inputs=data, network=predictor)
                else:
                    logits = predictor(data)

            if not logits.is_cuda:
                target = target.cpu()

            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.to(args.device)

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather([acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

    torch.cuda.empty_cache()
    return run_acc.avg


def save_checkpoint(model, epoch, args, best_dice=0, optimizer=None, scheduler=None):
    epoch_name = str(epoch)
    filename = f"checkpoint_{epoch_name}.pth"
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"model": state_dict, "args": args, "epoch": epoch, "best_dice": best_dice}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = Path(args.model_dir) / filename
    torch.save(save_dict, str(filename))


def run_training(
    args,
    model,
    autoencoder,
    MeDUET,
    train_dataset,
    train_loader,
    val_loader,
    optimizer,
    loss_dice,
    acc_func,
    logger,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    scaler = GradScaler() if args.amp else None
    val_acc_max = 0.0
    ema_Es = None

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        train_loss, ema_Es = train_epoch(
            args, model, autoencoder, MeDUET, ema_Es, train_dataset, train_loader, optimizer, scaler, epoch, loss_dice, logger
        )

        if args.model_dir and (epoch % args.ckpt_interval == 0 or epoch + 1 == args.max_epochs):
            save_checkpoint(model, epoch, args, best_dice=val_acc_max, optimizer=optimizer, scheduler=scheduler)

        if (epoch + 1) % args.val_every == 0 or epoch == 0:
            val_avg_acc = val_epoch(args, model, autoencoder, val_loader, epoch, acc_func, model_inferer, post_label, post_pred, logger)
            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print(f"Validation Epoch:{epoch}, Dice Score:{val_avg_acc:.4f}")
                logger.info(f"Validation Epoch:{epoch}, Dice Score:{val_avg_acc:.4f}")
                if val_avg_acc > val_acc_max:
                    print(f"new best ({val_acc_max:.6f} --> {val_avg_acc:.6f}).")
                    logger.info(f"new best ({val_acc_max:.6f} --> {val_avg_acc:.6f}).")
                    val_acc_max = val_avg_acc

        if scheduler is not None:
            scheduler.step(epoch)

    if args.rank == 0:
        print("Training Finished !, Best Dice: ", val_acc_max)
    logger.info("Training Finished !, Best Dice: %s", val_acc_max)
    return val_acc_max
