import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils.utils import AverageMeter, distributed_all_gather
from monai.data import decollate_batch, DataLoader
import torch.distributed as dist
from torch.utils.data import Subset


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

    length_tensor = torch.tensor([idx_all.numel()], dtype=torch.long, device=device) if rank == 0 else torch.zeros(1,
                                                                                                                   dtype=torch.long,
                                                                                                                   device=device)
    dist.broadcast(length_tensor, src=0)
    Llen = int(length_tensor.item())
    if rank != 0:
        idx_all = torch.empty(Llen, dtype=torch.long, device=device)
    dist.broadcast(idx_all, src=0)
    idx_all = idx_all.cpu()

    if Llen == 0:
        raise RuntimeError("No indices sampled for style bank (K_total=0?).")

    per_rank_idx = idx_all[rank::world]
    assert per_rank_idx.numel() > 0, "Ensure K_total >= world_size so that each rank gets samples."

    subset = Subset(target_dataset, per_rank_idx.tolist())
    loader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    S_part = None
    count_part = torch.tensor(0.0, device=device)

    for batch in loader:
        x = batch["image"].to(device)
        x = resize(x)

        with torch.no_grad():
            with autocast(enabled=args.amp):
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
        raise RuntimeError("No indices sampled for style bank.")

    g = torch.Generator()
    g.manual_seed(base_seed + epoch)
    K_eff = min(K_total, N)
    idx_all = torch.randperm(N, generator=g)[:K_eff].tolist()

    subset = Subset(target_dataset, idx_all)
    pin_mem = (device.type == "cuda")
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
        x = resize(x)

        with autocast(enabled=getattr(args, "amp", False)):
            latent = []
            x1, x2 = x.chunk(2, dim=0)
            for x_s in (x1, x2):
                x_mb = autoencoder.encode_stage_2_inputs(x_s)
                latent.append(x_mb)
            z = torch.cat(latent, dim=0)

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


def resize(img):
    size = 256
    b, _, c, h, w = img.size()
    new_img = []
    for i in range(b):
        im = img[i, :, :, :, :]
        im = F.interpolate(im, size=[size, size], mode='bilinear', align_corners=True)
        new_img.append(im.unsqueeze(0))
    new_img = torch.cat(new_img, dim=0)
    return new_img


def style_augmentation(z, MeDUET, style_token, alpha):
    z_c, z_s, cls = MeDUET.encode_full(z)
    a, b = alpha
    alpha = torch.empty(1).uniform_(a, b).item()
    z_s_mix = (1 - alpha) * z_s + alpha * style_token
    z_aug = MeDUET.decode_full(z_c, z_s_mix, cls)
    return z_aug


def train_epoch(args, model, autoencoder, MeDUET, ema_Es, dataset, loader, optimizer, scaler, epoch, logger):
    model.train()
    run_loss = AverageMeter()

    loss_cls = torch.nn.CrossEntropyLoss()

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
        data, target = data.to(args.device), target.to(args.device)
        data = resize(data)

        with torch.no_grad():
            with autocast(enabled=args.amp):
                z_parts = []
                x1, x2 = data.chunk(2, dim=0)
                for x_s in (x1, x2):
                    z_parts.append(autoencoder.encode_stage_2_inputs(x_s))
                z = torch.cat(z_parts, dim=0)

                y_parts = []
                y1, y2 = target.chunk(2, dim=0)
                for y_s in (y1, y2):
                    y_parts.append(y_s)
                y = torch.cat(y_parts, dim=0)

        z_chunks = z.chunk(2, dim=0)
        y_chunks = y.chunk(2, dim=0)
        tgt_chunks = target.chunk(2, dim=0)
        num_mb = 2 

        loss_cls_total = 0.0
        loss_cls_aug_total = 0.0

        do_aug = (torch.rand(1, device=args.device).item() < args.p_aug)

        if args.amp:
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad(set_to_none=True)

        for z_mb, y_mb, tgt_mb in zip(z_chunks, y_chunks, tgt_chunks):
            with autocast(enabled=args.amp):
                z_pred_mb = model(z_mb)
                loss_cls_mb = loss_cls(z_pred_mb, y_mb) / num_mb

                if do_aug:
                    with torch.no_grad():
                        z_aug_mb = style_augmentation(z_mb, MeDUET, ema_Es, args.alpha)
                    z_pred_aug_mb = model(z_aug_mb)

                    loss_cls_aug_mb = loss_cls(z_pred_aug_mb, tgt_mb) / num_mb
                    loss_mb = (
                            loss_cls_mb
                            + args.lambda_aug * loss_cls_aug_mb
                    )
                    loss_cls_aug_total += float(loss_cls_aug_mb.detach())
                else:
                    loss_mb = loss_cls_mb

                loss_cls_total += float(loss_cls_mb.detach())

            if args.amp:
                scaler.scale(loss_mb).backward()
            else:
                loss_mb.backward()

        if args.amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        loss_total = (
                loss_cls_total
                + (args.lambda_aug * loss_cls_aug_total if do_aug else 0.0)
        )

        loss = torch.as_tensor(loss_total, device=args.device)
        run_loss.update(loss.item(), n=args.batch_size)

        steps = len(loader)
        interval = 1 if steps == 1 else max(1, steps // 2)
        log_now = (idx % interval == 0) and (steps == 1 or idx != 0)
        if log_now:
            if args.rank == 0:
                print(
                    "Epoch:{}, Cls_Loss:{:.4f}, Cls_Loss_Aug:{:.4f}, Lr:{:.6f}".format(
                        epoch, loss_cls_total, loss_cls_aug_total,
                        optimizer.param_groups[0]['lr']))
            logger.info(
                    "Epoch:{}, Cls_Loss:{:.4f}, Cls_Loss_Aug:{:.4f}, Lr:{:.6f}".format(
                        epoch, loss_cls_total, loss_cls_aug_total,
                        optimizer.param_groups[0]['lr']))
    return run_loss.avg, ema_Es


def val_epoch(args, model, autoencoder, loader, epoch, logger):
    model.eval()

    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]

            data = resize(data)
            data, target = data.cuda(args.rank), target.cuda(args.rank)

            with autocast(enabled=args.amp):
                z = autoencoder.encode_stage_2_inputs(data)
                logits = model(z)

            value = torch.eq(logits.argmax(dim=1), target)
            metric_count += len(value)
            num_correct += value.sum().item()
            metric = num_correct / metric_count

        return metric


def save_checkpoint(model, epoch, args, best_dice=0, optimizer=None, scheduler=None):
    epoch_name = str(epoch)
    filename = 'checkpoint_%s.pth' % epoch_name
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"model": state_dict, "args": args, "epoch": epoch, "best_dice": best_dice}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.model_dir, filename)
    torch.save(save_dict, filename)


def run_training(
    args,
    model,
    autoencoder,
    MeDUET,
    train_dataset,
    train_loader,
    val_loader,
    optimizer,
    logger,
    scheduler=None,
    start_epoch=0,
):

    scaler = None
    if args.amp:
        scaler = GradScaler()

    val_acc_max = 0.0
    ema_Es = None

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        train_loss, ema_Es = train_epoch(args, model, autoencoder, MeDUET, ema_Es, train_dataset, train_loader, optimizer, scaler, epoch, logger)

        if args.model_dir and (epoch % args.ckpt_interval == 0 or epoch + 1 == args.max_epochs):
            save_checkpoint(model, epoch, args, best_dice=val_acc_max, optimizer=optimizer, scheduler=scheduler)

        if (epoch + 1) % args.val_every == 0 or epoch == 0:
            val_avg_acc = val_epoch(args, model, autoencoder, val_loader, epoch, logger)
            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print("Validation Epoch:{}, Acc:{:.4f}".format(
                    epoch, val_avg_acc))
                logger.info("Validation Epoch:{}, Acc:{:.4f}".format(
                    epoch, val_avg_acc))

                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    logger.info("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc

        if scheduler is not None:
            scheduler.step(epoch)
    if args.rank == 0:
        print("Training Finished !, Best Acc: ", val_acc_max)
    logger.info("Training Finished !, Best Acc: %s", val_acc_max)

    return val_acc_max