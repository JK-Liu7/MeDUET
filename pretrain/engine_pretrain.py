import math
import sys
import timeit
import torch
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from util.utils import *



def train_one_epoch(args, student, teacher, MFTD, SiQC, data_loader,
                    optimizer, momentum_schedule, epoch, scaler, logger):

    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), ncols=100)
    progress_bar.set_description(f"Epoch {epoch}")

    for i, batch in progress_bar:
        it = len(data_loader) * epoch + i

        x = batch["latent"].to(args.device).contiguous()

        domain = batch["domain"].to(args.device)
        voxel = batch["voxel"].to(args.device)
        body_part = batch["bp"].to(args.device)
        parameter = [domain, voxel, body_part]


        with autocast(enabled=args.amp):
            x_rec, loss_status, z_S, mask = student(x, parameter, args.mask_ratio)

            l_sc = SiQC(x, z_S, student, teacher, mask)

            if epoch >= args.MFTD_epoch:
                l_fd = MFTD(x, z_S, student, teacher, mask)
            else:
                l_fd = torch.zeros(1, device=args.device)

            loss_status.update({'sc_loss': l_sc})
            loss_status.update({'fd_loss': l_fd})
            l_total = train_loss_weighted_sum(args, loss_status)

        loss_value = l_total.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        l_total /= args.accum_iter

        if args.amp:
            scaler.scale(l_total).backward()
            if (i + 1) % args.accum_iter == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            l_total.backward()
            if (i + 1) % args.accum_iter == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        momentum_teacher = momentum_schedule[it]
        if args.distributed:
            update_ema(teacher, student.module, momentum_teacher)
        else:
            update_ema(teacher, student, momentum_teacher)

        if (i != 0 and i % (len(data_loader) // 2) == 0) or (i == len(data_loader) - 1):
            if args.rank == 0:
                print("Epoch:{}, Rec_Loss:{:.3f}, D_Loss_C:{:.3f}, D_Loss_S:{:.3f}, Fd_Loss:{:.3f}, Sc_Loss:{:.3f}, Lr:{:.6f}".format(
                        epoch, loss_status['rec_loss'].item(), loss_status['domain_loss_c'].item(), loss_status['domain_loss_s'].item(),
                         loss_status['fd_loss'].item(), loss_status['sc_loss'].item(), optimizer.param_groups[0]['lr']))
            logger.info("Epoch:{}, Rec_Loss:{:.3f}, D_Loss_C:{:.3f}, D_Loss_S:{:.3f}, Fd_Loss:{:.3f}, Sc_Loss:{:.3f}, Lr:{:.6f}".format(
                        epoch, loss_status['rec_loss'].item(), loss_status['domain_loss_c'].item(), loss_status['domain_loss_s'].item(),
                         loss_status['fd_loss'].item(), loss_status['sc_loss'].item(), optimizer.param_groups[0]['lr']))


