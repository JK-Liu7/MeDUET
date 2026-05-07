import torch
import torch.nn as nn
import torch.nn.functional as F



class MFTD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_cosine = False
        self.eps = 1e-6


    def _masked_mean_tokens(self, dist_2d: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        if mask.ndim == 3:  # (B,L,1) -> (B,L)
            mask = mask[..., 0]
        mask = mask.to(dtype=dist_2d.dtype)
        denom = mask.sum()
        return (dist_2d * mask).sum() / denom


    def token_loss(self, x_pred, x_tgt, mask):
        if self.use_cosine:
            x_pred = F.normalize(x_pred, dim=-1, eps=self.eps)
            x_tgt = F.normalize(x_tgt, dim=-1, eps=self.eps)
            dist = (x_pred - x_tgt).pow(2).sum(dim=-1)  # (B, L)
        else:
            dist = (x_pred - x_tgt).pow(2).mean(dim=-1)     # (B, L)
        return self._masked_mean_tokens(dist, mask)


    def forward(self, x, z_S, student, teacher, M):
        B = x.size(0)
        assert B % 2 == 0

        zc1_S, zc2_S, zs1_S, zs2_S, cls_S = z_S

        if self.args.distributed:
            x_rec_S = student.module.reconstruct(zc1_S, zc2_S, zs1_S, zs2_S, cls_S, M)
        else:
            x_rec_S = student.reconstruct(zc1_S, zc2_S, zs1_S, zs2_S, cls_S, M)

        if self.args.distributed:
            x_rec_S_ = student.module.unpatchify_image(x_rec_S)
            zc_S, zs_S, _ = student.module.encode_full(x_rec_S_)
        else:
            x_rec_S_ = student.unpatchify_image(x_rec_S)
            zc_S, zs_S, _ = student.encode_full(x_rec_S_)

        zc1_S_r, zc2_S_r = zc_S[:B], zc_S[B:]
        zs1_S_r, zs2_S_r = zs_S[:B], zs_S[B:]

        with torch.no_grad():
            zc1_T, zs1_T, _ = teacher.encode_full(x)
            zc2_T, zs2_T, _ = teacher.encode_full(x.flip(0))

        m1_masked = (M == 1)
        m2_masked = (M == 0)

        loss_c1 = self.token_loss(zc1_S_r, zc1_T, m1_masked)
        loss_c2 = self.token_loss(zc2_S_r, zc2_T, m2_masked)
        loss_c = loss_c1 + loss_c2

        loss_s1 = self.token_loss(zs1_S_r, zs1_T, m1_masked)
        loss_s2 = self.token_loss(zs2_S_r, zs2_T, m2_masked)
        loss_s = loss_s1 + loss_s2

        loss = self.args.lambda_c * loss_c + loss_s

        return loss