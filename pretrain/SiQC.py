import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiQC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tau_c = getattr(self.args, "tau_c", None)
        self.tau_s = getattr(self.args, "tau_s", None)
        self.logit_scale_c = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.logit_scale_s = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @staticmethod
    def l2norm(x, dim=1, eps=1e-8):
        return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

    @staticmethod
    def token_pool(feat):
        feat = F.layer_norm(feat, feat.shape[-1:])
        return feat.mean(dim=1)

    @staticmethod
    def _soft_agreement_weights(weight_logits: torch.Tensor,
                                pos_mask: torch.Tensor,
                                temperature: float = 1.0,
                                eps: float = 1e-6,
                                clip: tuple | None = (0.1, 1.5)):
        prob = (weight_logits.detach() / temperature).softmax(dim=-1)  # (N, C)

        W = prob @ prob.t()                                            # (N, N)
        W.fill_diagonal_(0.0)

        if clip is not None:
            W = torch.clamp(W, min=clip[0], max=clip[1])

        row_sum = W.sum(dim=1, keepdim=True).clamp_min(eps)
        pos_cnt = pos_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        W = W / row_sum * pos_cnt

        return W

    def supcon_qk_loss(self,
                       q_feats: torch.Tensor,
                       k_feats: torch.Tensor,
                       labels: torch.Tensor,
                       tau: float | None,
                       logit_scale: torch.Tensor | None,
                       *,
                       pos_weight_logits: torch.Tensor | None = None,
                       weight_temperature: float = 1.0,
                       weight_clip: tuple | None = (0.1, 1.5)):

        q = self.l2norm(q_feats, dim=1)
        with torch.no_grad():
            k = self.l2norm(k_feats, dim=1)

        sim = q @ k.t()

        if logit_scale is not None:
            ls = torch.clamp(logit_scale, min=math.log(1e-3), max=math.log(100.0))
            sim = sim * ls.exp()
        else:
            assert tau is not None, "Use either tau or logit_scale."
            sim = sim / tau

        N = sim.size(0)
        labels = labels.view(-1, 1)
        device = sim.device

        pos_mask = (labels == labels.t()) & (~torch.eye(N, dtype=torch.bool, device=device))
        pos_mask = pos_mask.float()

        sim = sim - torch.eye(N, device=device) * 1e9
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        if pos_weight_logits is not None:
            W = self._soft_agreement_weights(
                weight_logits=pos_weight_logits, pos_mask=pos_mask,
                temperature=weight_temperature, clip=weight_clip
            )
            pos_mask = pos_mask * W

        pos_counts = pos_mask.sum(dim=1).clamp_min(1e-6)
        loss = -(log_prob * pos_mask).sum(dim=1) / pos_counts
        return loss.mean()

    def forward(self,
                x, z_S, student, teacher, M,
                weight_temperature: float = 1.0,
                weight_clip: tuple | None = (0.1, 1.5)):
        device = x.device
        B = x.size(0)
        assert B % 2 == 0

        M_T = 1.0 - M

        zc1_S, zc2_S, zs1_S, zs2_S, cls_S = z_S

        if self.args.distributed:
            x_swap_S = student.module.reconstruct(zc1_S, zc2_S, zs2_S, zs1_S, cls_S, M)
        else:
            x_swap_S = student.reconstruct(zc1_S, zc2_S, zs2_S, zs1_S, cls_S, M)

        if self.args.distributed:
            x_swap_S_ = student.module.unpatchify_image(x_swap_S)
            zc_S, zs_S, _ = student.module.encode_full(x_swap_S_)
        else:
            x_swap_S_ = student.unpatchify_image(x_swap_S)
            zc_S, zs_S, _ = student.encode_full(x_swap_S_)

        zc_12_S, zc_21_S = zc_S[:B], zc_S[B:]
        zs_12_S, zs_21_S = zs_S[:B], zs_S[B:]

        qc = torch.cat([self.token_pool(zc1_S),
                        self.token_pool(zc2_S),
                        self.token_pool(zc_12_S),
                        self.token_pool(zc_21_S)], dim=0)
        qs = torch.cat([self.token_pool(zs1_S),
                        self.token_pool(zs2_S),
                        self.token_pool(zs_12_S),
                        self.token_pool(zs_21_S)], dim=0)

        with torch.no_grad():
            zc1_T, zc2_T, zs1_T, zs2_T, _, cls_T = teacher.encode_demix(x, M_T)
            x_swap_T = teacher.reconstruct(zc1_T, zc2_T, zs2_T, zs1_T, cls_T, M_T)
            x_swap_T_ = teacher.unpatchify_image(x_swap_T)
            zc_T, zs_T, _ = teacher.encode_full(x_swap_T_)
            zc_12_T, zc_21_T = zc_T[:B], zc_T[B:]
            zs_12_T, zs_21_T = zs_T[:B], zs_T[B:]

            kc = torch.cat([self.token_pool(zc1_T),
                            self.token_pool(zc2_T),
                            self.token_pool(zc_12_T),
                            self.token_pool(zc_21_T)], dim=0)
            ks = torch.cat([self.token_pool(zs1_T),
                            self.token_pool(zs2_T),
                            self.token_pool(zs_12_T),
                            self.token_pool(zs_21_T)], dim=0)


        perm = torch.arange(B - 1, -1, -1, device=device)
        cid = torch.cat([torch.arange(B, device=device),
                         perm,
                         torch.arange(B, device=device),
                         perm], dim=0)
        sid = torch.cat([torch.arange(B, device=device),
                         perm,
                         perm,
                         torch.arange(B, device=device)], 0)

        Lc = self.supcon_qk_loss(
            qc, kc, cid, self.tau_c, logit_scale=self.logit_scale_c,
            pos_weight_logits=None,
            weight_temperature=weight_temperature,
            weight_clip=weight_clip
        )
        Ls = self.supcon_qk_loss(
            qs, ks, sid, self.tau_s, logit_scale=self.logit_scale_s,
            pos_weight_logits=None,
            weight_temperature=weight_temperature,
            weight_clip=weight_clip
        )

        return Lc + Ls
