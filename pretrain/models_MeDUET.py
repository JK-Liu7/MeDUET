from models_mae import *
from util.pos_embed import *


class PatchEmbed_3D(nn.Module):
    """ Voxel to Patch Embedding
    """
    def __init__(self, voxel_size=(16, 16, 16), patch_size=2, in_chans=4, embed_dim=768, bias=True):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        num_patches = (voxel_size[0] // patch_size[0]) * (voxel_size[1] // patch_size[1]) * (voxel_size[2] // patch_size[2])
        self.patch_xyz = (voxel_size[0] // patch_size[0], voxel_size[1] // patch_size[1], voxel_size[2] // patch_size[2])
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Disentanglement(nn.Module):
    def __init__(self, emb_dim, s_ratio=0.25):
        super().__init__()
        c_dim = emb_dim
        s_dim = int(emb_dim * s_ratio)
        self.proj_c = nn.Conv1d(emb_dim, c_dim, 1, bias=False)   # content head
        self.proj_s = nn.Conv1d(emb_dim, s_dim, 1, bias=False)   # style head
        self.smooth = nn.Conv1d(s_dim, s_dim, 3, padding=1, groups=s_dim, bias=False)

    def forward(self, x):
        x_ = x.transpose(1, 2)
        content = self.proj_c(x_)    # content map
        style = self.smooth(self.proj_s(x_))  # style map
        content = content.transpose(1, 2)
        style = style.transpose(1, 2)
        return content, style


class Entanglement(nn.Module):
    def __init__(self, emb_dim, s_ratio=0.25):
        super().__init__()
        c_dim = emb_dim
        s_dim = int(emb_dim * s_ratio)
        self.proj = nn.Conv1d(
            in_channels=c_dim + s_dim,
            out_channels=emb_dim,
            kernel_size=1,
            padding=0,
            bias=True
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        x = x.transpose(1, 2)
        x_fused = self.proj(x)
        x_fused = x_fused.transpose(1, 2)
        return x_fused


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class Domain_classifier(nn.Module):
    def __init__(self, c_dim, s_dim, num_domain, lambda_adv=1.0):
        super().__init__()
        self.domain_classifier_c = nn.Sequential(
                            nn.Linear(c_dim, c_dim * 2),
                            nn.ReLU(),
                            nn.Linear(c_dim * 2, c_dim * 2),
                            nn.ReLU(),
                            nn.Linear(c_dim * 2, num_domain)
                            )

        self.domain_classifier_s = nn.Sequential(
                            nn.Linear(s_dim, s_dim * 2),
                            nn.ReLU(),
                            nn.Linear(s_dim * 2, s_dim * 2),
                            nn.ReLU(),
                            nn.Linear(s_dim * 2, num_domain)
                            )
        self.grl = GradientReversal(lambda_adv)
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, z_c, z_s, domain):
        z_c_ = z_c.mean(dim=1)
        z_s_ = z_s.mean(dim=1)
        domain = domain.view(-1).long()

        zc_rev = self.grl(z_c_)
        c_pred = self.domain_classifier_c(zc_rev)
        l_c = self.cls_loss(c_pred, domain)

        s_pred = self.domain_classifier_s(z_s_)
        l_s = self.cls_loss(s_pred, domain)
        return l_c, l_s


class MeDUET(MaskedAutoencoderViT):
    def __init__(self, args, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, norm_image=False):
        super().__init__(img_size, patch_size, in_chans,
                         embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_pix_loss)

        self.args = args
        self.img_size = img_size
        self.patch_size= patch_size
        self.in_chans = in_chans

        grid_size = img_size // patch_size
        self.grid_size = grid_size

        self.embed_layer = PatchEmbed_3D(to_3tuple(img_size), patch_size, in_chans, embed_dim)
        num_patches = self.embed_layer.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.disentangled_factorizer = Disentanglement(embed_dim, s_ratio=self.args.s_ratio)
        self.latent_entanglement = Entanglement(embed_dim, s_ratio=self.args.s_ratio)

        self.domain_classifier = Domain_classifier(embed_dim, int(embed_dim * self.args.s_ratio), self.args.num_domain, 1.0)

        self.initialize_weights()


    def patchify_image(self, x):
        B, C, H, W, D = x.shape
        patch_size = to_3tuple(self.patch_size)
        grid_size = (H // patch_size[0], W // patch_size[1], D // patch_size[2])

        x = x.reshape(B, C, grid_size[0], patch_size[0], grid_size[1], patch_size[1], grid_size[2],
                      patch_size[2])  # [B,C,gh,ph,gw,pw,gd,pd]
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, np.prod(grid_size),
                                                      np.prod(patch_size) * C)  # [B,gh*gw*gd,ph*pw*pd*C]
        return x

    def unpatchify_image(self, x):
        """
        input: (N, T, patch_size * patch_size * patch_size * C)
        voxels: (N, C, H, W, D)
        """
        c = self.in_chans
        p = self.patch_size
        h = w = d = self.grid_size
        assert h * w * d == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, c))
        x = torch.einsum('nxyzpqrc->ncxpyqzr', x)
        points = x.reshape(shape=(x.shape[0], c, h * p, w * p, d * p))
        return points

    def random_mixing(self, x, mask_ratio):
        """
        x: [N, C, H, W, D],
        """
        N, C, H, W, D = x.shape
        L = self.grid_size ** 3
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is for x1, 1 is for x2
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask_expanded = mask.unsqueeze(-1)    # [N, L, 1]
        return mask_expanded, ids_restore

    def unmixng(self, x_mixed, mask):
        """
        x_mixed: [N, L, D],
        mask: [N, L, 1]
        """
        N, L, D = x_mixed.shape

        patches = x_mixed[:, 1:, :]
        m1 = (mask == 0).expand(-1, -1, D)
        m2 = (mask == 1).expand(-1, -1, D)

        x1_kept = torch.masked_select(patches, m1).view(N, -1, D)
        x2_kept = torch.masked_select(patches, m2).view(N, -1, D)
        return x1_kept, x2_kept

    def separation(self, z1, z2, mask):
        assert mask.dim() in (2, 3), "mask must be [N, L] or [N, L, 1]"
        if mask.dim() == 3:
            mask2d = mask.squeeze(-1)
        else:
            mask2d = mask

        m0 = (mask2d == 0)  # [N, L] (bool)
        m1 = ~m0  # [N, L] (bool)

        N, L = mask2d.shape
        D = z1.shape[-1] if z1.numel() > 0 else z2.shape[-1]
        device = z1.device if z1.numel() > 0 else z2.device
        dtype = z1.dtype if z1.numel() > 0 else z2.dtype

        len_keep = z1.shape[1]
        len_take = z2.shape[1]
        assert len_keep + len_take == L, "len_keep + (L-len_keep) must equal L"
        rank0 = torch.cumsum(m0.int(), dim=1) - 1  # [-1, ..., len_keep-1]
        rank1 = torch.cumsum(m1.int(), dim=1) - 1  # [-1, ..., len_take-1]

        if len_keep > 0:
            idx0 = rank0.clamp(min=0).unsqueeze(-1).expand(-1, -1, D)  # [N, L, D]
            z1_filled = torch.gather(z1, dim=1, index=idx0)  # [N, L, D]
        else:
            z1_filled = torch.zeros(N, L, D, device=device, dtype=dtype)
        if len_take > 0:
            idx1 = rank1.clamp(min=0).unsqueeze(-1).expand(-1, -1, D)  # [N, L, D]
            z2_filled = torch.gather(z2, dim=1, index=idx1)  # [N, L, D]
        else:
            z2_filled = torch.zeros(N, L, D, device=device, dtype=dtype)
        return z1_filled, z2_filled

    def remixing(self, z1, z2, mask):
        assert mask.dim() in (2, 3), "mask must be [N, L] or [N, L, 1]"
        if mask.dim() == 3:
            mask2d = mask.squeeze(-1)
        else:
            mask2d = mask

        m0 = (mask2d == 0)  # [N, L]
        z1_filled, z2_filled = self.separation(z1, z2, mask)    # [N, L, D]
        z_mixed = torch.where(m0.unsqueeze(-1), z1_filled, z2_filled)   # [N, L, D]
        return z_mixed

    def disentanglement(self, z):
        content, style = self.disentangled_factorizer(z)
        return content, style

    def entanglement(self, z1, z2):
        z_fused = self.latent_entanglement(z1, z2)
        return z_fused

    def domain_classification(self, z_c, z_s, domain):
        l_c, l_s = self.domain_classifier(z_c, z_s, domain)
        return l_c, l_s

    def decorrelation(self, c, s, use_correlation=True, eps=1e-6):
        assert c.ndim == 3 and s.ndim == 3, "Expected [N, L, D] tensors"
        N, L, D1 = c.shape
        _, _, D2 = s.shape
        M = N * L

        c2 = c.reshape(M, D1).float()
        s2 = s.reshape(M, D2).float()
        c2 = c2 - c2.mean(dim=0, keepdim=True)
        s2 = s2 - s2.mean(dim=0, keepdim=True)

        if use_correlation:
            c_std = c2.std(dim=0, keepdim=True).clamp_min(eps)
            s_std = s2.std(dim=0, keepdim=True).clamp_min(eps)
            c2n = c2 / c_std
            s2n = s2 / s_std
        else:
            c2n, s2n = c2, s2

        denom = float(max(int(c2n.shape[0]) - 1, 1)) 

        Ccs = (c2n.transpose(0, 1) @ s2n) / denom 
        loss = (Ccs.pow(2)).mean()
        return loss


    def encode_demix(self, x, mask):
        z, x_ = self.forward_encoder(x, mask)       # [N, L + 1, D]
        cls = z[:, :1, :]
        z1, z2 = self.unmixng(z, mask)      # [N, L * mask_ratio, D]
        z_c1, z_s1 = self.disentanglement(z1)
        z_c2, z_s2 = self.disentanglement(z2)
        return z_c1, z_c2, z_s1, z_s2, x_, cls

    def encode_full(self, x):
        # embed patches
        x = self.embed_layer(x)
        x = x + self.pos_embed[:, 1:, :]

        # # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        z = self.norm(x)

        z_patches = z[:, 1:, :]
        cls_token = z[:, :1, :]
        z_c, z_s = self.disentanglement(z_patches)
        return z_c, z_s, cls_token


    def decode_full(self, z_c, z_s, cls):
        z = self.entanglement(z_c, z_s)
        z = torch.cat([cls, z], dim=1)

        # embed tokens
        z = self.decoder_embed(z)
        # add pos embed
        z = z + self.decoder_pos_embed
        # apply Transformer blocks
        for idx, blk in enumerate(self.decoder_blocks):
            z = blk(z)
        z = self.decoder_norm(z)
        # predictor projection
        z = self.decoder_pred(z)
        # remove cls token
        z = z[:, 1:, :]
        z_rec = self.unpatchify_image(z)
        return z_rec


    def reconstruct(self, z_c1, z_c2, z_s1, z_s2, cls, mask):
        z1 = self.entanglement(z_c1, z_s1)
        z2 = self.entanglement(z_c2, z_s2)
        z = self.remixing(z1, z2, mask)

        # append cls token
        z = torch.cat([cls, z], dim=1)
        x_rec = self.forward_decoder(z, cls, mask)
        return x_rec


    def forward_encoder(self, x, mask):
        # embed patches
        x = self.embed_layer(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        x_ = x.clone()
        # Mixing: 0 for x1, 1 for x2
        x = x * (1. - mask) + x.flip(0) * mask
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, x_

    def forward_decoder(self, x, cls, mask):
        # embed tokens
        x = self.decoder_embed(x)
        B, L, C = x.shape

        cls, patches = x[:, :1, :], x[:, 1:, :]

        mask_tokens = self.mask_token.expand(B, patches.shape[1], C)  # (B, L, C)
        p1 = patches * (1 - mask) + mask_tokens * mask
        p2 = patches * mask + mask_tokens * (1 - mask)

        x1 = torch.cat([cls, p1], dim=1)  # (B, 1+L, C)
        x2 = torch.cat([cls, p2], dim=1)  # (B, 1+L, C)
        x = torch.cat([x1, x2], dim=0)  # (2B, 1+L, C)

        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for idx, blk in enumerate(self.decoder_blocks):
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]
        return x

    def forward_loss(self, x, x_rec, mask):
        B, L, C = x_rec.shape

        # unmix tokens
        x1_rec = x_rec[:B//2]
        x2_rec = x_rec[B//2:]

        target = self.patchify_image(x)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        unmix_x_rec = x1_rec * mask + x2_rec.flip(0) * (1 - mask)
        loss_rec = (unmix_x_rec - target) ** 2
        loss_rec = loss_rec.mean()
        return loss_rec

    def forward(self, x, parameter, mask_ratio):

        domain, voxel, body_part =  parameter

        mask, ids_restore = self.random_mixing(x, mask_ratio)       # [N, L, 1]

        z_c1, z_c2, z_s1, z_s2, x_, cls = self.encode_demix(x, mask)
        z_S = (z_c1, z_c2, z_s1, z_s2, cls)

        # Reconstruction Loss
        x_rec = self.reconstruct(z_c1, z_c2, z_s1, z_s2, cls, mask)
        l_rec = self.forward_loss(x, x_rec, mask)

        # Domain Classifier Loss
        l_d_c1, l_d_s1 = self.domain_classification(z_c1, z_s1, domain)
        l_d_c2, l_d_s2 = self.domain_classification(z_c2, z_s2, domain.flip(0))
        l_d_c = (l_d_c1 + l_d_c2) * 0.5
        l_d_s = (l_d_s1 + l_d_s2) * 0.5

        loss_status = {
            'rec_loss': l_rec,
            'domain_loss_c': l_d_c,
            'domain_loss_s': l_d_s,
        }

        return x_rec, loss_status, z_S, mask


def MeDUET_vit_base(args, **kwargs):
    model = MeDUET(args,
        patch_size=4, in_chans=4, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

