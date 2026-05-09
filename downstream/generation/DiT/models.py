# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import torch.nn.functional as F

# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class PatchEmbed_3D(nn.Module):
    """ Voxel to Patch Embedding
    """
    def __init__(self, voxel_size=(64, 64, 32), patch_size=4, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (voxel_size[0] // patch_size[0]) * (voxel_size[1] // patch_size[1]) * (voxel_size[2] // patch_size[2])
        self.patch_xyz = (voxel_size[0] // patch_size[0], voxel_size[1] // patch_size[1], voxel_size[2] // patch_size[2])
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = x.float()
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class CSEmbedder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob = 0.0, kind = "content",
                 tau = 0.1, learnable = True):
        super().__init__()
        assert kind in ("content", "style"), "kind must be 'content' or 'style'"
        self.kind = kind
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_prob = float(dropout_prob)
        self.tau = nn.Parameter(torch.tensor(float(tau))) if learnable else torch.tensor(float(tau))

        self.embedding_table = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_size, hidden_size, bias=True)
        )

        self.c_null = nn.Parameter(torch.zeros(input_size))
        self.s_null = nn.Parameter(torch.zeros(input_size))
        nn.init.normal_(self.c_null, mean=0.0, std=0.02)
        nn.init.normal_(self.s_null, mean=0.0, std=0.02)

    @property
    def _null_vec(self) -> torch.Tensor:
        return self.c_null if self.kind == "content" else self.s_null

    def _norm_temp(self, v: torch.Tensor) -> torch.Tensor:
        v = F.normalize(v, p=2, dim=-1, eps=1e-6)
        tau = self.tau
        if not torch.is_tensor(tau):
            tau = torch.tensor(tau, device=v.device, dtype=v.dtype)
        return v / tau.clamp_min(1e-3)

    def token_drop(self, embs: torch.Tensor, train: bool, force_drop_ids: torch.Tensor | None):
        B = embs.shape[0]
        device = embs.device

        if force_drop_ids is None:
            if self.dropout_prob > 0 and train:
                drop_mask = torch.rand(B, device=device) < self.dropout_prob
            else:
                return embs
        else:
            drop_mask = (force_drop_ids.bool() if force_drop_ids.dim() == 0
                         else force_drop_ids.to(device).bool())

        if not drop_mask.any():
            return embs

        null_batch = repeat(self._null_vec.to(device, dtype=embs.dtype), 'd -> b d', b=B)
        embs = torch.where(drop_mask.unsqueeze(-1), null_batch, embs)
        return embs

    def forward(self, embs: torch.Tensor, train: bool = True, force_drop_ids: torch.Tensor | None = None):
        embs = self.token_drop(embs, train=train, force_drop_ids=force_drop_ids)
        embs = self._norm_temp(embs)
        return self.embedding_table(embs)

    def null_embedding(self, batch_size: int, device=None, dtype=None):
        vec = self._null_vec
        if device is not None: vec = vec.to(device)
        if dtype  is not None: vec = vec.to(dtype)
        vec = repeat(vec, 'd -> b d', b=batch_size)
        vec = self._norm_temp(vec)
        return self.embedding_table(vec)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with AdaLN-Zero where (γ, β) are produced
    separately from content and style then summed.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.mod_content = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )
        self.mod_style = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )
        self.mod_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, c_cond, s_cond):
        shift_msa_c, scale_msa_c, shift_mlp_c, scale_mlp_c = self.mod_content(c_cond).chunk(4, dim=1)
        shift_msa_s, scale_msa_s, shift_mlp_s, scale_mlp_s = self.mod_style(s_cond).chunk(4, dim=1)

        shift_msa = shift_msa_c + shift_msa_s
        scale_msa = scale_msa_c + scale_msa_s
        shift_mlp = shift_mlp_c + shift_mlp_s
        scale_mlp = scale_mlp_c + scale_mlp_s

        gate_msa, gate_mlp = self.mod_gate(t).chunk(2, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTBlock_VAE_simple(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT. AdaLN-Zero with separate (time/content/style) modulation.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True
        )
        self.mod_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.mod_content = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.mod_style = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, c_cond, s_cond):
        shift_t, scale_t = self.mod_time(t).chunk(2, dim=1)
        shift_c, scale_c = self.mod_content(c_cond).chunk(2, dim=1)
        shift_s, scale_s = self.mod_style(s_cond).chunk(2, dim=1)

        shift = shift_t + shift_c + shift_s
        scale = scale_t + scale_c + scale_s

        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=64,
        input_depth=32,
        patch_size=4,
        in_channels=4,
        s_ratio=0.25,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.2,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_depth = input_depth
        self.s_ratio = s_ratio
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed_3D((input_size, input_size, input_depth), patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.content_embedder = CSEmbedder(hidden_size, hidden_size, class_dropout_prob, kind='content', learnable=True)
        self.style_embedder = CSEmbedder(int(hidden_size * self.s_ratio), hidden_size, class_dropout_prob, kind='style', learnable=True)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.input_size//self.patch_size), int(self.input_size//self.patch_size),
                                            int(self.input_depth//self.patch_size))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.mod_content[-1].weight, 0)
            nn.init.constant_(block.mod_content[-1].bias, 0)
            nn.init.constant_(block.mod_style[-1].weight, 0)
            nn.init.constant_(block.mod_style[-1].bias, 0)
            nn.init.constant_(block.mod_gate[-1].weight, 0)
            nn.init.constant_(block.mod_gate[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.mod_time[-1].weight, 0)
        nn.init.constant_(self.final_layer.mod_time[-1].bias, 0)
        nn.init.constant_(self.final_layer.mod_content[-1].weight, 0)
        nn.init.constant_(self.final_layer.mod_content[-1].bias, 0)
        nn.init.constant_(self.final_layer.mod_style[-1].weight, 0)
        nn.init.constant_(self.final_layer.mod_style[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def unpatchify_3D(self, x0):
        """
        input: (N, T, patch_size * patch_size * patch_size * C)    (N, 64, 8*8*8*3)
        voxels: (N, C, X, Y, Z)          (N, 3, 32, 32, 32)
        """
        c = self.out_channels
        p = self.patch_size
        x = y = self.input_size // self.patch_size
        z = self.input_depth // self.patch_size
        assert x * y * z == x0.shape[1]

        x0 = x0.reshape(shape=(x0.shape[0], x, y, z, p, p, p, c))
        x0 = torch.einsum('nxyzpqrc->ncxpyqzr', x0)
        points = x0.reshape(shape=(x0.shape[0], c, x * p, y * p, z * p))
        return points

    def forward(self, x, t, content, style):
        """
        Forward pass of DiT.
        x: (N, C, H, W, D) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W * D / patch_size ** 3
        t = self.t_embedder(t)                   # (N, D)

        if self.training:
            c_emb = self.content_embedder(content, self.training)
            s_emb = self.style_embedder(style, self.training)
        else:
            if content is None or style is None:
                B = x.shape[0]
                c_emb = self.content_embedder.null_embedding(B, x.device, x.dtype)
                s_emb = self.style_embedder.null_embedding(B, x.device, x.dtype)
            else:
                c_emb = content
                s_emb = style

        for block in self.blocks:
            x = block(x, t, c_emb, s_emb)                      # (N, T, D)
        x = self.final_layer(x, t, c_emb, s_emb)                # (N, T, patch_size ** 3 * out_channels)

        x = self.unpatchify_3D(x)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, content, style, cfg_scale_c, cfg_scale_s):
        """
        Forward pass of SiT, but also batches the uncondional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb

        half = x[: len(x) // 3]
        combined = torch.cat([half, half, half], dim=0)

        B = half.shape[0]
        c_emb = self.content_embedder(content, self.training)
        s_emb = self.style_embedder(style, self.training)
        c_emb_null = self.content_embedder.null_embedding(B, x.device, x.dtype)
        s_emb_null = self.style_embedder.null_embedding(B, x.device, x.dtype)
        c_emb_total = torch.cat([c_emb_null, c_emb_null, c_emb], dim=0)
        s_emb_total = torch.cat([s_emb_null, s_emb, s_emb_null], dim=0)

        model_out = self.forward(combined, t, c_emb_total, s_emb_total)

        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        nn_eps, ns_eps, cn_eps = torch.split(eps, len(eps) // 3, dim=0)
        half_eps = nn_eps + cfg_scale_c * (ns_eps - nn_eps) + cfg_scale_s * (cn_eps - nn_eps)
        eps = torch.cat([half_eps, half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class DiT_VAE_simple(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock_VAE_simple(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_3d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, grid_size_z, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_x = np.arange(grid_size_h, dtype=np.float32)
    grid_y = np.arange(grid_size_w, dtype=np.float32)
    grid_z = np.arange(grid_size_z, dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')  # here y goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size_h, grid_size_w, grid_size_z])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0
    # use half of dimensions to encode grid_h
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (X*Y*Z, D/3)
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (X*Y*Z, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (X*Y*Z, D/3)

    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1) # (X*Y*Z, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_B_2_VAE_simple(**kwargs):
    return DiT_VAE_simple(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_XL_2_VAE_simple(**kwargs):
    return DiT_VAE_simple(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,

    # custom
    'DiT-B/2-VAE-simple':   DiT_B_2_VAE_simple,
    'DiT-XL/2-VAE-simple':   DiT_XL_2_VAE_simple,
}
