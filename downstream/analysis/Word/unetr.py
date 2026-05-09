from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock, UnetResBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep


class DynamicFusionUpBlock(nn.Module):
    """
    Final lightweight decoder block.
    Upsamples the transformer feature to the spatial size of the skip feature,
    concatenates them, and fuses with a residual conv block.
    This avoids the hard-coded latent size=(24,24,24) in the original code.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
    ) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.fuse = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.proj(x)
        x = torch.cat((x, skip), dim=1)
        x = self.fuse(x)
        return x


class UNETR(nn.Module):
    """
    UNETR encoder + simple multi-class decoder.

    Input: latent volume z with shape [B, Cz, Dz, Hy, Wx]
    Output: K-class logits at the same latent resolution as the input z.
    The trainer can then upsample logits to the original label resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        proj_type: str = "conv",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(4, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False

        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = DynamicFusionUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            norm_name=norm_name,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x: torch.Tensor) -> torch.Tensor:
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward_features(self, x_in: torch.Tensor):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(self.proj_feat(hidden_states_out[3]))
        enc3 = self.encoder3(self.proj_feat(hidden_states_out[6]))
        enc4 = self.encoder4(self.proj_feat(hidden_states_out[9]))
        bottleneck = self.proj_feat(x)
        return enc1, enc2, enc3, enc4, bottleneck

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        enc1, enc2, enc3, enc4, bottleneck = self.forward_features(x_in)
        dec3 = self.decoder5(bottleneck, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits
