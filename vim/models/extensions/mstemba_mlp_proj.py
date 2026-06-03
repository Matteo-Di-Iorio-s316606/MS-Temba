"""MSTemba variant with MLP input projection (E3).

Replaces the single Linear(in_feat_dim, embed_dims[0]) projection of MSTemba
with a 2-layer MLP: in_feat_dim → hidden_dim → embed_dims[0], with LayerNorm
and GELU. Designed to test whether SCD-Net (D=4096) features need a deeper
projection to expose useful signal that a single Linear may collapse.

All other blocks are inherited unchanged from MSTemba.
"""
from typing import Any

import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from models_MSTemba import MSTemba


class MLPProjection(nn.Module):
    """2-layer MLP projection with LayerNorm + GELU.

    Input:  x of shape (B, T, in_dim)
    Output: y of shape (B, T, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in_dim) -> (B, T, out_dim)"""
        x = self.fc1(x)        # (B, T, hidden_dim)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)        # (B, T, out_dim)
        return x


class MSTembaMLPProj(MSTemba):
    """MSTemba with MLP input projection. Inherits all other behavior."""

    def __init__(
        self,
        in_feat_dim: int = 4096,
        proj_hidden_dim: int = 1024,
        proj_dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_feat_dim=in_feat_dim, **kwargs)
        # Override the parent's self.proj (Linear) with an MLP.
        # All weights re-initialized by self.apply(self._init_weights) in parent.
        self.proj = MLPProjection(
            in_dim=in_feat_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=self.embed_dims[0],
            dropout=proj_dropout,
        )
        # Re-apply init to the new module (parent already ran apply, but the
        # new submodules were created after that call).
        self.proj.apply(self._init_weights)


@register_model
def mstemba_mlp_proj(pretrained: bool = False, **kwargs: Any) -> MSTembaMLPProj:
    model = MSTembaMLPProj(
        embed_dims=[256, 384, 576],
        depths=[1, 1, 1],
        d_state=16,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=False,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError("No pretrained weights for mstemba_mlp_proj")
    return model