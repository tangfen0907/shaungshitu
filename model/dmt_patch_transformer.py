import math
from typing import Dict

import torch
from torch import nn

from utils.dmt_patching import patchify_1d


class ChannelIndependentPatchTransformer(nn.Module):
    """
    DMT-M1 View1 encoder.

    Data flow:
        x_blt: [B, L, C]
        x_bcl = transpose(x_blt): [B, C, L]
        patches: [B, C, P, patch_len]
        patch_embed: [B, C, P, D]
        shared Transformer over patches: [B*C, P, D]
        H1: [B, C, P, D]

    The Transformer weights are shared by all variables, and variables are not
    mixed before the patch-level temporal encoder.
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        patch_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)
        self.num_patches = int(math.ceil(self.seq_len / float(self.patch_len)))

        self.patch_embed = nn.Linear(self.patch_len, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=self.d_model * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.d_model)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x_blt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_blt: Input window with shape [B, L, C].

        Returns:
            dict with:
                H1: [B, C, P, D]
                patches: [B, C, P, patch_len]
                patch_meta: patch metadata for unpatchify_1d
        """
        if x_blt.dim() != 3:
            raise ValueError(f"x_blt must be [B, L, C], got shape {tuple(x_blt.shape)}")
        bsz, seq_len, channels = x_blt.shape
        if channels != self.in_channels:
            raise ValueError(f"expected C={self.in_channels}, got C={channels}")

        x_bcl = x_blt.transpose(1, 2).contiguous()  # [B, C, L]
        patches, meta = patchify_1d(x_bcl, self.patch_len, stride=self.patch_len, pad=True)
        _, _, num_patches, _ = patches.shape
        if num_patches > self.pos_embed.size(1):
            raise ValueError(
                f"input length produces P={num_patches}, but encoder was initialized for "
                f"P={self.pos_embed.size(1)}"
            )

        tokens = self.patch_embed(patches)  # [B, C, P, D]
        tokens = tokens.reshape(bsz * channels, num_patches, self.d_model)
        tokens = tokens + self.pos_embed[:, :num_patches, :]
        tokens = self.encoder(tokens)  # [B*C, P, D]
        tokens = self.norm(tokens)
        h1 = tokens.reshape(bsz, channels, num_patches, self.d_model).contiguous()
        return {
            "H1": h1,
            "patches": patches,
            "patch_meta": meta,
        }
