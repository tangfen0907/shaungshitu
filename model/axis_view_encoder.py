from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tcn_encoder import build_activation


class FlattenedSegmentConvBranch(nn.Module):
    """
    Conv1d branch over a right-aligned flattened block window.

    Input:
        x_flat_cf: [B, 1, T * C]
    Output:
        features: [B, T, out_channels]
    """

    def __init__(
        self,
        in_channels_per_step: int,
        num_blocks: int,
        out_channels: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.in_channels_per_step = max(1, int(in_channels_per_step))
        self.num_blocks = max(1, int(num_blocks))
        self.out_channels = int(out_channels)
        self.kernel_size = self.in_channels_per_step * self.num_blocks
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.in_channels_per_step,
        )
        self.activation = build_activation(str(activation).strip().lower())
        self.dropout = nn.Dropout(float(dropout))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x_flat_cf: torch.Tensor, first_block_flat: torch.Tensor) -> torch.Tensor:
        if x_flat_cf.dim() != 3 or x_flat_cf.size(1) != 1:
            raise ValueError(f"FlattenedSegmentConvBranch expects [B, 1, T*C], got {tuple(x_flat_cf.shape)}")
        if first_block_flat.dim() != 3 or first_block_flat.size(1) != 1:
            raise ValueError(
                "first_block_flat should be [B, 1, C], "
                f"got {tuple(first_block_flat.shape)}."
            )
        if first_block_flat.size(-1) != self.in_channels_per_step:
            raise ValueError(
                f"Expected first block width C={self.in_channels_per_step}, "
                f"got C={int(first_block_flat.size(-1))}."
            )

        if self.num_blocks > 1:
            # Repeat the earliest complete block on the left so every original
            # time point still gets one complete right-aligned history window.
            prefix = first_block_flat.repeat(1, 1, self.num_blocks - 1)
            conv_input = torch.cat([prefix, x_flat_cf], dim=-1)
        else:
            conv_input = x_flat_cf

        # [B, 1, T*C + (K-1)*C] -> [B, out_channels, T]
        features = self.conv(conv_input)
        features = self.activation(features)
        features = self.dropout(features)
        # [B, out_channels, T] -> [B, T, out_channels]
        return features.transpose(1, 2).contiguous()


class PointwiseDualEncoder(nn.Module):
    """
    Point-level dual-view encoder for multivariate time-series windows.

    Input:
        x: [B, C, T]

    View1:
        F1: [B, T, 3C + L]
        H1: [B, T, d_model]

    View2:
        x_flat: [B, T*C, 1]
        F2: [B, T, current_out + short_out + long_out]
        H2: [B, T, d_model]
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        history_len: int = 20,
        current_out: int = 8,
        short_out: int = 16,
        long_out: int = 16,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.in_channels = max(1, int(in_channels))
        self.d_model = int(d_model)
        self.history_len = max(1, int(history_len))
        self.short_blocks = max(1, self.history_len // 2)
        self.long_blocks = self.history_len
        self.current_out = int(current_out)
        self.short_out = int(short_out)
        self.long_out = int(long_out)
        self.view1_dim = 3 * self.in_channels + self.history_len
        self.view2_dim = self.current_out + self.short_out + self.long_out

        relation_hidden = max(8, self.in_channels)
        self.view1_relation_summary = nn.Sequential(
            nn.LayerNorm(self.in_channels),
            nn.Linear(self.in_channels, relation_hidden),
            build_activation(str(activation).strip().lower()),
            nn.Linear(relation_hidden, 1),
        )
        self.view1_projection = nn.Linear(self.view1_dim, self.d_model)

        self.view2_current = FlattenedSegmentConvBranch(
            in_channels_per_step=self.in_channels,
            num_blocks=1,
            out_channels=self.current_out,
            dropout=dropout,
            activation=activation,
        )
        self.view2_short = FlattenedSegmentConvBranch(
            in_channels_per_step=self.in_channels,
            num_blocks=self.short_blocks,
            out_channels=self.short_out,
            dropout=dropout,
            activation=activation,
        )
        self.view2_long = FlattenedSegmentConvBranch(
            in_channels_per_step=self.in_channels,
            num_blocks=self.long_blocks,
            out_channels=self.long_out,
            dropout=dropout,
            activation=activation,
        )
        self.view2_projection = nn.Linear(self.view2_dim, self.d_model)

    @staticmethod
    def _right_aligned_mean(x: torch.Tensor, window: int) -> torch.Tensor:
        # x: [B, C, T] -> [B, C, T]
        window = max(1, int(window))
        if window == 1:
            return x
        padded = F.pad(x, (window - 1, 0), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=window, stride=1)

    def _view1_features(self, x: torch.Tensor) -> torch.Tensor:
        # Horizontal summaries: [B, C, T] -> [B, T, C] each.
        current = x.transpose(1, 2).contiguous()
        short = self._right_aligned_mean(x, self.short_blocks).transpose(1, 2).contiguous()
        long = self._right_aligned_mean(x, self.long_blocks).transpose(1, 2).contiguous()

        # One full-variable relation summary per original time point:
        # [B, T, C] -> [B, T, 1].
        relation_summary = self.view1_relation_summary(current)
        # Gather a right-aligned L-length history of relation summaries:
        # [B, T, 1] -> [B, 1, T] -> [B, T, L].
        relation_cf = relation_summary.transpose(1, 2).contiguous()
        relation_padded = F.pad(relation_cf, (self.history_len - 1, 0), mode="replicate")
        relation_history = relation_padded.unfold(
            dimension=-1,
            size=self.history_len,
            step=1,
        ).squeeze(1)

        # [B, T, C + C + C + L] = [B, T, 3C + L].
        return torch.cat([current, short, long, relation_history], dim=-1)

    def _view2_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, channels, length = x.shape
        x_time = x.transpose(1, 2).contiguous()  # [B, T, C]
        x_flat = x_time.reshape(batch_size, length * channels, 1)  # [B, T*C, 1]
        x_flat_cf = x_flat.transpose(1, 2).contiguous()  # [B, 1, T*C]
        first_block_flat = x_time[:, :1, :].reshape(batch_size, 1, channels)

        current = self.view2_current(x_flat_cf, first_block_flat)
        short = self.view2_short(x_flat_cf, first_block_flat)
        long = self.view2_long(x_flat_cf, first_block_flat)
        f2 = torch.cat([current, short, long], dim=-1)
        return {
            "x_flat": x_flat,
            "current": current,
            "short": short,
            "long": long,
            "F2": f2,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"PointwiseDualEncoder expects [B, C, T], got {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"PointwiseDualEncoder expected C={self.in_channels}, got C={int(x.size(1))}"
            )

        f1 = self._view1_features(x)
        view2 = self._view2_features(x)
        f2 = view2["F2"]
        h1 = self.view1_projection(f1)
        h2 = self.view2_projection(f2)

        return {
            "F1": f1,
            "H1": h1,
            "x_flat": view2["x_flat"],
            "F2": f2,
            "H2": h2,
            "view2_current": view2["current"],
            "view2_short": view2["short"],
            "view2_long": view2["long"],
        }
