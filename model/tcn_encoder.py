from typing import Iterable, Optional, Sequence, Union

import torch
import torch.nn as nn


def build_activation(name: str) -> nn.Module:
    key = str(name).strip().lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported TCN activation: {name}")


class Chomp1d(nn.Module):
    """
    裁掉因 padding 多出来的未来信息，保证卷积严格因果。
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class CausalConvBlock(nn.Module):
    """
    TCN 残差块。

    结构：
    CausalConv1d -> GELU -> Dropout -> CausalConv1d -> GELU -> Dropout + Residual
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation
        activation_name = str(activation).strip().lower()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.net = nn.Sequential(
            self.conv1,
            Chomp1d(padding),
            build_activation(activation_name),
            nn.Dropout(dropout),
            self.conv2,
            Chomp1d(padding),
            build_activation(activation_name),
            nn.Dropout(dropout),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        self.out_activation = build_activation(activation_name)
        self._init_weights()

    def _init_weights(self):
        # Explicit small-variance init helps keep early training stable
        # in this no-normalization TCN stack.
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        if isinstance(self.shortcut, nn.Conv1d):
            nn.init.normal_(self.shortcut.weight, mean=0.0, std=0.01)
            if self.shortcut.bias is not None:
                nn.init.zeros_(self.shortcut.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        return self.out_activation(self.net(x) + residual)


class AttentivePooling1d(nn.Module):
    """Temporal attentive pooling that returns a window-level vector [B, C]."""

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=-1)
        return torch.sum(x * weights, dim=-1)


class TCNEncoder(nn.Module):
    """
    基于因果膨胀卷积的编码器。

    输入：
        x, 形状 [B, C, L]
    输出：
        z, 形状 [B, D]
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        tcn_layers: Sequence[int],
        kernel_size: Union[int, Sequence[int]] = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        use_attentive_pooling: bool = False,
        dilations: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        if not isinstance(tcn_layers, Iterable) or len(tuple(tcn_layers)) == 0:
            raise ValueError("tcn_layers 必须是非空层宽列表，例如 (64, 128, 128)。")

        hidden_channels = list(tcn_layers)
        if isinstance(kernel_size, Iterable) and not isinstance(kernel_size, (str, bytes)):
            kernel_values = [int(value) for value in kernel_size]
            if len(kernel_values) != len(hidden_channels):
                raise ValueError(
                    "kernel_size sequence should have the same length as tcn_layers: "
                    f"got {len(kernel_values)} vs {len(hidden_channels)}."
                )
        else:
            kernel_values = [int(kernel_size)] * len(hidden_channels)
        if any(value <= 0 for value in kernel_values):
            raise ValueError("All TCN kernel sizes should be positive integers.")
        self.kernel_sizes = tuple(kernel_values)

        if dilations is None:
            dilation_values = [2 ** layer_idx for layer_idx in range(len(hidden_channels))]
        else:
            dilation_values = [int(value) for value in dilations]
            if len(dilation_values) != len(hidden_channels):
                raise ValueError(
                    "dilations should have the same length as tcn_layers: "
                    f"got {len(dilation_values)} vs {len(hidden_channels)}."
                )
            if any(value <= 0 for value in dilation_values):
                raise ValueError("All TCN dilation values should be positive integers.")
        self.dilations = tuple(dilation_values)

        blocks = []
        prev_channels = in_channels
        for layer_idx, out_channels in enumerate(hidden_channels):
            dilation = self.dilations[layer_idx]
            block_kernel_size = self.kernel_sizes[layer_idx]
            blocks.append(
                CausalConvBlock(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    kernel_size=block_kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation,
                )
            )
            prev_channels = out_channels

        self.tcn = nn.Sequential(*blocks)
        self.use_attentive_pooling = bool(use_attentive_pooling)
        self.pool = (
            AttentivePooling1d(prev_channels)
            if self.use_attentive_pooling
            else nn.AdaptiveAvgPool1d(1)
        )
        self.projection = nn.Linear(prev_channels, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TCNEncoder 期望输入为 [B, C, L]，实际得到 {tuple(x.shape)}")

        features = self.tcn(x)
        pooled = self.pool(features)
        if not self.use_attentive_pooling:
            pooled = pooled.squeeze(-1)
        z = self.projection(pooled)
        return z
