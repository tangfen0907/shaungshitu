import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tcn_encoder import build_activation


def _as_layer_tuple(tcn_layers: Sequence[int]) -> tuple:
    if not isinstance(tcn_layers, Iterable):
        raise ValueError("tcn_layers should be a non-empty sequence.")
    layers = tuple(int(value) for value in tcn_layers)
    if not layers or any(value <= 0 for value in layers):
        raise ValueError("tcn_layers should contain positive integers.")
    return layers


def _resolve_kernel_size(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size <= 0:
        raise ValueError("kernel_size should be a positive integer.")
    if kernel_size % 2 == 0:
        raise ValueError("Same-padding temporal conv expects an odd kernel_size, e.g. 5 or 7.")
    return kernel_size


class TokenAttentionReadout(nn.Module):
    """
    Learnable token readout.

    Input:
        tokens: [B, L, H]
    Output:
        z: [B, D]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = None,
        hidden_dim: int = None,
        dropout: float = 0.0,
        readout: str = "attention",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim) if output_dim is not None else int(input_dim)
        self.readout = str(readout or "attention").strip().lower()
        if self.readout not in {"attention", "attn", "last"}:
            raise ValueError("readout should be 'attention' or 'last'.")

        attn_hidden = int(hidden_dim) if hidden_dim is not None else max(16, self.input_dim // 2)
        self.attention_score = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, attn_hidden),
            nn.Tanh(),
            nn.Dropout(float(dropout)),
            nn.Linear(attn_hidden, 1),
        )
        self.projection = (
            nn.Identity()
            if self.output_dim == self.input_dim
            else nn.Linear(self.input_dim, self.output_dim)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"TokenAttentionReadout expects [B, L, H], got {tuple(tokens.shape)}")
        if tokens.size(-1) != self.input_dim:
            raise ValueError(
                f"TokenAttentionReadout expected H={self.input_dim}, got H={int(tokens.size(-1))}"
            )

        # tokens: [B, L, H]
        if self.readout == "last":
            context = tokens[:, -1, :]
        else:
            # scores: [B, L, 1], weights: [B, L, 1]
            scores = self.attention_score(tokens)
            weights = torch.softmax(scores, dim=1)
            # context: [B, H]
            context = torch.sum(tokens * weights, dim=1)
        # z: [B, D]
        return self.projection(context)


class TemporalAttentionReadout(TokenAttentionReadout):
    """Alias kept for call sites that want a temporal-specific name."""


class AnomalySensitiveReadout(nn.Module):
    """
    Attention + local-evidence token readout.

    This module avoids reducing a token sequence with only global attention or
    averaging. The default mode keeps three complementary summaries:

        attention context: [B, L, H] -> [B, H]
        top-k mean context: [B, L, H] -> [B, H]
        max context: [B, L, H] -> [B, H]

    The concatenated evidence [B, 3H] is projected back to [B, D].

    Input:
        tokens: [B, L, H]
    Output:
        z: [B, D]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = None,
        hidden_dim: int = None,
        dropout: float = 0.0,
        mode: str = "attn_topk_max",
        topk_ratio: float = 0.1,
        topk_k: int = 0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim) if output_dim is not None else int(input_dim)
        self.mode = str(mode or "attn_topk_max").strip().lower()
        self.topk_ratio = float(topk_ratio)
        self.topk_k = int(topk_k)
        if self.mode not in {"attn_topk_max", "attention", "attn", "last", "topk_max"}:
            raise ValueError(
                "readout mode should be one of: 'attn_topk_max', 'attention', 'last', 'topk_max'."
            )

        attn_hidden = int(hidden_dim) if hidden_dim is not None else max(16, self.input_dim // 2)
        self.attention_score = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, attn_hidden),
            nn.Tanh(),
            nn.Dropout(float(dropout)),
            nn.Linear(attn_hidden, 1),
        )
        if self.mode == "attn_topk_max":
            projection_in = self.input_dim * 3
        elif self.mode == "topk_max":
            projection_in = self.input_dim * 2
        else:
            projection_in = self.input_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(projection_in),
            nn.Dropout(float(dropout)),
            nn.Linear(projection_in, self.output_dim),
        )

    def _topk_count(self, length: int) -> int:
        if self.topk_k > 0:
            return max(1, min(int(self.topk_k), int(length)))
        ratio = self.topk_ratio if self.topk_ratio > 0 else 0.1
        return max(1, min(int(length), int(math.ceil(float(length) * ratio))))

    def _attention_context(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, L, H] -> scores/weights: [B, L, 1] -> context: [B, H]
        scores = self.attention_score(tokens)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(tokens * weights, dim=1)

    def _topk_context(self, tokens: torch.Tensor) -> torch.Tensor:
        # Rank tokens by their strongest channel response, then average top-k.
        # tokens: [B, L, H] -> evidence: [B, L]
        k = self._topk_count(tokens.size(1))
        evidence = torch.amax(torch.abs(tokens), dim=-1)
        indices = torch.topk(evidence, k=k, dim=1, largest=True, sorted=False).indices
        gather_index = indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
        # selected: [B, K, H] -> [B, H]
        selected = torch.gather(tokens, dim=1, index=gather_index)
        return torch.mean(selected, dim=1)

    def _max_context(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, L, H] -> [B, H]
        return torch.amax(tokens, dim=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"AnomalySensitiveReadout expects [B, L, H], got {tuple(tokens.shape)}")
        if tokens.size(-1) != self.input_dim:
            raise ValueError(
                f"AnomalySensitiveReadout expected H={self.input_dim}, got H={int(tokens.size(-1))}"
            )

        # tokens: [B, L, H]
        if self.mode == "last":
            evidence = tokens[:, -1, :]
        elif self.mode in {"attention", "attn"}:
            evidence = self._attention_context(tokens)
        elif self.mode == "topk_max":
            evidence = torch.cat(
                [self._topk_context(tokens), self._max_context(tokens)],
                dim=-1,
            )
        else:
            evidence = torch.cat(
                [
                    self._attention_context(tokens),
                    self._topk_context(tokens),
                    self._max_context(tokens),
                ],
                dim=-1,
            )
        # evidence: [B, H] or [B, 2H] or [B, 3H] -> z: [B, D]
        return self.projection(evidence)


class SamePadResidualConvBlock(nn.Module):
    """
    Ordinary same-padding residual temporal Conv1d block.

    There is no dilation and no causal chomp. Both convolutions use
    dilation=1 and padding=(kernel_size - 1) / 2, so the time length is kept.

    Input:
        x: [B, C_in, T]
    Output:
        y: [B, C_out, T]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dropout: float = 0.1,
        activation: str = "relu",
        norm: str = "batch",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _resolve_kernel_size(kernel_size)
        padding = self.kernel_size // 2
        activation_name = str(activation).strip().lower()
        norm_key = str(norm or "batch").strip().lower()

        self.conv1 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=padding,
            dilation=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=padding,
            dilation=1,
        )
        self.norm1 = self._build_norm(self.out_channels, norm_key)
        self.norm2 = self._build_norm(self.out_channels, norm_key)
        self.activation1 = build_activation(activation_name)
        self.activation2 = build_activation(activation_name)
        self.dropout1 = nn.Dropout(float(dropout))
        self.dropout2 = nn.Dropout(float(dropout))
        self.shortcut = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, dilation=1)
        )
        self.out_activation = build_activation(activation_name)
        self._init_weights()

    @staticmethod
    def _build_norm(channels: int, norm: str) -> nn.Module:
        if norm in {"batch", "bn", "batchnorm"}:
            return nn.BatchNorm1d(int(channels))
        if norm in {"layer", "ln", "layernorm"}:
            return nn.GroupNorm(1, int(channels))
        if norm in {"none", "identity"}:
            return nn.Identity()
        raise ValueError("norm should be one of: batch, layer, none.")

    def _init_weights(self):
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
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.activation1(y)
        y = self.dropout1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.activation2(y)
        y = self.dropout2(y)
        return self.out_activation(y + residual)


class MultiScalePatchConvBlock(nn.Module):
    """
    Multi-scale ordinary Conv1d block over patch tokens.

    Input:
        tokens: [B, P, D]
    Output:
        tokens: [B, P, D]
    """

    def __init__(
        self,
        dim: int,
        kernels: Sequence[int] = (3, 5, 7),
        dropout: float = 0.1,
        activation: str = "relu",
        norm: str = "batch",
    ):
        super().__init__()
        self.dim = int(dim)
        self.kernels = tuple(_resolve_kernel_size(k) for k in kernels)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.dim,
                    out_channels=self.dim,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    dilation=1,
                )
                for kernel in self.kernels
            ]
        )
        self.mix = nn.Conv1d(self.dim * len(self.branches), self.dim, kernel_size=1, dilation=1)
        norm_key = str(norm or "batch").strip().lower()
        if norm_key in {"batch", "bn", "batchnorm"}:
            self.norm = nn.BatchNorm1d(self.dim)
        elif norm_key in {"layer", "ln", "layernorm"}:
            self.norm = nn.GroupNorm(1, self.dim)
        elif norm_key in {"none", "identity"}:
            self.norm = nn.Identity()
        else:
            raise ValueError("norm should be one of: batch, layer, none.")
        self.activation = build_activation(str(activation).strip().lower())
        self.dropout = nn.Dropout(float(dropout))
        self._init_weights()

    def _init_weights(self):
        for branch in self.branches:
            nn.init.normal_(branch.weight, mean=0.0, std=0.01)
            if branch.bias is not None:
                nn.init.zeros_(branch.bias)
        nn.init.normal_(self.mix.weight, mean=0.0, std=0.01)
        if self.mix.bias is not None:
            nn.init.zeros_(self.mix.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"MultiScalePatchConvBlock expects [B, P, D], got {tuple(tokens.shape)}")
        if tokens.size(-1) != self.dim:
            raise ValueError(f"MultiScalePatchConvBlock expected D={self.dim}, got D={int(tokens.size(-1))}")

        # [B, P, D] -> [B, D, P]
        conv_input = tokens.transpose(1, 2).contiguous()
        # each branch: [B, D, P], concat: [B, D * K, P]
        multi_scale = torch.cat([branch(conv_input) for branch in self.branches], dim=1)
        mixed = self.mix(multi_scale)
        mixed = self.norm(mixed)
        mixed = self.activation(mixed)
        mixed = self.dropout(mixed)
        # [B, D, P] -> [B, P, D]
        return tokens + mixed.transpose(1, 2).contiguous()


class ChannelIndependentPatchTemporalEncoder(nn.Module):
    """
    View1 core: channel-independent patch-level temporal encoder.

    Semantics:
        Each variable is encoded as its own patch sequence. Variables are not
        mixed in the first layer; the same patch encoder is shared across all
        variables and all datasets.

    Input:
        x: [B, N, T]
    Intermediate:
        reshape: [B, N, T] -> [B * N, T]
        patchify: [B * N, T] -> [B * N, P, L_patch]
        patch embedding: [B * N, P, L_patch] -> [B * N, P, D1]
        multi-scale ordinary conv over patches: [B * N, P, D1] -> [B * N, P, D1]
    Output:
        h1_patch: [B, N, P, D1]
    """

    def __init__(
        self,
        patch_len: int = 16,
        patch_stride: int = 8,
        dim: int = 64,
        num_blocks: int = 3,
        kernels: Sequence[int] = (3, 5, 7),
        dropout: float = 0.1,
        activation: str = "relu",
        norm: str = "batch",
    ):
        super().__init__()
        self.patch_len = max(1, int(patch_len))
        self.patch_stride = max(1, int(patch_stride))
        self.dim = int(dim)
        self.num_blocks = max(1, int(num_blocks))
        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_len),
            nn.Linear(self.patch_len, self.dim),
            build_activation(str(activation).strip().lower()),
            nn.Dropout(float(dropout)),
        )
        self.patch_blocks = nn.Sequential(
            *[
                MultiScalePatchConvBlock(
                    dim=self.dim,
                    kernels=kernels,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                )
                for _ in range(self.num_blocks)
            ]
        )

    def num_patches(self, length: int) -> int:
        length = int(length)
        if length <= self.patch_len:
            return 1
        return int(math.ceil(float(length - self.patch_len) / float(self.patch_stride))) + 1

    def _patchify(self, per_variable: torch.Tensor) -> torch.Tensor:
        # per_variable: [B * N, T]
        length = int(per_variable.size(-1))
        patch_count = self.num_patches(length)
        total_len = (patch_count - 1) * self.patch_stride + self.patch_len
        if total_len > length:
            per_variable = F.pad(per_variable, (0, total_len - length), mode="replicate")
        # [B * N, T_pad] -> [B * N, P, L_patch]
        return per_variable.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"ChannelIndependentPatchTemporalEncoder expects [B, N, T], got {tuple(x.shape)}")

        batch_size, num_variables, length = x.shape
        # [B, N, T] -> [B * N, T]
        per_variable = x.reshape(batch_size * num_variables, length)
        # [B * N, T] -> [B * N, P, L_patch]
        patches = self._patchify(per_variable)
        # [B * N, P, L_patch] -> [B * N, P, D1]
        patch_tokens = self.patch_embedding(patches)
        # [B * N, P, D1] -> [B * N, P, D1]
        patch_tokens = self.patch_blocks(patch_tokens)
        patch_count = int(patch_tokens.size(1))
        # [B * N, P, D1] -> [B, N, P, D1]
        return patch_tokens.reshape(batch_size, num_variables, patch_count, self.dim)


def _choose_num_heads(dim: int, preferred: int = 4) -> int:
    dim = int(dim)
    for heads in (preferred, 8, 4, 2, 1):
        if heads > 0 and dim % heads == 0:
            return heads
    return 1


class VariableTokenRelationEncoder(nn.Module):
    """
    View2 core: variable-token relation encoder.

    Semantics:
        N is treated as the number of tokens. The relation encoder never uses
        Linear(N, D) or Conv1d(in_channels=N), so the same module can process
        different datasets with different variable counts.

    Input:
        patch_tokens: [B, N, P, D1]
    Intermediate:
        patch readout per variable: [B * N, P, D1] -> [B * N, Dv]
        var_tokens: [B, N, Dv]
        TransformerEncoder over variable tokens: [B, N, D2] -> [B, N, D2]
        fusion back to patches: [B, N, P, D1 + D2] -> [B, N, P, D2]
    Output:
        h2_var: [B, N, D2]
        h2_patch: [B, N, P, D2]
    """

    def __init__(
        self,
        input_dim: int,
        relation_dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
        readout: str = "attn_topk_max",
        topk_ratio: float = 0.1,
        topk_k: int = 0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.relation_dim = int(relation_dim)
        self.variable_patch_readout = AnomalySensitiveReadout(
            input_dim=self.input_dim,
            output_dim=self.relation_dim,
            dropout=dropout,
            mode=readout,
            topk_ratio=topk_ratio,
            topk_k=topk_k,
        )
        heads = _choose_num_heads(self.relation_dim, preferred=int(num_heads))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.relation_dim,
            nhead=heads,
            dim_feedforward=max(self.relation_dim * 4, 128),
            dropout=float(dropout),
            activation="gelu" if str(activation).strip().lower() == "gelu" else "relu",
            batch_first=True,
            norm_first=True,
        )
        self.relation_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, int(num_layers)),
        )
        self.relation_norm = nn.LayerNorm(self.relation_dim)
        self.patch_fusion = nn.Sequential(
            nn.LayerNorm(self.input_dim + self.relation_dim),
            nn.Linear(self.input_dim + self.relation_dim, self.relation_dim),
            build_activation(str(activation).strip().lower()),
            nn.Dropout(float(dropout)),
        )

    def forward(self, patch_tokens: torch.Tensor) -> tuple:
        if patch_tokens.dim() != 4:
            raise ValueError(f"VariableTokenRelationEncoder expects [B, N, P, D1], got {tuple(patch_tokens.shape)}")
        if patch_tokens.size(-1) != self.input_dim:
            raise ValueError(
                f"VariableTokenRelationEncoder expected D1={self.input_dim}, got D1={int(patch_tokens.size(-1))}"
            )

        batch_size, num_variables, patch_count, _ = patch_tokens.shape
        # [B, N, P, D1] -> [B * N, P, D1]
        per_variable_patches = patch_tokens.reshape(batch_size * num_variables, patch_count, self.input_dim)
        # [B * N, P, D1] -> [B * N, Dv] -> [B, N, D2]
        var_tokens = self.variable_patch_readout(per_variable_patches).reshape(
            batch_size,
            num_variables,
            self.relation_dim,
        )
        # [B, N, D2] -> [B, N, D2]
        h2_var = self.relation_norm(self.relation_encoder(var_tokens))
        # [B, N, D2] -> [B, N, P, D2]
        relation_context = h2_var.unsqueeze(2).expand(-1, -1, patch_count, -1)
        # [B, N, P, D1 + D2] -> [B, N, P, D2]
        h2_patch = self.patch_fusion(torch.cat([patch_tokens, relation_context], dim=-1))
        return h2_var, h2_patch


class PatchRelationDualEncoder(nn.Module):
    """
    Token-first dual-view MTS encoder.

    Core representations:
        h1_patch: [B, N, P, D1]
            Channel-independent patch temporal tokens.
        h2_input_patch: [B, N, P, D1]
            Independent View2 patch tokens from the raw input.
        h2_var: [B, N, D2]
            Variable-token relation/system state tokens.
        h2_patch: [B, N, P, D2]
            Relation-aware patch tokens.

    Compatibility outputs:
        z1_global: [B, latent_dim]
        z2_global: [B, latent_dim]

    Input:
        x: [B, N, T]
    Output:
        dict with h1_patch, h2_var, h2_patch, z1_global, z2_global.
    """

    def __init__(
        self,
        latent_dim: int,
        tcn_layers: Sequence[int],
        patch_len: int = 16,
        patch_stride: int = 8,
        patch_blocks: int = 3,
        relation_layers: int = 1,
        relation_heads: int = 4,
        kernels: Sequence[int] = (3, 5, 7),
        dropout: float = 0.1,
        activation: str = "relu",
        readout: str = "attn_topk_max",
        topk_ratio: float = 0.1,
        topk_k: int = 0,
    ):
        super().__init__()
        layers = _as_layer_tuple(tcn_layers)
        self.latent_dim = int(latent_dim)
        self.d1 = int(layers[0])
        self.d2 = int(layers[0])
        self.patch_len = max(1, int(patch_len))
        self.patch_stride = max(1, int(patch_stride))
        self.kernels = tuple(_resolve_kernel_size(k) for k in kernels)

        self.view1_patch_encoder = ChannelIndependentPatchTemporalEncoder(
            patch_len=self.patch_len,
            patch_stride=self.patch_stride,
            dim=self.d1,
            num_blocks=patch_blocks,
            kernels=self.kernels,
            dropout=dropout,
            activation=activation,
        )
        self.view2_patch_encoder = ChannelIndependentPatchTemporalEncoder(
            patch_len=self.patch_len,
            patch_stride=self.patch_stride,
            dim=self.d1,
            num_blocks=patch_blocks,
            kernels=self.kernels,
            dropout=dropout,
            activation=activation,
        )
        self.view2_relation_encoder = VariableTokenRelationEncoder(
            input_dim=self.d1,
            relation_dim=self.d2,
            num_layers=relation_layers,
            num_heads=relation_heads,
            dropout=dropout,
            activation=activation,
            readout=readout,
            topk_ratio=topk_ratio,
            topk_k=topk_k,
        )
        self.z1_readout = AnomalySensitiveReadout(
            input_dim=self.d1,
            output_dim=self.latent_dim,
            dropout=dropout,
            mode=readout,
            topk_ratio=topk_ratio,
            topk_k=topk_k,
        )
        self.z2_readout = AnomalySensitiveReadout(
            input_dim=self.d2,
            output_dim=self.latent_dim,
            dropout=dropout,
            mode=readout,
            topk_ratio=topk_ratio,
            topk_k=topk_k,
        )

    def num_patches(self, length: int) -> int:
        return self.view1_patch_encoder.num_patches(length)

    def forward(self, x: torch.Tensor) -> dict:
        if x.dim() != 3:
            raise ValueError(f"PatchRelationDualEncoder expects [B, N, T], got {tuple(x.shape)}")

        batch_size, num_variables, _, = x.shape
        # View1 core: [B, N, T] -> [B, N, P, D1]
        h1_patch = self.view1_patch_encoder(x)
        patch_count = int(h1_patch.size(2))
        # View2 core: [B, N, T] -> [B, N, P, D1] -> [B, N, D2], [B, N, P, D2]
        h2_input_patch = self.view2_patch_encoder(x)
        h2_var, h2_patch = self.view2_relation_encoder(h2_input_patch)

        # Compatibility readouts only. Core representations stay token-level.
        # [B, N, P, D1] -> [B, N * P, D1] -> [B, latent_dim]
        z1_global = self.z1_readout(h1_patch.reshape(batch_size, num_variables * patch_count, self.d1))
        # [B, N, P, D2] -> [B, N * P, D2] -> [B, latent_dim]
        z2_global = self.z2_readout(h2_patch.reshape(batch_size, num_variables * patch_count, self.d2))

        return {
            "h1_patch": h1_patch,
            "h2_input_patch": h2_input_patch,
            "h2_var": h2_var,
            "h2_patch": h2_patch,
            "z1_global": z1_global,
            "z2_global": z2_global,
            "z1": z1_global,
            "z2": z2_global,
        }
