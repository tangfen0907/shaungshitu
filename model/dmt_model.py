from typing import Dict, Optional

import torch
from torch import nn

from model.dmt_memory import PrototypeMemory
from model.dmt_patch_transformer import ChannelIndependentPatchTransformer
from utils.dmt_patching import unpatchify_1d


class DMTPatchMemoryModel(nn.Module):
    """
    DMT-M1: View1 single-variable Patch Transformer + K-means Memory.

    Input:
        x_blt: [B, L, C]

    Core representation:
        H1: [B, C, P, D]

    Modes:
        pretrain:      H1 -> decoder_pre -> x_hat [B, L, C]
        memory_train: H1 -> memory read -> decoder_mem -> x_hat [B, L, C]
        test:         memory-guided reconstruction + window_score [B]
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        patch_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        n_memory: int,
        temperature: float = 0.1,
        topk_ratio: float = 0.05,
        memory_init: Optional[torch.Tensor] = None,
        memory_trainable: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)
        self.n_memory = int(n_memory)
        self.temperature = float(temperature)
        self.topk_ratio = float(topk_ratio)

        self.encoder_v1 = ChannelIndependentPatchTransformer(
            in_channels=self.in_channels,
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            d_model=self.d_model,
            n_heads=int(n_heads),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        self.memory_v1 = PrototypeMemory(
            n_memory=self.n_memory,
            d_model=self.d_model,
            temperature=self.temperature,
            init_memory=memory_init,
            trainable=bool(memory_trainable),
        )
        self.decoder_pre = nn.Linear(self.d_model, self.patch_len)
        self.decoder_mem = nn.Linear(self.d_model * 2, self.patch_len)

    @torch.no_grad()
    def set_memory(self, memory: torch.Tensor) -> None:
        self.memory_v1.set_memory(memory)

    def _decode_to_blt(self, patch_hat: torch.Tensor, patch_meta: Dict[str, int]) -> torch.Tensor:
        # patch_hat: [B, C, P, patch_len] -> x_hat_bcl: [B, C, L] -> x_hat_blt: [B, L, C]
        x_hat_bcl = unpatchify_1d(patch_hat, patch_meta)
        return x_hat_bcl.transpose(1, 2).contiguous()

    def _topk_mean(self, token_score: torch.Tensor) -> torch.Tensor:
        # token_score: [B, C*P] -> window_score: [B]
        num_tokens = token_score.size(1)
        k = max(1, int(num_tokens * self.topk_ratio))
        k = min(k, num_tokens)
        return token_score.topk(k=k, dim=1).values.mean(dim=1)

    def forward(self, x_blt: torch.Tensor, mode: str = "pretrain") -> Dict[str, torch.Tensor]:
        if mode not in {"pretrain", "memory_train", "test"}:
            raise ValueError(f"unsupported DMT mode: {mode}")
        if x_blt.dim() != 3:
            raise ValueError(f"x_blt must be [B, L, C], got shape {tuple(x_blt.shape)}")

        enc = self.encoder_v1(x_blt)
        h1 = enc["H1"]  # [B, C, P, D]
        patches = enc["patches"]  # [B, C, P, patch_len]
        patch_meta = enc["patch_meta"]

        if mode == "pretrain":
            patch_hat = self.decoder_pre(h1)  # [B, C, P, patch_len]
            x_hat = self._decode_to_blt(patch_hat, patch_meta)  # [B, L, C]
            return {
                "x_hat": x_hat,
                "H1": h1,
                "patches": patches,
                "patch_hat": patch_hat,
            }

        h1_aug, attn, retrieved = self.memory_v1.read(h1)  # [B, C, P, 2D], [B, C, P, K], [B, C, P, D]
        patch_hat = self.decoder_mem(h1_aug)  # [B, C, P, patch_len]
        x_hat = self._decode_to_blt(patch_hat, patch_meta)  # [B, L, C]

        if mode == "memory_train":
            return {
                "x_hat": x_hat,
                "H1": h1,
                "H1_aug": h1_aug,
                "attn": attn,
                "retrieved": retrieved,
                "patches": patches,
                "patch_hat": patch_hat,
            }

        lsd = self.memory_v1.nearest_distance(h1)  # [B, C, P]
        isd = (patch_hat - patches).pow(2).mean(dim=-1)  # [B, C, P]
        bsz = x_blt.size(0)
        flat_lsd = lsd.reshape(bsz, -1)
        flat_isd = isd.reshape(bsz, -1)
        weights = torch.softmax(flat_lsd / self.temperature, dim=1)
        token_score = weights * flat_isd  # [B, C*P]
        window_score = self._topk_mean(token_score)  # [B]
        return {
            "window_score": window_score,
            "x_hat": x_hat,
            "H1": h1,
            "H1_aug": h1_aug,
            "LSD": lsd,
            "ISD": isd,
            "token_score": token_score,
            "attn": attn,
            "retrieved": retrieved,
            "patches": patches,
            "patch_hat": patch_hat,
        }
