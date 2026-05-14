from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class PrototypeMemory(nn.Module):
    """
    Fixed or trainable prototype memory for DMT-M1 patch tokens.

    memory: [K, D]

    DMT-M1 keeps the memory fixed by default after K-means initialization.
    """

    def __init__(
        self,
        n_memory: int,
        d_model: int,
        temperature: float = 0.1,
        init_memory: Optional[torch.Tensor] = None,
        trainable: bool = False,
    ):
        super().__init__()
        self.n_memory = int(n_memory)
        self.d_model = int(d_model)
        self.temperature = float(temperature)
        self.trainable = bool(trainable)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

        memory = self._build_initial_memory(init_memory)
        if self.trainable:
            self.memory = nn.Parameter(memory)
        else:
            self.register_buffer("memory", memory)

    def _build_initial_memory(self, init_memory: Optional[torch.Tensor]) -> torch.Tensor:
        if init_memory is None:
            memory = torch.empty(self.n_memory, self.d_model, dtype=torch.float32)
            nn.init.xavier_uniform_(memory)
            return memory

        memory = init_memory.detach().clone().float()
        if memory.shape != (self.n_memory, self.d_model):
            raise ValueError(
                f"init_memory must be [{self.n_memory}, {self.d_model}], got {tuple(memory.shape)}"
            )
        return memory

    @torch.no_grad()
    def set_memory(self, init_memory: torch.Tensor) -> None:
        memory = self._build_initial_memory(init_memory).to(self.memory.device)
        if isinstance(self.memory, nn.Parameter):
            self.memory.data.copy_(memory)
        else:
            self.memory.copy_(memory)

    def read(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: Token tensor with shape [..., D].

        Returns:
            aug_tokens: [..., 2D]
            attn: [..., K]
            retrieved: [..., D]
        """
        if tokens.size(-1) != self.d_model:
            raise ValueError(f"expected last dim D={self.d_model}, got {tokens.size(-1)}")

        original_shape = tokens.shape[:-1]
        tokens_flat = tokens.reshape(-1, self.d_model)  # [Q, D]
        memory = self.memory.to(tokens_flat.dtype)

        tokens_norm = F.normalize(tokens_flat, dim=-1)
        memory_norm = F.normalize(memory, dim=-1)
        logits = torch.matmul(tokens_norm, memory_norm.t()) / self.temperature  # [Q, K]
        attn_flat = torch.softmax(logits, dim=-1)  # [Q, K]
        retrieved_flat = torch.matmul(attn_flat, memory)  # [Q, D]
        aug_flat = torch.cat([tokens_flat, retrieved_flat], dim=-1)  # [Q, 2D]

        aug_tokens = aug_flat.reshape(*original_shape, self.d_model * 2)
        attn = attn_flat.reshape(*original_shape, self.n_memory)
        retrieved = retrieved_flat.reshape(*original_shape, self.d_model)
        return aug_tokens, attn, retrieved

    def nearest_distance(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Token tensor with shape [..., D].

        Returns:
            dist: Squared L2 distance to the nearest memory item, shape [...].
        """
        if tokens.size(-1) != self.d_model:
            raise ValueError(f"expected last dim D={self.d_model}, got {tokens.size(-1)}")

        original_shape = tokens.shape[:-1]
        tokens_flat = tokens.reshape(-1, self.d_model)  # [Q, D]
        memory = self.memory.to(tokens_flat.dtype)
        dist = torch.cdist(tokens_flat, memory, p=2).pow(2)  # [Q, K]
        nearest = dist.min(dim=-1).values
        return nearest.reshape(*original_shape)
