from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset


def _extract_window(sample: Any) -> Any:
    """
    Support the two dataset return styles used in this repo:
    1. window only
    2. (window, label_window)
    """
    if isinstance(sample, (tuple, list)):
        return sample[0]
    return sample


def _to_channel_first_tensor(
    window: Any,
    *,
    in_channels: int = None,
    seq_len: int = None,
) -> torch.Tensor:
    """
    Normalize each window to float tensor [C, L].
    Existing loaders mostly return [L, C], so we transpose when needed.
    """
    if isinstance(window, np.ndarray):
        tensor = torch.from_numpy(window).float()
    elif isinstance(window, torch.Tensor):
        tensor = window.detach().clone().float()
    else:
        tensor = torch.tensor(window, dtype=torch.float32)

    if tensor.dim() != 2:
        raise ValueError(f"Window tensor should be 2D, got {tuple(tensor.shape)}")

    if in_channels is not None and seq_len is not None:
        in_channels = int(in_channels)
        seq_len = int(seq_len)
        if tensor.shape[0] == seq_len and tensor.shape[1] == in_channels:
            tensor = tensor.transpose(0, 1).contiguous()
        elif tensor.shape[0] == in_channels and tensor.shape[1] == seq_len:
            tensor = tensor.contiguous()
        elif tensor.shape[0] > tensor.shape[1]:
            tensor = tensor.transpose(0, 1).contiguous()
    elif tensor.shape[0] > tensor.shape[1]:
        tensor = tensor.transpose(0, 1).contiguous()

    return tensor


class Stage1AdjacentPairDataset(Dataset):
    """
    Stage 1 anchor-positive sampler.

    It yields exactly one temporal neighbor for each anchor. The default
    direction is "past", so positive_idx = anchor_idx - positive_offset. This
    avoids using future windows when Stage 1 is used as a causal warmup.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        positive_offset: int = 1,
        positive_direction: str = "past",
        active_mask=None,
        in_channels: int = None,
        seq_len: int = None,
    ):
        self.base_dataset = base_dataset
        self.positive_offset = max(1, int(positive_offset))
        self.positive_direction = str(positive_direction).strip().lower()
        self.in_channels = None if in_channels is None else int(in_channels)
        self.seq_len = None if seq_len is None else int(seq_len)

        num_items = int(len(base_dataset))
        if num_items <= self.positive_offset:
            raise RuntimeError(
                "Stage1AdjacentPairDataset requires more windows than positive_offset."
            )

        if self.positive_direction in {"past", "previous", "prev", "behind"}:
            self.anchor_indices = np.arange(self.positive_offset, num_items, dtype=np.int64)
            self.positive_indices = self.anchor_indices - self.positive_offset
        elif self.positive_direction in {"future", "next"}:
            self.anchor_indices = np.arange(0, num_items - self.positive_offset, dtype=np.int64)
            self.positive_indices = self.anchor_indices + self.positive_offset
        else:
            raise ValueError(
                "positive_direction should be one of: past, previous, prev, behind, future, next."
            )

        if active_mask is not None:
            active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
            if active_mask.shape[0] != num_items:
                raise ValueError(
                    "active_mask length should match base_dataset length: "
                    f"{active_mask.shape[0]} vs {num_items}"
                )
            pair_mask = active_mask[self.anchor_indices] & active_mask[self.positive_indices]
            self.anchor_indices = self.anchor_indices[pair_mask]
            self.positive_indices = self.positive_indices[pair_mask]
            if self.anchor_indices.size == 0:
                raise RuntimeError(
                    "Stage1AdjacentPairDataset has no active adjacent pairs after active-pool filtering."
                )

    def __len__(self) -> int:
        return int(self.anchor_indices.shape[0])

    def __getitem__(self, idx: int):
        anchor_idx = int(self.anchor_indices[idx])
        positive_idx = int(self.positive_indices[idx])
        anchor = _to_channel_first_tensor(
            _extract_window(self.base_dataset[anchor_idx]),
            in_channels=self.in_channels,
            seq_len=self.seq_len,
        )
        positive = _to_channel_first_tensor(
            _extract_window(self.base_dataset[positive_idx]),
            in_channels=self.in_channels,
            seq_len=self.seq_len,
        )
        return anchor, positive
