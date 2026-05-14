import math

import numpy as np
import torch


def _patch_count(length: int, patch_len: int, patch_stride: int) -> int:
    length = int(length)
    patch_len = max(1, int(patch_len))
    patch_stride = max(1, int(patch_stride))
    if length <= patch_len:
        return 1
    return int(math.ceil(float(length - patch_len) / float(patch_stride))) + 1


def point_mask_to_patch_mask(mask, patch_len: int, patch_stride: int):
    if isinstance(mask, torch.Tensor):
        if mask.dim() != 3:
            raise ValueError(f"point mask should be [B, N, L], got {tuple(mask.shape)}")
        batch_size, num_variables, length = mask.shape
        patch_count = _patch_count(length, patch_len, patch_stride)
        patch_mask = torch.zeros(
            batch_size,
            num_variables,
            patch_count,
            dtype=torch.bool,
            device=mask.device,
        )
        bool_mask = mask.bool()
        for patch_idx in range(patch_count):
            start = patch_idx * max(1, int(patch_stride))
            end = min(start + max(1, int(patch_len)), int(length))
            patch_mask[:, :, patch_idx] = bool_mask[:, :, start:end].any(dim=-1)
        return patch_mask

    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 3:
        raise ValueError(f"point mask should be [B, N, L], got {arr.shape}")
    batch_size, num_variables, length = arr.shape
    patch_count = _patch_count(length, patch_len, patch_stride)
    patch_mask = np.zeros((batch_size, num_variables, patch_count), dtype=bool)
    for patch_idx in range(patch_count):
        start = patch_idx * max(1, int(patch_stride))
        end = min(start + max(1, int(patch_len)), int(length))
        patch_mask[:, :, patch_idx] = np.any(arr[:, :, start:end], axis=-1)
    return patch_mask
