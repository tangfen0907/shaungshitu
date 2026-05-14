import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def patchify_1d(
    x_bcl: torch.Tensor,
    patch_len: int,
    stride: int = None,
    pad: bool = True,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Split each variable's 1D window into non-overlapping temporal patches.

    Args:
        x_bcl: Tensor with shape [B, C, L].
        patch_len: Length of each temporal patch.
        stride: First DMT-M1 version only supports stride == patch_len.
        pad: If True, right-pad time dimension so every point is covered.

    Returns:
        patches: Tensor with shape [B, C, P, patch_len].
        meta: Metadata needed by unpatchify_1d.
    """
    if x_bcl.dim() != 3:
        raise ValueError(f"x_bcl must be [B, C, L], got shape {tuple(x_bcl.shape)}")
    patch_len = int(patch_len)
    if patch_len <= 0:
        raise ValueError("patch_len must be positive")

    stride = patch_len if stride is None else int(stride)
    if stride != patch_len:
        raise NotImplementedError("DMT-M1 only supports non-overlap patches: stride == patch_len")

    bsz, channels, original_len = x_bcl.shape
    if original_len <= 0:
        raise ValueError("time length L must be positive")

    if pad:
        num_patches = int(math.ceil(original_len / float(patch_len)))
        padded_len = num_patches * patch_len
        pad_len = padded_len - original_len
        if pad_len > 0:
            x_bcl = F.pad(x_bcl, (0, pad_len))
    else:
        if original_len % patch_len != 0:
            raise ValueError("pad=False requires L to be divisible by patch_len")
        num_patches = original_len // patch_len
        padded_len = original_len
        pad_len = 0

    patches = x_bcl.reshape(bsz, channels, num_patches, patch_len).contiguous()
    meta = {
        "original_len": int(original_len),
        "padded_len": int(padded_len),
        "patch_len": int(patch_len),
        "stride": int(stride),
        "pad_len": int(pad_len),
        "num_patches": int(num_patches),
    }
    return patches, meta


def unpatchify_1d(patches: torch.Tensor, meta: Dict[str, int]) -> torch.Tensor:
    """
    Rebuild a [B, C, L] window from non-overlapping temporal patches.

    Args:
        patches: Tensor with shape [B, C, P, patch_len].
        meta: Metadata returned by patchify_1d.

    Returns:
        x_hat_bcl: Tensor with shape [B, C, original_len].
    """
    if patches.dim() != 4:
        raise ValueError(f"patches must be [B, C, P, patch_len], got shape {tuple(patches.shape)}")

    bsz, channels, num_patches, patch_len = patches.shape
    expected_patch_len = int(meta["patch_len"])
    expected_patches = int(meta["num_patches"])
    if patch_len != expected_patch_len or num_patches != expected_patches:
        raise ValueError(
            "patch shape does not match metadata: "
            f"patches={tuple(patches.shape)}, meta P={expected_patches}, patch_len={expected_patch_len}"
        )

    padded_len = int(meta["padded_len"])
    original_len = int(meta["original_len"])
    x_hat_bcl = patches.reshape(bsz, channels, padded_len).contiguous()
    return x_hat_bcl[:, :, :original_len].contiguous()
