"""Shape smoke test for the token-first patch/relation dual encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.axis_view_encoder import PatchRelationDualEncoder


def _assert_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    actual = tuple(int(dim) for dim in tensor.shape)
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def main() -> int:
    batch_size = 2
    seq_len = 100
    latent_dim = 64
    encoder = PatchRelationDualEncoder(
        latent_dim=latent_dim,
        tcn_layers=(64, 128, 128),
        patch_len=16,
        patch_stride=8,
        patch_blocks=3,
        relation_layers=1,
        relation_heads=4,
        kernels=(3, 5, 7),
        dropout=0.1,
        activation="relu",
        readout="attn_topk_max",
        topk_ratio=0.1,
        topk_k=0,
    )
    encoder.eval()
    patch_count = encoder.num_patches(seq_len)

    for num_variables in (9, 18, 25, 38, 51):
        x = torch.randn(batch_size, num_variables, seq_len)
        with torch.no_grad():
            out = encoder(x)
        _assert_shape("h1_patch", out["h1_patch"], (batch_size, num_variables, patch_count, encoder.d1))
        _assert_shape("h2_var", out["h2_var"], (batch_size, num_variables, encoder.d2))
        _assert_shape("h2_patch", out["h2_patch"], (batch_size, num_variables, patch_count, encoder.d2))
        _assert_shape("z1_global", out["z1_global"], (batch_size, latent_dim))
        _assert_shape("z2_global", out["z2_global"], (batch_size, latent_dim))
        print(
            f"[PASS] N={num_variables} | P={patch_count} | "
            f"h1_patch={tuple(out['h1_patch'].shape)} | "
            f"h2_var={tuple(out['h2_var'].shape)} | "
            f"z=({tuple(out['z1_global'].shape)}, {tuple(out['z2_global'].shape)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
