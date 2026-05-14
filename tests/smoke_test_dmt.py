import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.dmt_model import DMTPatchMemoryModel


def main():
    bsz, seq_len, channels = 2, 100, 8
    patch_len = 10
    d_model = 128
    n_memory = 20

    model = DMTPatchMemoryModel(
        in_channels=channels,
        seq_len=seq_len,
        patch_len=patch_len,
        d_model=d_model,
        n_heads=4,
        num_layers=2,
        n_memory=n_memory,
        temperature=0.1,
        topk_ratio=0.05,
    )

    x = torch.randn(bsz, seq_len, channels)

    out = model(x, mode="pretrain")
    assert out["H1"].shape == (bsz, channels, 10, d_model)
    assert out["patches"].shape == (bsz, channels, 10, patch_len)
    assert out["x_hat"].shape == (bsz, seq_len, channels)

    fake_memory = torch.randn(n_memory, d_model)
    model.set_memory(fake_memory)

    out = model(x, mode="memory_train")
    assert out["x_hat"].shape == (bsz, seq_len, channels)
    assert out["attn"].shape == (bsz, channels, 10, n_memory)
    assert out["H1_aug"].shape == (bsz, channels, 10, d_model * 2)

    out = model(x, mode="test")
    assert out["window_score"].shape == (bsz,)
    assert out["LSD"].shape == (bsz, channels, 10)
    assert out["ISD"].shape == (bsz, channels, 10)
    assert out["token_score"].shape == (bsz, channels * 10)

    print("DMT smoke test passed.")


if __name__ == "__main__":
    main()
