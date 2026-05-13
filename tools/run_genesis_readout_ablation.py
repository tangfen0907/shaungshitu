import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_Genesis import LOCAL_CONFIG_OVERRIDES
from train import run_dataset_preset


def main():
    overrides = dict(LOCAL_CONFIG_OVERRIDES)
    overrides.update(
        {
            "readout_mode": "attn_topk_max",
            "topk_ratio": 0.1,
            "topk_k": 0,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "num_workers": 0,
        }
    )
    run_dataset_preset("Genesis", "genesis_readout_ablation", overrides, argparse.Namespace())


if __name__ == "__main__":
    main()
