"""Unified local smoke test entrypoint."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.smoke_test_all_presets import main as smoke_all_presets
from tools.smoke_test_patch_relation_encoder import main as smoke_patch_relation_encoder


def main():
    smoke_patch_relation_encoder()
    smoke_all_presets()


if __name__ == "__main__":
    main()
