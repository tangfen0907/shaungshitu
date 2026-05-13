import argparse
import os
from typing import Optional

from utils.config import dataset_entry_meta
from utils.run_entry import load_checkpoint_config, run_evaluation


SOURCE_RUN_NAME = "genesis_experiment001"
RUN_NAME = "genesis_test_eval"
CHECKPOINT_NAME = "model_last.pt"

LOCAL_EVAL_OVERRIDES = {
    # Test-set visualization should stay on for Genesis because the test split has labels.
    "enable_stage_visualization": True,
    "visualization_method": "pca",
    # Match the Genesis test-window anomaly ratio: 249 / 6389 = 3.8973%.
    "decision_quantile": 0.98,
}


def _latest_run_name(results_root: str, prefix: str) -> Optional[str]:
    if not os.path.isdir(results_root):
        return None

    candidates = []
    for entry in os.listdir(results_root):
        full_path = os.path.join(results_root, entry)
        if not os.path.isdir(full_path):
            continue
        if not entry.startswith(prefix):
            continue
        checkpoint_path = os.path.join(full_path, CHECKPOINT_NAME)
        if not os.path.isfile(checkpoint_path):
            continue
        suffix = entry[len(prefix):]
        if suffix.isdigit():
            candidates.append((int(suffix), entry))

    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def resolve_checkpoint_path() -> str:
    meta = dataset_entry_meta("Genesis")
    results_root = str(meta["results_root"])

    run_name = str(SOURCE_RUN_NAME).strip()
    if not run_name:
        run_name = _latest_run_name(results_root, "genesis_experiment")
        if not run_name:
            raise FileNotFoundError(
                f"No Genesis checkpoint run found under: {os.path.abspath(results_root)}"
            )

    checkpoint_path = os.path.join(results_root, run_name, CHECKPOINT_NAME)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {os.path.abspath(checkpoint_path)}")
    return checkpoint_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Genesis checkpoint.")
    parser.add_argument("--visualization_method", choices=["pca", "tsne", "umap"], default=None)
    return parser.parse_args()


def build_eval_config(checkpoint_path: str, visualization_method=None):
    config = load_checkpoint_config(checkpoint_path)
    for key, value in LOCAL_EVAL_OVERRIDES.items():
        setattr(config, key, value)
    if visualization_method is not None:
        setattr(config, "visualization_method", str(visualization_method).strip().lower())
    return config


def main():
    args = _parse_args()
    meta = dataset_entry_meta("Genesis")
    checkpoint_path = resolve_checkpoint_path()
    config = build_eval_config(
        checkpoint_path,
        visualization_method=args.visualization_method,
    )
    run_evaluation(
        checkpoint_path=checkpoint_path,
        config=config,
        run_name=RUN_NAME,
        results_root=str(meta["results_root"]),
        experiment_name="Genesis Test Eval",
        show_visualization_artifacts=True,
    )


if __name__ == "__main__":
    main()
