import argparse
import os

from utils.run_entry import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint without retraining.")
    parser.add_argument("checkpoint", type=str, help="Path to model_last.pt.")
    parser.add_argument("--decision_quantile", type=float, default=None)
    parser.add_argument("--run_name", type=str, default="eval")
    parser.add_argument("--results_root", type=str, default="")
    parser.add_argument("--experiment_name", type=str, default="Evaluation")
    parser.add_argument("--print_visualization_artifacts", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.decision_quantile is not None:
        overrides["decision_quantile"] = float(args.decision_quantile)
    results_root = args.results_root.strip() or None
    run_evaluation(
        checkpoint_path=os.path.abspath(args.checkpoint),
        config=overrides or None,
        run_name=args.run_name,
        results_root=results_root,
        experiment_name=args.experiment_name,
        show_visualization_artifacts=bool(args.print_visualization_artifacts),
    )


if __name__ == "__main__":
    main()
