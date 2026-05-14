import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DMT-M1: single-variable Patch Transformer + K-means Memory"
    )
    parser.add_argument("--mode", choices=["pretrain", "init_memory", "memory_train", "test"], required=True)
    parser.add_argument("--dataset", type=str, default="SKAB")
    parser.add_argument("--data_path", type=str, default="./dataset/SKAB")
    parser.add_argument("--win_size", type=int, default=100)
    parser.add_argument("--input_c", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--patch_len", type=int, default=10)
    parser.add_argument("--n_memory", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lambda_ent", type=float, default=0.01)
    parser.add_argument("--topk_ratio", type=float, default=0.05)
    parser.add_argument("--anormly_ratio", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model_save_path", type=str, default="checkpoints_dmt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_memory_tokens", type=int, default=200000)
    parser.add_argument("--memory_trainable", action="store_true")
    parser.add_argument("--scaler_fit_mode", type=str, default="train")
    parser.add_argument("--cache_windows", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)

    # Debug-only limits. Leave at 0 for normal full-dataset runs.
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from solver_dmt import DMTSolver

    solver = DMTSolver(args)
    solver.run()


if __name__ == "__main__":
    main()
