def main():
    raise SystemExit(
        "core/main.py is a deprecated legacy entry.\n"
        "Use train.py / eval.py or the dataset-specific main_*.py scripts instead.\n"
        "The active Stage 2 path is train.py -> utils/run_entry.py -> trainer/solver.py -> trainer/stage2.py."
    )


if __name__ == "__main__":
    main()
