import argparse
from data import summarize_dataset, build_dataloaders


def main():
    parser = argparse.ArgumentParser(description="HWRS640 Assignment 4 CLI")
    parser.add_argument(
        "command",
        choices=["summarize-data", "debug-data", "train", "evaluate", "plot"],
        help="Command to run",
    )

    args = parser.parse_args()

    if args.command == "summarize-data":
        summarize_dataset()

    elif args.command == "debug-data":
        train_loader, val_loader, test_loader, meta = build_dataloaders(
            seq_len=60,
            batch_size=32,
        )

        print("Train samples:", meta["n_train_samples"])
        print("Val samples:", meta["n_val_samples"])
        print("Test samples:", meta["n_test_samples"])

        batch = next(iter(train_loader))
        print("x_seq shape:", batch["x_seq"].shape)
        print("x_static shape:", batch["x_static"].shape)
        print("y shape:", batch["y"].shape)
        print("Example basin:", batch["basin_id"][0])
        print("Example pred time:", batch["pred_time"][0])

    elif args.command == "train":
        print("Train command not implemented yet.")

    elif args.command == "evaluate":
        print("Evaluate command not implemented yet.")

    elif args.command == "plot":
        print("Plot command not implemented yet.")


if __name__ == "__main__":
    main()
