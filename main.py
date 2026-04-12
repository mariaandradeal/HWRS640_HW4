import argparse

from data import summarize_dataset, build_dataloaders
from model import create_model, count_trainable_parameters
from train import train_model, evaluate_model
from visualization import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description="HWRS640 Assignment 4 CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # summarize-data
    subparsers.add_parser("summarize-data", help="Print dataset summary")

    # debug-data
    debug_data_parser = subparsers.add_parser("debug-data", help="Debug dataloaders")
    debug_data_parser.add_argument("--seq-len", type=int, default=60)
    debug_data_parser.add_argument("--batch-size", type=int, default=32)

    # debug-model
    debug_model_parser = subparsers.add_parser("debug-model", help="Debug model forward pass")
    debug_model_parser.add_argument("--seq-len", type=int, default=60)
    debug_model_parser.add_argument("--batch-size", type=int, default=32)

    # train
    train_parser = subparsers.add_parser("train", help="Train the LSTM model")
    train_parser.add_argument("--seq-len", type=int, default=60)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--hidden-size", type=int, default=64)
    train_parser.add_argument("--num-layers", type=int, default=1)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--static-hidden-size", type=int, default=32)
    train_parser.add_argument("--fusion-hidden-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--output-dir", type=str, default="outputs")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved model on the test set")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--batch-size", type=int, default=64)

    # plot
    plot_parser = subparsers.add_parser("plot", help="Generate figures")
    plot_parser.add_argument("--checkpoint", type=str, required=True)
    plot_parser.add_argument("--history", type=str, default="outputs/metrics/training_history.json")
    plot_parser.add_argument("--output-dir", type=str, default="outputs/figures")
    plot_parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    if args.command == "summarize-data":
        summarize_dataset()

    elif args.command == "debug-data":
        train_loader, val_loader, test_loader, meta = build_dataloaders(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
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

    elif args.command == "debug-model":
        train_loader, _, _, meta = build_dataloaders(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
        batch = next(iter(train_loader))

        model = create_model(
            num_dynamic_features=len(meta["dynamic_vars"]),
            num_static_features=len(meta["static_vars"]),
            hidden_size=64,
            num_layers=1,
            dropout=0.1,
            static_hidden_size=32,
            fusion_hidden_size=32,
        )

        y_pred = model(batch["x_seq"], batch["x_static"])

        print("Input x_seq shape:", batch["x_seq"].shape)
        print("Input x_static shape:", batch["x_static"].shape)
        print("Output y_pred shape:", y_pred.shape)
        print("Trainable parameters:", count_trainable_parameters(model))

    elif args.command == "train":
        results = train_model(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            static_hidden_size=args.static_hidden_size,
            fusion_hidden_size=args.fusion_hidden_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            seed=args.seed,
            output_dir=args.output_dir,
        )

        print("\nTraining complete.")
        print("Best checkpoint:", results["best_checkpoint_path"])
        print("History saved to:", results["history_path"])

    elif args.command == "evaluate":
        results = evaluate_model(
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
        )

        print("\n===== Test Results =====")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
        print("========================\n")

    elif args.command == "plot":
        saved = generate_all_plots(
            checkpoint_path=args.checkpoint,
            history_path=args.history,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

        print("\nGenerated plots:")
        for key, value in saved.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
