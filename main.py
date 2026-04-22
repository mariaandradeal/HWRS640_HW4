import argparse

from data import summarize_dataset
from train import train_model, evaluate_model
from visualization import generate_all_plots, generate_exploratory_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HWRS640 Assignment 4 CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # summarize-data
    subparsers.add_parser(
        "summarize-data",
        help="Print dataset summary",
    )

    # explore-data
    explore_parser = subparsers.add_parser(
        "explore-data",
        help="Generate exploratory plots",
    )
    explore_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/exploration",
        help="Directory to save exploratory figures",
    )

    explore_parser.add_argument(
        "--n-basins",
        type=int,
        default=4,
        help="Number of basins to plot hydrographs for",
    )

    # train
    train_parser = subparsers.add_parser(
        "train",
        help="Train the LSTM model",
    )
    train_parser.add_argument("--seq-len", type=int, default=60, help="Input sequence length")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size")
    train_parser.add_argument("--num-layers", type=int, default=1, help="Number of LSTM layers")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    train_parser.add_argument(
        "--static-hidden-size",
        type=int,
        default=32,
        help="Hidden size for static attribute encoder",
    )
    train_parser.add_argument(
        "--fusion-hidden-size",
        type=int,
        default=32,
        help="Hidden size for fusion MLP",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save checkpoints, metrics, and figures",
    )

    # evaluate
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a saved model on the test set",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )

    # plot
    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate evaluation figures",
    )
    plot_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    plot_parser.add_argument(
        "--history",
        type=str,
        default="outputs/metrics/training_history.json",
        help="Path to saved training history JSON",
    )
    plot_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save figures",
    )
    plot_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prediction/plotting",
    )

    return parser


def run_command(args: argparse.Namespace) -> None:
    if args.command == "summarize-data":
        summarize_dataset()
        return

    if args.command == "explore-data":
        saved = generate_exploratory_plots(
            output_dir=args.output_dir,
            n_basins=args.n_basins,
        )

    if args.command == "train":
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
        print(f"Best checkpoint: {results['best_checkpoint_path']}")
        print(f"History saved to: {results['history_path']}")
        return

    if args.command == "evaluate":
        results = evaluate_model(
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
        )

        print("\n===== Test Results =====")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
        print("========================\n")
        return

    if args.command == "plot":
        saved = generate_all_plots(
            checkpoint_path=args.checkpoint,
            history_path=args.history,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

        print("\nGenerated plots:")
        for key, value in saved.items():
            if key != "best_worst_summary":
                print(f"{key}: {value}")

        summary = saved.get("best_worst_summary")
        if summary is not None:
            print("\nBest basin:")
            print(f"  basin_id: {summary['best_basin_id']}")
            print(f"  NSE: {summary['best_basin_nse']:.4f}")
            print(f"  RMSE: {summary['best_basin_rmse']:.4f}")
            print(f"  MAE: {summary['best_basin_mae']:.4f}")
            print(f"  plot: {summary['best_plot']}")

            print("\nWorst basin:")
            print(f"  basin_id: {summary['worst_basin_id']}")
            print(f"  NSE: {summary['worst_basin_nse']:.4f}")
            print(f"  RMSE: {summary['worst_basin_rmse']:.4f}")
            print(f"  MAE: {summary['worst_basin_mae']:.4f}")
            print(f"  plot: {summary['worst_plot']}")
        return

    raise ValueError(f"Unknown command: {args.command}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
