    elif args.command == "plot":
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

        if "best_worst_summary" in saved:
            s = saved["best_worst_summary"]
            print("\nBest basin:")
            print(f"  basin_id: {s['best_basin_id']}")
            print(f"  NSE: {s['best_basin_nse']:.4f}")
            print(f"  RMSE: {s['best_basin_rmse']:.4f}")
            print(f"  MAE: {s['best_basin_mae']:.4f}")
            print(f"  plot: {s['best_plot']}")

            print("\nWorst basin:")
            print(f"  basin_id: {s['worst_basin_id']}")
            print(f"  NSE: {s['worst_basin_nse']:.4f}")
            print(f"  RMSE: {s['worst_basin_rmse']:.4f}")
            print(f"  MAE: {s['worst_basin_mae']:.4f}")
            print(f"  plot: {s['worst_plot']}")
