"""
Run multiple training experiments automatically and save results to CSV.

This script:
1. Defines a grid of hyperparameter combinations
2. Runs `python main.py train ...` for each combination
3. Shows stdout live during training
4. Parses the final epoch metrics
5. Saves all experiment results to a CSV file
6. Saves one log file per run
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a grid of streamflow training experiments and save results."
    )

    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable to use for training runs."
    )
    parser.add_argument(
        "--main-script",
        type=str,
        default="main.py",
        help="Path to main.py training script."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="experiment_results.csv",
        help="Path to output CSV file."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="experiment_logs",
        help="Directory to store logs for each run."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for each experiment."
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[30, 60, 90],
        help="Sequence lengths to test."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="Batch sizes to test."
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-3, 5e-4, 1e-4],
        help="Learning rates to test."
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[64],
        help="Hidden sizes to test."
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        nargs="+",
        default=[1],
        help="Number of LSTM layers to test."
    )
    parser.add_argument(
        "--dropouts",
        type=float,
        nargs="+",
        default=[0.1],
        help="Dropout values to test."
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        nargs="*",
        default=[],
        help="Any additional arguments to pass to main.py train."
    )

    return parser.parse_args()


def safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def normalize_output_for_parsing(text: str) -> str:
    """
    Clean training stdout so regex parsing still works even if some metrics
    wrap to the next line.

    Example of wrapped output:
    Epoch 001/005 | Train Loss: ... | Val NSE: 0.6277Train KGE: 0.6044 | Val KGE:
    0.6305

    This function inserts missing separators and flattens line breaks.
    """
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    # If Val NSE and Train KGE got glued together
    cleaned = re.sub(r"(Val NSE:\s*[-+eE0-9\.]+)(Train KGE:)", r"\1 | \2", cleaned)

    # If Val KGE: is followed by a newline and then the number
    cleaned = re.sub(r"(Val KGE:\s*)\n\s*([-+eE0-9\.]+)", r"\1\2", cleaned)

    # Flatten remaining newlines to spaces for robust regex parsing
    cleaned = cleaned.replace("\n", " ")

    # Collapse repeated spaces
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned


def parse_final_metrics(stdout_text: str) -> Dict[str, Optional[float]]:
    """
    Parse the final epoch metrics from training stdout.

    Expected style:
    Epoch 005/010 | Train Loss: 0.1004 | Val Loss: 0.0742 |
    Train NSE: 0.5565 | Val NSE: 0.6206 | Train KGE: 0.6142 | Val KGE: 0.6185

    Handles wrapped lines as well.
    """
    metrics = {
        "train_loss": None,
        "val_loss": None,
        "train_nse": None,
        "val_nse": None,
        "train_kge": None,
        "val_kge": None,
        "final_epoch": None,
        "total_epochs": None,
    }

    cleaned = normalize_output_for_parsing(stdout_text)

    pattern = re.compile(
        r"Epoch\s*\[?(?P<epoch>\d+)\s*/\s*(?P<total>\d+)\]?"
        r".*?Train Loss:\s*(?P<train_loss>[-+eE0-9\.]+)"
        r".*?Val Loss:\s*(?P<val_loss>[-+eE0-9\.]+)"
        r".*?Train NSE:\s*(?P<train_nse>[-+eE0-9\.]+)"
        r".*?Val NSE:\s*(?P<val_nse>[-+eE0-9\.]+)"
        r".*?Train KGE:\s*(?P<train_kge>[-+eE0-9\.]+)"
        r".*?Val KGE:\s*(?P<val_kge>[-+eE0-9\.]+)",
        re.IGNORECASE
    )

    matches = list(pattern.finditer(cleaned))
    if not matches:
        return metrics

    last = matches[-1]
    metrics["final_epoch"] = int(last.group("epoch"))
    metrics["total_epochs"] = int(last.group("total"))
    metrics["train_loss"] = safe_float(last.group("train_loss"))
    metrics["val_loss"] = safe_float(last.group("val_loss"))
    metrics["train_nse"] = safe_float(last.group("train_nse"))
    metrics["val_nse"] = safe_float(last.group("val_nse"))
    metrics["train_kge"] = safe_float(last.group("train_kge"))
    metrics["val_kge"] = safe_float(last.group("val_kge"))

    return metrics


def build_command(
    python_exe: str,
    main_script: str,
    seq_len: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    extra_args: List[str],
) -> List[str]:
    """
    Build the subprocess command for one experiment.
    """
    cmd = [
        python_exe,
        main_script,
        "train",
        "--seq-len", str(seq_len),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--learning-rate", str(learning_rate),
        "--hidden-size", str(hidden_size),
        "--num-layers", str(num_layers),
        "--dropout", str(dropout),
    ]

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def run_one_experiment(
    run_id: int,
    python_exe: str,
    main_script: str,
    log_dir: Path,
    seq_len: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    extra_args: List[str],
) -> Dict[str, object]:
    """
    Run a single training experiment and return a result dictionary.
    Shows live output while also capturing it for parsing and logging.
    """
    cmd = build_command(
        python_exe=python_exe,
        main_script=main_script,
        seq_len=seq_len,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        extra_args=extra_args,
    )

    print("=" * 100)
    print(f"Run {run_id}")
    print("Command:", " ".join(cmd))

    start_time = datetime.now()
    stdout_lines: List[str] = []

    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as process:
            assert process.stdout is not None

            for line in process.stdout:
                print(line, end="")   # live display
                stdout_lines.append(line)

            process.wait()
            return_code = process.returncode

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return_code = -999

    end_time = datetime.now()
    runtime_seconds = (end_time - start_time).total_seconds()

    stdout_text = "".join(stdout_lines)

    log_path = log_dir / f"run_{run_id:03d}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write("OUTPUT:\n")
        f.write(stdout_text)

    metrics = parse_final_metrics(stdout_text)

    result = {
        "run_id": run_id,
        "timestamp_start": start_time.isoformat(timespec="seconds"),
        "timestamp_end": end_time.isoformat(timespec="seconds"),
        "runtime_seconds": runtime_seconds,
        "return_code": return_code,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "train_loss": metrics["train_loss"],
        "val_loss": metrics["val_loss"],
        "train_nse": metrics["train_nse"],
        "val_nse": metrics["val_nse"],
        "train_kge": metrics["train_kge"],
        "val_kge": metrics["val_kge"],
        "final_epoch": metrics["final_epoch"],
        "total_epochs_reported": metrics["total_epochs"],
        "log_file": str(log_path),
        "command": " ".join(cmd),
    }

    print("-" * 100)
    print(f"Run {run_id} finished")
    print(f"Return code: {return_code}")
    print(f"Runtime (s): {runtime_seconds:.1f}")
    print(
        "Parsed metrics:",
        {
            "train_loss": result["train_loss"],
            "val_loss": result["val_loss"],
            "train_nse": result["train_nse"],
            "val_nse": result["val_nse"],
            "train_kge": result["train_kge"],
            "val_kge": result["val_kge"],
        }
    )

    return result


def append_result_to_csv(csv_path: Path, row: Dict[str, object]) -> None:
    """
    Append one result row to CSV, writing header if file does not exist yet.
    """
    file_exists = csv_path.exists()

    fieldnames = [
        "run_id",
        "timestamp_start",
        "timestamp_end",
        "runtime_seconds",
        "return_code",
        "seq_len",
        "batch_size",
        "epochs",
        "learning_rate",
        "hidden_size",
        "num_layers",
        "dropout",
        "train_loss",
        "val_loss",
        "train_nse",
        "val_nse",
        "train_kge",
        "val_kge",
        "final_epoch",
        "total_epochs_reported",
        "log_file",
        "command",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main() -> None:
    args = parse_args()

    main_script = Path(args.main_script)
    if not main_script.exists():
        raise FileNotFoundError(f"Could not find main script: {main_script}")

    output_csv = Path(args.output_csv)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    combinations = list(
        itertools.product(
            args.seq_lens,
            args.batch_sizes,
            args.learning_rates,
            args.hidden_sizes,
            args.num_layers,
            args.dropouts,
        )
    )

    print(f"Total experiments to run: {len(combinations)}")
    print(f"Results CSV: {output_csv.resolve()}")
    print(f"Logs directory: {log_dir.resolve()}")

    for run_id, (
        seq_len,
        batch_size,
        learning_rate,
        hidden_size,
        num_layers,
        dropout,
    ) in enumerate(combinations, start=1):
        result = run_one_experiment(
            run_id=run_id,
            python_exe=args.python_exe,
            main_script=str(main_script),
            log_dir=log_dir,
            seq_len=seq_len,
            batch_size=batch_size,
            epochs=args.epochs,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            extra_args=args.extra_args,
        )

        append_result_to_csv(output_csv, result)

        # Optional: stop if user interrupted
        if result["return_code"] == -999:
            print("\nExperiment loop stopped by user.")
            break

    print("\nAll requested experiments completed.")
    print(f"Results saved to: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
