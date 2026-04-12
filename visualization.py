"""
Plotting and visualization functions for streamflow prediction.
"""

import json
import os
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import build_dataloaders
from train import load_checkpoint, inverse_transform_target
from utils import ensure_dir


def load_history(history_path: str) -> Dict:
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_clean_style():
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
    })


def plot_training_history(
    history: Dict,
    output_dir: str = "outputs/figures",
) -> List[str]:
    """
    Plot training/validation loss and NSE curves.
    """
    ensure_dir(output_dir)
    _apply_clean_style()

    saved_paths = []
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend(frameon=True)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "training_validation_loss.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_paths.append(loss_path)

    # NSE
    plt.figure()
    plt.plot(epochs, history["train_nse"], label="Train NSE")
    plt.plot(epochs, history["val_nse"], label="Validation NSE")
    plt.xlabel("Epoch")
    plt.ylabel("NSE")
    plt.title("Training and Validation NSE")
    plt.legend(frameon=True)
    plt.tight_layout()
    nse_path = os.path.join(output_dir, "training_validation_nse.png")
    plt.savefig(nse_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_paths.append(nse_path)

    return saved_paths


def collect_test_predictions(
    checkpoint_path: str,
    batch_size: int = 64,
) -> Dict:
    """
    Run the saved model on the test set and collect predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)

    seq_len = checkpoint["meta"]["seq_len"]
    log_target = checkpoint["meta"]["log_target"]

    _, _, test_loader, _ = build_dataloaders(
        seq_len=seq_len,
        batch_size=batch_size,
        log_target=log_target,
    )

    all_obs = []
    all_pred = []
    all_basin = []
    all_time = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x_seq = batch["x_seq"].to(device)
            x_static = batch["x_static"].to(device)

            y_true = batch["y"].cpu().numpy().reshape(-1)
            y_pred = model(x_seq, x_static).cpu().numpy().reshape(-1)

            y_true = inverse_transform_target(y_true, log_target=log_target)
            y_pred = inverse_transform_target(y_pred, log_target=log_target)

            all_obs.append(y_true)
            all_pred.append(y_pred)
            all_basin.extend(batch["basin_id"])
            all_time.extend(batch["pred_time"])

    obs = np.concatenate(all_obs)
    pred = np.concatenate(all_pred)
    pred_time = np.array(all_time, dtype="datetime64[D]")
    basin_id = np.array(all_basin)

    return {
        "obs": obs,
        "pred": pred,
        "basin_id": basin_id,
        "pred_time": pred_time,
    }


def plot_parity(
    obs: np.ndarray,
    pred: np.ndarray,
    output_dir: str = "outputs/figures",
    max_points: int = 4000,
) -> str:
    """
    Create a cleaner parity plot.
    """
    ensure_dir(output_dir)
    _apply_clean_style()

    if len(obs) > max_points:
        idx = np.random.choice(len(obs), size=max_points, replace=False)
        obs_plot = obs[idx]
        pred_plot = pred[idx]
    else:
        obs_plot = obs
        pred_plot = pred

    min_val = float(min(obs_plot.min(), pred_plot.min()))
    max_val = float(max(obs_plot.max(), pred_plot.max()))

    plt.figure(figsize=(6.5, 6))
    plt.scatter(obs_plot, pred_plot, s=12, alpha=0.35)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.5)
    plt.xlabel("Observed Streamflow")
    plt.ylabel("Predicted Streamflow")
    plt.title("Parity Plot: Predicted vs Observed")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "parity_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path


def select_basin_for_timeseries(results: Dict) -> str:
    """
    Pick the basin with the most test samples.
    """
    basin_ids = results["basin_id"]
    unique_basins, counts = np.unique(basin_ids, return_counts=True)
    return str(unique_basins[np.argmax(counts)])


def plot_test_timeseries(
    results: Dict,
    basin_id: Optional[str] = None,
    output_dir: str = "outputs/figures",
    max_points: int = 365,
    date_format: str = "%Y-%m",
) -> str:
    """
    Plot observed vs predicted streamflow for one basin.
    By default, plot only the first 365 points for readability.
    """
    ensure_dir(output_dir)
    _apply_clean_style()

    if basin_id is None:
        basin_id = select_basin_for_timeseries(results)

    mask = results["basin_id"] == basin_id
    obs = results["obs"][mask]
    pred = results["pred"][mask]
    times = results["pred_time"][mask]

    sort_idx = np.argsort(times)
    obs = obs[sort_idx]
    pred = pred[sort_idx]
    times = times[sort_idx]

    if len(obs) > max_points:
        obs = obs[:max_points]
        pred = pred[:max_points]
        times = times[:max_points]

    times = np.array(times, dtype="datetime64[D]").astype("datetime64[ms]").astype(object)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(times, obs, label="Observed", linewidth=2.2)
    ax.plot(times, pred, label="Predicted", linewidth=2.0, alpha=0.9)

    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow")
    ax.set_title(f"Observed vs Predicted Streamflow\nTest Basin: {basin_id}")
    ax.legend(frameon=True)

    # Cleaner date axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    out_path = os.path.join(output_dir, f"timeseries_basin_{basin_id}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path


def plot_test_timeseries_full_years(
    results: Dict,
    basin_id: Optional[str] = None,
    output_dir: str = "outputs/figures",
) -> str:
    """
    Plot the full available basin test period with yearly ticks.
    """
    ensure_dir(output_dir)
    _apply_clean_style()

    if basin_id is None:
        basin_id = select_basin_for_timeseries(results)

    mask = results["basin_id"] == basin_id
    obs = results["obs"][mask]
    pred = results["pred"][mask]
    times = results["pred_time"][mask]

    sort_idx = np.argsort(times)
    obs = obs[sort_idx]
    pred = pred[sort_idx]
    times = times[sort_idx]

    times = np.array(times, dtype="datetime64[D]").astype("datetime64[ms]").astype(object)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(times, obs, label="Observed", linewidth=2.2)
    ax.plot(times, pred, label="Predicted", linewidth=2.0, alpha=0.9)

    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow")
    ax.set_title(f"Observed vs Predicted Streamflow (Full Test Period)\nTest Basin: {basin_id}")
    ax.legend(frameon=True)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    out_path = os.path.join(output_dir, f"timeseries_full_basin_{basin_id}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path


def generate_all_plots(
    checkpoint_path: str,
    history_path: str = "outputs/metrics/training_history.json",
    output_dir: str = "outputs/figures",
    batch_size: int = 64,
) -> Dict:
    """
    Generate all main plots.
    """
    ensure_dir(output_dir)

    saved = {}

    if os.path.exists(history_path):
        history = load_history(history_path)
        saved["history_plots"] = plot_training_history(history, output_dir=output_dir)
    else:
        saved["history_plots"] = []

    results = collect_test_predictions(checkpoint_path, batch_size=batch_size)

    saved["parity_plot"] = plot_parity(results["obs"], results["pred"], output_dir=output_dir)
    saved["timeseries_plot"] = plot_test_timeseries(
        results,
        basin_id=None,
        output_dir=output_dir,
        max_points=365,
    )
    saved["timeseries_full_plot"] = plot_test_timeseries_full_years(
        results,
        basin_id=None,
        output_dir=output_dir,
    )

    return saved
