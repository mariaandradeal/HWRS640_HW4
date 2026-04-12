"""
Plotting and visualization functions for streamflow prediction.
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import build_dataloaders
from model import create_model
from train import load_checkpoint, inverse_transform_target
from utils import ensure_dir

import matplotlib.dates as mdates

def load_history(history_path: str) -> Dict:
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_history(
    history: Dict,
    output_dir: str = "outputs/figures",
) -> List[str]:
    """
    Plot:
      - train/val loss vs epoch
      - validation NSE vs epoch
    """
    ensure_dir(output_dir)
    saved_paths = []

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(output_dir, "training_validation_loss.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()
    saved_paths.append(loss_path)

    # NSE plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_nse"], label="Train NSE")
    plt.plot(epochs, history["val_nse"], label="Validation NSE")
    plt.xlabel("Epoch")
    plt.ylabel("NSE")
    plt.title("Training and Validation NSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    nse_path = os.path.join(output_dir, "training_validation_nse.png")
    plt.tight_layout()
    plt.savefig(nse_path, dpi=300)
    plt.close()
    saved_paths.append(nse_path)

    return saved_paths


def collect_test_predictions(
    checkpoint_path: str,
    batch_size: int = 64,
) -> Dict:
    """
    Load checkpoint, run on test set, and return predictions plus metadata.
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
            y = batch["y"].cpu().numpy().reshape(-1)
            y_pred = model(x_seq, x_static).cpu().numpy().reshape(-1)

            y = inverse_transform_target(y, log_target=log_target)
            y_pred = inverse_transform_target(y_pred, log_target=log_target)

            all_obs.append(y)
            all_pred.append(y_pred)
            all_basin.extend(batch["basin_id"])
            all_time.extend(batch["pred_time"])

    obs = np.concatenate(all_obs)
    pred = np.concatenate(all_pred)

    return {
        "obs": obs,
        "pred": pred,
        "basin_id": np.array(all_basin),
        "pred_time": np.array(all_time),
    }


def plot_parity(
    obs: np.ndarray,
    pred: np.ndarray,
    output_dir: str = "outputs/figures",
    max_points: int = 5000,
) -> str:
    """
    Parity plot of predicted vs observed streamflow.
    """
    ensure_dir(output_dir)

    if len(obs) > max_points:
        idx = np.random.choice(len(obs), size=max_points, replace=False)
        obs_plot = obs[idx]
        pred_plot = pred[idx]
    else:
        obs_plot = obs
        pred_plot = pred

    plt.figure(figsize=(6, 6))
    plt.scatter(obs_plot, pred_plot, alpha=0.4, s=10)

    min_val = min(obs_plot.min(), pred_plot.min())
    max_val = max(obs_plot.max(), pred_plot.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Observed Streamflow")
    plt.ylabel("Predicted Streamflow")
    plt.title("Parity Plot: Predicted vs Observed")
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "parity_plot.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path


def select_basin_for_timeseries(results: Dict) -> str:
    """
    Select one basin for time series visualization.
    Chooses the basin with the most test samples.
    """
    basin_ids = results["basin_id"]
    unique_basins, counts = np.unique(basin_ids, return_counts=True)
    selected_basin = unique_basins[np.argmax(counts)]
    return str(selected_basin)


def plot_test_timeseries(
    results: Dict,
    basin_id: str = None,
    output_dir: str = "outputs/figures",
    max_points: int = 365,
) -> str:
    """
    Plot observed vs predicted streamflow time series for one basin.
    """
    ensure_dir(output_dir)

    if basin_id is None:
        basin_id = select_basin_for_timeseries(results)

    mask = results["basin_id"] == basin_id
    obs = results["obs"][mask]
    pred = results["pred"][mask]
    times = results["pred_time"][mask]

    # Sort by time just in case
    sort_idx = np.argsort(times)
    obs = obs[sort_idx]
    pred = pred[sort_idx]
    times = times[sort_idx]

    # Limit number of plotted points for readability
    if len(obs) > max_points:
        obs = obs[:max_points]
        pred = pred[:max_points]
        times = times[:max_points]

    plt.figure(figsize=(10, 5))
    plt.plot(times, obs, label="Observed")
    plt.plot(times, pred, label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Streamflow")
    plt.title(f"Observed vs Predicted Streamflow\nTest Basin: {basin_id}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    

    out_path = os.path.join(output_dir, f"timeseries_basin_{basin_id}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path


def generate_all_plots(
    checkpoint_path: str,
    history_path: str = "outputs/metrics/training_history.json",
    output_dir: str = "outputs/figures",
    batch_size: int = 64,
) -> Dict:
    """
    Generate all required plots for the assignment.
    """
    ensure_dir(output_dir)

    saved = {}

    # Training history plots
    if os.path.exists(history_path):
        history = load_history(history_path)
        history_paths = plot_training_history(history, output_dir=output_dir)
        saved["history_plots"] = history_paths
    else:
        saved["history_plots"] = []

    # Test predictions
    results = collect_test_predictions(checkpoint_path, batch_size=batch_size)

    parity_path = plot_parity(results["obs"], results["pred"], output_dir=output_dir)
    saved["parity_plot"] = parity_path

    ts_path = plot_test_timeseries(results, basin_id=None, output_dir=output_dir)
    saved["timeseries_plot"] = ts_path

    return saved
