"""
Plotting and visualization functions for streamflow prediction.
"""

import json
import os
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import build_dataloaders
from minicamels import MiniCamels
from train import load_checkpoint, inverse_transform_target
from utils import ensure_dir, nse, rmse, mae


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
    ensure_dir(output_dir)
    _apply_clean_style()

    saved_paths = []
    epochs = np.arange(1, len(history["train_loss"]) + 1)

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
    Load checkpoint, run on test set, and collect predictions.
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
    basin_id = np.array(all_basin)
    pred_time = np.array(all_time, dtype="datetime64[D]")

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


def compute_per_basin_metrics(results: Dict) -> pd.DataFrame:
    """
    Compute per-basin test metrics.
    """
    basin_ids = np.unique(results["basin_id"])
    rows = []

    for basin in basin_ids:
        mask = results["basin_id"] == basin
        obs = results["obs"][mask]
        pred = results["pred"][mask]

        if len(obs) < 2:
            continue

        rows.append({
            "basin_id": str(basin),
            "n_samples": int(len(obs)),
            "nse": float(nse(obs, pred)),
            "rmse": float(rmse(obs, pred)),
            "mae": float(mae(obs, pred)),
        })

    df = pd.DataFrame(rows).sort_values("nse", ascending=False).reset_index(drop=True)
    return df


def get_basin_metadata() -> pd.DataFrame:
    """
    Load basin coordinates and static attributes from MiniCamels.
    """
    ds = MiniCamels()
    basins_df = ds.basins().copy()
    attrs_df = ds.attributes().copy()

    attrs_df["basin_id"] = basins_df["basin_id"].astype(str).values
    meta_df = attrs_df.copy()

    if "basin_name" in basins_df.columns:
        meta_df["basin_name"] = basins_df["basin_name"].values

    return meta_df


def get_best_and_worst_basin(metrics_df: pd.DataFrame) -> Dict:
    valid_df = metrics_df.dropna(subset=["nse"]).copy()
    best = valid_df.iloc[0].to_dict()
    worst = valid_df.iloc[-1].to_dict()

    return {
        "best": best,
        "worst": worst,
    }


def plot_test_timeseries(
    results: Dict,
    basin_id: str,
    output_dir: str = "outputs/figures",
    max_points: int = 365,
    title_prefix: str = "",
    filename_prefix: str = "",
) -> str:
    ensure_dir(output_dir)
    _apply_clean_style()

    mask = results["basin_id"] == basin_id
    obs = results["obs"][mask]
    pred = results["pred"][mask]
    times = results["pred_time"][mask]

    sort_idx = np.argsort(times)
    obs = obs[sort_idx]
    pred = pred[sort_idx]
    times = times[sort_idx]

    basin_nse = nse(obs, pred)
    basin_rmse = rmse(obs, pred)
    basin_mae = mae(obs, pred)

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
    ax.set_title(
        f"{title_prefix} Basin: {basin_id}\n"
        f"NSE = {basin_nse:.3f}, RMSE = {basin_rmse:.3f}, MAE = {basin_mae:.3f}"
    )
    ax.legend(frameon=True)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{filename_prefix}basin_{basin_id}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path


def plot_best_and_worst_basins(
    results: Dict,
    metrics_df: pd.DataFrame,
    output_dir: str = "outputs/figures",
    max_points: int = 365,
) -> Dict:
    ensure_dir(output_dir)

    summary = get_best_and_worst_basin(metrics_df)
    best = summary["best"]
    worst = summary["worst"]

    best_path = plot_test_timeseries(
        results=results,
        basin_id=best["basin_id"],
        output_dir=output_dir,
        max_points=max_points,
        title_prefix="Best Test",
        filename_prefix="best_",
    )

    worst_path = plot_test_timeseries(
        results=results,
        basin_id=worst["basin_id"],
        output_dir=output_dir,
        max_points=max_points,
        title_prefix="Worst Test",
        filename_prefix="worst_",
    )

    return {
        "best_basin_id": best["basin_id"],
        "best_basin_nse": best["nse"],
        "best_basin_rmse": best["rmse"],
        "best_basin_mae": best["mae"],
        "best_plot": best_path,
        "worst_basin_id": worst["basin_id"],
        "worst_basin_nse": worst["nse"],
        "worst_basin_rmse": worst["rmse"],
        "worst_basin_mae": worst["mae"],
        "worst_plot": worst_path,
    }


def plot_nse_map(
    metrics_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    output_dir: str = "outputs/figures",
) -> str:
    """
    Map of basin NSE across all catchments.
    """
    ensure_dir(output_dir)
    _apply_clean_style()

    df = metrics_df.merge(meta_df, on="basin_id", how="left")

    plt.figure(figsize=(11, 6))
    sc = plt.scatter(
        df["lon"],
        df["lat"],
        c=df["nse"],
        s=55,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.3,
    )
    plt.colorbar(sc, label="Test NSE")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Basin-Level Test NSE Across MiniCAMELS")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "basin_nse_map.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path


def plot_ranked_nse(
    metrics_df: pd.DataFrame,
    output_dir: str = "outputs/figures",
) -> str:
    """
    Ranked basin NSE plot.
    """
    ensure_dir(output_dir)
    _apply_clean_style()

    df = metrics_df.sort_values("nse", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(df) + 1), df["nse"], marker="o", linewidth=1.5)
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xlabel("Basin Rank")
    plt.ylabel("Test NSE")
    plt.title("Ranked Basin-Level Test NSE")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "ranked_basin_nse.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path


def plot_nse_vs_aridity(
    metrics_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    output_dir: str = "outputs/figures",
) -> str:
    """
    Scatter plot of NSE versus aridity.
    """
    ensure_dir(output_dir)
    _apply_clean_style()

    df = metrics_df.merge(meta_df[["basin_id", "aridity"]], on="basin_id", how="left")

    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(df["aridity"], df["nse"], s=45, alpha=0.8)
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xlabel("Aridity")
    plt.ylabel("Test NSE")
    plt.title("Basin-Level Test NSE vs Aridity")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "nse_vs_aridity.png")
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
    Generate all main plots, including basin-level variability plots.
    """
    ensure_dir(output_dir)
    saved = {}

    if os.path.exists(history_path):
        history = load_history(history_path)
        saved["history_plots"] = plot_training_history(history, output_dir=output_dir)
    else:
        saved["history_plots"] = []

    results = collect_test_predictions(checkpoint_path, batch_size=batch_size)
    metrics_df = compute_per_basin_metrics(results)
    meta_df = get_basin_metadata()

    saved["parity_plot"] = plot_parity(results["obs"], results["pred"], output_dir=output_dir)

    best_worst = plot_best_and_worst_basins(
        results=results,
        metrics_df=metrics_df,
        output_dir=output_dir,
        max_points=365,
    )
    saved["best_worst_summary"] = best_worst

    saved["nse_map"] = plot_nse_map(metrics_df, meta_df, output_dir=output_dir)
    saved["ranked_nse"] = plot_ranked_nse(metrics_df, output_dir=output_dir)
    saved["nse_vs_aridity"] = plot_nse_vs_aridity(metrics_df, meta_df, output_dir=output_dir)

    # Also save the full metrics table for inspection
    metrics_csv = os.path.join(output_dir, "per_basin_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    saved["per_basin_metrics_csv"] = metrics_csv

    return saved
