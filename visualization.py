"""
Plotting and visualization functions for streamflow prediction.
Includes:
  - Exploratory plots for Problem 1
  - Training/evaluation plots for Problems 3 and 4
"""

import json
import os
from typing import Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import build_dataloaders
from minicamels import MiniCamels
from train import load_checkpoint, inverse_transform_target
from utils import ensure_dir, nse, rmse, mae, kge


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_history(history_path: str) -> Dict:
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_clean_style():
    """
    Compact gray-background style closer to the example figures.
    """
    plt.rcParams.update({
        "figure.figsize": (6.2, 3.1),
        "figure.facecolor": "white",
        "axes.facecolor": "#e6e6e6",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#b8b8b8",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        "lines.linewidth": 0.9,
        "savefig.dpi": 300,
    })


# ============================================================
# PROBLEM 1 — EXPLORATORY DATA ANALYSIS
# ============================================================

def _load_all_basins_raw() -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load basin index, attributes, and raw time series for all basins.
    """
    ds = MiniCamels()

    basins_df = ds.basins().copy()
    attrs_df = ds.attributes().copy()

    basin_timeseries = {}
    for basin_id in basins_df["basin_id"].astype(str):
        ts = ds.load_basin(basin_id)
        df = ts.to_dataframe().reset_index()
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        basin_timeseries[basin_id] = df

    return basins_df, attrs_df, basin_timeseries


def plot_streamflow_multiple_basins(
    basin_timeseries: Dict[str, pd.DataFrame],
    basin_ids: List[str] = None,
    output_dir: str = "outputs/exploration",
    max_points: int = 1500,
) -> str:
    """
    Plot streamflow time series for several basins on the same figure.
    """
    ensure_dir(output_dir)
    _apply_clean_style()

    available_ids = list(basin_timeseries.keys())

    if basin_ids is None:
        basin_ids = available_ids[:4]

    plt.figure(figsize=(6.3, 3.2))

    colors = ["black", "royalblue", "seagreen", "purple"]

    for i, basin_id in enumerate(basin_ids):
        if basin_id not in basin_timeseries:
            continue

        df = basin_timeseries[basin_id].copy()
        df = df[["time", "qobs"]].dropna()

        if len(df) > max_points:
            step = max(1, len(df) // max_points)
            df = df.iloc[::step].copy()

        plt.plot(
            df["time"],
            df["qobs"],
            color=colors[i % len(colors)],
            linewidth=0.8,
            alpha=0.95,
            label=f"{basin_id}",
        )

    plt.xlabel("Date")
    plt.ylabel("Q [mm/day]")
    plt.title("Observed Streamflow for Multiple Basins")
    plt.legend(loc="upper right", fontsize=6)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "explore_streamflow_multiple_basins.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return out_path


def plot_precip_and_streamflow_one_basin(
    basin_timeseries: Dict[str, pd.DataFrame],
    basin_id: str = None,
    output_dir: str = "outputs/exploration",
    max_points: int = 2200,
) -> str:
    """
    Plot precipitation and streamflow for one basin over the same period.
    Styled to resemble the example hydrograph figure.
    """
    ensure_dir(output_dir)
    _apply_clean_style()

    if basin_id is None:
        basin_id = list(basin_timeseries.keys())[0]

    df = basin_timeseries[basin_id].copy()
    df = df[["time", "prcp", "qobs"]].dropna().sort_values("time")

    if len(df) > max_points:
        df = df.iloc[:max_points].copy()

    fig, ax1 = plt.subplots(figsize=(6.5, 3.0))
    ax2 = ax1.twinx()

    q_line, = ax1.plot(
        df["time"],
        df["qobs"],
        color="black",
        linewidth=0.8,
        label="Q",
        zorder=3,
    )

    p_pts = ax2.scatter(
        df["time"],
        df["prcp"],
        color="red",
        s=5,
        marker="s",
        alpha=0.85,
        linewidths=0.0,
        label="P",
        zorder=2,
    )

    ax1.set_xlabel("Time [Test Period Days]")
    ax1.set_ylabel("Q [mm/day]")
    ax2.set_ylabel("P [mm/day]")

    ax2.invert_yaxis()

    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax1.legend(
        handles=[q_line, p_pts],
        labels=["Q", "P"],
        loc="lower left",
        fontsize=6,
        frameon=True,
        borderpad=0.25,
        handlelength=1.2,
    )

    plt.title(f"Basin {basin_id}: Streamflow and Precipitation", pad=4)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"explore_precip_streamflow_basin_{basin_id}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return out_path


def plot_qobs_histogram(
    basin_timeseries: Dict[str, pd.DataFrame],
    output_dir: str = "outputs/exploration",
    sample_size_per_basin: int = 2000,
) -> str:
    """
    Histogram of qobs values aggregated across basins.
    """
    ensure_dir(output_dir)
    _apply_clean_style()

    all_qobs = []

    for basin_id, df in basin_timeseries.items():
        q = df["qobs"].dropna()
        q = q[q >= 0]

        if len(q) > sample_size_per_basin:
            q = q.sample(sample_size_per_basin, random_state=42)

        all_qobs.append(q.values)

    qobs = np.concatenate(all_qobs)

    plt.figure(figsize=(5.1, 3.6))
    plt.hist(
        qobs,
        bins=50,
        color="#6baed6",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
    )
    plt.xlabel("Observed Streamflow (qobs)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Observed Streamflow")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "explore_qobs_histogram.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return out_path


def plot_static_attribute_scatter(
    attrs_df: pd.DataFrame,
    output_dir: str = "outputs/exploration",
    x_attr: str = "aridity",
    y_attr: str = "runoff_ratio",
) -> str:
    """
    Scatterplot using static basin attributes.
    """
    ensure_dir(output_dir)
    _apply_clean_style()

    df = attrs_df.copy().reset_index(drop=True)
    df = df[[x_attr, y_attr]].dropna()

    plt.figure(figsize=(5.0, 3.8))
    plt.scatter(
        df[x_attr],
        df[y_attr],
        s=16,
        color="#6baed6",
        edgecolor="black",
        linewidth=0.25,
        alpha=0.85,
    )
    plt.xlabel(x_attr.replace("_", " ").title())
    plt.ylabel(y_attr.replace("_", " ").title())
    plt.title(f"{x_attr.replace('_', ' ').title()} vs {y_attr.replace('_', ' ').title()}")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"explore_static_scatter_{x_attr}_vs_{y_attr}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return out_path


def generate_exploratory_plots(
    output_dir: str = "outputs/exploration",
    basin_ids: List[str] = None,
    single_basin_id: str = None,
) -> Dict[str, str]:
    """
    Generate a set of exploratory plots for Problem 1.
    """
    ensure_dir(output_dir)

    basins_df, attrs_df, basin_timeseries = _load_all_basins_raw()

    available_ids = basins_df["basin_id"].astype(str).tolist()

    if basin_ids is None:
        basin_ids = available_ids[:4]

    if single_basin_id is None:
        single_basin_id = available_ids[0]

    saved = {}

    saved["streamflow_multiple_basins"] = plot_streamflow_multiple_basins(
        basin_timeseries=basin_timeseries,
        basin_ids=basin_ids,
        output_dir=output_dir,
    )

    saved["precip_streamflow_one_basin"] = plot_precip_and_streamflow_one_basin(
        basin_timeseries=basin_timeseries,
        basin_id=single_basin_id,
        output_dir=output_dir,
    )

    saved["qobs_histogram"] = plot_qobs_histogram(
        basin_timeseries=basin_timeseries,
        output_dir=output_dir,
    )

    saved["scatter_aridity_vs_runoff_ratio"] = plot_static_attribute_scatter(
        attrs_df=attrs_df,
        output_dir=output_dir,
        x_attr="aridity",
        y_attr="runoff_ratio",
    )

    saved["scatter_area_km2_vs_q_mean"] = plot_static_attribute_scatter(
        attrs_df=attrs_df,
        output_dir=output_dir,
        x_attr="area_km2",
        y_attr="q_mean",
    )

    saved["scatter_elev_mean_vs_frac_snow"] = plot_static_attribute_scatter(
        attrs_df=attrs_df,
        output_dir=output_dir,
        x_attr="elev_mean",
        y_attr="frac_snow",
    )

    saved["scatter_baseflow_index_vs_hfd_mean"] = plot_static_attribute_scatter(
        attrs_df=attrs_df,
        output_dir=output_dir,
        x_attr="baseflow_index",
        y_attr="hfd_mean",
    )

    return saved


# ============================================================
# PROBLEM 3/4 — TRAINING + EVALUATION PLOTS
# ============================================================

def plot_training_history(
    history: Dict,
    output_dir: str = "outputs/figures",
) -> List[str]:
    ensure_dir(output_dir)
    _apply_clean_style()

    saved_paths = []
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(5.3, 3.6))
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="black", linewidth=0.9)
    plt.plot(epochs, history["val_loss"], label="Validation Loss", color="royalblue", linewidth=0.9)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend(frameon=True, fontsize=7)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "training_validation_loss.png")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()
    saved_paths.append(loss_path)

    plt.figure(figsize=(5.3, 3.6))
    plt.plot(epochs, history["train_nse"], label="Train NSE", color="black", linewidth=0.9)
    plt.plot(epochs, history["val_nse"], label="Validation NSE", color="royalblue", linewidth=0.9)
    plt.xlabel("Epoch")
    plt.ylabel("NSE")
    plt.title("Training and Validation NSE")
    plt.legend(frameon=True, fontsize=7)
    plt.tight_layout()
    nse_path = os.path.join(output_dir, "training_validation_nse.png")
    plt.savefig(nse_path, bbox_inches="tight")
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
    """
    Parity plot styled close to the example:
      - compact portrait figure
      - pale blue points
      - red dashed 1:1 line
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

    plt.figure(figsize=(3.3, 4.0))
    plt.scatter(
        obs_plot,
        pred_plot,
        s=12,
        color="#6baed6",
        edgecolor="none",
        alpha=0.8,
    )
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=0.9,
        color="red",
        label="Prediction",
    )
    plt.xlabel("Observed Streamflow")
    plt.ylabel("Predicted Streamflow")
    plt.title("Observed vs Predicted Streamflow")
    plt.legend(loc="upper left", fontsize=6)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "parity_plot.png")
    plt.savefig(out_path, bbox_inches="tight")
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
            "kge": float(kge(obs, pred)),
        })

    df = pd.DataFrame(rows).sort_values("nse", ascending=False).reset_index(drop=True)
    return df


def get_basin_metadata() -> pd.DataFrame:
    """
    Load basin coordinates and static attributes from MiniCamels.
    Returns a dataframe where basin_id is a normal column.
    """
    ds = MiniCamels()
    basins_df = ds.basins().copy()
    attrs_df = ds.attributes().copy()

    meta_df = attrs_df.copy().reset_index(drop=True)
    meta_df["basin_id"] = basins_df["basin_id"].astype(str).values

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

    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.plot(times, obs, label="Observed", linewidth=0.9, color="black")
    ax.plot(times, pred, label="Predicted", linewidth=0.9, alpha=0.9, color="royalblue")

    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow")
    ax.set_title(
        f"{title_prefix} Basin: {basin_id}\n"
        f"NSE = {basin_nse:.3f}, RMSE = {basin_rmse:.3f}, MAE = {basin_mae:.3f}"
    )
    ax.legend(frameon=True, fontsize=6)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{filename_prefix}basin_{basin_id}.png")
    plt.savefig(out_path, bbox_inches="tight")
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
    shapefile_dir: str = "shp",
) -> str:
    """
    Basin-level NSE map using user-provided shapefiles for continent,
    drainage network, and lakes.
    """
    ensure_dir(output_dir)
    _apply_clean_style()

    import geopandas as gpd

    df = metrics_df.merge(meta_df, on="basin_id", how="left").copy()

    bins = [-np.inf, 0.0, 0.3, 0.5, 0.65, 0.8, np.inf]
    labels = [
        "NSE < 0",
        "0.00 – 0.30",
        "0.30 – 0.50",
        "0.50 – 0.65",
        "0.65 – 0.80",
        "NSE > 0.80",
    ]
    colors = [
        "#d73027",
        "#fc8d59",
        "#fee08b",
        "#91cf60",
        "#1a9850",
        "#2c7bb6",
    ]

    df["nse_class"] = pd.cut(df["nse"], bins=bins, labels=labels, include_lowest=True)

    continent_path = os.path.join(shapefile_dir, "Americas.shp")
    drainage_path = os.path.join(shapefile_dir, "Estados_Unidos_Hidrografia.shp")
    lakes_path = os.path.join(shapefile_dir, "Estados_Unidos_Lagos.shp")

    df_continent = gpd.read_file(continent_path)
    df_drainage_network = gpd.read_file(drainage_path)
    df_lakes = gpd.read_file(lakes_path)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    df_continent.boundary.plot(color="black", ax=ax, lw=0.6, alpha=0.6)
    df_drainage_network.plot(color="steelblue", ax=ax, lw=0.4, alpha=0.20)
    df_lakes.plot(color="steelblue", ax=ax, lw=0.4, alpha=0.25)

    for cls, color in zip(labels, colors):
        subset = df[df["nse_class"] == cls]
        if len(subset) == 0:
            continue

        ax.scatter(
            subset["lon"],
            subset["lat"],
            marker="o",
            s=20,
            color=color,
            edgecolor="black",
            linewidth=0.25,
            label=cls,
            zorder=3,
        )

    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.set_xlim([-125, -66])
    ax.set_ylim([20, 50])
    ax.set_title("Basin-Level Test NSE Across MiniCAMELS", fontsize=10)

    legend = ax.legend(
        title="NSE class",
        loc="lower left",
        frameon=True,
        fontsize=6,
        title_fontsize=7,
    )
    legend.get_frame().set_alpha(0.95)

    plt.tight_layout()

    out_path = os.path.join(output_dir, "basin_nse_map_classified.png")
    plt.savefig(out_path, bbox_inches="tight")
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

    plt.figure(figsize=(6.0, 3.2))
    plt.plot(np.arange(1, len(df) + 1), df["nse"], marker="o", markersize=2.5, linewidth=0.8, color="black")
    plt.axhline(0.0, linestyle="--", linewidth=0.8, color="red")
    plt.xlabel("Basin Rank")
    plt.ylabel("Test NSE")
    plt.title("Ranked Basin-Level Test NSE")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "ranked_basin_nse.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return out_path


def plot_nse_vs_aridity(
    metrics_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    output_dir: str = "outputs/figures",
) -> str:
    """
    Scatter plot of basin NSE versus aridity.
    """
    ensure_dir(output_dir)
    _apply_clean_style()

    df = metrics_df.merge(meta_df[["basin_id", "aridity"]], on="basin_id", how="left")
    df = df.dropna(subset=["aridity", "nse"]).copy()

    plt.figure(figsize=(5.0, 3.8))
    plt.scatter(
        df["aridity"],
        df["nse"],
        s=16,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.25,
        color="#6baed6",
    )

    plt.axhline(0.0, linestyle="--", linewidth=0.8, color="red")

    best_idx = df["nse"].idxmax()
    worst_idx = df["nse"].idxmin()

    plt.annotate(
        df.loc[best_idx, "basin_id"],
        (df.loc[best_idx, "aridity"], df.loc[best_idx, "nse"]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=6,
    )

    plt.annotate(
        df.loc[worst_idx, "basin_id"],
        (df.loc[worst_idx, "aridity"], df.loc[worst_idx, "nse"]),
        xytext=(4, -8),
        textcoords="offset points",
        fontsize=6,
    )

    plt.xlabel("Aridity")
    plt.ylabel("Test NSE")
    plt.title("Basin-Level Test NSE vs Aridity")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "nse_vs_aridity.png")
    plt.savefig(out_path, bbox_inches="tight")
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

    metrics_csv = os.path.join(output_dir, "per_basin_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    saved["per_basin_metrics_csv"] = metrics_csv

    return saved
