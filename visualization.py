"""
Organized plotting and visualization utilities.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import build_dataloaders
from minicamels import MiniCamels
from train import load_checkpoint, inverse_transform_target
from utils import ensure_dir, nse, rmse, mae, kge


# =====================================================================
# SECTION 1 — GLOBAL STYLE / METADATA (AESTHETIC PALETTE)
# =====================================================================

# --- Custom aesthetic palette (clean, modern, not the example colors)
PRIMARY_BLUE   = "#3A86FF"   # vivid blue
SOFT_CYAN      = "#4CC9F0"   # light cyan
DEEP_PURPLE    = "#5E548E"   # muted purple
SOFT_PINK      = "#F28482"   # soft red/pink
EARTH_GREEN    = "#84A98C"   # desaturated green
WARM_SAND      = "#E9C46A"   # warm yellow
DARK_CHARCOAL  = "#2B2D42"   # neutral dark
LIGHT_GRAY     = "#ADB5BD"   # subtle grid tone

TRAIN_COLOR = PRIMARY_BLUE
VAL_COLOR   = SOFT_CYAN
TEST_COLOR  = DEEP_PURPLE

PRECIP_COLOR = SOFT_CYAN
QOBS_COLOR   = DARK_CHARCOAL
QPRED_COLOR  = SOFT_PINK
SCATTER_COLOR = PRIMARY_BLUE

FORCING_META: Dict[str, Tuple[str, str]] = {
    "prcp": ("Precipitation (mm/day)", SOFT_CYAN),
    "tmax": ("T max (°C)", SOFT_PINK),
    "tmin": ("T min (°C)", DEEP_PURPLE),
    "srad": ("Solar radiation (W/m²)", WARM_SAND),
    "vp": ("Vapor pressure (Pa)", EARTH_GREEN),
}

ATTRIBUTE_HIST_COLORS = [
    PRIMARY_BLUE,
    SOFT_CYAN,
    DEEP_PURPLE,
    SOFT_PINK,
    EARTH_GREEN,
    WARM_SAND,
    "#90DBF4",
    "#B8C0FF",
    "#FFD6A5",
    "#CDB4DB",
]

DEFAULT_ATTRIBUTE_VARS = [
    "aridity",
    "runoff_ratio",
    "area_km2",
    "q_mean",
    "elev_mean",
    "frac_snow",
    "baseflow_index",
    "hfd_mean",
]


# =====================================================================
# SECTION 2 — GENERAL HELPERS
# =====================================================================


def reset_plot_style() -> None:
    """Apply a clean, publication-style aesthetic."""
    plt.rcdefaults()
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.color": LIGHT_GRAY,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
    })



def load_history(history_path: str) -> Dict:
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)



def get_dataset() -> MiniCamels:
    return MiniCamels()



def get_basin_metadata() -> pd.DataFrame:
    ds = get_dataset()
    basins_df = ds.basins().copy()
    attrs_df = ds.attributes().copy().reset_index(drop=True)

    meta_df = attrs_df.copy()
    meta_df["basin_id"] = basins_df["basin_id"].astype(str).values

    if "basin_name" in basins_df.columns:
        meta_df["basin_name"] = basins_df["basin_name"].values

    return meta_df



def load_all_basins_raw() -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load basin metadata, static attributes, and full time series for all basins."""
    ds = get_dataset()

    basins_df = ds.basins().copy()
    attrs_df = ds.attributes().copy()

    basin_timeseries: Dict[str, pd.DataFrame] = {}
    for basin_id in basins_df["basin_id"].astype(str):
        ts = ds.load_basin(basin_id)
        df = ts.to_dataframe().reset_index()
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        basin_timeseries[basin_id] = df

    return basins_df, attrs_df, basin_timeseries


# =====================================================================
# SECTION 3 — EXPLORATORY PLOTS
# =====================================================================


def plot_precip_and_streamflow_one_basin(
    basin_timeseries: Dict[str, pd.DataFrame],
    basin_id: str,
    output_dir: str = "outputs/exploration",
    max_points: int = 2200,
) -> str:
    """Plot streamflow and precipitation for one basin."""
    ensure_dir(output_dir)
    reset_plot_style()

    df = basin_timeseries[basin_id][["time", "prcp", "qobs"]].dropna().sort_values("time").copy()
    if len(df) > max_points:
        df = df.iloc[:max_points].copy()

    x = np.arange(len(df))
    q_obs = df["qobs"].values
    p = df["prcp"].values

    fig, ax_q = plt.subplots(figsize=(13, 4.5))

    ax_q.plot(x, q_obs, color=QOBS_COLOR, linewidth=1.2, label="Qobs", zorder=3)
    ax_q.set_ylabel(r"$Q\ [mm/day]$")
    ax_q.set_xlabel("Time (days)")
    ax_q.grid(True, alpha=0.35)

    ax_p = ax_q.twinx()
    ax_p.bar(x, p, color=PRECIP_COLOR, width=1.0, alpha=0.9, label="P", zorder=1)
    ax_p.set_ylabel(r"$P\ [mm/day]$")
    ax_p.invert_yaxis()

    pmax = np.nanmax(p) if len(p) > 0 else 1.0
    ax_p.set_ylim(pmax * 1.05 if pmax > 0 else 1, 0)

    h_q, l_q = ax_q.get_legend_handles_labels()
    h_p, l_p = ax_p.get_legend_handles_labels()
    ax_q.legend(h_p + h_q, l_p + l_q, loc="lower left", framealpha=0.95)

    plt.title(f"Basin {basin_id}: Streamflow and Precipitation")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"explore_precip_streamflow_basin_{basin_id}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path



def plot_streamflow_multiple_basins(
    basin_timeseries: Dict[str, pd.DataFrame],
    basin_ids: Optional[List[str]] = None,
    n_basins: int = 4,
    random_seed: int = 42,
    output_dir: str = "outputs/exploration/random_basin_hydrographs",
    max_points: int = 2200,
) -> List[str]:
    """Generate repeated hydrograph plots for several basins."""
    ensure_dir(output_dir)

    available_ids = list(basin_timeseries.keys())
    if basin_ids is None:
        rng = random.Random(random_seed)
        basin_ids = rng.sample(available_ids, min(n_basins, len(available_ids)))
    else:
        basin_ids = [str(b) for b in basin_ids]

    saved_paths = []
    for basin_id in basin_ids:
        if basin_id not in basin_timeseries:
            continue
        saved_paths.append(
            plot_precip_and_streamflow_one_basin(
                basin_timeseries=basin_timeseries,
                basin_id=basin_id,
                output_dir=output_dir,
                max_points=max_points,
            )
        )

    return saved_paths



def plot_qobs_histogram(
    basin_timeseries: Dict[str, pd.DataFrame],
    output_dir: str = "outputs/exploration",
    sample_size_per_basin: int = 2000,
) -> str:
    """Histogram of observed streamflow across all basins."""
    ensure_dir(output_dir)
    reset_plot_style()

    all_qobs = []
    for _, df in basin_timeseries.items():
        q = df["qobs"].dropna()
        q = q[q >= 0]
        if len(q) > sample_size_per_basin:
            q = q.sample(sample_size_per_basin, random_state=42)
        all_qobs.append(q.values)

    qobs = np.concatenate(all_qobs)

    plt.figure(figsize=(6, 4))
    plt.hist(qobs, bins=50, color=SCATTER_COLOR, edgecolor="black", alpha=0.85)
    plt.xlabel("Observed Streamflow (qobs)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Observed Streamflow")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "explore_qobs_histogram.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path



def plot_attribute_histograms(
    attrs_df: pd.DataFrame,
    output_dir: str = "outputs/exploration",
    attribute_vars: Optional[List[str]] = None,
    bins: int = 30,
) -> str:
    """Single multi-panel figure with histograms for static attributes."""
    ensure_dir(output_dir)
    reset_plot_style()

    if attribute_vars is None:
        attribute_vars = DEFAULT_ATTRIBUTE_VARS

    cols = [c for c in attribute_vars if c in attrs_df.columns]
    n = len(cols)

    if n == 0:
        raise ValueError("No valid attribute columns were found in attrs_df.")

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for i, (ax, col) in enumerate(zip(axes_flat, cols)):
        values = attrs_df[col].dropna()
        color = ATTRIBUTE_HIST_COLORS[i % len(ATTRIBUTE_HIST_COLORS)]
        ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.4)
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.set_title(col.replace("_", " ").title(), fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Static Attribute Distributions", fontsize=12, fontweight="bold")

    out_path = os.path.join(output_dir, "explore_attribute_histograms.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path



def plot_forcing_histograms(
    ds: MiniCamels,
    basin_ids: List[str],
    start_date: str,
    end_date: str,
    forcing_vars: List[str],
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Histograms of forcing variables across selected basins and time window."""
    reset_plot_style()

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    data = {v: [] for v in forcing_vars}
    for basin_id in basin_ids:
        forcings = ds.get_forcings(basin_id, start, end).to_array()
        all_vars = forcings.coords["variable"].values.tolist()
        for var in forcing_vars:
            if var in all_vars:
                data[var].append(forcings.sel(variable=var).values.ravel())

    n = len(forcing_vars)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, var in zip(axes_flat, forcing_vars):
        values = data[var]
        if not values:
            ax.set_visible(False)
            continue
        combined = pd.concat([pd.Series(v) for v in values], ignore_index=True).dropna()
        label, color = FORCING_META.get(var, (var, "#7f7f7f"))
        ax.hist(combined, bins=50, color=color, edgecolor="white", linewidth=0.3)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(var, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Forcing Distributions ({start_date} – {end_date})", fontsize=12, fontweight="bold")

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path

    plt.show()
    return None



def plot_data_availability(
    ds: MiniCamels,
    periods: Dict[str, Tuple[str, str]],
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Heatmap of valid streamflow fraction by basin and year for train/val/eval splits."""
    reset_plot_style()

    basin_ids = ds.basins()["basin_id"].tolist()
    all_starts = [pd.to_datetime(v[0]) for v in periods.values()]
    all_ends = [pd.to_datetime(v[1]) for v in periods.values()]
    global_start = min(all_starts)
    global_end = max(all_ends)

    series = {}
    for basin_id in basin_ids:
        q = ds.get_streamflow(basin_id, global_start, global_end)
        series[basin_id] = pd.Series(q.values, index=pd.to_datetime(q.coords["time"].values))

    n_splits = len(periods)
    panel_height = max(4, len(basin_ids) * 0.18)

    fig, axes = plt.subplots(
        1,
        n_splits,
        figsize=(sum(max(3, (pd.to_datetime(end).year - pd.to_datetime(start).year + 1) * 0.45)
                    for _, (start, end) in periods.items()), panel_height),
        constrained_layout=True,
    )

    if n_splits == 1:
        axes = [axes]

    for ax, (split, (start_str, end_str)) in zip(axes, periods.items()):
        t0 = pd.to_datetime(start_str)
        t1 = pd.to_datetime(end_str)
        years = list(range(t0.year, t1.year + 1))

        matrix = np.full((len(basin_ids), len(years)), np.nan)
        for i, basin_id in enumerate(basin_ids):
            ser = series[basin_id]
            for j, yr in enumerate(years):
                yr_data = ser[ser.index.year == yr]
                if len(yr_data) == 0:
                    continue
                matrix[i, j] = (~np.isnan(yr_data.values)).mean()

        im = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="jet_r",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
        )

        ax.set_xticks(range(len(years)))
        ax.set_xticklabels([str(y) if y % 5 == 0 or len(years) <= 10 else "" for y in years],
                           fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(basin_ids)))
        ax.set_yticklabels(basin_ids, fontsize=6)
        ax.set_xlabel("Year")
        ax.set_ylabel("Basin ID")
        ax.set_title(split, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Fraction valid", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Streamflow Data Availability", fontsize=12, fontweight="bold")

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path

    plt.show()
    return None



def generate_exploratory_plots(
    output_dir: str = "outputs/exploration",
    basin_ids: Optional[List[str]] = None,
    single_basin_id: Optional[str] = None,
    n_random_basins: int = 4,
    random_seed: int = 42,
    attribute_vars: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Generate the full set of exploratory figures."""
    ensure_dir(output_dir)

    basins_df, attrs_df, basin_timeseries = load_all_basins_raw()
    available_ids = basins_df["basin_id"].astype(str).tolist()

    if single_basin_id is None:
        single_basin_id = available_ids[0]

    saved: Dict[str, object] = {}

    saved["precip_streamflow_one_basin"] = plot_precip_and_streamflow_one_basin(
        basin_timeseries=basin_timeseries,
        basin_id=single_basin_id,
        output_dir=output_dir,
    )

    saved["random_basin_hydrographs"] = plot_streamflow_multiple_basins(
        basin_timeseries=basin_timeseries,
        basin_ids=basin_ids,
        n_basins=n_random_basins,
        random_seed=random_seed,
        output_dir=os.path.join(output_dir, "random_basin_hydrographs"),
    )

    saved["qobs_histogram"] = plot_qobs_histogram(
        basin_timeseries=basin_timeseries,
        output_dir=output_dir,
    )

    saved["attribute_histograms"] = plot_attribute_histograms(
        attrs_df=attrs_df,
        output_dir=output_dir,
        attribute_vars=attribute_vars,
    )

    return saved


# =====================================================================
# SECTION 4 — TRAINING HISTORY PLOTS
# =====================================================================


def plot_training_history(
    history: Dict,
    output_dir: str = "outputs/figures",
) -> List[str]:
    """Plot training history metrics."""
    ensure_dir(output_dir)
    reset_plot_style()

    saved_paths = []
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss", color=TRAIN_COLOR, linewidth=1.3)
    plt.plot(epochs, history["val_loss"], label="Validation Loss", color=VAL_COLOR, linewidth=1.3)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "training_validation_loss.png")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()
    saved_paths.append(loss_path)

    if "train_nse" in history and "val_nse" in history:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["train_nse"], label="Train NSE", color=TRAIN_COLOR, linewidth=1.3)
        plt.plot(epochs, history["val_nse"], label="Validation NSE", color=VAL_COLOR, linewidth=1.3)
        plt.xlabel("Epoch")
        plt.ylabel("NSE")
        plt.title("Training and Validation NSE")
        plt.legend()
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        nse_path = os.path.join(output_dir, "training_validation_nse.png")
        plt.savefig(nse_path, bbox_inches="tight")
        plt.close()
        saved_paths.append(nse_path)

    return saved_paths


# =====================================================================
# SECTION 5 — MODEL PREDICTIONS / METRICS
# =====================================================================


def collect_test_predictions(
    checkpoint_path: str,
    batch_size: int = 64,
) -> Dict:
    """Load model checkpoint and collect test predictions."""
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



def compute_per_basin_metrics(results: Dict) -> pd.DataFrame:
    """Compute evaluation metrics for each basin."""
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

    return pd.DataFrame(rows).sort_values("nse", ascending=False).reset_index(drop=True)



def get_best_and_worst_basin(metrics_df: pd.DataFrame) -> Dict:
    """Return the best and worst basins ranked by NSE."""
    valid_df = metrics_df.dropna(subset=["nse"]).copy()
    best = valid_df.iloc[0].to_dict()
    worst = valid_df.iloc[-1].to_dict()
    return {"best": best, "worst": worst}


# =====================================================================
# SECTION 6 — EVALUATION PLOTS
# =====================================================================


def plot_parity(
    obs: np.ndarray,
    pred: np.ndarray,
    output_dir: str = "outputs/figures",
    max_points: int = 4000,
) -> str:
    """Observed vs predicted scatter plot."""
    ensure_dir(output_dir)
    reset_plot_style()

    if len(obs) > max_points:
        idx = np.random.choice(len(obs), size=max_points, replace=False)
        obs_plot = obs[idx]
        pred_plot = pred[idx]
    else:
        obs_plot = obs
        pred_plot = pred

    plt.figure(figsize=(6, 6))
    plt.scatter(obs_plot, pred_plot, alpha=0.5, s=14, color=SCATTER_COLOR)
    plt.plot([obs_plot.min(), obs_plot.max()], [obs_plot.min(), obs_plot.max()], "r--", label="1:1 line")
    plt.xlabel("Observed Streamflow")
    plt.ylabel("Predicted Streamflow")
    plt.title("Observed vs Predicted Streamflow")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "parity_plot.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path



def plot_test_timeseries(
    results: Dict,
    basin_id: str,
    output_dir: str = "outputs/figures",
    max_points: int = 365,
    title_prefix: str = "",
    filename_prefix: str = "",
) -> str:
    """Observed vs predicted streamflow time series for a selected basin."""
    ensure_dir(output_dir)
    reset_plot_style()

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

    plt.figure(figsize=(12, 5))
    plt.plot(times, obs, label="Observed", linewidth=2, color=TRAIN_COLOR)
    plt.plot(times, pred, label="Predicted", linewidth=2, alpha=0.85, color=QPRED_COLOR)
    plt.xlabel("Date")
    plt.ylabel("Streamflow")
    plt.title(f"{title_prefix} Basin {basin_id} | NSE={basin_nse:.3f}, RMSE={basin_rmse:.3f}, MAE={basin_mae:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.35)
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
    """Generate time series plots for the best and worst basins."""
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



def plot_ranked_nse(
    metrics_df: pd.DataFrame,
    output_dir: str = "outputs/figures",
) -> str:
    """Ranked NSE across basins."""
    ensure_dir(output_dir)
    reset_plot_style()

    df = metrics_df.sort_values("nse", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(df) + 1), df["nse"], marker="o", markersize=3, linewidth=0.8, color="black")
    plt.axhline(0.0, linestyle="--", linewidth=0.8, color="red")
    plt.xlabel("Basin Rank")
    plt.ylabel("Test NSE")
    plt.title("Ranked Basin-Level Test NSE")
    plt.grid(True, alpha=0.35)
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
    """Scatter plot of basin NSE versus aridity."""
    ensure_dir(output_dir)
    reset_plot_style()

    df = metrics_df.merge(meta_df[["basin_id", "aridity"]], on="basin_id", how="left")
    df = df.dropna(subset=["aridity", "nse"]).copy()

    plt.figure(figsize=(6, 5))
    plt.scatter(df["aridity"], df["nse"], s=18, alpha=0.8, edgecolor="black", linewidth=0.2, color=SCATTER_COLOR)
    plt.axhline(0.0, linestyle="--", linewidth=0.8, color="red")

    best_idx = df["nse"].idxmax()
    worst_idx = df["nse"].idxmin()

    plt.annotate(df.loc[best_idx, "basin_id"],
                 (df.loc[best_idx, "aridity"], df.loc[best_idx, "nse"]),
                 xytext=(4, 4), textcoords="offset points", fontsize=6)
    plt.annotate(df.loc[worst_idx, "basin_id"],
                 (df.loc[worst_idx, "aridity"], df.loc[worst_idx, "nse"]),
                 xytext=(4, -8), textcoords="offset points", fontsize=6)

    plt.xlabel("Aridity")
    plt.ylabel("Test NSE")
    plt.title("Basin-Level Test NSE vs Aridity")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "nse_vs_aridity.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path



def plot_nse_map(
    metrics_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    output_dir: str = "outputs/figures",
    shapefile_dir: str = "shp",
) -> str:
    """Map of basin-level NSE classes."""
    ensure_dir(output_dir)
    reset_plot_style()

    import geopandas as gpd

    df = metrics_df.merge(meta_df, on="basin_id", how="left").copy()

    bins = [-np.inf, 0.0, 0.3, 0.5, 0.65, 0.8, np.inf]
    labels = ["NSE < 0", "0.00 – 0.30", "0.30 – 0.50", "0.50 – 0.65", "0.65 – 0.80", "NSE > 0.80"]
    colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850", "#2c7bb6"]
    df["nse_class"] = pd.cut(df["nse"], bins=bins, labels=labels, include_lowest=True)

    continent_path = os.path.join(shapefile_dir, "Americas.shp")
    drainage_path = os.path.join(shapefile_dir, "Estados_Unidos_Hidrografia.shp")
    lakes_path = os.path.join(shapefile_dir, "Estados_Unidos_Lagos.shp")

    df_continent = gpd.read_file(continent_path)
    df_drainage_network = gpd.read_file(drainage_path)
    df_lakes = gpd.read_file(lakes_path)

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    df_continent.boundary.plot(color="black", ax=ax, lw=0.6, alpha=0.6)
    df_drainage_network.plot(color="steelblue", ax=ax, lw=0.4, alpha=0.20)
    df_lakes.plot(color="steelblue", ax=ax, lw=0.4, alpha=0.25)

    for cls, color in zip(labels, colors):
        subset = df[df["nse_class"] == cls]
        if len(subset) == 0:
            continue
        ax.scatter(subset["lon"], subset["lat"], marker="o", s=20, color=color,
                   edgecolor="black", linewidth=0.2, label=cls, zorder=3)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim([-125, -66])
    ax.set_ylim([20, 50])
    ax.set_title("Basin-Level Test NSE Across MiniCAMELS")
    ax.legend(title="NSE class", loc="lower left", frameon=True, fontsize=6, title_fontsize=7)

    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "basin_nse_map_classified.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


# =====================================================================
# SECTION 7 — MASTER DRIVER
# =====================================================================


def generate_all_plots(
    checkpoint_path: str,
    history_path: str = "outputs/metrics/training_history.json",
    output_dir: str = "outputs/figures",
    batch_size: int = 64,
) -> Dict:
    """Run the full evaluation plotting workflow."""
    ensure_dir(output_dir)
    saved: Dict[str, object] = {}

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
