"""
Plotting and visualization functions for streamflow prediction.
Includes:
  - Exploratory plots for Problem 1
  - Training/evaluation plots for Problems 3 and 4
"""

import json
import os
import random
from typing import Dict, List

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


def _reset_plot_style():
    """
    Keep styling very close to the user's reference snippet:
    simple matplotlib defaults + moderate grid + compact fonts.
    """
    plt.rcdefaults()
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "savefig.dpi": 300,
    })


# ============================================================
# PROBLEM 1 — EXPLORATORY DATA ANALYSIS
# ============================================================

def _load_all_basins_raw() -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
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


def plot_precip_and_streamflow_one_basin(
    basin_timeseries: Dict[str, pd.DataFrame],
    basin_id: str,
    output_dir: str = "outputs/exploration",
    max_points: int = 2200,
) -> str:
    """
    One-basin hydrograph styled like the user's reference snippet:
      - black line for Q
      - red points for Qobs
      - blue precipitation bars on inverted right axis
    """
    ensure_dir(output_dir)
    _reset_plot_style()

    df = basin_timeseries[basin_id][["time", "prcp", "qobs"]].dropna().sort_values("time").copy()

    if len(df) > max_points:
        df = df.iloc[:max_points].copy()

    x = np.arange(len(df))
    q_obs = df["qobs"].values
    p = df["prcp"].values

    fig, axQ = plt.subplots(figsize=(13, 4.5))

    # ---- Streamflow (left axis)
    q_line, = axQ.plot(
        x,
        q_obs,
        color="black",
        linewidth=1.2,
        label="Q",
        zorder=3,
    )

    q_pts = axQ.scatter(
        x,
        q_obs,
        color="red",
        s=10,
        label="Qobs",
        zorder=4,
    )

    axQ.set_ylabel(r"$Q\ [mm/day]$")
    axQ.set_xlabel("Time (Test Period Days)")
    axQ.grid(True, alpha=0.35)

    # ---- Precipitation (right axis, inverted)
    axP = axQ.twinx()
    p_bar = axP.bar(
        x,
        p,
        color="blue",
        width=1.0,
        alpha=0.85,
        label="P",
        zorder=1,
    )

    axP.set_ylabel(r"$P\ [mm/day]$")
    axP.invert_yaxis()

    pmax = np.nanmax(p)
    axP.set_ylim(pmax * 1.05 if pmax > 0 else 1, 0)

    # ---- Combined legend
    hQ, lQ = axQ.get_legend_handles_labels()
    hP, lP = axP.get_legend_handles_labels()
    axQ.legend(hP + hQ, lP + lQ, loc="lower left", framealpha=0.95)

    plt.title(f"Basin {basin_id}: Streamflow and Precipitation")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"explore_precip_streamflow_basin_{basin_id}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return out_path


def plot_streamflow_multiple_basins(
    basin_timeseries: Dict[str, pd.DataFrame],
    basin_ids: List[str] = None,
    n_basins: int = 4,
    random_seed: int = 42,
    output_dir: str = "outputs/exploration/random_basin_hydrographs",
    max_points: int = 2200,
) -> List[str]:
    """
    Instead of plotting many basins in one panel, this function randomly selects
    N basins and repeats the one-basin hydrograph plot for each of them.

    Returns
    -------
    saved_paths : list of str
        Paths to the generated figures.
    """
    ensure_dir(output_dir)

    available_ids = list(basin_timeseries.keys())

    if basin_ids is None:
        rng = random.Random(random_seed)
        n_basins = min(n_basins, len(available_ids))
        basin_ids = rng.sample(available_ids, n_basins)
    else:
        basin_ids = [str(b) for b in basin_ids]

    saved_paths = []

    for basin_id in basin_ids:
        if basin_id not in basin_timeseries:
            continue

        out_path = plot_precip_and_streamflow_one_basin(
            basin_timeseries=basin_timeseries,
            basin_id=basin_id,
            output_dir=output_dir,
            max_points=max_points,
        )
        saved_paths.append(out_path)

    return saved_paths


def plot_qobs_histogram(
    basin_timeseries: Dict[str, pd.DataFrame],
    output_dir: str = "outputs/exploration",
    sample_size_per_basin: int = 2000,
) -> str:
    ensure_dir(output_dir)
    _reset_plot_style()

    all_qobs = []

    for _, df in basin_timeseries.items():
        q = df["qobs"].dropna()
        q = q[q >= 0]

        if len(q) > sample_size_per_basin:
            q = q.sample(sample_size_per_basin, random_state=42)

        all_qobs.append(q.values)

    qobs = np.concatenate(all_qobs)

    plt.figure(figsize=(6, 4))
    plt.hist(qobs, bins=50, color="#6BAED6", edgecolor="black", alpha=0.85)
    plt.xlabel("Observed Streamflow (qobs)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Observed Streamflow")
    plt.grid(True, alpha=0.35)
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
    ensure_dir(output_dir)
    _reset_plot_style()

    df = attrs_df[[x_attr, y_attr]].dropna().copy()

    plt.figure(figsize=(6, 5))
    plt.scatter(
        df[x_attr],
        df[y_attr],
        alpha=0.6,
        s=22,
        color="#6BAED6",
        edgecolors="black",
        linewidths=0.3,
    )
    plt.xlabel(x_attr.replace("_", " ").title())
    plt.ylabel(y_attr.replace("_", " ").title())
    plt.title(f"{x_attr.replace('_', ' ').title()} vs {y_attr.replace('_', ' ').title()}")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"explore_static_scatter_{x_attr}_vs_{y_attr}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return out_path


def generate_exploratory_plots(
    output_dir: str = "outputs/exploration",
    basin_ids: List[str] = None,
    single_basin_id: str = None,
    n_random_basins: int = 4,
    random_seed: int = 42,
) -> Dict[str, object]:
    """
    Generate the full set of exploratory plots for Problem 1.
    """
    ensure_dir(output_dir)

    basins_df, attrs_df, basin_timeseries = _load_all_basins_raw()
    available_ids = basins_df["basin_id"].astype(str).tolist()

    if single_basin_id is None:
        single_basin_id = available_ids[0]

    saved = {}

    # One detailed hydrograph
    saved["precip_streamflow_one_basin"] = plot_precip_and_streamflow_one_basin(
        basin_timeseries=basin_timeseries,
        basin_id=single_basin_id,
        output_dir=output_dir,
    )

    # Random repeated hydrographs
    saved["random_basin_hydrographs"] = plot_streamflow_multiple_basins(
        basin_timeseries=basin_timeseries,
        basin_ids=basin_ids,
        n_basins=n_random_basins,
        random_seed=random_seed,
        output_dir=os.path.join(output_dir, "random_basin_hydrographs"),
    )

    # Histogram
    saved["qobs_histogram"] = plot_qobs_histogram(
        basin_timeseries=basin_timeseries,
        output_dir=output_dir,
    )

    # Attribute plots
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
    _reset_plot_style()

    saved_paths = []
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="black", linewidth=1.0)
    plt.plot(epochs, history["val_loss"], label="Validation Loss", color="royalblue", linewidth=1.0)
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

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_nse"], label="Train NSE", color="black", linewidth=1.0)
    plt.plot(epochs, history["val_nse"], label="Validation NSE", color="royalblue", linewidth=1.0)
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


def collect_test_predictions(
    checkpoint_path: str,
    batch_size: int = 64,
) -> Dict:
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
    _reset_plot_style()

    if len(obs) > max_points:
        idx = np.random.choice(len(obs), size=max_points, replace=False)
        obs_plot = obs[idx]
        pred_plot = pred[idx]
    else:
        obs_plot = obs
        pred_plot = pred

    plt.figure(figsize=(6, 6))
    plt.scatter(obs_plot, pred_plot, alpha=0.5, s=14, color="#1f77b4")

    plt.plot(
        [obs_plot.min(), obs_plot.max()],
        [obs_plot.min(), obs_plot.max()],
        "r--",
        label="Prediction"
    )

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


def compute_per_basin_metrics(results: Dict) -> pd.DataFrame:
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


def get_basin_metadata() -> pd.DataFrame:
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
    return {"best": best, "worst": worst}


def plot_test_timeseries(
    results: Dict,
    basin_id: str,
    output_dir: str = "outputs/figures",
    max_points: int = 365,
    title_prefix: str = "",
    filename_prefix: str = "",
) -> str:
    ensure_dir(output_dir)
    _reset_plot_style()

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
    plt.plot(times, obs, label="Observed", linewidth=2, color="#6BAED6")
    plt.plot(times, pred, label="Predicted", linewidth=2, alpha=0.8, color="#F4A6B5")
    plt.xlabel("Date")
    plt.ylabel("Streamflow")
    plt.title(
        f"{title_prefix} Basin {basin_id} | NSE={basin_nse:.3f}, RMSE={basin_rmse:.3f}, MAE={basin_mae:.3f}"
    )
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
    ensure_dir(output_dir)
    _reset_plot_style()

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
        ax.scatter(
            subset["lon"],
            subset["lat"],
            marker="o",
            s=20,
            color=color,
            edgecolor="black",
            linewidth=0.2,
            label=cls,
            zorder=3,
        )

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


def plot_ranked_nse(
    metrics_df: pd.DataFrame,
    output_dir: str = "outputs/figures",
) -> str:
    ensure_dir(output_dir)
    _reset_plot_style()

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
    ensure_dir(output_dir)
    _reset_plot_style()

    df = metrics_df.merge(meta_df[["basin_id", "aridity"]], on="basin_id", how="left")
    df = df.dropna(subset=["aridity", "nse"]).copy()

    plt.figure(figsize=(6, 5))
    plt.scatter(
        df["aridity"],
        df["nse"],
        s=18,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.2,
        color="#6BAED6",
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
    plt.grid(True, alpha=0.35)
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
