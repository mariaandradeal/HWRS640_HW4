"""
Data loading and preprocessing using MiniCamels
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from minicamels import MiniCamels


DYNAMIC_VARS = ["prcp", "tmax", "tmin", "srad", "vp"]
TARGET_VAR = "qobs"
STATIC_VARS = [
    "lat",
    "lon",
    "elev_mean",
    "slope_mean",
    "area_km2",
    "mean_prcp",
    "mean_pet",
    "aridity",
    "frac_snow",
    "q_mean",
    "runoff_ratio",
    "hfd_mean",
    "baseflow_index",
    "soil_depth_pelletier",
    "frac_forest",
    "lai_max",
]


@dataclass
class NormalizationStats:
    dynamic_mean: np.ndarray
    dynamic_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray


class StreamflowDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        return {
            "x_seq": torch.tensor(s["x_seq"], dtype=torch.float32),
            "x_static": torch.tensor(s["x_static"], dtype=torch.float32),
            "y": torch.tensor([s["y"]], dtype=torch.float32),
            "basin_id": str(s["basin_id"]),
            "pred_time": str(s["pred_time"]),
        }


def summarize_dataset() -> None:
    ds = MiniCamels()

    basins_df = ds.basins()
    attrs_df = ds.attributes()

    print("Basins columns:", basins_df.columns.tolist())
    print("Attributes columns:", attrs_df.columns.tolist())

    example_basin_id = str(basins_df.iloc[0]["basin_id"])

    ts = ds.load_basin(example_basin_id)
    df = ts.to_dataframe().reset_index()

    print("Time series columns:", df.columns.tolist())

    start_date = df["time"].min()
    end_date = df["time"].max()

    print("\n===== MiniCAMELS Dataset Summary =====")
    print(f"Number of basins: {len(basins_df)}")
    print(f"Time span: {start_date} to {end_date}")
    print(f"Dynamic variables: {', '.join(DYNAMIC_VARS)}")
    print(f"Target variable: {TARGET_VAR}")
    print(f"Number of static attributes: {attrs_df.shape[1]}")
    print(f"Example basin ID: {example_basin_id}")
    print("======================================\n")


def load_all_basin_data() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    ds = MiniCamels()

    basins_df = ds.basins().copy()
    attrs_df = ds.attributes().copy()

    attrs_df["basin_id"] = basins_df["basin_id"].values.astype(str)
    attrs_df = attrs_df.set_index("basin_id")

    basin_timeseries: Dict[str, pd.DataFrame] = {}

    for basin_id in basins_df["basin_id"].astype(str):
        ts = ds.load_basin(basin_id)
        df = ts.to_dataframe().reset_index()

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        df = df[["time"] + DYNAMIC_VARS + [TARGET_VAR]].copy()
        df = df.dropna().reset_index(drop=True)
        df = df[df[TARGET_VAR] >= 0].reset_index(drop=True)

        basin_timeseries[basin_id] = df

    return attrs_df, basin_timeseries


def split_timeseries_by_time(
    df: pd.DataFrame,
    train_end: str = "2000-09-30",
    val_end: str = "2005-09-30",
    test_end: str = "2010-09-30",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    test_end = pd.Timestamp(test_end)

    train_df = df[df["time"] <= train_end].copy()
    val_df = df[(df["time"] > train_end) & (df["time"] <= val_end)].copy()
    test_df = df[(df["time"] > val_end) & (df["time"] <= test_end)].copy()

    return train_df, val_df, test_df


def compute_normalization_stats(
    attrs_df: pd.DataFrame,
    train_timeseries: Dict[str, pd.DataFrame],
) -> NormalizationStats:
    dynamic_all = []
    static_all = []

    for basin_id, df in train_timeseries.items():
        if len(df) == 0:
            continue

        dynamic_all.append(df[DYNAMIC_VARS].values.astype(np.float32))
        static_all.append(attrs_df.loc[basin_id, STATIC_VARS].values.astype(np.float32))

    if len(dynamic_all) == 0:
        raise ValueError("No training time series available to compute normalization stats.")

    dynamic_all = np.vstack(dynamic_all)
    static_all = np.vstack(static_all)

    dynamic_mean = dynamic_all.mean(axis=0)
    dynamic_std = dynamic_all.std(axis=0)
    dynamic_std = np.where(dynamic_std < 1e-8, 1.0, dynamic_std)

    static_mean = static_all.mean(axis=0)
    static_std = static_all.std(axis=0)
    static_std = np.where(static_std < 1e-8, 1.0, static_std)

    return NormalizationStats(
        dynamic_mean=dynamic_mean.astype(np.float32),
        dynamic_std=dynamic_std.astype(np.float32),
        static_mean=static_mean.astype(np.float32),
        static_std=static_std.astype(np.float32),
    )


def normalize_dynamic(x: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (x - stats.dynamic_mean) / stats.dynamic_std


def normalize_static(x: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (x - stats.static_mean) / stats.static_std


def build_samples_for_one_split(
    split_timeseries: Dict[str, pd.DataFrame],
    attrs_df: pd.DataFrame,
    stats: NormalizationStats,
    seq_len: int = 60,
    log_target: bool = True,
) -> List[Dict]:
    samples: List[Dict] = []

    for basin_id, df in split_timeseries.items():
        if len(df) <= seq_len:
            continue

        x_dyn = df[DYNAMIC_VARS].values.astype(np.float32)
        y = df[TARGET_VAR].values.astype(np.float32)
        times = df["time"].values

        x_dyn = normalize_dynamic(x_dyn, stats)

        x_static = attrs_df.loc[basin_id, STATIC_VARS].values.astype(np.float32)
        x_static = normalize_static(x_static, stats)

        if log_target:
            y = np.log1p(y)

        for i in range(len(df) - seq_len):
            x_seq = x_dyn[i : i + seq_len]
            y_target = y[i + seq_len]
            pred_time = str(pd.Timestamp(times[i + seq_len]).date())

            if np.isnan(x_seq).any() or np.isnan(y_target):
                continue

            samples.append(
                {
                    "x_seq": x_seq,
                    "x_static": x_static,
                    "y": float(y_target),
                    "basin_id": str(basin_id),
                    "pred_time": pred_time,
                }
            )

    return samples


def build_dataloaders(
    seq_len: int = 60,
    batch_size: int = 64,
    log_target: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    attrs_df, basin_timeseries = load_all_basin_data()

    train_ts: Dict[str, pd.DataFrame] = {}
    val_ts: Dict[str, pd.DataFrame] = {}
    test_ts: Dict[str, pd.DataFrame] = {}

    for basin_id, df in basin_timeseries.items():
        train_df, val_df, test_df = split_timeseries_by_time(df)
        train_ts[basin_id] = train_df
        val_ts[basin_id] = val_df
        test_ts[basin_id] = test_df

    stats = compute_normalization_stats(attrs_df, train_ts)

    train_samples = build_samples_for_one_split(
        train_ts, attrs_df, stats, seq_len=seq_len, log_target=log_target
    )
    val_samples = build_samples_for_one_split(
        val_ts, attrs_df, stats, seq_len=seq_len, log_target=log_target
    )
    test_samples = build_samples_for_one_split(
        test_ts, attrs_df, stats, seq_len=seq_len, log_target=log_target
    )

    train_dataset = StreamflowDataset(train_samples)
    val_dataset = StreamflowDataset(val_samples)
    test_dataset = StreamflowDataset(test_samples)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    metadata = {
        "dynamic_vars": DYNAMIC_VARS,
        "static_vars": STATIC_VARS,
        "target_var": TARGET_VAR,
        "seq_len": seq_len,
        "n_train_samples": len(train_dataset),
        "n_val_samples": len(val_dataset),
        "n_test_samples": len(test_dataset),
        "stats": stats,
    }

    return train_loader, val_loader, test_loader, metadata
