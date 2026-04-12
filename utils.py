"""
Utility functions for metrics, seeding, and file I/O.
"""

import json
import os
import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Nash-Sutcliffe Efficiency.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 1e-12:
        return float("nan")

    num = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - num / denom)


def save_json(data: Dict, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
