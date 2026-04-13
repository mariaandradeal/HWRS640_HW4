"""
Training and validation routines for streamflow prediction.
"""

import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from data import build_dataloaders
from model import create_model
from utils import rmse, mae, nse, kge, ensure_dir, save_json, set_seed


def inverse_transform_target(y: np.ndarray, log_target: bool = True) -> np.ndarray:
    """
    Convert target values back to original streamflow scale.
    """
    if log_target:
        return np.expm1(y)
    return y


def run_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Run one training or evaluation epoch.

    Returns
    -------
    epoch_loss : float
    y_true_all : np.ndarray
    y_pred_all : np.ndarray
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_count = 0

    all_true = []
    all_pred = []

    for batch in dataloader:
        x_seq = batch["x_seq"].to(device)
        x_static = batch["x_static"].to(device)
        y = batch["y"].to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            y_pred = model(x_seq, x_static)
            loss = criterion(y_pred, y)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = x_seq.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        all_true.append(y.detach().cpu().numpy())
        all_pred.append(y_pred.detach().cpu().numpy())

    epoch_loss = total_loss / max(total_count, 1)
    y_true_all = np.vstack(all_true).reshape(-1)
    y_pred_all = np.vstack(all_pred).reshape(-1)

    return epoch_loss, y_true_all, y_pred_all


def compute_epoch_metrics(
    y_true_transformed: np.ndarray,
    y_pred_transformed: np.ndarray,
    log_target: bool = True,
) -> Dict[str, float]:
    """
    Compute metrics on original streamflow scale.
    """
    y_true = inverse_transform_target(y_true_transformed, log_target=log_target)
    y_pred = inverse_transform_target(y_pred_transformed, log_target=log_target)

    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nse": nse(y_true, y_pred),
        "kge": kge(y_true, y_pred),
    }


def train_model(
    seq_len: int = 60,
    batch_size: int = 64,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.1,
    static_hidden_size: int = 32,
    fusion_hidden_size: int = 32,
    learning_rate: float = 1e-3,
    epochs: int = 20,
    log_target: bool = True,
    seed: int = 42,
    output_dir: str = "outputs",
) -> Dict:
    """
    Full training routine.

    Returns
    -------
    results : dict
        Dictionary containing history and checkpoint path.
    """
    set_seed(seed)

    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "checkpoints"))
    ensure_dir(os.path.join(output_dir, "metrics"))

    train_loader, val_loader, test_loader, meta = build_dataloaders(
        seq_len=seq_len,
        batch_size=batch_size,
        log_target=log_target,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(
        num_dynamic_features=len(meta["dynamic_vars"]),
        num_static_features=len(meta["static_vars"]),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        static_hidden_size=static_hidden_size,
        fusion_hidden_size=fusion_hidden_size,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_rmse": [],
        "val_rmse": [],
        "train_mae": [],
        "val_mae": [],
        "train_nse": [],
        "val_nse": [],
        "train_kge": [],
        "val_kge": [],
    }

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(output_dir, "checkpoints", "best_model.pt")

    print(f"Using device: {device}")
    print(f"Training samples: {meta['n_train_samples']}")
    print(f"Validation samples: {meta['n_val_samples']}")
    print(f"Test samples: {meta['n_test_samples']}")

    for epoch in range(1, epochs + 1):
        train_loss, y_train_true, y_train_pred = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_loss, y_val_true, y_val_pred = run_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        train_metrics = compute_epoch_metrics(
            y_train_true, y_train_pred, log_target=log_target
        )
        val_metrics = compute_epoch_metrics(
            y_val_true, y_val_pred, log_target=log_target
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(train_metrics["rmse"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["train_mae"].append(train_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])
        history["train_nse"].append(train_metrics["nse"])
        history["val_nse"].append(val_metrics["nse"])
        history["train_kge"].append(train_metrics["kge"])
        history["val_kge"].append(val_metrics["kge"])

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train NSE: {train_metrics['nse']:.4f} | Val NSE: {val_metrics['nse']:.4f}"
            f"Train KGE: {train_metrics['kge']:.4f} | Val KGE: {val_metrics['kge']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "num_dynamic_features": len(meta["dynamic_vars"]),
                        "num_static_features": len(meta["static_vars"]),
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "static_hidden_size": static_hidden_size,
                        "fusion_hidden_size": fusion_hidden_size,
                    },
                    "meta": {
                        "dynamic_vars": meta["dynamic_vars"],
                        "static_vars": meta["static_vars"],
                        "target_var": meta["target_var"],
                        "seq_len": meta["seq_len"],
                        "log_target": log_target,
                    },
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                best_checkpoint_path,
            )

    history_path = os.path.join(output_dir, "metrics", "training_history.json")
    save_json(history, history_path)

    results = {
        "history": history,
        "best_checkpoint_path": best_checkpoint_path,
        "history_path": history_path,
        "output_dir": output_dir,
    }

    return results


def load_checkpoint(checkpoint_path: str, device: torch.device = None):
    """
    Load a trained model checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    cfg = checkpoint["model_config"]
    model = create_model(
        num_dynamic_features=cfg["num_dynamic_features"],
        num_static_features=cfg["num_static_features"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        static_hidden_size=cfg["static_hidden_size"],
        fusion_hidden_size=cfg["fusion_hidden_size"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def evaluate_model(
    checkpoint_path: str,
    batch_size: int = 64,
    log_target: bool = True,
) -> Dict:
    """
    Evaluate the saved model on the test set.
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

    criterion = nn.MSELoss()

    test_loss, y_test_true, y_test_pred = run_one_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )

    test_metrics = compute_epoch_metrics(
        y_test_true,
        y_test_pred,
        log_target=log_target,
    )

    results = {
        "test_loss": test_loss,
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_nse": test_metrics["nse"],
        "test_kge": test_metrics["kge"],
    }

    return results
