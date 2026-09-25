"""Model training and evaluation helpers for alloy density prediction."""

from __future__ import annotations

import gc
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset

from aviary.core import Normalizer
from aviary.roost.data import CompositionData, collate_batch
from aviary.roost.model import Roost
from aviary.utils import (
    initialize_losses,
    initialize_model,
    initialize_optim,
    init_normalizers,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RF_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_density.pkl"
DEFAULT_ROOST_MODEL_DIR = PROJECT_ROOT / "models" / "roost_density"
DEFAULT_ROOST_MP_PRETRAIN_DIR = PROJECT_ROOT / "models" / "roost_mp_pretrain"

TARGET_NAME = "density"
TASK_DICT = {TARGET_NAME: "regression"}
LOSS_DICT = {TARGET_NAME: "L1"}


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert merged density JSON records to a Roost-ready DataFrame."""
    return pd.DataFrame(
        {
            "material_id": range(len(records)),
            "composition": [record["composition_str"] for record in records],
            TARGET_NAME: [record["density"] for record in records],
        }
    )


def regression_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> dict[str, float]:
    """Compute MAE, RMSE, and R2 for regression."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "R2": float(r2_score(y_true_arr, y_pred_arr)),
    }


def evaluate_mean_density_baseline(
    y_train: np.ndarray | list[float],
    y_test: np.ndarray | list[float],
) -> dict[str, Any]:
    """Predict train-set mean density for all samples; return metrics."""
    train_mean = float(np.mean(y_train))
    train_pred = np.full(len(y_train), train_mean)
    test_pred = np.full(len(y_test), train_mean)
    return {
        "train_mean": train_mean,
        "train_metrics": regression_metrics(y_train, train_pred),
        "test_metrics": regression_metrics(y_test, test_pred),
    }


def plot_parity(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot predicted vs actual values with a y=x reference line."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_arr, y_pred_arr, alpha=0.7, edgecolors="k", linewidths=0.3)
    lo = min(y_true_arr.min(), y_pred_arr.min())
    hi = max(y_true_arr.max(), y_pred_arr.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("Actual density (g/cm³)")
    ax.set_ylabel("Predicted density (g/cm³)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    return ax


def plot_parity_train_test(
    y_train: np.ndarray | list[float],
    y_train_pred: np.ndarray | list[float],
    y_test: np.ndarray | list[float],
    y_test_pred: np.ndarray | list[float],
    model_name: str,
) -> plt.Figure:
    """Plot side-by-side parity charts for train and test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_parity(y_train, y_train_pred, f"{model_name} (train)", ax=axes[0])
    plot_parity(y_test, y_test_pred, f"{model_name} (test)", ax=axes[1])
    fig.tight_layout()
    return fig


def plot_roost_training_history(
    history: pd.DataFrame | Path | str,
    metric: str = "MAE",
    target: str = TARGET_NAME,
    ax: plt.Axes | None = None,
    label_prefix: str | None = None,
) -> plt.Axes:
    """Plot a Roost training-history metric for train and validation splits."""
    history_df = pd.read_csv(history) if not isinstance(history, pd.DataFrame) else history.copy()
    required_columns = {"epoch", "split", "target", metric}
    missing_columns = required_columns.difference(history_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Training history is missing required columns: {missing}")

    target_history = history_df[history_df["target"] == target].copy()
    if target_history.empty:
        raise ValueError(f"No training-history rows found for target {target!r}")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    for split_name, split_history in target_history.groupby("split", sort=False):
        split_history = split_history.sort_values("epoch")
        label = str(split_name)
        if label_prefix is not None:
            label = f"{label_prefix} {label}"
        ax.plot(
            split_history["epoch"],
            split_history[metric],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=label,
        )

    ax.set_xlabel("Epoch")
    ylabel = f"{metric} (g/cm³)" if metric == "MAE" else metric
    ax.set_ylabel(ylabel)
    ax.set_title(f"{target} {metric} by epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def _ensemble_prediction_stats(member_preds: np.ndarray) -> dict[str, np.ndarray]:
    """Aggregate member predictions into mean, std, and confidence scores."""
    pred_mean = member_preds.mean(axis=0)
    pred_std = member_preds.std(axis=0, ddof=0)
    confidence = 1.0 / (1.0 + pred_std)
    return {
        "y_pred_mean": pred_mean,
        "y_pred_std": pred_std,
        "confidence": confidence,
    }


def train_predict_rf_ensemble(
    X_train: pd.DataFrame | np.ndarray,
    y_train: list[float] | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: list[float] | np.ndarray,
    n_members: int = 5,
    base_seed: int = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train an RF ensemble and return mean/std/confidence predictions."""
    train_preds = []
    test_preds = []
    for member_idx in range(n_members):
        model = train_random_forest(
            X_train,
            y_train,
            random_state=base_seed + member_idx,
            **kwargs,
        )
        train_preds.append(model.predict(X_train))
        test_preds.append(model.predict(X_test))

    train_member_preds = np.asarray(train_preds, dtype=float)
    test_member_preds = np.asarray(test_preds, dtype=float)
    train_stats = _ensemble_prediction_stats(train_member_preds)
    test_stats = _ensemble_prediction_stats(test_member_preds)

    y_train_arr = np.asarray(y_train, dtype=float)
    y_test_arr = np.asarray(y_test, dtype=float)
    return {
        "train_metrics": regression_metrics(y_train_arr, train_stats["y_pred_mean"]),
        "test_metrics": regression_metrics(y_test_arr, test_stats["y_pred_mean"]),
        "y_train_true": y_train_arr,
        "y_train_pred": train_stats["y_pred_mean"],
        "y_train_pred_std": train_stats["y_pred_std"],
        "y_train_confidence": train_stats["confidence"],
        "y_test_true": y_test_arr,
        "y_test_pred": test_stats["y_pred_mean"],
        "y_test_pred_std": test_stats["y_pred_std"],
        "y_test_confidence": test_stats["confidence"],
        "member_test_preds": test_member_preds,
    }


def train_test_metrics_table(
    model_name: str,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> pd.Series:
    """Flatten train and test metrics into one Series for comparison tables."""
    return pd.Series(
        {
            **{f"{model_name}_train_{k}": v for k, v in train_metrics.items()},
            **{f"{model_name}_test_{k}": v for k, v in test_metrics.items()},
        }
    )


def train_random_forest(
    X_train: pd.DataFrame | np.ndarray,
    y_train: list[float] | np.ndarray,
    **kwargs: Any,
) -> RandomForestRegressor:
    """Train a random forest regressor with sensible defaults."""
    params = {
        "n_estimators": 500,
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(kwargs)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def save_sklearn_model(model: RandomForestRegressor, path: Path | str | None = None) -> Path:
    """Persist a scikit-learn model to disk."""
    save_path = Path(path) if path is not None else DEFAULT_RF_MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as handle:
        pickle.dump(model, handle)
    return save_path


def load_sklearn_model(path: Path | str | None = None) -> RandomForestRegressor:
    """Load a scikit-learn model from disk."""
    load_path = Path(path) if path is not None else DEFAULT_RF_MODEL_PATH
    with load_path.open("rb") as handle:
        return pickle.load(handle)


def _make_roost_loaders(
    dataset: CompositionData,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders for Roost."""
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": False,
        "collate_fn": collate_batch,
    }
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        shuffle=False,
        batch_size=max(batch_size, 64),
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        shuffle=False,
        batch_size=max(batch_size, 64),
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_batch,
    )
    return train_loader, val_loader, test_loader


def _default_roost_model_params(
    n_targets: int,
    device: str,
    elem_embedding: str = "matscholar200",
) -> dict[str, Any]:
    """Return default Roost architecture parameters."""
    return {
        "task_dict": TASK_DICT,
        "robust": False,
        "n_targets": n_targets,
        "elem_embedding": elem_embedding,
        "elem_fea_len": 64,
        "n_graph": 3,
        "elem_heads": 3,
        "elem_gate": [256],
        "elem_msg": [256],
        "cry_heads": 3,
        "cry_gate": [256],
        "cry_msg": [256],
        "trunk_hidden": [1024, 512],
        "out_hidden": [256, 128, 64],
        "device": device,
    }


def _default_roost_setup_params(
    device: str,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-6,
) -> dict[str, Any]:
    """Return default Roost optimizer settings."""
    return {
        "optim": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": 0.9,
        "device": device,
    }


def _set_roost_embedding_trainable(model: Roost, trainable: bool) -> None:
    """Toggle gradient updates for Roost element-embedding parameters."""
    for name, param in model.named_parameters():
        if "embed" in name.lower():
            param.requires_grad = trainable


def _evaluate_roost_split(
    model: Roost,
    dataset: CompositionData,
    indices: list[int],
    normalizer_dict: dict[str, Normalizer | None],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Run Roost inference on an index subset and return metrics."""
    loader = DataLoader(
        Subset(dataset, indices),
        shuffle=False,
        batch_size=max(batch_size, 64),
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_batch,
    )
    y_true, y_pred = predict_roost(model, loader, normalizer_dict)
    return y_true, y_pred, regression_metrics(y_true, y_pred)


def _ensure_roost_validation_tracking(model: Any) -> None:
    """Initialize Aviary validation tracking so early stopping can see improvements."""
    best_val_scores = dict(getattr(model, "best_val_scores", {}) or {})
    for target_name in model.task_dict:
        best_val_scores.setdefault(target_name, float("inf"))
    model.best_val_scores = best_val_scores
    model.es_patience = 0


def _roost_metric_history_rows(
    epoch: int,
    split: str,
    metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten Aviary epoch metrics into rows for CSV persistence."""
    rows: list[dict[str, Any]] = []
    for target_name, target_metrics in metrics.items():
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "split": split,
            "target": target_name,
        }
        for metric_name, metric_value in target_metrics.items():
            row[metric_name] = float(metric_value)
        rows.append(row)
    return rows


def save_roost_training_history(
    history: pd.DataFrame | list[dict[str, Any]],
    path: Path | str,
) -> Path:
    """Persist Roost per-epoch metrics in a stable long CSV format."""
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history_df = history if isinstance(history, pd.DataFrame) else pd.DataFrame(history)
    preferred_columns = ["epoch", "split", "target", "Loss", "MAE", "RMSE", "Accuracy", "F1"]
    if history_df.empty:
        history_df = pd.DataFrame(columns=preferred_columns[:6])
    else:
        ordered_columns = [
            column for column in preferred_columns if column in history_df.columns
        ] + [
            column for column in history_df.columns if column not in preferred_columns
        ]
        history_df = history_df[ordered_columns].sort_values(
            ["epoch", "target", "split"],
            kind="stable",
        )

    history_df.to_csv(history_path, index=False)
    return history_path


def _run_roost_fit(
    model: Roost,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    normalizer_dict: dict[str, Normalizer | None],
    epochs: int,
    patience: int,
    run_id: int,
    verbose: bool,
    model_name: str = "roost_density",
    history_path: Path | str | None = None,
) -> pd.DataFrame:
    """Execute the Roost training loop with early stopping and metric history."""
    loss_func_dict = initialize_losses(TASK_DICT, LOSS_DICT, robust=False)
    _ensure_roost_validation_tracking(model)
    history_rows: list[dict[str, Any]] = []
    start_epoch = model.epoch

    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            model.epoch += 1
            completed_epoch = int(model.epoch)
            if verbose:
                print(f"Epoch: [{epoch}/{start_epoch + epochs - 1}]")

            train_metrics = model.evaluate(
                train_loader,
                loss_dict=loss_func_dict,
                optimizer=optimizer,
                normalizer_dict=normalizer_dict,
                action="train",
                verbose=verbose,
            )
            history_rows.extend(
                _roost_metric_history_rows(completed_epoch, "train", train_metrics)
            )

            stop_early = False
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = model.evaluate(
                        val_loader,
                        loss_dict=loss_func_dict,
                        optimizer=None,
                        normalizer_dict=normalizer_dict,
                        action="evaluate",
                        verbose=verbose,
                    )
                history_rows.extend(
                    _roost_metric_history_rows(
                        completed_epoch,
                        "validation",
                        val_metrics,
                    )
                )

                is_best: list[bool] = []
                for target_name in model.best_val_scores:
                    score_name = (
                        "MAE"
                        if model.task_dict[target_name] == "regression"
                        else "Accuracy"
                    )
                    score = val_metrics[target_name][score_name]
                    prev_best = model.best_val_scores[target_name]
                    is_best.append(score < prev_best)
                    model.best_val_scores[target_name] = min(prev_best, score)

                if any(is_best):
                    model.es_patience = 0
                else:
                    model.es_patience += 1
                    if patience and model.es_patience > patience:
                        print(
                            f"No improvement on validation set for {patience} "
                            "epochs, stopping early"
                        )
                        stop_early = True

            if stop_early:
                break

            scheduler.step()
            gc.collect()
    except KeyboardInterrupt:
        pass

    history_df = pd.DataFrame(history_rows)
    if history_path is not None:
        save_roost_training_history(history_df, history_path)
    return history_df


def _roost_training_results(
    model: Roost,
    dataset: CompositionData,
    train_idx: list[int],
    val_idx: list[int] | None,
    test_idx: list[int],
    normalizer_dict: dict[str, Normalizer | None],
    loss_func_dict: dict[str, Any],
    checkpoint_path: Path,
    batch_size: int,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    verbose: bool,
    training_history: pd.DataFrame | None = None,
    history_path: Path | None = None,
) -> dict[str, Any]:
    """Collect predictions, metrics, and artifacts after Roost training."""
    y_true, y_pred = predict_roost(model, test_loader, normalizer_dict)
    test_metrics = regression_metrics(y_true, y_pred)
    y_train_true, y_train_pred, train_metrics = _evaluate_roost_split(
        model, dataset, train_idx, normalizer_dict, batch_size
    )
    if val_idx is not None:
        y_val_true, y_val_pred, val_metrics = _evaluate_roost_split(
            model, dataset, val_idx, normalizer_dict, batch_size
        )
    else:
        y_val_true = np.asarray([], dtype=float)
        y_val_pred = np.asarray([], dtype=float)
        val_metrics = {}

    if verbose:
        print(
            "Roost train metrics: "
            f"MAE={train_metrics['MAE']:.4f}, "
            f"RMSE={train_metrics['RMSE']:.4f}, "
            f"R2={train_metrics['R2']:.4f}"
        )
        if val_metrics:
            print(
                "Roost validation metrics: "
                f"MAE={val_metrics['MAE']:.4f}, "
                f"RMSE={val_metrics['RMSE']:.4f}, "
                f"R2={val_metrics['R2']:.4f}"
            )
        print(
            "Roost test metrics: "
            f"MAE={test_metrics['MAE']:.4f}, "
            f"RMSE={test_metrics['RMSE']:.4f}, "
            f"R2={test_metrics['R2']:.4f}"
        )
        print(f"Saved checkpoint to {checkpoint_path}")

    return {
        "model": model,
        "normalizer_dict": normalizer_dict,
        "loss_func_dict": loss_func_dict,
        "checkpoint_path": checkpoint_path,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "y_train_true": y_train_true,
        "y_train_pred": y_train_pred,
        "y_val_true": y_val_true,
        "y_val_pred": y_val_pred,
        "y_true": y_true,
        "y_pred": y_pred,
        "test_loader": test_loader,
        "optimizer": optimizer,
        "training_history": (
            training_history if training_history is not None else pd.DataFrame()
        ),
        "history_path": history_path,
    }


def _fit_target_normalizer(
    dataset: CompositionData,
    train_idx: list[int],
    normalizer_dict: dict[str, Normalizer | None],
) -> None:
    """Fit the regression target normalizer on training targets."""
    normalizer = normalizer_dict[TARGET_NAME]
    if normalizer is None:
        return
    train_targets = torch.tensor(
        dataset.df.iloc[train_idx][TARGET_NAME].to_numpy(),
        dtype=torch.float32,
    )
    normalizer.fit(train_targets)


def save_roost_checkpoint(
    model: Roost,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    normalizer_dict: dict[str, Normalizer | None],
    model_dir: Path | str,
    run_id: int = 0,
) -> Path:
    """Save a Roost checkpoint to a project-local directory."""
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir_path / f"checkpoint-r{run_id}.pth.tar"
    checkpoint = {
        "model_params": model.model_params,
        "state_dict": model.state_dict(),
        "epoch": model.epoch,
        "best_val_score": model.best_val_scores,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "normalizer_dict": {
            task: normalizer.state_dict() if isinstance(normalizer, Normalizer) else None
            for task, normalizer in normalizer_dict.items()
        },
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_roost_checkpoint(
    path: Path | str,
    device: str | None = None,
) -> tuple[Roost, dict[str, Normalizer | None], dict[str, Any]]:
    """Load a Roost model and normalizers from a checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_params = dict(checkpoint["model_params"])
    model_params["device"] = device
    model = Roost(**model_params)
    model.to(device)
    model.device = torch.device(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.epoch = checkpoint.get("epoch", 0)
    model.best_val_scores = checkpoint.get("best_val_score", {})

    normalizer_dict: dict[str, Normalizer | None] = {}
    for task, state_dict in checkpoint["normalizer_dict"].items():
        normalizer_dict[task] = (
            Normalizer.from_state_dict(state_dict) if state_dict is not None else None
        )

    return model, normalizer_dict, checkpoint


def predict_roost(
    model: Roost,
    data_loader: DataLoader,
    normalizer_dict: dict[str, Normalizer | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict density values and return denormalized targets and predictions."""
    targets, preds, _ = model.predict(data_loader)
    normalizer = normalizer_dict[TARGET_NAME]
    assert normalizer is not None

    y_true = targets[0]
    y_pred = normalizer.denorm(preds[0].squeeze()).cpu().numpy()
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def evaluate_roost_checkpoint(
    df: pd.DataFrame,
    train_idx: list[int],
    test_idx: list[int],
    checkpoint_path: Path | str,
    val_idx: list[int] | None = None,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, Any]:
    """Evaluate a saved Roost checkpoint without training (zero-shot inference)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer_dict, _ = load_roost_checkpoint(checkpoint_path, device=device)
    model.eval()

    dataset = CompositionData(df=df, task_dict=TASK_DICT)
    loader_kwargs = {
        "batch_size": max(batch_size, 64),
        "num_workers": 0,
        "pin_memory": False,
        "collate_fn": collate_batch,
    }
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        shuffle=False,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(
            Subset(dataset, val_idx),
            shuffle=False,
            **loader_kwargs,
        )
        if val_idx is not None
        else None
    )

    y_train_true, y_train_pred = predict_roost(model, train_loader, normalizer_dict)
    y_test_true, y_test_pred = predict_roost(model, test_loader, normalizer_dict)
    if val_loader is not None:
        y_val_true, y_val_pred = predict_roost(model, val_loader, normalizer_dict)
        val_metrics = regression_metrics(y_val_true, y_val_pred)
    else:
        y_val_true = np.asarray([], dtype=float)
        y_val_pred = np.asarray([], dtype=float)
        val_metrics = {}

    return {
        "model": model,
        "normalizer_dict": normalizer_dict,
        "checkpoint_path": Path(checkpoint_path),
        "train_metrics": regression_metrics(y_train_true, y_train_pred),
        "val_metrics": val_metrics,
        "test_metrics": regression_metrics(y_test_true, y_test_pred),
        "y_train_true": y_train_true,
        "y_train_pred": y_train_pred,
        "y_val_true": y_val_true,
        "y_val_pred": y_val_pred,
        "y_true": y_test_true,
        "y_pred": y_test_pred,
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred,
    }


def train_roost(
    df: pd.DataFrame,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    model_dir: Path | str | None = None,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 20,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-6,
    elem_embedding: str = "matscholar200",
    device: str | None = None,
    run_id: int = 0,
    verbose: bool = True,
    history_path: Path | str | None = None,
) -> dict[str, Any]:
    """Train a Roost model from scratch and save checkpoints."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir_path = Path(model_dir) if model_dir is not None else DEFAULT_ROOST_MODEL_DIR
    history_path_obj = Path(history_path) if history_path is not None else None

    dataset = CompositionData(df=df, task_dict=TASK_DICT)
    train_loader, val_loader, test_loader = _make_roost_loaders(
        dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=batch_size,
    )

    model_params = _default_roost_model_params(
        n_targets=dataset.n_targets,
        device=device,
        elem_embedding=elem_embedding,
    )
    setup_params = _default_roost_setup_params(
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    model = initialize_model(
        model_class=Roost,
        model_params=model_params,
        device=device,
    )
    optimizer, scheduler = initialize_optim(model, **setup_params)
    loss_func_dict = initialize_losses(TASK_DICT, LOSS_DICT, robust=False)
    normalizer_dict = init_normalizers(TASK_DICT, device=device)
    _fit_target_normalizer(dataset, train_idx, normalizer_dict)

    if verbose:
        print(f"Training Roost on {device} for up to {epochs} epochs")

    training_history = _run_roost_fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        normalizer_dict=normalizer_dict,
        epochs=epochs,
        patience=patience,
        run_id=run_id,
        verbose=verbose,
        history_path=history_path_obj,
    )

    checkpoint_path = save_roost_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        normalizer_dict=normalizer_dict,
        model_dir=model_dir_path,
        run_id=run_id,
    )

    return _roost_training_results(
        model=model,
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        normalizer_dict=normalizer_dict,
        loss_func_dict=loss_func_dict,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        test_loader=test_loader,
        optimizer=optimizer,
        verbose=verbose,
        training_history=training_history,
        history_path=history_path_obj,
    )


def pretrain_roost(
    df: pd.DataFrame,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    model_dir: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Pretrain Roost on a source dataset (e.g. Materials Project density)."""
    model_dir_path = Path(model_dir) if model_dir is not None else DEFAULT_ROOST_MP_PRETRAIN_DIR
    return train_roost(
        df=df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        model_dir=model_dir_path,
        **kwargs,
    )


def finetune_roost(
    df: pd.DataFrame,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    checkpoint_path: Path | str,
    model_dir: Path | str | None = None,
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 20,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
    device: str | None = None,
    run_id: int = 0,
    freeze_embeddings: bool = False,
    verbose: bool = True,
    history_path: Path | str | None = None,
) -> dict[str, Any]:
    """Fine-tune a pretrained Roost checkpoint on a target dataset."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir_path = Path(model_dir) if model_dir is not None else DEFAULT_ROOST_MODEL_DIR
    history_path_obj = Path(history_path) if history_path is not None else None

    dataset = CompositionData(df=df, task_dict=TASK_DICT)
    train_loader, val_loader, test_loader = _make_roost_loaders(
        dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=batch_size,
    )

    model, _, _ = load_roost_checkpoint(checkpoint_path, device=device)
    model.epoch = 0
    model.best_val_scores = {}

    for param in model.parameters():
        param.requires_grad = True
    if freeze_embeddings:
        _set_roost_embedding_trainable(model, trainable=False)

    setup_params = _default_roost_setup_params(
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer, scheduler = initialize_optim(model, **setup_params)
    loss_func_dict = initialize_losses(TASK_DICT, LOSS_DICT, robust=False)
    normalizer_dict = init_normalizers(TASK_DICT, device=device)
    _fit_target_normalizer(dataset, train_idx, normalizer_dict)

    if verbose:
        print(f"Fine-tuning Roost on {device} for up to {epochs} epochs")
        print(f"Loaded checkpoint from {checkpoint_path}")

    training_history = _run_roost_fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        normalizer_dict=normalizer_dict,
        epochs=epochs,
        patience=patience,
        run_id=run_id,
        verbose=verbose,
        model_name="roost_density_finetune",
        history_path=history_path_obj,
    )

    checkpoint_path_out = save_roost_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        normalizer_dict=normalizer_dict,
        model_dir=model_dir_path,
        run_id=run_id,
    )

    return _roost_training_results(
        model=model,
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        normalizer_dict=normalizer_dict,
        loss_func_dict=loss_func_dict,
        checkpoint_path=checkpoint_path_out,
        batch_size=batch_size,
        test_loader=test_loader,
        optimizer=optimizer,
        verbose=verbose,
        training_history=training_history,
        history_path=history_path_obj,
    )


def train_predict_roost_ensemble(
    df: pd.DataFrame,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    y_train: list[float] | np.ndarray,
    y_test: list[float] | np.ndarray,
    model_dir: Path | str,
    n_members: int = 5,
    base_seed: int = 0,
    split_seed: int | None = None,
    on_member_start: Any | None = None,
    history_dir: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train a Roost ensemble and return mean/std/confidence test predictions."""
    model_dir_path = Path(model_dir)
    history_dir_path = Path(history_dir) if history_dir is not None else None
    split_label = split_seed if split_seed is not None else "unknown"

    train_preds = []
    test_preds = []
    checkpoint_paths: list[Path] = []
    history_paths: list[Path] = []

    for member_idx in range(n_members):
        member_seed = base_seed + member_idx
        if on_member_start is not None:
            on_member_start(member_idx, n_members, member_seed)

        torch.manual_seed(member_seed)
        np.random.seed(member_seed)

        member_dir = model_dir_path / f"split{split_label}" / f"member{member_idx}"
        member_kwargs = {"verbose": False, **kwargs}
        if history_dir_path is not None:
            member_kwargs["history_path"] = (
                history_dir_path / f"split{split_label}_member{member_idx}.csv"
            )
        results = train_roost(
            df=df,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            model_dir=member_dir,
            run_id=member_idx,
            **member_kwargs,
        )
        checkpoint_paths.append(results["checkpoint_path"])
        if results["history_path"] is not None:
            history_paths.append(results["history_path"])
        train_preds.append(results["y_train_pred"])
        test_preds.append(results["y_pred"])

    train_member_preds = np.asarray(train_preds, dtype=float)
    test_member_preds = np.asarray(test_preds, dtype=float)
    train_stats = _ensemble_prediction_stats(train_member_preds)
    test_stats = _ensemble_prediction_stats(test_member_preds)

    y_train_arr = np.asarray(y_train, dtype=float)
    y_test_arr = np.asarray(y_test, dtype=float)
    return {
        "train_metrics": regression_metrics(y_train_arr, train_stats["y_pred_mean"]),
        "test_metrics": regression_metrics(y_test_arr, test_stats["y_pred_mean"]),
        "y_train_true": y_train_arr,
        "y_train_pred": train_stats["y_pred_mean"],
        "y_train_pred_std": train_stats["y_pred_std"],
        "y_train_confidence": train_stats["confidence"],
        "y_test_true": y_test_arr,
        "y_test_pred": test_stats["y_pred_mean"],
        "y_test_pred_std": test_stats["y_pred_std"],
        "y_test_confidence": test_stats["confidence"],
        "member_test_preds": test_member_preds,
        "checkpoint_paths": checkpoint_paths,
        "history_paths": history_paths,
    }


def train_predict_roost_transfer_ensemble(
    df: pd.DataFrame,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    y_train: list[float] | np.ndarray,
    y_test: list[float] | np.ndarray,
    checkpoint_path: Path | str,
    model_dir: Path | str,
    n_members: int = 5,
    base_seed: int = 0,
    split_seed: int | None = None,
    on_member_start: Any | None = None,
    history_dir: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fine-tune a pretrained Roost ensemble and return aggregated predictions."""
    model_dir_path = Path(model_dir)
    history_dir_path = Path(history_dir) if history_dir is not None else None
    split_label = split_seed if split_seed is not None else "unknown"

    train_preds = []
    test_preds = []
    checkpoint_paths: list[Path] = []
    history_paths: list[Path] = []

    for member_idx in range(n_members):
        member_seed = base_seed + member_idx
        if on_member_start is not None:
            on_member_start(member_idx, n_members, member_seed)

        torch.manual_seed(member_seed)
        np.random.seed(member_seed)

        member_dir = model_dir_path / f"split{split_label}" / f"member{member_idx}"
        member_kwargs = {"verbose": False, **kwargs}
        if history_dir_path is not None:
            member_kwargs["history_path"] = (
                history_dir_path / f"split{split_label}_member{member_idx}.csv"
            )
        results = finetune_roost(
            df=df,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            checkpoint_path=checkpoint_path,
            model_dir=member_dir,
            run_id=member_idx,
            **member_kwargs,
        )
        checkpoint_paths.append(results["checkpoint_path"])
        if results["history_path"] is not None:
            history_paths.append(results["history_path"])
        train_preds.append(results["y_train_pred"])
        test_preds.append(results["y_pred"])

    train_member_preds = np.asarray(train_preds, dtype=float)
    test_member_preds = np.asarray(test_preds, dtype=float)
    train_stats = _ensemble_prediction_stats(train_member_preds)
    test_stats = _ensemble_prediction_stats(test_member_preds)

    y_train_arr = np.asarray(y_train, dtype=float)
    y_test_arr = np.asarray(y_test, dtype=float)
    return {
        "train_metrics": regression_metrics(y_train_arr, train_stats["y_pred_mean"]),
        "test_metrics": regression_metrics(y_test_arr, test_stats["y_pred_mean"]),
        "y_train_true": y_train_arr,
        "y_train_pred": train_stats["y_pred_mean"],
        "y_train_pred_std": train_stats["y_pred_std"],
        "y_train_confidence": train_stats["confidence"],
        "y_test_true": y_test_arr,
        "y_test_pred": test_stats["y_pred_mean"],
        "y_test_pred_std": test_stats["y_pred_std"],
        "y_test_confidence": test_stats["confidence"],
        "member_test_preds": test_member_preds,
        "checkpoint_paths": checkpoint_paths,
        "history_paths": history_paths,
    }


def augment_with_physical_features(X: pd.DataFrame) -> pd.DataFrame:
    """Append mixing-rule and atoms-density columns to a composition matrix."""
    from features import physical_density_features

    return pd.concat([X, physical_density_features(X)], axis=1)


def train_predict_rf_ensemble_physical(
    X_train: pd.DataFrame | np.ndarray,
    y_train: list[float] | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: list[float] | np.ndarray,
    n_members: int = 5,
    base_seed: int = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train an RF ensemble on composition plus physical density features."""
    X_train_aug = augment_with_physical_features(pd.DataFrame(X_train))
    X_test_aug = augment_with_physical_features(pd.DataFrame(X_test))
    return train_predict_rf_ensemble(
        X_train_aug,
        y_train,
        X_test_aug,
        y_test,
        n_members=n_members,
        base_seed=base_seed,
        **kwargs,
    )


def train_predict_roost_ensemble_physics_embed(
    df: pd.DataFrame,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    y_train: list[float] | np.ndarray,
    y_test: list[float] | np.ndarray,
    model_dir: Path | str,
    n_members: int = 5,
    base_seed: int = 0,
    split_seed: int | None = None,
    on_member_start: Any | None = None,
    elem_embedding: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train a Roost ensemble with radius+mass element embeddings."""
    from features import DEFAULT_PHYSICAL_EMBEDDING_PATH

    embedding_path = (
        str(elem_embedding)
        if elem_embedding is not None
        else str(DEFAULT_PHYSICAL_EMBEDDING_PATH)
    )
    return train_predict_roost_ensemble(
        df=df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        y_train=y_train,
        y_test=y_test,
        model_dir=model_dir,
        n_members=n_members,
        base_seed=base_seed,
        split_seed=split_seed,
        on_member_start=on_member_start,
        elem_embedding=embedding_path,
        **kwargs,
    )
