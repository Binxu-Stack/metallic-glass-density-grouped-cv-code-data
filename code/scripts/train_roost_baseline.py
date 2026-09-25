"""Run Roost scratch under the Phase 4 random/grouped CV protocol."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from data_loader import load_merged_density_data
from features import elements_composition_features
from models import (
    _ensemble_prediction_stats,
    evaluate_roost_checkpoint,
    records_to_dataframe,
    regression_metrics,
    train_roost,
)
from train_test_split import group_kfold_indices, group_split_indices, random_kfold_indices

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase4_cv_roost_scratch"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "roost_scratch_cv"
DEFAULT_CLUSTER_ASSIGNMENTS = PROJECT_ROOT / "data" / "composition_families_k36.csv"
DEFAULT_PHYSICAL_SUMMARY = PROJECT_ROOT / "results" / "phase4_cv_physical_baselines" / "physical_baseline_cv_summary.csv"
DEFAULT_PHYSICAL_OOF = PROJECT_ROOT / "results" / "phase4_cv_physical_baselines" / "physical_baseline_cv_oof_summary.csv"
DEFAULT_RF_SUMMARY = PROJECT_ROOT / "results" / "phase4_cv_random_forest" / "rf_cv_summary.csv"
DEFAULT_RF_OOF = PROJECT_ROOT / "results" / "phase4_cv_random_forest" / "rf_cv_oof_summary.csv"
BP_SI_ELEMENTS = {"B", "P", "Si"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=None,
        help="Directory for per-member training-history CSVs. Defaults to output-dir/training_history.",
    )
    parser.add_argument("--cluster-assignments", type=Path, default=DEFAULT_CLUSTER_ASSIGNMENTS)
    parser.add_argument("--physical-summary", type=Path, default=DEFAULT_PHYSICAL_SUMMARY)
    parser.add_argument("--physical-oof-summary", type=Path, default=DEFAULT_PHYSICAL_OOF)
    parser.add_argument("--rf-summary", type=Path, default=DEFAULT_RF_SUMMARY)
    parser.add_argument("--rf-oof-summary", type=Path, default=DEFAULT_RF_OOF)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--protocols",
        choices=["random", "grouped"],
        nargs="+",
        default=["random", "grouped"],
        help="CV protocols to run. Defaults to both random and grouped.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Specific fold numbers to run. Defaults to all folds.",
    )
    parser.add_argument(
        "--validation-protocol",
        choices=["auto", "group_val", "random_val"],
        default="auto",
        help="Validation split inside each CV training pool. Auto uses random_val for random CV and group_val for grouped CV.",
    )
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--member-start", type=int, default=0)
    parser.add_argument("--n-members", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Retrain even if a checkpoint exists.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _load_inputs(
    cluster_assignments_path: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    records = load_merged_density_data()
    roost_df = records_to_dataframe(records)
    df_comp = elements_composition_features([record["composition"] for record in records])
    y = pd.Series([record["density"] for record in records], name="density")

    cluster_assignments = pd.read_csv(cluster_assignments_path)
    required_columns = {"record_index", "cluster_kmeans_36", "family_label"}
    missing = required_columns - set(cluster_assignments.columns)
    if missing:
        raise ValueError(f"Cluster assignment file is missing columns: {sorted(missing)}")
    cluster_assignments = cluster_assignments.set_index("record_index").sort_index()

    expected_index = pd.Index(range(len(records)), name="record_index")
    if not cluster_assignments.index.equals(expected_index):
        raise ValueError("Cluster assignments must contain one row per record_index in order.")

    groups = cluster_assignments["cluster_kmeans_36"]
    return records, roost_df, df_comp, y, cluster_assignments, groups


def _has_bp_si(record: dict[str, Any]) -> bool:
    composition = record["composition"]
    return any(composition.get(element, 0.0) > 0.0 for element in BP_SI_ELEMENTS)


def _cluster_set(indices: list[int], cluster_assignments: pd.DataFrame) -> set[int]:
    return set(cluster_assignments.loc[indices, "cluster_kmeans_36"].astype(int))


def _validation_protocol_for(cv_protocol: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "group_val" if cv_protocol.startswith("grouped_") else "random_val"


def _protocol_seed_offset(cv_protocol: str) -> int:
    return 1 if cv_protocol.startswith("grouped_") else 0


def _split_train_validation(
    df_comp: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    train_pool_idx: list[int],
    validation_protocol: str,
    val_size: float,
    random_state: int,
) -> tuple[list[int], list[int]]:
    if validation_protocol == "group_val":
        train_pool = df_comp.loc[train_pool_idx]
        y_pool = y.loc[train_pool_idx]
        groups_pool = groups.loc[train_pool_idx]
        return group_split_indices(
            train_pool,
            y_pool,
            groups_pool,
            test_size=val_size,
            random_state=random_state,
        )

    train_idx, val_idx = train_test_split(
        train_pool_idx,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )
    return list(train_idx), list(val_idx)


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    best_val_score = {
        str(key): float(value)
        for key, value in dict(checkpoint.get("best_val_score", {}) or {}).items()
    }
    return {
        "checkpoint_path": _path_label(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "best_val_score": json.dumps(best_val_score, sort_keys=True),
        "best_val_score_nonempty": bool(best_val_score),
    }


def _member_indices(start: int, count: int) -> range:
    return range(start, start + count)


def _upsert_csv(path: Path, rows: list[dict[str, Any]], key_columns: list[str]) -> pd.DataFrame:
    new_df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=key_columns, keep="last")
    combined = combined.sort_values(key_columns).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def _metric_row(
    cv_protocol: str,
    fold: int,
    validation_protocol: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    member_start: int,
    n_members: int,
    epochs: int,
    patience: int,
    base_seed: int,
) -> dict[str, Any]:
    return {
        "cv_protocol": cv_protocol,
        "fold": fold,
        "validation_protocol": validation_protocol,
        "split": split,
        "model": "Roost_scratch",
        "member_start": member_start,
        "n_members": n_members,
        "epochs_requested": epochs,
        "patience": patience,
        "base_seed": base_seed,
        "n_records": len(y_true),
        **regression_metrics(y_true, y_pred),
    }


def _prediction_rows(
    records: list[dict[str, Any]],
    cluster_assignments: pd.DataFrame,
    cv_protocol: str,
    fold: int,
    validation_protocol: str,
    split: str,
    member_start: int,
    n_members: int,
    epochs: int,
    patience: int,
    indices: list[int],
    y_true: np.ndarray,
    stats: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, record_index in enumerate(indices):
        cluster_row = cluster_assignments.loc[record_index]
        prediction = float(stats["y_pred_mean"][position])
        true_value = float(y_true[position])
        residual = prediction - true_value
        rows.append(
            {
                "cv_protocol": cv_protocol,
                "fold": fold,
                "validation_protocol": validation_protocol,
                "split": split,
                "model": "Roost_scratch",
                "member_start": member_start,
                "n_members": n_members,
                "epochs_requested": epochs,
                "patience": patience,
                "record_index": int(record_index),
                "composition_str": records[record_index]["composition_str"],
                "density": true_value,
                "prediction": prediction,
                "prediction_std": float(stats["y_pred_std"][position]),
                "confidence": float(stats["confidence"][position]),
                "residual_pred_minus_true": residual,
                "abs_error": abs(residual),
                "squared_error": residual**2,
                "cluster_kmeans_36": int(cluster_row["cluster_kmeans_36"]),
                "family_label": cluster_row["family_label"],
                "has_B_P_Si": _has_bp_si(records[record_index]),
            }
        )
    return rows


def _provenance_row(
    cv_protocol: str,
    fold: int,
    validation_protocol: str,
    train_pool_idx: list[int],
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    cluster_assignments: pd.DataFrame,
) -> dict[str, Any]:
    train_clusters = _cluster_set(train_idx, cluster_assignments)
    val_clusters = _cluster_set(val_idx, cluster_assignments)
    test_clusters = _cluster_set(test_idx, cluster_assignments)
    return {
        "cv_protocol": cv_protocol,
        "fold": fold,
        "validation_protocol": validation_protocol,
        "n_cv_train_pool": len(train_pool_idx),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "n_train_clusters": len(train_clusters),
        "n_val_clusters": len(val_clusters),
        "n_test_clusters": len(test_clusters),
        "train_val_overlap": len(set(train_idx) & set(val_idx)),
        "train_test_overlap": len(set(train_idx) & set(test_idx)),
        "val_test_overlap": len(set(val_idx) & set(test_idx)),
        "train_val_cluster_overlap": len(train_clusters & val_clusters),
        "train_test_cluster_overlap": len(train_clusters & test_clusters),
        "val_test_cluster_overlap": len(val_clusters & test_clusters),
        "test_clusters": ";".join(str(cluster) for cluster in sorted(test_clusters)),
    }


def _run_fold(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    roost_df: pd.DataFrame,
    df_comp: pd.DataFrame,
    y: pd.Series,
    cluster_assignments: pd.DataFrame,
    groups: pd.Series,
    cv_protocol: str,
    fold: int,
    train_pool_idx: list[int],
    test_idx: list[int],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    validation_protocol = _validation_protocol_for(cv_protocol, args.validation_protocol)
    protocol_offset = _protocol_seed_offset(cv_protocol)
    validation_seed = args.random_state + protocol_offset * 1000 + fold
    train_idx, val_idx = _split_train_validation(
        df_comp=df_comp,
        y=y,
        groups=groups,
        train_pool_idx=train_pool_idx,
        validation_protocol=validation_protocol,
        val_size=args.val_size,
        random_state=validation_seed,
    )

    provenance = _provenance_row(
        cv_protocol=cv_protocol,
        fold=fold,
        validation_protocol=validation_protocol,
        train_pool_idx=train_pool_idx,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        cluster_assignments=cluster_assignments,
    )

    y_train = y.loc[train_idx].to_numpy(dtype=float)
    y_val = y.loc[val_idx].to_numpy(dtype=float)
    y_test = y.loc[test_idx].to_numpy(dtype=float)
    fold_base_seed = args.base_seed + protocol_offset * 10000 + fold * 100

    member_rows: list[dict[str, Any]] = []
    train_preds: list[np.ndarray] = []
    val_preds: list[np.ndarray] = []
    test_preds: list[np.ndarray] = []
    for member_idx in _member_indices(args.member_start, args.n_members):
        member_seed = fold_base_seed + member_idx
        torch.manual_seed(member_seed)
        np.random.seed(member_seed)

        member_model_dir = (
            args.model_dir
            / cv_protocol
            / f"fold{fold}"
            / validation_protocol
            / f"member{member_idx}"
        )
        checkpoint_path = member_model_dir / f"checkpoint-r{member_idx}.pth.tar"
        history_path = (
            args.history_dir
            / f"{cv_protocol}_fold{fold}_{validation_protocol}_member{member_idx}.csv"
        )

        if checkpoint_path.exists() and not args.force:
            result = evaluate_roost_checkpoint(
                df=roost_df,
                train_idx=train_idx,
                test_idx=test_idx,
                checkpoint_path=checkpoint_path,
                val_idx=val_idx,
                batch_size=args.batch_size,
                device=args.device,
            )
            action = "evaluated_existing"
        else:
            result = train_roost(
                df=roost_df,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                model_dir=member_model_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
                device=args.device,
                run_id=member_idx,
                verbose=args.verbose,
                history_path=history_path,
            )
            checkpoint_path = Path(result["checkpoint_path"])
            action = "trained"

        checkpoint_meta = _checkpoint_metadata(checkpoint_path)
        if not checkpoint_meta["best_val_score_nonempty"]:
            raise RuntimeError(f"Checkpoint has empty best_val_score: {checkpoint_path}")

        result_history_path = result.get("history_path")
        if result_history_path is not None:
            history_path = Path(result_history_path)
        history_path_label = _path_label(history_path) if history_path.exists() else ""

        train_pred = np.asarray(result["y_train_pred"], dtype=float)
        val_pred = np.asarray(result["y_val_pred"], dtype=float)
        test_pred = np.asarray(result.get("y_pred", result.get("y_test_pred")), dtype=float)
        train_preds.append(train_pred)
        val_preds.append(val_pred)
        test_preds.append(test_pred)

        member_rows.append(
            {
                "cv_protocol": cv_protocol,
                "fold": fold,
                "validation_protocol": validation_protocol,
                "model": "Roost_scratch",
                "member_index": member_idx,
                "member_seed": member_seed,
                "action": action,
                "epochs_requested": args.epochs,
                "patience": args.patience,
                "n_cv_train_pool": len(train_pool_idx),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "training_history_path": history_path_label,
                "train_MAE": result["train_metrics"]["MAE"],
                "train_RMSE": result["train_metrics"]["RMSE"],
                "train_R2": result["train_metrics"]["R2"],
                "validation_MAE": result["val_metrics"]["MAE"],
                "validation_RMSE": result["val_metrics"]["RMSE"],
                "validation_R2": result["val_metrics"]["R2"],
                "test_MAE": result["test_metrics"]["MAE"],
                "test_RMSE": result["test_metrics"]["RMSE"],
                "test_R2": result["test_metrics"]["R2"],
                **checkpoint_meta,
            }
        )

    train_stats = _ensemble_prediction_stats(np.asarray(train_preds, dtype=float))
    val_stats = _ensemble_prediction_stats(np.asarray(val_preds, dtype=float))
    test_stats = _ensemble_prediction_stats(np.asarray(test_preds, dtype=float))
    metric_rows = [
        _metric_row(
            cv_protocol=cv_protocol,
            fold=fold,
            validation_protocol=validation_protocol,
            split="train",
            y_true=y_train,
            y_pred=train_stats["y_pred_mean"],
            member_start=args.member_start,
            n_members=args.n_members,
            epochs=args.epochs,
            patience=args.patience,
            base_seed=fold_base_seed,
        ),
        _metric_row(
            cv_protocol=cv_protocol,
            fold=fold,
            validation_protocol=validation_protocol,
            split="validation",
            y_true=y_val,
            y_pred=val_stats["y_pred_mean"],
            member_start=args.member_start,
            n_members=args.n_members,
            epochs=args.epochs,
            patience=args.patience,
            base_seed=fold_base_seed,
        ),
        _metric_row(
            cv_protocol=cv_protocol,
            fold=fold,
            validation_protocol=validation_protocol,
            split="test",
            y_true=y_test,
            y_pred=test_stats["y_pred_mean"],
            member_start=args.member_start,
            n_members=args.n_members,
            epochs=args.epochs,
            patience=args.patience,
            base_seed=fold_base_seed,
        ),
    ]
    validation_residual_rows = _prediction_rows(
        records=records,
        cluster_assignments=cluster_assignments,
        cv_protocol=cv_protocol,
        fold=fold,
        validation_protocol=validation_protocol,
        split="validation",
        member_start=args.member_start,
        n_members=args.n_members,
        epochs=args.epochs,
        patience=args.patience,
        indices=val_idx,
        y_true=y_val,
        stats=val_stats,
    )
    test_residual_rows = _prediction_rows(
        records=records,
        cluster_assignments=cluster_assignments,
        cv_protocol=cv_protocol,
        fold=fold,
        validation_protocol=validation_protocol,
        split="test",
        member_start=args.member_start,
        n_members=args.n_members,
        epochs=args.epochs,
        patience=args.patience,
        indices=test_idx,
        y_true=y_test,
        stats=test_stats,
    )

    return provenance, member_rows, metric_rows, validation_residual_rows, test_residual_rows


def _test_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    test_metrics = metrics_df[metrics_df["split"] == "test"]
    if test_metrics.empty:
        return pd.DataFrame()
    return (
        test_metrics.groupby(["cv_protocol", "model", "validation_protocol"], dropna=False)
        .agg(
            n_folds=("fold", "nunique"),
            n_test_records_total=("n_records", "sum"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MAE_min=("MAE", "min"),
            MAE_max=("MAE", "max"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .reset_index()
        .sort_values(["cv_protocol", "MAE_mean", "model"])
    )


def _split_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    return (
        metrics_df.groupby(
            ["cv_protocol", "model", "validation_protocol", "split"],
            dropna=False,
        )
        .agg(
            n_folds=("fold", "nunique"),
            n_records_total=("n_records", "sum"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MAE_min=("MAE", "min"),
            MAE_max=("MAE", "max"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .reset_index()
        .sort_values(["cv_protocol", "validation_protocol", "split", "MAE_mean"])
    )


def _oof_summary(residuals_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if residuals_df.empty:
        return pd.DataFrame()
    for (protocol, model, validation_protocol), group in residuals_df.groupby(
        ["cv_protocol", "model", "validation_protocol"],
        dropna=False,
    ):
        metrics = regression_metrics(group["density"].to_numpy(), group["prediction"].to_numpy())
        rows.append(
            {
                "cv_protocol": protocol,
                "model": model,
                "validation_protocol": validation_protocol,
                "n_records": len(group),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["cv_protocol", "MAE", "model"]).reset_index(drop=True)


def _protocol_contrast(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    value_cols = ["MAE_mean", "MAE_std", "RMSE_mean", "R2_mean", "R2_std"]
    wide = summary.pivot(
        index=["model", "validation_protocol"],
        columns="cv_protocol",
        values=value_cols,
    )
    wide.columns = [f"{protocol}_{metric}" for metric, protocol in wide.columns]
    wide = wide.reset_index()
    protocols = sorted(summary["cv_protocol"].unique())
    random_protocol = next((protocol for protocol in protocols if protocol.startswith("random_")), None)
    grouped_protocol = next((protocol for protocol in protocols if protocol.startswith("grouped_")), None)
    for metric in ("MAE_mean", "RMSE_mean", "R2_mean"):
        random_col = f"{random_protocol}_{metric}" if random_protocol is not None else ""
        grouped_col = f"{grouped_protocol}_{metric}" if grouped_protocol is not None else ""
        if random_col in wide.columns and grouped_col in wide.columns:
            wide[f"random_minus_grouped_{metric}"] = wide[random_col] - wide[grouped_col]
    sort_col = f"{grouped_protocol}_MAE_mean" if grouped_protocol is not None else "model"
    return wide.sort_values(sort_col).reset_index(drop=True)


def _cluster_test_summary(residuals_df: pd.DataFrame) -> pd.DataFrame:
    if residuals_df.empty:
        return pd.DataFrame()
    return (
        residuals_df.groupby(
            ["cv_protocol", "model", "cluster_kmeans_36", "family_label"],
            dropna=False,
        )
        .agg(
            n_records=("record_index", "size"),
            MAE=("abs_error", "mean"),
            RMSE=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
            mean_residual=("residual_pred_minus_true", "mean"),
            mean_density=("density", "mean"),
            mean_prediction_std=("prediction_std", "mean"),
            has_B_P_Si_count=("has_B_P_Si", "sum"),
        )
        .reset_index()
        .sort_values(["cv_protocol", "model", "MAE"], ascending=[True, True, False])
    )


def _with_model_family(df: pd.DataFrame) -> pd.DataFrame:
    family_map = {
        "Roost_scratch": "composition_neural",
        "random_forest_mendeleev": "composition_ml",
        "mixing_rule": "physical_baseline",
        "packing_eta_train_median": "physical_baseline",
        "packing_eta_train_mae_opt": "physical_baseline",
        "atoms_density_diagnostic": "physical_diagnostic",
    }
    output = df.copy()
    output["model_family"] = output["model"].map(family_map).fillna("unknown")
    return output


def _comparison_summary(
    roost_summary: pd.DataFrame,
    physical_summary_path: Path,
    rf_summary_path: Path,
) -> pd.DataFrame:
    frames = [_with_model_family(roost_summary)]
    for path in (physical_summary_path, rf_summary_path):
        if path.exists():
            frames.append(_with_model_family(pd.read_csv(path)))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.sort_values(["cv_protocol", "MAE_mean", "model"]).reset_index(drop=True)


def _comparison_oof_summary(
    roost_oof: pd.DataFrame,
    physical_oof_path: Path,
    rf_oof_path: Path,
) -> pd.DataFrame:
    frames = [_with_model_family(roost_oof)]
    for path in (physical_oof_path, rf_oof_path):
        if path.exists():
            frames.append(_with_model_family(pd.read_csv(path)))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.sort_values(["cv_protocol", "MAE", "model"]).reset_index(drop=True)


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    display = df.head(max_rows)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(_format_value(value) for value in row) + " |"
        for row in display.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _write_report(
    output_dir: Path,
    provenance_df: pd.DataFrame,
    roost_split_summary: pd.DataFrame,
    roost_summary: pd.DataFrame,
    roost_oof_summary: pd.DataFrame,
    protocol_contrast: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    comparison_oof_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    provenance_summary = (
        provenance_df.groupby(["cv_protocol", "validation_protocol"])
        .agg(
            n_folds=("fold", "nunique"),
            n_test_min=("n_test", "min"),
            n_test_max=("n_test", "max"),
            n_test_clusters_min=("n_test_clusters", "min"),
            n_test_clusters_max=("n_test_clusters", "max"),
            max_train_test_cluster_overlap=("train_test_cluster_overlap", "max"),
            max_val_test_cluster_overlap=("val_test_cluster_overlap", "max"),
        )
        .reset_index()
        if not provenance_df.empty
        else pd.DataFrame()
    )
    grouped_label = f"grouped_{args.n_splits}fold"
    random_label = f"random_{args.n_splits}fold"
    group_comparison = comparison_summary[comparison_summary["cv_protocol"] == grouped_label]
    random_comparison = comparison_summary[comparison_summary["cv_protocol"] == random_label]

    report = "\n".join(
        [
            "# Phase 4 CV Roost Scratch Report",
            "",
            f"Generated: {date.today().isoformat()}",
            "",
            "## Scope",
            "",
            "- 10-fold random CV is the interpolation benchmark.",
            "- 10-fold grouped CV uses Phase 3 k=36 cluster labels as groups.",
            f"- Roost scratch is evaluated with {args.n_members} member(s) per fold in this run.",
            f"- Requested epochs/patience: {args.epochs}/{args.patience}.",
            f"- Validation protocol request: `{args.validation_protocol}`.",
            "- Auto validation uses random validation for random CV and group validation for grouped CV.",
            "- Features for fold grouping come from parsed composition dictionaries; Roost receives the established composition-string dataframe.",
            "",
            "## Output Files",
            "",
            "- `roost_cv_fold_provenance.csv`",
            "- `roost_cv_member_metadata.csv`",
            "- `roost_cv_metrics.csv`",
            "- `roost_cv_split_summary.csv`",
            "- `roost_cv_validation_residuals.csv`",
            "- `roost_cv_test_residuals.csv`",
            "- `roost_cv_summary.csv`",
            "- `roost_cv_oof_summary.csv`",
            "- `roost_cv_protocol_contrast.csv`",
            "- `roost_cv_cluster_summary.csv`",
            "- `roost_cv_vs_baselines_summary.csv`",
            "- `roost_cv_vs_baselines_oof_summary.csv`",
            "- `phase4_cv_roost_scratch_report.md`",
            "",
            "## Fold Provenance Summary",
            "",
            _markdown_table(provenance_summary),
            "",
            "## Roost Fold-Mean Split Summary",
            "",
            _markdown_table(roost_split_summary),
            "",
            "## Roost Fold-Mean Test Summary",
            "",
            _markdown_table(roost_summary),
            "",
            "## Roost Pooled Out-of-Fold Summary",
            "",
            _markdown_table(roost_oof_summary),
            "",
            "## Roost Random-CV Minus Grouped-CV Contrast",
            "",
            _markdown_table(protocol_contrast),
            "",
            "## Grouped-CV Model Comparison",
            "",
            _markdown_table(group_comparison),
            "",
            "## Random-CV Model Comparison",
            "",
            _markdown_table(random_comparison),
            "",
            "## Pooled OOF Model Comparison",
            "",
            _markdown_table(comparison_oof_summary),
            "",
            "## Interpretation Notes",
            "",
            "- This report is comparable to the physical-baseline and RF CV packages only when the same fold count and selected folds are complete.",
            "- Grouped-CV results are the primary generalization yardstick.",
            "- Random-CV results are interpolation performance.",
            "",
        ]
    )
    (output_dir / "phase4_cv_roost_scratch_report.md").write_text(report, encoding="utf-8")


def _fold_maps(
    df_comp: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    protocols: list[str],
    n_splits: int,
    random_state: int,
) -> dict[str, list[tuple[list[int], list[int]]]]:
    maps: dict[str, list[tuple[list[int], list[int]]]] = {}
    if "random" in protocols:
        maps[f"random_{n_splits}fold"] = random_kfold_indices(
            df_comp,
            n_splits=n_splits,
            random_state=random_state,
        )
    if "grouped" in protocols:
        maps[f"grouped_{n_splits}fold"] = group_kfold_indices(
            df_comp,
            y,
            groups,
            n_splits=n_splits,
            random_state=random_state,
            shuffle=False,
        )
    return maps


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    if args.history_dir is None:
        args.history_dir = args.output_dir / "training_history"
    args.history_dir.mkdir(parents=True, exist_ok=True)

    records, roost_df, df_comp, y, cluster_assignments, groups = _load_inputs(args.cluster_assignments)
    fold_maps = _fold_maps(
        df_comp=df_comp,
        y=y,
        groups=groups,
        protocols=args.protocols,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    selected_folds = set(args.folds) if args.folds is not None else None

    provenance_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    validation_residual_rows: list[dict[str, Any]] = []
    test_residual_rows: list[dict[str, Any]] = []
    for cv_protocol, folds in fold_maps.items():
        for fold, (train_pool_idx, test_idx) in enumerate(folds):
            if selected_folds is not None and fold not in selected_folds:
                continue
            provenance, members, metrics, validation_residuals, test_residuals = _run_fold(
                args=args,
                records=records,
                roost_df=roost_df,
                df_comp=df_comp,
                y=y,
                cluster_assignments=cluster_assignments,
                groups=groups,
                cv_protocol=cv_protocol,
                fold=fold,
                train_pool_idx=train_pool_idx,
                test_idx=test_idx,
            )
            provenance_rows.append(provenance)
            member_rows.extend(members)
            metric_rows.extend(metrics)
            validation_residual_rows.extend(validation_residuals)
            test_residual_rows.extend(test_residuals)
            validation_metric = next(row for row in metrics if row["split"] == "validation")
            test_metric = next(row for row in metrics if row["split"] == "test")
            print(
                f"{cv_protocol} fold={fold} "
                f"{provenance['validation_protocol']} n={args.n_members} "
                f"val_MAE={validation_metric['MAE']:.4f} "
                f"test_MAE={test_metric['MAE']:.4f} test_R2={test_metric['R2']:.4f}"
            )

    if not metric_rows:
        raise ValueError("No folds were selected to run.")

    provenance_df = _upsert_csv(
        args.output_dir / "roost_cv_fold_provenance.csv",
        provenance_rows,
        ["cv_protocol", "fold", "validation_protocol"],
    )
    member_df = _upsert_csv(
        args.output_dir / "roost_cv_member_metadata.csv",
        member_rows,
        ["cv_protocol", "fold", "validation_protocol", "member_index", "epochs_requested", "patience"],
    )
    metrics_df = _upsert_csv(
        args.output_dir / "roost_cv_metrics.csv",
        metric_rows,
        [
            "cv_protocol",
            "fold",
            "validation_protocol",
            "split",
            "model",
            "member_start",
            "n_members",
            "epochs_requested",
            "patience",
        ],
    )
    validation_residuals_df = _upsert_csv(
        args.output_dir / "roost_cv_validation_residuals.csv",
        validation_residual_rows,
        [
            "cv_protocol",
            "fold",
            "validation_protocol",
            "model",
            "member_start",
            "n_members",
            "epochs_requested",
            "patience",
            "split",
            "record_index",
        ],
    )
    test_residuals_df = _upsert_csv(
        args.output_dir / "roost_cv_test_residuals.csv",
        test_residual_rows,
        [
            "cv_protocol",
            "fold",
            "validation_protocol",
            "model",
            "member_start",
            "n_members",
            "epochs_requested",
            "patience",
            "split",
            "record_index",
        ],
    )

    roost_split_summary = _split_summary(metrics_df)
    roost_summary = _test_summary(metrics_df)
    roost_oof_summary = _oof_summary(test_residuals_df)
    protocol_contrast = _protocol_contrast(roost_summary)
    cluster_summary = _cluster_test_summary(test_residuals_df)
    comparison_summary = _comparison_summary(roost_summary, args.physical_summary, args.rf_summary)
    comparison_oof_summary = _comparison_oof_summary(
        roost_oof_summary,
        args.physical_oof_summary,
        args.rf_oof_summary,
    )

    roost_split_summary.to_csv(args.output_dir / "roost_cv_split_summary.csv", index=False)
    roost_summary.to_csv(args.output_dir / "roost_cv_summary.csv", index=False)
    roost_oof_summary.to_csv(args.output_dir / "roost_cv_oof_summary.csv", index=False)
    protocol_contrast.to_csv(args.output_dir / "roost_cv_protocol_contrast.csv", index=False)
    cluster_summary.to_csv(args.output_dir / "roost_cv_cluster_summary.csv", index=False)
    comparison_summary.to_csv(args.output_dir / "roost_cv_vs_baselines_summary.csv", index=False)
    comparison_oof_summary.to_csv(args.output_dir / "roost_cv_vs_baselines_oof_summary.csv", index=False)
    _write_report(
        output_dir=args.output_dir,
        provenance_df=provenance_df,
        roost_split_summary=roost_split_summary,
        roost_summary=roost_summary,
        roost_oof_summary=roost_oof_summary,
        protocol_contrast=protocol_contrast,
        comparison_summary=comparison_summary,
        comparison_oof_summary=comparison_oof_summary,
        args=args,
    )

    summary = {
        "results_dir": _path_label(args.output_dir),
        "model_dir": _path_label(args.model_dir),
        "history_dir": _path_label(args.history_dir),
        "fold_rows_written": len(provenance_df),
        "member_rows_written": len(member_df),
        "metric_rows_written": len(metrics_df),
        "validation_residual_rows_written": len(validation_residuals_df),
        "test_residual_rows_written": len(test_residuals_df),
        "n_splits": args.n_splits,
        "protocols_requested": args.protocols,
        "folds_requested": args.folds,
        "n_members": args.n_members,
        "epochs": args.epochs,
        "patience": args.patience,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
