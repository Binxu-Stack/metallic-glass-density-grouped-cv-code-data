"""Run 10-fold random and grouped CV for leak-safe physical baselines."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from data_loader import load_merged_density_data
from features import (
    calculate_density_mixing_rule,
    calulate_atoms_density,
    elements_composition_features,
    estimate_packing_efficiency_from_atoms,
)
from models import regression_metrics
from train_test_split import group_kfold_indices, random_kfold_indices

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase4_cv_physical_baselines"
DEFAULT_CLUSTER_ASSIGNMENTS = PROJECT_ROOT / "data" / "composition_families_k36.csv"
BP_SI_ELEMENTS = {"B", "P", "Si"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cluster-assignments", type=Path, default=DEFAULT_CLUSTER_ASSIGNMENTS)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _load_inputs(
    cluster_assignments_path: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    records = load_merged_density_data()
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
    return records, df_comp, y, cluster_assignments, groups


def _has_bp_si(record: dict[str, Any]) -> bool:
    composition = record["composition"]
    return any(composition.get(element, 0.0) > 0.0 for element in BP_SI_ELEMENTS)


def _cluster_set(indices: list[int], cluster_assignments: pd.DataFrame) -> set[int]:
    return set(cluster_assignments.loc[indices, "cluster_kmeans_36"].astype(int))


def _fold_provenance_rows(
    protocol: str,
    folds: list[tuple[list[int], list[int]]],
    cluster_assignments: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        train_clusters = _cluster_set(train_idx, cluster_assignments)
        test_clusters = _cluster_set(test_idx, cluster_assignments)
        rows.append(
            {
                "cv_protocol": protocol,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_train_clusters": len(train_clusters),
                "n_test_clusters": len(test_clusters),
                "train_test_cluster_overlap": len(train_clusters & test_clusters),
                "test_clusters": ";".join(str(cluster) for cluster in sorted(test_clusters)),
            }
        )
    return rows


def _physical_predictions(
    mixing_rule: np.ndarray,
    atoms_density: np.ndarray,
    y: pd.Series,
    train_idx: list[int],
) -> dict[str, dict[str, Any]]:
    eta_median = estimate_packing_efficiency_from_atoms(
        atoms_density[train_idx],
        y.loc[train_idx],
        method="median",
    )
    eta_mae = estimate_packing_efficiency_from_atoms(
        atoms_density[train_idx],
        y.loc[train_idx],
        method="mae_opt",
    )
    return {
        "mixing_rule": {
            "prediction": mixing_rule,
            "eta_method": None,
            "eta": None,
        },
        "atoms_density_diagnostic": {
            "prediction": atoms_density,
            "eta_method": None,
            "eta": None,
        },
        "packing_eta_train_median": {
            "prediction": atoms_density * eta_median,
            "eta_method": "train_median",
            "eta": eta_median,
        },
        "packing_eta_train_mae_opt": {
            "prediction": atoms_density * eta_mae,
            "eta_method": "train_mae_opt",
            "eta": eta_mae,
        },
    }


def _metric_row(
    protocol: str,
    fold: int,
    split: str,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eta_method: str | None,
    eta: float | None,
) -> dict[str, Any]:
    return {
        "cv_protocol": protocol,
        "fold": fold,
        "split": split,
        "model": model,
        "eta_method": eta_method,
        "eta": eta,
        "n_records": len(y_true),
        **regression_metrics(y_true, y_pred),
    }


def _evaluate_physical_baselines(
    records: list[dict[str, Any]],
    df_comp: pd.DataFrame,
    y: pd.Series,
    cluster_assignments: pd.DataFrame,
    fold_maps: dict[str, list[tuple[list[int], list[int]]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mixing_rule = np.asarray(calculate_density_mixing_rule(df_comp), dtype=float)
    atoms_density = np.asarray(calulate_atoms_density(df_comp), dtype=float)

    metric_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for protocol, folds in fold_maps.items():
        for fold, (train_idx, test_idx) in enumerate(folds):
            model_predictions = _physical_predictions(
                mixing_rule=mixing_rule,
                atoms_density=atoms_density,
                y=y,
                train_idx=train_idx,
            )
            for model_name, model_data in model_predictions.items():
                predictions = model_data["prediction"]
                eta_method = model_data["eta_method"]
                eta = model_data["eta"]

                for split, indices in (("train", train_idx), ("test", test_idx)):
                    y_true = y.loc[indices].to_numpy(dtype=float)
                    y_pred = predictions[indices]
                    metric_rows.append(
                        _metric_row(
                            protocol=protocol,
                            fold=fold,
                            split=split,
                            model=model_name,
                            y_true=y_true,
                            y_pred=y_pred,
                            eta_method=eta_method,
                            eta=eta,
                        )
                    )

                y_test = y.loc[test_idx].to_numpy(dtype=float)
                y_test_pred = predictions[test_idx]
                for record_index, true_value, pred_value in zip(test_idx, y_test, y_test_pred):
                    cluster_row = cluster_assignments.loc[record_index]
                    residual = float(pred_value - true_value)
                    residual_rows.append(
                        {
                            "cv_protocol": protocol,
                            "fold": fold,
                            "split": "test",
                            "model": model_name,
                            "record_index": int(record_index),
                            "composition_str": records[record_index]["composition_str"],
                            "density": true_value,
                            "prediction": float(pred_value),
                            "residual_pred_minus_true": residual,
                            "abs_error": abs(residual),
                            "squared_error": residual**2,
                            "cluster_kmeans_36": int(cluster_row["cluster_kmeans_36"]),
                            "family_label": cluster_row["family_label"],
                            "has_B_P_Si": _has_bp_si(records[record_index]),
                        }
                    )

    return pd.DataFrame(metric_rows), pd.DataFrame(residual_rows)


def _test_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    test_metrics = metrics_df[metrics_df["split"] == "test"]
    return (
        test_metrics.groupby(["cv_protocol", "model"], dropna=False)
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
            eta_mean=("eta", "mean"),
            eta_std=("eta", "std"),
        )
        .reset_index()
        .sort_values(["cv_protocol", "MAE_mean", "model"])
    )


def _oof_summary(residuals_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (protocol, model), group in residuals_df.groupby(["cv_protocol", "model"], dropna=False):
        metrics = regression_metrics(group["density"].to_numpy(), group["prediction"].to_numpy())
        rows.append(
            {
                "cv_protocol": protocol,
                "model": model,
                "n_records": len(group),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["cv_protocol", "MAE", "model"]).reset_index(drop=True)


def _protocol_contrast(summary: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["MAE_mean", "MAE_std", "RMSE_mean", "R2_mean", "R2_std"]
    wide = summary.pivot(index="model", columns="cv_protocol", values=value_cols)
    wide.columns = [f"{protocol}_{metric}" for metric, protocol in wide.columns]
    wide = wide.reset_index()
    for metric in ("MAE_mean", "RMSE_mean", "R2_mean"):
        random_col = f"random_10fold_{metric}"
        grouped_col = f"grouped_10fold_{metric}"
        if random_col in wide.columns and grouped_col in wide.columns:
            wide[f"random_minus_grouped_{metric}"] = wide[random_col] - wide[grouped_col]
    return wide.sort_values("grouped_10fold_MAE_mean").reset_index(drop=True)


def _cluster_test_summary(residuals_df: pd.DataFrame) -> pd.DataFrame:
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
            has_B_P_Si_count=("has_B_P_Si", "sum"),
        )
        .reset_index()
        .sort_values(["cv_protocol", "model", "MAE"], ascending=[True, True, False])
    )


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
    summary_df: pd.DataFrame,
    oof_summary_df: pd.DataFrame,
    contrast_df: pd.DataFrame,
) -> None:
    overlap_summary = (
        provenance_df.groupby("cv_protocol")
        .agg(
            n_folds=("fold", "nunique"),
            n_test_min=("n_test", "min"),
            n_test_max=("n_test", "max"),
            n_test_clusters_min=("n_test_clusters", "min"),
            n_test_clusters_max=("n_test_clusters", "max"),
            max_train_test_cluster_overlap=("train_test_cluster_overlap", "max"),
        )
        .reset_index()
    )

    report = "\n".join(
        [
            "# Phase 4 CV Physical Baseline Report",
            "",
            f"Generated: {date.today().isoformat()}",
            "",
            "## Scope",
            "",
            "- 10-fold random CV is the interpolation benchmark.",
            "- 10-fold grouped CV uses Phase 3 k=36 cluster labels as groups and is the generalization benchmark.",
            "- Packing-efficiency `eta` is fit on each fold's training records only.",
            "- Raw atomic density is diagnostic only.",
            "",
            "## Output Files",
            "",
            "- `cv_fold_provenance.csv`",
            "- `physical_baseline_cv_metrics.csv`",
            "- `physical_baseline_cv_test_residuals.csv`",
            "- `physical_baseline_cv_summary.csv`",
            "- `physical_baseline_cv_oof_summary.csv`",
            "- `physical_baseline_cv_protocol_contrast.csv`",
            "- `physical_baseline_cv_cluster_summary.csv`",
            "- `phase4_cv_physical_baseline_report.md`",
            "",
            "## Fold Provenance Summary",
            "",
            _markdown_table(overlap_summary),
            "",
            "## Test Metric Summary",
            "",
            _markdown_table(summary_df),
            "",
            "## Pooled Out-of-Fold Test Metrics",
            "",
            "These metrics pool each record's single out-of-fold test prediction within each CV protocol.",
            "",
            _markdown_table(oof_summary_df),
            "",
            "## Random-CV Minus Grouped-CV Contrast",
            "",
            _markdown_table(contrast_df),
            "",
            "## Interpretation Notes",
            "",
            "- Grouped-CV rows should have zero train/test cluster overlap.",
            "- Random-CV rows are expected to have train/test cluster overlap and represent interpolation performance.",
            "- These cheap deterministic baselines validate the CV protocol before RF/Roost CV runs.",
            "",
        ]
    )
    (output_dir / "phase4_cv_physical_baseline_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, df_comp, y, cluster_assignments, groups = _load_inputs(args.cluster_assignments)
    random_folds = random_kfold_indices(
        df_comp,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    grouped_folds = group_kfold_indices(
        df_comp,
        y,
        groups,
        n_splits=args.n_splits,
        random_state=args.random_state,
        shuffle=False,
    )
    fold_maps = {
        f"random_{args.n_splits}fold": random_folds,
        f"grouped_{args.n_splits}fold": grouped_folds,
    }

    provenance_rows: list[dict[str, Any]] = []
    for protocol, folds in fold_maps.items():
        provenance_rows.extend(_fold_provenance_rows(protocol, folds, cluster_assignments))
    provenance_df = pd.DataFrame(provenance_rows)

    metrics_df, residuals_df = _evaluate_physical_baselines(
        records=records,
        df_comp=df_comp,
        y=y,
        cluster_assignments=cluster_assignments,
        fold_maps=fold_maps,
    )
    summary_df = _test_summary(metrics_df)
    oof_summary_df = _oof_summary(residuals_df)
    contrast_df = _protocol_contrast(summary_df)
    cluster_summary_df = _cluster_test_summary(residuals_df)

    provenance_df.to_csv(args.output_dir / "cv_fold_provenance.csv", index=False)
    metrics_df.to_csv(args.output_dir / "physical_baseline_cv_metrics.csv", index=False)
    residuals_df.to_csv(args.output_dir / "physical_baseline_cv_test_residuals.csv", index=False)
    summary_df.to_csv(args.output_dir / "physical_baseline_cv_summary.csv", index=False)
    oof_summary_df.to_csv(args.output_dir / "physical_baseline_cv_oof_summary.csv", index=False)
    contrast_df.to_csv(args.output_dir / "physical_baseline_cv_protocol_contrast.csv", index=False)
    cluster_summary_df.to_csv(args.output_dir / "physical_baseline_cv_cluster_summary.csv", index=False)
    _write_report(args.output_dir, provenance_df, summary_df, oof_summary_df, contrast_df)

    print(f"Wrote CV physical baselines to {args.output_dir.relative_to(PROJECT_ROOT)}")
    print(f"fold_rows={len(provenance_df)}")
    print(f"metric_rows={len(metrics_df)}")
    print(f"test_residual_rows={len(residuals_df)}")


if __name__ == "__main__":
    main()
