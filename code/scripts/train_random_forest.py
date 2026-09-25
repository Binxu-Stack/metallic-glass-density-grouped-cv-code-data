"""Run Random Forest under the Phase 4 10-fold random/grouped CV protocol."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from data_loader import load_merged_density_data
from features import elements_composition_features, magpie_features, mendeleev_composition_features
from train_test_split import group_kfold_indices, random_kfold_indices

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase4_cv_random_forest"
DEFAULT_CLUSTER_ASSIGNMENTS = PROJECT_ROOT / "data" / "composition_families_k36.csv"
DEFAULT_PHYSICAL_SUMMARY = PROJECT_ROOT / "results" / "phase4_cv_physical_baselines" / "physical_baseline_cv_summary.csv"
DEFAULT_PHYSICAL_OOF = PROJECT_ROOT / "results" / "phase4_cv_physical_baselines" / "physical_baseline_cv_oof_summary.csv"
BP_SI_ELEMENTS = {"B", "P", "Si"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cluster-assignments", type=Path, default=DEFAULT_CLUSTER_ASSIGNMENTS)
    parser.add_argument("--physical-summary", type=Path, default=DEFAULT_PHYSICAL_SUMMARY)
    parser.add_argument("--physical-oof-summary", type=Path, default=DEFAULT_PHYSICAL_OOF)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=2000)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument(
        "--feature-set",
        choices=["mendeleev", "magpie"],
        default="mendeleev",
        help=(
            "RF feature set. `mendeleev` is the Python 3.12-compatible fallback; "
            "`magpie` is intended for the separate .venv-magpie environment."
        ),
    )
    return parser.parse_args()


def regression_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "R2": float(r2_score(y_true_arr, y_pred_arr)),
    }


def _ensemble_prediction_stats(member_preds: np.ndarray) -> dict[str, np.ndarray]:
    pred_mean = member_preds.mean(axis=0)
    pred_std = member_preds.std(axis=0, ddof=0)
    confidence = 1.0 / (1.0 + pred_std)
    return {
        "y_pred_mean": pred_mean,
        "y_pred_std": pred_std,
        "confidence": confidence,
    }


def train_predict_rf_ensemble(
    x_train: pd.DataFrame | np.ndarray,
    y_train: list[float] | np.ndarray,
    x_test: pd.DataFrame | np.ndarray,
    y_test: list[float] | np.ndarray,
    n_members: int,
    base_seed: int,
    n_estimators: int,
) -> dict[str, Any]:
    train_preds = []
    test_preds = []
    for member_idx in range(n_members):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=base_seed + member_idx,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        train_preds.append(model.predict(x_train))
        test_preds.append(model.predict(x_test))

    train_stats = _ensemble_prediction_stats(np.asarray(train_preds, dtype=float))
    test_stats = _ensemble_prediction_stats(np.asarray(test_preds, dtype=float))
    y_train_arr = np.asarray(y_train, dtype=float)
    y_test_arr = np.asarray(y_test, dtype=float)
    return {
        "train_metrics": regression_metrics(y_train_arr, train_stats["y_pred_mean"]),
        "test_metrics": regression_metrics(y_test_arr, test_stats["y_pred_mean"]),
        "y_train_pred": train_stats["y_pred_mean"],
        "y_train_pred_std": train_stats["y_pred_std"],
        "y_train_confidence": train_stats["confidence"],
        "y_test_pred": test_stats["y_pred_mean"],
        "y_test_pred_std": test_stats["y_pred_std"],
        "y_test_confidence": test_stats["confidence"],
    }


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


def _build_feature_matrix(records: list[dict[str, Any]], feature_set: str) -> pd.DataFrame:
    compositions = [record["composition"] for record in records]
    if feature_set == "magpie":
        features = magpie_features(compositions).add_prefix("magpie_")
    else:
        fraction_features = elements_composition_features(compositions).add_prefix("frac_")
        mendeleev_features = mendeleev_composition_features(compositions).add_prefix("mendeleev_")
        features = pd.concat([fraction_features, mendeleev_features], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def _impute_fold_features(
    x_features: pd.DataFrame,
    train_idx: list[int],
    test_idx: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(x_features.loc[train_idx])
    x_test = imputer.transform(x_features.loc[test_idx])
    return x_train, x_test


def _metric_row(
    protocol: str,
    fold: int,
    split: str,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_members: int,
    n_estimators: int,
    base_seed: int,
) -> dict[str, Any]:
    return {
        "cv_protocol": protocol,
        "fold": fold,
        "split": split,
        "model": model,
        "n_members": n_members,
        "n_estimators": n_estimators,
        "base_seed": base_seed,
        "n_records": len(y_true),
        **regression_metrics(y_true, y_pred),
    }


def _evaluate_rf_cv(
    records: list[dict[str, Any]],
    x_features: pd.DataFrame,
    y: pd.Series,
    cluster_assignments: pd.DataFrame,
    fold_maps: dict[str, list[tuple[list[int], list[int]]]],
    n_members: int,
    base_seed: int,
    n_estimators: int,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    model_name = f"random_forest_{feature_set}"

    for protocol, folds in fold_maps.items():
        for fold, (train_idx, test_idx) in enumerate(folds):
            x_train, x_test = _impute_fold_features(x_features, train_idx, test_idx)
            y_train = y.loc[train_idx].to_numpy(dtype=float)
            y_test = y.loc[test_idx].to_numpy(dtype=float)
            fold_base_seed = base_seed + fold * 100
            result = train_predict_rf_ensemble(
                x_train,
                y_train,
                x_test,
                y_test,
                n_members=n_members,
                base_seed=fold_base_seed,
                n_estimators=n_estimators,
            )

            metric_rows.append(
                _metric_row(
                    protocol=protocol,
                    fold=fold,
                    split="train",
                    model=model_name,
                    y_true=y_train,
                    y_pred=result["y_train_pred"],
                    n_members=n_members,
                    n_estimators=n_estimators,
                    base_seed=fold_base_seed,
                )
            )
            metric_rows.append(
                _metric_row(
                    protocol=protocol,
                    fold=fold,
                    split="test",
                    model=model_name,
                    y_true=y_test,
                    y_pred=result["y_test_pred"],
                    n_members=n_members,
                    n_estimators=n_estimators,
                    base_seed=fold_base_seed,
                )
            )

            for position, record_index in enumerate(test_idx):
                true_value = float(y_test[position])
                prediction = float(result["y_test_pred"][position])
                residual = prediction - true_value
                cluster_row = cluster_assignments.loc[record_index]
                residual_rows.append(
                    {
                        "cv_protocol": protocol,
                        "fold": fold,
                        "split": "test",
                        "model": model_name,
                        "record_index": int(record_index),
                        "composition_str": records[record_index]["composition_str"],
                        "density": true_value,
                        "prediction": prediction,
                        "prediction_std": float(result["y_test_pred_std"][position]),
                        "confidence": float(result["y_test_confidence"][position]),
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
        "random_forest_mendeleev": "composition_ml",
        "random_forest_magpie": "composition_ml",
        "mixing_rule": "physical_baseline",
        "packing_eta_train_median": "physical_baseline",
        "packing_eta_train_mae_opt": "physical_baseline",
        "atoms_density_diagnostic": "physical_diagnostic",
    }
    output = df.copy()
    output["model_family"] = output["model"].map(family_map).fillna("unknown")
    return output


def _comparison_summary(
    rf_summary: pd.DataFrame,
    physical_summary_path: Path,
) -> pd.DataFrame:
    frames = [_with_model_family(rf_summary)]
    if physical_summary_path.exists():
        frames.append(_with_model_family(pd.read_csv(physical_summary_path)))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.sort_values(["cv_protocol", "MAE_mean", "model"]).reset_index(drop=True)


def _comparison_oof_summary(
    rf_oof: pd.DataFrame,
    physical_oof_path: Path,
) -> pd.DataFrame:
    frames = [_with_model_family(rf_oof)]
    if physical_oof_path.exists():
        frames.append(_with_model_family(pd.read_csv(physical_oof_path)))
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
        "| " + " | ".join(_format_value(value) for value in row)
        + " |"
        for row in display.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _write_report(
    output_dir: Path,
    provenance_df: pd.DataFrame,
    rf_summary: pd.DataFrame,
    rf_oof_summary: pd.DataFrame,
    rf_protocol_contrast: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    comparison_oof_summary: pd.DataFrame,
    n_members: int,
    n_estimators: int,
    feature_set: str,
) -> None:
    provenance_summary = (
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
    group_comparison = comparison_summary[comparison_summary["cv_protocol"] == "grouped_10fold"]
    random_comparison = comparison_summary[comparison_summary["cv_protocol"] == "random_10fold"]

    report = "\n".join(
        [
            "# Phase 4 CV Random Forest Report",
            "",
            f"Generated: {date.today().isoformat()}",
            "",
            "## Scope",
            "",
            "- 10-fold random CV is the interpolation benchmark.",
            "- 10-fold grouped CV uses Phase 3 k=36 cluster labels as groups.",
            f"- RF is evaluated as an ensemble of {n_members} members per fold.",
            f"- Each RF member uses {n_estimators} trees.",
            f"- Feature set: `{feature_set}`.",
            "- Features are built from parsed composition dictionaries, not raw alloy formula strings.",
            "- Feature imputation is fit on each fold's training rows only.",
            "",
            "## Output Files",
            "",
            "- `rf_cv_fold_provenance.csv`",
            "- `rf_cv_metrics.csv`",
            "- `rf_cv_test_residuals.csv`",
            "- `rf_cv_summary.csv`",
            "- `rf_cv_oof_summary.csv`",
            "- `rf_cv_protocol_contrast.csv`",
            "- `rf_cv_cluster_summary.csv`",
            "- `rf_vs_physical_cv_summary.csv`",
            "- `rf_vs_physical_cv_oof_summary.csv`",
            "- `phase4_cv_random_forest_report.md`",
            "",
            "## Fold Provenance Summary",
            "",
            _markdown_table(provenance_summary),
            "",
            "## RF Fold-Mean Test Summary",
            "",
            _markdown_table(rf_summary),
            "",
            "## RF Pooled Out-of-Fold Summary",
            "",
            _markdown_table(rf_oof_summary),
            "",
            "## RF Random-CV Minus Grouped-CV Contrast",
            "",
            _markdown_table(rf_protocol_contrast),
            "",
            "## Grouped-CV Model Comparison",
            "",
            _markdown_table(
                group_comparison[
                    [
                        "model",
                        "model_family",
                        "n_folds",
                        "MAE_mean",
                        "MAE_std",
                        "RMSE_mean",
                        "R2_mean",
                        "R2_std",
                    ]
                ]
            ),
            "",
            "## Random-CV Model Comparison",
            "",
            _markdown_table(
                random_comparison[
                    [
                        "model",
                        "model_family",
                        "n_folds",
                        "MAE_mean",
                        "MAE_std",
                        "RMSE_mean",
                        "R2_mean",
                        "R2_std",
                    ]
                ]
            ),
            "",
            "## Pooled OOF Model Comparison",
            "",
            _markdown_table(comparison_oof_summary[["cv_protocol", "model", "model_family", "n_records", "MAE", "RMSE", "R2"]]),
            "",
            "## Interpretation Notes",
            "",
            "- Grouped-CV comparison is the main generalization yardstick.",
            "- Random-CV comparison is interpolation performance.",
            "- RF should be compared against train-only packing-efficiency baselines before Roost CV runs are prioritized.",
            "",
        ]
    )
    (output_dir / "phase4_cv_random_forest_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, df_comp, y, cluster_assignments, groups = _load_inputs(args.cluster_assignments)
    x_features = _build_feature_matrix(records, feature_set=args.feature_set)
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

    metrics_df, residuals_df = _evaluate_rf_cv(
        records=records,
        x_features=x_features,
        y=y,
        cluster_assignments=cluster_assignments,
        fold_maps=fold_maps,
        n_members=args.n_members,
        base_seed=args.base_seed,
        n_estimators=args.n_estimators,
        feature_set=args.feature_set,
    )
    rf_summary = _test_summary(metrics_df)
    rf_oof_summary = _oof_summary(residuals_df)
    rf_protocol_contrast = _protocol_contrast(rf_summary)
    cluster_summary = _cluster_test_summary(residuals_df)
    comparison_summary = _comparison_summary(rf_summary, args.physical_summary)
    comparison_oof_summary = _comparison_oof_summary(rf_oof_summary, args.physical_oof_summary)

    provenance_df.to_csv(args.output_dir / "rf_cv_fold_provenance.csv", index=False)
    metrics_df.to_csv(args.output_dir / "rf_cv_metrics.csv", index=False)
    residuals_df.to_csv(args.output_dir / "rf_cv_test_residuals.csv", index=False)
    rf_summary.to_csv(args.output_dir / "rf_cv_summary.csv", index=False)
    rf_oof_summary.to_csv(args.output_dir / "rf_cv_oof_summary.csv", index=False)
    rf_protocol_contrast.to_csv(args.output_dir / "rf_cv_protocol_contrast.csv", index=False)
    cluster_summary.to_csv(args.output_dir / "rf_cv_cluster_summary.csv", index=False)
    comparison_summary.to_csv(args.output_dir / "rf_vs_physical_cv_summary.csv", index=False)
    comparison_oof_summary.to_csv(args.output_dir / "rf_vs_physical_cv_oof_summary.csv", index=False)
    _write_report(
        output_dir=args.output_dir,
        provenance_df=provenance_df,
        rf_summary=rf_summary,
        rf_oof_summary=rf_oof_summary,
        rf_protocol_contrast=rf_protocol_contrast,
        comparison_summary=comparison_summary,
        comparison_oof_summary=comparison_oof_summary,
        n_members=args.n_members,
        n_estimators=args.n_estimators,
        feature_set=args.feature_set,
    )

    try:
        output_label = args.output_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        output_label = args.output_dir
    print(f"Wrote RF CV results to {output_label}")
    print(f"feature_rows={len(x_features)}")
    print(f"feature_cols={x_features.shape[1]}")
    print(f"feature_set={args.feature_set}")
    print(f"metric_rows={len(metrics_df)}")
    print(f"test_residual_rows={len(residuals_df)}")


if __name__ == "__main__":
    main()
