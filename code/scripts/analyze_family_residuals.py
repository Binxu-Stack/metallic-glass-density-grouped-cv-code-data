"""Analyze hard grouped-CV residual clusters across Phase 4 models."""

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

from models import regression_metrics

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase4_hard_grouped_residuals"
DEFAULT_ROOST_RESIDUALS = (
    PROJECT_ROOT / "results" / "phase4_cv_roost_scratch" / "roost_cv_test_residuals.csv"
)
DEFAULT_RF_RESIDUALS = (
    PROJECT_ROOT / "results" / "phase4_cv_random_forest" / "rf_cv_test_residuals.csv"
)
DEFAULT_PHYSICAL_RESIDUALS = (
    PROJECT_ROOT
    / "results"
    / "phase4_cv_physical_baselines"
    / "physical_baseline_cv_test_residuals.csv"
)

CV_PROTOCOL = "grouped_10fold"
ROOST_MODEL = "Roost_scratch"
RF_MODEL = "random_forest_mendeleev"
DIAGNOSTIC_MODEL = "atoms_density_diagnostic"
PHYSICAL_MODELS = [
    "packing_eta_train_median",
    "packing_eta_train_mae_opt",
    "mixing_rule",
]
MODEL_ORDER = [
    "packing_eta_train_median",
    "packing_eta_train_mae_opt",
    "mixing_rule",
    RF_MODEL,
    ROOST_MODEL,
    DIAGNOSTIC_MODEL,
]
MODEL_FAMILY = {
    "packing_eta_train_median": "physical_baseline",
    "packing_eta_train_mae_opt": "physical_baseline",
    "mixing_rule": "physical_baseline",
    RF_MODEL: "composition_ml",
    ROOST_MODEL: "composition_neural",
    DIAGNOSTIC_MODEL: "physical_diagnostic",
}
RESIDUAL_COLUMNS = [
    "cv_protocol",
    "fold",
    "model",
    "record_index",
    "composition_str",
    "density",
    "prediction",
    "residual_pred_minus_true",
    "abs_error",
    "squared_error",
    "cluster_kmeans_36",
    "family_label",
    "has_B_P_Si",
]
OPTIONAL_RESIDUAL_COLUMNS = ["validation_protocol", "prediction_std", "confidence"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--roost-residuals", type=Path, default=DEFAULT_ROOST_RESIDUALS)
    parser.add_argument("--rf-residuals", type=Path, default=DEFAULT_RF_RESIDUALS)
    parser.add_argument("--physical-residuals", type=Path, default=DEFAULT_PHYSICAL_RESIDUALS)
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


def _load_residuals(path: Path, source: str) -> pd.DataFrame:
    available_columns = set(pd.read_csv(path, nrows=0).columns)
    usecols = [
        column
        for column in [*RESIDUAL_COLUMNS, *OPTIONAL_RESIDUAL_COLUMNS]
        if column in available_columns
    ]
    df = pd.read_csv(path, usecols=usecols)
    if "validation_protocol" not in df.columns:
        df["validation_protocol"] = ""
    if "prediction_std" not in df.columns:
        df["prediction_std"] = np.nan
    missing = set(RESIDUAL_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df[df["cv_protocol"] == CV_PROTOCOL].copy()
    df["source_file"] = source
    df["model_family"] = df["model"].map(MODEL_FAMILY).fillna("unknown")
    return df


def _combined_residuals(args: argparse.Namespace) -> pd.DataFrame:
    frames = [
        _load_residuals(args.roost_residuals, "roost"),
        _load_residuals(args.rf_residuals, "rf"),
        _load_residuals(args.physical_residuals, "physical"),
    ]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    present_models = set(combined["model"])
    expected_models = set(MODEL_ORDER)
    missing_models = expected_models.difference(present_models)
    if missing_models:
        raise ValueError(f"Missing expected models in residual files: {sorted(missing_models)}")

    record_counts = combined.groupby("model")["record_index"].nunique()
    bad_counts = record_counts[record_counts != 380]
    if not bad_counts.empty:
        raise ValueError(f"Grouped residuals should contain 380 records per model: {bad_counts.to_dict()}")

    return combined


def _rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(values.to_numpy(dtype=float))))


def _model_summary(residuals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        model_rows = residuals[residuals["model"] == model]
        metrics = regression_metrics(model_rows["density"], model_rows["prediction"])
        rows.append(
            {
                "cv_protocol": CV_PROTOCOL,
                "model": model,
                "model_family": MODEL_FAMILY[model],
                "n_records": len(model_rows),
                **metrics,
                "mean_residual": float(model_rows["residual_pred_minus_true"].mean()),
                "median_abs_error": float(model_rows["abs_error"].median()),
                "max_abs_error": float(model_rows["abs_error"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["MAE", "model"]).reset_index(drop=True)


def _cluster_model_metrics(residuals: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        residuals.groupby(
            ["cluster_kmeans_36", "family_label", "fold", "model", "model_family"],
            dropna=False,
        )
        .agg(
            n_records=("record_index", "size"),
            MAE=("abs_error", "mean"),
            RMSE=("squared_error", _rmse),
            mean_residual=("residual_pred_minus_true", "mean"),
            median_abs_error=("abs_error", "median"),
            max_abs_error=("abs_error", "max"),
            mean_density=("density", "mean"),
            mean_prediction=("prediction", "mean"),
            has_B_P_Si_count=("has_B_P_Si", "sum"),
        )
        .reset_index()
    )
    roost_rank = (
        metrics[metrics["model"] == ROOST_MODEL]
        .sort_values(["MAE", "cluster_kmeans_36"], ascending=[False, True])
        .assign(roost_hard_rank=lambda df: range(1, len(df) + 1))
        [["cluster_kmeans_36", "roost_hard_rank"]]
    )
    metrics = metrics.merge(roost_rank, on="cluster_kmeans_36", how="left")
    return metrics.sort_values(["roost_hard_rank", "model"]).reset_index(drop=True)


def _hard_cluster_tables(
    cluster_metrics: pd.DataFrame,
    residuals: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hard_clusters = (
        cluster_metrics[cluster_metrics["model"] == ROOST_MODEL]
        .sort_values(["MAE", "cluster_kmeans_36"], ascending=[False, True])
        .head(top_n)
        [["cluster_kmeans_36", "family_label", "fold", "n_records", "MAE", "mean_density"]]
        .rename(columns={"MAE": "roost_MAE"})
    )
    hard_ids = hard_clusters["cluster_kmeans_36"].tolist()
    hard_long = cluster_metrics[cluster_metrics["cluster_kmeans_36"].isin(hard_ids)].copy()
    model_order = {model: position for position, model in enumerate(MODEL_ORDER)}
    hard_long["_model_order"] = hard_long["model"].map(model_order)
    hard_long = (
        hard_long.sort_values(["roost_hard_rank", "_model_order"])
        .drop(columns=["_model_order"])
        .reset_index(drop=True)
    )

    mae_wide = (
        hard_long[
            [
                "roost_hard_rank",
                "cluster_kmeans_36",
                "family_label",
                "fold",
                "n_records",
                "model",
                "MAE",
            ]
        ]
        .pivot(
            index=["roost_hard_rank", "cluster_kmeans_36", "family_label", "fold", "n_records"],
            columns="model",
            values="MAE",
        )
        .reset_index()
    )
    mae_wide.columns.name = None

    delta_rows: list[dict[str, Any]] = []
    for _, row in mae_wide.iterrows():
        physical_values = {
            model: float(row[model])
            for model in PHYSICAL_MODELS
            if model in row and not pd.isna(row[model])
        }
        best_physical_model = min(physical_values, key=physical_values.get)
        best_physical_mae = physical_values[best_physical_model]
        candidate_models = [*PHYSICAL_MODELS, RF_MODEL]
        candidate_values = {
            model: float(row[model])
            for model in candidate_models
            if model in row and not pd.isna(row[model])
        }
        best_non_roost_model = min(candidate_values, key=candidate_values.get)
        best_non_roost_mae = candidate_values[best_non_roost_model]
        roost_mae = float(row[ROOST_MODEL])
        delta_rows.append(
            {
                "roost_hard_rank": int(row["roost_hard_rank"]),
                "cluster_kmeans_36": int(row["cluster_kmeans_36"]),
                "family_label": row["family_label"],
                "fold": int(row["fold"]),
                "n_records": int(row["n_records"]),
                "roost_MAE": roost_mae,
                "random_forest_mendeleev_MAE": float(row[RF_MODEL]),
                "packing_eta_train_median_MAE": float(row["packing_eta_train_median"]),
                "mixing_rule_MAE": float(row["mixing_rule"]),
                "best_physical_model": best_physical_model,
                "best_physical_MAE": best_physical_mae,
                "best_non_roost_model": best_non_roost_model,
                "best_non_roost_MAE": best_non_roost_mae,
                "delta_roost_minus_rf": roost_mae - float(row[RF_MODEL]),
                "delta_roost_minus_packing_median": roost_mae - float(row["packing_eta_train_median"]),
                "delta_roost_minus_best_physical": roost_mae - best_physical_mae,
                "delta_roost_minus_best_non_roost": roost_mae - best_non_roost_mae,
            }
        )
    deltas = pd.DataFrame(delta_rows)

    hard_records = residuals[residuals["cluster_kmeans_36"].isin(hard_ids)].copy()
    hard_records["_model_order"] = hard_records["model"].map(model_order)
    hard_records = (
        hard_records.sort_values(["cluster_kmeans_36", "record_index", "_model_order"])
        .drop(columns=["_model_order"])
        .reset_index(drop=True)
    )

    return hard_long, mae_wide, deltas, hard_records


def _record_level_wide(hard_records: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "cluster_kmeans_36",
        "family_label",
        "fold",
        "record_index",
        "composition_str",
        "density",
        "has_B_P_Si",
    ]
    wide = hard_records[base_cols].drop_duplicates().reset_index(drop=True)
    for model in MODEL_ORDER:
        model_rows = hard_records[hard_records["model"] == model][
            [
                *base_cols,
                "prediction",
                "residual_pred_minus_true",
                "abs_error",
            ]
        ].rename(
            columns={
                "prediction": f"{model}_prediction",
                "residual_pred_minus_true": f"{model}_residual",
                "abs_error": f"{model}_abs_error",
            }
        )
        wide = wide.merge(model_rows, on=base_cols, how="left", validate="one_to_one")
    return wide.sort_values(["cluster_kmeans_36", "record_index"]).reset_index(drop=True)


def _bp_si_summary(residuals: pd.DataFrame) -> pd.DataFrame:
    return (
        residuals.groupby(["has_B_P_Si", "model", "model_family"], dropna=False, observed=True)
        .agg(
            n_records=("record_index", "size"),
            MAE=("abs_error", "mean"),
            RMSE=("squared_error", _rmse),
            mean_residual=("residual_pred_minus_true", "mean"),
            max_abs_error=("abs_error", "max"),
        )
        .reset_index()
        .sort_values(["has_B_P_Si", "MAE", "model"])
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
    model_summary: pd.DataFrame,
    hard_wide: pd.DataFrame,
    deltas: pd.DataFrame,
    bp_si_summary: pd.DataFrame,
    top_n: int,
) -> None:
    hard_display_cols = [
        "roost_hard_rank",
        "cluster_kmeans_36",
        "family_label",
        "fold",
        "n_records",
        "packing_eta_train_median",
        "mixing_rule",
        RF_MODEL,
        ROOST_MODEL,
    ]
    delta_display_cols = [
        "roost_hard_rank",
        "family_label",
        "roost_MAE",
        "best_non_roost_model",
        "best_non_roost_MAE",
        "delta_roost_minus_best_non_roost",
    ]
    report = "\n".join(
        [
            "# Phase 4 Hard Grouped-CV Residual Analysis",
            "",
            f"Generated: {date.today().isoformat()}",
            "",
            "## Scope",
            "",
            "- Uses grouped_10fold test residuals only.",
            "- Ranks hard clusters by Roost scratch cluster MAE.",
            "- Compares Roost to RF and leak-safe physical baselines on the same held-out records.",
            "",
            "## Output Files",
            "",
            "- `grouped_model_summary.csv`",
            "- `grouped_cluster_model_metrics.csv`",
            "- `hard_roost_cluster_model_metrics.csv`",
            "- `hard_roost_cluster_mae_wide.csv`",
            "- `hard_roost_cluster_deltas.csv`",
            "- `hard_roost_record_predictions.csv`",
            "- `bp_si_grouped_summary.csv`",
            "- `phase4_hard_grouped_residual_report.md`",
            "",
            "## Grouped-CV Model Summary",
            "",
            _markdown_table(
                model_summary[
                    [
                        "model",
                        "model_family",
                        "n_records",
                        "MAE",
                        "RMSE",
                        "R2",
                        "mean_residual",
                        "max_abs_error",
                    ]
                ]
            ),
            "",
            f"## Top {top_n} Roost-Hard Clusters",
            "",
            _markdown_table(hard_wide[hard_display_cols], max_rows=top_n),
            "",
            "## Roost Minus Best Non-Roost Baseline",
            "",
            _markdown_table(deltas[delta_display_cols], max_rows=top_n),
            "",
            "## B/P/Si Diagnostic Summary",
            "",
            _markdown_table(bp_si_summary, max_rows=20),
            "",
            "## Interpretation Notes",
            "",
            "- Positive Roost-minus-baseline deltas mean Roost is worse on that held-out cluster.",
            "- Very hard Roost clusters should be inspected before spending compute on larger Roost ensembles.",
            "- Physical baselines provide a useful check for whether the error is a neural-model failure or a broader composition-only extrapolation issue.",
            "",
        ]
    )
    (output_dir / "phase4_hard_grouped_residual_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    residuals = _combined_residuals(args)
    model_summary = _model_summary(residuals)
    cluster_metrics = _cluster_model_metrics(residuals)
    hard_long, hard_wide, deltas, hard_records = _hard_cluster_tables(
        cluster_metrics=cluster_metrics,
        residuals=residuals,
        top_n=args.top_n,
    )
    hard_record_wide = _record_level_wide(hard_records)
    bp_si_summary = _bp_si_summary(residuals)

    residuals.to_csv(args.output_dir / "grouped_residuals_long.csv", index=False)
    model_summary.to_csv(args.output_dir / "grouped_model_summary.csv", index=False)
    cluster_metrics.to_csv(args.output_dir / "grouped_cluster_model_metrics.csv", index=False)
    hard_long.to_csv(args.output_dir / "hard_roost_cluster_model_metrics.csv", index=False)
    hard_wide.to_csv(args.output_dir / "hard_roost_cluster_mae_wide.csv", index=False)
    deltas.to_csv(args.output_dir / "hard_roost_cluster_deltas.csv", index=False)
    hard_record_wide.to_csv(args.output_dir / "hard_roost_record_predictions.csv", index=False)
    bp_si_summary.to_csv(args.output_dir / "bp_si_grouped_summary.csv", index=False)
    _write_report(
        output_dir=args.output_dir,
        model_summary=model_summary,
        hard_wide=hard_wide,
        deltas=deltas,
        bp_si_summary=bp_si_summary,
        top_n=args.top_n,
    )

    summary = {
        "output_dir": str(args.output_dir.resolve().relative_to(PROJECT_ROOT)),
        "cv_protocol": CV_PROTOCOL,
        "n_models": int(residuals["model"].nunique()),
        "n_records_per_model": 380,
        "n_cluster_model_rows": len(cluster_metrics),
        "top_n": args.top_n,
        "hardest_cluster": hard_wide.iloc[0]["family_label"],
        "hardest_cluster_roost_mae": float(hard_wide.iloc[0][ROOST_MODEL]),
    }
    (args.output_dir / "run_summary.json").write_text(
        pd.Series(summary).to_json(indent=2),
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
