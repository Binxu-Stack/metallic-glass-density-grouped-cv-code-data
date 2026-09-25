"""Bootstrap uncertainty for Phase 4 pooled OOF metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase4_pooled_oof_uncertainty"
RANDOM_PROTOCOL = "random_10fold"
GROUPED_PROTOCOL = "grouped_10fold"


@dataclass(frozen=True)
class ResidualSource:
    name: str
    path: Path
    default_validation_protocol: str = "none"
    notes: str = ""


RESIDUAL_SOURCES = [
    ResidualSource(
        name="physical_baselines",
        path=PROJECT_ROOT
        / "results"
        / "phase4_cv_physical_baselines"
        / "physical_baseline_cv_test_residuals.csv",
        notes="Leak-safe train-estimated physical baselines.",
    ),
    ResidualSource(
        name="random_forest_mendeleev",
        path=PROJECT_ROOT / "results" / "phase4_cv_random_forest" / "rf_cv_test_residuals.csv",
        notes="Element fractions plus local Mendeleev aggregates.",
    ),
    ResidualSource(
        name="random_forest_magpie",
        path=PROJECT_ROOT
        / "results"
        / "phase4_cv_random_forest_magpie"
        / "rf_cv_test_residuals.csv",
        notes="Magpie features from the separate .venv-magpie environment.",
    ),
    ResidualSource(
        name="roost_scratch",
        path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_scratch"
        / "roost_cv_test_residuals.csv",
        notes="Primary scratch Roost CV.",
    ),
    ResidualSource(
        name="roost_physics",
        path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_physics"
        / "roost_physics_cv_test_residuals.csv",
        notes="Roost physical embedding and physics-trunk variants.",
    ),
    ResidualSource(
        name="roost_transfer",
        path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_transfer"
        / "roost_transfer_cv_test_residuals.csv",
        notes="MP-pretrained Roost fine-tuning.",
    ),
]

MODEL_FAMILIES = {
    "mixing_rule": "physical_baseline",
    "packing_eta_train_median": "physical_baseline",
    "packing_eta_train_mae_opt": "physical_baseline",
    "atoms_density_diagnostic": "physical_diagnostic",
    "random_forest_mendeleev": "random_forest",
    "random_forest_magpie": "random_forest",
    "Roost_scratch": "composition_neural",
    "Roost_physics_embed": "composition_neural_physics",
    "Roost_trunk_mixing_rule": "composition_neural_physics",
    "Roost_trunk_atoms_density": "composition_neural_physics",
    "Roost_transfer": "composition_neural_transfer",
}

DISPLAY_NAMES = {
    "mixing_rule": "Mixing rule",
    "packing_eta_train_median": "Packing eta, train median",
    "packing_eta_train_mae_opt": "Packing eta, train MAE-opt",
    "atoms_density_diagnostic": "Raw atomic density diagnostic",
    "random_forest_mendeleev": "Random Forest, Mendeleev",
    "random_forest_magpie": "Random Forest, Magpie",
    "Roost_scratch": "Roost scratch",
    "Roost_physics_embed": "Roost + physical embedding",
    "Roost_trunk_mixing_rule": "Roost trunk + mixing rule",
    "Roost_trunk_atoms_density": "Roost trunk + atomic density",
    "Roost_transfer": "Roost transfer, MP pretrain",
}

PRIMARY_EXCLUDED_MODELS = {"atoms_density_diagnostic"}
GROUP_COLS = [
    "source_result",
    "model",
    "display_name",
    "model_family",
    "variant",
]
PROTOCOL_GROUP_COLS = [
    *GROUP_COLS,
    "validation_protocol",
]


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    tss = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if tss <= 0:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum(residual**2) / tss)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _metrics_from_sums(
    n_records: float,
    sum_y: float,
    sum_y2: float,
    sum_abs_error: float,
    sum_squared_error: float,
) -> dict[str, float]:
    mae = float(sum_abs_error / n_records)
    rmse = float(np.sqrt(sum_squared_error / n_records))
    tss = float(sum_y2 - (sum_y**2) / n_records)
    if tss <= 0:
        r2 = float("nan")
    else:
        r2 = float(1.0 - sum_squared_error / tss)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _aggregate_for_bootstrap(group: pd.DataFrame, by: str) -> pd.DataFrame:
    residual = group["prediction"].to_numpy(dtype=float) - group["density"].to_numpy(dtype=float)
    work = pd.DataFrame(
        {
            by: group[by].to_numpy(),
            "n_records": 1.0,
            "sum_y": group["density"].to_numpy(dtype=float),
            "sum_y2": group["density"].to_numpy(dtype=float) ** 2,
            "sum_abs_error": np.abs(residual),
            "sum_squared_error": residual**2,
        }
    )
    return (
        work.groupby(by, sort=True, as_index=False)
        .sum(numeric_only=True)
        .reset_index(drop=True)
    )


def _metrics_from_aggregate_sample(
    aggregate: pd.DataFrame,
    sampled_positions: np.ndarray,
) -> tuple[dict[str, float], int, int]:
    values = aggregate[
        ["n_records", "sum_y", "sum_y2", "sum_abs_error", "sum_squared_error"]
    ].to_numpy(dtype=float)
    sums = values[sampled_positions].sum(axis=0)
    metrics = _metrics_from_sums(
        n_records=sums[0],
        sum_y=sums[1],
        sum_y2=sums[2],
        sum_abs_error=sums[3],
        sum_squared_error=sums[4],
    )
    return metrics, int(sums[0]), int(len(sampled_positions))


def _normalize_residuals(include_diagnostic: bool = False) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    wanted = [
        "cv_protocol",
        "fold",
        "validation_protocol",
        "model",
        "variant",
        "record_index",
        "composition_str",
        "density",
        "prediction",
        "cluster_kmeans_36",
        "family_label",
        "has_B_P_Si",
    ]
    for source in RESIDUAL_SOURCES:
        if not source.path.exists():
            raise FileNotFoundError(source.path)
        df = pd.read_csv(source.path)
        if "validation_protocol" not in df.columns:
            df["validation_protocol"] = source.default_validation_protocol
        df["validation_protocol"] = df["validation_protocol"].fillna("none")
        if "variant" not in df.columns:
            df["variant"] = df["model"]
        for col in wanted:
            if col not in df.columns:
                df[col] = pd.NA
        out = df[wanted].copy()
        out["source_result"] = source.name
        out["source_path"] = _path_label(source.path)
        out["source_notes"] = source.notes
        rows.append(out)
    residuals = pd.concat(rows, ignore_index=True, sort=False)
    residuals["model_family"] = residuals["model"].map(MODEL_FAMILIES).fillna("unknown")
    residuals["display_name"] = residuals["model"].map(DISPLAY_NAMES).fillna(residuals["model"])
    residuals["is_diagnostic_row"] = residuals["model"].isin(PRIMARY_EXCLUDED_MODELS)
    residuals["is_primary_row"] = ~residuals["is_diagnostic_row"]
    residuals["density"] = pd.to_numeric(residuals["density"], errors="raise")
    residuals["prediction"] = pd.to_numeric(residuals["prediction"], errors="raise")
    residuals["cluster_kmeans_36"] = pd.to_numeric(
        residuals["cluster_kmeans_36"], errors="raise"
    ).astype(int)
    if not include_diagnostic:
        residuals = residuals[residuals["is_primary_row"]].copy()
    return residuals.reset_index(drop=True)


def _point_metrics(residuals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key_values, group in residuals.groupby([*PROTOCOL_GROUP_COLS, "cv_protocol"], dropna=False):
        row = dict(zip([*PROTOCOL_GROUP_COLS, "cv_protocol"], key_values, strict=False))
        row["n_records"] = int(len(group))
        row["n_clusters"] = int(group["cluster_kmeans_36"].nunique())
        row.update(
            _metrics(
                group["density"].to_numpy(dtype=float),
                group["prediction"].to_numpy(dtype=float),
            )
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["cv_protocol", "model_family", "model", "variant"]
    ).reset_index(drop=True)


def _cluster_sample(group: pd.DataFrame, sampled_clusters: np.ndarray) -> pd.DataFrame:
    cluster_frames = {
        int(cluster_id): cluster_df
        for cluster_id, cluster_df in group.groupby("cluster_kmeans_36", sort=False)
    }
    return pd.concat(
        [cluster_frames[int(cluster_id)] for cluster_id in sampled_clusters],
        ignore_index=True,
        sort=False,
    )


def _record_sample(group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    indices = rng.integers(0, len(group), size=len(group))
    return group.iloc[indices].reset_index(drop=True)


def _bootstrap_protocol_metrics(
    residuals: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    bootstrap_unit: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for key_values, group in residuals.groupby([*PROTOCOL_GROUP_COLS, "cv_protocol"], dropna=False):
        key = dict(zip([*PROTOCOL_GROUP_COLS, "cv_protocol"], key_values, strict=False))
        group = group.reset_index(drop=True)
        aggregate_by_cluster = _aggregate_for_bootstrap(group, "cluster_kmeans_36")
        aggregate_by_record = _aggregate_for_bootstrap(group, "record_index")
        if bootstrap_unit == "cluster":
            aggregate = aggregate_by_cluster
            record_cluster_lookup = None
        elif bootstrap_unit == "record":
            aggregate = aggregate_by_record
            record_cluster_lookup = group.drop_duplicates("record_index").set_index(
                "record_index"
            )["cluster_kmeans_36"]
        else:
            raise ValueError(f"Unknown bootstrap unit: {bootstrap_unit}")
        n_units = len(aggregate)
        for iteration in range(n_bootstrap):
            sampled_positions = rng.integers(0, n_units, size=n_units)
            metrics, n_records_sampled, n_units_sampled = _metrics_from_aggregate_sample(
                aggregate,
                sampled_positions,
            )
            if bootstrap_unit == "record":
                sampled_record_ids = aggregate.iloc[sampled_positions]["record_index"].to_numpy()
                n_clusters_sampled = int(record_cluster_lookup.loc[sampled_record_ids].nunique())
            else:
                n_clusters_sampled = n_units_sampled
            rows.append(
                {
                    **key,
                    "bootstrap_unit": bootstrap_unit,
                    "iteration": iteration,
                    "n_records_sampled": n_records_sampled,
                    "n_clusters_sampled": n_clusters_sampled,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_gap_metrics(
    residuals: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    bootstrap_unit: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for key_values, model_group in residuals.groupby(GROUP_COLS, dropna=False):
        key = dict(zip(GROUP_COLS, key_values, strict=False))
        random_group = model_group[model_group["cv_protocol"] == RANDOM_PROTOCOL].reset_index(
            drop=True
        )
        grouped_group = model_group[
            model_group["cv_protocol"] == GROUPED_PROTOCOL
        ].reset_index(drop=True)
        if random_group.empty or grouped_group.empty:
            continue
        clusters = np.array(
            sorted(
                set(random_group["cluster_kmeans_36"].unique())
                & set(grouped_group["cluster_kmeans_36"].unique())
            ),
            dtype=int,
        )
        random_cluster_agg = _aggregate_for_bootstrap(random_group, "cluster_kmeans_36")
        grouped_cluster_agg = _aggregate_for_bootstrap(grouped_group, "cluster_kmeans_36")
        random_cluster_agg = random_cluster_agg[
            random_cluster_agg["cluster_kmeans_36"].isin(clusters)
        ].sort_values("cluster_kmeans_36").reset_index(drop=True)
        grouped_cluster_agg = grouped_cluster_agg[
            grouped_cluster_agg["cluster_kmeans_36"].isin(clusters)
        ].sort_values("cluster_kmeans_36").reset_index(drop=True)
        record_ids = np.array(
            sorted(
                set(random_group["record_index"].unique())
                & set(grouped_group["record_index"].unique())
            ),
            dtype=int,
        )
        random_record_agg = _aggregate_for_bootstrap(random_group, "record_index")
        grouped_record_agg = _aggregate_for_bootstrap(grouped_group, "record_index")
        random_record_agg = random_record_agg[
            random_record_agg["record_index"].isin(record_ids)
        ].sort_values("record_index").reset_index(drop=True)
        grouped_record_agg = grouped_record_agg[
            grouped_record_agg["record_index"].isin(record_ids)
        ].sort_values("record_index").reset_index(drop=True)
        for iteration in range(n_bootstrap):
            if bootstrap_unit == "cluster":
                sampled_positions = rng.integers(0, len(clusters), size=len(clusters))
                random_metrics, _, n_clusters_sampled = _metrics_from_aggregate_sample(
                    random_cluster_agg,
                    sampled_positions,
                )
                grouped_metrics, _, _ = _metrics_from_aggregate_sample(
                    grouped_cluster_agg,
                    sampled_positions,
                )
            elif bootstrap_unit == "record":
                # Paired by sorted record_index so the gap uses identical records for both protocols.
                sampled_positions = rng.integers(0, len(record_ids), size=len(record_ids))
                random_metrics, _, _ = _metrics_from_aggregate_sample(
                    random_record_agg,
                    sampled_positions,
                )
                grouped_metrics, _, _ = _metrics_from_aggregate_sample(
                    grouped_record_agg,
                    sampled_positions,
                )
                sampled_record_ids = record_ids[sampled_positions]
                record_clusters = random_group.drop_duplicates("record_index").set_index(
                    "record_index"
                )["cluster_kmeans_36"]
                n_clusters_sampled = int(record_clusters.loc[sampled_record_ids].nunique())
            else:
                raise ValueError(f"Unknown bootstrap unit: {bootstrap_unit}")
            rows.append(
                {
                    **key,
                    "bootstrap_unit": bootstrap_unit,
                    "iteration": iteration,
                    "n_clusters_sampled": n_clusters_sampled,
                    "random_R2": random_metrics["R2"],
                    "grouped_R2": grouped_metrics["R2"],
                    "random_minus_grouped_R2": random_metrics["R2"]
                    - grouped_metrics["R2"],
                    "random_MAE": random_metrics["MAE"],
                    "grouped_MAE": grouped_metrics["MAE"],
                    "grouped_minus_random_MAE": grouped_metrics["MAE"]
                    - random_metrics["MAE"],
                    "random_RMSE": random_metrics["RMSE"],
                    "grouped_RMSE": grouped_metrics["RMSE"],
                    "grouped_minus_random_RMSE": grouped_metrics["RMSE"]
                    - random_metrics["RMSE"],
                }
            )
    return pd.DataFrame(rows)


def _ci_summary(
    samples: pd.DataFrame,
    point_metrics: pd.DataFrame,
    gap_samples: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = ["MAE", "RMSE", "R2"]
    for key_values, group in samples.groupby(
        [*PROTOCOL_GROUP_COLS, "cv_protocol", "bootstrap_unit"],
        dropna=False,
    ):
        key = dict(zip([*PROTOCOL_GROUP_COLS, "cv_protocol", "bootstrap_unit"], key_values, strict=False))
        point_row = point_metrics[
            (point_metrics["source_result"] == key["source_result"])
            & (point_metrics["model"] == key["model"])
            & (point_metrics["variant"] == key["variant"])
            & (point_metrics["validation_protocol"] == key["validation_protocol"])
            & (point_metrics["cv_protocol"] == key["cv_protocol"])
        ].iloc[0]
        for metric in metric_cols:
            values = group[metric].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    **key,
                    "comparison": "protocol",
                    "metric": metric,
                    "point_estimate": float(point_row[metric]),
                    "bootstrap_mean": float(np.mean(values)),
                    "bootstrap_std": float(np.std(values, ddof=1)),
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "n_bootstrap": int(len(group)),
                    "n_valid_bootstrap": int(len(values)),
                    "n_records": int(point_row["n_records"]),
                    "n_clusters": int(point_row["n_clusters"]),
                }
            )
    gap_metric_cols = [
        "random_minus_grouped_R2",
        "grouped_minus_random_MAE",
        "grouped_minus_random_RMSE",
    ]
    point_wide = _wide_point_metrics(point_metrics)
    for key_values, group in gap_samples.groupby([*GROUP_COLS, "bootstrap_unit"], dropna=False):
        key = dict(zip([*GROUP_COLS, "bootstrap_unit"], key_values, strict=False))
        point_row = point_wide[
            (point_wide["source_result"] == key["source_result"])
            & (point_wide["model"] == key["model"])
            & (point_wide["variant"] == key["variant"])
        ].iloc[0]
        for metric in gap_metric_cols:
            values = group[metric].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    **key,
                    "cv_protocol": "random_vs_grouped",
                    "comparison": "gap",
                    "metric": metric,
                    "point_estimate": float(point_row[metric]),
                    "bootstrap_mean": float(np.mean(values)),
                    "bootstrap_std": float(np.std(values, ddof=1)),
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "n_bootstrap": int(len(group)),
                    "n_valid_bootstrap": int(len(values)),
                    "n_records": int(point_row["grouped_n_records"]),
                    "n_clusters": int(point_row["grouped_n_clusters"]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["comparison", "cv_protocol", "metric", "bootstrap_unit", "model"]
    ).reset_index(drop=True)


def _wide_point_metrics(point_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key_values, group in point_metrics.groupby(GROUP_COLS, dropna=False):
        row = dict(zip(GROUP_COLS, key_values, strict=False))
        for protocol, prefix in [(RANDOM_PROTOCOL, "random"), (GROUPED_PROTOCOL, "grouped")]:
            protocol_rows = group[group["cv_protocol"] == protocol]
            if protocol_rows.empty:
                continue
            selected = protocol_rows.iloc[0]
            row[f"{prefix}_validation_protocol"] = selected["validation_protocol"]
            for metric in ("n_records", "n_clusters", "MAE", "RMSE", "R2"):
                row[f"{prefix}_{metric}"] = selected[metric]
        if "random_R2" in row and "grouped_R2" in row:
            row["random_minus_grouped_R2"] = row["random_R2"] - row["grouped_R2"]
        if "random_MAE" in row and "grouped_MAE" in row:
            row["grouped_minus_random_MAE"] = row["grouped_MAE"] - row["random_MAE"]
        if "random_RMSE" in row and "grouped_RMSE" in row:
            row["grouped_minus_random_RMSE"] = row["grouped_RMSE"] - row["random_RMSE"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("grouped_R2", ascending=False).reset_index(drop=True)


def _write_report(
    output_dir: Path,
    point_wide: pd.DataFrame,
    summary: pd.DataFrame,
    n_bootstrap: int,
) -> None:
    grouped_cluster_r2 = summary[
        (summary["comparison"] == "protocol")
        & (summary["cv_protocol"] == GROUPED_PROTOCOL)
        & (summary["metric"] == "R2")
        & (summary["bootstrap_unit"] == "cluster")
    ].copy()
    grouped_cluster_r2 = grouped_cluster_r2.sort_values("point_estimate", ascending=False)
    lines = [
        "# Phase 4 Pooled OOF Uncertainty",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "- Computes pooled out-of-fold metrics from per-record test residuals.",
        "- Headline uncertainty uses composition-family / k=36 cluster bootstrap.",
        "- Record bootstrap is included as a supplementary sensitivity check.",
        f"- Bootstrap iterations per unit: {n_bootstrap}.",
        "",
        "## Headline Grouped Pooled OOF R2",
        "",
        "| Model | Family | R2 | 95% cluster-bootstrap CI | Grouped MAE | Grouped RMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in grouped_cluster_r2.iterrows():
        point_row = point_wide[point_wide["model"] == row["model"]].iloc[0]
        lines.append(
            "| "
            f"{row['display_name']} | "
            f"{row['model_family']} | "
            f"{row['point_estimate']:.4f} | "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | "
            f"{point_row['grouped_MAE']:.4f} | "
            f"{point_row['grouped_RMSE']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Physical packing baselines, physics-augmented Roost variants, and transfer learning form the high-performing group under grouped pooled OOF R2.",
            "- Scratch Roost and Magpie RF remain substantially lower under grouped validation despite strong random-CV metrics.",
            "- Use the cluster-bootstrap intervals as the main uncertainty statement because composition families are the unit of discovery-relevant generalization.",
            "",
            "## Generated Files",
            "",
            "- `pooled_oof_residuals_long.csv`",
            "- `pooled_oof_point_metrics.csv`",
            "- `pooled_oof_point_metrics_wide.csv`",
            "- `pooled_oof_bootstrap_samples.csv`",
            "- `pooled_oof_gap_bootstrap_samples.csv`",
            "- `pooled_oof_bootstrap_summary.csv`",
            "- `pooled_oof_grouped_r2_ci.png`",
            "- `phase4_pooled_oof_uncertainty_report.md`",
        ]
    )
    (output_dir / "phase4_pooled_oof_uncertainty_report.md").write_text(
        "\n".join(lines) + "\n"
    )


def _plot_grouped_r2_ci(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary[
        (summary["comparison"] == "protocol")
        & (summary["cv_protocol"] == GROUPED_PROTOCOL)
        & (summary["metric"] == "R2")
        & (summary["bootstrap_unit"] == "cluster")
    ].copy()
    plot_df = plot_df.sort_values("point_estimate", ascending=True)
    y = np.arange(len(plot_df))
    lower = plot_df["point_estimate"] - plot_df["ci_lower"]
    upper = plot_df["ci_upper"] - plot_df["point_estimate"]
    fig_height = max(4.8, 0.42 * len(plot_df))
    fig, ax = plt.subplots(figsize=(8.8, fig_height))
    ax.barh(y, plot_df["point_estimate"], color="#4C78A8", alpha=0.85)
    ax.errorbar(
        plot_df["point_estimate"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="black",
        elinewidth=1.1,
        capsize=3,
    )
    ax.set_yticks(y, plot_df["display_name"])
    ax.set_xlabel("Grouped pooled OOF R2")
    ax.set_title("Grouped pooled OOF R2 with composition-family bootstrap CI")
    ax.set_xlim(left=min(-0.05, float(plot_df["ci_lower"].min()) - 0.05), right=1.03)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "pooled_oof_grouped_r2_ci.png", dpi=220)
    plt.close(fig)


def _check_against_unified(point_wide: pd.DataFrame, tolerance: float = 1e-10) -> None:
    unified_path = (
        PROJECT_ROOT / "results" / "phase4_unified_model_matrix" / "phase4_oof_model_matrix.csv"
    )
    if not unified_path.exists():
        return
    unified = pd.read_csv(unified_path)
    merged = point_wide.merge(
        unified[
            [
                "model",
                "random_MAE_mean",
                "random_RMSE_mean",
                "random_R2_mean",
                "grouped_MAE_mean",
                "grouped_RMSE_mean",
                "grouped_R2_mean",
            ]
        ],
        on="model",
        how="left",
        validate="one_to_one",
    )
    checks = [
        ("random_MAE", "random_MAE_mean"),
        ("random_RMSE", "random_RMSE_mean"),
        ("random_R2", "random_R2_mean"),
        ("grouped_MAE", "grouped_MAE_mean"),
        ("grouped_RMSE", "grouped_RMSE_mean"),
        ("grouped_R2", "grouped_R2_mean"),
    ]
    mismatches: list[str] = []
    for left, right in checks:
        diff = (merged[left] - merged[right]).abs().max()
        if pd.notna(diff) and diff > tolerance:
            mismatches.append(f"{left} vs {right}: max diff {diff}")
    if mismatches:
        raise RuntimeError("Point metrics do not match unified OOF matrix: " + "; ".join(mismatches))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--include-diagnostic", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    residuals = _normalize_residuals(include_diagnostic=args.include_diagnostic)
    point = _point_metrics(residuals)
    point_wide = _wide_point_metrics(point)
    _check_against_unified(point_wide)

    cluster_samples = _bootstrap_protocol_metrics(
        residuals,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        bootstrap_unit="cluster",
    )
    record_samples = _bootstrap_protocol_metrics(
        residuals,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 1,
        bootstrap_unit="record",
    )
    samples = pd.concat([cluster_samples, record_samples], ignore_index=True, sort=False)

    cluster_gap_samples = _bootstrap_gap_metrics(
        residuals,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 2,
        bootstrap_unit="cluster",
    )
    record_gap_samples = _bootstrap_gap_metrics(
        residuals,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 3,
        bootstrap_unit="record",
    )
    gap_samples = pd.concat(
        [cluster_gap_samples, record_gap_samples], ignore_index=True, sort=False
    )
    summary = _ci_summary(samples, point, gap_samples)

    residuals.to_csv(output_dir / "pooled_oof_residuals_long.csv", index=False)
    point.to_csv(output_dir / "pooled_oof_point_metrics.csv", index=False)
    point_wide.to_csv(output_dir / "pooled_oof_point_metrics_wide.csv", index=False)
    samples.to_csv(output_dir / "pooled_oof_bootstrap_samples.csv", index=False)
    gap_samples.to_csv(output_dir / "pooled_oof_gap_bootstrap_samples.csv", index=False)
    summary.to_csv(output_dir / "pooled_oof_bootstrap_summary.csv", index=False)
    _plot_grouped_r2_ci(summary, output_dir)
    _write_report(output_dir, point_wide, summary, args.n_bootstrap)

    run_summary = {
        "generated": date.today().isoformat(),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "n_residual_rows": int(len(residuals)),
        "n_point_rows": int(len(point)),
        "n_bootstrap_sample_rows": int(len(samples)),
        "n_gap_bootstrap_sample_rows": int(len(gap_samples)),
        "output_dir": _path_label(output_dir),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
