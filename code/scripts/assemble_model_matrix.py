"""Assemble the unified Phase 4 model matrix from completed CV outputs."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase4_unified_model_matrix"


@dataclass(frozen=True)
class SummarySource:
    name: str
    summary_path: Path
    oof_path: Path | None = None
    split_summary_path: Path | None = None
    provenance_path: Path | None = None
    default_validation_protocol: str = "none"
    diagnostic_only: bool = False
    notes: str = ""


SUMMARY_SOURCES = [
    SummarySource(
        name="physical_baselines",
        summary_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_physical_baselines"
        / "physical_baseline_cv_summary.csv",
        oof_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_physical_baselines"
        / "physical_baseline_cv_oof_summary.csv",
        provenance_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_physical_baselines"
        / "cv_fold_provenance.csv",
        notes="Leak-safe train-estimated physical baselines.",
    ),
    SummarySource(
        name="random_forest_mendeleev",
        summary_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_random_forest"
        / "rf_cv_summary.csv",
        oof_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_random_forest"
        / "rf_cv_oof_summary.csv",
        provenance_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_random_forest"
        / "rf_cv_fold_provenance.csv",
        notes="Element fractions plus local Mendeleev aggregates.",
    ),
    SummarySource(
        name="random_forest_magpie",
        summary_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_random_forest_magpie"
        / "rf_cv_summary.csv",
        oof_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_random_forest_magpie"
        / "rf_cv_oof_summary.csv",
        provenance_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_random_forest_magpie"
        / "rf_cv_fold_provenance.csv",
        notes="Magpie features from the separate .venv-magpie environment.",
    ),
    SummarySource(
        name="roost_scratch",
        summary_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_scratch"
        / "roost_cv_summary.csv",
        oof_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_scratch"
        / "roost_cv_oof_summary.csv",
        provenance_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_scratch"
        / "roost_cv_fold_provenance.csv",
        notes="Primary scratch Roost CV: random validation for random CV, group validation for grouped CV.",
    ),
    SummarySource(
        name="roost_scratch_random_val_diagnostic",
        summary_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_scratch_random_val"
        / "roost_cv_summary.csv",
        oof_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_scratch_random_val"
        / "roost_cv_oof_summary.csv",
        split_summary_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_scratch_random_val"
        / "roost_cv_split_summary.csv",
        provenance_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_scratch_random_val"
        / "roost_cv_fold_provenance.csv",
        diagnostic_only=True,
        notes="Grouped test with random validation inside grouped train pool.",
    ),
    SummarySource(
        name="roost_physics",
        summary_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_physics"
        / "roost_physics_cv_summary.csv",
        oof_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_physics"
        / "roost_physics_cv_oof_summary.csv",
        split_summary_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_physics"
        / "roost_physics_cv_split_summary.csv",
        provenance_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_physics"
        / "roost_physics_cv_fold_provenance.csv",
        notes="Roost physical embedding and physics-trunk variants.",
    ),
    SummarySource(
        name="roost_transfer",
        summary_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_transfer"
        / "roost_transfer_cv_summary.csv",
        oof_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_transfer"
        / "roost_transfer_cv_oof_summary.csv",
        split_summary_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_transfer"
        / "roost_transfer_cv_split_summary.csv",
        provenance_path=PROJECT_ROOT
        / "results"
        / "phase4_cv_roost_transfer"
        / "roost_transfer_cv_fold_provenance.csv",
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
RANDOM_PROTOCOL = "random_10fold"
GROUPED_PROTOCOL = "grouped_10fold"


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _records_column(df: pd.DataFrame) -> pd.Series:
    if "n_test_records_total" in df.columns:
        return df["n_test_records_total"]
    if "n_records_total" in df.columns:
        return df["n_records_total"]
    if "n_records" in df.columns:
        return df["n_records"]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _normalize_summary(source: SummarySource) -> pd.DataFrame:
    if not source.summary_path.exists():
        raise FileNotFoundError(source.summary_path)
    df = pd.read_csv(source.summary_path)
    output = df.copy()
    if "validation_protocol" not in output.columns:
        output["validation_protocol"] = source.default_validation_protocol
    output["validation_protocol"] = output["validation_protocol"].fillna("none")
    if "variant" not in output.columns:
        output["variant"] = output["model"]
    output["n_records_total"] = _records_column(output)
    output["source_result"] = source.name
    output["source_path"] = _path_label(source.summary_path)
    output["model_family"] = output["model"].map(MODEL_FAMILIES).fillna("unknown")
    output["display_name"] = output["model"].map(DISPLAY_NAMES).fillna(output["model"])
    output["source_notes"] = source.notes
    output["is_diagnostic_row"] = source.diagnostic_only | output["model"].isin(PRIMARY_EXCLUDED_MODELS)
    output["is_primary_row"] = ~output["is_diagnostic_row"]
    wanted = [
        "source_result",
        "source_path",
        "cv_protocol",
        "model",
        "display_name",
        "model_family",
        "variant",
        "validation_protocol",
        "n_folds",
        "n_records_total",
        "MAE_mean",
        "MAE_std",
        "MAE_min",
        "MAE_max",
        "RMSE_mean",
        "RMSE_std",
        "R2_mean",
        "R2_std",
        "is_primary_row",
        "is_diagnostic_row",
        "source_notes",
    ]
    for col in wanted:
        if col not in output.columns:
            output[col] = pd.NA
    return output[wanted]


def _normalize_oof(source: SummarySource) -> pd.DataFrame:
    if source.oof_path is None or not source.oof_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(source.oof_path)
    output = df.copy()
    if "validation_protocol" not in output.columns:
        output["validation_protocol"] = source.default_validation_protocol
    output["validation_protocol"] = output["validation_protocol"].fillna("none")
    if "variant" not in output.columns:
        output["variant"] = output["model"]
    output["source_result"] = source.name
    output["source_path"] = _path_label(source.oof_path)
    output["model_family"] = output["model"].map(MODEL_FAMILIES).fillna("unknown")
    output["display_name"] = output["model"].map(DISPLAY_NAMES).fillna(output["model"])
    output["is_diagnostic_row"] = source.diagnostic_only | output["model"].isin(PRIMARY_EXCLUDED_MODELS)
    output["is_primary_row"] = ~output["is_diagnostic_row"]
    wanted = [
        "source_result",
        "source_path",
        "cv_protocol",
        "model",
        "display_name",
        "model_family",
        "variant",
        "validation_protocol",
        "n_records",
        "MAE",
        "RMSE",
        "R2",
        "is_primary_row",
        "is_diagnostic_row",
    ]
    for col in wanted:
        if col not in output.columns:
            output[col] = pd.NA
    return output[wanted]


def _wide_metric_table(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    keys = ["model", "display_name", "model_family", "variant"]
    primary = df[df["is_primary_row"]].copy()
    rows: list[dict[str, Any]] = []
    for key_values, group in primary.groupby(keys, dropna=False):
        row = dict(zip(keys, key_values, strict=False))
        for protocol, suffix in [(RANDOM_PROTOCOL, "random"), (GROUPED_PROTOCOL, "grouped")]:
            protocol_rows = group[group["cv_protocol"] == protocol]
            if protocol_rows.empty:
                continue
            if len(protocol_rows) > 1:
                protocol_rows = protocol_rows.sort_values("validation_protocol")
            selected = protocol_rows.iloc[0]
            row[f"{suffix}_validation_protocol"] = selected.get("validation_protocol", "none")
            for col in metric_cols:
                row[f"{suffix}_{col}"] = selected.get(col, pd.NA)
        rows.append(row)
    wide = pd.DataFrame(rows)
    if wide.empty:
        return wide
    if "grouped_MAE_mean" in wide.columns and "random_MAE_mean" in wide.columns:
        wide["grouped_minus_random_MAE_mean"] = (
            wide["grouped_MAE_mean"] - wide["random_MAE_mean"]
        )
        wide["grouped_to_random_MAE_ratio"] = (
            wide["grouped_MAE_mean"] / wide["random_MAE_mean"]
        )
    if "grouped_RMSE_mean" in wide.columns and "random_RMSE_mean" in wide.columns:
        wide["grouped_minus_random_RMSE_mean"] = (
            wide["grouped_RMSE_mean"] - wide["random_RMSE_mean"]
        )
    if "grouped_R2_mean" in wide.columns and "random_R2_mean" in wide.columns:
        wide["random_minus_grouped_R2_mean"] = (
            wide["random_R2_mean"] - wide["grouped_R2_mean"]
        )
    return wide.sort_values(["grouped_MAE_mean", "random_MAE_mean", "model"]).reset_index(drop=True)


def _wide_oof_table(oof_df: pd.DataFrame) -> pd.DataFrame:
    renamed = oof_df.rename(
        columns={
            "MAE": "MAE_mean",
            "RMSE": "RMSE_mean",
            "R2": "R2_mean",
        }
    )
    return _wide_metric_table(renamed, ["n_records", "MAE_mean", "RMSE_mean", "R2_mean"])


def _load_split_summaries() -> pd.DataFrame:
    rows = []
    for source in SUMMARY_SOURCES:
        if source.split_summary_path is None or not source.split_summary_path.exists():
            continue
        df = pd.read_csv(source.split_summary_path)
        if "validation_protocol" not in df.columns:
            df["validation_protocol"] = source.default_validation_protocol
        if "variant" not in df.columns:
            df["variant"] = df["model"]
        df["source_result"] = source.name
        df["model_family"] = df["model"].map(MODEL_FAMILIES).fillna("unknown")
        df["display_name"] = df["model"].map(DISPLAY_NAMES).fillna(df["model"])
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def _validation_diagnostic(split_df: pd.DataFrame) -> pd.DataFrame:
    if split_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = [
        "source_result",
        "cv_protocol",
        "model",
        "display_name",
        "model_family",
        "variant",
        "validation_protocol",
    ]
    for key_values, group in split_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key_values, strict=False))
        for split in ("train", "validation", "test"):
            split_rows = group[group["split"] == split]
            if split_rows.empty:
                continue
            selected = split_rows.iloc[0]
            for metric in ("n_folds", "n_records_total", "MAE_mean", "MAE_std", "RMSE_mean", "R2_mean"):
                row[f"{split}_{metric}"] = selected.get(metric, pd.NA)
        if "validation_MAE_mean" in row and "test_MAE_mean" in row:
            row["test_minus_validation_MAE_mean"] = (
                row["test_MAE_mean"] - row["validation_MAE_mean"]
            )
            row["test_to_validation_MAE_ratio"] = (
                row["test_MAE_mean"] / row["validation_MAE_mean"]
            )
        rows.append(row)
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    return output.sort_values(["cv_protocol", "model_family", "model", "validation_protocol"]).reset_index(drop=True)


def _scratch_validation_protocol_contrast(all_summary: pd.DataFrame) -> pd.DataFrame:
    scratch = all_summary[
        (all_summary["model"] == "Roost_scratch")
        & (all_summary["cv_protocol"] == GROUPED_PROTOCOL)
    ].copy()
    if scratch.empty:
        return pd.DataFrame()
    cols = ["validation_protocol", "MAE_mean", "MAE_std", "RMSE_mean", "R2_mean"]
    scratch = scratch[cols].drop_duplicates("validation_protocol")
    row: dict[str, Any] = {"model": "Roost_scratch", "cv_protocol": GROUPED_PROTOCOL}
    for _, values in scratch.iterrows():
        prefix = str(values["validation_protocol"])
        for metric in ("MAE_mean", "MAE_std", "RMSE_mean", "R2_mean"):
            row[f"{prefix}_{metric}"] = values[metric]
    if "random_val_MAE_mean" in row and "group_val_MAE_mean" in row:
        row["random_val_minus_group_val_MAE_mean"] = (
            row["random_val_MAE_mean"] - row["group_val_MAE_mean"]
        )
        row["random_val_minus_group_val_R2_mean"] = (
            row.get("random_val_R2_mean", np.nan) - row.get("group_val_R2_mean", np.nan)
        )
    return pd.DataFrame([row])


def _provenance_audit() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in SUMMARY_SOURCES:
        if source.provenance_path is None or not source.provenance_path.exists():
            continue
        df = pd.read_csv(source.provenance_path)
        if "validation_protocol" not in df.columns:
            df["validation_protocol"] = source.default_validation_protocol
        if "variant" not in df.columns:
            df["variant"] = "all"
        for key_values, group in df.groupby(
            ["cv_protocol", "validation_protocol", "variant"],
            dropna=False,
        ):
            protocol, validation_protocol, variant = key_values
            row: dict[str, Any] = {
                "source_result": source.name,
                "source_path": _path_label(source.provenance_path),
                "cv_protocol": protocol,
                "validation_protocol": validation_protocol,
                "variant": variant,
                "n_folds": group["fold"].nunique() if "fold" in group.columns else len(group),
                "n_test_min": group["n_test"].min() if "n_test" in group.columns else pd.NA,
                "n_test_max": group["n_test"].max() if "n_test" in group.columns else pd.NA,
                "max_train_test_cluster_overlap": group["train_test_cluster_overlap"].max()
                if "train_test_cluster_overlap" in group.columns
                else pd.NA,
                "max_val_test_cluster_overlap": group["val_test_cluster_overlap"].max()
                if "val_test_cluster_overlap" in group.columns
                else pd.NA,
                "max_train_test_overlap": group["train_test_overlap"].max()
                if "train_test_overlap" in group.columns
                else pd.NA,
                "max_val_test_overlap": group["val_test_overlap"].max()
                if "val_test_overlap" in group.columns
                else pd.NA,
            }
            if str(protocol).startswith("grouped_"):
                train_cluster_ok = row["max_train_test_cluster_overlap"] == 0
                val_cluster_ok = (
                    pd.isna(row["max_val_test_cluster_overlap"])
                    or row["max_val_test_cluster_overlap"] == 0
                )
                train_record_ok = (
                    pd.isna(row["max_train_test_overlap"])
                    or row["max_train_test_overlap"] == 0
                )
                val_record_ok = (
                    pd.isna(row["max_val_test_overlap"])
                    or row["max_val_test_overlap"] == 0
                )
                row["grouped_holdout_safe"] = (
                    train_cluster_ok and val_cluster_ok and train_record_ok and val_record_ok
                )
            else:
                row["grouped_holdout_safe"] = pd.NA
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["source_result", "cv_protocol", "variant"]).reset_index(drop=True)


def _modnet_archive_note() -> pd.DataFrame:
    archive_dir = PROJECT_ROOT / "results" / "archive" / "2026-07-14_pre_early_stop_fix"
    sources = {
        "random_repeated_split_archive": archive_dir / "modnet_random_split_ensemble.csv",
        "group_repeated_split_archive": archive_dir / "modnet_group_split_ensemble.csv",
    }
    rows: list[dict[str, Any]] = []
    for protocol, path in sources.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        summary = df[["test_MAE", "test_RMSE", "test_R2"]].agg(["mean", "std"])
        rows.append(
            {
                "model": "MODNet",
                "protocol": protocol,
                "n_runs": len(df),
                "mean_test_MAE": summary.loc["mean", "test_MAE"],
                "std_test_MAE": summary.loc["std", "test_MAE"],
                "mean_test_RMSE": summary.loc["mean", "test_RMSE"],
                "std_test_RMSE": summary.loc["std", "test_RMSE"],
                "mean_test_R2": summary.loc["mean", "test_R2"],
                "std_test_R2": summary.loc["std", "test_R2"],
                "source_path": _path_label(path),
                "comparability_note": "Archived repeated-split result, not current 10-fold CV; do not rank in main table.",
            }
        )
    return pd.DataFrame(rows)


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    display = df[columns].head(max_rows).copy()
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(_format_value(value) for value in row) + " |"
        for row in display.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _plot_grouped_mae(main_matrix: pd.DataFrame, output_dir: Path) -> None:
    plot_df = main_matrix.sort_values("grouped_MAE_mean", ascending=True)
    fig_height = max(4.5, 0.38 * len(plot_df))
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["grouped_MAE_mean"], color="#4C78A8")
    ax.set_yticks(y_pos, plot_df["display_name"])
    ax.invert_yaxis()
    ax.set_xlabel("Grouped 10-fold MAE (g/cm3)")
    ax.set_title("Phase 4 grouped-CV model ranking")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_grouped_mae_ranking.png", dpi=200)
    plt.close(fig)


def _plot_generalization_gap(main_matrix: pd.DataFrame, output_dir: Path) -> None:
    plot_df = main_matrix.sort_values("grouped_minus_random_MAE_mean", ascending=True)
    fig_height = max(4.5, 0.38 * len(plot_df))
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["grouped_minus_random_MAE_mean"], color="#F58518")
    ax.set_yticks(y_pos, plot_df["display_name"])
    ax.invert_yaxis()
    ax.set_xlabel("Grouped MAE - random MAE (g/cm3)")
    ax.set_title("Random-to-grouped generalization gap")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_generalization_gap.png", dpi=200)
    plt.close(fig)


def _write_report(
    output_dir: Path,
    all_summary: pd.DataFrame,
    main_matrix: pd.DataFrame,
    validation_diagnostic: pd.DataFrame,
    provenance: pd.DataFrame,
    modnet_note: pd.DataFrame,
) -> None:
    grouped_safe = provenance[
        provenance["cv_protocol"].astype(str).str.startswith("grouped_")
    ]
    all_grouped_safe = bool(
        grouped_safe["grouped_holdout_safe"].dropna().astype(bool).all()
    )
    top_cols = [
        "display_name",
        "model_family",
        "grouped_MAE_mean",
        "grouped_MAE_std",
        "random_MAE_mean",
        "grouped_minus_random_MAE_mean",
        "grouped_R2_mean",
        "random_R2_mean",
        "random_minus_grouped_R2_mean",
    ]
    validation_cols = [
        "display_name",
        "cv_protocol",
        "validation_protocol",
        "validation_MAE_mean",
        "test_MAE_mean",
        "test_minus_validation_MAE_mean",
        "test_to_validation_MAE_ratio",
    ]
    lines = [
        "# Phase 4 Unified Model Matrix",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "- Primary ranking uses grouped 10-fold fold-mean test MAE.",
        "- Random 10-fold is retained as the interpolation benchmark.",
        "- Raw atomic density and archived MODNet repeated-split outputs are diagnostic/comparability rows, not primary-ranked models.",
        "- Roost scratch grouped-CV random-validation is reported as a validation-protocol diagnostic.",
        "",
        "## Generated Files",
        "",
        "- `phase4_model_matrix_all.csv`",
        "- `phase4_main_model_matrix.csv`",
        "- `phase4_oof_model_matrix.csv`",
        "- `phase4_validation_diagnostic.csv`",
        "- `phase4_scratch_validation_protocol_contrast.csv`",
        "- `phase4_provenance_audit.csv`",
        "- `phase4_modnet_archive_note.csv`",
        "- `phase4_grouped_mae_ranking.png`",
        "- `phase4_generalization_gap.png`",
        "- `phase4_unified_model_matrix_report.md`",
        "",
        "## Primary Grouped-CV Ranking",
        "",
        _markdown_table(main_matrix, top_cols, max_rows=30),
        "",
        "## Validation Diagnostics",
        "",
        _markdown_table(validation_diagnostic, validation_cols, max_rows=30),
        "",
        "## Provenance Check",
        "",
        f"- Grouped holdout safe across audited sources: `{all_grouped_safe}`.",
        "- Grouped-CV rows require zero train/test cluster overlap; validation/test cluster overlap is also checked when validation exists.",
        "",
        "## MODNet Comparability Note",
        "",
        _markdown_table(
            modnet_note,
            [
                "protocol",
                "n_runs",
                "mean_test_MAE",
                "std_test_MAE",
                "mean_test_R2",
                "comparability_note",
            ],
            max_rows=10,
        ),
        "",
        "## Immediate Interpretation",
        "",
        "- The best grouped-CV rows are physical packing and Roost physics-trunk variants; they are much closer to each other than to scratch Roost.",
        "- Transfer learning is excellent under random CV, but grouped CV remains substantially harder.",
        "- The Roost scratch random-validation diagnostic shows low validation error can coexist with much larger grouped test error.",
        "- The next manuscript step is to decide which primary rows belong in the main text table versus supplementary tables.",
        "",
        "## Source Row Counts",
        "",
        _markdown_table(
            all_summary.groupby("source_result").size().reset_index(name="n_rows"),
            ["source_result", "n_rows"],
        ),
        "",
    ]
    (output_dir / "phase4_unified_model_matrix_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary = pd.concat(
        [_normalize_summary(source) for source in SUMMARY_SOURCES],
        ignore_index=True,
        sort=False,
    )
    all_oof = pd.concat(
        [_normalize_oof(source) for source in SUMMARY_SOURCES],
        ignore_index=True,
        sort=False,
    )
    split_df = _load_split_summaries()

    main_matrix = _wide_metric_table(
        all_summary,
        ["n_folds", "n_records_total", "MAE_mean", "MAE_std", "RMSE_mean", "R2_mean"],
    )
    oof_matrix = _wide_oof_table(all_oof)
    validation_diagnostic = _validation_diagnostic(split_df)
    scratch_validation_contrast = _scratch_validation_protocol_contrast(all_summary)
    provenance = _provenance_audit()
    modnet_note = _modnet_archive_note()

    all_summary.to_csv(output_dir / "phase4_model_matrix_all.csv", index=False)
    all_oof.to_csv(output_dir / "phase4_oof_summary_all.csv", index=False)
    main_matrix.to_csv(output_dir / "phase4_main_model_matrix.csv", index=False)
    oof_matrix.to_csv(output_dir / "phase4_oof_model_matrix.csv", index=False)
    validation_diagnostic.to_csv(output_dir / "phase4_validation_diagnostic.csv", index=False)
    scratch_validation_contrast.to_csv(
        output_dir / "phase4_scratch_validation_protocol_contrast.csv",
        index=False,
    )
    provenance.to_csv(output_dir / "phase4_provenance_audit.csv", index=False)
    modnet_note.to_csv(output_dir / "phase4_modnet_archive_note.csv", index=False)

    _plot_grouped_mae(main_matrix, output_dir)
    _plot_generalization_gap(main_matrix, output_dir)
    _write_report(
        output_dir=output_dir,
        all_summary=all_summary,
        main_matrix=main_matrix,
        validation_diagnostic=validation_diagnostic,
        provenance=provenance,
        modnet_note=modnet_note,
    )

    summary = {
        "results_dir": _path_label(output_dir),
        "n_all_summary_rows": len(all_summary),
        "n_primary_models": len(main_matrix),
        "n_oof_rows": len(all_oof),
        "n_validation_diagnostic_rows": len(validation_diagnostic),
        "n_provenance_rows": len(provenance),
        "grouped_holdout_safe": bool(
            provenance[
                provenance["cv_protocol"].astype(str).str.startswith("grouped_")
            ]["grouped_holdout_safe"].dropna().astype(bool).all()
        ),
        "best_grouped_model": (
            main_matrix.iloc[0]["model"] if not main_matrix.empty else None
        ),
        "best_grouped_MAE_mean": (
            float(main_matrix.iloc[0]["grouped_MAE_mean"]) if not main_matrix.empty else None
        ),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
