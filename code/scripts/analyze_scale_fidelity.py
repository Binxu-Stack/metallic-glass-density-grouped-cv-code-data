"""Scale fidelity of out-of-fold predictions: the calibration slope diagnostic.

A model can rank held-out densities correctly while systematically refusing their
absolute scale, hedging every prediction toward the training mean. R2 and MAE do not
separate these two failures. This script measures scale fidelity directly by
regressing pooled out-of-fold predictions on the measured densities,

    prediction = a + b * density,

and reporting the slope `b` per model under both cross-validation protocols:

  - b ~ 1   the model reproduces the real spread of densities
  - b < 1   compression: heavy alloys underpredicted, light alloys overpredicted
  - b > 1   over-expansion: the predicted spread exceeds the real one

The scientific question is whether random-split validation can see this. Under random
CV every fold contains near-neighbours of its test records, so a locally-fitted model
looks well-calibrated; only family holdout removes that support. If `b` tracks grouped
error but not random error, then standard validation is structurally blind to whether a
model has the right functional form, and grouped holdout is the only protocol that
exposes it.

`b` requires no physical knowledge -- only held-out predictions and measured values --
so it transfers to targets whose governing physics is unknown. Metallic-glass density
is the case where it can be checked against physics we already have.

Caveat carried into the manuscript: b < 1 signals broken scale fidelity from *either*
overfitting *or* a biased physical reference state. The deterministic mixing rule has
zero fitted parameters and still compresses, because pure-element B/P/Si reference
densities understate their density in metallic glasses. `b` is a scale-fidelity
diagnostic, not an overfitting detector.

Reads the canonical per-model out-of-fold residual files written by the Phase 4 CV
runners; fits nothing and trains nothing, so it cannot introduce leakage.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from models import regression_metrics

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase4_scale_fidelity"
RESULTS_DIR = PROJECT_ROOT / "results"

#: Canonical out-of-fold residual sources, one per CV runner. Every model carries both
#: `grouped_10fold` and `random_10fold` rows, so both protocols come from one read.
RESIDUAL_SOURCES: tuple[tuple[str, str], ...] = (
    ("phase4_cv_physical_baselines/physical_baseline_cv_test_residuals.csv", "physical_baseline"),
    ("phase4_cv_random_forest/rf_cv_test_residuals.csv", "random_forest"),
    ("phase4_cv_random_forest_magpie/rf_cv_test_residuals.csv", "random_forest"),
    ("phase4_cv_roost_scratch/roost_cv_test_residuals.csv", "composition_neural"),
    ("phase4_cv_roost_physics/roost_physics_cv_test_residuals.csv", "composition_neural_physics"),
    ("phase4_cv_roost_transfer/roost_transfer_cv_test_residuals.csv", "composition_neural_transfer"),
)

#: How much of the mass/volume ratio each model is *given* rather than asked to learn.
#: Density is rho = sum(x*M) / sum(x*V) -- a ratio of two composition-weighted sums.
#: Roost pools element embeddings into a weighted sum and trees average training
#: targets, so neither has division in its inductive bias; both must approximate the
#: quotient. This column orders the models by how much of that quotient is supplied.
PHYSICS_SUPPLIED: dict[str, str] = {
    "atoms_density_diagnostic": "full ratio, eta=1 (uncalibrated)",
    "packing_eta_train_median": "full ratio + fitted eta",
    "packing_eta_train_mae_opt": "full ratio + fitted eta",
    "mixing_rule": "full ratio, fixed (no fitting)",
    "Roost_trunk_atoms_density": "ratio as trunk scalar",
    "Roost_trunk_mixing_rule": "ratio as trunk scalar",
    "Roost_physics_embed": "mass + radius as element features",
    "Roost_transfer": "external corpus only",
    "random_forest_mendeleev": "mass + radius statistics in features",
    "random_forest_magpie": "elemental-property statistics in features",
    "Roost_scratch": "nothing",
}

#: Diagnostic rows excluded from cross-model correlations: `atoms_density_diagnostic`
#: is the uncalibrated eta=1 reference, reported for interpretation but absent from
#: the headline model matrix.
DIAGNOSTIC_MODELS: frozenset[str] = frozenset({"atoms_density_diagnostic"})

PROTOCOLS: tuple[str, ...] = ("grouped_10fold", "random_10fold")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--cluster-column",
        default="cluster_kmeans_36",
        help="Resampling unit for the family bootstrap (composition family, not record).",
    )
    return parser.parse_args()


def calibration_slope(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Regress predictions on measured values; return the scale-fidelity summary.

    The slope is the diagnostic. `range_ratio` is a cruder, outlier-sensitive version
    of the same quantity, kept as an independent check that the slope is not driven by
    a handful of extreme records.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 3:
        raise ValueError(f"need at least 3 records to fit a slope, got {y_true.size}")

    fit = stats.linregress(y_true, y_pred)
    true_span = float(y_true.max() - y_true.min())
    return {
        "slope_b": float(fit.slope),
        "slope_stderr": float(fit.stderr),
        "intercept": float(fit.intercept),
        "range_ratio": float((y_pred.max() - y_pred.min()) / true_span) if true_span > 0 else np.nan,
        "pred_min": float(y_pred.min()),
        "pred_max": float(y_pred.max()),
        "true_min": float(y_true.min()),
        "true_max": float(y_true.max()),
    }


def _bootstrap_slope_ci(
    frame: pd.DataFrame,
    cluster_column: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Percentile CI for the slope, resampling composition families with replacement.

    Families are the resampling unit rather than records: the claim is about
    generalization to unseen families, so the uncertainty must reflect family-level
    variability. Record-level resampling would understate it.
    """
    if cluster_column not in frame.columns:
        return {"slope_ci_lower": np.nan, "slope_ci_upper": np.nan, "n_clusters": np.nan}

    by_cluster = [group for _, group in frame.groupby(cluster_column)]
    n_clusters = len(by_cluster)
    if n_clusters < 3:
        return {"slope_ci_lower": np.nan, "slope_ci_upper": np.nan, "n_clusters": float(n_clusters)}

    slopes: list[float] = []
    for _ in range(n_bootstrap):
        picks = rng.integers(0, n_clusters, size=n_clusters)
        sample = pd.concat([by_cluster[i] for i in picks], ignore_index=True)
        # A resample can collapse onto near-constant density (one narrow family drawn
        # repeatedly); the slope is then undefined rather than merely uncertain.
        if sample["density"].nunique() < 3:
            continue
        try:
            slopes.append(calibration_slope(sample["density"], sample["prediction"])["slope_b"])
        except ValueError:
            continue

    if len(slopes) < 2:
        return {"slope_ci_lower": np.nan, "slope_ci_upper": np.nan, "n_clusters": float(n_clusters)}
    return {
        "slope_ci_lower": float(np.percentile(slopes, 2.5)),
        "slope_ci_upper": float(np.percentile(slopes, 97.5)),
        "n_clusters": float(n_clusters),
        "n_bootstrap_valid": float(len(slopes)),
    }


def load_oof_predictions() -> pd.DataFrame:
    """Concatenate pooled out-of-fold predictions from every canonical CV runner."""
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for relative_path, family in RESIDUAL_SOURCES:
        path = RESULTS_DIR / relative_path
        if not path.exists():
            missing.append(relative_path)
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["cv_protocol"].isin(PROTOCOLS)].copy()
        # Physical-baseline and neural runners also emit validation rows.
        if "split" in frame.columns:
            frame = frame[frame["split"] == "test"].copy()
        if frame.empty:
            missing.append(f"{relative_path} (no test rows for {PROTOCOLS})")
            continue

        keep = ["cv_protocol", "model", "density", "prediction", "record_index"]
        if "cluster_kmeans_36" in frame.columns:
            keep.append("cluster_kmeans_36")
        if "family_label" in frame.columns:
            keep.append("family_label")
        frame = frame[keep].copy()
        frame["model_family"] = family
        frame["source_file"] = relative_path
        frames.append(frame)

    if missing:
        raise FileNotFoundError(
            "missing or empty out-of-fold residual sources:\n  " + "\n  ".join(missing)
        )

    combined = pd.concat(frames, ignore_index=True)
    duplicated = combined.duplicated(subset=["cv_protocol", "model", "record_index"])
    if duplicated.any():
        offenders = combined.loc[duplicated, "model"].unique().tolist()
        raise ValueError(f"duplicate (protocol, model, record) rows for: {offenders}")
    return combined


def summarize(
    oof: pd.DataFrame, cluster_column: str, n_bootstrap: int, random_state: int
) -> pd.DataFrame:
    """Per-model, per-protocol scale fidelity with family-bootstrap CIs on the slope."""
    rows: list[dict[str, Any]] = []
    for (protocol, model), frame in oof.groupby(["cv_protocol", "model"]):
        rng = np.random.default_rng(random_state)
        metrics = regression_metrics(frame["density"].to_numpy(), frame["prediction"].to_numpy())
        row: dict[str, Any] = {
            "cv_protocol": protocol,
            "model": model,
            "model_family": frame["model_family"].iloc[0],
            "physics_supplied": PHYSICS_SUPPLIED.get(model, "unlabelled"),
            "n_records": int(len(frame)),
            **calibration_slope(frame["density"], frame["prediction"]),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "is_diagnostic": model in DIAGNOSTIC_MODELS,
            "source_file": frame["source_file"].iloc[0],
        }
        row.update(_bootstrap_slope_ci(frame, cluster_column, n_bootstrap, rng))
        rows.append(row)

    summary = pd.DataFrame(rows)
    ordered = ["cv_protocol", "model", "physics_supplied", "slope_b", "range_ratio", "MAE", "R2"]
    rest = [c for c in summary.columns if c not in ordered]
    return summary[ordered + rest].sort_values(
        ["cv_protocol", "slope_b"], ascending=[True, False]
    ).reset_index(drop=True)


def protocol_contrast(summary: pd.DataFrame) -> pd.DataFrame:
    """Per-model random-vs-grouped slope drop: is the compression protocol-dependent?"""
    wide = summary.pivot_table(
        index=["model", "physics_supplied"], columns="cv_protocol", values=["slope_b", "MAE"]
    )
    wide.columns = [f"{metric}_{proto.replace('_10fold', '')}" for metric, proto in wide.columns]
    wide = wide.reset_index()
    if {"slope_b_grouped", "slope_b_random"} <= set(wide.columns):
        wide["slope_drop_random_minus_grouped"] = wide["slope_b_random"] - wide["slope_b_grouped"]
    return wide.sort_values("slope_drop_random_minus_grouped", ascending=False).reset_index(drop=True)


def slope_error_correlations(summary: pd.DataFrame) -> pd.DataFrame:
    """Does the slope explain error, and does it do so under both protocols?

    The finding is the *contrast* between the two protocols, not either coefficient
    alone: with roughly ten model classes a null result is weak evidence of
    independence, but a coefficient near -1 in one protocol and near 0 in the other is
    informative about which protocol can see functional-form fidelity.
    """
    rows: list[dict[str, Any]] = []
    matrix = summary[~summary["is_diagnostic"]]
    for protocol, frame in matrix.groupby("cv_protocol"):
        for metric in ("MAE", "RMSE", "R2"):
            valid = frame[["slope_b", metric]].dropna()
            if len(valid) < 3:
                continue
            pearson_r, pearson_p = stats.pearsonr(valid["slope_b"], valid[metric])
            spearman_r, spearman_p = stats.spearmanr(valid["slope_b"], valid[metric])
            rows.append(
                {
                    "cv_protocol": protocol,
                    "metric": metric,
                    "n_models": int(len(valid)),
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_rho": float(spearman_r),
                    "spearman_p": float(spearman_p),
                }
            )
    return pd.DataFrame(rows)


def cross_protocol_correlations(summary: pd.DataFrame) -> pd.DataFrame:
    """Grouped slope against *random* error, and vice versa.

    This is the blindness test. If the grouped slope is uncorrelated with random-CV
    error, a practitioner reading only random scores has no signal about scale
    fidelity.
    """
    matrix = summary[~summary["is_diagnostic"]]
    wide = matrix.pivot_table(index="model", columns="cv_protocol", values=["slope_b", "MAE"])
    wide.columns = [f"{metric}_{proto.replace('_10fold', '')}" for metric, proto in wide.columns]
    wide = wide.dropna()

    pairs = (
        ("slope_b_grouped", "MAE_grouped", "grouped slope vs grouped error"),
        ("slope_b_grouped", "MAE_random", "grouped slope vs RANDOM error (blindness test)"),
        ("slope_b_random", "MAE_random", "random slope vs random error"),
        ("slope_b_random", "MAE_grouped", "random slope vs grouped error"),
    )
    rows: list[dict[str, Any]] = []
    for x_col, y_col, label in pairs:
        if not {x_col, y_col} <= set(wide.columns):
            continue
        pearson_r, pearson_p = stats.pearsonr(wide[x_col], wide[y_col])
        spearman_r, spearman_p = stats.spearmanr(wide[x_col], wide[y_col])
        rows.append(
            {
                "comparison": label,
                "x": x_col,
                "y": y_col,
                "n_models": int(len(wide)),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_rho": float(spearman_r),
                "spearman_p": float(spearman_p),
            }
        )
    return pd.DataFrame(rows)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if np.isnan(value):
            return "-"
        return f"{value:.4f}" if abs(value) < 0.01 else f"{value:.3f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "_No rows._"
    display = df.head(max_rows)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = [
        "| " + " | ".join(_format_value(v) for v in row) + " |" for row in display.to_numpy()
    ]
    return "\n".join([header, separator, *body])


def _build_report(
    summary: pd.DataFrame,
    contrast: pd.DataFrame,
    within: pd.DataFrame,
    across: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    ladder_cols = [
        "model", "physics_supplied", "slope_b", "slope_ci_lower", "slope_ci_upper",
        "range_ratio", "pred_min", "pred_max", "MAE",
    ]
    grouped = summary[summary["cv_protocol"] == "grouped_10fold"]
    random_cv = summary[summary["cv_protocol"] == "random_10fold"]

    def _lookup(frame: pd.DataFrame, comparison: str, field: str) -> str:
        hit = frame[frame["comparison"] == comparison]
        return _format_value(float(hit[field].iloc[0])) if not hit.empty else "n/a"

    blind = "grouped slope vs RANDOM error (blindness test)"
    return "\n".join(
        [
            "# Scale Fidelity of Out-of-Fold Predictions",
            "",
            f"Generated: {date.today().isoformat()}",
            "",
            "## Question",
            "",
            "Can random-split validation see whether a model reproduces the absolute scale",
            "of a held-out property, or only whether it ranks nearby records correctly?",
            "",
            "## Method",
            "",
            "For each model and protocol, pooled out-of-fold predictions are regressed on the",
            "measured densities as `prediction = a + b * density`. The slope `b` is the",
            "diagnostic: b ~ 1 reproduces the real spread, b < 1 compresses toward the",
            "training mean, b > 1 over-expands. Confidence intervals resample composition",
            f"families (`{args.cluster_column}`) with replacement, {args.n_bootstrap} iterations,",
            "because the claim concerns generalization to unseen families.",
            "",
            "Nothing is fitted or trained here; the script reads the canonical out-of-fold",
            "residual files, so it cannot introduce leakage.",
            "",
            "## Grouped CV: the physics-supplied ladder",
            "",
            "Ordered by slope. `physics_supplied` records how much of the mass/volume ratio",
            "each model is given rather than asked to learn.",
            "",
            _markdown_table(grouped[ladder_cols]),
            "",
            "## Random CV: the same models, near-neighbour support intact",
            "",
            _markdown_table(random_cv[ladder_cols]),
            "",
            "## Per-model protocol contrast",
            "",
            _markdown_table(contrast),
            "",
            "## Slope versus error, within each protocol",
            "",
            _markdown_table(within),
            "",
            "## Slope versus error, across protocols",
            "",
            _markdown_table(across),
            "",
            "## Interpretation",
            "",
            f"Grouped slope against grouped error: Pearson r = "
            f"{_lookup(across, 'grouped slope vs grouped error', 'pearson_r')}. Grouped slope",
            f"against random error: Pearson r = {_lookup(across, blind, 'pearson_r')} "
            f"(p = {_lookup(across, blind, 'pearson_p')}).",
            "",
            "Read the contrast between those two coefficients, not either one alone. With",
            "roughly ten model classes, a coefficient near zero is weak evidence of",
            "independence on its own; the informative result is that scale fidelity tracks",
            "error under family holdout while being far weaker against random-split error.",
            "",
            "That blindness is a property of the random-split *error*, not of the slope. The",
            "slope measured under random CV still rank-orders which models will lose scale",
            f"under family holdout (Spearman "
            f"{_lookup(across, 'random slope vs grouped error', 'spearman_rho')} against grouped",
            f"error, versus Pearson "
            f"{_lookup(across, 'random slope vs grouped error', 'pearson_r')}), but it",
            "understates the magnitude badly: random-CV slopes occupy a narrow band near 1",
            "while grouped slopes spread far lower. So computing the slope on a random split is",
            "a useful early warning that costs nothing, and is not a substitute for family",
            "holdout when the size of the degradation matters.",
            "",
            "The deterministic models (`mixing_rule`, `atoms_density_diagnostic`) return",
            "identical rows under both protocols. They fit nothing, so the protocol cannot",
            "change their predictions -- a useful check that the pooling is wired correctly.",
            "",
            "Two caveats belong with any use of these numbers. First, b < 1 indicates broken",
            "scale fidelity from *either* overfitting *or* a biased physical reference state:",
            "the deterministic mixing rule has no fitted parameters and still compresses,",
            "because pure-element B/P/Si reference densities understate those elements in",
            "metallic glasses. Second, the uncalibrated `atoms_density_diagnostic` row",
            "over-expands rather than compresses, which is why it is excluded from the",
            "cross-model correlations.",
            "",
        ]
    )


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    oof = load_oof_predictions()
    summary = summarize(oof, args.cluster_column, args.n_bootstrap, args.random_state)
    contrast = protocol_contrast(summary)
    within = slope_error_correlations(summary)
    across = cross_protocol_correlations(summary)

    summary.to_csv(args.output_dir / "scale_fidelity_summary.csv", index=False)
    contrast.to_csv(args.output_dir / "scale_fidelity_protocol_contrast.csv", index=False)
    within.to_csv(args.output_dir / "scale_fidelity_within_protocol_correlations.csv", index=False)
    across.to_csv(args.output_dir / "scale_fidelity_across_protocol_correlations.csv", index=False)
    (args.output_dir / "scale_fidelity_report.md").write_text(
        _build_report(summary, contrast, within, across, args), encoding="utf-8"
    )

    show = ["model", "physics_supplied", "slope_b", "slope_ci_lower", "slope_ci_upper", "MAE"]
    print("=== Grouped CV: physics-supplied ladder ===")
    print(summary[summary["cv_protocol"] == "grouped_10fold"][show].to_string(index=False))
    print("\n=== Random CV ===")
    print(summary[summary["cv_protocol"] == "random_10fold"][show].to_string(index=False))
    print("\n=== Slope vs error across protocols ===")
    print(across.to_string(index=False))
    print(f"\nWrote 4 CSVs and a report to {args.output_dir}")


if __name__ == "__main__":
    main()
