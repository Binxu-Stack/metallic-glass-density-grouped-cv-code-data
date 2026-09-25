"""Recompute pooled metrics and composition-family bootstrap intervals from released OOF predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    total = np.sum((y - y.mean()) ** 2)
    slope = float(np.polyfit(y, prediction, 1)[0]) if len(np.unique(y)) >= 3 else np.nan
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": float(1 - np.sum(error**2) / total) if total > 0 else np.nan,
        "prediction_observation_slope": slope,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "recomputed")
    args = parser.parse_args()

    predictions = pd.read_csv(ROOT / "results" / "out_of_fold_predictions.csv")
    families = pd.read_csv(ROOT / "data" / "composition_families_k36.csv")
    predictions = predictions.merge(
        families[["record_index", "cluster_kmeans_36"]],
        on="record_index", validate="many_to_one",
    )
    if not (predictions["cluster_id"] == predictions["cluster_kmeans_36"]).all():
        raise ValueError("Prediction and family labels differ")

    point_rows = []
    interval_rows = []
    for (model, protocol), group in predictions.groupby(["model_id", "cv_protocol"], sort=True):
        group = group.sort_values("record_index")
        if len(group) != 380 or group["record_index"].nunique() != 380:
            raise ValueError(f"Incomplete OOF predictions: {model}, {protocol}")
        y = group["measured_density_g_cm3"].to_numpy(dtype=float)
        predicted = group["predicted_density_g_cm3"].to_numpy(dtype=float)
        cluster = group["cluster_id"].to_numpy(dtype=int)
        values = metrics(y, predicted)
        point_rows.append({"model_id": model, "cv_protocol": protocol, **values})

        cluster_ids = np.unique(cluster)
        positions = {label: np.flatnonzero(cluster == label) for label in cluster_ids}
        for seed, n_resamples, requested_metrics in (
            (20260716, 5000, ("MAE", "RMSE", "R2")),
            (42, 2000, ("prediction_observation_slope",)),
        ):
            rng = np.random.default_rng(seed)
            samples = {metric: [] for metric in requested_metrics}
            for _ in range(n_resamples):
                drawn = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
                selected = np.concatenate([positions[label] for label in drawn])
                sampled = metrics(y[selected], predicted[selected])
                for metric in requested_metrics:
                    if np.isfinite(sampled[metric]):
                        samples[metric].append(sampled[metric])
            for metric, valid in samples.items():
                interval_rows.append({
                    "model_id": model,
                    "cv_protocol": protocol,
                    "metric": metric,
                    "point_estimate": values[metric],
                    "ci_lower": float(np.percentile(valid, 2.5)),
                    "ci_upper": float(np.percentile(valid, 97.5)),
                    "n_resamples": n_resamples,
                    "n_valid_resamples": len(valid),
                    "seed": seed,
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(point_rows).to_csv(args.output_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(args.output_dir / "bootstrap_intervals.csv", index=False)
    print(args.output_dir)


if __name__ == "__main__":
    main()
