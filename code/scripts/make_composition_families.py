#!/usr/bin/env python3
"""Generate the Phase 3 composition-cluster analysis package."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from data_loader import load_merged_density_data
from features import elements_composition_features

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "cluster_analysis"
PRIMARY_K = 36
RANDOM_STATE = 42
K_MIN = 2
K_MAX = 100
FAMILY_LABEL_MAX_ELEMENTS = 4
FAMILY_LABEL_MIN_FRACTION = 0.05
TOP_ELEMENTS_MAX_ELEMENTS = 5
TOP_ELEMENTS_MIN_FRACTION = 0.01


def _format_top_elements(
    means: pd.Series,
    *,
    max_elements: int = TOP_ELEMENTS_MAX_ELEMENTS,
    min_fraction: float = TOP_ELEMENTS_MIN_FRACTION,
) -> str:
    parts = []
    for element, fraction in means.sort_values(ascending=False).head(max_elements).items():
        if float(fraction) >= min_fraction:
            parts.append(f"{element}:{float(fraction):.3f}")
    return "; ".join(parts)


def _family_label(
    means: pd.Series,
    *,
    max_elements: int = FAMILY_LABEL_MAX_ELEMENTS,
    min_fraction: float = FAMILY_LABEL_MIN_FRACTION,
) -> str:
    elements = [
        str(element)
        for element, fraction in means.sort_values(ascending=False).head(max_elements).items()
        if float(fraction) >= min_fraction
    ]
    if not elements:
        elements = [str(means.sort_values(ascending=False).index[0])]
    return "-".join(elements)


def _load_dataset() -> tuple[list[dict[str, Any]], pd.DataFrame, pd.Series]:
    records = load_merged_density_data()
    compositions = [record["composition"] for record in records]
    composition_features = elements_composition_features(compositions).fillna(0.0)
    density = pd.Series(
        [record["density"] for record in records],
        name="density",
        index=composition_features.index,
    )
    return records, composition_features, density


def _kmeans_sweep(
    x: np.ndarray,
    linkage_matrix: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans_labels = kmeans.fit_predict(x)
        rows.append(
            {
                "method": "kmeans",
                "k": k,
                "inertia": float(kmeans.inertia_),
                "silhouette": float(silhouette_score(x, kmeans_labels)),
                "davies_bouldin": float(davies_bouldin_score(x, kmeans_labels)),
                "calinski_harabasz": float(calinski_harabasz_score(x, kmeans_labels)),
            }
        )

        ward_labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
        rows.append(
            {
                "method": "ward",
                "k": k,
                "inertia": np.nan,
                "silhouette": float(silhouette_score(x, ward_labels)),
                "davies_bouldin": float(davies_bouldin_score(x, ward_labels)),
                "calinski_harabasz": float(calinski_harabasz_score(x, ward_labels)),
            }
        )
    return pd.DataFrame(rows)


def _hdbscan_sensitivity(x: np.ndarray) -> pd.DataFrame:
    try:
        import hdbscan
    except ImportError:
        return pd.DataFrame(
            [
                {
                    "min_cluster_size": np.nan,
                    "min_samples": np.nan,
                    "n_clusters": np.nan,
                    "n_noise": np.nan,
                    "noise_fraction": np.nan,
                    "silhouette_non_noise": np.nan,
                    "status": "hdbscan_not_available",
                }
            ]
        )

    rows: list[dict[str, float | int | str | None]] = []
    settings = [
        (3, None),
        (5, None),
        (8, None),
        (10, None),
        (5, 1),
        (5, 3),
    ]
    for min_cluster_size, min_samples in settings:
        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        ).fit_predict(x)
        non_noise = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(~non_noise))
        silhouette = np.nan
        if n_clusters > 1 and int(np.sum(non_noise)) > n_clusters:
            silhouette = float(silhouette_score(x[non_noise], labels[non_noise]))
        rows.append(
            {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_fraction": float(n_noise / len(labels)),
                "silhouette_non_noise": silhouette,
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def _cluster_summary(
    composition_features: pd.DataFrame,
    density: pd.Series,
    records: list[dict[str, Any]],
    labels: np.ndarray,
    centroids: np.ndarray,
    sample_centroid_distances: np.ndarray,
) -> tuple[pd.DataFrame, dict[int, str]]:
    df = composition_features.copy()
    df["density"] = density
    df["cluster"] = labels
    df["distance_to_centroid"] = sample_centroid_distances

    family_labels: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    feature_cols = list(composition_features.columns)

    for cluster_id, group in df.groupby("cluster", sort=True):
        feature_group = group[feature_cols]
        means = feature_group.mean()
        family = _family_label(means)
        family_labels[int(cluster_id)] = family

        centroid = centroids[int(cluster_id)]
        distances_to_centroid = group["distance_to_centroid"].to_numpy(dtype=float)
        nearest_row_index = int(
            group.index[np.argmin(cdist(feature_group.to_numpy(dtype=float), centroid.reshape(1, -1)).ravel())]
        )
        rows.append(
            {
                "cluster": int(cluster_id),
                "family_label": family,
                "size": int(len(group)),
                "density_mean": float(group["density"].mean()),
                "density_std": float(group["density"].std(ddof=1)),
                "density_min": float(group["density"].min()),
                "density_max": float(group["density"].max()),
                "density_iqr": float(group["density"].quantile(0.75) - group["density"].quantile(0.25)),
                "within_centroid_distance_mean": float(np.mean(distances_to_centroid)),
                "within_centroid_distance_median": float(np.median(distances_to_centroid)),
                "within_centroid_distance_max": float(np.max(distances_to_centroid)),
                "top_elements": _format_top_elements(means),
                "representative_record_index": nearest_row_index,
                "representative_composition": records[nearest_row_index].get("composition_str", ""),
                "representative_density": float(records[nearest_row_index]["density"]),
            }
        )

    return pd.DataFrame(rows), family_labels


def _cluster_assignments(
    records: list[dict[str, Any]],
    density: pd.Series,
    labels: np.ndarray,
    family_labels: dict[int, str],
    sample_centroid_distances: np.ndarray,
    ward_labels: np.ndarray,
    hdbscan_labels: np.ndarray | None,
    pca_coords: np.ndarray,
    tsne_coords: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        cluster = int(labels[idx])
        row = {
            "record_index": idx,
            "composition_str": record.get("composition_str", ""),
            "density": float(density.iloc[idx]),
            "cluster_kmeans_36": cluster,
            "family_label": family_labels[cluster],
            "distance_to_kmeans_centroid": float(sample_centroid_distances[idx]),
            "cluster_ward_k36": int(ward_labels[idx]),
            "pca_1": float(pca_coords[idx, 0]),
            "pca_2": float(pca_coords[idx, 1]),
            "tsne_1": float(tsne_coords[idx, 0]),
            "tsne_2": float(tsne_coords[idx, 1]),
        }
        if "material_id" in record:
            row["material_id"] = record["material_id"]
        if hdbscan_labels is not None:
            row["cluster_hdbscan_mcs5"] = int(hdbscan_labels[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def _distance_metrics(
    labels: np.ndarray,
    sample_centroid_distances: np.ndarray,
    centroids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    centroid_distance_matrix = squareform(pdist(centroids, metric="euclidean"))
    matrix_df = pd.DataFrame(
        centroid_distance_matrix,
        index=[f"cluster_{i}" for i in range(len(centroids))],
        columns=[f"cluster_{i}" for i in range(len(centroids))],
    )

    inter_distances = centroid_distance_matrix[np.triu_indices_from(centroid_distance_matrix, k=1)]
    rows: list[dict[str, Any]] = [
        {
            "scope": "global",
            "cluster": np.nan,
            "metric": "sample_to_centroid_distance_mean",
            "value": float(np.mean(sample_centroid_distances)),
            "other_cluster": np.nan,
        },
        {
            "scope": "global",
            "cluster": np.nan,
            "metric": "sample_to_centroid_distance_median",
            "value": float(np.median(sample_centroid_distances)),
            "other_cluster": np.nan,
        },
        {
            "scope": "global",
            "cluster": np.nan,
            "metric": "inter_centroid_distance_mean",
            "value": float(np.mean(inter_distances)),
            "other_cluster": np.nan,
        },
        {
            "scope": "global",
            "cluster": np.nan,
            "metric": "inter_centroid_distance_median",
            "value": float(np.median(inter_distances)),
            "other_cluster": np.nan,
        },
        {
            "scope": "global",
            "cluster": np.nan,
            "metric": "inter_to_intra_mean_ratio",
            "value": float(np.mean(inter_distances) / np.mean(sample_centroid_distances)),
            "other_cluster": np.nan,
        },
    ]

    for cluster_id in range(len(centroids)):
        mask = labels == cluster_id
        distances = sample_centroid_distances[mask]
        nearest_distances = centroid_distance_matrix[cluster_id].copy()
        nearest_distances[cluster_id] = np.inf
        nearest_cluster = int(np.argmin(nearest_distances))
        rows.extend(
            [
                {
                    "scope": "cluster",
                    "cluster": cluster_id,
                    "metric": "size",
                    "value": int(np.sum(mask)),
                    "other_cluster": np.nan,
                },
                {
                    "scope": "cluster",
                    "cluster": cluster_id,
                    "metric": "sample_to_centroid_distance_mean",
                    "value": float(np.mean(distances)),
                    "other_cluster": np.nan,
                },
                {
                    "scope": "cluster",
                    "cluster": cluster_id,
                    "metric": "sample_to_centroid_distance_max",
                    "value": float(np.max(distances)),
                    "other_cluster": np.nan,
                },
                {
                    "scope": "cluster",
                    "cluster": cluster_id,
                    "metric": "nearest_centroid_distance",
                    "value": float(nearest_distances[nearest_cluster]),
                    "other_cluster": nearest_cluster,
                },
            ]
        )
    return pd.DataFrame(rows), matrix_df


def _plot_k_sweep(k_sweep: pd.DataFrame, output_path: Path, primary_k: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("silhouette", "Silhouette higher is better"),
        ("davies_bouldin", "Davies-Bouldin lower is better"),
        ("calinski_harabasz", "Calinski-Harabasz higher is better"),
        ("inertia", "K-means inertia lower is tighter"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for method, group in k_sweep.groupby("method"):
            if metric == "inertia" and method != "kmeans":
                continue
            ax.plot(group["k"], group[metric], label=method, linewidth=1.5)
        ax.axvline(primary_k, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("k")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_embedding(
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="tab20",
        s=34,
        alpha=0.82,
        edgecolor="none",
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("K-means cluster")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _preserve_original_figure(path: Path, backup_path: Path) -> None:
    """Keep the first generated unlabeled figure as a stable visual fallback."""
    if path.exists() and not backup_path.exists():
        shutil.copy2(path, backup_path)


def _plot_tsne_with_family_labels(
    coords: np.ndarray,
    labels: np.ndarray,
    cluster_summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    min_size: int = 1,
) -> None:
    """Plot t-SNE clusters with family labels placed near cluster centers."""
    summary = cluster_summary.set_index("cluster")
    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="tab20",
        s=34,
        alpha=0.78,
        edgecolor="none",
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("K-means cluster")

    plot_center = coords.mean(axis=0)
    for cluster_id in sorted(np.unique(labels)):
        row = summary.loc[int(cluster_id)]
        if int(row["size"]) < min_size:
            continue

        cluster_coords = coords[labels == cluster_id]
        center = np.median(cluster_coords, axis=0)
        direction = center - plot_center
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            direction = np.array([1.0, 0.0])
        else:
            direction = direction / norm
        text_xy = center + 0.7 * direction

        label = f"{int(cluster_id)} {row['family_label']}\nn={int(row['size'])}"
        ax.annotate(
            label,
            xy=center,
            xytext=text_xy,
            fontsize=8,
            ha="center",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": "0.35",
                "lw": 0.7,
                "alpha": 0.65,
            },
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "white",
                "ec": "0.65",
                "alpha": 0.82,
                "lw": 0.6,
            },
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_sizes(cluster_summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = cluster_summary.sort_values("size", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(plot_df["cluster"].astype(str), plot_df["size"], color="#4C78A8")
    ax.set_title("K-means k=36 cluster sizes")
    ax.set_xlabel("Cluster sorted by size")
    ax.set_ylabel("Number of records")
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_centroid_heatmap(centroid_distances: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(centroid_distances.to_numpy(dtype=float), cmap="viridis")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Euclidean distance")
    ax.set_title("K-means centroid distance matrix")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cluster")
    ax.set_xticks(range(len(centroid_distances)))
    ax.set_yticks(range(len(centroid_distances)))
    ax.set_xticklabels(range(len(centroid_distances)), fontsize=6, rotation=90)
    ax.set_yticklabels(range(len(centroid_distances)), fontsize=6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_dendrogram(
    linkage_matrix: np.ndarray,
    output_path: Path,
    *,
    truncate_p: int = 40,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=truncate_p,
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True,
        color_threshold=0.7 * float(np.max(linkage_matrix[:, 2])),
        ax=ax,
    )
    ax.set_title(f"Ward hierarchical dendrogram, last {truncate_p} merges")
    ax.set_xlabel("Sample index or cluster size")
    ax.set_ylabel("Merge distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output_path: Path,
    *,
    records: list[dict[str, Any]],
    composition_features: pd.DataFrame,
    density: pd.Series,
    k_sweep: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    distance_metrics: pd.DataFrame,
    hdbscan_sensitivity: pd.DataFrame,
    cophenetic_correlation: float,
    primary_k: int,
) -> None:
    kmeans_best = (
        k_sweep[k_sweep["method"] == "kmeans"]
        .sort_values("silhouette", ascending=False)
        .iloc[0]
    )
    ward_best = (
        k_sweep[k_sweep["method"] == "ward"]
        .sort_values("silhouette", ascending=False)
        .iloc[0]
    )
    primary_row = k_sweep[
        (k_sweep["method"] == "kmeans") & (k_sweep["k"] == primary_k)
    ].iloc[0]
    size_series = cluster_summary["size"]
    largest = cluster_summary.sort_values("size", ascending=False).head(8)
    small_count = int((size_series < 5).sum())
    large_count = int((size_series >= 20).sum())
    ratio = float(
        distance_metrics.loc[
            distance_metrics["metric"] == "inter_to_intra_mean_ratio",
            "value",
        ].iloc[0]
    )

    lines = [
        "# Phase 3 Cluster Analysis Report",
        "",
        "## Scope",
        "",
        "This report characterizes the clustered structure of the 380-record experimental amorphous-alloy density dataset using element atomic-fraction features. It does not train or evaluate Roost models.",
        "",
        "## Dataset",
        "",
        f"- Records: {len(records)}",
        f"- Element-fraction features: {composition_features.shape[1]}",
        f"- Density range: {density.min():.3f} to {density.max():.3f} g/cm^3",
        f"- Density mean: {density.mean():.3f} g/cm^3",
        "",
        "## Cluster Count",
        "",
        f"- Primary grouping: K-means with k={primary_k}, random_state={RANDOM_STATE}, standardize=False.",
        f"- K-means best silhouette over k={K_MIN}-{K_MAX}: k={int(kmeans_best['k'])}, silhouette={kmeans_best['silhouette']:.4f}.",
        f"- Ward best silhouette over k={K_MIN}-{K_MAX}: k={int(ward_best['k'])}, silhouette={ward_best['silhouette']:.4f}.",
        f"- Ward dendrogram cophenetic correlation: {cophenetic_correlation:.4f}.",
        f"- Primary k={primary_k} K-means silhouette: {primary_row['silhouette']:.4f}.",
        f"- Primary k={primary_k} K-means Davies-Bouldin: {primary_row['davies_bouldin']:.4f}.",
        f"- Primary k={primary_k} K-means Calinski-Harabasz: {primary_row['calinski_harabasz']:.2f}.",
        "",
        "## Cluster Family Label Method",
        "",
        "Family labels are descriptive summaries of each composition cluster, not thermodynamic phase labels.",
        "",
        "For each k=36 K-means cluster:",
        "",
        "1. Compute the arithmetic mean atomic fraction of every element across all records in the cluster.",
        "2. Sort elements by descending mean atomic fraction.",
        f"3. Build the `family_label` from up to {FAMILY_LABEL_MAX_ELEMENTS} elements whose mean fraction is at least {FAMILY_LABEL_MIN_FRACTION:.2f}.",
        "4. Join those element symbols with hyphens, e.g. `Zr-Cu-Ni-Al`.",
        "5. If no element passes the threshold, use the single largest-mean element as a fallback.",
        "",
        f"The `top_elements` column is more detailed: it lists up to {TOP_ELEMENTS_MAX_ELEMENTS} elements with mean fraction at least {TOP_ELEMENTS_MIN_FRACTION:.2f}, including their mean fractions. The representative composition is the real record nearest to the K-means centroid in the original element-fraction feature space.",
        "",
        "Interpretation limits:",
        "",
        "- Minor alloying elements can be omitted from `family_label` even when they are physically important.",
        "- Small clusters have less stable labels because their mean composition is based on fewer records.",
        "- A label such as `Cu-Zr` means the cluster is Cu/Zr-dominant on average; it does not mean every record is binary Cu-Zr.",
        "",
        "## How To Read t-SNE",
        "",
        "- t-SNE converts high-dimensional composition vectors into a 2-D map by trying to preserve local neighborhoods.",
        "- In the original feature space, each alloy is represented by its element atomic fractions. t-SNE assigns high similarity to nearby compositions and low similarity to distant compositions.",
        "- In the 2-D map, t-SNE optimizes point positions so that alloys with high original-space similarity remain close. It uses a heavy-tailed low-dimensional distribution, which helps separate neighboring groups visually.",
        "- The plot is strongest evidence for local neighborhood structure: compact islands usually mean chemically similar composition families.",
        "- The absolute x/y axes are not physical coordinates, and large global distances or rotations should be interpreted qualitatively rather than as exact chemical distances.",
        "- Because random_state is fixed here, the current projection is reproducible for this dataset and parameter choice.",
        "",
        "## Cluster Sizes",
        "",
        f"- Cluster size min/median/max: {int(size_series.min())} / {size_series.median():.1f} / {int(size_series.max())}.",
        f"- Clusters with fewer than 5 records: {small_count}.",
        f"- Clusters with at least 20 records: {large_count}.",
        "",
        "Largest clusters:",
        "",
        "| cluster | size | family label | density mean | top elements |",
        "|---:|---:|---|---:|---|",
    ]
    for _, row in largest.iterrows():
        lines.append(
            f"| {int(row['cluster'])} | {int(row['size'])} | {row['family_label']} | "
            f"{row['density_mean']:.3f} | {row['top_elements']} |"
        )

    lines.extend(
        [
            "",
            "## Distance Structure",
            "",
            f"- Mean inter-centroid distance is {ratio:.2f} times the mean sample-to-centroid distance.",
            "- This supports treating the dataset as clustered composition families rather than a uniformly sampled composition space.",
            "- The many small clusters mean some group-heldout test splits can be dominated by sparse or chemically narrow families; report split-level variability and cluster composition alongside model metrics.",
            "",
            "## HDBSCAN Sensitivity",
            "",
        ]
    )
    if "status" in hdbscan_sensitivity.columns:
        lines.append(hdbscan_sensitivity.to_markdown(index=False))

    lines.extend(
        [
            "",
            "## Implication For The Generalization Yardstick",
            "",
            "- Random splits can put near-neighbor compositions from the same family on both sides of the split.",
            "- Group splits based on k=36 K-means are a reasonable primary extrapolation yardstick because the clusters are chemically interpretable and quantitatively separated.",
            "- Phase 4 model claims should use post-early-stop-fix reruns and report the random-vs-group gap with these cluster assignments preserved for provenance.",
            "",
            "## Generated Files",
            "",
            "- `cluster_assignments_k36.csv`",
            "- `cluster_summary_k36.csv`",
            "- `k_sweep_metrics.csv`",
            "- `distance_metrics.csv`",
            "- `centroid_distance_matrix_k36.csv`",
            "- `hdbscan_sensitivity.csv`",
            "- `figures/k_sweep_metrics.png`",
            "- `figures/pca_kmeans_k36.png`",
            "- `figures/tsne_kmeans_k36.png`",
            "- `figures/tsne_kmeans_k36_unlabeled_original.png`",
            "- `figures/tsne_kmeans_k36_labeled_major.png`",
            "- `figures/tsne_kmeans_k36_labeled_all.png`",
            "- `figures/cluster_size_distribution_k36.png`",
            "- `figures/centroid_distance_heatmap_k36.png`",
            "- `figures/ward_dendrogram_truncated.png`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_cluster_analysis(
    output_dir: Path,
    *,
    primary_k: int,
    k_min: int,
    k_max: int,
    random_state: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    records, composition_features, density = _load_dataset()
    x = composition_features.to_numpy(dtype=float)

    linkage_matrix = linkage(x, method="ward", metric="euclidean")
    cophenetic_correlation, _ = cophenet(linkage_matrix, pdist(x, metric="euclidean"))

    k_sweep = _kmeans_sweep(
        x,
        linkage_matrix,
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
    )
    k_sweep.to_csv(output_dir / "k_sweep_metrics.csv", index=False)

    kmeans = KMeans(n_clusters=primary_k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(x)
    centroids = kmeans.cluster_centers_
    sample_centroid_distances = np.linalg.norm(x - centroids[labels], axis=1)
    ward_labels = fcluster(linkage_matrix, t=primary_k, criterion="maxclust")

    hdbscan_sensitivity = _hdbscan_sensitivity(x)
    hdbscan_sensitivity.to_csv(output_dir / "hdbscan_sensitivity.csv", index=False)
    hdbscan_labels: np.ndarray | None = None
    try:
        import hdbscan

        hdbscan_labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(x)
    except ImportError:
        hdbscan_labels = None

    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(x)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    tsne_coords = tsne.fit_transform(x)

    cluster_summary, family_labels = _cluster_summary(
        composition_features=composition_features,
        density=density,
        records=records,
        labels=labels,
        centroids=centroids,
        sample_centroid_distances=sample_centroid_distances,
    )
    cluster_summary.to_csv(output_dir / "cluster_summary_k36.csv", index=False)

    assignments = _cluster_assignments(
        records=records,
        density=density,
        labels=labels,
        family_labels=family_labels,
        sample_centroid_distances=sample_centroid_distances,
        ward_labels=ward_labels,
        hdbscan_labels=hdbscan_labels,
        pca_coords=pca_coords,
        tsne_coords=tsne_coords,
    )
    assignments.to_csv(output_dir / "cluster_assignments_k36.csv", index=False)

    distance_metrics, centroid_distance_matrix = _distance_metrics(
        labels=labels,
        sample_centroid_distances=sample_centroid_distances,
        centroids=centroids,
    )
    distance_metrics.to_csv(output_dir / "distance_metrics.csv", index=False)
    centroid_distance_matrix.to_csv(output_dir / "centroid_distance_matrix_k36.csv")

    _plot_k_sweep(k_sweep, figures_dir / "k_sweep_metrics.png", primary_k=primary_k)
    _plot_embedding(
        pca_coords,
        labels,
        figures_dir / "pca_kmeans_k36.png",
        title="PCA projection colored by K-means k=36",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
    )
    tsne_original_path = figures_dir / "tsne_kmeans_k36.png"
    _preserve_original_figure(
        tsne_original_path,
        figures_dir / "tsne_kmeans_k36_unlabeled_original.png",
    )
    _plot_embedding(
        tsne_coords,
        labels,
        tsne_original_path,
        title="t-SNE projection colored by K-means k=36",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
    )
    _plot_tsne_with_family_labels(
        tsne_coords,
        labels,
        cluster_summary,
        figures_dir / "tsne_kmeans_k36_labeled_major.png",
        title="t-SNE projection with major alloy-family labels",
        min_size=8,
    )
    _plot_tsne_with_family_labels(
        tsne_coords,
        labels,
        cluster_summary,
        figures_dir / "tsne_kmeans_k36_labeled_all.png",
        title="t-SNE projection with all K-means cluster labels",
        min_size=1,
    )
    _plot_cluster_sizes(cluster_summary, figures_dir / "cluster_size_distribution_k36.png")
    _plot_centroid_heatmap(
        centroid_distance_matrix,
        figures_dir / "centroid_distance_heatmap_k36.png",
    )
    _plot_dendrogram(linkage_matrix, figures_dir / "ward_dendrogram_truncated.png")

    _write_report(
        output_dir / "phase3_cluster_analysis_report.md",
        records=records,
        composition_features=composition_features,
        density=density,
        k_sweep=k_sweep,
        cluster_summary=cluster_summary,
        distance_metrics=distance_metrics,
        hdbscan_sensitivity=hdbscan_sensitivity,
        cophenetic_correlation=float(cophenetic_correlation),
        primary_k=primary_k,
    )

    print(f"Records: {len(records)}")
    print(f"Element features: {composition_features.shape[1]}")
    print(f"Primary K-means k: {primary_k}")
    print(f"Outputs written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated cluster-analysis outputs.",
    )
    parser.add_argument("--primary-k", type=int, default=PRIMARY_K)
    parser.add_argument("--k-min", type=int, default=K_MIN)
    parser.add_argument("--k-max", type=int, default=K_MAX)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    run_cluster_analysis(
        args.output_dir,
        primary_k=args.primary_k,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
