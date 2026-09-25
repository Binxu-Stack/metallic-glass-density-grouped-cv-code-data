#!/usr/bin/env python3
"""Rebuild the simplified manuscript figures from archived numerical results.

Adapted from Evalute_ML_models/scripts/build_manuscript_figures.py and
build_manuscript_figure3.py. All writes are confined to this revised package.
Run with --source-project once to snapshot inputs, then omit it to reproduce.
"""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / 'figures'
DATA_DIR = PROJECT_ROOT / 'figure_data'

CLUSTER_DIR = DATA_DIR / "clustering"
POOLED_DIR = DATA_DIR / "uncertainty"
HARD_DIR = DATA_DIR / "family_residuals"
SCALE_DIR = DATA_DIR / "scale_fidelity"

OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

MODEL_ROLE = {
    "packing_eta_train_mae_opt": "Train-calibrated physics-based",
    "packing_eta_train_median": "Train-calibrated physics-based",
    "mixing_rule": "Zero-parameter physics-based",
    "random_forest_mendeleev": "Composition-based ML",
    "random_forest_magpie": "Composition-based ML",
    "Roost_scratch": "Composition-based ML",
    "Roost_physics_embed": "Physics-augmented ML",
    "Roost_trunk_atoms_density": "Physics-augmented ML",
    "Roost_trunk_mixing_rule": "Physics-augmented ML",
    "Roost_transfer": "Transfer-learning ML",
}

MODEL_ROLE_COLORS = {
    "Train-calibrated physics-based": OKABE_ITO["bluish_green"],
    "Zero-parameter physics-based": OKABE_ITO["sky_blue"],
    "Composition-based ML": OKABE_ITO["blue"],
    "Physics-augmented ML": OKABE_ITO["reddish_purple"],
    "Transfer-learning ML": OKABE_ITO["orange"],
}

ROLE_ORDER = [
    "Train-calibrated physics-based",
    "Zero-parameter physics-based",
    "Physics-augmented ML",
    "Transfer-learning ML",
    "Composition-based ML",
]

SCALE_DISPLAY = {
    "packing_eta_train_median": r"Packing model, median $\eta$",
    "packing_eta_train_mae_opt": r"Packing model, MAE-optimized $\eta$",
    "Roost_trunk_atoms_density": r"Roost + $\rho_{\mathrm{HS}}$ feature",
    "Roost_trunk_mixing_rule": r"Roost + $\rho_{\mathrm{ROM}}$ feature",
    "Roost_physics_embed": "Roost + mass–radius element embedding",
    "Roost_transfer": "Materials Project-pretrained Roost",
    "mixing_rule": "ROM density",
    "random_forest_mendeleev": "RF Mendeleev",
    "random_forest_magpie": "RF Magpie",
    "Roost_scratch": "Baseline Roost",
}

COMPACT_DISPLAY = {
    "packing_eta_train_median": "Packing model\nmedian $\\eta$",
    "packing_eta_train_mae_opt": "Packing model\nMAE-optimized $\\eta$",
    "Roost_trunk_atoms_density": "Roost + $\\rho_{\\mathrm{HS}}$\nfeature",
    "Roost_trunk_mixing_rule": "Roost + $\\rho_{\\mathrm{ROM}}$\nfeature",
    "Roost_physics_embed": "Roost + mass–radius\nelement embedding",
    "Roost_transfer": "Materials Project-\npretrained Roost",
    "mixing_rule": "ROM density",
    "random_forest_mendeleev": "RF Mendeleev",
    "random_forest_magpie": "RF Magpie",
    "Roost_scratch": "Baseline\nRoost",
}

def configure_style() -> None:
    """Apply a compact, journal-scale Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

def wrap_labels(labels: Iterable[object], width: int = 26) -> list[str]:
    return [
        textwrap.fill(str(label), width=width, break_long_words=False)
        for label in labels
    ]

def model_display(model: str, *, compact: bool = False) -> str:
    """Return the canonical reader-facing model label for an internal model ID."""
    labels = COMPACT_DISPLAY if compact else SCALE_DISPLAY
    return labels.get(str(model), str(model).replace("_", " "))

def role_for(model: str) -> str:
    return MODEL_ROLE.get(str(model), "Composition-based ML")

def role_color(role: str) -> str:
    return MODEL_ROLE_COLORS.get(str(role), "#666666")

def role_handles(roles: Iterable[str]) -> list[patches.Patch]:
    present = set(roles)
    return [
        patches.Patch(color=role_color(role), alpha=0.78, label=role)
        for role in ROLE_ORDER
        if role in present
    ]

def load_model_matrix() -> pd.DataFrame:
    return pd.read_csv(POOLED_DIR / "pooled_oof_point_metrics_wide.csv")

def build_figure1() -> plt.Figure:
    rng = np.random.default_rng(42)
    centers = np.array([[0.25, 0.70], [0.56, 0.42], [0.78, 0.76]])
    clouds = [center + rng.normal(0, 0.045, size=(22, 2)) for center in centers]
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.45), sharex=True, sharey=True)
    titles = [
        "Study-centered sampling",
        "Random split",
        "Composition-family\ngrouped split",
    ]

    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, pad=7)
        ax.set_xlim(0.08, 0.94)
        ax.set_ylim(0.18, 0.92)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for cloud in clouds:
        axes[0].scatter(cloud[:, 0], cloud[:, 1], s=16, color="#6E6E6E", alpha=0.7)
    axes[0].text(
        0.5,
        -0.05,
        "Published studies often sample local\ncomposition neighborhoods",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=7,
    )

    for cloud in clouds:
        mask = rng.random(len(cloud)) < 0.72
        axes[1].scatter(
            cloud[mask, 0], cloud[mask, 1], s=16, color=OKABE_ITO["blue"]
        )
        axes[1].scatter(
            cloud[~mask, 0],
            cloud[~mask, 1],
            s=28,
            marker="x",
            color=OKABE_ITO["orange"],
            linewidths=1.0,
        )
    axes[1].text(
        0.5,
        -0.05,
        "Train and test share nearby\ncomposition-family points",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=7,
    )

    for index, cloud in enumerate(clouds):
        held_out = index == 2
        color = OKABE_ITO["orange"] if held_out else OKABE_ITO["blue"]
        axes[2].scatter(
            cloud[:, 0],
            cloud[:, 1],
            s=30 if held_out else 16,
            marker="x" if held_out else "o",
            color=color,
            linewidths=1.0 if held_out else None,
        )
        axes[2].add_patch(
            patches.Ellipse(
                xy=centers[index],
                width=0.26,
                height=0.20,
                fill=False,
                lw=1.0,
                ls="--",
                edgecolor=color,
                alpha=0.9,
            )
        )
    axes[2].text(
        0.5,
        -0.05,
        "Held-out families approximate\nnew composition-domain discovery",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=7,
    )

    for label, ax in zip("abc", axes, strict=True):
        panel_label(ax, label)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=OKABE_ITO["blue"],
            markeredgecolor=OKABE_ITO["blue"],
            markersize=5,
            label="Train",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            color=OKABE_ITO["orange"],
            markersize=5,
            label="Test / held out",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.subplots_adjust(wspace=0.20, bottom=0.24)
    return fig

def build_figure2() -> plt.Figure:
    assignments = pd.read_csv(CLUSTER_DIR / "cluster_assignments_k36.csv")
    summary = pd.read_csv(CLUSTER_DIR / "cluster_summary_k36.csv")
    distance = pd.read_csv(CLUSTER_DIR / "distance_metrics.csv")
    k_sweep = pd.read_csv(CLUSTER_DIR / "k_sweep_metrics.csv")

    global_metrics = distance[distance["scope"] == "global"].set_index("metric")["value"]
    ratio = float(global_metrics["inter_to_intra_mean_ratio"])
    intra = float(global_metrics["sample_to_centroid_distance_mean"])
    inter = float(global_metrics["inter_centroid_distance_mean"])
    silhouette = float(
        k_sweep[(k_sweep["method"] == "kmeans") & (k_sweep["k"] == 36)][
            "silhouette"
        ].iloc[0]
    )
    fig = plt.figure(figsize=(7.2, 3.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.48)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_c = fig.add_subplot(grid[0, 1])

    clusters = assignments["cluster_kmeans_36"].to_numpy()
    colors = [plt.cm.tab20((int(cluster) % 20) / 20) for cluster in clusters]
    ax_a.scatter(
        assignments["tsne_1"],
        assignments["tsne_2"],
        s=16,
        c=colors,
        alpha=0.78,
        linewidths=0,
    )
    for row in summary.sort_values("size", ascending=False).head(10).itertuples():
        points = assignments[assignments["cluster_kmeans_36"] == row.cluster]
        ax_a.text(
            float(points["tsne_1"].median()),
            float(points["tsne_2"].median()),
            str(row.family_label),
            ha="center",
            va="center",
            fontsize=6,
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": "#BDBDBD",
                "lw": 0.4,
                "alpha": 0.82,
            },
        )
    ax_a.set_title("t-SNE projection colored by k=36 family")
    ax_a.set_xlabel("t-SNE 1")
    ax_a.set_ylabel("t-SNE 2")
    panel_label(ax_a, "a")

    ax_c.barh(
        ["Within-family\nmean", "Between-centroid\nmean"],
        [intra, inter],
        color=[OKABE_ITO["sky_blue"], OKABE_ITO["vermillion"]],
    )
    ax_c.set_title("Separation of families")
    ax_c.set_xlabel("Euclidean distance")
    ax_c.text(
        0.98,
        0.08,
        f"inter / intra = {ratio:.2f}x\nsilhouette = {silhouette:.3f}",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    panel_label(ax_c, "b")
    return fig

def build_figure4(matrix: pd.DataFrame) -> plt.Figure:
    gaps = matrix.sort_values("grouped_minus_random_MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    roles = [role_for(model) for model in gaps["model"]]
    ax.barh(np.arange(len(gaps)), gaps["grouped_minus_random_MAE"],
            color=[role_color(role) for role in roles])
    ax.set_yticks(np.arange(len(gaps)),
                  [model_display(model) for model in gaps["model"]])
    ax.invert_yaxis()
    ax.set_xlabel("Grouped MAE - random MAE (g cm$^{-3}$)")
    ax.axvline(0, color="#333333", lw=0.8)
    fig.legend(handles=role_handles(roles), frameon=False, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.37, right=0.98, bottom=0.23, top=0.98)
    return fig

def build_figure5() -> plt.Figure:
    summary = pd.read_csv(SCALE_DIR / "scale_fidelity_summary.csv")
    matrix = summary[~summary["is_diagnostic"]]
    grouped = matrix[matrix["cv_protocol"] == "grouped_10fold"].copy()
    random_cv = matrix[matrix["cv_protocol"] == "random_10fold"].copy()
    grouped["role"] = [role_for(model) for model in grouped["model"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    ax = axes[0]
    for role, block in grouped.groupby("role"):
        ax.errorbar(
            block["slope_b"],
            block["MAE"],
            xerr=np.vstack(
                [
                    block["slope_b"] - block["slope_ci_lower"],
                    block["slope_ci_upper"] - block["slope_b"],
                ]
            ),
            fmt="o",
            markersize=5,
            color=role_color(role),
            ecolor=role_color(role),
            elinewidth=0.8,
            capsize=1.8,
            zorder=3,
        )
    ax.axvline(1.0, color="#666666", lw=0.8, ls=":")
    ax.annotate(
        "$b=1$",
        xy=(1.0, 0.97),
        xycoords=("data", "axes fraction"),
        xytext=(-3, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=7,
        color="#666666",
    )
    ax.set_xlabel("Prediction–observation slope $b$\n(grouped cross-validation)")
    ax.set_ylabel("Held-out-family MAE (g cm$^{-3}$)")
    panel_label(ax, "a")

    ax = axes[1]
    pair = grouped[["model", "slope_b", "role"]].merge(
        random_cv[["model", "slope_b"]],
        on="model",
        suffixes=("_grouped", "_random"),
    )
    pair = pair.sort_values("slope_b_grouped")
    y = np.arange(len(pair))
    for index, row in enumerate(pair.itertuples()):
        ax.plot(
            [row.slope_b_random, row.slope_b_grouped],
            [index, index],
            color=role_color(row.role),
            lw=1.4,
            alpha=0.55,
            zorder=2,
        )
    ax.scatter(
        pair["slope_b_random"],
        y,
        marker="o",
        s=22,
        facecolor="white",
        edgecolor="#222222",
        linewidth=0.9,
        zorder=3,
    )
    ax.scatter(
        pair["slope_b_grouped"],
        y,
        marker="o",
        s=24,
        color=[role_color(role) for role in pair["role"]],
        zorder=4,
    )
    protocol_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor="#222222",
            label="Random cross-validation",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=5,
            markerfacecolor="#666666",
            markeredgecolor="#666666",
            label="Grouped cross-validation (color = role)",
        ),
    ]
    ax.axvline(1.0, color="#666666", lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([model_display(model, compact=True) for model in pair["model"]])
    ax.set_xlabel("Prediction–observation slope $b$")
    ax.legend(
        handles=protocol_handles,
        frameon=False,
        loc="upper left",
        fontsize=6.5,
        handletextpad=0.4,
    )
    panel_label(ax, "b")

    fig.legend(
        handles=role_handles(grouped["role"]),
        frameon=False,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    return fig

def build_figure6() -> plt.Figure:
    hard = pd.read_csv(HARD_DIR / "hard_roost_cluster_deltas.csv")
    model_metrics = pd.read_csv(HARD_DIR / "hard_roost_cluster_model_metrics.csv")
    bp = pd.read_csv(HARD_DIR / "bp_si_grouped_summary.csv")

    fig = plt.figure(figsize=(7.3, 5.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.05], hspace=0.58, wspace=0.50)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    top = hard.head(8).copy()
    y = np.arange(len(top))
    height = 0.36
    ax_a.barh(
        y - height / 2,
        top["roost_MAE"],
        height=height,
        color=OKABE_ITO["vermillion"],
        label="Baseline Roost",
    )
    ax_a.barh(
        y + height / 2,
        top["best_non_roost_MAE"],
        height=height,
        color=OKABE_ITO["bluish_green"],
        label="Best non-Roost",
    )
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(
        [f"{row.family_label} (n={int(row.n_records)})" for row in top.itertuples()]
    )
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Family MAE (g cm$^{-3}$)")
    ax_a.legend(frameon=False, loc="lower right")
    panel_label(ax_a, "a")

    selected_families = ["Pt-P-Ni", "Ca-Mg-Cu", "Mg-Cu-Gd"]
    selected = hard[hard["family_label"].isin(selected_families)].copy()
    selected["order"] = selected["family_label"].map(
        {name: index for index, name in enumerate(selected_families)}
    )
    selected = selected.sort_values("order")
    y2 = np.arange(len(selected))
    ax_b.set_yticks(y2)
    ax_b.set_yticklabels(selected["family_label"])
    for index, row in enumerate(selected.itertuples()):
        rows = model_metrics[model_metrics["family_label"] == row.family_label]
        measured = float(rows["mean_density"].iloc[0])
        roost_prediction = float(
            rows[rows["model"] == "Roost_scratch"]["mean_prediction"].iloc[0]
        )
        best_prediction = float(
            rows[rows["model"] == str(row.best_non_roost_model)]["mean_prediction"].iloc[0]
        )
        ax_b.plot(
            [measured, roost_prediction],
            [index, index],
            color="#BDBDBD",
            lw=1.1,
            zorder=1,
        )
        ax_b.scatter(
            measured,
            index,
            color="black",
            s=26,
            label="Measured mean" if index == 0 else None,
            zorder=3,
        )
        ax_b.scatter(
            roost_prediction,
            index,
            color=OKABE_ITO["vermillion"],
            marker="x",
            s=34,
            label="Baseline Roost mean" if index == 0 else None,
            zorder=3,
        )
        ax_b.scatter(
            best_prediction,
            index,
            color=OKABE_ITO["bluish_green"],
            marker="D",
            s=22,
            label="Best non-Roost mean" if index == 0 else None,
            zorder=3,
        )
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Density (g cm$^{-3}$)")
    ax_b.legend(frameon=False, loc="lower right", fontsize=6)
    panel_label(ax_b, "b")

    compare_models = ["mixing_rule", "packing_eta_train_mae_opt"]
    labels = {
        "mixing_rule": "ROM density",
        "packing_eta_train_mae_opt": "Packing model",
        "Roost_scratch": "Baseline Roost",
    }
    subset = bp[bp["model"].isin(compare_models)].copy()
    x = np.arange(len(compare_models))
    width = 0.36
    without = subset[~subset["has_B_P_Si"].astype(bool)].set_index("model")
    with_bpsi = subset[subset["has_B_P_Si"].astype(bool)].set_index("model")
    ax_c.bar(
        x - width / 2,
        [without.loc[model, "MAE"] for model in compare_models],
        width=width,
        color=OKABE_ITO["sky_blue"],
        label="No B/P/Si",
    )
    ax_c.bar(
        x + width / 2,
        [with_bpsi.loc[model, "MAE"] for model in compare_models],
        width=width,
        color=OKABE_ITO["orange"],
        label="B/P/Si",
    )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([labels[model] for model in compare_models], rotation=28, ha="right")
    ax_c.set_ylabel("MAE (g cm$^{-3}$)")
    ax_c.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax_c.text(
        0.98,
        0.86,
        f"B/P/Si n={int(with_bpsi['n_records'].iloc[0])}\nother n={int(without['n_records'].iloc[0])}",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )
    panel_label(ax_c, "c")
    return fig

def build_s1() -> plt.Figure:
    metrics = pd.read_csv(CLUSTER_DIR / "k_sweep_metrics.csv")
    subset = metrics[metrics["method"].eq("kmeans")]
    assert len(subset) == 99
    fig, ax = plt.subplots(figsize=(4.4, 3.0), constrained_layout=True)
    ax.plot(subset["k"], subset["silhouette"], lw=1.5, color=OKABE_ITO["blue"])
    best = subset.loc[subset["silhouette"].idxmax()]
    assert int(best["k"]) == 36
    ax.axvline(36, color="0.3", lw=0.8, ls="--")
    ax.scatter([36], [best["silhouette"]], color=OKABE_ITO["orange"], zorder=3)
    ax.annotate("k = 36", (36, best["silhouette"]), xytext=(8, -16),
                textcoords="offset points", fontsize=8)
    ax.set_xlabel("Number of clusters, k")
    ax.set_ylabel("Silhouette score")
    return fig

def build_s2() -> plt.Figure:
    """Build high-error family residuals used as Supplementary Fig. S2."""
    data = pd.read_csv(HARD_DIR / "hard_roost_cluster_model_metrics.csv")
    keep_models = [
        "Roost_scratch",
        "mixing_rule",
        "packing_eta_train_median",
        "random_forest_mendeleev",
    ]
    data = data[
        data["model"].isin(keep_models) & data["roost_hard_rank"].le(8)
    ].copy()
    order = (
        data[data["model"].eq("Roost_scratch")]
        .sort_values("roost_hard_rank")["family_label"]
        .tolist()
    )
    pivot = (
        data.pivot_table(index="family_label", columns="model", values="MAE", aggfunc="first")
        .loc[order]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    width = 0.19
    x = np.arange(len(pivot.index))
    colors = [
        OKABE_ITO["vermillion"],
        OKABE_ITO["sky_blue"],
        OKABE_ITO["bluish_green"],
        OKABE_ITO["blue"],
    ]
    for index, model in enumerate(keep_models):
        offset = (index - 1.5) * width
        ax.bar(
            x + offset,
            pivot[model],
            width=width,
            label=model_display(model),
            color=colors[index],
            edgecolor="0.25",
            linewidth=0.25,
        )
    ax.set_xticks(x, pivot.index, rotation=30, ha="right")
    ax.set_ylabel("Family MAE (g cm$^{-3}$)")
    ax.legend(frameon=False, ncols=2)
    return fig


def build_s3() -> plt.Figure:
    """Build paired parity plots for three Roost information regimes."""
    residuals = pd.read_csv(POOLED_DIR / "pooled_oof_residuals_long.csv")
    models = ["Roost_scratch", "Roost_transfer", "Roost_trunk_atoms_density"]
    protocols = ["random_10fold", "grouped_10fold"]
    data = residuals[
        residuals["model"].isin(models)
        & residuals["cv_protocol"].isin(protocols)
    ].copy()

    counts = data.groupby(["model", "cv_protocol"])["record_index"].agg(
        ["size", "nunique"]
    )
    expected_pairs = {(model, protocol) for model in models for protocol in protocols}
    if set(counts.index) != expected_pairs:
        missing = sorted(expected_pairs - set(counts.index))
        raise ValueError(f"Missing Roost parity model-protocol pairs: {missing}")
    if not (counts["size"].eq(380).all() and counts["nunique"].eq(380).all()):
        raise ValueError("Parity data must contain 380 unique records per panel")
    if data.duplicated(["model", "cv_protocol", "record_index"]).any():
        raise ValueError("Duplicate model-protocol-record keys in parity data")

    reference = (
        data[
            data["model"].eq("Roost_scratch")
            & data["cv_protocol"].eq("random_10fold")
        ]
        .set_index("record_index")["density"]
        .sort_index()
    )
    for (model, protocol), panel in data.groupby(["model", "cv_protocol"]):
        measured = panel.set_index("record_index")["density"].sort_index()
        if not reference.index.equals(measured.index) or not np.array_equal(
            reference.to_numpy(), measured.to_numpy()
        ):
            raise ValueError(
                f"Parity panel {model}/{protocol} does not use the common records"
            )

    expected = load_model_matrix().set_index("model").loc[models]
    expected_slopes = (
        pd.read_csv(SCALE_DIR / "scale_fidelity_summary.csv")
        .query("model in @models and cv_protocol in @protocols")
        .set_index(["model", "cv_protocol"])["slope_b"]
    )
    colors = {
        "random_10fold": OKABE_ITO["sky_blue"],
        "grouped_10fold": OKABE_ITO["vermillion"],
    }
    titles = {
        "random_10fold": "Random cross-validation",
        "grouped_10fold": "Composition-family grouped\ncross-validation",
    }
    row_labels = {
        "Roost_scratch": "Baseline Roost",
        "Roost_transfer": "Materials Project-pretrained Roost",
        "Roost_trunk_atoms_density": r"Roost + $\rho_{\mathrm{HS}}$ feature",
    }

    lower = np.floor(data[["density", "prediction"]].to_numpy().min())
    upper = np.ceil(data[["density", "prediction"]].to_numpy().max())
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.25), sharex=True, sharey=True)

    panel_labels = iter("abcdef")
    for row, model in enumerate(models):
        for column, protocol in enumerate(protocols):
            ax = axes[row, column]
            panel = data[
                data["model"].eq(model) & data["cv_protocol"].eq(protocol)
            ].sort_values("record_index")
            measured = panel["density"].to_numpy()
            predicted = panel["prediction"].to_numpy()
            residual = predicted - measured
            mae = float(np.mean(np.abs(residual)))
            r2 = float(
                1.0
                - np.sum(residual**2)
                / np.sum((measured - measured.mean()) ** 2)
            )
            slope, intercept = np.polyfit(measured, predicted, 1)
            metric_prefix = "random" if protocol == "random_10fold" else "grouped"
            if not np.isclose(
                mae, expected.loc[model, f"{metric_prefix}_MAE"], atol=1e-12
            ):
                raise ValueError(
                    f"Parity MAE disagrees with archived matrix for {model}/{protocol}"
                )
            if not np.isclose(
                r2, expected.loc[model, f"{metric_prefix}_R2"], atol=1e-12
            ):
                raise ValueError(
                    f"Parity R2 disagrees with archived matrix for {model}/{protocol}"
                )
            if not np.isclose(
                slope, expected_slopes.loc[(model, protocol)], atol=1e-12
            ):
                raise ValueError(
                    f"Parity slope disagrees with archived summary for {model}/{protocol}"
                )

            ax.scatter(
                measured,
                predicted,
                s=10,
                color=colors[protocol],
                alpha=0.55,
                edgecolors="none",
                zorder=2,
            )
            ax.plot(
                [lower, upper],
                [lower, upper],
                color="0.25",
                lw=0.8,
                ls="--",
                zorder=1,
            )
            fit_x = np.array([measured.min(), measured.max()])
            ax.plot(
                fit_x,
                intercept + slope * fit_x,
                color=colors[protocol],
                lw=1.3,
                zorder=3,
            )
            ax.text(
                0.04,
                0.96,
                (
                    f"n = {len(panel)}\n"
                    f"MAE = {mae:.3f}\n"
                    f"$R^2$ = {r2:.3f}\n"
                    f"$b$ = {slope:.3f}"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "fc": "white",
                    "ec": "0.75",
                    "lw": 0.45,
                    "alpha": 0.9,
                },
            )
            if row == 0:
                ax.set_title(titles[protocol], pad=7)
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.set_aspect("equal", adjustable="box")
            panel_label(ax, next(panel_labels))

    line_handles = [
        Line2D([], [], color="0.25", lw=0.9, ls="--", label="Identity line"),
        Line2D([], [], color="0.35", lw=1.5, label="OLS fit"),
    ]
    fig.legend(
        handles=line_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.53, 0.005),
    )
    fig.supxlabel("Measured density (g cm$^{-3}$)", y=0.045)
    fig.supylabel("Predicted density (g cm$^{-3}$)", x=0.025)
    fig.subplots_adjust(
        left=0.18,
        right=0.985,
        bottom=0.115,
        top=0.945,
        hspace=0.20,
        wspace=0.12,
    )
    for row, model in enumerate(models):
        position = axes[row, 0].get_position()
        fig.text(
            0.105,
            (position.y0 + position.y1) / 2,
            row_labels[model],
            rotation=90,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    return fig

from matplotlib.ticker import MultipleLocator

MODEL_LABELS = {
    "packing_eta_train_median": r"Packing model, median $\eta$",
    "packing_eta_train_mae_opt": r"Packing model, MAE-optimized $\eta$",
    "Roost_trunk_atoms_density": r"Roost + $\rho_{\mathrm{HS}}$ feature",
    "Roost_trunk_mixing_rule": r"Roost + $\rho_{\mathrm{ROM}}$ feature",
    "Roost_transfer": "Materials Project-pretrained Roost",
    "Roost_physics_embed": "Roost + mass–radius element embedding",
    "mixing_rule": "ROM density",
    "random_forest_mendeleev": "Random Forest, Mendeleev",
    "Roost_scratch": "Baseline Roost",
    "random_forest_magpie": "Random Forest, Magpie",
}

MODEL_ROLES = {
    "packing_eta_train_median": "Train-calibrated physics-based",
    "packing_eta_train_mae_opt": "Train-calibrated physics-based",
    "Roost_trunk_atoms_density": "Physics-augmented ML",
    "Roost_trunk_mixing_rule": "Physics-augmented ML",
    "Roost_physics_embed": "Physics-augmented ML",
    "Roost_transfer": "Transfer-learning ML",
    "mixing_rule": "Zero-parameter physics-based",
    "random_forest_mendeleev": "Composition-based ML",
    "Roost_scratch": "Composition-based ML",
    "random_forest_magpie": "Composition-based ML",
}

ROLE_STYLES = {
    "Train-calibrated physics-based": {"color": "#E69F00", "marker": "o"},
    "Physics-augmented ML": {"color": "#0072B2", "marker": "s"},
    "Transfer-learning ML": {"color": "#009E73", "marker": "D"},
    "Zero-parameter physics-based": {"color": "#4D4D4D", "marker": "^"},
    "Composition-based ML": {"color": "#D55E00", "marker": "X"},
}


def load_grouped_mae(bootstrap_csv: Path) -> pd.DataFrame:
    """Load cluster-bootstrap grouped-CV MAE estimates for the ten main models."""
    data = pd.read_csv(bootstrap_csv)
    selected = data.loc[
        (data["bootstrap_unit"] == "cluster")
        & (data["comparison"] == "protocol")
        & (data["metric"] == "MAE")
        & (data["cv_protocol"] == "grouped_10fold")
    ].copy()

    selected = selected.loc[selected["model"].isin(MODEL_LABELS)].copy()
    missing = sorted(set(MODEL_LABELS) - set(selected["model"]))
    if missing:
        raise ValueError(f"Missing grouped-MAE rows for: {missing}")
    if selected["model"].duplicated().any():
        duplicated = selected.loc[selected["model"].duplicated(), "model"].tolist()
        raise ValueError(f"Duplicate grouped-MAE rows for: {duplicated}")

    selected["label"] = selected["model"].map(MODEL_LABELS)
    selected["role"] = selected["model"].map(MODEL_ROLES)
    selected = selected.sort_values("point_estimate", ascending=True).reset_index(drop=True)
    return selected


def build_figure3(data: pd.DataFrame) -> plt.Figure:
    """Create the grouped-MAE ranking as a publication-ready dot-whisker plot."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    y_positions = range(len(data))

    for y, row in zip(y_positions, data.itertuples(index=False), strict=True):
        style = ROLE_STYLES[row.role]
        lower = row.point_estimate - row.ci_lower
        upper = row.ci_upper - row.point_estimate
        ax.errorbar(
            row.point_estimate,
            y,
            xerr=[[lower], [upper]],
            fmt=style["marker"],
            color=style["color"],
            ecolor=style["color"],
            markeredgecolor="black",
            markeredgewidth=0.55,
            markersize=6.2,
            elinewidth=1.2,
            capsize=2.5,
            capthick=1.0,
            zorder=3,
        )

    ax.set_yticks(list(y_positions), data["label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel(r"Held-out-family MAE (g cm$^{-3}$)")
    ax.set_xlim(0.0, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=3, width=0.7)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            linestyle="none",
            markerfacecolor=style["color"],
            markeredgecolor="black",
            markeredgewidth=0.55,
            markersize=5.5,
            label=role,
        )
        for role, style in ROLE_STYLES.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.57, 0.015),
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.36, right=0.985, top=0.98, bottom=0.18)
    return fig



INPUTS = {
    "clustering": [
        "cluster_assignments_k36.csv", "cluster_summary_k36.csv",
        "distance_metrics.csv", "k_sweep_metrics.csv"],
    "uncertainty": [
        "pooled_oof_point_metrics_wide.csv", "pooled_oof_bootstrap_summary.csv",
        "pooled_oof_residuals_long.csv"],
    "family_residuals": [
        "hard_roost_cluster_deltas.csv", "hard_roost_cluster_model_metrics.csv",
        "bp_si_grouped_summary.csv"],
    "scale_fidelity": ["scale_fidelity_summary.csv"],
}

SOURCE_DIRS = {
    "clustering": "cluster_analysis",
    "uncertainty": "phase4_pooled_oof_uncertainty",
    "family_residuals": "phase4_hard_grouped_residuals",
    "scale_fidelity": "phase4_scale_fidelity",
}

def stage_inputs(source: Path) -> None:
    """Snapshot only existing results; never fit a model or alter source results."""
    import hashlib
    import json
    manifest = []
    for folder, files in INPUTS.items():
        for name in files:
            src = source / "results" / SOURCE_DIRS[folder] / name
            data = pd.read_csv(src)
            if name == "k_sweep_metrics.csv":
                data = data[data["method"].eq("kmeans")]
            if name == "cluster_assignments_k36.csv":
                data = data[[
                    "record_index", "composition_str", "density",
                    "cluster_kmeans_36", "family_label", "tsne_1", "tsne_2",
                ]]
            if "model" in data:
                data = data[data["model"].isin(MODEL_ROLE)]
            if name == "pooled_oof_bootstrap_summary.csv":
                data = data[data["bootstrap_unit"].eq("cluster")]
            if name == "pooled_oof_residuals_long.csv":
                data = data[
                    data["model"].isin([
                        "Roost_scratch",
                        "Roost_transfer",
                        "Roost_trunk_atoms_density",
                    ])
                    & data["cv_protocol"].isin(["random_10fold", "grouped_10fold"])
                ][[
                    "cv_protocol", "fold", "model", "record_index",
                    "composition_str", "density", "prediction",
                    "cluster_kmeans_36", "family_label",
                ]]
            for col in data.select_dtypes(include=["object", "str"]):
                data[col] = data[col].str.replace("Baseline Roost", "Baseline Roost", regex=False)
            dest = DATA_DIR / folder / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(dest, index=False)
            manifest.append({
                "source": str(src.relative_to(source)), "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                "snapshot": str(dest.relative_to(PROJECT_ROOT)), "rows": len(data)})
    (DATA_DIR / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path,
                        help="Read-only source project; omit to use the saved figure_data.")
    args = parser.parse_args()
    if args.source_project:
        stage_inputs(args.source_project.resolve())
    configure_style()
    plt.rcParams["svg.fonttype"] = "none"
    # Keep the same role colors across all model-comparison figures.
    for role in ROLE_STYLES:
        ROLE_STYLES[role]["color"] = role_color(role)
    grouped_mae = load_grouped_mae(POOLED_DIR / "pooled_oof_bootstrap_summary.csv")
    figures = [
        ("figure1_group_aware_validation_schematic", build_figure1()),
        ("figure2_cluster_evidence", build_figure2()),
        ("figure3_grouped_mae_ranking", build_figure3(grouped_mae)),
        ("figure4_validation_gap_diagnostic", build_figure4(load_model_matrix())),
        ("figure5_scale_fidelity", build_figure5()),
        ("figure6_physical_interpretation", build_figure6()),
        ("figure_s1_clustering_diagnostics", build_s1()),
        ("figure_s2_hard_cluster_residuals", build_s2()),
        ("figure_s3_random_grouped_parity", build_s3()),
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem, fig in figures:
        for ext in ("pdf", "svg", "png"):
            fig.savefig(OUTPUT_DIR / f"{stem}.{ext}", bbox_inches="tight",
                        facecolor="white", dpi=240)
        plt.close(fig)
    print(f"Built {len(figures)} figures from archived results; no models retrained.")

if __name__ == "__main__":
    main()
