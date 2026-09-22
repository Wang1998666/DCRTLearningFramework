"""Figures for the leakage-resistant, sea-route DCRank experiment (v8).

Every plotted value is read from a saved result file. There are no manually
entered performance or sensitivity values in this module.
Publication-quality aesthetic refactor with Times New Roman and unified palette.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "02_数据_output" / "ml_bunkering_v1"
FIG = PROJECT / "04_图表_figures" / "ml_bunkering_v1"
LATEX_FIG = PROJECT / "05_手稿" / "Latex_ML"
FIG.mkdir(parents=True, exist_ok=True)

# Publication styling: Times New Roman & STIX math
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 200,
})

# Paper-wide unified color palette
COLOR_NAVY = "#1E40AF"     # Primary model / Top-1 / Learned
COLOR_GREEN = "#059669"    # Top-5 / Secondary metric
COLOR_AMBER = "#D97706"    # Top-3
COLOR_GREY = "#64748B"     # Baseline / Shortest route
COLOR_CRIMSON = "#DC2626"  # Highlight / Crisis / Selected budget


def save(fig, stem: str):
    """Save to both figure archive and LaTeX manuscript folder."""
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    if LATEX_FIG.exists():
        fig.savefig(LATEX_FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(LATEX_FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}")


def main_performance():
    metrics = pd.read_csv(OUT / "dcrank_searoute_metrics_v8.csv").set_index("model")
    curve = pd.read_csv(OUT / "dcrank_searoute_budget_curve_v8.csv")
    report = json.loads((OUT / "dcrank_searoute_report_v8.json").read_text(encoding="utf-8"))

    models = ["shortest_rolling", "fixed50_rolling", "learned_global_rolling"]
    labels = ["Shortest sea route", "Fixed 50 nm", "Learned budget"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2))

    # --- Panel (a): Grouped accuracy comparison ---
    x = np.arange(len(models))
    width = 0.32
    top1 = metrics.loc[models, "top1_end_to_end"].to_numpy() * 100
    top5 = metrics.loc[models, "top5_end_to_end"].to_numpy() * 100

    b1 = ax1.bar(x - width / 2, top1, width, label="Top-1 accuracy",
                 color=COLOR_NAVY, edgecolor="#0F172A", linewidth=0.6, alpha=0.92)
    b5 = ax1.bar(x + width / 2, top5, width, label="Top-5 accuracy",
                 color=COLOR_GREEN, edgecolor="#0F172A", linewidth=0.6, alpha=0.88)

    for bars in (b1, b5):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.0,
                     f"{h:.1f}%", ha="center", va="bottom", fontsize=8.8,
                     fontweight="bold", color="#1E293B")

    ax1.set_xticks(x, labels, fontsize=9.5)
    ax1.set_ylabel("End-to-end accuracy (%)", fontsize=10.0)
    ax1.set_ylim(0, 108)
    ax1.set_title("(a) Hormuz-period container vessels", fontsize=10.5,
                  fontweight="bold", loc="left", pad=8)
    ax1.grid(axis="y", linestyle=":", alpha=0.5, color="#CBD5E1")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0",
               framealpha=0.95, loc="upper left")

    # --- Panel (b): Distance budget sensitivity curve ---
    for split, label, style in [
        ("validation", "2023 validation", "-"),
        ("hormuz_test_descriptive", "2026 test (descriptive)", "--"),
    ]:
        z = curve[curve.split.eq(split)]
        ax2.plot(z.budget_nm, z.top1_end_to_end * 100, style, color=COLOR_NAVY,
                 lw=1.9, label=f"Top-1, {label}")
        ax2.plot(z.budget_nm, z.top5_end_to_end * 100, style, color=COLOR_GREEN,
                 lw=1.9, label=f"Top-5, {label}")

    learned = float(report["learned_global_budget_nm"])
    ax2.axvline(50, color=COLOR_GREY, linestyle=":", lw=1.5,
                label="Fixed 50 nm", zorder=3)
    ax2.axvline(learned, color=COLOR_CRIMSON, linestyle="-.", lw=1.6,
                label=f"NLL-selected {learned:.1f} nm", zorder=3)

    ax2.set_xlabel("Distance budget (nautical miles)", fontsize=10.0)
    ax2.set_ylabel("End-to-end accuracy (%)", fontsize=10.0)
    ax2.set_title("(b) Computed budget sensitivity", fontsize=10.5,
                  fontweight="bold", loc="left", pad=8)
    ax2.set_ylim(35, 100)
    ax2.grid(linestyle=":", alpha=0.5, color="#CBD5E1")
    ax2.legend(frameon=True, facecolor="white", framealpha=0.92,
               edgecolor="#E2E8F0", loc="lower right", fontsize=8.2)

    fig.tight_layout(w_pad=2.2)
    save(fig, "fig_dcrank_searoute_main_v8")


def diagnostics():
    metrics = pd.read_csv(OUT / "dcrank_searoute_metrics_v8.csv").set_index("model")
    crude = pd.read_csv(OUT / "dcrank_searoute_crude_metrics_v8.csv").set_index("model")
    seeds = pd.read_csv(OUT / "dcrank_searoute_seed_stability_v8.csv")
    shuffle = pd.read_csv(OUT / "dcrank_searoute_shuffle_v8.csv")
    budgets = pd.read_parquet(OUT / "dcrank_searoute_dynamic_budgets_v8.parquet")

    fig = plt.figure(figsize=(12.4, 3.8))
    gs_main = fig.add_gridspec(1, 3, wspace=0.28)

    # --- Panel (a): Near-constant dynamic budgets ---
    ax_a = fig.add_subplot(gs_main[0])
    ax_a.hist(budgets.dynamic_budget_nm, bins=25, color=COLOR_NAVY,
              edgecolor="#FFFFFF", linewidth=0.5, alpha=0.88)
    mean_b = budgets.dynamic_budget_nm.mean()
    ax_a.axvline(mean_b, color=COLOR_CRIMSON, linestyle="--", lw=1.6,
                 label=f"Mean: {mean_b:.1f} nm")
    ax_a.set_xlabel("Event-specific budget (nm)", fontsize=10.0)
    ax_a.set_ylabel("Voyage events", fontsize=10.0)
    ax_a.set_title("(a) Event-specific dynamic budgets", fontsize=10.5,
                   fontweight="bold", loc="left", pad=8)
    ax_a.grid(axis="y", linestyle=":", alpha=0.5, color="#CBD5E1")
    ax_a.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper left")

    # --- Panel (b): Falsification and seed stability (broken axis) ---
    # Left sub-axis shows the 100 within-event shuffles; right sub-axis shows 5 seeds & observed.
    gs_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1], width_ratios=[1.2, 1.0], wspace=0.18)
    ax_b1 = fig.add_subplot(gs_b[0])
    ax_b2 = fig.add_subplot(gs_b[1])

    # Left: Shuffle distribution (null model)
    shuf_vals = shuffle.top1_end_to_end.to_numpy() * 100
    ax_b1.hist(shuf_vals, bins=12, color=COLOR_GREY, edgecolor="#FFFFFF",
               linewidth=0.5, alpha=0.85, label="100 shuffles")
    ax_b1.axvline(shuf_vals.mean(), color="#334155", linestyle=":", lw=1.4,
                  label=f"Mean {shuf_vals.mean():.1f}%")
    ax_b1.set_xlim(21.5, 23.5)
    ax_b1.set_xlabel("Accuracy (%)", fontsize=9.0)
    ax_b1.set_ylabel("Shuffle count", fontsize=10.0)
    ax_b1.grid(axis="y", linestyle=":", alpha=0.5, color="#CBD5E1")
    ax_b1.legend(frameon=False, fontsize=7.8, loc="upper right")

    # Right: Observed and seed stability
    observed = metrics.loc["learned_global_rolling", "top1_end_to_end"] * 100
    seed_values = seeds.top1_end_to_end.to_numpy() * 100
    ax_b2.axvline(observed, color=COLOR_CRIMSON, linewidth=2.0, zorder=3,
                  label=f"Observed: {observed:.1f}%")
    ax_b2.scatter(seed_values, np.linspace(2, 10, len(seed_values)), marker="o", s=32,
                  color=COLOR_NAVY, edgecolor="#FFFFFF", linewidth=0.7, zorder=5,
                  label=f"5 seeds ({seed_values.mean():.2f}%)")
    ax_b2.set_xlim(62.4, 63.0)
    ax_b2.set_xlabel("Accuracy (%)", fontsize=9.0)
    ax_b2.set_yticks([])
    ax_b2.grid(axis="x", linestyle=":", alpha=0.5, color="#CBD5E1")
    ax_b2.legend(frameon=False, fontsize=7.8, loc="upper right")

    # Broken axis tick marks
    d = 0.02
    kwargs = dict(transform=ax_b1.transAxes, color="#64748B", clip_on=False, linewidth=1.0)
    ax_b1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_b1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_b2.transAxes)
    ax_b2.plot((-d, +d), (-d, +d), **kwargs)
    ax_b2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    ax_b1.spines["right"].set_visible(False)
    ax_b2.spines["left"].set_visible(False)
    ax_b1.set_title("(b) Falsification & stability", fontsize=10.5,
                    fontweight="bold", loc="left", pad=8)

    # --- Panel (c): Cross-fleet transfer ---
    ax_c = fig.add_subplot(gs_main[2])
    fleets = ["Container", "Crude tanker"]
    shortest = [metrics.loc["shortest_rolling", "top1_end_to_end"] * 100,
                crude.loc["shortest_rolling", "top1_end_to_end"] * 100]
    learned = [metrics.loc["learned_global_rolling", "top1_end_to_end"] * 100,
               crude.loc["learned_global_rolling", "top1_end_to_end"] * 100]
    x_c = np.arange(2)
    w_c = 0.34
    bc1 = ax_c.bar(x_c - w_c / 2, shortest, w_c, color=COLOR_GREY,
                   edgecolor="#0F172A", linewidth=0.6, alpha=0.9, label="Shortest route")
    bc2 = ax_c.bar(x_c + w_c / 2, learned, w_c, color=COLOR_NAVY,
                   edgecolor="#0F172A", linewidth=0.6, alpha=0.92, label="Learned global")

    for bars in (bc1, bc2):
        for bar in bars:
            h = bar.get_height()
            ax_c.text(bar.get_x() + bar.get_width() / 2, h + 1.0,
                      f"{h:.1f}%", ha="center", va="bottom", fontsize=8.8,
                      fontweight="bold", color="#1E293B")

    ax_c.set_xticks(x_c, fleets, fontsize=9.5)
    ax_c.set_ylim(0, 75)
    ax_c.set_ylabel("End-to-end Top-1 (%)", fontsize=10.0)
    ax_c.set_title("(c) Cross-fleet parameter transfer", fontsize=10.5,
                   fontweight="bold", loc="left", pad=8)
    ax_c.grid(axis="y", linestyle=":", alpha=0.5, color="#CBD5E1")
    ax_c.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper right")

    save(fig, "fig_dcrank_searoute_diagnostics_v8")


def main():
    main_performance()
    diagnostics()
    print("Saved v8 sea-route figures successfully.")


if __name__ == "__main__":
    main()
