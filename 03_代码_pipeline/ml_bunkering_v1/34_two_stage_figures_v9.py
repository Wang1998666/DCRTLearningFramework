"""Publication figures for the current two-stage v9 system.

Publication-quality aesthetic refactor with Times New Roman, unified palette,
and complete elimination of legend occlusions over data labels.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

COLOR_NAVY = "#1E40AF"     # Top-1 / Proposed / Updated
COLOR_GREEN = "#059669"    # Top-5 / Secondary metric
COLOR_GREY = "#64748B"     # Normal frozen / Baseline
COLOR_CRIMSON = "#DC2626"  # Red Sea transfer highlight


def save(fig, stem: str):
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    if LATEX_FIG.exists():
        fig.savefig(LATEX_FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(LATEX_FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}")


def main():
    occurrence = pd.read_csv(OUT / "two_stage_occurrence_metrics_v9.csv")
    port = pd.read_csv(OUT / "redsea_transfer_dcrank_metrics_v9.csv")
    system = pd.read_csv(OUT / "two_stage_system_metrics_v9.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))

    # --- (a) Stage 1: occurrence PR-AUC ---
    h = occurrence[occurrence.split.eq("hormuz_post")].set_index("model")
    names = ["normal_frozen", "hormuz_pre_updated", "redsea_plus_hormuz_pre"]
    labels = ["Normal\nfrozen", "Hormuz-pre\nupdated", "+ Red Sea\nexperience"]
    x = np.arange(3)
    values = h.loc[names, "pr_auc"].to_numpy() * 100

    bars_a = axes[0].bar(x, values, width=0.55,
                         color=[COLOR_GREY, COLOR_NAVY, COLOR_CRIMSON],
                         edgecolor="#0F172A", linewidth=0.6, alpha=0.92)
    for bar in bars_a:
        v = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.8,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=8.8,
                     fontweight="bold", color="#1E293B")

    axes[0].set_xticks(x, labels, fontsize=9.2)
    axes[0].set_ylim(0, 58)
    axes[0].set_ylabel("PR-AUC (%)", fontsize=10.0)
    axes[0].set_title("(a) Stage 1: Refuelling occurrence", fontsize=10.5,
                      fontweight="bold", loc="left", pad=8)
    axes[0].grid(axis="y", linestyle=":", alpha=0.5, color="#CBD5E1")

    # --- (b) Stage 2: conditional port ranking ---
    q = port[port.split.eq("hormuz_post")].set_index("model")
    names2 = ["normal_v8", "redsea_adapted_v9"]
    labels2 = ["Normal v8", "Red Sea adapted"]
    width = 0.32

    for j, (col, lab, color) in enumerate([
        ("top1_end_to_end", "Top-1", COLOR_NAVY),
        ("top5_end_to_end", "Top-5", COLOR_GREEN),
    ]):
        vals = q.loc[names2, col].to_numpy() * 100
        bars_b = axes[1].bar(np.arange(2) + (j - 0.5) * width, vals, width,
                             label=lab, color=color, edgecolor="#0F172A",
                             linewidth=0.6, alpha=0.9)
        for bar in bars_b:
            v = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, v + 1.2,
                         f"{v:.1f}%", ha="center", va="bottom", fontsize=8.8,
                         fontweight="bold", color="#1E293B")

    axes[1].set_xticks(np.arange(2), labels2, fontsize=9.5)
    # Give headroom so the legend at upper left has ZERO overlap with bar labels
    axes[1].set_ylim(0, 118)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10.0)
    axes[1].set_title("(b) Stage 2: Conditional port ranking", fontsize=10.5,
                      fontweight="bold", loc="left", pad=8)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="#E2E8F0",
                   framealpha=0.95, loc="upper left", ncol=2)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5, color="#CBD5E1")

    # --- (c) Linked two-stage system ---
    s = system.set_index("stage1_model")
    snames = ["hormuz_pre_updated", "redsea_plus_hormuz_pre"]
    slabels = ["No transfer", "Red Sea transfer"]

    for j, (col, lab, color) in enumerate([
        ("system_positive_top1", "Detected + Top-1", COLOR_NAVY),
        ("system_positive_top5", "Detected + Top-5", COLOR_GREEN),
    ]):
        vals = s.loc[snames, col].to_numpy() * 100
        bars_c = axes[2].bar(np.arange(2) + (j - 0.5) * width, vals, width,
                             label=lab, color=color, edgecolor="#0F172A",
                             linewidth=0.6, alpha=0.9)
        for bar in bars_c:
            v = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width() / 2, v + 0.8,
                         f"{v:.1f}%", ha="center", va="bottom", fontsize=8.8,
                         fontweight="bold", color="#1E293B")

    axes[2].set_xticks(np.arange(2), slabels, fontsize=9.5)
    axes[2].set_ylim(0, 62)
    axes[2].set_ylabel("Share of all refuelling events (%)", fontsize=10.0)
    axes[2].set_title("(c) Linked two-stage system", fontsize=10.5,
                      fontweight="bold", loc="left", pad=8)
    axes[2].legend(frameon=True, facecolor="white", edgecolor="#E2E8F0",
                   framealpha=0.95, loc="upper left", ncol=1, fontsize=8.2)
    axes[2].grid(axis="y", linestyle=":", alpha=0.5, color="#CBD5E1")

    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_two_stage_system_v9")


if __name__ == "__main__":
    main()
