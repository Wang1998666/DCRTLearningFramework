"""Standard machine-learning diagnostic figures for the two-stage v9 system.

Every plotted value is computed here from saved result files; nothing is
hand-entered. Outputs (PNG 300 dpi + PDF) go to
``04_图表_figures/ml_bunkering_v1`` and are synced to ``05_手稿/Latex_ML``.

Figures:
fig_stage1_roc_pr_v9            ROC and Precision-Recall curves, three Stage-1 models
fig_stage1_calibration_v9       reliability diagram + side-by-side confusion matrices
fig_stage1_permutation_v9       permutation feature importance (PR-AUC drop)
fig_stage2_topk_curve_v9        recall@k curves, Stage-2 rankers (container + crude)
fig_stage2_utility_weights_v9   learned utility weights and marginal rates of substitution
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

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

# Harmonious paper-wide color palette
COLOR_NAVY = "#1E40AF"     # Primary / Hormuz-Pre / Learned
COLOR_GREEN = "#059669"    # Crude / Secondary / Top-5
COLOR_GREY = "#64748B"     # Normal frozen / Baseline
COLOR_CRIMSON = "#DC2626"  # Red Sea transfer / Highlight
COLOR_AMBER = "#D97706"    # Top-3 / Trade-off

MODEL_COLORS = {
    "normal_frozen": COLOR_GREY,
    "hormuz_pre_updated": COLOR_NAVY,
    "redsea_plus_hormuz_pre": COLOR_CRIMSON,
}
MODEL_LABELS = {
    "normal_frozen": "Normal frozen",
    "hormuz_pre_updated": "Hormuz-Pre updated",
    "redsea_plus_hormuz_pre": "Red Sea + Hormuz-Pre",
}


def save(fig, stem: str):
    """Save to both figure directory and LaTeX manuscript folder."""
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    if LATEX_FIG.exists():
        fig.savefig(LATEX_FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(LATEX_FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}")


# ---------------------------------------------------------------- Stage 1 curves
def stage1_roc_pr():
    pred = pd.read_parquet(OUT / "two_stage_occurrence_predictions_v9.parquet")
    y = pred.y.to_numpy(int)
    prevalence = float(y.mean())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.3))

    for model, color in MODEL_COLORS.items():
        p = pred["p_refuel_" + model].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        auc = np.trapezoid(tpr, fpr)
        ax1.plot(fpr * 100, tpr * 100, color=color, lw=1.9,
                 label=f"{MODEL_LABELS[model]} (AUC = {auc:.3f})")

        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax2.plot(rec * 100, prec * 100, color=color, lw=1.9,
                 label=f"{MODEL_LABELS[model]} (AP = {ap:.3f})")

    ax1.plot([0, 100], [0, 100], color="#94A3B8", lw=1.2, ls="--", label="Random guess")
    ax1.set_xlabel("False positive rate (%)", fontsize=10.0)
    ax1.set_ylabel("True positive rate (%)", fontsize=10.0)
    ax1.set_title("(a) Receiver operating characteristic", fontsize=10.5,
                  fontweight="bold", loc="left", pad=8)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.grid(linestyle=":", alpha=0.5, color="#CBD5E1")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0",
               framealpha=0.95, loc="lower right")

    ax2.axhline(prevalence * 100, color="#94A3B8", lw=1.2, ls="--",
                label=f"No-skill rate ({prevalence * 100:.2f}%)")
    ax2.set_xlabel("Recall (%)", fontsize=10.0)
    ax2.set_ylabel("Precision (%)", fontsize=10.0)
    ax2.set_title("(b) Precision-recall curves", fontsize=10.5,
                  fontweight="bold", loc="left", pad=8)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.grid(linestyle=":", alpha=0.5, color="#CBD5E1")
    ax2.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0",
               framealpha=0.95, loc="upper right")

    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_stage1_roc_pr_v9")


# ------------------------------------------------------- calibration + confusion
def stage1_calibration():
    pred = pd.read_parquet(OUT / "two_stage_occurrence_predictions_v9.parquet")
    system = json.loads((OUT / "two_stage_system_report_v9.json").read_text(encoding="utf-8"))
    y = pred.y.to_numpy(int)

    # Clean layout: Left is reliability diagram; Right is two side-by-side confusion matrices with shared colorbar
    fig = plt.figure(figsize=(13.2, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 0.92, 0.92, 0.05], wspace=0.32)

    ax1 = fig.add_subplot(gs[0])
    ax_cm1 = fig.add_subplot(gs[1])
    ax_cm2 = fig.add_subplot(gs[2])
    cax = fig.add_subplot(gs[3])

    # (a) Reliability diagram
    bins = np.quantile(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
    for model, color in MODEL_COLORS.items():
        p = pred["p_refuel_" + model].to_numpy()
        idx = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() < 50:
                continue
            xs.append(p[m].mean() * 100)
            ys.append(y[m].mean() * 100)
        ax1.plot(xs, ys, "o-", color=color, ms=5.0, lw=1.8, label=MODEL_LABELS[model])

    ax1.plot([0, 100], [0, 100], color="#94A3B8", lw=1.2, ls="--", label="Perfect calibration")
    ax1.set_xlabel("Mean predicted refuelling probability (%)", fontsize=10.0)
    ax1.set_ylabel("Observed refuelling rate (%)", fontsize=10.0)
    ax1.set_title("(a) Reliability diagram (deciles)", fontsize=10.5,
                  fontweight="bold", loc="left", pad=8)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.grid(linestyle=":", alpha=0.5, color="#CBD5E1")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0",
               framealpha=0.95, loc="upper left")

    # (b) Confusion matrices side-by-side (NO overlapping insets!)
    cfgs = [
        ("hormuz_pre_updated", "Without transfer", ax_cm1, "(b) Without transfer"),
        ("redsea_plus_hormuz_pre", "Red Sea transfer", ax_cm2, "With Red Sea transfer"),
    ]
    mats = []
    for name, _, _, _ in cfgs:
        row = next(r for r in system["metrics"] if r["stage1_model"] == name)
        theta = row["pre_hormuz_threshold"]
        p = pred["p_refuel_" + name].to_numpy()
        pos = p >= theta
        tp = int((pos & (y == 1)).sum())
        fp = int((pos & (y == 0)).sum())
        fn = int((~pos & (y == 1)).sum())
        tn = int((~pos & (y == 0)).sum())
        mats.append((theta, np.array([[tn, fp], [fn, tp]])))

    vmax = max(m[1].max() for m in mats)

    for k, (name, label_txt, ax, title_prefix) in enumerate(cfgs):
        theta, mat = mats[k]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)

        for (i, j), v in np.ndenumerate(mat):
            # Dynamic text contrast: white text on dark background
            txt_color = "white" if v > vmax * 0.45 else "#0F172A"
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color=txt_color)

        ax.set_xticks([0, 1], ["Pred. 0", "Pred. 1"], fontsize=9.0)
        if k == 0:
            ax.set_yticks([0, 1], ["Actual 0", "Actual 1"], fontsize=9.0)
            ax.set_title(f"{title_prefix}\n($\\theta^* = {theta:.4f}$)",
                         fontsize=10.0, fontweight="bold", pad=8)
        else:
            ax.set_yticks([0, 1], ["", ""], fontsize=9.0)
            ax.set_title(f"{title_prefix}\n($\\theta^* = {theta:.4f}$)",
                         fontsize=10.0, fontweight="bold", pad=8)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Voyage count", fontsize=9.0)
    cb.ax.tick_params(labelsize=8.5)

    save(fig, "fig_stage1_calibration_v9")


# ------------------------------------------------------ permutation importance
FEATURE_PRETTY = {
    "od_rate": "O–D pair refuel base rate",
    "teu": "Nominal vessel capacity (TEU)",
    "dest_rate": "Destination port base rate",
    "berthDuration": "Berth duration (hours)",
    "days_since_refuel": "Days since last bunker call",
    "vessel_rate": "Vessel historical refuel rate",
    "prior10_rate": "Prior 10-leg refuel frequency",
    "origin_rate": "Origin port base rate",
    "operator_rate": "Carrier operator base rate",
    "power_kw_total": "Total engine power (kW)",
    "nm_since_refuel": "Distance since last bunker (nm)",
    "cargoLoad": "Cargo load (draft ratio)",
    "averageSpeed": "Average voyage speed (knots)",
    "isFullLoad": "Full load indicator",
    "ship_age": "Vessel age (years)",
    "month_cos": "Departure month (cosine)",
    "prior3_rate": "Prior 3-leg refuel frequency",
    "prior_y": "Prior leg refuel indicator",
    "month_sin": "Departure month (sine)",
}


def stage1_permutation(repeats=3):
    import joblib
    bundle = joblib.load(OUT / "two_stage_occurrence_models_v9.joblib")
    model, features = bundle["transfer_model"], bundle["features"]
    feats = pd.read_parquet(OUT / "two_stage_occurrence_features_v9.parquet")
    feats.legStartTime = pd.to_datetime(feats.legStartTime)
    test = feats[feats.legStartTime.between("2026-02-28", "2026-07-22 23:59:59")]
    X = test[features].to_numpy(float)
    y = test.y.to_numpy(int)
    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    rng = np.random.default_rng(20260921)
    drops = np.zeros((len(features), repeats))

    for r in range(repeats):
        for k, f in enumerate(features):
            Xp = X.copy()
            Xp[:, k] = Xp[rng.permutation(len(Xp)), k]
            drops[k, r] = base - average_precision_score(y, model.predict_proba(Xp)[:, 1])

    mean, std = drops.mean(axis=1), drops.std(axis=1)
    order = np.argsort(mean)

    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    # Human-readable professional academic labels (NO raw LaTeX backslash escapes)
    labels = [FEATURE_PRETTY.get(features[i], features[i]) for i in order]

    ax.barh(np.arange(len(features)), mean[order] * 100, xerr=std[order] * 100,
            color=COLOR_NAVY, edgecolor="#0F172A", linewidth=0.6, alpha=0.9,
            capsize=3.5, error_kw={"elinewidth": 0.9, "ecolor": "#334155"})

    ax.set_yticks(np.arange(len(features)), labels, fontsize=9.2)
    ax.set_xlabel("Decrease in PR-AUC when feature is permuted (percentage points)", fontsize=10.0)
    # Standalone title removed per publication requirements (caption in LaTeX)
    ax.grid(axis="x", linestyle=":", alpha=0.5, color="#CBD5E1")
    fig.tight_layout()
    save(fig, "fig_stage1_permutation_v9")


# ------------------------------------------------------------- Stage-2 top-k
class _UtilityLinear(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def utility(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)


def _utility(frame, checkpoint, features):
    x = frame[features].to_numpy(np.float32)
    x = np.nan_to_num((x - np.asarray(checkpoint["mean"])) / np.asarray(checkpoint["std"]))
    xt = torch.from_numpy(x.astype(np.float32))
    vals = []
    for state in checkpoint["states"]:
        m = _UtilityLinear(len(features))
        m.load_state_dict(state)
        m.eval()
        with torch.no_grad():
            vals.append(m.utility(xt).numpy().ravel())
    return np.mean(vals, axis=0)


def _recall_at_k(frame, scores, ks=range(1, 11)):
    z = pd.DataFrame({"event_id": frame.event_id.to_numpy(),
                      "y": frame.y.to_numpy(), "score": scores})
    z["rank"] = z.groupby("event_id").score.rank(method="first", ascending=False)
    r = z.loc[z.y.eq(1), "rank"]
    n = z.event_id.nunique()
    return np.array([(r <= k).sum() / n * 100 for k in ks])


def stage2_topk():
    ck = torch.load(OUT / "dcrank_searoute_model_v8.pt", map_location="cpu", weights_only=False)
    feats = ck["features"]
    b_learned, b_fixed = float(ck["budget_knm"]), 0.05
    con = pd.read_parquet(OUT / "dcrank_searoute_hormuz_candidates_v8.parquet")
    crude = pd.read_parquet(OUT / "dcrank_searoute_crude_candidates_v8.parquet")
    u_con = _utility(con, ck, feats)
    u_crude = _utility(crude, ck, feats)
    det_c = con.sea_detour_knm.to_numpy()
    det_t = crude.sea_detour_knm.to_numpy()
    ks = np.arange(1, 11)

    curves = {
        "Shortest sea detour": (_recall_at_k(con, -det_c), COLOR_GREY, "-", "^"),
        "Fixed 50 nm budget": (_recall_at_k(con, -det_c + b_fixed * u_con), "#2563EB", "-", "s"),
        "Learned global budget": (_recall_at_k(con, -det_c + b_learned * u_con), COLOR_CRIMSON, "-", "o"),
        "Learned budget, crude tankers": (_recall_at_k(crude, -det_t + b_learned * u_crude), COLOR_GREEN, "--", "D"),
    }

    fig, ax = plt.subplots(figsize=(9.8, 4.3))
    offsets = {
        "Shortest sea detour": (8, -8),
        "Fixed 50 nm budget": (-8, 6),
        "Learned global budget": (8, 4),
        "Learned budget, crude tankers": (8, 4),
    }

    for label, (vals, color, ls, marker) in curves.items():
        ax.plot(ks, vals, ls, color=color, lw=2.0, marker=marker, ms=5.0, label=label)
        if label == "Learned global budget":
            ax.annotate(f"{vals[0]:.1f}%", (1, vals[0]), textcoords="offset points",
                        xytext=(8, 3), fontsize=8.5, fontweight="bold", color=color)
        elif label == "Learned budget, crude tankers":
            ax.annotate(f"{vals[0]:.1f}%", (1, vals[0]), textcoords="offset points",
                        xytext=(8, 3), fontsize=8.5, fontweight="bold", color=color)
        elif label == "Shortest sea detour":
            ax.annotate(f"{vals[0]:.1f}%", (1, vals[0]), textcoords="offset points",
                        xytext=(8, -8), fontsize=8.5, fontweight="bold", color=color)

    ax.set_xlabel("Rank cut-off ($k$)", fontsize=10.0)
    ax.set_ylabel("End-to-end Recall@$k$ (%)", fontsize=10.0)
    # Standalone title removed per publication requirements (caption in LaTeX)
    ax.set_xticks(ks)
    ax.set_xlim(0.8, 10.2)
    ax.set_ylim(32, 100)
    ax.grid(linestyle=":", alpha=0.5, color="#CBD5E1")
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0",
              framealpha=0.95, loc="lower right", fontsize=8.8)
    fig.tight_layout()
    save(fig, "fig_stage2_topk_curve_v9")


# ------------------------------------------------------- utility weights / MRS
def stage2_utility_weights():
    w = pd.read_csv(OUT / "dcrank_searoute_weights_v8.csv")
    g = w.groupby("feature").agg(
        weight=("weight", "mean"), weight_sd=("weight", "std"),
        mrs=("mrs_nm_per_unit_at_learned_budget", "mean"),
        mrs_sd=("mrs_nm_per_unit_at_learned_budget", "std")
    )
    pretty = {
        "log_lag4_refuels": "log(1 + bunkering calls)",
        "lag4_refuel_share": "Bunkering share",
        "log_lag4_calls": "log(1 + vessel calls)",
        "lag4_berth_h": "Median berth hours",
    }
    order = g.weight.abs().sort_values().index
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2))

    # Directional colors: Navy for positive (attractive), Crimson for negative (penalizing)
    weights = g.loc[order, "weight"].to_numpy()
    colors_w = [COLOR_NAVY if val >= 0 else COLOR_CRIMSON for val in weights]

    ax1.barh(np.arange(len(g)), weights, xerr=g.loc[order, "weight_sd"],
             color=colors_w, edgecolor="#0F172A", linewidth=0.6, alpha=0.9,
             capsize=3.5, error_kw={"elinewidth": 0.9, "ecolor": "#334155"})
    ax1.axvline(0, color="#475569", lw=0.9)
    ax1.set_yticks(np.arange(len(g)), [pretty[f] for f in order], fontsize=9.2)
    ax1.set_xlabel("Utility weight $w_k$ (standardized)", fontsize=10.0)
    ax1.set_title("(a) Learned utility weights (5-seed ensemble)", fontsize=10.5,
                  fontweight="bold", loc="left", pad=8)
    ax1.grid(axis="x", linestyle=":", alpha=0.5, color="#CBD5E1")

    # Panel (b): Marginal rate of substitution
    mrs_vals = g.loc[order, "mrs"].to_numpy()
    ax2.barh(np.arange(len(g)), mrs_vals, xerr=g.loc[order, "mrs_sd"],
             color=COLOR_AMBER, edgecolor="#0F172A", linewidth=0.6, alpha=0.9,
             capsize=3.5, error_kw={"elinewidth": 0.9, "ecolor": "#334155"})
    ax2.axvline(0, color="#475569", lw=0.9)
    ax2.set_yticks(np.arange(len(g)), ["" for _ in order])
    ax2.set_xlabel("Marginal rate of substitution (nm per unit)", fontsize=10.0)
    ax2.set_title("(b) Implied detour trade-off at $B^*$ (nm)", fontsize=10.5,
                  fontweight="bold", loc="left", pad=8)
    ax2.grid(axis="x", linestyle=":", alpha=0.5, color="#CBD5E1")
    ax2.set_xlim(-5, 175)

    # Annotate top bar (shifted down to avoid error bar collision)
    top_mrs = mrs_vals[-1]
    ax2.text(top_mrs + 4, len(g) - 1 - 0.36, f"{top_mrs:.1f} nm", va="top",
             fontsize=8.8, fontweight="bold", color="#1E293B")

    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_stage2_utility_weights_v9")


def main():
    print("Stage-1 ROC / PR curves ...", flush=True)
    stage1_roc_pr()
    print("Stage-1 calibration + confusion ...", flush=True)
    stage1_calibration()
    print("Stage-1 permutation importance ...", flush=True)
    stage1_permutation()
    print("Stage-2 recall@k curves ...", flush=True)
    stage2_topk()
    print("Stage-2 utility weights / MRS ...", flush=True)
    stage2_utility_weights()
    print("ML diagnostic figures complete and saved.")


if __name__ == "__main__":
    main()
