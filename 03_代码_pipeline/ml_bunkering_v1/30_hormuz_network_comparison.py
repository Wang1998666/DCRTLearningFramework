"""Hold events and trained DCRank fixed; switch only the route network.

This is a predictive route-scenario comparison, not a causal estimate of the
Hormuz closure.  Both scenarios use exactly the same dated refuelling events,
trained weights, lagged port information, candidate cap, and distance budget.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
PROJECT = HERE.resolve().parents[1]
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


def load_repaired():
    spec = importlib.util.spec_from_file_location("dcrank_searoute_v8", HERE / "27_dcrank_searoute_repaired.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_repaired()


def chosen_table(frame, utility, budget, scenario):
    z = frame[["event_id", "mmsi", "actual", "alt", "exposed",
               "sea_detour_knm", "direct_sea_nm"]].copy()
    z["score"] = -z.sea_detour_knm.to_numpy() + budget * utility
    z = z.loc[z.groupby("event_id").score.idxmax()].copy()
    z["correct"] = z.alt.eq(z.actual)
    return z.rename(columns={"alt": f"prediction_{scenario}",
                             "sea_detour_knm": f"chosen_detour_knm_{scenario}",
                             "direct_sea_nm": f"direct_sea_nm_{scenario}",
                             "correct": f"correct_{scenario}"})


def main():
    print("Building the same Hormuz-period events under two route networks...", flush=True)
    code_area, geo, _, mats = M.resources()
    raw = pd.read_parquet(OUT / "hormuz_container_legs_2026.parquet")
    raw.mmsi = pd.to_numeric(raw.mmsi, errors="coerce").astype("Int64")
    raw.legStartTime = pd.to_datetime(raw.legStartTime)
    raw.legEndTime = pd.to_datetime(raw.legEndTime)
    raw = M.canonicalize(raw)
    pre = raw.legStartTime.between("2026-01-23", "2026-02-27 23:59:59")
    exposed = set(raw.loc[pre & raw.legCrossNodeList.astype(str).str.contains(
        "HORMUZ", case=False, na=False), "mmsi"].dropna())
    events, source_events, selected_events = M.event_frame(
        raw, code_area, "2026-02-28", "2026-07-22 23:59:59", "HC", exposure=exposed)

    normal, normal_meta, _ = M.network_candidates(events, geo, mats, "normal", "HC")
    blocked, blocked_meta, _ = M.network_candidates(events, geo, mats, "hormuz", "HC")
    common = sorted(set(normal_meta.event_id).intersection(blocked_meta.event_id))
    normal = normal[normal.event_id.isin(common)].reset_index(drop=True)
    blocked = blocked[blocked.event_id.isin(common)].reset_index(drop=True)
    meta = normal_meta[normal_meta.event_id.isin(common)].sort_values("event_id").reset_index(drop=True)

    panel = pd.read_parquet(M.CORE / "legs_panel_v2_areas.parquet")
    panel.mmsi = pd.to_numeric(panel.mmsi, errors="coerce").astype("Int64")
    panel.legStartTime = pd.to_datetime(panel.legStartTime)
    panel.isRefueled = pd.to_numeric(panel.isRefueled).fillna(0)
    historical_lag = M.D.historical_lags(panel)
    training_lag = historical_lag[historical_lag.week.lt("2023-01-01")]
    fallback = training_lag.groupby("area")[M.LAG_COLS].median().to_dict()
    lag26 = M.D.hormuz_lags(raw, code_area)
    normal = M.add_lags_and_frozen(normal, lag26, fallback, "2026-02-23")
    blocked = M.add_lags_and_frozen(blocked, lag26, fallback, "2026-02-23")
    habitual = M.R.history_maps(panel)[2]

    checkpoint = torch.load(OUT / "dcrank_searoute_model_v8.pt", map_location="cpu", weights_only=False)
    models = []
    for state in checkpoint["states"]:
        model = M.UtilityLinear(len(checkpoint["features"]))
        model.load_state_dict(state); model.eval(); models.append(model)
    budget = float(checkpoint["budget_knm"])
    mean, std = np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"])
    normal_u, _ = M.utilities(models, normal, M.FEATURES, mean, std)
    blocked_u, _ = M.utilities(models, blocked, M.FEATURES, mean, std)

    rows, details = [], {}
    for cohort, event_ids in [
        ("all_events", set(meta.event_id)),
        ("historically_hormuz_exposed", set(meta.loc[meta.exposed.eq(1), "event_id"])),
    ]:
        cohort_meta = meta[meta.event_id.isin(event_ids)]
        for scenario, full_frame, full_utility in [
            ("normal_network", normal, normal_u),
            ("hormuz_blocked_network", blocked, blocked_u),
        ]:
            keep = full_frame.event_id.isin(event_ids).to_numpy()
            frame, utility = full_frame.loc[keep].reset_index(drop=True), full_utility[keep]
            for method, u, b in [
                ("shortest", np.zeros(len(frame)), 0.0),
                ("fixed50", utility, .05),
                ("learned_global", utility, budget),
            ]:
                name = f"{method}_{scenario}_{cohort}"
                metric, detail, _ = M.score_metrics(name, frame, cohort_meta, habitual, u, b)
                rows.append({"cohort": cohort, "scenario": scenario, "method": method, **metric})
                details[name] = detail
    metrics = pd.DataFrame(rows)

    normal_chosen = chosen_table(normal, normal_u, budget, "normal")
    blocked_chosen = chosen_table(blocked, blocked_u, budget, "blocked")
    event_comparison = normal_chosen.merge(
        blocked_chosen.drop(columns=["mmsi", "actual", "exposed"]), on="event_id", how="inner")
    event_comparison["prediction_changed"] = event_comparison.prediction_normal.ne(
        event_comparison.prediction_blocked)
    event_comparison["direct_route_change_nm"] = (
        event_comparison.direct_sea_nm_blocked - event_comparison.direct_sea_nm_normal)

    common_pairs = normal[["event_id", "alt", "sea_detour_knm"]].merge(
        blocked[["event_id", "alt", "sea_detour_knm"]], on=["event_id", "alt"],
        suffixes=("_normal", "_blocked"))
    common_pairs["detour_change_nm"] = 1000 * (
        common_pairs.sea_detour_knm_blocked - common_pairs.sea_detour_knm_normal)

    comparisons = {}
    for cohort in ["all_events", "historically_hormuz_exposed"]:
        comparisons[cohort] = M.E.paired_cluster_bootstrap(
            details[f"learned_global_normal_network_{cohort}"],
            details[f"learned_global_hormuz_blocked_network_{cohort}"])
    summary = {
        "design": "same Hormuz-period events, trained model, lagged features, cap and budget; only route network changes",
        "interpretation": "predictive route-scenario comparison, not a causal estimate",
        "network_source": "TRE-10CHOKE AIS corridor network: baseline versus HORMUZ_STRAIT disruption scenario",
        "source_refuel_events": source_events,
        "events_after_required_fields": selected_events,
        "common_geographically_eligible_events": len(common),
        "learned_budget_nm": budget * 1000,
        "metrics": metrics.to_dict("records"),
        "blocked_minus_normal_bootstrap": comparisons,
        "prediction_changed_share": float(event_comparison.prediction_changed.mean()),
        "direct_route_changed_over_1nm_share": float(event_comparison.direct_route_change_nm.abs().gt(1).mean()),
        "mean_direct_route_change_nm": float(event_comparison.direct_route_change_nm.mean()),
        "common_candidate_pairs": len(common_pairs),
        "candidate_detour_changed_over_1nm_share": float(common_pairs.detour_change_nm.abs().gt(1).mean()),
        "historically_exposed_events": int(event_comparison.exposed.eq(1).sum()),
        "exposed_prediction_changed_share": float(event_comparison.loc[
            event_comparison.exposed.eq(1), "prediction_changed"].mean()),
        "exposed_direct_route_changed_over_1nm_share": float(event_comparison.loc[
            event_comparison.exposed.eq(1), "direct_route_change_nm"].abs().gt(1).mean()),
    }
    metrics.to_csv(OUT / "hormuz_network_scenario_metrics_v8.csv", index=False)
    event_comparison.to_parquet(OUT / "hormuz_network_event_comparison_v8.parquet", index=False)
    (OUT / "hormuz_network_scenario_report_v8.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_figure(metrics)

    print(metrics[["cohort", "scenario", "method", "events", "candidate_recall",
                   "top1_end_to_end", "top3_end_to_end", "top5_end_to_end",
                   "mrr_end_to_end"]].to_string(index=False))
    print(json.dumps({k: summary[k] for k in [
        "common_geographically_eligible_events", "prediction_changed_share",
        "direct_route_changed_over_1nm_share", "mean_direct_route_change_nm",
        "candidate_detour_changed_over_1nm_share", "blocked_minus_normal_bootstrap"]}, indent=2))


def plot_figure(metrics: pd.DataFrame):
    plot = metrics[metrics.method.eq("learned_global") & metrics.cohort.eq("all_events")].set_index("scenario")
    labels = ["Normal network", "Hormuz-blocked network"]
    scenarios = ["normal_network", "hormuz_blocked_network"]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(2)
    width = 0.24

    for j, (column, label, color) in enumerate([
        ("top1_end_to_end", "Top-1", "#1E40AF"),   # Deep Navy
        ("top3_end_to_end", "Top-3", "#D97706"),   # Warm Amber
        ("top5_end_to_end", "Top-5", "#059669"),   # Emerald Green
    ]):
        values = plot.loc[scenarios, column].to_numpy() * 100
        bars = ax.bar(x + (j - 1) * width, values, width, label=label, color=color,
                      edgecolor="#0F172A", linewidth=0.6, alpha=0.92)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8.8,
                    fontweight="bold", color="#1E293B")

    ax.set_xticks(x, labels, fontsize=9.5)
    ax.set_ylabel("End-to-end accuracy (%)", fontsize=10.0)
    ax.set_ylim(0, 110)
    # Standalone title removed per publication requirements (caption in LaTeX)
    ax.grid(axis="y", linestyle=":", alpha=0.5, color="#CBD5E1")
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0",
              framealpha=0.95, loc="upper left", ncol=3)
    fig.tight_layout()

    fig.savefig(FIG / "fig_hormuz_network_scenario_v8.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig_hormuz_network_scenario_v8.pdf", bbox_inches="tight")
    if LATEX_FIG.exists():
        fig.savefig(LATEX_FIG / "fig_hormuz_network_scenario_v8.png", dpi=300, bbox_inches="tight")
        fig.savefig(LATEX_FIG / "fig_hormuz_network_scenario_v8.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_hormuz_network_scenario_v8")


if __name__ == "__main__":
    import sys
    csv_path = OUT / "hormuz_network_scenario_metrics_v8.csv"
    if "--plot-only" in sys.argv and csv_path.exists():
        plot_figure(pd.read_csv(csv_path))
    else:
        main()
