"""Reviewer-requested control experiments for the two-stage manuscript (v9).

Three controls that close gaps flagged in the GLM-R1 review:

1. Great-circle substitution.  Holding the trained five-seed utility and the
   2023-calibrated global budget fixed, the great-circle detour replaces the
   AIS sea-lane shortest-path detour everywhere it is used (candidate
   retrieval and the ranking score), on the same 23,296 blind-test events.
2. Oracle injection.  The candidate set is augmented with the realized port
   whenever retrieval missed it, quantifying the optimism that oracle-based
   choice experiments buy.
3. Stage-1 cold start.  Occurrence accuracy is split by whether the
   origin-destination pair (or the vessel) was observed in 2021-2022 training.

The script first rebuilds the published sea-lane frame and verifies that the
reported metrics are reproduced exactly before reporting the two controls.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

HERE = Path(__file__).parent
PROJECT = HERE.resolve().parents[1]
OUT = PROJECT / "02_数据_output" / "ml_bunkering_v1"
CORE = PROJECT / "02_数据_output" / "refuel_panel_v2"

PANEL_COLS = ["mmsi", "legStartTime", "legStartPortCode", "legEndPortCode",
              "berthDuration", "call_time", "area_30", "refuel_area_30", "year"]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E = load_module("e2e_ctrl", "12_end_to_end_port_ranker.py")
S = load_module("searoute_ctrl", "27_dcrank_searoute_repaired.py")
# Module 27 carries its own instances of 08/10 with the shared AREA_ATTR state
# that resources() populates, so all ranker state must be reached through S.
D, R = S.D, S.R


class UtilityLinear(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def utility(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)


def build_test_inputs():
    """Rebuild the blind-test event frame exactly as the v8 pipeline does."""
    code_area, geo, vessels, mats = S.resources()
    raw26 = pd.read_parquet(OUT / "hormuz_container_legs_2026.parquet")
    raw26.mmsi = pd.to_numeric(raw26.mmsi, errors="coerce").astype("Int64")
    raw26.legStartTime = pd.to_datetime(raw26.legStartTime)
    raw26.legEndTime = pd.to_datetime(raw26.legEndTime)
    raw26 = S.canonicalize(raw26)
    pre = raw26.legStartTime.between("2026-01-23", "2026-02-27 23:59:59")
    exposed = set(raw26.loc[pre & raw26.legCrossNodeList.astype(str).str.contains(
        "HORMUZ", case=False, na=False), "mmsi"].dropna())
    test_ev, initial, selected = S.event_frame(
        raw26, code_area, "2026-02-28", "2026-07-22 23:59:59", "HT", exposure=exposed)

    panel = pd.read_parquet(CORE / "legs_panel_v2_areas.parquet", columns=PANEL_COLS)
    panel.mmsi = pd.to_numeric(panel.mmsi, errors="coerce").astype("Int64")
    hist_lag = D.historical_lags(panel)
    train_lag = hist_lag[hist_lag.week.lt("2023-01-01")]
    fallback = train_lag.groupby("area")[S.LAG_COLS].median().to_dict()
    lag26 = D.hormuz_lags(raw26, code_area)
    habitual = R.history_maps(panel)[2]
    return code_area, geo, mats, test_ev, initial, selected, lag26, fallback, habitual


def candidate_frames(test_ev, geo, mats):
    """One long row per event/candidate for three candidate-set definitions.

    Frames share lag features; each frame keeps its own detour column:
    ``sea_detour_knm`` (AIS sea-lane) and ``gc_detour_knm`` (great circle).
    """
    code_node, areas, area_node = S.node_maps(geo, mats)
    matrix = mats[4]  # HORMUZ-stressed sea-lane network
    attr = R.AREA_ATTR
    area_lat = attr.lat.to_numpy()
    area_lon = attr.lon.to_numpy()
    true_index = {a: i for i, a in enumerate(areas)}
    rows, meta_rows = [], []
    for ev in test_ev.itertuples():
        oc, dc = str(ev.legStartPortCode), str(ev.legEndPortCode)
        if oc not in code_node or dc not in code_node or ev.actual not in true_index:
            continue
        oi, di = code_node[oc], code_node[dc]
        o_lat, o_lon = float(geo.loc[oc, "lat"]), float(geo.loc[oc, "lon"])
        d_lat, d_lon = float(geo.loc[dc, "lat"]), float(geo.loc[dc, "lon"])
        direct_km = float(matrix[oi, di])
        if not np.isfinite(direct_km):
            continue
        via_km = matrix[oi, area_node] + matrix[area_node, di]
        valid = np.isfinite(via_km)
        detour_nm = np.maximum(via_km - direct_km, 0) / 1.852
        feasible = np.flatnonzero(valid & (detour_nm <= S.MAX_RETRIEVAL_DETOUR_NM))
        order = feasible[np.lexsort((areas[feasible], detour_nm[feasible]))]

        gc_direct = float(R.gc_nm(o_lat, o_lon, d_lat, d_lon))
        gc_via = (R.gc_nm(o_lat, o_lon, area_lat, area_lon)
                  + R.gc_nm(area_lat, area_lon, d_lat, d_lon))
        gc_detour_nm = np.maximum(np.asarray(gc_via, dtype=float) - gc_direct, 0)
        gc_feasible = np.flatnonzero(valid & (gc_detour_nm <= S.MAX_RETRIEVAL_DETOUR_NM))
        gc_order = gc_feasible[np.lexsort((areas[gc_feasible], gc_detour_nm[gc_feasible]))]

        sea_set = set(order[:S.CAP].tolist())
        gc_set = set(gc_order[:S.CAP].tolist())
        oracle_set = sea_set | {true_index[ev.actual]}
        base = {"event_id": ev.event_id, "mmsi": ev.mmsi, "call_time": ev.legStartTime,
                "legStartPortCode": oc, "legEndPortCode": dc, "actual": ev.actual,
                "exposed": ev.exposed}
        for j in sorted(sea_set | gc_set | oracle_set):
            row = dict(base)
            row.update({
                "alt": areas[j], "y": int(areas[j] == ev.actual),
                "sea_detour_knm": float(detour_nm[j] / 1000),
                "gc_detour_knm": float(gc_detour_nm[j] / 1000),
                "direct_sea_nm": float(direct_km / 1.852),
                "in_sea": int(j in sea_set), "in_gc": int(j in gc_set),
                "in_oracle": int(j in oracle_set)})
            rows.append(row)
        meta_rows.append({"event_id": ev.event_id, "mmsi": ev.mmsi,
                          "actual": ev.actual, "exposed": ev.exposed,
                          "sea_recall": int(true_index[ev.actual] in sea_set),
                          "gc_recall": int(true_index[ev.actual] in gc_set)})
    long = pd.DataFrame(rows)
    meta = pd.DataFrame(meta_rows)
    frames = {"sealane": long[long.in_sea.eq(1)].copy(),
              "greatcircle": long[long.in_gc.eq(1)].copy(),
              "oracle": long[long.in_oracle.eq(1)].copy()}
    long = pd.concat([v.assign(frame=k) for k, v in frames.items()], ignore_index=True)
    return long, meta


def main():
    print("Rebuilding blind-test inputs...", flush=True)
    (code_area, geo, mats, test_ev, initial, selected,
     lag26, fallback, habitual) = build_test_inputs()
    print(f"source events={initial:,}; eligible events={selected:,}", flush=True)
    long, meta = candidate_frames(test_ev, geo, mats)
    long = S.add_lags_and_frozen(long, lag26, fallback)
    print(f"candidate rows={len(long):,}; events={long.event_id.nunique():,}", flush=True)

    ckpt = torch.load(OUT / "dcrank_searoute_model_v8.pt", map_location="cpu",
                      weights_only=False)
    models = []
    for state in ckpt["states"]:
        m = UtilityLinear(len(ckpt["features"]))
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    mean, std = np.asarray(ckpt["mean"]), np.asarray(ckpt["std"])
    b_knm = float(ckpt["budget_knm"])
    x = np.nan_to_num((long[ckpt["features"]].to_numpy(np.float32) - mean) / std).astype(np.float32)
    with torch.no_grad():
        utility = np.mean([m.utility(torch.from_numpy(x)).numpy() for m in models], axis=0)
    long = long.assign(utility=utility)

    rows, details = [], {}
    for frame_name, detour_col in [("sealane", "sea_detour_knm"),
                                   ("greatcircle", "gc_detour_knm"),
                                   ("oracle", "sea_detour_knm")]:
        f = long[long.frame.eq(frame_name)].copy()
        detour = f[detour_col].to_numpy()
        for label, score in [("shortest", -detour),
                             ("dcrank", -detour + b_knm * f.utility.to_numpy())]:
            metric, detail = E.end_to_end_metrics(
                f"{frame_name}_{label}", score, f, meta, habitual)
            rows.append({"frame": frame_name, "model": label, **metric})
            details[f"{frame_name}_{label}"] = detail
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "stage2_metric_oracle_controls_v9.csv", index=False)
    print(table[["frame", "model", "events", "candidate_recall", "top1_end_to_end",
                 "top3_end_to_end", "top5_end_to_end", "mrr_end_to_end",
                 "conditional_top1", "conditional_top5"]].to_string(index=False), flush=True)

    pub = table[table.model.eq("sealane_dcrank")].iloc[0]
    expected = {"top1_end_to_end": 0.6266311813186813, "top5_end_to_end": 0.9312757554945055,
                "mrr_end_to_end": 0.7536799794980792,
                "candidate_recall": 0.9843320741758241}
    checks = {k: bool(abs(float(pub[k]) - v) < 1e-9) for k, v in expected.items()}
    print("reproduction checks:", checks, flush=True)
    assert all(checks.values()), "sea-lane frame does not reproduce the published metrics"

    comparisons = {
        "gc_dcrank_vs_sealane_dcrank": E.paired_cluster_bootstrap(
            details["sealane_dcrank"], details["greatcircle_dcrank"]),
        "gc_dcrank_vs_gc_shortest": E.paired_cluster_bootstrap(
            details["greatcircle_shortest"], details["greatcircle_dcrank"]),
        "oracle_dcrank_vs_sealane_dcrank": E.paired_cluster_bootstrap(
            details["sealane_dcrank"], details["oracle_dcrank"]),
        "gc_shortest_vs_sealane_shortest": E.paired_cluster_bootstrap(
            details["sealane_shortest"], details["greatcircle_shortest"]),
    }
    for name, comp in comparisons.items():
        print(name, json.dumps(comp, indent=1), flush=True)

    # Rank distribution of the realized port among events where sea-lane
    # retrieval missed it and oracle injection appended it.
    injected = details["oracle_dcrank"].merge(
        meta[["event_id", "sea_recall"]], on="event_id", how="left")
    missed = injected[injected.sea_recall.eq(0)]
    dist = missed["rank"].value_counts().sort_index()
    print("injected-port rank distribution over retrieval failures:",
          dist.head(15).to_dict(), flush=True)
    print("failure events:", len(missed),
          "| rank<=1:", int(missed["rank"].le(1).sum()),
          "| rank<=5:", int(missed["rank"].le(5).sum()),
          "| rank<=40:", int(missed["rank"].le(40).sum()),
          "| mean rank:", float(missed["rank"].mean()), flush=True)

    feats = pd.read_parquet(OUT / "two_stage_occurrence_features_v9.parquet",
                            columns=["mmsi", "legStartTime", "origin", "dest", "y"])
    feats.legStartTime = pd.to_datetime(feats.legStartTime)
    train = feats[feats.legStartTime.between("2021-01-01", "2022-12-31 23:59:59")]
    seen_od = set(zip(train.origin.astype(str), train.dest.astype(str)))
    seen_ship = set(train.mmsi.dropna())
    pred = pd.read_parquet(OUT / "two_stage_occurrence_predictions_v9.parquet")
    pred["origin"] = pred.origin.astype(str)
    pred["dest"] = pred.dest.astype(str)
    pred["seen_od"] = [od in seen_od for od in zip(pred.origin, pred.dest)]
    pred["seen_vessel"] = pred.mmsi.isin(seen_ship)
    cold_rows = []
    for label, mask in [("all", pred.y.notna()),
                        ("od_seen", pred.seen_od),
                        ("od_unseen", ~pred.seen_od),
                        ("vessel_known", pred.seen_vessel),
                        ("vessel_new", ~pred.seen_vessel)]:
        z = pred[mask]
        for model in ["normal_frozen", "hormuz_pre_updated", "redsea_plus_hormuz_pre"]:
            p = z["p_refuel_" + model].to_numpy()
            cold_rows.append({"subset": label, "rows": int(mask.sum()),
                              "refuel_rate": float(z.y.mean()), "model": model,
                              "roc_auc": float(roc_auc_score(z.y, p)),
                              "pr_auc": float(average_precision_score(z.y, p)),
                              "log_loss": float(log_loss(z.y, p))})
    cold = pd.DataFrame(cold_rows)
    cold.to_csv(OUT / "stage1_coldstart_od_vessel_v9.csv", index=False)
    print(cold.to_string(index=False), flush=True)

    summary = {
        "control_metrics": table.to_dict("records"),
        "paired_vessel_bootstrap": comparisons,
        "injected_rank_distribution": {str(k): int(v) for k, v in dist.items()},
        "injected_failure_events": int(len(missed)),
        "injected_rank_le1": int(missed["rank"].le(1).sum()),
        "injected_rank_le5": int(missed["rank"].le(5).sum()),
        "injected_rank_mean": float(missed["rank"].mean()),
        "reproduction_checks": checks,
        "source_events": int(initial), "eligible_events": int(selected),
        "budget_knm": b_knm,
        "coldstart": cold.to_dict("records"),
        "protocol": {
            "greatcircle": "great-circle detour replaces sea-lane detour in retrieval and scoring; trained utility and calibrated budget unchanged",
            "oracle": "realized port appended to the top-40 sea-lane candidate set when retrieval missed it",
            "coldstart": "occurrence accuracy split by OD-pair and vessel visibility in 2021-2022 training data",
        },
    }
    (OUT / "reviewer_controls_report_v9.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved stage2_metric_oracle_controls_v9.csv, stage1_coldstart_od_vessel_v9.csv, "
          "reviewer_controls_report_v9.json", flush=True)


if __name__ == "__main__":
    main()
