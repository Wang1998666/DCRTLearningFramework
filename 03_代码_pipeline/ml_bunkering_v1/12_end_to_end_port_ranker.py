"""Leakage-resistant, end-to-end bunkering-port retrieval and ranking.

Unlike the v1/v2 experiments, the true port is never injected into the
candidate set.  Candidate recall and ranking quality are therefore reported
separately and jointly.  Every popularity/preference statistic and every
fallback value is learned from 2021-2022 only.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker

HERE = Path(__file__).parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load_module("ranker_v1", "08_shock_port_ranker.py")
D = load_module("dynamic_v2", "10_dynamic_port_ranker.py")
OUT = R.OUT


def setup_training_data():
    """Rebuild v2 training data with train-period-only imputation fallbacks."""
    code_area, geo, cent, vessels = R.resources()
    panel = pd.read_parquet(R.CORE / "legs_panel_v2_areas.parquet")
    panel = panel.merge(vessels, on="mmsi", how="left")
    panel["operator"] = panel.operator.fillna("Unknown")
    panel = panel.sort_values(["mmsi", "legStartTime", "legEndTime"]).reset_index(drop=True)
    panel["event_id"] = panel.index.astype("int64")
    R.VESSEL_PREF, R.ORIGIN_PREF, habitual = R.history_maps(panel)

    choice = pd.read_parquet(R.CORE / "choice_long_v2.parquet")
    train_choice = choice[choice.year.isin([2021, 2022])]
    R.AREA_ATTR = (train_choice.groupby("alt").agg(
        dwell_10h=("dwell_10h", "first"), log_calls=("log_calls", "first"),
        propensity=("propensity", "first"), events=("y", "sum"))
        .join(cent, how="inner").dropna())
    # dynamic_v2 loads its own module instance; keep its shared area metadata aligned.
    D.R.AREA_ATTR = R.AREA_ATTR
    pref = train_choice.groupby(["operator", "alt"]).pref.first().to_dict()
    exposure = pd.read_parquet(
        R.CORE / "red_sea_historical_exposure_v2.parquet",
        columns=["mmsi", "historically_exposed", "error"])
    exposure = exposure[exposure.error.eq("")]
    exposure.mmsi = pd.to_numeric(exposure.mmsi, errors="coerce").astype("Int64")
    normal, red = R.historical_long(panel, vessels, exposure[["mmsi", "historically_exposed"]])
    normal = normal[normal.alt.isin(R.AREA_ATTR.index)].copy()
    red = red[red.alt.isin(R.AREA_ATTR.index)].copy()
    destinations = panel.set_index("event_id").legEndPortCode
    normal["legEndPortCode"] = normal.event_id.map(destinations)
    red["legEndPortCode"] = red.event_id.map(destinations)

    lagh = D.historical_lags(panel)
    train_lagh = lagh[lagh.week.lt("2023-01-01")]
    fallback = train_lagh.groupby("area")[[
        "lag4_calls", "lag4_refuels", "lag4_refuel_share", "lag4_berth_h"
    ]].median().to_dict()
    normal = D.add_lags(normal, lagh, fallback)
    red = D.add_lags(red, lagh, fallback)
    mats = D.graph_matrices(geo)
    normal = D.add_network(normal, "normal", geo, mats)
    red = D.add_network(red, "redsea", geo, mats)
    red_train = red[red.call_time.lt("2024-04-01")]
    red_test = red[red.call_time.ge("2024-04-01")].copy()
    training = pd.concat([normal, red_train], ignore_index=True)
    return code_area, geo, vessels, panel, pref, habitual, training, red_test, fallback, mats


def build_hormuz_candidates(code_area, geo, vessels, pref, caps=(10, 20, 40),
                             thresholds=(500, 1000, 2500, 5000),
                             event_start="2026-02-28", event_end=None, prefix="H3"):
    """Create deployable candidates without consulting the realized port."""
    cols = ["mmsi", "legStartTime", "legEndTime", "legStartPortCode", "legEndPortCode",
            "isRefueled", "refuelPortCodes", "legCrossNodeList", "berthDuration"]
    raw = pd.read_parquet(OUT / "hormuz_container_legs_2026.parquet", columns=cols)
    raw.mmsi = pd.to_numeric(raw.mmsi, errors="coerce").astype("Int64")
    raw.legStartTime = pd.to_datetime(raw.legStartTime)
    raw = raw.merge(vessels, on="mmsi", how="left")
    raw["operator"] = raw.operator.fillna("Unknown")
    raw = raw.sort_values(["mmsi", "legStartTime", "legEndTime"]).drop_duplicates(
        ["mmsi", "legStartTime", "legEndTime"], keep="last")
    pre = raw.legStartTime.lt("2026-02-28") & raw.legCrossNodeList.astype(str).str.contains(
        "HORMUZ", case=False, na=False)
    exposed = set(raw.loc[pre, "mmsi"].dropna())
    event_mask = raw.isRefueled.eq(1) & raw.legStartTime.ge(event_start)
    if event_end is not None:
        event_mask &= raw.legStartTime.le(event_end)
    events = raw[event_mask].copy()
    events["actual_code"] = events.refuelPortCodes.astype(str).str.split(",").str[0]
    events["actual"] = events.actual_code.map(code_area)
    events = events[
        events.actual.isin(R.AREA_ATTR.index)
        & events.legStartPortCode.astype(str).isin(geo.index)
        & events.legEndPortCode.astype(str).isin(geo.index)
    ].copy()
    events["exposed"] = events.mmsi.isin(exposed).astype("int8")
    events["rand"] = np.random.default_rng(20260921).random(len(events))
    events = events.sort_values("rand").groupby("exposed", group_keys=False).head(3500).reset_index(drop=True)
    events["event_id"] = [f"{prefix}_{i}" for i in range(len(events))]

    names = R.AREA_ATTR.index.to_numpy()
    popularity = R.AREA_ATTR.events.to_dict()
    rows = []
    sensitivity = []
    for event in events.itertuples():
        origin = geo.loc[str(event.legStartPortCode)]
        destination = geo.loc[str(event.legEndPortCode)]
        direct = R.gc_nm(origin.lat, origin.lon, destination.lat, destination.lon)
        route = (R.gc_nm(origin.lat, origin.lon, R.AREA_ATTR.lat, R.AREA_ATTR.lon)
                 + R.gc_nm(R.AREA_ATTR.lat, R.AREA_ATTR.lon, destination.lat, destination.lon))
        detour_nm = np.asarray(route - direct)
        priority = np.array([
            2.0 * np.log1p(R.VESSEL_PREF.get((event.mmsi, area), 0))
            + 1.5 * np.log1p(R.ORIGIN_PREF.get((event.legStartPortCode, area), 0))
            + float(pref.get((event.operator, area), 0.0))
            + 0.35 * np.log1p(popularity.get(area, 0))
            - 0.08 * max(detour_nm[j], 0) / 1000
            for j, area in enumerate(names)
        ])
        for threshold in thresholds:
            feasible = np.flatnonzero((detour_nm >= -50) & (detour_nm <= threshold / 1.852))
            ordered = feasible[np.argsort(-priority[feasible])]
            for cap in caps:
                selected = ordered[:cap]
                sensitivity.append({
                    "event_id": event.event_id, "threshold_km": threshold, "cap": cap,
                    "candidate_count": len(selected),
                    "actual_recalled": int(event.actual in set(names[selected]))
                })
        feasible = np.flatnonzero((detour_nm >= -50) & (detour_nm <= 2500 / 1.852))
        selected = feasible[np.argsort(-priority[feasible])][:40]
        for j in selected:
            area = names[j]
            rows.append({
                "event_id": event.event_id, "mmsi": event.mmsi, "call_time": event.legStartTime,
                "operator": event.operator, "legStartPortCode": event.legStartPortCode,
                "legEndPortCode": event.legEndPortCode, "actual": event.actual, "alt": area,
                "y": int(area == event.actual), "detour_knm": float(max(detour_nm[j], 0) / 1000),
                "dwell_10h": R.AREA_ATTR.loc[area, "dwell_10h"],
                "log_calls": R.AREA_ATTR.loc[area, "log_calls"],
                "propensity": R.AREA_ATTR.loc[area, "propensity"],
                "pref": pref.get((event.operator, area), 0.0), "exposed": event.exposed,
                "days_since_shock": (event.legStartTime - pd.Timestamp("2026-02-28")).total_seconds() / 86400,
                "vessel_pref_count": np.log1p(R.VESSEL_PREF.get((event.mmsi, area), 0)),
                "origin_pref_count": np.log1p(R.ORIGIN_PREF.get((event.legStartPortCode, area), 0)),
                "candidate_priority": priority[j], "train_popularity": np.log1p(popularity.get(area, 0)),
            })
    long = R.shock_features(pd.DataFrame(rows), "hormuz")
    meta = events[["event_id", "mmsi", "actual", "exposed"]].copy()
    return raw, long, meta, pd.DataFrame(sensitivity)


def end_to_end_metrics(name, scores, long, meta, habitual):
    z = long[["event_id", "mmsi", "alt", "actual", "y"]].copy()
    z["score"] = np.asarray(scores)
    z["rank"] = z.groupby("event_id").score.rank(method="first", ascending=False)
    chosen = z[z.y.eq(1)][["event_id", "rank"]]
    event = meta.merge(chosen, on="event_id", how="left")
    pred = z.loc[z.groupby("event_id").score.idxmax(), ["event_id", "alt"]].rename(columns={"alt": "pred"})
    event = event.merge(pred, on="event_id", how="left")
    event["habitual"] = event.mmsi.map(habitual)
    valid = event.habitual.notna() & event.pred.notna()
    actual_change = event.loc[valid, "actual"].ne(event.loc[valid, "habitual"])
    pred_change = event.loc[valid, "pred"].ne(event.loc[valid, "habitual"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_change, pred_change, average="binary", zero_division=0)
    rank = event["rank"]
    return {
        "model": name, "events": len(event), "candidate_recall": float(rank.notna().mean()),
        "top1_end_to_end": float(rank.le(1).mean()), "top3_end_to_end": float(rank.le(3).mean()),
        "top5_end_to_end": float(rank.le(5).mean()),
        "mrr_end_to_end": float((1 / rank).fillna(0).mean()),
        "conditional_top1": float(rank.dropna().le(1).mean()),
        "conditional_top5": float(rank.dropna().le(5).mean()),
        "habit_change_events": int(valid.sum()), "change_precision": float(precision),
        "change_recall": float(recall), "change_f1": float(f1),
    }, event[["event_id", "mmsi", "rank"]]


def paired_cluster_bootstrap(a, b, reps=2000):
    z = a.merge(b, on=["event_id", "mmsi"], suffixes=("_a", "_b"))
    for side in ["a", "b"]:
        z[f"top1_{side}"] = z[f"rank_{side}"].le(1).astype(float)
        z[f"top5_{side}"] = z[f"rank_{side}"].le(5).astype(float)
        z[f"mrr_{side}"] = (1 / z[f"rank_{side}"]).fillna(0)
    agg = z.groupby("mmsi").agg(n=("event_id", "size"),
        d_top1=("top1_b", "sum"), a_top1=("top1_a", "sum"),
        d_top5=("top5_b", "sum"), a_top5=("top5_a", "sum"),
        d_mrr=("mrr_b", "sum"), a_mrr=("mrr_a", "sum"))
    rng = np.random.default_rng(20260921)
    result = {}
    for metric in ["top1", "top5", "mrr"]:
        delta = agg[f"d_{metric}"] - agg[f"a_{metric}"]
        arr = np.c_[agg.n.to_numpy(), delta.to_numpy()]
        draws = []
        for _ in range(reps):
            sample = arr[rng.integers(0, len(arr), len(arr))]
            draws.append(sample[:, 1].sum() / sample[:, 0].sum())
        result[metric] = {"difference": float(delta.sum() / agg.n.sum()),
                          "p025": float(np.quantile(draws, .025)),
                          "p975": float(np.quantile(draws, .975))}
    return result


def within_event_rank_score(event_id, score):
    frame = pd.DataFrame({"event_id": np.asarray(event_id), "score": np.asarray(score)})
    rank = frame.groupby("event_id").score.rank(method="average", ascending=True)
    size = frame.groupby("event_id").score.transform("size")
    return ((rank - 1) / np.maximum(size - 1, 1)).to_numpy()


def tune_blend(dev, learned_score):
    """Choose the distance/learned blend on the Red Sea development holdout only."""
    distance = within_event_rank_score(dev.event_id, -dev.detour_knm)
    learned = within_event_rank_score(dev.event_id, learned_score)
    rows = []
    for alpha in np.linspace(0, 1, 11):
        score = alpha * distance + (1 - alpha) * learned
        z = dev[["event_id", "y"]].copy(); z["score"] = score
        z["rank"] = z.groupby("event_id").score.rank(method="first", ascending=False)
        ranks = z.loc[z.y.eq(1), "rank"]
        rows.append({"distance_weight": float(alpha), "mrr": float((1 / ranks).mean()),
                     "top1": float(ranks.le(1).mean()), "top5": float(ranks.le(5).mean())})
    table = pd.DataFrame(rows)
    best = table.sort_values(["mrr", "top5"], ascending=False).iloc[0]
    return float(best.distance_weight), table


def main():
    (code_area, geo, vessels, panel, pref, habitual, training, red_dev, fallback, mats) = setup_training_data()
    raw, test, meta, sensitivity = build_hormuz_candidates(code_area, geo, vessels, pref)
    lag26 = D.hormuz_lags(raw, code_area)
    test = D.add_lags(test, lag26, fallback)
    test = D.add_network(test, "hormuz", geo, mats)

    base = ["detour_knm", "dwell_10h", "log_calls", "propensity", "pref",
            "vessel_pref_count", "origin_pref_count"]
    shock = base + ["shock_proximity", "inside_region", "days_x_proximity", "exposed_x_proximity"]
    dynamic = shock + ["log_lag4_calls", "log_lag4_refuels", "lag4_refuel_share", "lag4_berth_h"]
    features = dynamic + ["network_base_detour_knm", "shock_network_extra_knm",
                          "shock_network_unreachable", "network_missing"]
    training = training.sort_values("event_id").reset_index(drop=True)
    test = test.sort_values("event_id").reset_index(drop=True)
    xtr, ytr = training[features], training.y

    models = {}
    models["linear_choice"] = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=.3, max_iter=500, class_weight="balanced", random_state=20260921))
    models["pointwise_hgb"] = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.06, max_leaf_nodes=31, min_samples_leaf=80,
        l2_regularization=1.5, random_state=20260921)
    for model in models.values():
        model.fit(xtr, ytr)

    qid = pd.factorize(training.event_id, sort=False)[0]
    ranker = XGBRanker(objective="rank:ndcg", eval_metric="ndcg@5", n_estimators=350,
        max_depth=6, learning_rate=.045, subsample=.85, colsample_bytree=.85,
        min_child_weight=20, reg_lambda=2.0, random_state=20260921, tree_method="hist")
    ranker.fit(xtr, ytr, qid=qid, verbose=False)

    red_dev = red_dev.sort_values("event_id").reset_index(drop=True)
    red_learned = ranker.predict(red_dev[features])
    distance_weight, blend_tuning = tune_blend(red_dev, red_learned)
    test_distance_rank = within_event_rank_score(test.event_id, -test.detour_knm)
    test_learned_rank = within_event_rank_score(test.event_id, ranker.predict(test[features]))

    scores = {
        "shortest_detour": -test.detour_knm.to_numpy(),
        "training_popularity": test.train_popularity.to_numpy(),
        "vessel_habit": test.vessel_pref_count.to_numpy() + 1e-3 * test.train_popularity.to_numpy(),
        "origin_markov": test.origin_pref_count.to_numpy() + 1e-3 * test.train_popularity.to_numpy(),
        "operator_preference": test.pref.to_numpy() + 1e-3 * test.train_popularity.to_numpy(),
        "retrieval_priority": test.candidate_priority.to_numpy(),
        "linear_choice": models["linear_choice"].predict_proba(test[features])[:, 1],
        "pointwise_hgb": models["pointwise_hgb"].predict_proba(test[features])[:, 1],
        "lambdamart": ranker.predict(test[features]),
        "redsea_tuned_hybrid": distance_weight * test_distance_rank + (1 - distance_weight) * test_learned_rank,
    }
    rows, details = [], {}
    for name, score in scores.items():
        metric, detail = end_to_end_metrics(name, score, test, meta, habitual)
        rows.append(metric); details[name] = detail
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "end_to_end_port_ranker_metrics_v3.csv", index=False)
    sensitivity_summary = sensitivity.groupby(["threshold_km", "cap"]).agg(
        events=("event_id", "nunique"), mean_candidates=("candidate_count", "mean"),
        candidate_recall=("actual_recalled", "mean")).reset_index()
    sensitivity_summary.to_csv(OUT / "candidate_retrieval_sensitivity_v3.csv", index=False)
    blend_tuning.to_csv(OUT / "ranker_blend_tuning_redsea_v3.csv", index=False)
    test.to_parquet(OUT / "hormuz_port_candidates_no_oracle_v3.parquet", index=False)
    comparisons = {
        "lambdamart_vs_shortest": paired_cluster_bootstrap(details["shortest_detour"], details["lambdamart"]),
        "lambdamart_vs_pointwise_hgb": paired_cluster_bootstrap(details["pointwise_hgb"], details["lambdamart"]),
        "redsea_tuned_hybrid_vs_shortest": paired_cluster_bootstrap(
            details["shortest_detour"], details["redsea_tuned_hybrid"]),
    }
    report = {
        "candidate_rule": "no realized-port injection; 2500 km detour limit; top 40 by train-only retrieval priority",
        "training_information": "all popularity, preferences, attributes, and imputation fallbacks use 2021-2022 only",
        "test_events": len(meta), "test_rows": len(test),
        "redsea_selected_distance_weight": distance_weight,
        "metrics": metrics.to_dict("records"),
        "candidate_sensitivity": sensitivity_summary.to_dict("records"),
        "paired_vessel_bootstrap": comparisons,
        "warning": "Predictive evaluation, not a causal counterfactual. Provider-confirmed labels are not an independent blind audit.",
    }
    (OUT / "end_to_end_port_ranker_report_v3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print("\nCandidate sensitivity\n", sensitivity_summary.to_string(index=False))


if __name__ == "__main__":
    main()
