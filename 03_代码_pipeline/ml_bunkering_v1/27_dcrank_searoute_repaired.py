"""Leakage-resistant DCRank with AIS-derived sea-route distances (v8).

This is a clean rebuild rather than a patch over v6/v7 outputs.

Design
------
* Candidate retrieval and detours use shortest paths on the observed sea-lane
  network.  Great-circle detours are not used by the ranker.
* Training events are rebuilt directly from dated source legs; no stale
  ``event_id`` join is used.
* Port-state features use weeks w-4,...,w-1 only.  Training imputation is fit
  before 2023 and the missing rates are audited.
* Utility is trained with a fixed, declared loss temperature.  The global and
  context-dependent distance budgets are calibrated on 2023 validation data,
  never on the 2026 Hormuz test.
* Five deterministic seeds, cluster bootstrap, within-event permutation, a
  frozen-pre-shock information test, and crude-tanker transfer are reported.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).parent
PROJECT = HERE.resolve().parents[1]
OUT = PROJECT / "02_数据_output" / "ml_bunkering_v1"
CORE = PROJECT / "02_数据_output" / "refuel_panel_v2"
SEEDS = [20260921, 20260922, 20260923, 20260924, 20260925]
FEATURES = ["log_lag4_refuels", "lag4_refuel_share", "log_lag4_calls", "lag4_berth_h"]
LAG_COLS = ["lag4_calls", "lag4_refuels", "lag4_refuel_share", "lag4_berth_h"]
REFERENCE_B_KNM = .05
TAU_KNM = .05
B_MAX_KNM = .20
MAX_RETRIEVAL_DETOUR_NM = 1350.0
CAP = 40


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E = load_module("e2e_v8", "12_end_to_end_port_ranker.py")
D, R = E.D, E.R


def resources():
    code_area, geo, cent, vessels = R.resources()
    choice = pd.read_parquet(CORE / "choice_long_v2.parquet")
    train_choice = choice[choice.year.isin([2021, 2022])]
    R.AREA_ATTR = (train_choice.groupby("alt").agg(
        dwell_10h=("dwell_10h", "first"), log_calls=("log_calls", "first"),
        propensity=("propensity", "first"), events=("y", "sum"))
        .join(cent, how="inner").dropna())
    D.R.AREA_ATTR = R.AREA_ATTR
    mats = D.graph_matrices(geo)
    return code_area, geo, vessels, mats


def canonicalize(raw):
    z = raw.copy()
    z["_quality"] = (z.isRefueled.notna().astype(int) * 100
                     + z.refuelPortCodes.notna().astype(int) * 20
                     + z.notna().sum(axis=1))
    keys = ["mmsi", "legStartTime", "legEndTime"]
    return (z.sort_values(keys + ["_quality"], kind="mergesort")
            .drop_duplicates(keys, keep="last").drop(columns="_quality").reset_index(drop=True))


def node_maps(geo, mats):
    ports, ix, normal, red, hormuz, tree, gp = mats
    geo_codes = geo.index.astype(str)
    code_node = {}
    exact = geo_codes.isin(ix)
    for code in geo_codes[exact]:
        code_node[code] = ix[code]
    missing = geo_codes[~exact]
    if len(missing):
        near = D.nearest_codes(geo.loc[missing, "lat"], geo.loc[missing, "lon"], tree, gp)
        code_node.update({str(c): ix[str(n)] for c, n in zip(missing, near)})
    areas = R.AREA_ATTR.index.to_numpy()
    near_area = D.nearest_codes(R.AREA_ATTR.lat, R.AREA_ATTR.lon, tree, gp)
    area_node = np.array([ix[str(n)] for n in near_area], dtype=int)
    return code_node, areas, area_node


def event_frame(raw, code_area, start, end, prefix, exposure=None, sample_caps=None):
    z = raw[raw.legStartTime.between(start, end) & raw.isRefueled.eq(1)].copy()
    initial = len(z)
    z["actual_code"] = z.refuelPortCodes.astype(str).str.split(",").str[0]
    z["actual"] = z.actual_code.map(code_area)
    z = z.dropna(subset=["mmsi", "legStartTime", "legStartPortCode", "legEndPortCode", "actual"])
    z["exposed"] = z.mmsi.isin(exposure or set()).astype("int8")
    if sample_caps:
        z["_rand"] = np.random.default_rng(SEEDS[0]).random(len(z))
        z = (z.sort_values("_rand").groupby("exposed", group_keys=False)
             .head(sample_caps).drop(columns="_rand"))
    z = z.reset_index(drop=True)
    z["event_id"] = [f"{prefix}_{i}" for i in range(len(z))]
    return z, initial, len(z)


def network_candidates(events, geo, mats, scenario, prefix):
    """Retrieve candidates using only AIS sea-network paths; never append truth."""
    code_node, areas, area_node = node_maps(geo, mats)
    matrix = {"normal": mats[2], "redsea": mats[3], "hormuz": mats[4]}[scenario]
    rows, sens, meta_rows = [], [], []
    area_set = set(areas)
    for ev in events.itertuples():
        oc, dc = str(ev.legStartPortCode), str(ev.legEndPortCode)
        if oc not in code_node or dc not in code_node or ev.actual not in area_set:
            continue
        oi, di = code_node[oc], code_node[dc]
        direct_km = float(matrix[oi, di])
        if not np.isfinite(direct_km):
            continue
        via_km = matrix[oi, area_node] + matrix[area_node, di]
        valid = np.isfinite(via_km)
        detour_nm = np.maximum(via_km - direct_km, 0) / 1.852
        feasible = np.flatnonzero(valid & (detour_nm <= MAX_RETRIEVAL_DETOUR_NM))
        order = feasible[np.lexsort((areas[feasible], detour_nm[feasible]))]
        selected = order[:CAP]
        meta_rows.append({"event_id": ev.event_id, "mmsi": ev.mmsi,
                          "actual": ev.actual, "exposed": ev.exposed})
        for cap in [20, 40, 80]:
            take = order[:cap]
            sens.append({"event_id": ev.event_id, "cap": cap,
                         "actual_recalled": int(ev.actual in set(areas[take])),
                         "candidate_count": len(take)})
        for j in selected:
            rows.append({
                "event_id": ev.event_id, "mmsi": ev.mmsi, "call_time": ev.legStartTime,
                "legStartPortCode": oc, "legEndPortCode": dc, "actual": ev.actual,
                "alt": areas[j], "y": int(areas[j] == ev.actual), "exposed": ev.exposed,
                "sea_detour_knm": float(detour_nm[j] / 1000),
                "direct_sea_nm": float(direct_km / 1.852),
                "via_sea_nm": float(via_km[j] / 1.852),
                "network_scenario": scenario,
            })
    return pd.DataFrame(rows), pd.DataFrame(meta_rows), pd.DataFrame(sens)


def add_lags_and_frozen(long, lag, fallback, freeze_week=None):
    z = long.copy()
    z["week"] = z.call_time.dt.to_period("W").dt.start_time
    z = z.merge(lag, left_on=["alt", "week"], right_on=["area", "week"], how="left").drop(columns="area")
    for c in LAG_COLS:
        z[c + "_missing"] = z[c].isna().astype("int8")
        z[c] = z[c].fillna(z.alt.map(fallback[c])).fillna(0)
    z["log_lag4_calls"] = np.log1p(z.lag4_calls)
    z["log_lag4_refuels"] = np.log1p(z.lag4_refuels)
    if freeze_week is not None:
        fw = pd.Timestamp(freeze_week).to_period("W").start_time
        state = lag[lag.week.eq(fw)].drop_duplicates("area").set_index("area")
        for c in LAG_COLS:
            z["frozen_" + c] = z.alt.map(state[c]).fillna(z.alt.map(fallback[c])).fillna(0)
        z["frozen_log_lag4_calls"] = np.log1p(z.frozen_lag4_calls)
        z["frozen_log_lag4_refuels"] = np.log1p(z.frozen_lag4_refuels)
    return z


class GroupDataset(Dataset):
    def __init__(self, frame, features, mean=None, std=None):
        z = frame.sort_values(["event_id", "alt"], kind="mergesort").reset_index(drop=True)
        raw = z[features].to_numpy(np.float32)
        self.mean = np.nanmean(raw, axis=0) if mean is None else np.asarray(mean)
        self.std = np.nanstd(raw, axis=0) if std is None else np.asarray(std)
        self.std[self.std < 1e-6] = 1
        x = np.nan_to_num((raw - self.mean) / self.std).astype(np.float32)
        self.items = []
        for _, g in z.groupby("event_id", sort=False):
            idx = g.index.to_numpy(); positives = np.flatnonzero(g.y.to_numpy() == 1)
            if len(positives) == 1:
                self.items.append((x[idx], g.sea_detour_knm.to_numpy(np.float32), int(positives[0])))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch):
    n, k, d = len(batch), max(len(x[0]) for x in batch), batch[0][0].shape[1]
    feat = torch.zeros((n, k, d)); det = torch.zeros((n, k)); mask = torch.zeros((n, k), dtype=torch.bool)
    target = torch.empty(n, dtype=torch.long)
    for i, (x, distance, y) in enumerate(batch):
        m = len(distance); feat[i, :m] = torch.from_numpy(x); det[i, :m] = torch.from_numpy(distance)
        mask[i, :m] = True; target[i] = y
    return feat, det, mask, target


class UtilityLinear(nn.Module):
    def __init__(self, n_features):
        super().__init__(); self.linear = nn.Linear(n_features, 1)
    def utility(self, x): return torch.sigmoid(self.linear(x)).squeeze(-1)


def train_one(dataset, seed, epochs=35):
    torch.manual_seed(seed); np.random.seed(seed)
    model = UtilityLinear(len(FEATURES))
    loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=5e-4)
    for _ in range(epochs):
        model.train()
        for feat, detour, mask, target in loader:
            score = (-detour + REFERENCE_B_KNM * model.utility(feat)) / TAU_KNM
            score = score.masked_fill(~mask, -1e9)
            loss = nn.functional.cross_entropy(score, target)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
    model.eval(); return model


def utilities(models, frame, feature_names, mean, std):
    x = frame[feature_names].to_numpy(np.float32)
    x = np.nan_to_num((x - mean) / std).astype(np.float32)
    with torch.no_grad():
        values = [m.utility(torch.from_numpy(x)).numpy() for m in models]
    return np.mean(values, axis=0), values


def positive_groups(frame):
    groups = []
    for _, g in frame.groupby("event_id", sort=False):
        positive = np.flatnonzero(g.y.to_numpy() == 1)
        if len(positive) == 1:
            groups.append((g.index.to_numpy(), int(positive[0])))
    return groups


def mean_nll(frame, utility, budget_for_event):
    losses = []
    for idx, target in positive_groups(frame):
        b = budget_for_event(frame.loc[idx[0]]) if callable(budget_for_event) else float(budget_for_event)
        score = (-frame.loc[idx, "sea_detour_knm"].to_numpy() + b * utility[idx]) / TAU_KNM
        score -= score.max(); losses.append(-score[target] + np.log(np.exp(score).sum()))
    return float(np.mean(losses))


def calibrate_budgets(validation, utility, train_direct):
    global_result = minimize_scalar(lambda b: mean_nll(validation, utility, b),
        bounds=(.005, B_MAX_KNM), method="bounded", options={"xatol": 1e-5})
    b_global = float(global_result.x)
    logged_direct = np.log(np.clip(train_direct, 10, None))
    center, scale = float(logged_direct.mean()), float(logged_direct.std())
    scale = max(scale, .1)
    def dynamic_budget(row, theta):
        z = (np.log(max(float(row.direct_sea_nm), 10)) - center) / scale
        return B_MAX_KNM * expit(theta[0] + theta[1] * z)
    def objective(theta): return mean_nll(validation, utility, lambda row: dynamic_budget(row, theta))
    start = [logit(np.clip(b_global / B_MAX_KNM, .01, .99)), 0.0]
    candidates = [minimize(objective, start, method="L-BFGS-B", bounds=[(-8, 8), (-8, 8)]),
                  minimize(objective, [0, 0], method="L-BFGS-B", bounds=[(-8, 8), (-8, 8)])]
    best = min(candidates, key=lambda r: r.fun)
    return b_global, np.asarray(best.x), center, scale, float(global_result.fun), float(best.fun)


def dynamic_budgets(frame, theta, center, scale):
    z = (np.log(frame.direct_sea_nm.clip(lower=10)) - center) / scale
    return B_MAX_KNM * expit(theta[0] + theta[1] * z.to_numpy())


def score_metrics(name, frame, meta, habitual, utility, budget):
    b = np.asarray(budget) if np.ndim(budget) else float(budget)
    score = -frame.sea_detour_knm.to_numpy() + b * utility
    metric, detail = E.end_to_end_metrics(name, score, frame.rename(
        columns={"sea_detour_knm": "detour_knm"}), meta, habitual)
    return metric, detail, score


def service_lag_sources(panel, raw26, crude, code_area):
    hist_lag = D.historical_lags(panel)
    train_lag = hist_lag[hist_lag.week.lt("2023-01-01")]
    fallback = train_lag.groupby("area")[LAG_COLS].median().to_dict()
    lag26 = D.hormuz_lags(raw26, code_area)
    crude_lag = D.hormuz_lags(crude, code_area)
    return hist_lag, lag26, crude_lag, fallback


def prepare_all():
    code_area, geo, vessels, mats = resources()
    panel = pd.read_parquet(CORE / "legs_panel_v2_areas.parquet")
    panel.mmsi = pd.to_numeric(panel.mmsi, errors="coerce").astype("Int64")
    panel.legStartTime = pd.to_datetime(panel.legStartTime); panel.isRefueled = pd.to_numeric(panel.isRefueled).fillna(0)
    train_ev, train_initial, train_selected = event_frame(
        panel, code_area, "2021-01-01", "2022-12-31 23:59:59", "TR", sample_caps=15000)
    val_ev, val_initial, val_selected = event_frame(
        panel, code_area, "2023-01-01", "2023-11-30 23:59:59", "VA", sample_caps=5000)

    raw26 = pd.read_parquet(OUT / "hormuz_container_legs_2026.parquet")
    raw26.mmsi = pd.to_numeric(raw26.mmsi, errors="coerce").astype("Int64")
    raw26.legStartTime = pd.to_datetime(raw26.legStartTime); raw26.legEndTime = pd.to_datetime(raw26.legEndTime)
    raw26 = canonicalize(raw26)
    pre = raw26.legStartTime.between("2026-01-23", "2026-02-27 23:59:59")
    exposed = set(raw26.loc[pre & raw26.legCrossNodeList.astype(str).str.contains(
        "HORMUZ", case=False, na=False), "mmsi"].dropna())
    test_ev, test_initial, test_selected = event_frame(
        raw26, code_area, "2026-02-28", "2026-07-22 23:59:59", "HT", exposure=exposed)

    crude = pd.read_parquet(OUT / "crude_tanker_legs_2024_2026_v1.parquet")
    crude.legStartTime = pd.to_datetime(crude.legStartTime)
    cpre = crude.legStartTime.between("2026-01-23", "2026-02-27 23:59:59")
    cexp = set(crude.loc[cpre & crude.legCrossNodeList.astype(str).str.contains(
        "HORMUZ", case=False, na=False), "mmsi"].dropna())
    crude_ev, crude_initial, crude_selected = event_frame(
        crude, code_area, "2026-02-28", "2026-07-22 23:59:59", "CT", exposure=cexp)

    train, train_meta, train_sens = network_candidates(train_ev, geo, mats, "normal", "TR")
    val, val_meta, val_sens = network_candidates(val_ev, geo, mats, "normal", "VA")
    test, test_meta, test_sens = network_candidates(test_ev, geo, mats, "hormuz", "HT")
    crude_long, crude_meta, crude_sens = network_candidates(crude_ev, geo, mats, "hormuz", "CT")
    hist_lag, lag26, crude_lag, fallback = service_lag_sources(panel, raw26, crude, code_area)
    train = add_lags_and_frozen(train, hist_lag, fallback)
    val = add_lags_and_frozen(val, hist_lag, fallback)
    test = add_lags_and_frozen(test, lag26, fallback, "2026-02-23")
    crude_long = add_lags_and_frozen(crude_long, crude_lag, fallback, "2026-02-23")
    habitual = R.history_maps(panel)[2]
    crude_hist = crude[crude.legStartTime.lt("2026-01-01") & crude.isRefueled.eq(1)].copy()
    crude_hist["area"] = crude_hist.refuelPortCodes.astype(str).str.split(",").str[0].map(code_area)
    crude_habitual = (crude_hist.dropna(subset=["area"]).groupby(["mmsi", "area"]).size()
                      .rename("n").reset_index().sort_values(["mmsi", "n"], ascending=[True, False])
                      .drop_duplicates("mmsi").set_index("mmsi").area)
    return ({"train": train, "val": val, "test": test, "crude": crude_long},
            {"train": train_meta, "val": val_meta, "test": test_meta, "crude": crude_meta},
            {"train": train_sens, "val": val_sens, "test": test_sens, "crude": crude_sens},
            habitual, crude_habitual,
            {"train": {"source": train_initial, "selected": train_selected},
             "val": {"source": val_initial, "selected": val_selected},
             "test": {"source": test_initial, "selected": test_selected},
             "crude": {"source": crude_initial, "selected": crude_selected}},
            len(mats[0]))


def evaluate_information_mode(models, frame, meta, habitual, mean, std, b_global, theta, center, scale, mode):
    names = FEATURES if mode == "rolling" else ["frozen_" + c for c in FEATURES]
    utility, seed_utilities = utilities(models, frame, names, mean, std)
    dynamic = dynamic_budgets(frame, theta, center, scale)
    rows, details = [], {}
    for name, u, b in [
        (f"shortest_{mode}", np.zeros(len(frame)), 0.0),
        (f"fixed50_{mode}", utility, .05),
        (f"learned_global_{mode}", utility, b_global),
        (f"learned_dynamic_{mode}", utility, dynamic),
    ]:
        metric, detail, score = score_metrics(name, frame, meta, habitual, u, b)
        rows.append(metric); details[name] = detail
    return rows, details, utility, seed_utilities, dynamic


def main():
    print("Building sea-route candidates and strictly lagged features...", flush=True)
    frames, metas, sensitivities, habitual, crude_habitual, initial, network_nodes = prepare_all()
    train, val, test, crude = (frames[k].reset_index(drop=True) for k in ["train", "val", "test", "crude"])
    metas = {k: v for k, v in metas.items()}
    train_ds = GroupDataset(train, FEATURES)
    val_ds = GroupDataset(val, FEATURES, train_ds.mean, train_ds.std)
    print(f"Train events with recalled truth={len(train_ds):,}; validation={len(val_ds):,}", flush=True)
    models = [train_one(train_ds, seed) for seed in SEEDS]
    val_u, _ = utilities(models, val, FEATURES, train_ds.mean, train_ds.std)
    train_direct = train.drop_duplicates("event_id").direct_sea_nm.to_numpy()
    b_global, theta, center, scale, global_nll, dynamic_nll = calibrate_budgets(
        val, val_u, train_direct)
    print(f"Validation-learned global budget={b_global*1000:.2f} nm; dynamic theta={theta}", flush=True)

    rows, details, test_u, seed_u, dynamic = evaluate_information_mode(
        models, test, metas["test"], habitual, train_ds.mean, train_ds.std,
        b_global, theta, center, scale, "rolling")
    frozen_rows, frozen_details, frozen_u, _, frozen_dynamic = evaluate_information_mode(
        models, test, metas["test"], habitual, train_ds.mean, train_ds.std,
        b_global, theta, center, scale, "frozen")
    rows += frozen_rows; details.update(frozen_details)
    metrics = pd.DataFrame(rows)

    # Seed stability uses the validation-selected global budget fixed above.
    seed_rows = []
    for seed, u in zip(SEEDS, seed_u):
        met, _, _ = score_metrics(f"seed_{seed}", test, metas["test"], habitual, u, b_global)
        seed_rows.append(met)
    seed_table = pd.DataFrame(seed_rows)

    # Actual, computed sensitivity curves; these are saved and read by figures.
    curve_rows = []
    for split, frame, meta, u in [("validation", val, metas["val"], val_u),
                                  ("hormuz_test_descriptive", test, metas["test"], test_u)]:
        for bnm in np.linspace(0, 200, 41):
            met, _, _ = score_metrics(f"budget_{bnm:.1f}", frame, meta, habitual, u, bnm / 1000)
            curve_rows.append({"split": split, "budget_nm": bnm, **met})
    curve = pd.DataFrame(curve_rows)

    comparisons = {
        "global_vs_shortest": E.paired_cluster_bootstrap(details["shortest_rolling"], details["learned_global_rolling"]),
        "global_vs_fixed50": E.paired_cluster_bootstrap(details["fixed50_rolling"], details["learned_global_rolling"]),
        "dynamic_vs_global": E.paired_cluster_bootstrap(details["learned_global_rolling"], details["learned_dynamic_rolling"]),
        "rolling_vs_frozen_global": E.paired_cluster_bootstrap(details["learned_global_frozen"], details["learned_global_rolling"]),
    }

    # Within-event permutation of all service features together.
    rng = np.random.default_rng(SEEDS[0]); groups = [g.to_numpy() for _, g in test.groupby("event_id").groups.items()]
    shuffle_rows = []
    raw_x = test[FEATURES].to_numpy(float)
    for rep in range(100):
        shuffled = raw_x.copy()
        for idx in groups: shuffled[idx] = raw_x[rng.permutation(idx)]
        q = test.copy(); q[FEATURES] = shuffled
        u, _ = utilities(models, q, FEATURES, train_ds.mean, train_ds.std)
        met, _, _ = score_metrics("shuffle", q, metas["test"], habitual, u, b_global)
        shuffle_rows.append({"rep": rep, **met})
    shuffle = pd.DataFrame(shuffle_rows)

    # Cross-fleet parameter transfer, reported for both rolling and frozen state.
    crude_rows, crude_details, crude_u, _, crude_dynamic = evaluate_information_mode(
        models, crude, metas["crude"], crude_habitual, train_ds.mean, train_ds.std,
        b_global, theta, center, scale, "rolling")
    crude_frozen_rows, _, _, _, _ = evaluate_information_mode(
        models, crude, metas["crude"], crude_habitual, train_ds.mean, train_ds.std,
        b_global, theta, center, scale, "frozen")
    crude_metrics = pd.DataFrame(crude_rows + crude_frozen_rows)
    crude_comparison = E.paired_cluster_bootstrap(
        crude_details["shortest_rolling"], crude_details["learned_global_rolling"])

    # The barrier is verified against the actual network-distance column.
    chosen = test.assign(score=-test.sea_detour_knm + b_global * test_u)
    chosen = chosen.loc[chosen.groupby("event_id").score.idxmax()]
    nearest = test.groupby("event_id").sea_detour_knm.min()
    excess = chosen.set_index("event_id").sea_detour_knm - nearest
    barrier_violations = int((excess > b_global + 1e-7).sum())

    # Average marginal substitution uses each observed sigmoid derivative, not 0.25.
    weight_rows = []
    train_x = np.nan_to_num((train[FEATURES].to_numpy(np.float32) - train_ds.mean) / train_ds.std)
    for seed, model in zip(SEEDS, models):
        w = model.linear.weight.detach().numpy().ravel(); bias = float(model.linear.bias.detach())
        z = train_x @ w + bias; derivative = (expit(z) * (1 - expit(z))).mean()
        for feature, coef, sd in zip(FEATURES, w, train_ds.std):
            weight_rows.append({"seed": seed, "feature": feature, "weight": float(coef),
                "mean_sigmoid_derivative": float(derivative),
                "mrs_nm_per_unit_at_learned_budget": float(b_global * 1000 * derivative * coef / sd)})
    weights = pd.DataFrame(weight_rows)

    metrics.to_csv(OUT / "dcrank_searoute_metrics_v8.csv", index=False)
    crude_metrics.to_csv(OUT / "dcrank_searoute_crude_metrics_v8.csv", index=False)
    seed_table.to_csv(OUT / "dcrank_searoute_seed_stability_v8.csv", index=False)
    curve.to_csv(OUT / "dcrank_searoute_budget_curve_v8.csv", index=False)
    shuffle.to_csv(OUT / "dcrank_searoute_shuffle_v8.csv", index=False)
    weights.to_csv(OUT / "dcrank_searoute_weights_v8.csv", index=False)
    pd.concat([v.assign(split=k) for k, v in sensitivities.items()]).to_csv(
        OUT / "dcrank_searoute_candidate_sensitivity_v8.csv", index=False)
    pd.DataFrame({"event_id": test.drop_duplicates("event_id").event_id,
                  "dynamic_budget_nm": dynamic[test.groupby("event_id").head(1).index] * 1000}).to_parquet(
        OUT / "dcrank_searoute_dynamic_budgets_v8.parquet", index=False)
    test.to_parquet(OUT / "dcrank_searoute_hormuz_candidates_v8.parquet", index=False)
    crude.to_parquet(OUT / "dcrank_searoute_crude_candidates_v8.parquet", index=False)
    torch.save({"states": [m.state_dict() for m in models], "features": FEATURES,
                "mean": train_ds.mean, "std": train_ds.std, "seeds": SEEDS,
                "budget_knm": b_global, "dynamic_theta": theta,
                "context_center": center, "context_scale": scale,
                "tau_knm": TAU_KNM, "reference_budget_knm": REFERENCE_B_KNM},
               OUT / "dcrank_searoute_model_v8.pt")

    audit = {}
    for name, frame in frames.items():
        sens40 = sensitivities[name][sensitivities[name].cap.eq(40)]
        audit[name] = {"source_refuel_events": initial[name]["source"],
                       "events_after_deterministic_sampling": initial[name]["selected"],
                       "geographically_eligible_events": int(metas[name].event_id.nunique()),
                       "candidate_rows": len(frame), "candidate_recall": float(frame.groupby("event_id").y.sum().gt(0).mean()),
                       "candidate_recall_including_zero_candidate_events": float(sens40.actual_recalled.mean()),
                       "zero_detour_candidate_share": float(frame.sea_detour_knm.eq(0).mean()),
                       "zero_direct_route_event_share": float(frame.drop_duplicates("event_id").direct_sea_nm.eq(0).mean()),
                       "lag_missing_share": {c: float(frame[c + "_missing"].mean()) for c in LAG_COLS}}
    report = {
        "framework": "DCRank-v8 with AIS-derived sea-lane shortest-path detours",
        "network_provenance": "TRE-10CHOKE observed AIS corridor network; baseline.parquet for training/validation and HORMUZ_STRAIT scenario for the 2026 test",
        "network_nodes": network_nodes,
        "distance_definition": "(shortest origin-candidate-destination sea-network path minus shortest origin-destination sea-network path) / 1.852; nautical miles",
        "candidate_rule": f"reachable under scenario network; <= {MAX_RETRIEVAL_DETOUR_NM} nm extra; closest {CAP}; no truth injection",
        "training_window": "2021-01-01 through 2022-12-31",
        "budget_validation_window": "2023-01-01 through 2023-11-30",
        "blind_test_window": "2026-02-28 through 2026-07-22",
        "loss_temperature_knm_fixed": TAU_KNM, "reference_training_budget_knm": REFERENCE_B_KNM,
        "learned_global_budget_nm": b_global * 1000, "global_validation_nll": global_nll,
        "dynamic_theta": theta.tolist(), "dynamic_validation_nll": dynamic_nll,
        "metrics": metrics.to_dict("records"), "bootstrap": comparisons,
        "seed_top1_mean": float(seed_table.top1_end_to_end.mean()),
        "seed_top1_std": float(seed_table.top1_end_to_end.std()),
        "shuffle_top1_mean": float(shuffle.top1_end_to_end.mean()),
        "shuffle_top1_p975": float(shuffle.top1_end_to_end.quantile(.975)),
        "crude_metrics": crude_metrics.to_dict("records"), "crude_bootstrap": crude_comparison,
        "network_barrier_violations": barrier_violations, "data_audit": audit,
        "interpretation": "rolling uses only prior complete weeks; frozen holds port state at the last pre-shock week",
    }
    (OUT / "dcrank_searoute_report_v8.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics[["model", "events", "candidate_recall", "top1_end_to_end", "top3_end_to_end", "top5_end_to_end", "mrr_end_to_end"]].to_string(index=False))
    print("\nCRUDE\n", crude_metrics[["model", "events", "candidate_recall", "top1_end_to_end", "top5_end_to_end"]].to_string(index=False))
    print(json.dumps({"learned_budget_nm": b_global*1000, "dynamic_theta": theta.tolist(),
                      "barrier_violations": barrier_violations, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
