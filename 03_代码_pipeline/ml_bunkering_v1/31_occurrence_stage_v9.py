"""Stage 1 of the current two-stage system: will this voyage refuel?

The experiment separates local pre-Hormuz updating from the incremental value
of early Red Sea observations.  All models are calibrated using only the
pre-shock window; no post-Hormuz label is used for training or calibration.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "02_数据_output" / "ml_bunkering_v1"
SEED = 20260921
FEATURES = ["prior_y", "prior3_rate", "prior10_rate", "days_since_refuel", "nm_since_refuel",
            "origin_rate", "dest_rate", "od_rate", "vessel_rate", "operator_rate", "month_sin",
            "month_cos", "teu", "power_kw_total", "ship_age", "isFullLoad", "cargoLoad",
            "berthDuration", "averageSpeed"]


def offset(prob, y):
    p = np.clip(np.asarray(prob), 1e-6, 1 - 1e-6)
    target = float(np.mean(y))
    return float(brentq(lambda d: expit(logit(p) + d).mean() - target, -20, 20))


def calibrated(model, calibration, test):
    delta = offset(model.predict_proba(calibration[FEATURES])[:, 1], calibration.y)
    raw = np.clip(model.predict_proba(test[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    return expit(logit(raw) + delta), delta


def fit(frame, weights=None):
    model = HistGradientBoostingClassifier(
        max_iter=220, learning_rate=.055, max_leaf_nodes=31,
        min_samples_leaf=100, l2_regularization=2.0, random_state=SEED)
    model.fit(frame[FEATURES], frame.y, sample_weight=weights)
    return model


def metrics(name, split, frame, prob):
    y = frame.y.to_numpy(int); p = np.clip(prob, 1e-6, 1-1e-6)
    return {"model": name, "split": split, "rows": len(frame), "refuel_events": int(y.sum()),
            "refuel_rate": float(y.mean()), "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)), "log_loss": float(log_loss(y, p)),
            "brier": float(brier_score_loss(y, p))}


def vessel_bootstrap(a, b, frame, reps=300):
    """Paired B-A differences; positive AUC/AP and negative loss favour B."""
    z = pd.DataFrame({"mmsi": frame.mmsi.to_numpy(), "y": frame.y.to_numpy(int),
                      "a": np.asarray(a), "b": np.asarray(b)})
    ships = z.mmsi.drop_duplicates().to_numpy(); groups = {s: q.index.to_numpy() for s, q in z.groupby("mmsi")}
    rng = np.random.default_rng(SEED); draws = {"roc_auc": [], "pr_auc": [], "log_loss": []}
    for _ in range(reps):
        sample = ships[rng.integers(0, len(ships), len(ships))]
        idx = np.concatenate([groups[s] for s in sample])
        y, pa, pb = z.y.to_numpy()[idx], z.a.to_numpy()[idx], z.b.to_numpy()[idx]
        if np.unique(y).size < 2:
            continue
        draws["roc_auc"].append(roc_auc_score(y, pb)-roc_auc_score(y, pa))
        draws["pr_auc"].append(average_precision_score(y, pb)-average_precision_score(y, pa))
        draws["log_loss"].append(log_loss(y, pb)-log_loss(y, pa))
    out = {}
    for key, values in draws.items():
        out[key] = {"difference": float(np.mean(values)),
                    "p025": float(np.quantile(values, .025)),
                    "p975": float(np.quantile(values, .975))}
    return out


def capped(frame, n, seed):
    return frame if len(frame) <= n else frame.sample(n=n, random_state=seed)


def main():
    d = pd.read_parquet(OUT / "two_stage_occurrence_features_v9.parquet")
    d.legStartTime = pd.to_datetime(d.legStartTime)
    normal = capped(d[d.legStartTime.between("2021-01-01", "2022-12-31 23:59:59")], 300000, SEED)
    red_early = capped(d[d.legStartTime.between("2023-12-15", "2024-03-31 23:59:59")], 120000, SEED+1)
    red_late = d[d.legStartTime.between("2024-04-01", "2024-06-30 23:59:59")].copy()
    hormuz_pre = capped(d[d.legStartTime.between("2026-01-23", "2026-02-27 23:59:59")], 120000, SEED+2)
    hormuz_post = d[d.legStartTime.between("2026-02-28", "2026-07-22 23:59:59")].copy()

    frozen = fit(normal)
    local_train = pd.concat([normal, hormuz_pre], ignore_index=True)
    local_weights = np.r_[np.ones(len(normal)), np.repeat(3.0, len(hormuz_pre))]
    local = fit(local_train, local_weights)
    transfer_train = pd.concat([normal, red_early, hormuz_pre], ignore_index=True)
    transfer_weights = np.r_[np.ones(len(normal)), np.repeat(3.0, len(red_early)+len(hormuz_pre))]
    transfer = fit(transfer_train, transfer_weights)

    rows, predictions, deltas = [], {}, {}
    for name, model in [("normal_frozen", frozen), ("hormuz_pre_updated", local),
                        ("redsea_plus_hormuz_pre", transfer)]:
        p, delta = calibrated(model, hormuz_pre, hormuz_post)
        rows.append(metrics(name, "hormuz_post", hormuz_post, p)); predictions[name] = p; deltas[name] = delta
    # Red Sea late holdout describes whether the early-redsea update learned within-crisis information.
    red_updated = fit(pd.concat([normal, red_early], ignore_index=True),
                      np.r_[np.ones(len(normal)), np.repeat(3.0, len(red_early))])
    for name, model in [("normal_frozen", frozen), ("redsea_early_updated", red_updated)]:
        p, _ = calibrated(model, red_early, red_late)
        rows.append(metrics(name, "redsea_late", red_late, p))

    table = pd.DataFrame(rows)
    comparison = vessel_bootstrap(predictions["hormuz_pre_updated"],
                                  predictions["redsea_plus_hormuz_pre"], hormuz_post)
    prediction = hormuz_post[["mmsi", "legStartTime", "legEndTime", "origin", "dest", "y"]].copy()
    for name, p in predictions.items(): prediction["p_refuel_"+name] = p
    prediction.to_parquet(OUT / "two_stage_occurrence_predictions_v9.parquet", index=False)
    table.to_csv(OUT / "two_stage_occurrence_metrics_v9.csv", index=False)
    joblib.dump({"model": local, "transfer_model": transfer, "features": FEATURES,
                 "calibration_delta": deltas["hormuz_pre_updated"],
                 "transfer_calibration_delta": deltas["redsea_plus_hormuz_pre"]},
                OUT / "two_stage_occurrence_models_v9.joblib")
    report = {
        "stage": "probability that a voyage refuels",
        "input_rows": len(d), "features": FEATURES,
        "normal_training_rows": len(normal), "redsea_early_rows": len(red_early),
        "redsea_late_rows": len(red_late), "hormuz_pre_rows": len(hormuz_pre),
        "hormuz_post_rows": len(hormuz_post),
        "time_rule": "no post-Hormuz label is used for fitting or calibration",
        "metrics": table.to_dict("records"),
        "incremental_redsea_transfer_on_hormuz": comparison,
        "interpretation": "compare redsea_plus_hormuz_pre with hormuz_pre_updated; the only extra labelled period is early Red Sea",
    }
    (OUT / "two_stage_occurrence_report_v9.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
