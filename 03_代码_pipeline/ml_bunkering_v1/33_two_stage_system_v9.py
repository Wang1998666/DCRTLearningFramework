"""Link refuelling occurrence and conditional port ranking into one system."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import precision_recall_curve
import torch

HERE = Path(__file__).parent
PROJECT = HERE.resolve().parents[1]
OUT = PROJECT / "02_数据_output" / "ml_bunkering_v1"


def load_v8():
    spec = importlib.util.spec_from_file_location("dcrank_v8_link", HERE / "27_dcrank_searoute_repaired.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


M = load_v8()


def threshold_from_pre(model, delta, d, features):
    raw = np.clip(model.predict_proba(d[features])[:, 1], 1e-6, 1-1e-6)
    p = expit(logit(raw)+delta)
    precision, recall, thresholds = precision_recall_curve(d.y, p)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-12)
    i = int(np.nanargmax(f1))
    return float(thresholds[i]), float(f1[i])


def port_probabilities():
    candidates = pd.read_parquet(OUT / "dcrank_searoute_hormuz_candidates_v8.parquet").reset_index(drop=True)
    checkpoint = torch.load(OUT / "dcrank_searoute_model_v8.pt", map_location="cpu", weights_only=False)
    models = []
    for state in checkpoint["states"]:
        model = M.UtilityLinear(len(checkpoint["features"])); model.load_state_dict(state); model.eval(); models.append(model)
    utility, _ = M.utilities(models, candidates, M.FEATURES,
                             np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"]))
    score = (-candidates.sea_detour_knm.to_numpy()+float(checkpoint["budget_knm"])*utility)/M.TAU_KNM
    z = candidates[["event_id", "mmsi", "call_time", "legStartPortCode", "legEndPortCode",
                    "actual", "alt", "y"]].copy()
    z["score"] = score
    z["score_centered"] = z.score-z.groupby("event_id").score.transform("max")
    z["exp_score"] = np.exp(z.score_centered)
    z["conditional_port_probability"] = z.exp_score/z.groupby("event_id").exp_score.transform("sum")
    z["port_rank"] = z.groupby("event_id").score.rank(method="first", ascending=False)
    pred = z.loc[z.groupby("event_id").score.idxmax(), ["event_id", "alt"]].rename(columns={"alt": "predicted_port"})
    truth = z[z.y.eq(1)][["event_id", "conditional_port_probability", "port_rank"]]
    event = z[["event_id", "mmsi", "call_time", "legStartPortCode", "legEndPortCode", "actual"]].drop_duplicates("event_id")
    return event.merge(pred, on="event_id").merge(truth, on="event_id", how="left")


def main():
    occurrence = pd.read_parquet(OUT / "two_stage_occurrence_predictions_v9.parquet")
    occurrence.legStartTime = pd.to_datetime(occurrence.legStartTime)
    port = port_probabilities()
    port.call_time = pd.to_datetime(port.call_time)
    port.mmsi = pd.to_numeric(port.mmsi, errors="coerce").astype("Int64")
    occurrence.mmsi = pd.to_numeric(occurrence.mmsi, errors="coerce").astype("Int64")
    joined = occurrence.merge(port, how="left",
        left_on=["mmsi", "legStartTime", "origin", "dest"],
        right_on=["mmsi", "call_time", "legStartPortCode", "legEndPortCode"],
        validate="one_to_one")

    bundle = joblib.load(OUT / "two_stage_occurrence_models_v9.joblib")
    raw = pd.read_parquet(OUT / "two_stage_occurrence_features_v9.parquet")
    raw.legStartTime = pd.to_datetime(raw.legStartTime)
    pre = raw[raw.legStartTime.between("2026-01-23", "2026-02-27 23:59:59")]
    configs = {
        "hormuz_pre_updated": (bundle["model"], bundle["calibration_delta"]),
        "redsea_plus_hormuz_pre": (bundle["transfer_model"], bundle["transfer_calibration_delta"]),
    }
    rows = []
    for name, (model, delta) in configs.items():
        threshold, pre_f1 = threshold_from_pre(model, delta, pre, bundle["features"])
        p = joined["p_refuel_"+name].to_numpy()
        y = joined.y.to_numpy(int); predicted_refuel = p >= threshold
        negative = y == 0; positive = y == 1
        linked = joined.event_id.notna().to_numpy()
        recalled = joined.port_rank.notna().to_numpy()
        top1_positive = predicted_refuel & linked & recalled & joined.port_rank.fillna(np.inf).le(1).to_numpy()
        top5_positive = predicted_refuel & linked & recalled & joined.port_rank.fillna(np.inf).le(5).to_numpy()
        correct_top1_action = (negative & ~predicted_refuel) | (positive & top1_positive)
        correct_top5_action = (negative & ~predicted_refuel) | (positive & top5_positive)
        specificity = float((~predicted_refuel[negative]).mean())
        sensitivity = float(predicted_refuel[positive].mean())
        positive_top1 = float(top1_positive[positive].mean())
        positive_top5 = float(top5_positive[positive].mean())

        class_prob = np.where(negative, 1-p, 1e-12)
        valid_positive = positive & linked & recalled
        class_prob[valid_positive] = (p[valid_positive]
            * joined.loc[valid_positive, "conditional_port_probability"].to_numpy())
        joint_nll = float(-np.log(np.clip(class_prob, 1e-12, 1)).mean())
        rows.append({"stage1_model": name, "pre_hormuz_threshold": threshold,
                     "pre_hormuz_threshold_f1": pre_f1, "voyages": len(joined),
                     "refuel_events": int(positive.sum()),
                     "positive_port_link_rate": float(linked[positive].mean()),
                     "positive_port_recall_rate_all_positives": float(recalled[positive].mean()),
                     "occurrence_sensitivity": sensitivity, "occurrence_specificity": specificity,
                     "system_positive_top1": positive_top1, "system_positive_top5": positive_top5,
                     "all_voyage_exact_top1_action_accuracy": float(correct_top1_action.mean()),
                     "all_voyage_exact_top5_action_accuracy": float(correct_top5_action.mean()),
                     "balanced_top1_action_accuracy": float((specificity+positive_top1)/2),
                     "balanced_top5_action_accuracy": float((specificity+positive_top5)/2),
                     "joint_no_refuel_or_actual_port_nll": joint_nll})
        joined[f"predict_refuel_{name}"] = predicted_refuel

    metrics = pd.DataFrame(rows)
    keep = ["mmsi", "legStartTime", "legEndTime", "origin", "dest", "y",
            "p_refuel_hormuz_pre_updated", "p_refuel_redsea_plus_hormuz_pre",
            "event_id", "actual", "predicted_port", "conditional_port_probability", "port_rank",
            "predict_refuel_hormuz_pre_updated", "predict_refuel_redsea_plus_hormuz_pre"]
    joined[keep].to_parquet(OUT / "two_stage_system_predictions_v9.parquet", index=False)
    metrics.to_csv(OUT / "two_stage_system_metrics_v9.csv", index=False)
    transfer = json.loads((OUT / "redsea_transfer_dcrank_report_v9.json").read_text(encoding="utf-8"))
    report = {
        "system": "P(no refuel)=1-P(refuel); P(port j)=P(refuel)*P(port j|refuel)",
        "stage1_primary": "redsea_plus_hormuz_pre is the pre-specified transfer model; hormuz_pre_updated is retained as the no-transfer comparator",
        "stage2_primary": "normal_v8 remains the pre-specified main ranker; Red Sea fine-tuning is evaluated once as a challenger",
        "threshold_rule": "maximum F1 on the pre-Hormuz window only",
        "metrics": metrics.to_dict("records"),
        "stage2_hormuz_transfer_bootstrap": transfer["paired_vessel_bootstrap"]["hormuz_post"],
        "warning": "all-voyage action accuracy is dominated by non-refuelling voyages; balanced accuracy and positive-event Top-k are reported",
    }
    (OUT / "two_stage_system_report_v9.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
