"""Stage-2 cross-crisis test under the repaired AIS sea-route DCRank.

Base v8 weights are fine-tuned on early Red Sea refuelling choices.  The number
of fine-tuning epochs is selected only on the later Red Sea holdout, and that
choice is then evaluated once on the 2026 Hormuz sample.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).parent
PROJECT = HERE.resolve().parents[1]
OUT = PROJECT / "02_数据_output" / "ml_bunkering_v1"


def load_v8():
    spec = importlib.util.spec_from_file_location("dcrank_v8", HERE / "27_dcrank_searoute_repaired.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


M = load_v8()
EPOCHS = [1, 3, 5]


def model_from_state(state, n_features):
    model = M.UtilityLinear(n_features); model.load_state_dict(state); model.eval(); return model


def fine_tune_snapshots(state, dataset, seed):
    model = model_from_state(state, len(M.FEATURES))
    loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=M.collate,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=5e-4)
    snapshots = {}
    for epoch in range(1, max(EPOCHS)+1):
        model.train()
        for feat, detour, mask, target in loader:
            score = (-detour + M.REFERENCE_B_KNM * model.utility(feat)) / M.TAU_KNM
            score = score.masked_fill(~mask, -1e9)
            loss = torch.nn.functional.cross_entropy(score, target)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
        if epoch in EPOCHS:
            snapshots[epoch] = copy.deepcopy(model.state_dict())
    return snapshots


def main():
    print("Constructing Red Sea choices with the v8 sea-route candidate rule...", flush=True)
    code_area, geo, _, mats = M.resources()
    panel = pd.read_parquet(M.CORE / "legs_panel_v2_areas.parquet")
    panel.mmsi = pd.to_numeric(panel.mmsi, errors="coerce").astype("Int64")
    panel.legStartTime = pd.to_datetime(panel.legStartTime)
    panel.isRefueled = pd.to_numeric(panel.isRefueled).fillna(0)
    early_ev, early_source, _ = M.event_frame(
        panel, code_area, "2023-12-15", "2024-03-31 23:59:59", "RE")
    late_ev, late_source, _ = M.event_frame(
        panel, code_area, "2024-04-01", "2024-06-30 23:59:59", "RL")
    early, early_meta, _ = M.network_candidates(early_ev, geo, mats, "redsea", "RE")
    late, late_meta, _ = M.network_candidates(late_ev, geo, mats, "redsea", "RL")

    lag = M.D.historical_lags(panel)
    training_lag = lag[lag.week.lt("2023-01-01")]
    fallback = training_lag.groupby("area")[M.LAG_COLS].median().to_dict()
    early = M.add_lags_and_frozen(early, lag, fallback).reset_index(drop=True)
    late = M.add_lags_and_frozen(late, lag, fallback).reset_index(drop=True)
    hormuz = pd.read_parquet(OUT / "dcrank_searoute_hormuz_candidates_v8.parquet").reset_index(drop=True)
    hormuz_meta = hormuz[["event_id", "mmsi", "actual", "exposed"]].drop_duplicates("event_id")
    habitual = M.R.history_maps(panel)[2]

    checkpoint = torch.load(OUT / "dcrank_searoute_model_v8.pt", map_location="cpu", weights_only=False)
    mean, std = np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"])
    budget = float(checkpoint["budget_knm"])
    base_models = [model_from_state(s, len(M.FEATURES)) for s in checkpoint["states"]]
    early_ds = M.GroupDataset(early, M.FEATURES, mean, std)
    print(f"Early Red Sea events with recalled truth={len(early_ds):,}; late eligible={len(late_meta):,}", flush=True)

    snapshots = [fine_tune_snapshots(state, early_ds, seed)
                 for state, seed in zip(checkpoint["states"], checkpoint["seeds"])]
    candidates = {0: base_models}
    for epoch in EPOCHS:
        candidates[epoch] = [model_from_state(s[epoch], len(M.FEATURES)) for s in snapshots]

    tuning_rows = []
    for epoch, models in candidates.items():
        u, _ = M.utilities(models, late, M.FEATURES, mean, std)
        nll = M.mean_nll(late, u, budget)
        metric, _, _ = M.score_metrics(f"redsea_epoch_{epoch}", late, late_meta, habitual, u, budget)
        tuning_rows.append({"fine_tune_epochs": epoch, "validation_nll": nll, **metric})
    tuning = pd.DataFrame(tuning_rows)
    selected_epoch = int(tuning.sort_values(["validation_nll", "fine_tune_epochs"]).iloc[0].fine_tune_epochs)
    adapted_models = candidates[selected_epoch]

    rows, details = [], {}
    for split, frame, meta in [("redsea_late", late, late_meta), ("hormuz_post", hormuz, hormuz_meta)]:
        for name, models in [("normal_v8", base_models), ("redsea_adapted_v9", adapted_models)]:
            u, _ = M.utilities(models, frame, M.FEATURES, mean, std)
            metric, detail, _ = M.score_metrics(name, frame, meta, habitual, u, budget)
            rows.append({"split": split, **metric}); details[(split, name)] = detail
    metrics = pd.DataFrame(rows)
    bootstrap = {
        split: M.E.paired_cluster_bootstrap(details[(split, "normal_v8")], details[(split, "redsea_adapted_v9")])
        for split in ["redsea_late", "hormuz_post"]
    }
    tuning.to_csv(OUT / "redsea_transfer_dcrank_tuning_v9.csv", index=False)
    metrics.to_csv(OUT / "redsea_transfer_dcrank_metrics_v9.csv", index=False)
    torch.save({"states": [m.state_dict() for m in adapted_models], "features": M.FEATURES,
                "mean": mean, "std": std, "budget_knm": budget,
                "fine_tune_epochs": selected_epoch, "source_checkpoint": "dcrank_searoute_model_v8.pt"},
               OUT / "redsea_adapted_dcrank_model_v9.pt")
    report = {
        "question": "does early Red Sea port-choice experience improve 2026 Hormuz port ranking?",
        "candidate_rule": "identical v8 AIS sea-route rule; reachable, <=1350 nm extra, closest 40, no truth injection",
        "early_redsea_source_events": early_source,
        "early_redsea_eligible_events": int(early_meta.event_id.nunique()),
        "early_redsea_recalled_events": len(early_ds),
        "late_redsea_source_events": late_source,
        "late_redsea_eligible_events": int(late_meta.event_id.nunique()),
        "selected_epochs_on_redsea_late_only": selected_epoch,
        "tuning": tuning.to_dict("records"), "metrics": metrics.to_dict("records"),
        "paired_vessel_bootstrap": bootstrap,
        "hormuz_labels_used_for_training_or_selection": False,
    }
    (OUT / "redsea_transfer_dcrank_report_v9.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nTUNING\n", tuning[["fine_tune_epochs", "validation_nll", "top1_end_to_end", "top5_end_to_end"]].to_string(index=False))
    print("\nFINAL\n", metrics[["split", "model", "events", "candidate_recall", "top1_end_to_end", "top5_end_to_end", "mrr_end_to_end"]].to_string(index=False))
    print(json.dumps(bootstrap, indent=2))


if __name__ == "__main__":
    main()
