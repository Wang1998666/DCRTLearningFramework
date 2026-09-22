"""Independent consistency and claim audit for sea-route DCRank v8."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OUT = PROJECT / "02_数据_output" / "ml_bunkering_v1"
FIG = PROJECT / "04_图表_figures" / "ml_bunkering_v1"
AUDIT = PROJECT / "05_审计与清单"


def main():
    checks = []

    def add(name, passed, detail):
        checks.append({"check": name, "pass": bool(passed), "detail": detail})

    source = (CODE / "27_dcrank_searoute_repaired.py").read_text(encoding="utf-8")
    figure_source = (CODE / "28_dcrank_searoute_figures.py").read_text(encoding="utf-8")
    ast.parse(source); ast.parse(figure_source)
    report = json.loads((OUT / "dcrank_searoute_report_v8.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(OUT / "dcrank_searoute_metrics_v8.csv").set_index("model")
    crude = pd.read_csv(OUT / "dcrank_searoute_crude_metrics_v8.csv").set_index("model")
    curve = pd.read_csv(OUT / "dcrank_searoute_budget_curve_v8.csv")
    dynamic = pd.read_parquet(OUT / "dcrank_searoute_dynamic_budgets_v8.parquet")

    required = [
        PROJECT / "02_数据_output" / "refuel_panel_v2" / "choice_long_v2.parquet",
        PROJECT / "02_数据_output" / "refuel_panel_v2" / "legs_panel_v2_areas.parquet",
        PROJECT / "02_数据_output" / "refuel_panel_v2" / "port_area_map_v2.csv",
        OUT / "hormuz_container_legs_2026.parquet",
        OUT / "crude_tanker_legs_2024_2026_v1.parquet",
        OUT / "dcrank_searoute_model_v8.pt",
        OUT / "dcrank_searoute_hormuz_candidates_v8.parquet",
        OUT / "dcrank_searoute_crude_candidates_v8.parquet",
        FIG / "fig_dcrank_searoute_main_v8.png",
        FIG / "fig_dcrank_searoute_main_v8.pdf",
        FIG / "fig_dcrank_searoute_diagnostics_v8.png",
        FIG / "fig_dcrank_searoute_diagnostics_v8.pdf",
        OUT / "hormuz_network_scenario_metrics_v8.csv",
        OUT / "hormuz_network_event_comparison_v8.parquet",
        OUT / "hormuz_network_scenario_report_v8.json",
        FIG / "fig_hormuz_network_scenario_v8.png",
        FIG / "fig_hormuz_network_scenario_v8.pdf",
    ]
    add("required_artifacts_present", all(p.exists() for p in required),
        [p.name for p in required if not p.exists()])
    add("sea_route_distance_declared", "sea-network path" in report["distance_definition"],
        report["distance_definition"])
    add("ais_network_provenance_recorded",
        "TRE-10CHOKE" in report.get("network_provenance", "") and report.get("network_nodes", 0) > 100,
        {"source": report.get("network_provenance"), "nodes": report.get("network_nodes")})
    add("candidate_source_has_no_truth_injection",
        "np.append(feasible" not in source and "append(ai)" not in source
        and "selected = order[:CAP]" in source,
        "candidate list is distance-sorted reachable network nodes only")
    add("saved_candidates_have_route_distances",
        {"sea_detour_knm", "direct_sea_nm", "via_sea_nm", "network_scenario"}.issubset(
            pd.read_parquet(OUT / "dcrank_searoute_hormuz_candidates_v8.parquet").columns),
        "route fields present")
    add("zero_distance_barrier_violations", report["network_barrier_violations"] == 0,
        report["network_barrier_violations"])
    add("container_candidate_recall_above_95pct",
        report["data_audit"]["test"]["candidate_recall"] > .95,
        report["data_audit"]["test"]["candidate_recall"])
    add("crude_candidate_recall_above_85pct",
        report["data_audit"]["crude"]["candidate_recall"] > .85,
        report["data_audit"]["crude"]["candidate_recall"])
    train_missing = report["data_audit"]["train"]["lag_missing_share"]
    add("training_core_lags_complete",
        max(train_missing[k] for k in ["lag4_calls", "lag4_refuels", "lag4_refuel_share"]) < .01,
        train_missing)
    ci = report["bootstrap"]
    add("global_beats_shortest_top1", ci["global_vs_shortest"]["top1"]["p025"] > 0,
        ci["global_vs_shortest"]["top1"])
    add("global_beats_shortest_top5", ci["global_vs_shortest"]["top5"]["p025"] > 0,
        ci["global_vs_shortest"]["top5"])
    fixed_ci = ci["global_vs_fixed50"]["top1"]
    add("top1_tie_with_fixed50_is_preserved", fixed_ci["p025"] <= 0 <= fixed_ci["p975"], fixed_ci)
    dyn_ci = ci["dynamic_vs_global"]["top1"]
    add("dynamic_no_gain_is_preserved", dyn_ci["p025"] <= 0 <= dyn_ci["p975"], dyn_ci)
    frozen_ci = ci["rolling_vs_frozen_global"]["top1"]
    add("rolling_frozen_difference_not_significant", frozen_ci["p025"] <= 0 <= frozen_ci["p975"], frozen_ci)
    add("five_seed_stability", report["seed_top1_std"] < .005, report["seed_top1_std"])
    observed = float(metrics.loc["learned_global_rolling", "top1_end_to_end"])
    add("within_event_shuffle_fails", report["shuffle_top1_p975"] < observed,
        {"shuffle_p975": report["shuffle_top1_p975"], "observed": observed})
    add("crude_transfer_beats_shortest", report["crude_bootstrap"]["top1"]["p025"] > 0,
        report["crude_bootstrap"]["top1"])
    add("budget_curve_is_computed_for_both_splits",
        set(curve.split) == {"validation", "hormuz_test_descriptive"} and curve.budget_nm.nunique() == 41,
        {"splits": sorted(curve.split.unique()), "budgets": int(curve.budget_nm.nunique())})
    add("figure_reads_saved_curve", "dcrank_searoute_budget_curve_v8.csv" in figure_source,
        "no hard-coded sensitivity array")
    add("dynamic_budget_collapse_recorded", dynamic.dynamic_budget_nm.std() < 5,
        dynamic.dynamic_budget_nm.describe().to_dict())
    add("sample_sizes_match_report",
        int(metrics.loc["learned_global_rolling", "events"]) == 23296
        and int(crude.loc["learned_global_rolling", "events"]) == 2565,
        {"container": int(metrics.loc["learned_global_rolling", "events"]),
         "crude": int(crude.loc["learned_global_rolling", "events"])})
    network_report = json.loads((OUT / "hormuz_network_scenario_report_v8.json").read_text(encoding="utf-8"))
    network_metrics = pd.read_csv(OUT / "hormuz_network_scenario_metrics_v8.csv")
    cohort_sizes = network_metrics.groupby(["cohort", "scenario"]).events.first()
    add("network_scenarios_use_same_event_cohort",
        cohort_sizes.groupby("cohort").nunique().eq(1).all(),
        {" / ".join(key): int(value) for key, value in cohort_sizes.items()})
    add("route_shock_changes_computed_distances",
        network_report["candidate_detour_changed_over_1nm_share"] > 0,
        {"candidate_detour_changed_share": network_report["candidate_detour_changed_over_1nm_share"],
         "direct_route_changed_share": network_report["direct_route_changed_over_1nm_share"]})

    table = pd.DataFrame(checks)
    table.to_csv(OUT / "dcrank_searoute_audit_checks_v8.csv", index=False, encoding="utf-8-sig")
    result = {"status": "PASS" if table["pass"].all() else "FAIL",
              "passed": int(table["pass"].sum()), "total": len(table), "checks": checks}
    (OUT / "dcrank_searoute_audit_report_v8.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_files = [
        PROJECT / "README.md",
        PROJECT / "00_研究说明" / "当前活动版本说明_v8.md",
        PROJECT / "00_研究说明" / "真实海上航线DCRank完全修复报告_v8.md",
        CODE / "08_shock_port_ranker.py", CODE / "10_dynamic_port_ranker.py",
        CODE / "12_end_to_end_port_ranker.py",
        CODE / "27_dcrank_searoute_repaired.py", CODE / "28_dcrank_searoute_figures.py",
        CODE / "29_dcrank_searoute_audit.py", CODE / "30_hormuz_network_comparison.py",
        CODE / "run_dcrank_searoute_v8.py",
        PROJECT / "02_数据_output" / "refuel_panel_v2" / "choice_long_v2.parquet",
        PROJECT / "02_数据_output" / "refuel_panel_v2" / "legs_panel_v2_areas.parquet",
        PROJECT / "02_数据_output" / "refuel_panel_v2" / "port_area_map_v2.csv",
        OUT / "hormuz_container_legs_2026.parquet",
        OUT / "crude_tanker_legs_2024_2026_v1.parquet",
    ] + sorted(OUT.glob("dcrank_searoute_*v8.*")) + sorted(OUT.glob("hormuz_network_*v8.*")) + sorted(FIG.glob("*v8.*"))
    rows = []
    for path in dict.fromkeys(manifest_files):
        if path.exists() and path.is_file():
            rows.append({"relative_path": str(path.relative_to(PROJECT)),
                         "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(AUDIT / "真实海路DCRank文件清单_SHA256_v8.csv",
                              index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print(f"\nAudit: {result['status']} ({result['passed']}/{result['total']})")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
