"""Independent audit of the current two-stage v9 system."""
from __future__ import annotations
import ast, hashlib, json
from pathlib import Path
import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).parent
OUT=PROJECT/"02_数据_output"/"ml_bunkering_v1"
FIG=PROJECT/"04_图表_figures"/"ml_bunkering_v1"
AUDIT=PROJECT/"05_审计与清单"


def main():
    checks=[]
    def add(name,passed,detail): checks.append({"check":name,"pass":bool(passed),"detail":detail})
    for name in ["31_occurrence_stage_v9.py","32_redsea_transfer_dcrank_v9.py",
                 "33_two_stage_system_v9.py","34_two_stage_figures_v9.py"]:
        ast.parse((CODE/name).read_text(encoding="utf-8"))
    required=[OUT/"two_stage_occurrence_features_v9.parquet",OUT/"two_stage_occurrence_models_v9.joblib",
              OUT/"two_stage_occurrence_metrics_v9.csv",OUT/"two_stage_occurrence_report_v9.json",
              OUT/"redsea_adapted_dcrank_model_v9.pt",OUT/"redsea_transfer_dcrank_metrics_v9.csv",
              OUT/"redsea_transfer_dcrank_report_v9.json",OUT/"two_stage_system_metrics_v9.csv",
              OUT/"two_stage_system_predictions_v9.parquet",OUT/"two_stage_system_report_v9.json",
              FIG/"fig_two_stage_system_v9.png",FIG/"fig_two_stage_system_v9.pdf"]
    add("all_v9_artifacts_present",all(p.exists() for p in required),[p.name for p in required if not p.exists()])
    occ=json.loads((OUT/"two_stage_occurrence_report_v9.json").read_text(encoding="utf-8"))
    port=json.loads((OUT/"redsea_transfer_dcrank_report_v9.json").read_text(encoding="utf-8"))
    system=json.loads((OUT/"two_stage_system_report_v9.json").read_text(encoding="utf-8"))
    add("no_post_hormuz_labels_in_stage1", "no post-Hormuz" in occ["time_rule"],occ["time_rule"])
    add("no_post_hormuz_labels_in_stage2",port["hormuz_labels_used_for_training_or_selection"] is False,
        port["hormuz_labels_used_for_training_or_selection"])
    ci=occ["incremental_redsea_transfer_on_hormuz"]
    add("redsea_improves_occurrence_pr_auc",ci["pr_auc"]["p025"]>0,ci["pr_auc"])
    add("redsea_improves_occurrence_log_loss",ci["log_loss"]["p975"]<0,ci["log_loss"])
    pci=port["paired_vessel_bootstrap"]["hormuz_post"]
    add("port_transfer_null_result_preserved",pci["top1"]["p025"]<=0<=pci["top1"]["p975"],pci["top1"])
    m=pd.read_csv(OUT/"two_stage_system_metrics_v9.csv").set_index("stage1_model")
    add("two_stage_link_rate_above_95pct",m.positive_port_link_rate.min()>.95,m.positive_port_link_rate.to_dict())
    add("redsea_transfer_improves_joint_nll",
        m.loc["redsea_plus_hormuz_pre","joint_no_refuel_or_actual_port_nll"]
        < m.loc["hormuz_pre_updated","joint_no_refuel_or_actual_port_nll"],
        m.joint_no_refuel_or_actual_port_nll.to_dict())
    add("redsea_transfer_improves_linked_top5",
        m.loc["redsea_plus_hormuz_pre","system_positive_top5"]
        > m.loc["hormuz_pre_updated","system_positive_top5"],m.system_positive_top5.to_dict())
    v8=json.loads((OUT/"dcrank_searoute_audit_report_v8.json").read_text(encoding="utf-8"))
    add("stage2_v8_audit_passes",v8["status"]=="PASS",v8["status"])
    table=pd.DataFrame(checks)
    table.to_csv(OUT/"two_stage_audit_checks_v9.csv",index=False,encoding="utf-8-sig")
    report={"status":"PASS" if table["pass"].all() else "FAIL","passed":int(table["pass"].sum()),
            "total":len(table),"checks":checks}
    (OUT/"two_stage_audit_report_v9.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    files=[PROJECT/"README.md",PROJECT/"00_研究说明"/"当前活动版本说明_v9.md",
           PROJECT/"00_研究说明"/"两阶段加注预测系统报告_v9.md",
           PROJECT/"00_研究说明"/"真实海上航线DCRank完全修复报告_v8.md"]
    files += [CODE/f for f in ["08_shock_port_ranker.py","10_dynamic_port_ranker.py","12_end_to_end_port_ranker.py",
        "27_dcrank_searoute_repaired.py","28_dcrank_searoute_figures.py","29_dcrank_searoute_audit.py",
        "30_hormuz_network_comparison.py","31_occurrence_stage_v9.py","32_redsea_transfer_dcrank_v9.py",
        "33_two_stage_system_v9.py","34_two_stage_figures_v9.py","35_two_stage_audit_v9.py","run_two_stage_v9.py"]]
    files += sorted(OUT.glob("*v9.*"))+sorted(FIG.glob("*v9.*"))
    rows=[]
    for p in dict.fromkeys(files):
        if p.exists() and p.is_file(): rows.append({"relative_path":str(p.relative_to(PROJECT)),"bytes":p.stat().st_size,
            "sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(AUDIT/"当前两阶段系统文件清单_SHA256_v9.csv",index=False,encoding="utf-8-sig")
    print(table.to_string(index=False)); print(f"\nTwo-stage audit: {report['status']} ({report['passed']}/{report['total']})")
    if report["status"]!="PASS": raise SystemExit(1)


if __name__=="__main__": main()
