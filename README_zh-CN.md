# 两阶段船舶加注预测项目（当前版本 v9）

当前活动版本为两阶段系统 v9：第一阶段预测航次是否加注，第二阶段使用真实海路 DCRank v8 预测条件加注港。旧版代码、结果、图表和研究说明保存在 `99_旧版本归档/20260921_v8之前版本`。

## 当前主结论

- 距离使用 TRE-10CHOKE 的 984 节点 AIS 海上航道网络最短路径，不使用经纬度大圆距离。
- 2021—2022 年训练，2023 年选择全局距离额度，2026 年霍尔木兹时期检验。
- 学习得到的全局距离额度为 194.45 海里。
- 集装箱船 23,296 个事件：Top-1 为 62.66%，Top-5 为 93.13%。
- 原油船 2,565 个事件：Top-1 为 49.04%，Top-5 为 81.99%。
- 第一阶段在245,912条霍尔木兹时期航次上的PR-AUC为46.52%；红海经验带来1.05个百分点的稳定增益。
- 两阶段连接后，在全部真实加注事件中同时“检出加注且港口Top-1正确”的比例为34.85%，Top-5为44.08%。
- 红海经验没有显著提高第二阶段的具体港口排序，因此只用于第一阶段。
- 动态距离额度没有优于全局额度；正常网络与当前霍尔木兹受阻网络的预测结果基本相同。这两项作为研究边界报告，不作为正面提升。

完整系统解释见 `00_研究说明/两阶段加注预测系统报告_v9.md`；第二阶段技术细节见 `00_研究说明/真实海上航线DCRank完全修复报告_v8.md`。

## 一键复现

```powershell
python -m pip install -r requirements.txt
python 03_代码_pipeline/ml_bunkering_v1/run_two_stage_v9.py
```

运行顺序为：重训真实海路v8、重训是否加注模型、红海迁移检验、连接两阶段、生成图件并执行双重审计。

## 当前活动代码

- `26_occurrence_features_v9.py`：第一阶段特征面板构建（自归档 `02_occurrence_ml.py` 恢复，输出 `two_stage_occurrence_features_v9.parquet`）。
- `08_shock_port_ranker.py`：v8 所需的港口资源和历史映射基础模块。
- `10_dynamic_port_ranker.py`：v8 所需的严格滞后变量和 AIS 航道网络基础模块。
- `12_end_to_end_port_ranker.py`：v8 所需的端到端指标和船舶层面重复抽样基础模块。
- `27_dcrank_searoute_repaired.py`：v8 主训练与验证程序。
- `28_dcrank_searoute_figures.py`：v8 图件。
- `29_dcrank_searoute_audit.py`：v8 独立审计和文件清单。
- `30_hormuz_network_comparison.py`：正常网络与霍尔木兹受阻网络对照。
- `31_occurrence_stage_v9.py`：第一阶段是否加注及红海迁移检验。
- `32_redsea_transfer_dcrank_v9.py`：第二阶段红海至霍尔木兹迁移检验。
- `33_two_stage_system_v9.py`：连接两个阶段并计算系统指标。
- `34_two_stage_figures_v9.py`：两阶段系统图件。
- `36_ml_diagnostics_figures_v9.py`：机器学习通用诊断图（ROC/PR 曲线、校准图与混淆矩阵、置换特征重要性、recall@k 曲线、效用权重与 MRS）。
- `37_geo_network_figures_v9.py`：地理组图（全球航道网络与咽喉要道、霍尔木兹情景断走廊、评估队列空间覆盖、加注港地理）。
- `35_two_stage_audit_v9.py`：两阶段系统独立审计。
- `run_two_stage_v9.py`：最新版唯一运行入口（先构建特征面板，再依次执行 27–35）。

前四个文件虽然保留早期编号或是自归档恢复，但它们是当前系统直接导入/执行的必要依赖，不代表旧结果仍在活动目录中。

## 当前活动数据

- `02_数据_output/refuel_panel_v2`：历史训练面板，是 v8 的规范化输入，不是待发表的旧结果。
- `02_数据_output/refuel_panel_v2/red_sea_historical_exposure_v2.parquet`：红海历史暴露标记，第一阶段特征面板输入（自归档恢复）。
- `02_数据_output/ml_bunkering_v1/hormuz_container_legs_2026.parquet`：集装箱船规范化源数据。
- `02_数据_output/ml_bunkering_v1/crude_tanker_legs_2024_2026_v1.parquet`：原油船规范化源数据；文件名中的 v1 是数据制备版本，仍是当前 v8 使用的唯一源文件。
- `02_数据_output/ml_bunkering_v1/dcrank_searoute_*_v8.*`：v8 模型、候选集、指标和审计结果。
- `02_数据_output/ml_bunkering_v1/hormuz_network_*_v8.*`：最新版航道网络对照结果。
- `02_数据_output/ml_bunkering_v1/two_stage_*_v9.*`：第一阶段、系统连接和v9审计结果。
- `02_数据_output/ml_bunkering_v1/redsea_*_v9.*`：真实海路框架下的跨冲击迁移检验。

外部航道、港口坐标和船舶资料保留在 `01_数据_input`，均为当前复现所需输入，不移动到旧版归档。

## 归档原则

归档只改变文件位置，不删除文件。归档清单记录原路径、新路径、字节数和 SHA-256；如需复核旧实验，可以从归档目录恢复，但旧文件不应再被论文正文或最新版程序引用。
