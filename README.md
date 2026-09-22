# Two-Stage Vessel Bunkering Prediction (v9) with Sea-Route DCRank (v8)

This repository accompanies the manuscript on predicting future vessel bunkering
(bunkering port choice and refuelling occurrence). The system predicts **(i)**
whether an upcoming voyage will involve a bunkering call (stage 1) and **(ii)**
which port will be used, conditional on a call (stage 2).

Stage 2 ranks candidate ports with a discrete-choice-inspired ranking model
(DCRank) in which "detour" is measured as the **shortest-path distance over an
AIS-derived sea-lane network** (TRE-10CHOKE, 984 nodes), not great-circle
distance. Stage 1 is a gradient-boosted occurrence model over voyage-level
features, including a cross-shock transfer setting (Red Sea data used as prior
experience for the Hormuz test period).

A Chinese version of the original project README is kept as
[README_zh-CN.md](README_zh-CN.md).

## Headline results

Evaluation protocol: train 2021–2022, select the global distance budget on 2023,
blind test on the 2026 Hormuz period (2026-02-28 to 2026-07-22). The learned
global distance budget is **194.45 nmi**.

| Setting | Events | Top-1 | Top-3 | Top-5 | MRR |
|---|---|---|---|---|---|
| Container ships, stage 2 end-to-end | 23,296 | 62.66% | 84.83% | 93.13% | 0.754 |
| Crude tankers, stage 2 end-to-end | 2,565 | 49.04% | 73.33% | 81.99% | 0.629 |

Stage 1 on 245,912 Hormuz-period voyages: PR-AUC **46.52%** (pre-specified
transfer model), ROC-AUC 88.29%; Red Sea experience adds a stable +1.05
percentage points of PR-AUC over the no-transfer comparator (45.46%).

Connecting both stages, the share of all real bunkering events where the system
both *detects* the call and puts the *correct port* at Top-1 is **34.85%**
(Top-5: 44.08%).

Reported as study boundaries rather than improvements: a dynamic
(event-dependent) distance budget did not outperform the global budget, and
predictions under the normal network were essentially identical to predictions
under the Hormuz-blocked network (see
`02_数据_output/ml_bunkering_v1/hormuz_network_scenario_report_v8.json`).
Bootstrap confidence intervals for all comparisons are stored in the
`*_report_v8.json` / `*_report_v9.json` files.

## Repository layout

```
.
├── README.md                          # this file
├── README_zh-CN.md                    # original Chinese project README
├── DATA_AVAILABILITY.md               # what is shipped and what is withheld, and why
├── MANIFEST.csv                       # path, byte size and SHA-256 of every shipped file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # keeps withheld raw data out of version control
├── 00_研究说明/                        # system documentation (stage-1/2 design, DCRank repair notes)
├── 02_数据_output/ml_bunkering_v1/     # publishable results: metrics, reports, model weights
├── 03_代码_pipeline/ml_bunkering_v1/   # all analysis code (the active v9/v8 pipeline)
└── 04_图表_figures/ml_bunkering_v1/    # manuscript figures (PDF + PNG)
```

The pipeline locates the project root via
`Path(__file__).resolve().parents[2]`, so the numbered directory layout above is
part of the interface: keep `03_代码_pipeline/ml_bunkering_v1` two levels below
the repository root.

## Quick start

```powershell
python -m pip install -r requirements.txt
python 03_代码_pipeline/ml_bunkering_v1/run_two_stage_v9.py
```

`run_two_stage_v9.py` rebuilds the feature panel, retrains sea-route DCRank v8,
retrains the stage-1 occurrence model, runs the Red Sea transfer check, connects
both stages, regenerates the figures, and re-runs both audits:

```
26 → 27 → 31 → 32 → 33 → 28 → 34 → 36 → 37 → 29 → 35
```

Two auxiliary scripts are not part of the one-command run and must be executed
separately:

- `30_hormuz_network_comparison.py` — normal vs. Hormuz-blocked network comparison.
- `38_reviewer_controls_v9.py` — cold-start and oracle control experiments.

The geographic figures additionally need Natural Earth 1:110m land/country
shapefiles, which the script reads from the Cartopy data directory
(`~/.local/share/cartopy/shapefiles` on Linux/macOS or the equivalent per-user
location on Windows); see `37_geo_network_figures_v9.py`.

### Required input data

The pipeline reads two families of input files that are **not included in this
repository** because they are licensed/commercial data and contain vessel- and
port-level records (see [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)):

1. `01_数据_input/外部参考/` — AIS sea-lane network caches (TRE-10CHOKE),
   the vessel–operator registry (TRE-09MEET), and port/strait geography tables.
2. `02_数据_output/refuel_panel_v2/` — the historical legs panel and choice
   panel (vessel-level voyage records), the port–area mapping table, and the
   vessel-level Red Sea exposure table.

To reproduce, restore both directories from the internal archive into the
layout expected by the scripts; every path is resolved relative to the
repository root.

## Code

| File | Role |
|---|---|
| `run_two_stage_v9.py` | Sole entry point for one-command reproduction |
| `26_occurrence_features_v9.py` | Stage-1 voyage-level feature panel |
| `27_dcrank_searoute_repaired.py` | Sea-route DCRank v8: main training and validation |
| `31_occurrence_stage_v9.py` | Stage-1 occurrence model + Red Sea transfer check |
| `32_redsea_transfer_dcrank_v9.py` | Stage-2 Red Sea → Hormuz transfer check |
| `33_two_stage_system_v9.py` | Connects both stages; computes system metrics |
| `28_dcrank_searoute_figures.py` | Stage-2 figures |
| `34_two_stage_figures_v9.py` | Two-stage system figures |
| `36_ml_diagnostics_figures_v9.py` | ML diagnostics (ROC/PR, calibration, permutation importance, recall@k, utility weights, MRS) |
| `37_geo_network_figures_v9.py` | Geographic figures (global sea-lane network, Hormuz corridor, coverage, bunkering ports) |
| `29_dcrank_searoute_audit.py` | Independent audit + file inventory for stage 2 |
| `35_two_stage_audit_v9.py` | Independent audit for the two-stage system |
| `30_hormuz_network_comparison.py` | Normal vs. blocked network comparison |
| `38_reviewer_controls_v9.py` | Cold-start and oracle reviewer controls |
| `08_shock_port_ranker.py` | Shared port resources and historical mappings (imported by the modules above) |
| `10_dynamic_port_ranker.py` | Strictly lagged variables and AIS network base module |
| `12_end_to_end_port_ranker.py` | End-to-end metrics and vessel-level resampling |

## Results shipped in this repository

Aggregated metrics and reports (CSV/JSON), trained model weights, and the
manuscript figures only. All event-level and vessel-level tables — candidate
sets, prediction tables, feature panels, legs panels, port mappings and exposure
tables — are withheld; see [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).
`MANIFEST.csv` records the SHA-256 of every shipped file so the package can be
verified after download.

## Reproducibility notes

- Deterministic sampling and fixed seeds are used throughout; seed-stability
  results are in `dcrank_searoute_seed_stability_v8.csv` (Top-1 std ≈ 0.0002).
- Two independent audit scripts (`29`, `35`) re-check data leakage, candidate
  recall, thresholds and metric definitions; their checklists are shipped.
- Model files are small parameter objects (scikit-learn
  `HistGradientBoostingClassifier` and PyTorch state dicts) and contain no raw
  records.
