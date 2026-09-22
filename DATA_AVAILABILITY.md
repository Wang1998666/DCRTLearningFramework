# Data availability

This repository ships the analysis **code**, the **aggregated results**, and the
**figures**. The underlying input data are commercial, licensed datasets and
contain vessel-level and port-level records; they are withheld from the public
release. Nothing in this repository contains AIS messages, MMSI identifiers,
vessel names or operators, per-voyage port calls, or commercial port reference
tables.

## Shipped (non-sensitive)

- `03_代码_pipeline/ml_bunkering_v1/` — all analysis scripts.
- `00_研究说明/` — system documentation (Chinese).
- `04_图表_figures/ml_bunkering_v1/` — manuscript figures (PDF/PNG).
- `02_数据_output/ml_bunkering_v1/` — aggregated result files only:
  - metric tables and audit checklists (`*_metrics_*.csv`,
    `*_audit_checks_*.csv`, `*_controls_*.csv`, `*_curve_*.csv`,
    `*_sensitivity_*.csv`, `*_stability_*.csv`, `*_shuffle_*.csv`,
    `*_weights_*.csv`, `*_tuning_*.csv`);
  - machine-readable reports with bootstrap confidence intervals
    (`*_report_*.json`);
  - trained model parameters (`dcrank_searoute_model_v8.pt`,
    `redsea_adapted_dcrank_model_v9.pt`,
    `two_stage_occurrence_models_v9.joblib`);
  - `dcrank_searoute_dynamic_budgets_v8.parquet` — per-event distance-budget
    values keyed only by opaque sequential event IDs, with no vessel, port or
    time identifiers.

## Withheld (sensitive)

### Raw external reference data (`01_数据_input/`)

| Category | Content | Reason withheld |
|---|---|---|
| AIS sea-lane network | TRE-10CHOKE network caches, corridor/strait shortest-path results, network node loading | Derived from licensed commercial AIS |
| Vessel registry | Vessel–operator mapping (TRE-09MEET processing result) | Vessel-level identifiers |
| Port geography | Port coordinates, port–area and strait tables | Commercial port reference data |

### Voyage-level and event-level outputs (`02_数据_output/`)

| File / directory | Content | Reason withheld |
|---|---|---|
| `refuel_panel_v2/legs_panel_v2_areas.parquet` | 2.1 M historical voyage legs with MMSI, port codes/names, AIS file provenance | AIS-derived per-voyage records |
| `refuel_panel_v2/choice_long_v2.parquet` | Port-choice training panel with MMSI and operator names | Vessel-level attribute data |
| `refuel_panel_v2/port_area_map_v2.csv` | Port-to-area mapping | Commercial port reference data |
| `refuel_panel_v2/red_sea_historical_exposure_v2.parquet` | Per-vessel Red Sea exposure flags (MMSI keyed) | Vessel-level identifiers |
| `ml_bunkering_v1/hormuz_container_legs_2026.parquet` | 321,892 container-ship legs with MMSI and refuelling ports | AIS-derived per-voyage records |
| `ml_bunkering_v1/crude_tanker_legs_2024_2026_v1.parquet` | 124,924 tanker legs with MMSI and refuelling ports | AIS-derived per-voyage records |
| `ml_bunkering_v1/two_stage_occurrence_features_v9.parquet` | 2.46 M voyage feature rows with MMSI, operator, origin/destination | AIS-derived per-voyage records |
| `ml_bunkering_v1/dcrank_searoute_hormuz_candidates_v8.parquet` | 738 K candidate-port rows per bunkering event | Vessel- and port-level records |
| `ml_bunkering_v1/dcrank_searoute_crude_candidates_v8.parquet` | 86 K candidate-port rows (tanker events) | Vessel- and port-level records |
| `ml_bunkering_v1/two_stage_occurrence_predictions_v9.parquet` | Per-voyage occurrence predictions | Voyage-level identifiers |
| `ml_bunkering_v1/two_stage_system_predictions_v9.parquet` | Per-voyage two-stage predictions with actual/predicted ports | Voyage-level identifiers |
| `ml_bunkering_v1/hormuz_network_event_comparison_v8.parquet` | Per-event normal vs. blocked network comparison | Voyage-level identifiers |

## Access

The withheld files remain in the authors' internal project archive under the
paths above. Researchers with a legitimate need may request access to the
aggregated outputs (already shipped) or ask the corresponding author about data
access terms for the licensed inputs. `.gitignore` in this repository blocks the
withheld directories so that they are never committed by accident.
