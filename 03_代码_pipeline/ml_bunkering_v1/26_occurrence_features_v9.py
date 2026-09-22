"""Stage-1 occurrence feature panel (v9).

Rebuilds ``two_stage_occurrence_features_v9.parquet``, the voyage-level feature
panel consumed by ``31_occurrence_stage_v9.py``.  The construction is restored
verbatim from the archived ``02_occurrence_ml.py`` (pre-v8 pipeline); only the
output file name changed when the two-stage v9 system was assembled.  The
model-fitting part of the archived script now lives in
``31_occurrence_stage_v9.py`` and is deliberately not duplicated here.

Construction summary
--------------------
* Historical legs (2021--2024 Red Sea era) come from
  ``refuel_panel_v2/legs_panel_v2_areas.parquet``; 2026 Hormuz legs come from
  ``ml_bunkering_v1/hormuz_container_legs_2026.parquet``.  The two sources are
  concatenated under a ``source_period`` tag, and every sequential feature is
  computed strictly within (source_period, vessel) so that no future or
  cross-source information can leak backwards.
* Vessel master attributes (operator, teu, power_kw_total, build_year) are
  joined from the TRE-09MEET vessel table.
* Historical base rates are shrunk toward the 2021--2022 global refuelling
  rate, never fitted on any post-2022 label.
* The Red Sea historical exposure flag is joined from
  ``refuel_panel_v2/red_sea_historical_exposure_v2.parquet``; the Hormuz
  exposure flag marks vessels that crossed the HORMUZ node before the shock.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
BASE_OUT = PROJECT / "02_数据_output" / "refuel_panel_v2"
OUT = PROJECT / "02_数据_output" / "ml_bunkering_v1"
OLD_VESSELS = PROJECT / "01_数据_input" / "外部参考" / "TRE-09MEET" / "vessels_container.parquet"
TARGET = OUT / "two_stage_occurrence_features_v9.parquet"

# Shrinkage strength per encoding (toward the 2021--2022 global rate).
RATE_ALPHAS = {"origin": 80, "dest": 80, "od": 40, "mmsi": 30, "operator": 100}


def load_panel():
    old = pd.read_parquet(
        BASE_OUT / "legs_panel_v2_areas.parquet",
        columns=["mmsi", "legStartTime", "legEndTime", "legStartPortCode", "legEndPortCode",
                 "refuel_any", "sailDistance", "isFullLoad", "cargoLoad", "berthDuration",
                 "averageSpeed", "year"])
    old = old.rename(columns={"refuel_any": "y", "legStartPortCode": "origin",
                              "legEndPortCode": "dest"})
    old["source_period"] = "historical"
    exposure = pd.read_parquet(BASE_OUT / "red_sea_historical_exposure_v2.parquet",
                               columns=["mmsi", "historically_exposed", "error"])
    exposure = exposure[exposure.error.eq("")]
    exposure.mmsi = pd.to_numeric(exposure.mmsi, errors="coerce").astype("Int64")
    old = old.merge(exposure[["mmsi", "historically_exposed"]], on="mmsi", how="left")
    old["redsea_exposed"] = old.historically_exposed.fillna(0).astype("int8")
    new = pd.read_parquet(
        OUT / "hormuz_container_legs_2026.parquet",
        columns=["mmsi", "legStartTime", "legEndTime", "legStartPortCode", "legEndPortCode",
                 "isRefueled", "sailDistance", "isFullLoad", "cargoLoad", "berthDuration",
                 "averageSpeed", "legCrossNodeList"])
    new = new.rename(columns={"isRefueled": "y", "legStartPortCode": "origin",
                              "legEndPortCode": "dest"})
    new["year"] = pd.to_datetime(new.legStartTime, errors="coerce").dt.year
    new["source_period"] = "hormuz2026"
    new.mmsi = pd.to_numeric(new.mmsi, errors="coerce").astype("Int64")
    pre_hormuz = new.legStartTime.lt("2026-02-28") & new.legCrossNodeList.astype(str).str.contains(
        "HORMUZ", case=False, na=False)
    exposed_ships = set(new.loc[pre_hormuz, "mmsi"].dropna())
    new["hormuz_exposed"] = new.mmsi.isin(exposed_ships).astype("int8")
    new["redsea_exposed"] = 0
    d = pd.concat([old, new], ignore_index=True)
    d.mmsi = pd.to_numeric(d.mmsi, errors="coerce").astype("Int64")
    d.y = pd.to_numeric(d.y, errors="coerce").fillna(0).clip(0, 1).astype("int8")
    d.legStartTime = pd.to_datetime(d.legStartTime, errors="coerce")
    d.legEndTime = pd.to_datetime(d.legEndTime, errors="coerce")
    d = d.dropna(subset=["mmsi", "legStartTime", "origin", "dest"])
    d = d.sort_values(["source_period", "mmsi", "legStartTime", "legEndTime"]).drop_duplicates(
        ["source_period", "mmsi", "legStartTime", "legEndTime"], keep="last").reset_index(drop=True)
    vessels = pd.read_parquet(OLD_VESSELS,
                              columns=["mmsi", "operator", "teu", "power_kw_total", "build_year"])
    vessels.mmsi = pd.to_numeric(vessels.mmsi, errors="coerce").astype("Int64")
    d = d.merge(vessels.drop_duplicates("mmsi"), on="mmsi", how="left")
    d["hormuz_exposed"] = d.get("hormuz_exposed", 0).fillna(0).astype("int8")
    return d


def add_sequential_features(d):
    key = [d.source_period, d.mmsi]
    d["prior_y"] = d.groupby(["source_period", "mmsi"]).y.shift(1)
    d["prior3_rate"] = d.groupby(["source_period", "mmsi"]).y.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    d["prior10_rate"] = d.groupby(["source_period", "mmsi"]).y.transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    marker = d.legStartTime.where(d.y.eq(1))
    last = marker.groupby(key).ffill()
    d["prior_refuel_time"] = last.groupby(key).shift()
    d["days_since_refuel"] = (d.legStartTime - d.prior_refuel_time).dt.total_seconds() / 86400
    d["lag_nm"] = d.groupby(["source_period", "mmsi"]).sailDistance.shift(1)
    block = d.groupby(["source_period", "mmsi"]).y.shift(1).fillna(0).groupby(key).cumsum()
    d["nm_since_refuel"] = d.lag_nm.fillna(0).groupby([d.source_period, d.mmsi, block]).cumsum()
    d["month_sin"] = np.sin(2 * np.pi * d.legStartTime.dt.month / 12)
    d["month_cos"] = np.cos(2 * np.pi * d.legStartTime.dt.month / 12)
    d["ship_age"] = (d.legStartTime.dt.year - d.build_year).clip(0, 80)
    return d


def smoothed_map(train, cols, alpha, global_rate):
    g = train.groupby(cols).y.agg(["size", "sum"])
    return ((g["sum"] + alpha * global_rate) / (g["size"] + alpha)).to_dict()


def add_train_encodings(d):
    tr = d[d.legStartTime.between("2021-01-01", "2022-12-31 23:59:59")]
    gp = float(tr.y.mean())
    maps = {
        "origin_rate": (["origin"], RATE_ALPHAS["origin"]),
        "dest_rate": (["dest"], RATE_ALPHAS["dest"]),
        "od_rate": (["origin", "dest"], RATE_ALPHAS["od"]),
        "vessel_rate": (["mmsi"], RATE_ALPHAS["mmsi"]),
        "operator_rate": (["operator"], RATE_ALPHAS["operator"]),
    }
    for name, (cols, a) in maps.items():
        mp = smoothed_map(tr, cols, a, gp)
        if len(cols) == 1:
            d[name] = d[cols[0]].map(mp).fillna(gp)
        else:
            d[name] = [mp.get(tuple(x), gp) for x in d[cols].itertuples(index=False, name=None)]
    d["global_rate"] = gp
    return d


def build():
    return add_train_encodings(add_sequential_features(load_panel()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    # Match the distributed artifact: legs whose vessel is absent from the
    # vessel master carry the literal string "None".  Applied after encoding so
    # that the operator base rate for these legs still falls back to the global
    # rate exactly as in the published panel.
    d["operator"] = d["operator"].fillna("None")
    d.to_parquet(TARGET, index=False)
    print(f"Wrote {TARGET} rows={len(d):,} cols={len(d.columns)}")


if __name__ == "__main__":
    main()
