"""Geographic figures for the two-stage v9 system.

All geospatial inputs are read from the project's own input folder:
  - ``01_数据_input/外部参考/TRE-10CHOKE/03_数据_初步验证/网络缓存``
    (graph_nodes.parquet: H3 cell coordinates; baseline.parquet: 6,953 corridors
    with their observed cell paths; new_cell_loading.parquet: AIS cell loading)
  - ``01_数据_input/外部参考/TRE-10CHOKE/03_数据_初步验证/临界性指标``
    (strait_corridor_results.parquet: per-strait corridor severance status)
  - ``01_数据_input/外部参考/港口地理`` (port coordinates, chokepoint segments,
    port-to-area mapping)

Figures
-------
fig_geo_network_v9    (a) global AIS sea-lane network, chokepoints, bunkering hubs
                      (b) Strait of Hormuz zoom with the 142 severed corridors
fig_geo_coverage_v9   spatial coverage of the 2026 evaluation cohorts
                      (a) container events (b) crude oil tanker events
fig_geo_hubs_v9       (a) leading bunkering areas by event count
                      (b) hub geography on a regional map
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

PROJECT = Path(__file__).resolve().parents[2]
GEO = PROJECT / "01_数据_input" / "外部参考"
TRE = GEO / "TRE-10CHOKE" / "03_数据_初步验证"
CACHE = TRE / "网络缓存"
CRIT = TRE / "临界性指标"
PORTS_DIR = GEO / "港口地理"
OUT = PROJECT / "02_数据_output" / "ml_bunkering_v1"
FIG = PROJECT / "04_图表_figures" / "ml_bunkering_v1"
FIG.mkdir(parents=True, exist_ok=True)
LATEX_FIG = PROJECT / "05_手稿" / "Latex_ML"
LATEX_FIG.mkdir(parents=True, exist_ok=True)

NE_SHAPE = (Path.home() / ".local" / "share" / "cartopy" / "shapefiles"
            / "natural_earth" / "cultural" / "ne_110m_admin_0_countries.shp")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8,
    "figure.dpi": 200,
})

# Academic palette
OCEAN = "#F5F8FA"
LAND = "#E6E9ED"
COAST = "#B8C1CC"
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]

NAVY = "#1E40AF"
DARK_NAVY = "#1E3A8A"
GREEN = "#059669"
AMBER = "#D97706"
DARK_AMBER = "#92400E"
CRIMSON = "#DC2626"
DARK_CRIMSON = "#991B1B"
GREY = "#64748B"
LIGHT_GREY = "#94A3B8"

CHOKEPOINTS = ["HORMUZ_STRAIT", "MALACCA_STRAIT", "SINGAPORE_STRAIT", "SUEZ_CANAL",
               "MANDAB_STRAIT", "GIBRALTAR_STRAIT", "DOVER_STRAIT", "PANAMA_CANAL",
               "BOSPORUS_STRAIT"]
HUBS = {"SGSGP": "Singapore", "AEFUJ": "Fujairah", "AEJAL": "Jebel Ali",
        "LKCOL": "Colombo", "NLROT": "Rotterdam"}


def save(fig, stem):
    for d in [FIG, LATEX_FIG]:
        fig.savefig(d / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem} to FIG and LATEX_FIG")


def basemap(ax, extent, land):
    ax.set_facecolor(OCEAN)
    land.plot(ax=ax, color=LAND, edgecolor=COAST, linewidth=0.35, zorder=0)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])


def load_geo():
    nodes = pd.read_parquet(CACHE / "graph_nodes.parquet")
    base = pd.read_parquet(CACHE / "baseline.parquet")
    load = pd.read_parquet(CACHE / "new_cell_loading.parquet",
                           columns=["lat", "lon", "load_freq_base"])
    straits = pd.read_csv(PORTS_DIR / "海峡数据.csv", encoding="utf-8-sig")
    ports = pd.read_csv(PORTS_DIR / "港口数据.csv", encoding="utf-8-sig")
    ports["portCode"] = ports.portCode.astype(str)
    amap = pd.read_csv(PROJECT / "02_数据_output" / "refuel_panel_v2" / "port_area_map_v2.csv",
                       encoding="utf-8-sig")
    amap["portCode"] = amap.portCode.astype(str)
    return nodes, base, load, straits, ports, amap


def corridor_segments(base, nodes, lon, lat, mask=None, max_cells=4000):
    """LineCollection of corridor paths (observed H3 cell trajectories)."""
    sub = base if mask is None else base[mask]
    segs = []
    for cells in sub["cells"]:
        if cells is None or len(cells) == 0 or len(cells) > max_cells:
            continue
        c = np.asarray(cells)
        x, y = lon[c], lat[c]
        jump = np.where(np.abs(np.diff(x)) > 180)[0] + 1
        for part in np.split(np.arange(len(x)), jump):
            if len(part) > 1:
                segs.append(np.column_stack([x[part], y[part]]))
    return segs


def area_centroids(ports, amap):
    return amap.groupby("area_30")[["lat", "lon"]].mean()


# --------------------------------------------------------------- Figure 10
def fig_geo_network(nodes, base, load, straits, ports, amap, land):
    lon, lat = nodes.lon.to_numpy(), nodes.lat.to_numpy()

    # Aspect ratio balance: (360/133) / (20/15) = 2.707 / 1.333 = 2.03
    fig = plt.figure(figsize=(12.2, 4.4))
    gs = GridSpec(1, 2, width_ratios=[2.05, 1.0], wspace=0.10)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # (a) global network
    basemap(ax1, [-180, 180, -58, 75], land)
    net = load[load.load_freq_base > 0]
    if len(net) > 400000:
        net = net.sample(400000, random_state=42)
    ax1.scatter(net.lon, net.lat, c=np.log10(net.load_freq_base + 1), s=0.25, lw=0,
                cmap="Blues", alpha=0.35, vmin=0, rasterized=True, zorder=1)
    segs = corridor_segments(base, nodes, lon, lat)
    ax1.add_collection(LineCollection(segs, colors=LIGHT_GREY, linewidths=0.30, alpha=0.45, zorder=2))

    srow = straits.set_index("nodeCode")
    # Custom non-overlapping annotation parameters for Chokepoints
    choke_annot = {
        "PANAMA_CANAL": ((-10, -12), "right"),
        "DOVER_STRAIT": ((-12, 10), "right"),
        "GIBRALTAR_STRAIT": ((-8, -8), "right"),
        "BOSPORUS_STRAIT": ((7, 6), "left"),
        "SUEZ_CANAL": ((-6, 6), "right"),
        "MANDAB_STRAIT": ((-6, -8), "right"),
        "HORMUZ_STRAIT": ((7, 9), "left"),
        "MALACCA_STRAIT": ((-8, 7), "right"),
        "SINGAPORE_STRAIT": ((8, 6), "left"),
    }

    for nc in CHOKEPOINTS:
        if nc not in srow.index:
            continue
        r = srow.loc[nc]
        ax1.plot([r.startLon, r.endLon], [r.startLat, r.endLat], color=CRIMSON, lw=2.5,
                 solid_capstyle="round", zorder=4, alpha=0.95)
        offset, ha = choke_annot.get(nc, ((4, 4), "left"))
        ax1.annotate(r.nameEn, (r.lon, r.lat), xytext=offset, textcoords="offset points",
                     fontsize=6.8, color=DARK_CRIMSON, fontweight="bold", ha=ha, zorder=6,
                     path_effects=HALO)

    # Custom non-overlapping annotation parameters for Hubs
    hub_annot = {
        "NLROT": ((8, 8), "left"),
        "AEFUJ": ((-6, -9), "right"),
        "AEJAL": None,  # in global panel, Fujairah represents the Persian Gulf entrance hub
        "LKCOL": ((7, -7), "left"),
        "SGSGP": ((8, -8), "left"),
    }

    for code, name in HUBS.items():
        row = ports[ports.portCode.eq(code)]
        if not len(row):
            continue
        r = row.iloc[0]
        ax1.scatter([r.lon], [r.lat], marker="*", s=110, c=AMBER, edgecolors=DARK_AMBER,
                    linewidths=0.5, zorder=6)
        cfg = hub_annot.get(code)
        if cfg is not None:
            offset, ha = cfg
            ax1.annotate(name, (r.lon, r.lat), xytext=offset, textcoords="offset points",
                         fontsize=7.2, color=DARK_AMBER, fontweight="bold", ha=ha, zorder=6,
                         path_effects=HALO)

    ax1.set_title("(a) Global AIS-derived sea-lane network (984 nodes, 6,953 corridors)",
                  fontsize=10.5, fontweight="bold")
    handles = [Line2D([], [], color=LIGHT_GREY, lw=1.2, alpha=0.7, label="Observed corridor"),
               Line2D([], [], color=CRIMSON, lw=2.2, label="Major chokepoint"),
               Line2D([], [], color="none", marker="*", ms=10, mfc=AMBER, mec=DARK_AMBER,
                      label="Bunkering hub"),
               Line2D([], [], color="none", marker="o", ms=4, mfc="#2563EB", mec="none", alpha=0.6,
                      label="AIS traffic density")]
    ax1.legend(handles=handles, loc="lower left", fontsize=7.2, frameon=True,
               framealpha=0.92, edgecolor="#CBD5E1")

    # (b) Hormuz zoom
    ext = [44, 64, 19, 34]
    basemap(ax2, ext, land)
    keep = []
    for i, cells in enumerate(base["cells"]):
        if cells is None or len(cells) == 0:
            continue
        c = np.asarray(cells)
        if (lon[c].min() < ext[1] and lon[c].max() > ext[0]
                and lat[c].min() < ext[3] and lat[c].max() > ext[2]):
            keep.append(i)
    keep = np.array(keep)
    segs = corridor_segments(base, nodes, lon, lat, mask=base.index.isin(base.index[keep]))
    ax2.add_collection(LineCollection(segs, colors=LIGHT_GREY, linewidths=0.65, alpha=0.6, zorder=2))

    st = pd.read_parquet(CRIT / "strait_corridor_results.parquet")
    hz = st[st.strait.eq("HORMUZ_STRAIT")]
    severed = base[base.corridor_id.isin(hz.corridor_id)]
    segs_hz = corridor_segments(severed, nodes, lon, lat)
    ax2.add_collection(LineCollection(segs_hz, colors=CRIMSON, linewidths=1.4, alpha=0.9, zorder=3))

    r = srow.loc["HORMUZ_STRAIT"]
    ax2.plot([r.startLon, r.endLon], [r.startLat, r.endLat], color=DARK_CRIMSON, lw=3.8,
             solid_capstyle="round", zorder=5)
    ax2.annotate("Strait of Hormuz", (r.lon, r.lat), xytext=(7, 10), textcoords="offset points",
                 fontsize=8.0, color=DARK_CRIMSON, fontweight="bold", ha="left", zorder=6,
                 path_effects=HALO)

    # Hub annotations in panel (b)
    # Jebel Ali: (55.06, 25.01) placed southwest into UAE
    # Fujairah: (56.37, 25.17) placed northeast into clear Gulf of Oman waters
    p_jal = ports[ports.portCode.eq("AEJAL")].iloc[0]
    ax2.scatter([p_jal.lon], [p_jal.lat], marker="*", s=130, c=AMBER, edgecolors=DARK_AMBER,
                linewidths=0.5, zorder=6)
    ax2.annotate("Jebel Ali", (p_jal.lon, p_jal.lat), xytext=(-8, -10), textcoords="offset points",
                 fontsize=7.8, color=DARK_AMBER, fontweight="bold", ha="right", zorder=6,
                 path_effects=HALO)

    p_fuj = ports[ports.portCode.eq("AEFUJ")].iloc[0]
    ax2.scatter([p_fuj.lon], [p_fuj.lat], marker="*", s=130, c=AMBER, edgecolors=DARK_AMBER,
                linewidths=0.5, zorder=6)
    ax2.annotate("Fujairah", (p_fuj.lon, p_fuj.lat), xytext=(10, 6), textcoords="offset points",
                 fontsize=7.8, color=DARK_AMBER, fontweight="bold", ha="left", zorder=6,
                 path_effects=HALO)

    # Persian Gulf ports
    gulf = ports[(ports.lon.between(ext[0], ext[1])) & (ports.lat.between(ext[2], ext[3]))]
    ax2.scatter(gulf.lon, gulf.lat, s=8, c="#64748B", alpha=0.50, zorder=4, lw=0)
    ax2.set_title(f"(b) Hormuz severance scenario ({len(hz)} severed corridors)",
                  fontsize=10.5, fontweight="bold")
    handles = [Line2D([], [], color=LIGHT_GREY, lw=1.2, label="Baseline corridor"),
               Line2D([], [], color=CRIMSON, lw=1.8, label="Severed corridor"),
               Line2D([], [], color="none", marker="*", ms=10, mfc=AMBER, mec=DARK_AMBER,
                      label="Bunkering hub")]
    ax2.legend(handles=handles, loc="lower left", fontsize=7.2, frameon=True,
               framealpha=0.92, edgecolor="#CBD5E1")
    save(fig, "fig_geo_network_v9")


# --------------------------------------------------------------- Figure 11
def fig_geo_coverage(nodes, ports, amap, land):
    cent = area_centroids(ports, amap)
    con = pd.read_parquet(OUT / "dcrank_searoute_hormuz_candidates_v8.parquet").drop_duplicates("event_id")
    crude = pd.read_parquet(OUT / "dcrank_searoute_crude_candidates_v8.parquet").drop_duplicates("event_id")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 7.2))

    panels = [
        (ax1, con, f"(a) Container voyages ({len(con):,} events)"),
        (ax2, crude, f"(b) Crude oil tanker voyages ({len(crude):,} events)")
    ]

    pc = ports.set_index("portCode")
    for ax, ev, title in panels:
        basemap(ax, [-180, 180, -55, 75], land)

        # Origins
        codes_orig = ev["legStartPortCode"].astype(str)
        hit_orig = codes_orig[codes_orig.isin(pc.index)]
        ax.scatter(pc.loc[hit_orig, "lon"], pc.loc[hit_orig, "lat"], s=6.5, c=LIGHT_GREY,
                   alpha=0.45, lw=0, zorder=2, rasterized=True)

        # Destinations
        codes_dest = ev["legEndPortCode"].astype(str)
        hit_dest = codes_dest[codes_dest.isin(pc.index)]
        ax.scatter(pc.loc[hit_dest, "lon"], pc.loc[hit_dest, "lat"], s=6.5, c=NAVY,
                   alpha=0.50, lw=0, zorder=3, rasterized=True)

        # Realized bunkering areas (top 60)
        counts = ev.actual.value_counts()
        top = counts.head(60)
        lat_c = cent.reindex(top.index).lat
        lon_c = cent.reindex(top.index).lon
        ok = lat_c.notna() & lon_c.notna()
        ax.scatter(lon_c[ok], lat_c[ok], s=18 + 180 * np.sqrt(top[ok] / top.max()),
                   c=CRIMSON, alpha=0.65, lw=0.5, edgecolors=DARK_CRIMSON, zorder=4)

        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)
        handles = [Line2D([], [], color="none", marker="o", ms=5.0, mfc=LIGHT_GREY, mec="none",
                          label="Origin port"),
                   Line2D([], [], color="none", marker="o", ms=5.0, mfc=NAVY, mec="none",
                          label="Destination port"),
                   Line2D([], [], color="none", marker="o", ms=6.5, mfc=CRIMSON, mec=DARK_CRIMSON,
                          label="Realized bunkering area (top 60)")]
        ax.legend(handles=handles, loc="lower left", fontsize=8.0, frameon=True,
                  framealpha=0.92, edgecolor="#CBD5E1")

    fig.tight_layout(h_pad=1.8)
    save(fig, "fig_geo_coverage_v9")


# --------------------------------------------------------------- Figure 12
def fig_geo_hubs(ports, amap, land):
    cent = area_centroids(ports, amap)
    con = pd.read_parquet(OUT / "dcrank_searoute_hormuz_candidates_v8.parquet").drop_duplicates("event_id")
    crude = pd.read_parquet(OUT / "dcrank_searoute_crude_candidates_v8.parquet").drop_duplicates("event_id")
    c_counts = con.actual.value_counts()
    t_counts = crude.actual.value_counts()
    top = c_counts.head(12).index

    # Enriched hub names for clear academic interpretation
    HUB_NAMES = {
        "SG@335": "Singapore",
        "CN@300": "Shanghai",
        "CN@720": "Guangzhou",
        "KR@511": "Busan",
        "HK@1255": "Hong Kong",
        "CN@1926": "Ningbo",
        "ES@88": "Algeciras",
        "BE@164": "Antwerp",
        "NL@79": "Rotterdam",
        "CN@2600": "Tianjin",
        "MY@2123": "Port Klang",
        "CN@1349": "Xiamen",
        "CN@2234": "Qingdao",
        "LK@674": "Colombo",
        "TH@1036": "Laem Chabang",
    }
    members = amap.groupby("area_30").portCode.apply(list)

    def label(area):
        name = HUB_NAMES.get(area)
        n = len(members.get(area, []))
        if name:
            if n > 1:
                return f"{area} — {name} ({n} ports)"
            return f"{area} — {name}"
        return f"{area} ({n} ports)"

    # Aspect ratio balance: bar chart on left, focused regional map on right
    fig = plt.figure(figsize=(12.2, 4.5))
    gs = GridSpec(1, 2, width_ratios=[1.05, 1.45], wspace=0.16)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # (a) Horizontal bar chart
    y = np.arange(len(top))
    ax1.barh(y + 0.2, c_counts.reindex(top).fillna(0), height=0.38, color=NAVY,
             edgecolor=DARK_NAVY, linewidth=0.4, label="Container vessels")
    ax1.barh(y - 0.2, t_counts.reindex(top).fillna(0), height=0.38, color=GREEN,
             edgecolor="#065F46", linewidth=0.4, label="Crude oil tankers")
    ax1.set_yticks(y)
    ax1.set_yticklabels([label(a) for a in top], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Bunkering events (2026 Hormuz evaluation window)", fontsize=9.5)
    ax1.set_title("(a) Leading bunkering areas by event count", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="lower right", fontsize=8)

    # (b) Focused regional map spanning Hormuz to East Asia
    # Extent: [42, 134, -6, 44]
    ext = [42, 134, -6, 44]
    basemap(ax2, ext, land)

    srow = pd.read_csv(PORTS_DIR / "海峡数据.csv", encoding="utf-8-sig").set_index("nodeCode")
    r = srow.loc["HORMUZ_STRAIT"]
    ax2.plot([r.startLon, r.endLon], [r.startLat, r.endLat], color=DARK_CRIMSON, lw=3.8,
             solid_capstyle="round", zorder=5)
    ax2.annotate("Strait of Hormuz", (r.lon, r.lat), xytext=(7, 9), textcoords="offset points",
                 fontsize=8.0, color=DARK_CRIMSON, fontweight="bold", ha="left", zorder=6,
                 path_effects=HALO)

    show = c_counts.head(20).index
    for area in show:
        if area not in cent.index:
            continue
        n_c, n_t = c_counts.get(area, 0), t_counts.get(area, 0)
        la, lo = cent.loc[area, "lat"], cent.loc[area, "lon"]
        # Container bubble (filled Navy)
        ax2.scatter([lo], [la], s=14 + 140 * np.sqrt(n_c / c_counts.max()), c=NAVY, alpha=0.65,
                    lw=0.45, edgecolors=DARK_NAVY, zorder=4)
        # Crude tanker bubble (ringed Green)
        if n_t >= 40:
            ax2.scatter([lo], [la], s=14 + 140 * np.sqrt(n_t / max(t_counts.max(), 1)), c="none",
                        lw=1.5, edgecolors=GREEN, zorder=5)

    # Handcrafted non-overlapping annotations for key regional hubs
    hub_regional_annot = {
        "SG@335": ("Singapore", (8, -8), "left"),
        "MY@2123": ("Port Klang", (-8, 6), "right"),
        "LK@674": ("Colombo", (8, -7), "left"),
        "TH@1036": ("Laem Chabang", (-8, 0), "right"),
        "HK@1255": ("Hong Kong", (-8, -8), "right"),
        "CN@1349": ("Xiamen", (-8, 6), "right"),
        "CN@1926": ("Ningbo", (8, -4), "left"),
        "CN@300": ("Shanghai", (-8, 6), "right"),
        "CN@2234": ("Qingdao", (-8, 4), "right"),
        "KR@511": ("Busan", (8, 2), "left"),
        "CN@2600": ("Tianjin", (-8, 6), "right"),
    }

    for area, (hub_title, offset, ha) in hub_regional_annot.items():
        if area in cent.index:
            la, lo = cent.loc[area, "lat"], cent.loc[area, "lon"]
            if ext[0] <= lo <= ext[1] and ext[2] <= la <= ext[3]:
                ax2.annotate(hub_title, (lo, la), xytext=offset, textcoords="offset points",
                             fontsize=7.0, color="#1F2937", fontweight="bold", ha=ha, zorder=6,
                             path_effects=HALO)

    ax2.set_title("(b) Regional bunkering hub distribution", fontsize=10.5, fontweight="bold")
    handles = [Line2D([], [], color="none", marker="o", ms=6, mfc=NAVY, mec=DARK_NAVY,
                      label="Container bunkering area (top 20)"),
               Line2D([], [], color="none", marker="o", ms=6, mfc="none", mec=GREEN, mew=1.5,
                      label="Crude tanker area ($\\geq$ 40 events)"),
               Line2D([], [], color=DARK_CRIMSON, lw=3, label="Strait of Hormuz")]
    ax2.legend(handles=handles, loc="lower left", fontsize=7.2, frameon=True,
               framealpha=0.92, edgecolor="#CBD5E1")

    save(fig, "fig_geo_hubs_v9")


def main():
    print("loading geospatial inputs ...", flush=True)
    nodes, base, load, straits, ports, amap = load_geo()
    land = gpd.read_file(NE_SHAPE)
    print("figure 10: sea-lane network ...", flush=True)
    fig_geo_network(nodes, base, load, straits, ports, amap, land)
    print("figure 11: evaluation coverage ...", flush=True)
    fig_geo_coverage(nodes, ports, amap, land)
    print("figure 12: bunkering hubs ...", flush=True)
    fig_geo_hubs(ports, amap, land)
    print("geographic figures complete.")


if __name__ == "__main__":
    main()
