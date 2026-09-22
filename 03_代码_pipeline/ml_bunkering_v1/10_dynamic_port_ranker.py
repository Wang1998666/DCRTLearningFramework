"""Add strictly lagged port activity/congestion and physical shock-network features.

All rolling features are shifted by one complete ISO week.  TRE-10CHOKE and the
raw 2026 source are read only; this script writes only below this repaired project.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location("ranker_v1",HERE/"08_shock_port_ranker.py")
R=importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
CHOKE=R.PROJECT/"01_数据_input"/"外部参考"/"TRE-10CHOKE"/"03_数据_初步验证"


def complete_lags(calls, refuels, start, end):
    """Return port-week values calculated from weeks w-4,...,w-1 only."""
    weeks=pd.date_range(pd.Timestamp(start).to_period("W").start_time,
                        pd.Timestamp(end).to_period("W").start_time,freq="7D")
    areas=sorted(set(calls.area.dropna()).union(refuels.area.dropna()))
    grid=pd.MultiIndex.from_product([areas,weeks],names=["area","week"]).to_frame(index=False)
    ca=(calls.groupby(["area","week"]).agg(calls=("area","size"),berth_h=("berth_h","median")).reset_index())
    re=refuels.groupby(["area","week"]).size().rename("refuels").reset_index()
    z=grid.merge(ca,on=["area","week"],how="left").merge(re,on=["area","week"],how="left")
    z[["calls","refuels"]]=z[["calls","refuels"]].fillna(0)
    z=z.sort_values(["area","week"])
    g=z.groupby("area",sort=False)
    z["lag4_calls"]=g.calls.transform(lambda x:x.shift(1).rolling(4,min_periods=1).sum())
    z["lag4_refuels"]=g.refuels.transform(lambda x:x.shift(1).rolling(4,min_periods=1).sum())
    z["lag4_berth_h"]=g.berth_h.transform(lambda x:x.shift(1).rolling(4,min_periods=1).median())
    z["lag4_refuel_share"]=z.lag4_refuels/(z.lag4_calls+1)
    return z[["area","week","lag4_calls","lag4_refuels","lag4_refuel_share","lag4_berth_h"]]


def add_lags(long, lag, fallback):
    x=long.copy(); x["week"]=x.call_time.dt.to_period("W").dt.start_time
    x=x.merge(lag,left_on=["alt","week"],right_on=["area","week"],how="left").drop(columns="area")
    for c in ["lag4_calls","lag4_refuels","lag4_refuel_share","lag4_berth_h"]:
        x[c+"_missing"]=x[c].isna().astype("int8")
        x[c]=x[c].fillna(x.alt.map(fallback[c])).fillna(0)
    x["log_lag4_calls"]=np.log1p(x.lag4_calls); x["log_lag4_refuels"]=np.log1p(x.lag4_refuels)
    return x


def historical_lags(panel):
    calls=panel[["call_time","area_30","berthDuration"]].rename(columns={"area_30":"area","berthDuration":"berth_h"}).dropna(subset=["call_time","area"])
    calls["week"]=calls.call_time.dt.to_period("W").dt.start_time
    refs=panel.loc[panel.refuel_area_30.eq(1),["call_time","area_30"]].rename(columns={"area_30":"area"})
    refs["week"]=refs.call_time.dt.to_period("W").dt.start_time
    return complete_lags(calls,refs,panel.call_time.min(),panel.call_time.max())


def hormuz_lags(raw, code_area):
    calls=raw[["legStartTime","legEndPortCode","berthDuration"]].copy()
    calls["area"]=calls.legEndPortCode.astype(str).map(code_area); calls["berth_h"]=pd.to_numeric(calls.berthDuration,errors="coerce")
    calls["week"]=calls.legStartTime.dt.to_period("W").dt.start_time
    refs=raw.loc[raw.isRefueled.eq(1),["legStartTime","refuelPortCodes"]].copy()
    refs["area"]=refs.refuelPortCodes.astype(str).str.split(",").str[0].map(code_area)
    refs["week"]=refs.legStartTime.dt.to_period("W").dt.start_time
    return complete_lags(calls.dropna(subset=["area"]),refs.dropna(subset=["area"]),raw.legStartTime.min(),raw.legStartTime.max())


def graph_matrices(geo):
    base=pd.read_parquet(CHOKE/"网络缓存"/"baseline.parquet",
                         columns=["corridor_id","origin_port","dest_port","base_km"])
    ports=sorted(set(base.origin_port).union(base.dest_port)); ix={p:i for i,p in enumerate(ports)}; n=len(ports)
    def matrix(update=None):
        b=base.copy(); b["w"]=b.base_km
        b["p0"]=np.minimum(b.origin_port,b.dest_port); b["p1"]=np.maximum(b.origin_port,b.dest_port)
        if update is not None:
            u=update.set_index("corridor_id")
            hit=b.corridor_id.isin(u.index); affected=b.loc[hit,["p0","p1","corridor_id"]].copy()
            affected["status"]=affected.corridor_id.map(u.status); affected["new_km"]=affected.corridor_id.map(u.new_km)
            # A sea lane is bidirectional even if only one observed OD direction
            # was tagged.  Replace/remove both directions of each affected pair.
            repl=affected.groupby(["p0","p1"]).apply(
                lambda q:q.loc[q.status.eq("rerouted"),"new_km"].dropna().min() if q.status.eq("rerouted").any() else np.inf,
                include_groups=False).to_dict()
            for pair,w in repl.items(): b.loc[b.p0.eq(pair[0])&b.p1.eq(pair[1]),"w"]=w
        # Collapse observed OD directions to one physical bidirectional lane.
        b=b.replace([np.inf,-np.inf],np.nan).dropna(subset=["w"]).groupby(["p0","p1"],as_index=False).w.min()
        mat=csr_matrix((b.w,([ix[x] for x in b.p0],[ix[x] for x in b.p1])),shape=(n,n))
        # Corridors are observed OD movements, while the underlying sea lane is
        # navigable in both directions; undirected routing avoids coverage being
        # determined by which direction happened to appear in the AIS sample.
        return dijkstra(mat,directed=False)
    scen=pd.read_parquet(CHOKE/"组合失效情景"/"scenario_corridor_results.parquet")
    red=scen[scen.scenario.eq("S2")][["corridor_id","status","new_km"]]
    st=pd.read_parquet(CHOKE/"临界性指标"/"strait_corridor_results.parquet")
    hor=st[st.strait.eq("HORMUZ_STRAIT")][["corridor_id","status","new_km"]]
    # Map every observed/code-area location to its nearest node with known coordinates.
    g=geo.loc[geo.index.intersection(ports)].copy(); xyz=np.c_[np.cos(np.radians(g.lat))*np.cos(np.radians(g.lon)),np.cos(np.radians(g.lat))*np.sin(np.radians(g.lon)),np.sin(np.radians(g.lat))]
    tree=cKDTree(xyz); gp=g.index.to_numpy()
    return ports,ix,matrix(),matrix(red),matrix(hor),tree,gp


def nearest_codes(lat,lon,tree,gp):
    lat=np.asarray(lat,float); lon=np.asarray(lon,float)
    xyz=np.c_[np.cos(np.radians(lat))*np.cos(np.radians(lon)),np.cos(np.radians(lat))*np.sin(np.radians(lon)),np.sin(np.radians(lat))]
    return gp[tree.query(xyz)[1]]


def add_network(long, kind, geo, mats):
    ports,ix,normal,red,hor,tree,gp=mats; shock={"normal":normal,"redsea":red,"hormuz":hor}[kind]
    x=long.copy(); o=x.legStartPortCode.astype(str); d=x.legEndPortCode.astype(str)
    # Unknown codes use nearest coordinate when available; candidate areas always use their centroid.
    def node_for_codes(s):
        out=s.where(s.isin(ix))
        miss=out.isna()&s.isin(geo.index)
        if miss.any(): out.loc[miss]=nearest_codes(geo.loc[s[miss],"lat"],geo.loc[s[miss],"lon"],tree,gp)
        return out
    on=node_for_codes(o); dn=node_for_codes(d)
    an=pd.Series(nearest_codes(x.alt.map(R.AREA_ATTR.lat),x.alt.map(R.AREA_ATTR.lon),tree,gp),index=x.index)
    valid=on.notna()&dn.notna(); oi=on.map(ix).fillna(0).astype(int).to_numpy(); di=dn.map(ix).fillna(0).astype(int).to_numpy(); ai=an.map(ix).astype(int).to_numpy()
    broute=normal[oi,ai]+normal[ai,di]; sroute=shock[oi,ai]+shock[ai,di]
    direct=normal[oi,di]; path_valid=valid.to_numpy()&np.isfinite(broute)&np.isfinite(direct)
    bdet=np.zeros(len(x)); bdet[path_valid]=np.maximum(broute[path_valid]-direct[path_valid],0)
    unreachable=(~np.isfinite(sroute))&np.isfinite(broute)&path_valid
    penalty=np.zeros(len(x)); both=path_valid&np.isfinite(sroute); penalty[both]=sroute[both]-broute[both]
    finite=penalty[both]
    cap=float(np.quantile(finite,0.99)) if len(finite) else 5000.
    x["network_base_detour_knm"]=bdet/1000
    x["shock_network_extra_knm"]=np.clip(penalty,0,cap)/1000
    x.loc[unreachable,"shock_network_extra_knm"]=cap/1000
    x["shock_network_unreachable"]=unreachable.astype("int8"); x["network_missing"]=(~path_valid).astype("int8")
    return x


def fit_metrics(name,tr,features,tests,habitual):
    m=HistGradientBoostingClassifier(max_iter=180,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=80,
        l2_regularization=1.5,random_state=20260921).fit(tr[features],tr.y)
    rows=[]; details={}
    for tn,te in tests:
        rows.append({"test":tn,**R.rank_metrics(name,m,features,te,habitual)})
        z=te[["event_id","mmsi","y"]].copy(); z["score"]=m.predict_proba(te[features])[:,1]
        z["rank"]=z.groupby("event_id").score.rank(method="first",ascending=False)
        details[tn]=z[z.y.eq(1)][["event_id","mmsi","rank"]]
    return rows,details


def paired_vessel_bootstrap(a,b,reps=2000):
    """Paired uncertainty, resampling vessels to respect repeated events."""
    z=a.merge(b,on=["event_id","mmsi"],suffixes=("_a","_b"))
    z["d_top1"]=(z.rank_b.le(1).astype(float)-z.rank_a.le(1).astype(float))
    z["d_top5"]=(z.rank_b.le(5).astype(float)-z.rank_a.le(5).astype(float))
    z["d_mrr"]=1/z.rank_b-1/z.rank_a
    agg=z.groupby("mmsi").agg(n=("event_id","size"),d_top1=("d_top1","sum"),d_top5=("d_top5","sum"),d_mrr=("d_mrr","sum"))
    arr=agg.to_numpy(); rng=np.random.default_rng(20260921); out={}
    for j,k in enumerate(["top1","top5","mrr"],start=1):
        draws=[]
        for _ in range(reps):
            q=arr[rng.integers(0,len(arr),len(arr))]; draws.append(q[:,j].sum()/q[:,0].sum())
        point=z["d_"+k].mean(); lo,hi=np.quantile(draws,[.025,.975])
        out[k]={"difference":float(point),"cluster_bootstrap_p025":float(lo),"cluster_bootstrap_p975":float(hi)}
    return out


def main():
    R.OUT.mkdir(parents=True,exist_ok=True); code_area,geo,cent,vessels=R.resources()
    panel=pd.read_parquet(R.CORE/"legs_panel_v2_areas.parquet"); panel=panel.merge(vessels,on="mmsi",how="left"); panel.operator=panel.operator.fillna("Unknown")
    panel=panel.sort_values(["mmsi","legStartTime","legEndTime"]).reset_index(drop=True); panel["event_id"]=panel.index.astype("int64")
    R.VESSEL_PREF,R.ORIGIN_PREF,habitual=R.history_maps(panel)
    choice=pd.read_parquet(R.CORE/"choice_long_v2.parquet")
    R.AREA_ATTR=(choice.groupby("alt").agg(dwell_10h=("dwell_10h","first"),log_calls=("log_calls","first"),propensity=("propensity","first"),events=("y","sum")).join(cent,how="inner").dropna())
    pref=choice.groupby(["operator","alt"]).pref.first().to_dict()
    exposure=pd.read_parquet(R.CORE/"red_sea_historical_exposure_v2.parquet",columns=["mmsi","historically_exposed","error"])
    exposure=exposure[exposure.error.eq("")]; exposure.mmsi=pd.to_numeric(exposure.mmsi,errors="coerce").astype("Int64")
    normal,red=R.historical_long(panel,vessels,exposure[["mmsi","historically_exposed"]])
    destinations=panel.set_index("event_id").legEndPortCode; normal["legEndPortCode"]=normal.event_id.map(destinations); red["legEndPortCode"]=red.event_id.map(destinations)
    hraw=R.hormuz_long(code_area,geo,cent,vessels,pref)
    raw=pd.read_parquet(R.OUT/"hormuz_container_legs_2026.parquet"); raw.legStartTime=pd.to_datetime(raw.legStartTime)
    # Recover destination for sampled H events by joining the identifying fields.
    key=raw.sort_values(["mmsi","legStartTime","legEndTime"]).drop_duplicates(["mmsi","legStartTime","legEndTime"],keep="last")
    kd=key[["mmsi","legStartTime","legEndPortCode"]].drop_duplicates(["mmsi","legStartTime"])
    hraw=hraw.merge(kd,left_on=["mmsi","call_time"],right_on=["mmsi","legStartTime"],how="left").drop(columns="legStartTime")

    lagh=historical_lags(panel); lag26=hormuz_lags(raw,code_area)
    fallback=lagh.groupby("area")[["lag4_calls","lag4_refuels","lag4_refuel_share","lag4_berth_h"]].median().to_dict()
    normal=add_lags(normal,lagh,fallback); red=add_lags(red,lagh,fallback); hraw=add_lags(hraw,lag26,fallback)
    mats=graph_matrices(geo)
    normal=add_network(normal,"normal",geo,mats); red=add_network(red,"redsea",geo,mats); hraw=add_network(hraw,"hormuz",geo,mats)
    red_train=red[red.call_time.lt("2024-04-01")]; red_test=red[red.call_time.ge("2024-04-01")]
    training=pd.concat([normal,red_train],ignore_index=True)
    base=["detour_knm","dwell_10h","log_calls","propensity","pref","vessel_pref_count","origin_pref_count"]
    shock=base+["shock_proximity","inside_region","days_x_proximity","exposed_x_proximity"]
    dyn=shock+["log_lag4_calls","log_lag4_refuels","lag4_refuel_share","lag4_berth_h"]
    net=dyn+["network_base_detour_knm","shock_network_extra_knm","shock_network_unreachable","network_missing"]
    tests=[("redsea_late_holdout",red_test),("hormuz_external",hraw)]
    rows=[]; detail={}
    for name,features in [("shock_aware_baseline",shock),("strict_lag_dynamic",dyn),("dynamic_plus_network",net)]:
        rr,dd=fit_metrics(name,training,features,tests,habitual); rows+=rr; detail[name]=dd
    result=pd.DataFrame(rows); result.to_csv(R.OUT/"dynamic_port_ranker_metrics_v2.csv",index=False)
    comparisons={}
    for tn,_ in tests:
        comparisons[tn]={
            "dynamic_vs_shock_baseline":paired_vessel_bootstrap(detail["shock_aware_baseline"][tn],detail["strict_lag_dynamic"][tn]),
            "network_vs_dynamic":paired_vessel_bootstrap(detail["strict_lag_dynamic"][tn],detail["dynamic_plus_network"][tn]),
            "full_vs_shock_baseline":paired_vessel_bootstrap(detail["shock_aware_baseline"][tn],detail["dynamic_plus_network"][tn])}
    audit={"historical_lag_rows":len(lagh),"hormuz_lag_rows":len(lag26),"graph_ports":len(mats[0]),
           "hormuz_network_missing_rate":float(hraw.network_missing.mean()),"hormuz_shock_unreachable_rate":float(hraw.shock_network_unreachable.mean())}
    report={"strict_time_rule":"week w uses only weeks w-4 through w-1","network_source":"TRE-10CHOKE baseline, S2 Red Sea, HORMUZ_STRAIT closure","metrics":result.to_dict("records"),"paired_vessel_bootstrap":comparisons,"feature_audit":audit,"warning":"Predictive validation; not a causal counterfactual."}
    (R.OUT/"dynamic_port_ranker_report_v2.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(result.to_string(index=False)); print("lag rows",len(lagh),len(lag26),"graph ports",len(mats[0]))


if __name__=="__main__": main()
