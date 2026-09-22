"""Rank likely bunkering ports with transferable, shock-aware features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support

PROJECT=Path(__file__).resolve().parents[2]
CORE=PROJECT/"02_数据_output"/"refuel_panel_v2"; OUT=PROJECT/"02_数据_output"/"ml_bunkering_v1"
OLD=PROJECT/"01_数据_input"/"外部参考"/"TRE-09MEET"/"vessels_container.parquet"
PORTS=PROJECT/"01_数据_input"/"外部参考"/"港口地理"/"港口数据.csv"


def gc_nm(lat1,lon1,lat2,lon2):
    a1,a2=np.radians(lat1),np.radians(lat2); dl=np.radians(np.asarray(lon2)-np.asarray(lon1))
    h=np.sin((a2-a1)/2)**2+np.cos(a1)*np.cos(a2)*np.sin(dl/2)**2
    return 2*3440.065*np.arcsin(np.sqrt(np.clip(h,0,1)))


def resources():
    amap=pd.read_csv(CORE/"port_area_map_v2.csv",encoding="utf-8-sig",usecols=["portCode","area_30"])
    amap.portCode=amap.portCode.astype(str); code_area=amap.drop_duplicates("portCode").set_index("portCode").area_30
    geo=pd.read_csv(PORTS,encoding="utf-8-sig",usecols=["portCode","lat","lon"]).dropna().drop_duplicates("portCode")
    geo.portCode=geo.portCode.astype(str); geo=geo.merge(amap,on="portCode",how="inner")
    cent=geo.groupby("area_30")[["lat","lon"]].mean()
    vessels=pd.read_parquet(OLD,columns=["mmsi","operator"]); vessels.mmsi=pd.to_numeric(vessels.mmsi,errors="coerce").astype("Int64")
    vessels.operator=vessels.operator.astype(str).str.strip().replace({"nan":"Unknown","δ֪":"Unknown"})
    return code_area,geo.set_index("portCode")[["lat","lon"]],cent,vessels.drop_duplicates("mmsi")


def history_maps(panel):
    tr=panel[panel.year.isin([2021,2022]) & panel.refuel_area_30.eq(1)]
    vessel=tr.groupby(["mmsi","area_30"]).size(); origin=tr.groupby(["legStartPortCode","area_30"]).size()
    habitual=(tr.groupby(["mmsi","area_30"]).size().rename("n").reset_index()
              .sort_values(["mmsi","n"],ascending=[True,False]).drop_duplicates("mmsi").set_index("mmsi").area_30)
    return vessel,origin,habitual


def shock_features(d,kind):
    if kind=="normal":
        d["shock_proximity"]=0.; d["inside_region"]=0.; d["days_x_proximity"]=0.; d["exposed_x_proximity"]=0.
        return d
    lat=d.alt.map(AREA_ATTR.lat); lon=d.alt.map(AREA_ATTR.lon)
    if kind=="redsea":
        dist=np.minimum(gc_nm(lat,lon,30.5,32.3),gc_nm(lat,lon,12.6,43.3))*1.852
        inside=lat.between(12,31)&lon.between(32,44)
    else:
        dist=gc_nm(lat,lon,26.5,56.3)*1.852
        inside=lat.between(23,31)&lon.between(47,56.8)
    d["shock_proximity"]=np.exp(-dist/1500); d["inside_region"]=inside.astype(int)
    d["days_x_proximity"]=d.days_since_shock.clip(0,250)/100*d.shock_proximity
    d["exposed_x_proximity"]=d.exposed*d.shock_proximity
    return d


def historical_long(panel,vessels,exposure):
    long=pd.read_parquet(CORE/"choice_long_v2.parquet")
    meta=panel[["event_id","call_time","legStartPortCode","mmsi"]].merge(vessels,on="mmsi",how="left")
    meta=meta.merge(exposure,on="mmsi",how="left"); meta["exposed"]=meta.historically_exposed.fillna(0).astype(int)
    long=long.drop(columns=["operator"],errors="ignore").merge(meta,on=["event_id","mmsi"],how="left")
    long["operator"]=long.operator.fillna("Unknown"); long["days_since_shock"]=0.
    long["vessel_pref_count"]=[np.log1p(VESSEL_PREF.get((m,a),0)) for m,a in long[["mmsi","alt"]].itertuples(index=False)]
    long["origin_pref_count"]=[np.log1p(ORIGIN_PREF.get((o,a),0)) for o,a in long[["legStartPortCode","alt"]].itertuples(index=False)]
    normal=long[long.year.isin([2021,2022])].copy(); normal=shock_features(normal,"normal")
    red=long[long.call_time.between("2023-12-15","2024-06-30 23:59:59")].copy()
    red["days_since_shock"]=(red.call_time-pd.Timestamp("2023-12-15")).dt.total_seconds()/86400
    red=shock_features(red,"redsea")
    return normal,red


def hormuz_long(code_area,geo,cent,vessels,pref):
    raw=pd.read_parquet(OUT/"hormuz_container_legs_2026.parquet",
        columns=["mmsi","legStartTime","legEndTime","legStartPortCode","legEndPortCode","isRefueled","refuelPortCodes","legCrossNodeList"])
    raw.mmsi=pd.to_numeric(raw.mmsi,errors="coerce").astype("Int64"); raw.legStartTime=pd.to_datetime(raw.legStartTime)
    raw=raw.merge(vessels,on="mmsi",how="left"); raw.operator=raw.operator.fillna("Unknown")
    raw=raw.sort_values(["mmsi","legStartTime","legEndTime"]).drop_duplicates(["mmsi","legStartTime","legEndTime"],keep="last")
    pre=raw.legStartTime.lt("2026-02-28")&raw.legCrossNodeList.astype(str).str.contains("HORMUZ",case=False,na=False)
    exp=set(raw.loc[pre,"mmsi"].dropna()); ev=raw[raw.isRefueled.eq(1)&raw.legStartTime.ge("2026-02-28")].copy()
    ev["actual_code"]=ev.refuelPortCodes.astype(str).str.split(",").str[0]; ev["actual"]=ev.actual_code.map(code_area)
    ev=ev[ev.actual.isin(AREA_ATTR.index)&ev.legStartPortCode.isin(geo.index)&ev.legEndPortCode.isin(geo.index)]
    ev["exposed"]=ev.mmsi.isin(exp).astype(int); ev["rand"]=np.random.default_rng(20260921).random(len(ev))
    ev=ev.sort_values("rand").groupby("exposed",group_keys=False).head(3500)
    rank=AREA_ATTR.events.rank(method="first",ascending=False).to_dict(); names=AREA_ATTR.index.to_numpy()
    rows=[]
    for i,r in enumerate(ev.itertuples()):
        o=geo.loc[str(r.legStartPortCode)]; n=geo.loc[str(r.legEndPortCode)]
        direct=gc_nm(o.lat,o.lon,n.lat,n.lon); route=gc_nm(o.lat,o.lon,AREA_ATTR.lat,AREA_ATTR.lon)+gc_nm(AREA_ATTR.lat,AREA_ATTR.lon,n.lat,n.lon)
        detour=np.asarray(route-direct); feasible=np.where((detour>=-50)&(detour<=2500))[0]
        ai=int(np.where(names==r.actual)[0][0])
        if ai not in feasible: feasible=np.append(feasible,ai)
        if len(feasible)>40: feasible=np.unique(np.array(sorted(feasible,key=lambda j:rank[names[j]])[:39]+[ai]))
        eid=f"H{i}"
        for j in feasible:
            a=names[j]
            rows.append({"event_id":eid,"mmsi":r.mmsi,"call_time":r.legStartTime,"operator":getattr(r,"operator","Unknown"),
                         "legStartPortCode":r.legStartPortCode,"actual":r.actual,"alt":a,"y":int(a==r.actual),
                         "detour_knm":float(max(detour[j],0)/1000),"dwell_10h":AREA_ATTR.loc[a,"dwell_10h"],
                         "log_calls":AREA_ATTR.loc[a,"log_calls"],"propensity":AREA_ATTR.loc[a,"propensity"],
                         "pref":pref.get((getattr(r,"operator","Unknown"),a),0.),"exposed":r.exposed,
                         "days_since_shock":(r.legStartTime-pd.Timestamp("2026-02-28")).total_seconds()/86400,
                         "vessel_pref_count":np.log1p(VESSEL_PREF.get((r.mmsi,a),0)),
                         "origin_pref_count":np.log1p(ORIGIN_PREF.get((r.legStartPortCode,a),0))})
    out=pd.DataFrame(rows); return shock_features(out,"hormuz")


def rank_metrics(name,model,features,d,habitual):
    z=d.copy(); z["score"]=model.predict_proba(z[features])[:,1]
    z["rank"]=z.groupby("event_id").score.rank(method="first",ascending=False)
    chosen=z[z.y.eq(1)].copy(); chosen_rank=chosen["rank"]
    pred=z.loc[z.groupby("event_id").score.idxmax(),["event_id","alt"]].rename(columns={"alt":"pred"})
    event=chosen[["event_id","mmsi","actual","exposed"]].merge(pred,on="event_id")
    event["habitual"]=event.mmsi.map(habitual); valid=event.habitual.notna()
    actual_change=event.loc[valid,"actual"].ne(event.loc[valid,"habitual"])
    pred_change=event.loc[valid,"pred"].ne(event.loc[valid,"habitual"])
    pr,rc,f1,_=precision_recall_fscore_support(actual_change,pred_change,average="binary",zero_division=0)
    return {"model":name,"events":len(chosen),"top1":float((chosen_rank<=1).mean()),"top3":float((chosen_rank<=3).mean()),
            "top5":float((chosen_rank<=5).mean()),"mrr":float((1/chosen_rank).mean()),
            "habit_change_events":int(valid.sum()),"change_precision":pr,"change_recall":rc,"change_f1":f1}


def main():
    global AREA_ATTR,VESSEL_PREF,ORIGIN_PREF
    OUT.mkdir(parents=True,exist_ok=True); code_area,geo,cent,vessels=resources()
    panel=pd.read_parquet(CORE/"legs_panel_v2_areas.parquet")
    panel=panel.merge(vessels,on="mmsi",how="left"); panel.operator=panel.operator.fillna("Unknown")
    panel=panel.sort_values(["mmsi","legStartTime","legEndTime"]).reset_index(drop=True); panel["event_id"]=panel.index.astype("int64")
    VESSEL_PREF,ORIGIN_PREF,habitual=history_maps(panel)
    choice=pd.read_parquet(CORE/"choice_long_v2.parquet")
    AREA_ATTR=(choice.groupby("alt").agg(dwell_10h=("dwell_10h","first"),log_calls=("log_calls","first"),
        propensity=("propensity","first"),events=("y","sum")).join(cent,how="inner").dropna())
    pref=choice.groupby(["operator","alt"]).pref.first().to_dict()
    exposure=pd.read_parquet(CORE/"red_sea_historical_exposure_v2.parquet",columns=["mmsi","historically_exposed","error"])
    exposure=exposure[exposure.error.eq("")]; exposure.mmsi=pd.to_numeric(exposure.mmsi,errors="coerce").astype("Int64")
    normal,red=historical_long(panel,vessels,exposure[["mmsi","historically_exposed"]])
    red_train=red[red.call_time.lt("2024-04-01")]; red_test=red[red.call_time.ge("2024-04-01")]
    hraw=hormuz_long(code_area,geo,cent,vessels,pref)
    base=["detour_knm","dwell_10h","log_calls","propensity","pref","vessel_pref_count","origin_pref_count"]
    shock=base+["shock_proximity","inside_region","days_x_proximity","exposed_x_proximity"]
    training=pd.concat([normal,red_train],ignore_index=True)
    models={
        "normal_frozen":(normal,base),
        "redsea_adapted_no_shock":(training,base),
        "shock_aware":(training,shock),
    }
    rows=[]
    for name,(tr,features) in models.items():
        fit=HistGradientBoostingClassifier(max_iter=180,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=80,
            l2_regularization=1.5,random_state=20260921).fit(tr[features],tr.y)
        for test_name,test in [("redsea_late_holdout",red_test),("hormuz_external",hraw)]:
            rows.append({"test":test_name,**rank_metrics(name,fit,features,test,habitual)})
    result=pd.DataFrame(rows); result.to_csv(OUT/"shock_port_ranker_metrics_v1.csv",index=False)
    report={"target":"actual bunkering port among feasible candidates","normal_train_events":normal.event_id.nunique(),
            "redsea_adaptation_events":red_train.event_id.nunique(),"redsea_holdout_events":red_test.event_id.nunique(),
            "hormuz_test_events":hraw.event_id.nunique(),"metrics":result.to_dict(orient="records"),
            "warning":"Port relocation is predictive, not a causal counterfactual."}
    (OUT/"shock_port_ranker_report_v1.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    hraw.to_parquet(OUT/"hormuz_port_choice_long_v1.parquet",index=False)
    print(result.to_string(index=False)); print(json.dumps({k:v for k,v in report.items() if k!="metrics"},indent=2))


if __name__=="__main__": main()
