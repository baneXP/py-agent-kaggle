"""Cross-validate a compact diverse portfolio and write ranked candidate CSVs."""
from __future__ import annotations
import sys,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import categorical_columns,emit_manifest,encoded_frames,load_task,native_frames,rank_unit,record_candidates,runtime_workdir,write_submission
SEED=20260710

def folds_for(y,n_rows):
    minority=int(np.bincount(y).min());return StratifiedKFold(n_splits=max(2,min(5,minority)),shuffle=True,random_state=SEED)
def score(y,p):return float(roc_auc_score(y,p))
def cv_catboost(xtr,y,xte,cats,splitter):
    from catboost import CatBoostClassifier
    oof=np.zeros(len(xtr));pred=np.zeros(len(xte))
    for fold,(fi,vi) in enumerate(splitter.split(xtr,y)):
        m=CatBoostClassifier(iterations=650 if len(xtr)>=2000 else 420,depth=6 if len(xtr)>=2000 else 5,learning_rate=.04,loss_function="Logloss",eval_metric="AUC",l2_leaf_reg=7.,random_strength=.35,random_seed=SEED+fold,verbose=False,allow_writing_files=False,thread_count=3)
        m.fit(xtr.iloc[fi],y[fi],cat_features=cats,eval_set=(xtr.iloc[vi],y[vi]),early_stopping_rounds=70,use_best_model=True,verbose=False)
        oof[vi]=m.predict_proba(xtr.iloc[vi])[:,1];pred+=m.predict_proba(xte)[:,1]/splitter.n_splits
    return oof,pred
def cv_lightgbm(xtr,y,xte,cats,splitter):
    import lightgbm as lgb
    xtr=xtr.copy();xte=xte.copy()
    for c in cats:xtr[c]=xtr[c].round().astype("int64").astype("category");xte[c]=xte[c].round().astype("int64").astype("category")
    oof=np.zeros(len(xtr));pred=np.zeros(len(xte));small=len(xtr)<2500
    for fold,(fi,vi) in enumerate(splitter.split(xtr,y)):
        m=lgb.LGBMClassifier(objective="binary",n_estimators=900 if not small else 500,learning_rate=.025 if not small else .04,num_leaves=15 if small else 31,max_depth=-1,min_child_samples=20 if small else 35,subsample=.85,colsample_bytree=.85,reg_alpha=.15,reg_lambda=2.5,random_state=SEED+fold,n_jobs=3,verbosity=-1)
        m.fit(xtr.iloc[fi],y[fi],eval_set=[(xtr.iloc[vi],y[vi])],eval_metric="auc",categorical_feature=cats,callbacks=[lgb.early_stopping(70,verbose=False),lgb.log_evaluation(0)])
        oof[vi]=m.predict_proba(xtr.iloc[vi])[:,1];pred+=m.predict_proba(xte)[:,1]/splitter.n_splits
    return oof,pred
def cv_extra_trees(xtr,y,xte,splitter):
    oof=np.zeros(len(xtr));pred=np.zeros(len(xte));leaf=2 if len(xtr)<3000 else 3
    for fold,(fi,vi) in enumerate(splitter.split(xtr,y)):
        m=ExtraTreesClassifier(n_estimators=500,max_features=.8,min_samples_leaf=leaf,class_weight="balanced",random_state=SEED+fold,n_jobs=3)
        m.fit(xtr.iloc[fi],y[fi]);oof[vi]=m.predict_proba(xtr.iloc[vi])[:,1];pred+=m.predict_proba(xte)[:,1]/splitter.n_splits
    return oof,pred
def cv_logistic(train,y,test,features,splitter):
    cats=categorical_columns(train,features);nums=[c for c in features if c not in cats]
    oof=np.zeros(len(train));pred=np.zeros(len(test))
    for fold,(fi,vi) in enumerate(splitter.split(train,y)):
        numeric=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]);categorical=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=2))])
        transformer=ColumnTransformer([("numeric",numeric,nums),("categorical",categorical,cats)])
        m=Pipeline([("features",transformer),("model",LogisticRegression(C=.25 if len(train)<2000 else .7,max_iter=1500,solver="liblinear",random_state=SEED+fold))])
        m.fit(train.iloc[fi][features],y[fi]);oof[vi]=m.predict_proba(train.iloc[vi][features])[:,1];pred+=m.predict_proba(test[features])[:,1]/splitter.n_splits
    return oof,pred

def main():
    started=time.time();workdir=runtime_workdir();train,test,sample,id_col,target_col,features,y=load_task(workdir)
    native_train,native_test,cats=native_frames(train,test,features);encoded_train,encoded_test,encoded_cats=encoded_frames(train,test,features);splitter=folds_for(y,len(train));models={};errors={}
    runners=[("catboost",lambda:cv_catboost(native_train,y,native_test,cats,splitter)),("lightgbm",lambda:cv_lightgbm(encoded_train,y,encoded_test,encoded_cats,splitter)),("extra_trees",lambda:cv_extra_trees(encoded_train,y,encoded_test,splitter)),("logistic",lambda:cv_logistic(train,y,test,features,splitter))]
    for name,runner in runners:
        t=time.time()
        try:
            oof,pred=runner()
            if not np.isfinite(oof).all() or np.std(oof)<=1e-9:raise ValueError("invalid or constant OOF predictions")
            models[name]={"oof":oof,"test":pred,"cv_auc":score(y,oof),"seconds":round(time.time()-t,3)};print(f"MODEL name={name} cv_auc={models[name]['cv_auc']:.6f} seconds={models[name]['seconds']}")
        except Exception as exc:errors[name]=f"{type(exc).__name__}:{exc}";print(f"MODEL_ERROR name={name} error={errors[name]}")
    if not models:
        prior=float(np.mean(y));path=write_submission(workdir/"portfolio_prior.csv",np.full(len(test),prior),test,sample,id_col,target_col);record_candidates(workdir,"portfolio",[{"path":path,"kind":"prior"}]);emit_manifest({"candidates":[path],"errors":errors,"models":{},"robust_choice":path,"total_seconds":round(time.time()-started,3)});return
    try:
        state = {}
        for core_name, core_data in models.items():
            state[f"{core_name}_oof"] = np.asarray(core_data["oof"], dtype=np.float32)
            state[f"{core_name}_test"] = np.asarray(core_data["test"], dtype=np.float32)
        np.savez_compressed(workdir / "portfolio_core_state.npz", **state)
    except Exception as exc:
        print(f"CORE_STATE_ERROR error={type(exc).__name__}:{exc}")

    ranked=sorted(models,key=lambda n:models[n]["cv_auc"],reverse=True);candidates=[]
    for name in ranked:
        path=write_submission(workdir/f"portfolio_{name}.csv",models[name]["test"],test,sample,id_col,target_col);candidates.append({"path":path,"cv_auc":models[name]["cv_auc"],"kind":name})
    top2=ranked[:2]
    if len(top2)==2:
        a,b=top2;ao,bo=rank_unit(models[a]["oof"]),rank_unit(models[b]["oof"]);at,bt=rank_unit(models[a]["test"]),rank_unit(models[b]["test"])
        oof=.5*ao+.5*bo;testp=.5*at+.5*bt;path=write_submission(workdir/"portfolio_rank_top2.csv",testp,test,sample,id_col,target_col);equal=score(y,oof);candidates.append({"path":path,"cv_auc":equal,"kind":"rank_top2"})
        opts=[]
        for w in (.25,.5,.75):opts.append((score(y,w*ao+(1-w)*bo),w))
        ws,w=max(opts)
        floor=max(models[n]["cv_auc"] for n in top2)
        if w!=.5 and ws>=max(equal,floor)+.0005:
            path=write_submission(workdir/"portfolio_rank_top2_weighted.csv",w*at+(1-w)*bt,test,sample,id_col,target_col);candidates.append({"path":path,"cv_auc":ws,"kind":"rank_top2_weighted","first_model":a,"first_weight":w,"second_model":b,"second_weight":1-w})
    top3=ranked[:3]
    if len(top3)==3:
        oof=np.mean([rank_unit(models[n]["oof"]) for n in top3],axis=0);testp=np.mean([rank_unit(models[n]["test"]) for n in top3],axis=0);path=write_submission(workdir/"portfolio_rank_top3.csv",testp,test,sample,id_col,target_col);candidates.append({"path":path,"cv_auc":score(y,oof),"kind":"rank_top3"})
    weights=np.array([max(models[n]["cv_auc"]-.5,.005)**2 for n in ranked]);weights/=weights.sum();all_oof=np.average(np.stack([rank_unit(models[n]["oof"]) for n in ranked]),axis=0,weights=weights);all_test=np.average(np.stack([rank_unit(models[n]["test"]) for n in ranked]),axis=0,weights=weights);robust=write_submission(workdir/"portfolio_rank_all.csv",all_test,test,sample,id_col,target_col);candidates.append({"path":robust,"cv_auc":score(y,all_oof),"kind":"rank_all"})

    specialist=None;specialist_info={"eligible":False,"accepted":False}
    all_categorical=len(cats)==len(features)
    if all_categorical and 3000<=len(train)<=50000:
        specialist_info["eligible"]=True
        try:
            from purecat_tabprep import cv_purecat_tabprep
            soof,stest,details=cv_purecat_tabprep(train,y,test,features,splitter)
            sauc=score(y,soof);ref_name=ranked[0];ref=models[ref_name]
            corr=float(pd.Series(ref["oof"]).corr(pd.Series(soof),method="spearman"))
            accepted=(np.isfinite(soof).all() and np.isfinite(stest).all() and np.std(soof)>1e-9 and sauc>=max(.55,ref["cv_auc"]-.005) and corr<.997)
            specialist_info.update({"accepted":accepted,"cv_auc":sauc,"reference_kind":ref_name,"reference_auc":ref["cv_auc"],"spearman":corr,"details":details})
            if accepted:
                specialist=write_submission(workdir/"portfolio_purecat_specialist.csv",stest,test,sample,id_col,target_col)
                candidates.append({"path":specialist,"cv_auc":sauc,"kind":"purecat_specialist","reference_kind":ref_name,"spearman":corr})
                print(f"PURECAT_ACCEPTED cv_auc={sauc:.6f} reference={ref_name} reference_auc={ref['cv_auc']:.6f} spearman={corr:.6f}")
            else:print(f"PURECAT_REJECTED cv_auc={sauc:.6f} reference_auc={ref['cv_auc']:.6f} spearman={corr:.6f}")
        except Exception as exc:
            specialist_info["error"]=f"{type(exc).__name__}:{exc}";print(f"PURECAT_ERROR error={specialist_info['error']}")

    candidates=sorted(candidates,key=lambda x:x["cv_auc"],reverse=True)
    if specialist is not None:
        special=next(x for x in candidates if x["kind"]=="purecat_specialist")
        core=[x for x in candidates if x["kind"]!="purecat_specialist"]
        chosen=core[:7]+[special]
    else:chosen=candidates[:8]
    paths=[x["path"] for x in chosen]
    record_candidates(workdir,"portfolio",[{"path":x["path"],"kind":x["kind"],"oof_auc":x["cv_auc"]} for x in chosen])
    emit_manifest({"candidates":paths,"candidate_metrics":candidates,"errors":errors,"models":{n:{"cv_auc":models[n]["cv_auc"],"seconds":models[n]["seconds"]} for n in ranked},"robust_choice":robust,"specialist_kind":"purecat_specialist" if specialist is not None else None,"specialist":specialist_info,"rows":len(train),"features":len(features),"folds":splitter.n_splits,"total_seconds":round(time.time()-started,3)})
if __name__=="__main__":main()
