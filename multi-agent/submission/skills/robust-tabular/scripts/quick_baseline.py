"""Write a fast valid baseline before the more expensive portfolio stage."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load_task,native_frames,record_candidates,runtime_workdir,write_submission

def main():
    workdir=runtime_workdir(); output=workdir/"quick_baseline.csv"
    train,test,sample,id_col,target_col,features,y=load_task(workdir)
    prior=float(np.clip(np.mean(y),1e-6,1-1e-6)); write_submission(output,np.full(len(test),prior),test,sample,id_col,target_col)
    method="prior"
    try:
        from catboost import CatBoostClassifier
        xtr,xte,cats=native_frames(train,test,features)
        model=CatBoostClassifier(iterations=260 if len(train)<3000 else 360,depth=5 if len(train)<2000 else 6,learning_rate=.06,loss_function="Logloss",eval_metric="AUC",l2_leaf_reg=6.,random_strength=.5,random_seed=20260710,verbose=False,allow_writing_files=False,thread_count=4)
        model.fit(xtr,y,cat_features=cats); pred=model.predict_proba(xte)[:,1]
        if np.std(pred)>1e-8:write_submission(output,pred,test,sample,id_col,target_col);method="catboost"
    except Exception as exc:print(f"quick_model_error={type(exc).__name__}:{exc}")
    record_candidates(workdir,"quick",[{"path":str(output),"kind":method}])
    print(f"QUICK_BASELINE path={output} method={method} rows={len(test)} features={len(features)}")
if __name__=="__main__":main()
