#!/usr/bin/env python3
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import encoded_frames, load_task, record_candidates, runtime_workdir, write_submission, rank_unit
SEED=20260710

def emit(x): print('ZEROSHOT_MANIFEST='+json.dumps(x,sort_keys=True,separators=(',',':')))
def auc(y,p): return float(roc_auc_score(y,p))
def folds(y):
    minority=int(np.bincount(np.asarray(y,int)).min())
    return StratifiedKFold(n_splits=max(2,min(5,minority)),shuffle=True,random_state=SEED)

CONFIGS={
 'ag_r177':dict(extra_trees=True,subsample=.8769107816033,colsample_bytree=.4622189821941,reg_alpha=.2375070586896,reg_lambda=.3551561351804,learning_rate=.0178593900218,num_leaves=39,min_child_samples=3),
 'ag_r163':dict(extra_trees=False,subsample=.9783898288461,colsample_bytree=.5279825611461,reg_alpha=.0269274915833,reg_lambda=.8375250972309,learning_rate=.0113913650333,num_leaves=84,min_child_samples=3),
 'ag_r120':dict(extra_trees=True,subsample=.8541333332514,colsample_bytree=.7334718346252,reg_alpha=.241338427726,reg_lambda=.298107723769,learning_rate=.0126654490778,num_leaves=5,min_child_samples=12),
}

def fit_config(xtr,y,xte,cats,sp,cfg):
 import lightgbm as lgb
 xtr=xtr.copy();xte=xte.copy()
 for c in cats:
  xtr[c]=xtr[c].round().astype('int64').astype('category');xte[c]=xte[c].round().astype('int64').astype('category')
 oof=np.zeros(len(xtr));pred=np.zeros(len(xte))
 for fold,(fi,vi) in enumerate(sp.split(xtr,y)):
  m=lgb.LGBMClassifier(objective='binary',n_estimators=1800,bagging_freq=1,random_state=SEED+fold,n_jobs=3,verbosity=-1,**cfg)
  m.fit(xtr.iloc[fi],y[fi],eval_set=[(xtr.iloc[vi],y[vi])],eval_metric='auc',categorical_feature=cats,callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
  oof[vi]=m.predict_proba(xtr.iloc[vi])[:,1];pred+=m.predict_proba(xte)[:,1]/sp.n_splits
 return oof,pred

def main():
 started=time.time();work=runtime_workdir();train,test,sample,idc,target,features,y=load_task(work)
 if len(train)>20000:
  emit({'candidates':[],'reason':'row_gate','rows':len(train),'seconds':round(time.time()-started,3)});return
 xtr,xte,cats=encoded_frames(train,test,features);sp=folds(y);models={};errors={}
 for name,cfg in CONFIGS.items():
  t=time.time()
  try:
   o,p=fit_config(xtr,y,xte,cats,sp,cfg)
   if not np.isfinite(o).all() or np.std(o)<1e-9: raise ValueError('invalid predictions')
   models[name]={'oof':o,'test':p,'oof_auc':auc(y,o),'seconds':round(time.time()-t,3)}
  except Exception as e: errors[name]=f'{type(e).__name__}:{e}'
 if not models: emit({'candidates':[],'errors':errors,'seconds':round(time.time()-started,3)});return
 ranked=sorted(models,key=lambda n:models[n]['oof_auc'],reverse=True)
 specs=[]
 for n in ranked: specs.append((n,models[n]['oof'],models[n]['test']))
 if len(ranked)>=2:
  a,b=ranked[:2]; specs.append(('ag_rank_top2',.5*rank_unit(models[a]['oof'])+.5*rank_unit(models[b]['oof']),.5*rank_unit(models[a]['test'])+.5*rank_unit(models[b]['test'])))
 cand=[];state={}
 for name,o,p in specs:
  s=auc(y,o);path=write_submission(work/f'zeroshot_{name}.csv',p,test,sample,idc,target)
  cand.append({'id':name,'path':path,'oof_auc':s});state[name+'_oof']=np.asarray(o,np.float32);state[name+'_test']=np.asarray(p,np.float32)
 cand.sort(key=lambda c:c['oof_auc'],reverse=True)
 np.savez_compressed(work/'zeroshot_lgb_state.npz',**state)
 record_candidates(work,'zeroshot',cand[:2])
 emit({'candidates':cand[:2],'all_metrics':[{k:v for k,v in c.items() if k!='path'} for c in cand],'errors':errors,'seconds':round(time.time()-started,3),'rows':len(train)})
if __name__=='__main__':main()
