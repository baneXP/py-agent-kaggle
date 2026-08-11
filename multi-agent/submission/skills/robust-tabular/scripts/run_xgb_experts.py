#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import encoded_frames,load_task,record_candidates,runtime_workdir,write_submission,rank_unit
SEED=20260721

def emit(x):print('XGB_MANIFEST='+json.dumps(x,sort_keys=True,separators=(',',':')))
def auc(y,p):return float(roc_auc_score(y,p))
def folds(y):
 m=int(np.bincount(np.asarray(y,int)).min());return StratifiedKFold(n_splits=max(2,min(5,m)),shuffle=True,random_state=SEED)
CONFIGS={
 'xgb_depth6':dict(max_depth=6,min_child_weight=5,learning_rate=.03,subsample=.85,colsample_bytree=.80,reg_alpha=.10,reg_lambda=2.0,gamma=0.0),
 'xgb_depth3':dict(max_depth=3,min_child_weight=2,learning_rate=.035,subsample=.90,colsample_bytree=.90,reg_alpha=.02,reg_lambda=1.3,gamma=.02),
}
def fit_one(X,y,XT,sp,cfg):
 from xgboost import XGBClassifier
 o=np.zeros(len(X));p=np.zeros(len(XT))
 for f,(tr,va) in enumerate(sp.split(X,y)):
  model=XGBClassifier(n_estimators=1600,objective='binary:logistic',eval_metric='auc',tree_method='hist',max_bin=256,n_jobs=3,random_state=SEED+f,early_stopping_rounds=80,**cfg)
  model.fit(X.iloc[tr],y[tr],eval_set=[(X.iloc[va],y[va])],verbose=False)
  o[va]=model.predict_proba(X.iloc[va])[:,1];p+=model.predict_proba(XT)[:,1]/sp.n_splits
 return o,p
def main():
 st=time.time();w=runtime_workdir();tr,te,sample,idc,target,features,y=load_task(w)
 if len(tr)>20000:emit({'candidates':[],'reason':'row_gate','rows':len(tr)});return
 X,XT,_=encoded_frames(tr,te,features);sp=folds(y);models={};errors={}
 for n,cfg in CONFIGS.items():
  t=time.time()
  try:
   o,p=fit_one(X,y,XT,sp,cfg)
   if not np.isfinite(o).all() or np.std(o)<1e-9:raise ValueError('invalid predictions')
   models[n]={'oof':o,'test':p,'oof_auc':auc(y,o),'seconds':round(time.time()-t,3)}
  except Exception as e:errors[n]=f'{type(e).__name__}:{e}'
 if not models:emit({'candidates':[],'errors':errors,'seconds':round(time.time()-st,3)});return
 ranked=sorted(models,key=lambda n:models[n]['oof_auc'],reverse=True);spec=[]
 for n in ranked:spec.append((n,models[n]['oof'],models[n]['test']))
 if len(ranked)>1:
  a,b=ranked[:2];spec.append(('xgb_rank_top2',.5*rank_unit(models[a]['oof'])+.5*rank_unit(models[b]['oof']),.5*rank_unit(models[a]['test'])+.5*rank_unit(models[b]['test'])))
 cand=[];state={}
 for n,o,p in spec:
  path=write_submission(w/f'{n}.csv',p,te,sample,idc,target);cand.append({'id':n,'path':path,'oof_auc':auc(y,o)});state[n+'_oof']=np.asarray(o,np.float32);state[n+'_test']=np.asarray(p,np.float32)
 cand.sort(key=lambda x:x['oof_auc'],reverse=True);np.savez_compressed(w/'xgb_expert_state.npz',**state)
 record_candidates(w,'xgb',cand[:2])
 emit({'candidates':cand[:2],'all_metrics':[{k:v for k,v in x.items() if k!='path'} for x in cand],'errors':errors,'seconds':round(time.time()-st,3),'rows':len(tr)})
if __name__=='__main__':main()
