#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import encoded_frames,load_task,runtime_workdir,write_submission,rank_unit
SEED=20260721

def emit(x):print('SMALL_MANIFEST='+json.dumps(x,sort_keys=True,separators=(',',':')))
def auc(y,p):return float(roc_auc_score(y,p))
def splitter(y):
 m=int(np.bincount(np.asarray(y,int)).min());return StratifiedKFold(n_splits=max(2,min(5,m)),shuffle=True,random_state=SEED)
def cv_model(X,y,XT,sp,model,decision=False):
 o=np.zeros(len(X));p=np.zeros(len(XT))
 for f,(tr,va) in enumerate(sp.split(X,y)):
  m=clone(model);m.fit(X.iloc[tr],y[tr])
  if decision:
   o[va]=expit(np.clip(m.decision_function(X.iloc[va]),-15,15));p+=expit(np.clip(m.decision_function(XT),-15,15))/sp.n_splits
  else:
   o[va]=m.predict_proba(X.iloc[va])[:,1];p+=m.predict_proba(XT)[:,1]/sp.n_splits
 return o,p
def main():
 st=time.time();w=runtime_workdir();tr,te,sample,idc,target,features,y=load_task(w)
 if not (300<=len(tr)<=3500):emit({'candidates':[],'reason':'row_gate','rows':len(tr)});return
 X,XT,cats=encoded_frames(tr,te,features);sp=splitter(y);models={};errors={}
 d=len(features);n=len(tr)
 specs={
  'rbf_c1':(Pipeline([('scale',StandardScaler()),('svc',SVC(C=1.0,gamma='scale',kernel='rbf',probability=False,class_weight='balanced',cache_size=700))]),True),
  'rbf_c4':(Pipeline([('scale',StandardScaler()),('svc',SVC(C=4.0,gamma='scale',kernel='rbf',probability=False,class_weight='balanced',cache_size=700))]),True),
  'hgb_small':(HistGradientBoostingClassifier(learning_rate=.035,max_iter=700,max_leaf_nodes=15,min_samples_leaf=max(8,min(30,n//40)),l2_regularization=5.0,early_stopping=True,validation_fraction=.15,random_state=SEED),False),
  'knn':(Pipeline([('scale',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=max(11,min(45,int(np.sqrt(n)))),weights='distance',p=2,n_jobs=3))]),False),
 }
 if not cats and d<=40:
  specs['qda']=(Pipeline([('scale',StandardScaler()),('qda',QuadraticDiscriminantAnalysis(reg_param=.25))]),False)
 for name,(m,decision) in specs.items():
  t=time.time()
  try:
   o,p=cv_model(X,y,XT,sp,m,decision)
   if not np.isfinite(o).all() or np.std(o)<1e-9:raise ValueError('invalid predictions')
   models[name]={'oof':o,'test':p,'oof_auc':auc(y,o),'seconds':round(time.time()-t,3)}
  except Exception as e:errors[name]=f'{type(e).__name__}:{e}'
 if not models:emit({'candidates':[],'errors':errors});return
 ranked=sorted(models,key=lambda n:models[n]['oof_auc'],reverse=True);spec=[]
 for n in ranked:spec.append((n,models[n]['oof'],models[n]['test']))
 if len(ranked)>1:
  a,b=ranked[:2];spec.append(('small_rank_top2',.5*rank_unit(models[a]['oof'])+.5*rank_unit(models[b]['oof']),.5*rank_unit(models[a]['test'])+.5*rank_unit(models[b]['test'])))
 cand=[];state={}
 for model_name,o,pred in spec:
  path=write_submission(w/f'{model_name}.csv',pred,te,sample,idc,target);cand.append({'id':model_name,'path':path,'oof_auc':auc(y,o)});state[model_name+'_oof']=np.asarray(o,np.float32);state[model_name+'_test']=np.asarray(pred,np.float32)
 cand.sort(key=lambda x:x['oof_auc'],reverse=True);np.savez_compressed(w/'small_expert_state.npz',**state)
 emit({'candidates':cand[:2],'all_metrics':[{k:v for k,v in x.items() if k!='path'} for x in cand],'errors':errors,'seconds':round(time.time()-st,3),'rows':len(tr)})
if __name__=='__main__':main()
