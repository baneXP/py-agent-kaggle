#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures,SplineTransformer,StandardScaler
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import categorical_columns,load_task,runtime_workdir,write_submission,rank_unit
SEED=20260721

def emit(x):print('GAM_MANIFEST='+json.dumps(x,sort_keys=True,separators=(',',':')))
def auc(y,p):return float(roc_auc_score(y,p))
def split(y):
 m=int(np.bincount(np.asarray(y,int)).min());return StratifiedKFold(n_splits=max(2,min(5,m)),shuffle=True,random_state=SEED)
def cv_model(train,y,test,features,sp,model):
 o=np.zeros(len(train));p=np.zeros(len(test))
 for tr,va in sp.split(train,y):
  m=clone(model);m.fit(train.iloc[tr][features],y[tr]);o[va]=m.predict_proba(train.iloc[va][features])[:,1];p+=m.predict_proba(test[features])[:,1]/sp.n_splits
 return o,p
def main():
 st=time.time();w=runtime_workdir();train,test,sample,idc,target,features,y=load_task(w)
 if len(train)>20000:emit({'candidates':[],'reason':'row_gate','rows':len(train)});return
 cats=categorical_columns(train,features);nums=[c for c in features if c not in cats]
 if not nums:emit({'candidates':[],'reason':'no_numeric','rows':len(train)});return
 cat_pipe=Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=2))])
 spline_num=Pipeline([('imp',SimpleImputer(strategy='median')),('spl',SplineTransformer(n_knots=6,degree=3,include_bias=False)),('sc',StandardScaler(with_mean=False))])
 spline_pre=ColumnTransformer([('num',spline_num,nums),('cat',cat_pipe,cats)],sparse_threshold=.3)
 spline=Pipeline([('pre',spline_pre),('lr',LogisticRegression(C=.25,max_iter=2500,solver='liblinear',class_weight='balanced'))])
 specs={'gam_spline':spline}
 if len(nums)<=30:
  quad_num=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('poly',PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)),('sc2',StandardScaler(with_mean=False))])
  quad_pre=ColumnTransformer([('num',quad_num,nums),('cat',cat_pipe,cats)],sparse_threshold=.3)
  specs['gam_quadratic']=Pipeline([('pre',quad_pre),('lr',LogisticRegression(C=.08,max_iter=3000,solver='liblinear',class_weight='balanced'))])
 sp=split(y);models={};errors={}
 for n,m in specs.items():
  t=time.time()
  try:
   o,p=cv_model(train,y,test,features,sp,m)
   if not np.isfinite(o).all() or np.std(o)<1e-9:raise ValueError('invalid')
   models[n]={'oof':o,'test':p,'oof_auc':auc(y,o),'seconds':round(time.time()-t,3)}
  except Exception as e:errors[n]=f'{type(e).__name__}:{e}'
 if not models:emit({'candidates':[],'errors':errors});return
 ranked=sorted(models,key=lambda n:models[n]['oof_auc'],reverse=True);spec=[]
 for n in ranked:spec.append((n,models[n]['oof'],models[n]['test']))
 if len(ranked)>1:
  a,b=ranked[:2];spec.append(('gam_rank_top2',.5*rank_unit(models[a]['oof'])+.5*rank_unit(models[b]['oof']),.5*rank_unit(models[a]['test'])+.5*rank_unit(models[b]['test'])))
 cand=[];state={}
 for n,o,p in spec:
  path=write_submission(w/f'{n}.csv',p,test,sample,idc,target);cand.append({'id':n,'path':path,'oof_auc':auc(y,o)});state[n+'_oof']=np.asarray(o,np.float32);state[n+'_test']=np.asarray(p,np.float32)
 cand.sort(key=lambda x:x['oof_auc'],reverse=True);np.savez_compressed(w/'gam_expert_state.npz',**state)
 emit({'candidates':cand[:2],'all_metrics':[{k:v for k,v in x.items() if k!='path'} for x in cand],'errors':errors,'seconds':round(time.time()-st,3),'rows':len(train)})
if __name__=='__main__':main()
