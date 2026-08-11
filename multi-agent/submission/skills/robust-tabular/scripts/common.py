"""Schema discovery and submission helpers for the robust tabular skill."""
from __future__ import annotations
import glob, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
BAD_TOKENS = ("solution", "answer", "truth", "ground", "labelled_test")

def runtime_workdir() -> Path:
    override = os.environ.get("ROBUST_TABULAR_WORKDIR")
    if override:
        candidate = Path(override).expanduser().resolve()
        if candidate.is_dir(): return candidate
    harness = Path("/work")
    if harness.is_dir():
        names = {path.name.lower() for path in harness.glob("*.csv")}
        if any("train" in n for n in names) and any("test" in n for n in names): return harness
    return Path.cwd()

def _safe_csvs(task_dir):
    paths = glob.glob(os.path.join(str(task_dir), "**", "*.csv"), recursive=True)
    return [p for p in paths if not any(t in os.path.basename(p).lower() for t in BAD_TOKENS)]

def discover_files(task_dir="."):
    csvs = _safe_csvs(task_dir)
    def pick(required, forbidden=()):
        for p in sorted(csvs, key=lambda v:(len(Path(v).parts),len(v))):
            n=os.path.basename(p).lower()
            if any(t in n for t in required) and not any(t in n for t in forbidden): return p
        return None
    train=pick(("train",),("sample","submission")); test=pick(("test",),("sample","submission")); sample=pick(("sample_submission","samplesubmission","sample-submission"))
    if train is None or test is None: raise FileNotFoundError(f"train/test CSV not found; visible CSV count={len(csvs)}")
    return train,test,sample

def infer_id(train,test,sample):
    if sample is not None and len(sample.columns) and sample.columns[0] in test.columns: return str(sample.columns[0])
    preferred=("row_id","id","sample_id","record_id","uid","index")
    for c in test.columns:
        low=str(c).lower()
        if low in preferred or low.endswith("_id"): return str(c)
    for c in test.columns:
        if c in train.columns and train[c].is_unique and test[c].is_unique: return str(c)
    return None

def infer_target(train,test,sample,id_col):
    train_only=[c for c in train.columns if c not in test.columns and c != id_col]
    preferred=("target","label","class","outcome","response","y")
    for c in train_only:
        if str(c).lower() in preferred:return str(c)
    if len(train_only)==1:return str(train_only[0])
    for c in train_only:
        if train[c].nunique(dropna=True)==2:return str(c)
    if sample is not None:
        for c in sample.columns[1:]:
            if c in train.columns:return str(c)
    raise ValueError(f"binary target could not be inferred from train-only columns={train_only}")

def normalize_binary(y):
    vals=list(pd.unique(y.dropna()))
    if len(vals)!=2: raise ValueError(f"expected binary target, found {len(vals)} values")
    if pd.api.types.is_numeric_dtype(y) and set(vals).issubset({0,1,0.0,1.0}): return y.astype(int).to_numpy()
    positive=sorted(vals,key=lambda v:str(v))[-1]
    return (y==positive).astype(int).to_numpy()

def load_task(task_dir="."):
    train_path,test_path,sample_path=discover_files(task_dir)
    train=pd.read_csv(train_path); test=pd.read_csv(test_path); sample=pd.read_csv(sample_path) if sample_path else None
    id_col=infer_id(train,test,sample); target_col=infer_target(train,test,sample,id_col)
    features=[c for c in test.columns if c in train.columns and c != id_col]
    if not features:raise ValueError("no shared modeling features")
    y=normalize_binary(train[target_col])
    return train,test,sample,id_col,target_col,features,y

def categorical_columns(frame,features):
    return [c for c in features if _string_like(frame[c])]

def native_frames(train,test,features):
    xtr=train[features].copy(); xte=test[features].copy(); cats=categorical_columns(xtr,features)
    for c in cats:
        xtr[c]=xtr[c].fillna("__MISSING__").astype(str); xte[c]=xte[c].fillna("__MISSING__").astype(str)
    for c in (x for x in features if x not in cats):
        xtr[c]=pd.to_numeric(xtr[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
        xte[c]=pd.to_numeric(xte[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    return xtr,xte,cats

def encoded_frames(train,test,features):
    xtr=train[features].copy(); xte=test[features].copy(); cats=categorical_columns(xtr,features)
    for c in features:
        if c in cats:
            combined=pd.concat([xtr[c],xte[c]],ignore_index=True).fillna("__MISSING__").astype(str)
            codes,_=pd.factorize(combined,sort=True); xtr[c]=codes[:len(xtr)]; xte[c]=codes[len(xtr):]
        else:
            tr=pd.to_numeric(xtr[c],errors="coerce").replace([np.inf,-np.inf],np.nan); te=pd.to_numeric(xte[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
            med=float(tr.median()) if tr.notna().any() else 0.0; xtr[c]=tr.fillna(med); xte[c]=te.fillna(med)
    return xtr.astype(float),xte.astype(float),cats

def write_submission(path,predictions,test,sample,id_col,target_col):
    p=np.asarray(predictions,dtype=float); p=np.nan_to_num(p,nan=.5,posinf=1.,neginf=0.); p=np.clip(p,1e-6,1-1e-6)
    if len(p)!=len(test):raise ValueError(f"prediction rows={len(p)} test rows={len(test)}")
    if sample is not None:
        out=sample.copy()
        if len(out)!=len(test):raise ValueError("sample-submission row count does not match test")
        pred_cols=[c for c in out.columns if c != id_col] or [out.columns[-1]]; out[pred_cols[-1]]=p
    else:
        ids=test[id_col] if id_col and id_col in test.columns else np.arange(len(test)); out=pd.DataFrame({id_col or "row_id":ids,target_col:p})
    if not np.isfinite(out.iloc[:,-1].to_numpy(dtype=float)).all():raise ValueError("non-finite predictions after sanitization")
    out.to_csv(path,index=False); return str(path)

def rank_unit(values): return pd.Series(np.asarray(values,dtype=float)).rank(method="average",pct=True).to_numpy()
def emit_manifest(payload): print("PORTFOLIO_MANIFEST="+json.dumps(payload,sort_keys=True,separators=(",",":")))

# --- V24 additions -----------------------------------------------------------
# A single append-only ledger lets the final selection controller reason about
# *which model* produced each public score. Without it, selection can only see
# an unlabelled vector of noisy leaderboard numbers.
LEDGER_NAME = "candidate_ledger.jsonl"

def rank_signature(values, digits=6):
    """Stable short fingerprint of a prediction vector's rank ordering."""
    r = np.round(rank_unit(values), digits)
    return hashlib.sha1(r.tobytes()).hexdigest()[:16]

def record_candidates(workdir, stage, entries):
    """Append the ordered list of paths this stage will ask the agent to submit.

    Ledger order mirrors manifest order, because a time-gated agent always
    submits a *prefix* of the manifest, never a permutation of it. Failures are
    logged and swallowed: the ledger helps the selection stage, it is never a
    dependency of the modelling stages.
    """
    try:
        path = Path(workdir) / LEDGER_NAME
        existing = 0
        if path.is_file():
            with open(path, "r", encoding="utf-8") as fh:
                existing = sum(1 for line in fh if line.strip())
        with open(path, "a", encoding="utf-8") as fh:
            for offset, item in enumerate(entries):
                row = {
                    "seq": existing + offset,
                    "stage": str(stage),
                    "path": str(item.get("path")),
                    "kind": str(item.get("kind", stage)),
                    "oof_auc": (float(item["oof_auc"]) if item.get("oof_auc") is not None else None),
                }
                fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        print(f"LEDGER stage={stage} appended={len(entries)} total={existing + len(entries)}")
    except Exception as exc:
        print(f"LEDGER_ERROR stage={stage} error={type(exc).__name__}:{exc}")

def _string_like(series):
    """Version-robust categorical detection.

    pandas >= 3 reads text columns as a dedicated string dtype rather than
    `object`, so an `is_object_dtype` test silently classifies every categorical
    column as numeric. Kaggle containers and local environments can disagree on
    the pandas major version, so the check must cover both.
    """
    dtype = series.dtype
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_bool_dtype(series):
        return True
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    string_dtype = getattr(pd, "StringDtype", None)
    if string_dtype is not None and isinstance(dtype, string_dtype):
        return True
    return bool(getattr(dtype, "kind", "") in ("O", "U", "S"))
