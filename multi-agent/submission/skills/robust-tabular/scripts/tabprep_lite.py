"""Fold-safe lightweight TabPrep-inspired specialist for binary tabular tasks."""
from __future__ import annotations
import hashlib, math, time
from dataclasses import dataclass
from typing import Iterable
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
SEED = 20260721
EPS = 1e-8

def _as_cat(s: pd.Series) -> pd.Series: return s.fillna("__MISSING__").astype(str)
def _as_num(s: pd.Series) -> pd.Series: return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        v = float(roc_auc_score(y, p))
        return max(v, 1.0 - v)
    except Exception: return 0.5

def _rank_scores_numeric(X: pd.DataFrame, y: np.ndarray, cols: list[str], seed: int) -> list[str]:
    if not cols: return []
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=12000, replace=False)) if n > 12000 else np.arange(n)
    A = []; valid_cols = []; aucs = {}
    for c in cols:
        s = _as_num(X[c])
        med = float(s.iloc[idx].median()) if s.iloc[idx].notna().any() else 0.0
        v = s.iloc[idx].fillna(med).to_numpy(dtype=np.float64)
        if np.nanstd(v) <= 1e-12: continue
        A.append(v); valid_cols.append(c); aucs[c] = _safe_auc(y[idx], v)
    if not valid_cols: return []
    mat = np.column_stack(A)
    try: mi = mutual_info_classif(mat, y[idx], random_state=seed, discrete_features=False)
    except Exception: mi = np.zeros(len(valid_cols))
    mi_order = {c: r for r, c in enumerate(np.array(valid_cols)[np.argsort(-mi)])}
    au_order = {c: r for r, c in enumerate(sorted(valid_cols, key=lambda z: aucs[z], reverse=True))}
    return sorted(valid_cols, key=lambda c: (mi_order[c] + au_order[c], -aucs[c], c))

def _rank_scores_cat(X: pd.DataFrame, y: np.ndarray, cols: list[str], seed: int) -> list[str]:
    if not cols: return []
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=15000, replace=False)) if n > 15000 else np.arange(n)
    scores = []
    for c in cols:
        s = _as_cat(X[c]).iloc[idx]
        codes, _ = pd.factorize(s, sort=True)
        if len(np.unique(codes)) <= 1: score = 0.0
        else:
            try: score = float(mutual_info_classif(codes.reshape(-1, 1), y[idx], discrete_features=True, random_state=seed)[0])
            except Exception: score = 0.0
        card = max(1, s.nunique(dropna=False))
        support = math.log1p(len(s) / card)
        scores.append((score * support, -card, c))
    scores.sort(reverse=True)
    return [c for _, _, c in scores]

def _fit_code(train_key: pd.Series, *apply_keys: pd.Series) -> tuple[np.ndarray, list[np.ndarray]]:
    vals = pd.Index(pd.unique(train_key))
    tr = vals.get_indexer(train_key).astype(np.float32)
    out = [vals.get_indexer(s).astype(np.float32) for s in apply_keys]
    return tr, out

def _freq_encode(train_key: pd.Series, *apply_keys: pd.Series) -> tuple[np.ndarray, list[np.ndarray]]:
    vc = train_key.value_counts(dropna=False)
    denom = max(len(train_key), 1)
    tr = train_key.map(vc).fillna(0).to_numpy(dtype=np.float32) / denom
    out = [s.map(vc).fillna(0).to_numpy(dtype=np.float32) / denom for s in apply_keys]
    return np.log1p(tr * denom).astype(np.float32), [np.log1p(v * denom).astype(np.float32) for v in out]

def _smoothed_mapping(key: pd.Series, y: np.ndarray, alpha: float, global_mean: float | None = None) -> pd.Series:
    if global_mean is None: global_mean = float(np.mean(y))
    df = pd.DataFrame({"key": key.to_numpy(), "y": y})
    st = df.groupby("key", sort=False, observed=True)["y"].agg(["sum", "count"])
    return ((st["sum"] + alpha * global_mean) / (st["count"] + alpha)).astype(float)

def _nested_target_encode(train_key: pd.Series, y: np.ndarray, apply_keys: list[pd.Series], seed: int, alpha: float) -> tuple[np.ndarray, list[np.ndarray]]:
    minority = int(np.bincount(y).min())
    n_splits = max(2, min(3, minority))
    inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=np.float32)
    for fi, vi in inner.split(np.zeros(len(y)), y):
        gm = float(np.mean(y[fi]))
        mapping = _smoothed_mapping(train_key.iloc[fi], y[fi], alpha, gm)
        oof[vi] = train_key.iloc[vi].map(mapping).fillna(gm).to_numpy(dtype=np.float32)
    gm = float(np.mean(y))
    mapping = _smoothed_mapping(train_key, y, alpha, gm)
    outs = [s.map(mapping).fillna(gm).to_numpy(dtype=np.float32) for s in apply_keys]
    return oof, outs

def _quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    v = values[np.isfinite(values)]
    if len(v) == 0: return np.array([-np.inf, np.inf])
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(v, qs))
    if len(edges) < 3: return np.array([-np.inf, np.inf])
    edges[0] = -np.inf; edges[-1] = np.inf
    return edges

def _bin_key(s: pd.Series, edges: np.ndarray) -> pd.Series:
    v = _as_num(s).to_numpy(dtype=float)
    codes = np.searchsorted(edges, v, side="right") - 1
    codes[~np.isfinite(v)] = -1
    return pd.Series(codes.astype(np.int32), index=s.index).astype(str)

def _hash_subset(frame: pd.DataFrame, cols: list[str], numeric_cols: set[str], round_digits: int = 2) -> pd.Series:
    pieces = []
    for c in cols:
        if c in numeric_cols: s = _as_num(frame[c]).round(round_digits).fillna(-9.87654321e37)
        else: s = _as_cat(frame[c])
        pieces.append(s.rename(c))
    if not pieces: return pd.Series(np.zeros(len(frame), dtype=np.uint64), index=frame.index)
    tmp = pd.concat(pieces, axis=1)
    return pd.util.hash_pandas_object(tmp, index=False).astype("uint64")

@dataclass
class FoldFeatures:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    names: list[str]
    details: dict

def _build_fold_features(Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, Xte: pd.DataFrame, seed: int) -> FoldFeatures:
    n = len(Xtr)
    object_cols = [c for c in Xtr.columns if pd.api.types.is_object_dtype(Xtr[c]) or pd.api.types.is_bool_dtype(Xtr[c]) or isinstance(Xtr[c].dtype, pd.CategoricalDtype)]
    numeric_all = [c for c in Xtr.columns if c not in object_cols]
    low_card_numeric = [c for c in numeric_all if Xtr[c].nunique(dropna=True) <= max(12, min(40, int(math.sqrt(max(n, 1)) / 2)))]
    cat_candidates = list(dict.fromkeys(object_cols + low_card_numeric))
    continuous = [c for c in numeric_all if c not in low_card_numeric and Xtr[c].nunique(dropna=True) >= 5]
    ranked_num = _rank_scores_numeric(Xtr, ytr, continuous or numeric_all, seed)
    ranked_cat = _rank_scores_cat(Xtr, ytr, cat_candidates, seed)
    tr_cols: list[np.ndarray] = []; va_cols: list[np.ndarray] = []; te_cols: list[np.ndarray] = []; names: list[str] = []

    def add(name: str, a, b, c):
        aa = np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        bb = np.nan_to_num(np.asarray(b, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        cc = np.nan_to_num(np.asarray(c, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        tr_cols.append(aa); va_cols.append(bb); te_cols.append(cc); names.append(name)

    for c in numeric_all:
        a = _as_num(Xtr[c]); b = _as_num(Xva[c]); d = _as_num(Xte[c])
        med = float(a.median()) if a.notna().any() else 0.0
        add(f"raw::{c}", a.fillna(med), b.fillna(med), d.fillna(med))
        if a.isna().any() or b.isna().any() or d.isna().any(): add(f"missing::{c}", a.isna(), b.isna(), d.isna())

    cat_te_cols = ranked_cat[: min(14, len(ranked_cat))]
    for j, c in enumerate(cat_te_cols):
        a, b, d = _as_cat(Xtr[c]), _as_cat(Xva[c]), _as_cat(Xte[c])
        code, (code_b, code_d) = _fit_code(a, b, d)
        freq, (freq_b, freq_d) = _freq_encode(a, b, d)
        card = max(a.nunique(dropna=False), 1)
        alpha = max(6.0, min(30.0, 0.5 * math.sqrt(n / card)))
        te, (te_b, te_d) = _nested_target_encode(a, ytr, [b, d], seed + 100 + j, alpha)
        add(f"cat_code::{c}", code, code_b, code_d)
        add(f"cat_freq::{c}", freq, freq_b, freq_d)
        add(f"cat_te::{c}", te, te_b, te_d)

    binned_keys: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}
    for j, c in enumerate(ranked_num[: min(10, len(ranked_num))]):
        a_num = _as_num(Xtr[c]).to_numpy(dtype=float)
        edges = _quantile_edges(a_num, bins=12 if n < 3000 else 20)
        a, b, d = _bin_key(Xtr[c], edges), _bin_key(Xva[c], edges), _bin_key(Xte[c], edges)
        binned_keys[c] = (a, b, d)
        code, (code_b, code_d) = _fit_code(a, b, d)
        te, (te_b, te_d) = _nested_target_encode(a, ytr, [b, d], seed + 300 + j, 20.0)
        add(f"qbin::{c}", code, code_b, code_d)
        add(f"qbin_te::{c}", te, te_b, te_d)

    pair_cats = ranked_cat[: min(7, len(ranked_cat))]
    pair_count = 0
    for i in range(len(pair_cats)):
        for j in range(i + 1, len(pair_cats)):
            c1, c2 = pair_cats[i], pair_cats[j]
            a = _as_cat(Xtr[c1]) + "\x1f" + _as_cat(Xtr[c2])
            b = _as_cat(Xva[c1]) + "\x1f" + _as_cat(Xva[c2])
            d = _as_cat(Xte[c1]) + "\x1f" + _as_cat(Xte[c2])
            freq, (freq_b, freq_d) = _freq_encode(a, b, d)
            alpha = max(12.0, min(50.0, math.sqrt(max(a.nunique(), 1))))
            te, (te_b, te_d) = _nested_target_encode(a, ytr, [b, d], seed + 500 + pair_count, alpha)
            add(f"cross_freq::{c1}::{c2}", freq, freq_b, freq_d)
            add(f"cross_te::{c1}::{c2}", te, te_b, te_d)
            pair_count += 1

    arith = ranked_num[: min(9, len(ranked_num))]
    num_cache = {}
    for c in arith:
        a = _as_num(Xtr[c]); med = float(a.median()) if a.notna().any() else 0.0
        num_cache[c] = (a.fillna(med).to_numpy(dtype=np.float32), _as_num(Xva[c]).fillna(med).to_numpy(dtype=np.float32), _as_num(Xte[c]).fillna(med).to_numpy(dtype=np.float32))
        x, xv, xt = num_cache[c]; scale = float(np.nanstd(x)) + EPS
        add(f"sq::{c}", np.square(np.clip(x / scale, -20, 20)), np.square(np.clip(xv / scale, -20, 20)), np.square(np.clip(xt / scale, -20, 20)))
        add(f"logabs::{c}", np.log1p(np.abs(x)), np.log1p(np.abs(xv)), np.log1p(np.abs(xt)))
    for i in range(len(arith)):
        for j in range(i + 1, len(arith)):
            c1, c2 = arith[i], arith[j]
            a1, b1, d1 = num_cache[c1]; a2, b2, d2 = num_cache[c2]
            s1, s2 = float(np.std(a1)) + EPS, float(np.std(a2)) + EPS
            z1, z2 = a1 / s1, a2 / s2
            zv1, zv2 = b1 / s1, b2 / s2
            zt1, zt2 = d1 / s1, d2 / s2
            add(f"mul::{c1}::{c2}", np.clip(z1 * z2, -50, 50), np.clip(zv1 * zv2, -50, 50), np.clip(zt1 * zt2, -50, 50))
            add(f"diff::{c1}::{c2}", z1 - z2, zv1 - zv2, zt1 - zt2)
            add(f"absdiff::{c1}::{c2}", np.abs(z1 - z2), np.abs(zv1 - zv2), np.abs(zt1 - zt2))
            add(f"ratio::{c1}::{c2}", np.clip(z1 / (np.abs(z2) + 0.25), -50, 50), np.clip(zv1 / (np.abs(zv2) + 0.25), -50, 50), np.clip(zt1 / (np.abs(zt2) + 0.25), -50, 50))

    group_cats = ranked_cat[: min(4, len(ranked_cat))]; group_nums = ranked_num[: min(5, len(ranked_num))]
    for ccat in group_cats:
        key_tr, key_va, key_te = _as_cat(Xtr[ccat]), _as_cat(Xva[ccat]), _as_cat(Xte[ccat])
        for cnum in group_nums:
            val_tr = _as_num(Xtr[cnum]); med = float(val_tr.median()) if val_tr.notna().any() else 0.0
            vtr, vva, vte = val_tr.fillna(med), _as_num(Xva[cnum]).fillna(med), _as_num(Xte[cnum]).fillna(med)
            tmp = pd.DataFrame({"k": key_tr.to_numpy(), "v": vtr.to_numpy()})
            stats = tmp.groupby("k", sort=False)["v"].agg(["mean", "std", "median", "count"])
            gm, gs = float(vtr.mean()), float(vtr.std()) + EPS
            mean_tr, mean_va, mean_te = key_tr.map(stats["mean"]).fillna(gm).to_numpy(float), key_va.map(stats["mean"]).fillna(gm).to_numpy(float), key_te.map(stats["mean"]).fillna(gm).to_numpy(float)
            std_tr, std_va, std_te = key_tr.map(stats["std"]).fillna(gs).to_numpy(float), key_va.map(stats["std"]).fillna(gs).to_numpy(float), key_te.map(stats["std"]).fillna(gs).to_numpy(float)
            add(f"group_diff::{cnum}::{ccat}", vtr.to_numpy() - mean_tr, vva.to_numpy() - mean_va, vte.to_numpy() - mean_te)
            add(f"group_ratio::{cnum}::{ccat}", vtr.to_numpy() / (np.abs(mean_tr) + 0.1 * gs), vva.to_numpy() / (np.abs(mean_va) + 0.1 * gs), vte.to_numpy() / (np.abs(mean_te) + 0.1 * gs))
            add(f"group_z::{cnum}::{ccat}", (vtr.to_numpy() - mean_tr) / (std_tr + 0.1 * gs), (vva.to_numpy() - mean_va) / (std_va + 0.1 * gs), (vte.to_numpy() - mean_te) / (std_te + 0.1 * gs))

    base_for_rsfc = list(dict.fromkeys(ranked_cat[: min(8, len(ranked_cat))] + ranked_num[: min(10, len(ranked_num))]))
    rng = np.random.default_rng(seed + 900); subsets: list[tuple[str, ...]] = []
    if len(base_for_rsfc) >= 2:
        subsets.append(tuple(base_for_rsfc[: min(12, len(base_for_rsfc))]))
        seen = set(subsets); tries = 0; max_subsets = 14 if n >= 1500 else 8
        while len(subsets) < max_subsets and tries < 300:
            tries += 1; k = int(rng.integers(2, min(5, len(base_for_rsfc)) + 1))
            s = tuple(sorted(rng.choice(base_for_rsfc, size=k, replace=False).tolist()))
            if s not in seen: seen.add(s); subsets.append(s)
    numeric_set = set(numeric_all)
    for j, subset in enumerate(subsets):
        a, b, d = _hash_subset(Xtr, list(subset), numeric_set), _hash_subset(Xva, list(subset), numeric_set), _hash_subset(Xte, list(subset), numeric_set)
        freq, (freq_b, freq_d) = _freq_encode(a, b, d)
        te, (te_b, te_d) = _nested_target_encode(a, ytr, [b, d], seed + 1000 + j, 20.0 if len(subset) <= 3 else 35.0)
        add(f"rsfc_freq::{j}", freq, freq_b, freq_d)
        add(f"rsfc_te::{j}", te, te_b, te_d)

    if numeric_all:
        def row_matrix(frame: pd.DataFrame): return np.column_stack([_as_num(frame[c]).to_numpy(float) for c in numeric_all])
        A, B, D = row_matrix(Xtr), row_matrix(Xva), row_matrix(Xte)
        for nm, fn in [("row_mean", lambda z: np.nanmean(z, axis=1)), ("row_std", lambda z: np.nanstd(z, axis=1)), ("row_min", lambda z: np.nanmin(z, axis=1)), ("row_max", lambda z: np.nanmax(z, axis=1)), ("row_median", lambda z: np.nanmedian(z, axis=1)), ("row_missing", lambda z: np.mean(~np.isfinite(z), axis=1))]:
            with np.errstate(all="ignore"): add(nm, fn(A), fn(B), fn(D))

    train_arr = np.column_stack(tr_cols).astype(np.float32, copy=False)
    valid_arr = np.column_stack(va_cols).astype(np.float32, copy=False)
    test_arr = np.column_stack(te_cols).astype(np.float32, copy=False)
    details = {"n_features": int(train_arr.shape[1]), "numeric": len(numeric_all), "continuous": len(continuous), "categorical_candidates": len(cat_candidates), "selected_numeric": ranked_num[:10], "selected_categorical": ranked_cat[:14], "cat_pairs": pair_count, "rsfc_subsets": len(subsets)}
    return FoldFeatures(train_arr, valid_arr, test_arr, names, details)

def cv_tabprep_lite(train: pd.DataFrame, y: np.ndarray, test: pd.DataFrame, features: list[str], splitter: StratifiedKFold, seed: int = SEED):
    import lightgbm as lgb
    X = train[features].reset_index(drop=True); T = test[features].reset_index(drop=True); y = np.asarray(y, dtype=int)
    oof = np.zeros(len(X), dtype=float); test_pred = np.zeros(len(T), dtype=float)
    fold_scores = []; fold_features = []; started = time.time()
    for fold, (fi, vi) in enumerate(splitter.split(X, y)):
        ff = _build_fold_features(X.iloc[fi].reset_index(drop=True), y[fi], X.iloc[vi].reset_index(drop=True), T, seed + 10000 * fold)
        n = len(fi)
        if n < 1000: leaves, child, l1, l2 = 7, 18, 0.3, 8.0
        elif n < 3000: leaves, child, l1, l2 = 15, 22, 0.2, 5.0
        elif n < 20000: leaves, child, l1, l2 = 31, 30, 0.15, 3.0
        else: leaves, child, l1, l2 = 31, 50, 0.15, 4.0
        model = lgb.LGBMClassifier(objective="binary", n_estimators=1400, learning_rate=0.02, num_leaves=leaves, max_depth=-1, min_child_samples=child, subsample=0.82, subsample_freq=1, colsample_bytree=0.72, reg_alpha=l1, reg_lambda=l2, max_bin=255, random_state=seed + fold, n_jobs=3, verbosity=-1)
        model.fit(ff.train, y[fi], eval_set=[(ff.valid, y[vi])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        pv = model.predict_proba(ff.valid)[:, 1]; pt = model.predict_proba(ff.test)[:, 1]
        oof[vi] = pv; test_pred += pt / splitter.n_splits
        fold_scores.append(float(roc_auc_score(y[vi], pv))); fold_features.append(ff.details)
    return oof, test_pred, {"fold_auc": fold_scores, "mean_fold_auc": float(np.mean(fold_scores)), "global_oof_auc": float(roc_auc_score(y, oof)), "seconds": round(time.time() - started, 3), "fold_features": fold_features}
