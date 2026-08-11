#!/usr/bin/env python3
"""Conservative date/time feature specialist for binary tabular tasks."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import encoded_frames, load_task, rank_unit, record_candidates, runtime_workdir, write_submission

SEED = 20260725
DATE_NAME = re.compile(r"(^|_)(date|datetime|timestamp|time|created|updated|event|start|end|birth|signup)(_|$)", re.I)


def emit(payload):
    print("TEMPORAL_MANIFEST=" + json.dumps(payload, sort_keys=True, separators=(",", ":")))


def auc(y, p):
    return float(roc_auc_score(y, p))


def folds(y):
    minority = int(np.bincount(np.asarray(y, int)).min())
    return StratifiedKFold(n_splits=max(2, min(5, minority)), shuffle=True, random_state=SEED)


def _parse_candidate(series: pd.Series, name: str):
    nonnull = series.dropna()
    if len(nonnull) < max(20, int(0.15 * len(series))):
        return None, 0.0

    parsed = None
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    elif pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        med = float(vals.dropna().median()) if vals.notna().any() else np.nan
        name_hint = bool(DATE_NAME.search(str(name)))
        # YYYYMMDD integers or plausible Unix timestamps; never parse arbitrary IDs as nanoseconds.
        if name_hint and np.isfinite(med):
            if 19000101 <= med <= 21001231:
                parsed = pd.to_datetime(vals.round().astype("Int64").astype(str), format="%Y%m%d", errors="coerce", utc=True)
            elif 1e9 <= med <= 2.5e9:
                parsed = pd.to_datetime(vals, unit="s", errors="coerce", utc=True)
            elif 1e12 <= med <= 2.5e12:
                parsed = pd.to_datetime(vals, unit="ms", errors="coerce", utc=True)
    else:
        text = series.astype("string")
        sample = text.dropna().head(2000)
        separator_rate = float(sample.str.contains(r"[-/:T ]", regex=True, na=False).mean()) if len(sample) else 0.0
        if DATE_NAME.search(str(name)) or separator_rate >= 0.70:
            try:
                parsed = pd.to_datetime(text, errors="coerce", utc=True, format="mixed")
            except TypeError:
                parsed = pd.to_datetime(text, errors="coerce", utc=True)

    if parsed is None:
        return None, 0.0
    rate = float(parsed.notna().mean())
    years = parsed.dt.year.dropna()
    plausible = float(years.between(1900, 2100).mean()) if len(years) else 0.0
    unique = int(parsed.nunique(dropna=True))
    if rate < 0.72 or plausible < 0.95 or unique < 8:
        return None, rate
    return parsed, rate


def detect_dates(train, test, features):
    combined = pd.concat([train[features], test[features]], axis=0, ignore_index=True)
    found = []
    for c in features:
        parsed, rate = _parse_candidate(combined[c], str(c))
        if parsed is not None:
            found.append((c, parsed, rate))
    found.sort(key=lambda x: (-x[2], str(x[0])))
    return found[:3]


def make_frames(train, test, features, detected):
    date_cols = [c for c, _, _ in detected]
    base_features = [c for c in features if c not in date_cols]
    if base_features:
        xtr, xte, cats = encoded_frames(train, test, base_features)
    else:
        xtr = pd.DataFrame(index=np.arange(len(train)))
        xte = pd.DataFrame(index=np.arange(len(test)))
        cats = []

    ntr = len(train)
    parsed_map = {}
    for col, parsed, _ in detected:
        parsed_map[col] = parsed
        trd = parsed.iloc[:ntr].reset_index(drop=True)
        ted = parsed.iloc[ntr:].reset_index(drop=True)
        safe = re.sub(r"\W+", "_", str(col)).strip("_") or "date"
        for frame, dt in ((xtr, trd), (xte, ted)):
            seconds = dt.astype("int64", copy=False).astype("float64") / 1e9
            seconds[dt.isna().to_numpy()] = np.nan
            frame[f"{safe}__ordinal_days"] = seconds / 86400.0
            frame[f"{safe}__year"] = dt.dt.year.astype("float64")
            frame[f"{safe}__month"] = dt.dt.month.astype("float64")
            frame[f"{safe}__day"] = dt.dt.day.astype("float64")
            frame[f"{safe}__dow"] = dt.dt.dayofweek.astype("float64")
            frame[f"{safe}__doy"] = dt.dt.dayofyear.astype("float64")
            frame[f"{safe}__quarter"] = dt.dt.quarter.astype("float64")
            try:
                frame[f"{safe}__week"] = dt.dt.isocalendar().week.astype("float64").to_numpy()
            except Exception:
                pass
            frame[f"{safe}__is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype("float64")
            frame[f"{safe}__missing"] = dt.isna().astype("float64")
            month = dt.dt.month.astype("float64")
            dow = dt.dt.dayofweek.astype("float64")
            frame[f"{safe}__month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
            frame[f"{safe}__month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
            frame[f"{safe}__dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
            frame[f"{safe}__dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    # Cross-date durations are often more causal and stable than absolute timestamps.
    if len(date_cols) >= 2:
        for i in range(len(date_cols)):
            for j in range(i + 1, len(date_cols)):
                a, b = date_cols[i], date_cols[j]
                pa, pb = parsed_map[a], parsed_map[b]
                diff = (pb - pa).dt.total_seconds() / 86400.0
                name = f"date_delta_{i}_{j}_days"
                xtr[name] = diff.iloc[:ntr].reset_index(drop=True)
                xte[name] = diff.iloc[ntr:].reset_index(drop=True)

    # Explicit row missingness is cheap and can expose collection-process signal.
    trmiss = train[features].isna()
    temiss = test[features].isna()
    xtr["__row_missing_count"] = trmiss.sum(axis=1).astype(float).to_numpy()
    xte["__row_missing_count"] = temiss.sum(axis=1).astype(float).to_numpy()
    xtr["__row_missing_fraction"] = trmiss.mean(axis=1).astype(float).to_numpy()
    xte["__row_missing_fraction"] = temiss.mean(axis=1).astype(float).to_numpy()

    xtr = xtr.replace([np.inf, -np.inf], np.nan)
    xte = xte.replace([np.inf, -np.inf], np.nan)
    return xtr, xte, cats


def fit_lgb(xtr, y, xte, cats, splitter):
    import lightgbm as lgb

    xtr = xtr.copy()
    xte = xte.copy()
    for c in cats:
        if c in xtr.columns:
            xtr[c] = xtr[c].round().astype("int64").astype("category")
            xte[c] = xte[c].round().astype("int64").astype("category")
    oof = np.zeros(len(xtr), dtype=float)
    pred = np.zeros(len(xte), dtype=float)
    fold_scores = []
    for fold, (fi, vi) in enumerate(splitter.split(xtr, y)):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1400,
            learning_rate=0.025,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=max(10, min(30, len(fi) // 120)),
            subsample=0.90,
            colsample_bytree=0.82,
            reg_alpha=0.08,
            reg_lambda=0.75,
            random_state=SEED + fold,
            n_jobs=3,
            verbosity=-1,
        )
        model.fit(
            xtr.iloc[fi], y[fi],
            eval_set=[(xtr.iloc[vi], y[vi])],
            eval_metric="auc",
            categorical_feature=[c for c in cats if c in xtr.columns],
            callbacks=[lgb.early_stopping(90, verbose=False), lgb.log_evaluation(0)],
        )
        oof[vi] = model.predict_proba(xtr.iloc[vi])[:, 1]
        pred += model.predict_proba(xte)[:, 1] / splitter.n_splits
        fold_scores.append(auc(y[vi], oof[vi]))
    return oof, pred, fold_scores


def core_sources(work, y):
    path = work / "portfolio_core_state.npz"
    if not path.is_file():
        return []
    zz = np.load(path)
    out = []
    for raw in sorted({k[:-4] for k in zz.files if k.endswith("_oof") and k[:-4] + "_test" in zz.files}):
        o = np.asarray(zz[raw + "_oof"], float)
        t = np.asarray(zz[raw + "_test"], float)
        if len(o) == len(y) and np.isfinite(o).all() and np.isfinite(t).all() and np.std(o) > 1e-9:
            out.append((raw, o, t, auc(y, o)))
    return out


def main():
    started = time.time()
    work = runtime_workdir()
    try:
        train, test, sample, idc, target, features, y = load_task(work)
        if len(train) > 30000 or len(features) > 300:
            emit({"candidate": None, "reason": "resource_gate", "rows": len(train), "features": len(features)})
            return
        detected = detect_dates(train, test, features)
        if not detected:
            emit({"candidate": None, "reason": "no_reliable_date_columns", "rows": len(train), "features": len(features)})
            return
        core = core_sources(work, y)
        if not core:
            emit({"candidate": None, "reason": "missing_core_state"})
            return
        best_name, best_oof, best_test, core_best = max(core, key=lambda x: x[3])
        xtr, xte, cats = make_frames(train, test, features, detected)
        oof, pred, fold_scores = fit_lgb(xtr, y, xte, cats, folds(y))
        temporal_auc = auc(y, oof)
        corr = float(pd.Series(rank_unit(best_oof)).corr(pd.Series(rank_unit(oof)), method="spearman"))

        # Persist only a competitive, non-duplicate source for the later cross-fit meta stage.
        source_ok = temporal_auc >= max(0.55, core_best - 0.02) and corr < 0.9985
        if source_ok:
            np.savez_compressed(
                work / "temporal_expert_state.npz",
                temporal_oof=np.asarray(oof, np.float32),
                temporal_test=np.asarray(pred, np.float32),
            )

        ro, rt = rank_unit(oof), rank_unit(pred)
        co, ct = rank_unit(best_oof), rank_unit(best_test)
        specs = [("temporal", temporal_auc, pred, 1.0)]
        for w in (0.20, 0.35, 0.50):
            blend_oof = w * ro + (1.0 - w) * co
            blend_test = w * rt + (1.0 - w) * ct
            specs.append((f"temporal_rankblend_{w:.2f}", auc(y, blend_oof), blend_test, w))
        kind, best_score, best_pred, weight = max(specs, key=lambda x: x[1])

        # Direct public candidate is emitted only after an actual OOF improvement over the frozen core.
        min_gain = 0.0005
        stable = float(np.std(fold_scores)) <= 0.10
        accepted = bool(source_ok and stable and best_score >= core_best + min_gain)
        candidate = None
        if accepted:
            candidate = write_submission(work / "temporal_structure_candidate.csv", best_pred, test, sample, idc, target)
            record_candidates(work, "temporal", [{"path": candidate, "kind": kind, "oof_auc": best_score}])

        emit({
            "candidate": candidate,
            "accepted": accepted,
            "source_saved": source_ok,
            "kind": kind,
            "blend_weight": weight,
            "oof_auc": best_score,
            "temporal_auc": temporal_auc,
            "core_best_oof": core_best,
            "core_model": best_name,
            "gain": best_score - core_best,
            "spearman": corr,
            "fold_std": float(np.std(fold_scores)),
            "date_columns": [str(c) for c, _, _ in detected],
            "parse_rates": {str(c): rate for c, _, rate in detected},
            "rows": len(train),
            "features_out": int(xtr.shape[1]),
            "seconds": round(time.time() - started, 3),
        })
    except Exception as exc:
        emit({"candidate": None, "reason": "error", "error": f"{type(exc).__name__}:{exc}", "seconds": round(time.time() - started, 3)})


if __name__ == "__main__":
    main()
