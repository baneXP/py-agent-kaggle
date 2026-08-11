#!/usr/bin/env python3
"""Random-subset feature-compression expert with a serve-consistent encoder.

V24 change: train-serve skew removal.

V23 built the model's training matrix from *inner* out-of-fold encoders (fit on
~53% of the rows) but built the validation and test matrices from an *outer*
encoder (fit on ~80% of the rows), and independently re-derived the
mutual-information feature subsets and the numeric bin edges on each side.
Three distinct skews followed:

  1. Column-identity skew. `mutual_info_classif` was recomputed on each fit set,
     so training column j and serving column j could encode different feature
     subsets. This is silent: the shapes match, the semantics do not.
  2. Discretisation skew. Quantile bin edges were recomputed per fit set, so the
     same raw value could land in different bins at train and serve time.
  3. Shrinkage / coverage skew. Smaller fit sets give smaller per-key counts,
     hence stronger shrinkage toward the prior and a higher unseen-key rate. The
     model learned split thresholds on one scale and then scored on another.

V24 freezes the schema (subsets + bin edges) once per outer fold from fit rows
only, and serves every matrix from encoders with an identical fit-set size. The
remaining difference is variance-only and in the benign direction: serve-time
features are the mean of K inner encoders, so they are slightly *cleaner* than
the single-encoder features the model trained on. A model robust to noisier
inputs than it receives is safe; the reverse is not.

Leakage safety is unchanged: no encoder ever sees the label of a row it scores.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import categorical_columns, load_task, record_candidates, runtime_workdir, write_submission

SEED = 20260721
ALPHA = 12.0          # frozen from V23: smoothing strength of the target encoder
N_INNER = 3           # frozen from V23: inner folds used to build training features
MAX_SUBSETS = 45      # frozen from V23
SEP = "\x1f"
SMD_LIMIT = 0.50      # calibrated: clean runs measured 0.12-0.23, V23-style breakage 14-17
UNSEEN_GAP_LIMIT = 0.02   # calibrated: clean runs measured 0.002-0.006, V23-style breakage 0.041-0.050


def emit(x):
    print("RSFC_MANIFEST=" + json.dumps(x, sort_keys=True, separators=(",", ":")))


def auc(y, p):
    return float(roc_auc_score(y, p))


def splitter(y):
    m = int(np.bincount(np.asarray(y, int)).min())
    return StratifiedKFold(n_splits=max(2, min(5, m)), shuffle=True, random_state=SEED)


def key_for(frame, cols):
    """Concatenate discretised columns into one interaction key (vectorised)."""
    s = frame[cols[0]].astype(str)
    for c in cols[1:]:
        s = s.str.cat(frame[c].astype(str), sep=SEP)
    return s


def freeze_schema(train, test, features, fit_idx, y_fit, top_k=8):
    """Derive bin edges and interaction subsets ONCE, from fit rows only.

    Both the training-side and the serving-side encoders consume this identical
    schema, which is what removes skews (1) and (2) from the module docstring.
    """
    A_cols, B_cols = {}, {}
    for c in features:
        tr, te = train[c], test[c]
        if pd.api.types.is_numeric_dtype(tr) and tr.nunique(dropna=True) > 24:
            x = pd.to_numeric(tr, errors="coerce")
            xt = pd.to_numeric(te, errors="coerce")
            vals = x.iloc[fit_idx].dropna().to_numpy()
            if len(vals):
                edges = np.unique(np.quantile(vals, np.linspace(0, 1, 13)))
                if len(edges) > 2:
                    A_cols[c] = pd.cut(x, edges, include_lowest=True, duplicates="drop").astype(str).fillna("__NA__")
                    B_cols[c] = pd.cut(xt, edges, include_lowest=True, duplicates="drop").astype(str).fillna("__NA__")
                    continue
        A_cols[c] = tr.fillna("__NA__").astype(str)
        B_cols[c] = te.fillna("__NA__").astype(str)
    A = pd.DataFrame(A_cols, columns=features)
    B = pd.DataFrame(B_cols, columns=features)
    M = np.column_stack([pd.factorize(A[c], sort=True)[0] for c in features])
    mi = mutual_info_classif(M[fit_idx], y_fit, discrete_features=True, random_state=SEED)
    top = [features[i] for i in np.argsort(-mi)[: min(top_k, len(features))]]
    subsets = [(c,) for c in top]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            subsets.append((top[i], top[j]))
    t5 = top[:5]
    for i in range(len(t5)):
        for j in range(i + 1, len(t5)):
            for k in range(j + 1, len(t5)):
                subsets.append((t5[i], t5[j], t5[k]))
    return A, B, subsets[:MAX_SUBSETS]


def fit_encoder(keys_fit, y_fit):
    prior = float(np.mean(y_fit))
    df = pd.DataFrame({"k": keys_fit.to_numpy(), "y": np.asarray(y_fit, float)})
    agg = df.groupby("k").y.agg(["mean", "count"])
    smooth = (agg["mean"] * agg["count"] + prior * ALPHA) / (agg["count"] + ALPHA)
    freq = agg["count"] / len(df)
    return smooth, freq, prior


def apply_encoder(enc, keys):
    smooth, freq, prior = enc
    mapped = keys.map(smooth)
    unseen = float(mapped.isna().mean())
    te = mapped.fillna(prior).to_numpy(float)
    fr = keys.map(freq).fillna(0.0).to_numpy(float)
    return te, fr, unseen


def skew_report(x_train, x_serve, unseen_train, unseen_serve):
    """Quantify residual train-serve mismatch on the encoded matrices."""
    mt, ms = x_train.mean(axis=0), x_serve.mean(axis=0)
    st, ss = x_train.std(axis=0), x_serve.std(axis=0)
    pooled = np.sqrt(0.5 * (st ** 2 + ss ** 2)) + 1e-9
    smd = np.abs(ms - mt) / pooled
    ratio = (ss + 1e-9) / (st + 1e-9)
    return {
        "smd_max": float(np.max(smd)),
        "smd_median": float(np.median(smd)),
        "sd_ratio_median": float(np.median(ratio)),
        "unseen_train": float(unseen_train),
        "unseen_serve": float(unseen_serve),
        "unseen_gap": float(abs(unseen_serve - unseen_train)),
    }


def build_fold_matrices(A, B, subsets, y, fit_rows, val_rows, inner_seed):
    """Training features are inner out-of-fold; serve features are the mean of
    the same inner encoders. Fit-set size, alpha and schema match on both sides,
    so shrinkage strength and unseen-key behaviour match by construction.
    """
    n_cols = 2 * len(subsets)
    keys_train = {cols: key_for(A, cols) for cols in subsets}
    keys_test = {cols: key_for(B, cols) for cols in subsets}
    x_fit = np.zeros((len(fit_rows), n_cols), dtype=float)
    x_val = np.zeros((len(val_rows), n_cols), dtype=float)
    x_test = np.zeros((len(B), n_cols), dtype=float)
    pos = {int(r): i for i, r in enumerate(fit_rows)}
    inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=inner_seed)
    unseen_train, unseen_serve = [], []
    for itr, iva in inner.split(np.zeros(len(fit_rows)), y[fit_rows]):
        rows_fit = fit_rows[itr]
        rows_out = fit_rows[iva]
        out_pos = np.array([pos[int(r)] for r in rows_out], dtype=int)
        for s, cols in enumerate(subsets):
            kt = keys_train[cols]
            enc = fit_encoder(kt.iloc[rows_fit], y[rows_fit])
            te, fr, u = apply_encoder(enc, kt.iloc[rows_out])
            x_fit[out_pos, 2 * s] = te
            x_fit[out_pos, 2 * s + 1] = fr
            unseen_train.append(u)
            te, fr, u = apply_encoder(enc, kt.iloc[val_rows])
            x_val[:, 2 * s] += te / N_INNER
            x_val[:, 2 * s + 1] += fr / N_INNER
            unseen_serve.append(u)
            te, fr, _ = apply_encoder(enc, keys_test[cols])
            x_test[:, 2 * s] += te / N_INNER
            x_test[:, 2 * s + 1] += fr / N_INNER
    return x_fit, x_val, x_test, float(np.mean(unseen_train)), float(np.mean(unseen_serve))


def main():
    st = time.time()
    w = runtime_workdir()
    try:
        train, test, sample, idc, target, features, y = load_task(w)
        cats = categorical_columns(train, features)
        if not (500 <= len(train) <= 20000 and len(features) <= 40 and 0 < len(cats) < len(features)):
            emit({"candidate": None, "reason": "schema_gate", "rows": len(train),
                  "features": len(features), "categorical": len(cats)})
            return
        import lightgbm as lgb

        sp = splitter(y)
        oof = np.zeros(len(train))
        pred = np.zeros(len(test))
        counts, skews = [], []
        for f, (tr, va) in enumerate(sp.split(train, y)):
            A, B, subsets = freeze_schema(train, test, features, tr, y[tr])
            counts.append(len(subsets))
            x_fit, x_val, x_test, u_tr, u_sv = build_fold_matrices(A, B, subsets, y, tr, va, SEED + f)
            skews.append(skew_report(x_fit, x_val, u_tr, u_sv))
            model = lgb.LGBMClassifier(objective="binary", n_estimators=900, learning_rate=.025, num_leaves=15,
                                       min_child_samples=20, subsample=.85, subsample_freq=1, colsample_bytree=.8,
                                       reg_alpha=.2, reg_lambda=2.0, random_state=SEED + f, n_jobs=3, verbosity=-1)
            model.fit(x_fit, y[tr], eval_set=[(x_val, y[va])], eval_metric="auc",
                      callbacks=[lgb.early_stopping(70, verbose=False), lgb.log_evaluation(0)])
            oof[va] = model.predict_proba(x_val)[:, 1]
            pred += model.predict_proba(x_test)[:, 1] / sp.n_splits

        s = auc(y, oof)
        skew = {k: float(np.max([d[k] for d in skews])) for k in ("smd_max", "unseen_gap")}
        skew["smd_median"] = float(np.median([d["smd_median"] for d in skews]))
        skew["sd_ratio_median"] = float(np.median([d["sd_ratio_median"] for d in skews]))
        skew["unseen_train"] = float(np.mean([d["unseen_train"] for d in skews]))
        skew["unseen_serve"] = float(np.mean([d["unseen_serve"] for d in skews]))
        skew_ok = skew["smd_max"] <= SMD_LIMIT and skew["unseen_gap"] <= UNSEEN_GAP_LIMIT

        path = write_submission(w / "rsfc_expert.csv", pred, test, sample, idc, target)
        np.savez_compressed(w / "rsfc_expert_state.npz",
                            rsfc_oof=np.asarray(oof, np.float32), rsfc_test=np.asarray(pred, np.float32))
        core_best = None
        core_path = w / "portfolio_core_state.npz"
        if core_path.is_file():
            try:
                z = np.load(core_path)
                core_best = max(auc(y, np.asarray(z[k], float)) for k in z.files if k.endswith("_oof"))
            except Exception:
                core_best = None
        accepted = bool((core_best is None or s >= core_best - 0.04) and skew_ok)
        if accepted:
            record_candidates(w, "rsfc", [{"path": path, "kind": "rsfc_expert", "oof_auc": s}])
        emit({"candidate": path if accepted else None, "accepted": accepted, "oof_auc": s,
              "core_best_oof": core_best, "subsets": counts, "skew": skew, "skew_ok": skew_ok,
              "seconds": round(time.time() - st, 3)})
    except Exception as e:
        emit({"candidate": None, "reason": "error", "error": f"{type(e).__name__}:{e}",
              "seconds": round(time.time() - st, 3)})


if __name__ == "__main__":
    main()
