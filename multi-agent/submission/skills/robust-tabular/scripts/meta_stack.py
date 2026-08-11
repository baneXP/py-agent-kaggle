#!/usr/bin/env python3
"""Cross-fit meta stacker with correlation-aware, in-fold source selection.

V24 change. V23 chose *which* prediction sources entered the stack using
full-data statistics (family AUC floors, exact-duplicate signatures) and only
cross-fitted the meta-weights afterwards. Selection is itself a fit: doing it on
all rows makes the reported meta OOF AUC optimistic by roughly the amount of
selection freedom used - which is precisely the quantity our 5e-4 promotion
margin is supposed to be measured against.

Three fixes:
  1. Source admission (family AUC floors) moves inside the outer meta fold and
     uses fit rows only.
  2. Exact-signature de-duplication is replaced by a correlation cap. Two
     sources at Spearman 0.998 are not duplicates by signature but are
     duplicates statistically: they inflate collinearity for the logistic
     stacker and let one model family dominate Caruana's greedy averaging.
  3. Emitted candidates are de-correlated on their *test* predictions, so the
     at most three submitted files are three different bets rather than three
     views of one bet. This matters because only two survive to private scoring.

The V23 dead `crossfit_simplex` / `compositions` path is removed: with eight
sources it would have enumerated tens of thousands of weight vectors inside a
60-minute session.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
for p in [HERE]:
    if (p / 'common.py').is_file():
        sys.path.insert(0, str(p))
        break
from common import load_task, record_candidates, runtime_workdir, write_submission

SEED = 20260710
CORR_CAP = 0.995        # source-level Spearman cap inside each meta fold
EMIT_CORR_CAP = 0.999   # test-prediction Spearman cap between emitted candidates
MAX_SOURCES = 8
FAMILY_MARGIN = {'small_': 0.06, 'rsfc_': 0.04, 'temporal_': 0.02}


def rank01(x):
    return pd.Series(np.asarray(x, float)).rank(method='average', pct=True).to_numpy()


def logit(x):
    x = np.clip(np.asarray(x, float), 1e-5, 1 - 1e-5)
    return np.clip(np.log(x / (1 - x)), -8, 8)


def auc(y, p):
    return float(roc_auc_score(y, p))


def emit(x):
    print('META_MANIFEST=' + json.dumps(x, sort_keys=True, separators=(',', ':')))


def splitter(y):
    minority = int(np.bincount(np.asarray(y, int)).min())
    return StratifiedKFold(n_splits=max(2, min(5, minority)), shuffle=True, random_state=SEED)


def in_fold_admit(names, R_fit, y_fit):
    """Family AUC floors evaluated on fit rows only."""
    aucs = {j: auc(y_fit, R_fit[:, j]) for j in range(R_fit.shape[1])}
    core = [aucs[j] for j, n in enumerate(names) if n.startswith('core_')]
    core_best = max(core) if core else max(aucs.values())
    admitted = []
    for j, name in enumerate(names):
        floor = None
        for family, margin in FAMILY_MARGIN.items():
            if name.startswith(family):
                floor = core_best - margin
                if family == 'small_' and core_best >= 0.84:
                    floor = max(core_best - 0.06, 0.84)
        if floor is not None and aucs[j] < floor:
            continue
        admitted.append(j)
    return admitted or list(range(R_fit.shape[1])), aucs, core_best


def greedy_decorrelate(candidates, aucs, R_fit, cap=CORR_CAP, max_k=MAX_SOURCES):
    """Keep the strongest source, then only sources that add a new direction."""
    order = sorted(candidates, key=lambda j: -aucs[j])
    kept = []
    for j in order:
        if kept:
            block = np.column_stack([R_fit[:, j]] + [R_fit[:, k] for k in kept])
            corr = np.abs(np.corrcoef(block.T)[0, 1:])
            if np.nanmax(corr) > cap:
                continue
        kept.append(int(j))
        if len(kept) >= max_k:
            break
    return kept or [int(order[0])]


def choose_caruana(R, y, max_members=20, min_gain=1e-5):
    single = [auc(y, R[:, j]) for j in range(R.shape[1])]
    chosen = [int(np.argmax(single))]
    current = R[:, chosen[0]].copy()
    score = single[chosen[0]]
    for _ in range(max_members - 1):
        best = (score, None, None)
        n = len(chosen)
        for j in range(R.shape[1]):
            p = (n * current + R[:, j]) / (n + 1)
            s = auc(y, p)
            if s > best[0] + 1e-12:
                best = (s, j, p)
        if best[1] is None or best[0] < score + min_gain:
            break
        score, j, current = best
        chosen.append(int(j))
    w = np.bincount(chosen, minlength=R.shape[1]).astype(float)
    w /= w.sum()
    return w


def crossfit(y, names, R, RT, X, XT, kind, C=None):
    """Cross-fit the *entire* procedure: admission, de-correlation and weights."""
    n_sources = R.shape[1]
    oof = np.zeros(len(y))
    tests, picks, neffs = [], [], []
    for tr, va in splitter(y).split(R, y):
        admitted, aucs, _ = in_fold_admit(names, R[tr], y[tr])
        keep = greedy_decorrelate(admitted, aucs, R[tr])
        picks.append([names[j] for j in keep])
        if kind == 'crossfit_logstack':
            cols = list(keep) + [j + n_sources for j in keep]
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=C, max_iter=2500, solver='lbfgs'))
            model.fit(X[np.ix_(tr, cols)], y[tr])
            oof[va] = model.predict_proba(X[np.ix_(va, cols)])[:, 1]
            tests.append(model.predict_proba(XT[:, cols])[:, 1])
            neffs.append(float(len(keep)))
        else:
            w = choose_caruana(R[np.ix_(tr, keep)], y[tr])
            w = .90 * w + .10 * np.ones(len(w)) / len(w)
            oof[va] = R[np.ix_(va, keep)] @ w
            tests.append(RT[:, keep] @ w)
            neffs.append(float(1.0 / np.sum(w ** 2)))
    return oof, np.mean(tests, axis=0), picks, float(np.mean(neffs))


def main():
    work = runtime_workdir()
    state = work / 'portfolio_core_state.npz'
    if not state.is_file():
        emit({'candidates': [], 'reason': 'missing_core_state'})
        return
    train, test, sample, idc, target, features, y = load_task(work)

    sources = []
    for prefix, spath in [('core', state), ('zs', work / 'zeroshot_lgb_state.npz'),
                          ('xgb', work / 'xgb_expert_state.npz'), ('small', work / 'small_expert_state.npz'),
                          ('gam', work / 'gam_expert_state.npz'), ('rsfc', work / 'rsfc_expert_state.npz'),
                          ('temporal', work / 'temporal_expert_state.npz'),
                          ('nn', work / 'neural_challenger_state.npz')]:
        if not spath.is_file():
            continue
        zz = np.load(spath)
        for raw in sorted({k[:-4] for k in zz.files if k.endswith('_oof') and k[:-4] + '_test' in zz.files}):
            o = np.asarray(zz[raw + '_oof'], float)
            t = np.asarray(zz[raw + '_test'], float)
            if len(o) != len(y) or len(t) != len(test):
                continue
            if not np.isfinite(o).all() or not np.isfinite(t).all() or np.std(o) < 1e-9:
                continue
            sources.append((f'{prefix}_{raw}', o, t))

    # Only exact duplicates are dropped here; statistical near-duplicates are
    # handled inside each fold, where the decision can be made without leakage.
    unique, signatures = [], set()
    for name, o, t in sources:
        sig = np.round(rank01(o), 10).tobytes()
        if sig in signatures:
            continue
        signatures.add(sig)
        unique.append((name, o, t))

    names = [x[0] for x in unique]
    if len(names) < 2:
        emit({'candidates': [], 'reason': 'insufficient_models', 'models': names})
        return

    O = np.column_stack([x[1] for x in unique])
    T = np.column_stack([x[2] for x in unique])
    R = np.column_stack([rank01(O[:, j]) for j in range(O.shape[1])])
    RT = np.column_stack([rank01(T[:, j]) for j in range(T.shape[1])])
    X = np.column_stack([logit(O[:, j]) for j in range(O.shape[1])] + [R[:, j] for j in range(O.shape[1])])
    XT = np.column_stack([logit(T[:, j]) for j in range(T.shape[1])] + [RT[:, j] for j in range(T.shape[1])])

    corr = np.corrcoef(R.T)
    redundancy = float(np.nanmax(corr - np.eye(len(names)))) if len(names) > 1 else 0.0

    specs = []
    for cid, C in [('meta_logstack_0p03', .03), ('meta_logstack_0p3', .3)]:
        o, p, picks, neff = crossfit(y, names, R, RT, X, XT, 'crossfit_logstack', C=C)
        specs.append((cid, 'crossfit_logstack', o, p, {'C': C, 'sources_per_fold': picks, 'n_eff': neff}))
    o, p, picks, neff = crossfit(y, names, R, RT, X, XT, 'crossfit_caruana')
    specs.append(('meta_caruana', 'crossfit_caruana', o, p, {'sources_per_fold': picks, 'n_eff': neff}))

    base_best = max(auc(y, O[:, j]) for j in range(O.shape[1]))
    scored = []
    for cid, kind, o, p, meta in specs:
        if not np.isfinite(o).all() or not np.isfinite(p).all() or np.std(o) < 1e-9:
            continue
        s = auc(y, o)
        if s < base_best - .02:
            continue
        scored.append({'id': cid, 'kind': kind, 'oof_auc': s, 'oof': o, 'test': p, 'meta': meta})
    scored.sort(key=lambda x: x['oof_auc'], reverse=True)

    # Diversity-aware emission on test-prediction ranks.
    emitted, save, metrics = [], {}, []
    for cand in scored:
        rt = rank01(cand['test'])
        dup = None
        for prev in emitted:
            rho = float(np.corrcoef(rt, rank01(prev['test']))[0, 1])
            if rho > EMIT_CORR_CAP:
                dup = rho
                break
        metrics.append({'id': cand['id'], 'kind': cand['kind'], 'oof_auc': cand['oof_auc'],
                        'n_eff': cand['meta'].get('n_eff'), 'suppressed_corr': dup})
        if dup is not None:
            continue
        path = write_submission(work / f"{cand['id']}.csv", cand['test'], test, sample, idc, target)
        cand['path'] = path
        emitted.append(cand)
        save[cand['id'] + '_oof'] = cand['oof']
        save[cand['id'] + '_test'] = cand['test']
        if len(emitted) >= 3:
            break

    if save:
        np.savez_compressed(work / 'meta_stack_state.npz', **save)
    payload = [{'id': c['id'], 'path': c['path'], 'kind': c['kind'], 'oof_auc': c['oof_auc'],
                'n_eff': c['meta'].get('n_eff')} for c in emitted]
    if payload:
        record_candidates(work, 'meta', payload)
    emit({'candidates': payload, 'all_metrics': metrics, 'models': names,
          'base_best_oof': base_best, 'source_max_corr': redundancy,
          'sources_per_fold': specs[-1][4]['sources_per_fold'], 'rows': len(y)})


if __name__ == '__main__':
    main()
