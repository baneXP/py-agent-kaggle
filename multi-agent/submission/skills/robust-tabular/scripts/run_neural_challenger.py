"""Schema-gated RealMLP/TabM-inspired numeric challenger."""
from __future__ import annotations
import copy, gc, json, os, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import categorical_columns, load_task, rank_unit, record_candidates, runtime_workdir, write_submission

SEED = 20260716

def _emit(payload: dict) -> None:
    print("SPECIALIST_MANIFEST=" + json.dumps(payload, sort_keys=True, separators=(",", ":")))

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))

def _raw_matrices(train, test, features, cat_cols):
    a_cols, b_cols = [], []
    cat_set = set(cat_cols)
    for column in features:
        if column in cat_set:
            tr = train[column].fillna("__NA__").astype(str)
            te = test[column].fillna("__NA__").astype(str)
            levels = {value: i for i, value in enumerate(sorted(pd.unique(tr)))}
            a_cols.append(tr.map(levels).fillna(-1).to_numpy(np.float32))
            b_cols.append(te.map(levels).fillna(-1).to_numpy(np.float32))
        else:
            a_cols.append(pd.to_numeric(train[column], errors="coerce").to_numpy(np.float32))
            b_cols.append(pd.to_numeric(test[column], errors="coerce").to_numpy(np.float32))
    if not a_cols:
        return np.zeros((len(train), 1), np.float32), np.zeros((len(test), 1), np.float32)
    return np.column_stack(a_cols).astype(np.float32), np.column_stack(b_cols).astype(np.float32)

def _transform_fold(A, B, fit_idx, *other_indices):
    med = np.nanmedian(A[fit_idx], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    full_a = np.where(np.isfinite(A), A, med)
    full_b = np.where(np.isfinite(B), B, med)
    q = QuantileTransformer(
        n_quantiles=min(256, len(fit_idx)),
        output_distribution="normal",
        subsample=None,
        random_state=SEED,
    )
    q.fit(full_a[fit_idx])
    return [q.transform(full_a[index]).astype(np.float32) for index in other_indices] + [q.transform(full_b).astype(np.float32)]

def _make_model(d: int, hidden: int = 96):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(0.12),
        nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(0.10),
        nn.Linear(hidden, 1),
    )

def _fit_one(tx, ty, vx, vy, bx, seed, max_epochs, patience, wall_seconds):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    _seed_everything(seed)
    model = _make_model(tx.shape[1], hidden=96)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty.astype(np.float32))),
        batch_size=min(512, max(64, len(tx))),
        shuffle=True,
    )
    vt, bt = torch.from_numpy(vx), torch.from_numpy(bx)
    best, state, stale = -1.0, None, 0
    started = time.monotonic()
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb).squeeze(1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch < 8 or epoch % 2:
            if time.monotonic() - started > wall_seconds:
                break
            continue
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(vt).squeeze(1)).cpu().numpy()
        score = float(roc_auc_score(vy, vp))
        if score > best + 1e-4:
            best, state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience or time.monotonic() - started > wall_seconds:
            break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        vp = torch.sigmoid(model(vt).squeeze(1)).cpu().numpy()
        bp = torch.sigmoid(model(bt).squeeze(1)).cpu().numpy()
    del optimizer, loader
    return model, vp, bp, float(roc_auc_score(vy, vp))

def probe_auc(A, y, B) -> float:
    fit_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=0.22, stratify=y, random_state=SEED + 17
    )
    tx, vx, bx = _transform_fold(A, B, fit_idx, fit_idx, val_idx)
    model, _, _, score = _fit_one(
        tx, y[fit_idx], vx, y[val_idx], bx,
        seed=SEED + 17, max_epochs=34, patience=5, wall_seconds=38.0,
    )
    del model, tx, vx, bx
    gc.collect()
    return score

def cv_mlp(A, y, B):
    folds = 3 if len(A) > 5000 else min(5, int(np.bincount(y).min()))
    folds = max(2, folds)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED + 301)
    oof = np.zeros(len(A), dtype=float)
    pred = np.zeros(len(B), dtype=float)
    fold_scores = []
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(A, y)):
        tx, vx, bx = _transform_fold(A, B, fit_idx, fit_idx, val_idx)
        model, vp, bp, score = _fit_one(
            tx, y[fit_idx], vx, y[val_idx], bx,
            seed=SEED + 301 + fold, max_epochs=76, patience=9, wall_seconds=68.0,
        )
        oof[val_idx] = vp
        pred += bp / folds
        fold_scores.append(score)
        del model, tx, vx, bx, vp, bp
        gc.collect()
    return oof, pred, fold_scores

def _forward_rank_ensemble(y, sources, max_members=5, min_gain=2e-5):
    oof_ranks = {name: rank_unit(values[0]) for name, values in sources.items()}
    test_ranks = {name: rank_unit(values[1]) for name, values in sources.items()}
    single_scores = {name: float(roc_auc_score(y, pred)) for name, pred in oof_ranks.items()}
    first = max(single_scores, key=single_scores.get)
    selected = [first]
    current = oof_ranks[first].copy()
    current_score = single_scores[first]
    history = [{"members": list(selected), "auc": current_score}]
    while len(selected) < max_members:
        best = None
        for name, pred in oof_ranks.items():
            candidate = (current * len(selected) + pred) / (len(selected) + 1)
            candidate_score = float(roc_auc_score(y, candidate))
            if best is None or candidate_score > best[0]:
                best = (candidate_score, name, candidate)
        if best is None or best[0] < current_score + min_gain:
            break
        current_score, name, current = best
        selected.append(name)
        history.append({"members": list(selected), "auc": current_score})
    test_prediction = np.mean([test_ranks[name] for name in selected], axis=0)
    weights = {name: selected.count(name) / len(selected) for name in sorted(set(selected))}
    return current_score, current, test_prediction, selected, weights, single_scores, history

def main() -> None:
    started = time.time()
    workdir = runtime_workdir()
    try:
        train, test, sample, id_col, target_col, features, y = load_task(workdir)
        cat_cols = categorical_columns(train, features)
        numeric_ratio = (len(features) - len(cat_cols)) / max(len(features), 1)
        profile = {
            "rows": len(train), "features": len(features),
            "categorical": len(cat_cols), "numeric_ratio": numeric_ratio,
        }
        eligible = (
            1500 <= len(train) <= 18000
            and 8 <= len(features) <= 80
            and numeric_ratio >= 0.85
            and int(np.bincount(y).min()) >= 80
        )
        if not eligible:
            _emit({"candidate": None, "accepted": False, "reason": "schema_gate", "profile": profile})
            return

        import torch
        A, B = _raw_matrices(train, test, features, cat_cols)
        probe = probe_auc(A, y, B)
        if not np.isfinite(probe) or probe < 0.885:
            _emit({"candidate": None, "accepted": False, "reason": "probe_gate", "probe_auc": probe, "profile": profile, "seconds": round(time.time()-started, 3)})
            return

        oof, mlp_test, fold_scores = cv_mlp(A, y, B)
        cv_auc = float(roc_auc_score(y, oof))
        fold_std = float(np.std(fold_scores))
        accepted = (
            np.isfinite(oof).all()
            and np.isfinite(mlp_test).all()
            and np.std(oof) > 1e-9
            and cv_auc >= 0.90
            and fold_std <= 0.04
        )
        if not accepted:
            _emit({"candidate": None, "accepted": False, "reason": "oof_gate", "probe_auc": probe, "cv_auc": cv_auc, "fold_auc": fold_scores, "fold_std": fold_std, "profile": profile, "seconds": round(time.time()-started, 3)})
            return

        sources = {"mlp": (oof, mlp_test)}
        source_details = {}
        state_path = workdir / "portfolio_core_state.npz"
        if state_path.is_file():
            try:
                state = np.load(state_path)
                if "catboost_oof" in state and "catboost_test" in state:
                    coof = np.asarray(state["catboost_oof"], dtype=float)
                    ctest = np.asarray(state["catboost_test"], dtype=float)
                    if len(coof) == len(y) and len(ctest) == len(test) and np.isfinite(coof).all() and np.isfinite(ctest).all():
                        sources["catboost"] = (coof, ctest)
            except Exception as exc:
                source_details["core_state_error"] = f"{type(exc).__name__}:{exc}"

        ensemble_auc, final_oof, final_test, members, weights, single_scores, history = _forward_rank_ensemble(
            y, sources, max_members=5, min_gain=2e-5
        )
        if ensemble_auc < cv_auc + 0.0003:
            final_oof = rank_unit(oof)
            final_test = rank_unit(mlp_test)
            members = ["mlp"]
            weights = {"mlp": 1.0}
            ensemble_auc = cv_auc
        kind = "numeric_oof_rank_ensemble" if len(members) > 1 else "numeric_mlp"

        path = write_submission(
            workdir / "portfolio_numeric_neural_challenger.csv",
            final_test, test, sample, id_col, target_col,
        )
        try:
            np.savez_compressed(
                workdir / "neural_challenger_state.npz",
                neural_oof=np.asarray(final_oof, dtype=np.float32),
                neural_test=np.asarray(final_test, dtype=np.float32),
            )
        except Exception as exc:
            source_details["state_error"] = f"{type(exc).__name__}:{exc}"
        record_candidates(workdir, "neural", [{"path": path, "kind": kind, "oof_auc": ensemble_auc}])
        _emit({
            "candidate": path, "accepted": True, "kind": kind,
            "probe_auc": probe, "cv_auc": cv_auc, "ensemble_auc": ensemble_auc,
            "members": members, "weights": weights, "single_scores": single_scores,
            "ensemble_history": history, "source_details": source_details,
            "fold_auc": fold_scores, "fold_std": fold_std, "profile": profile,
            "seconds": round(time.time()-started, 3),
        })
    except Exception as exc:
        _emit({"candidate": None, "accepted": False, "reason": "error", "error": f"{type(exc).__name__}:{exc}", "seconds": round(time.time()-started, 3)})

if __name__ == "__main__":
    main()
