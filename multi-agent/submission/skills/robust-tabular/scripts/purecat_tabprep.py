"""Fold-safe TabPrep-style specialist for sufficiently large all-categorical tables."""
from __future__ import annotations
import gc, math, time
from itertools import combinations
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
SEED = 20260721

def _select_columns(fit: pd.DataFrame, features: list[str], y: np.ndarray, max_cols: int = 7) -> list[str]:
    if len(features) <= max_cols:
        return list(features)
    prior = float(np.mean(y))
    scored = []
    for column in features:
        values = fit[column].fillna("__MISSING__").astype(str)
        stats = pd.DataFrame({"key": values.to_numpy(), "y": y}).groupby("key", sort=False).y.agg(["mean", "size"])
        between = float(np.average((stats["mean"] - prior) ** 2, weights=stats["size"]))
        scored.append((between / math.log1p(max(len(stats), 1)), column))
    return [column for _, column in sorted(scored, reverse=True)[:max_cols]]

def _keys(frame: pd.DataFrame, selected: list[str]) -> dict[str, pd.Series]:
    base = {column: frame[column].fillna("__MISSING__").astype(str) for column in selected}
    result = dict(base)
    for left, right in combinations(selected, 2):
        result[f"{left}__X__{right}"] = base[left].str.cat(base[right], sep="|")
    return result

def _mapping(values: pd.Series, y: np.ndarray, alpha: float = 10.0):
    prior = float(np.mean(y))
    stats = pd.DataFrame({"key": values.to_numpy(), "y": y}).groupby("key", sort=False).y.agg(["mean", "size"])
    encoded = ((stats["mean"] * stats["size"] + prior * alpha) / (stats["size"] + alpha)).to_dict()
    return encoded, prior

def _encode(fit_keys: dict[str, pd.Series], valid_keys: dict[str, pd.Series], test_keys: dict[str, pd.Series], y_fit: np.ndarray):
    fit_columns: dict[str, np.ndarray] = {}; valid_columns: dict[str, np.ndarray] = {}; test_columns: dict[str, np.ndarray] = {}
    minority = int(np.bincount(y_fit).min())
    inner = StratifiedKFold(n_splits=max(2, min(4, minority)), shuffle=True, random_state=SEED + 991)
    for index, name in enumerate(fit_keys):
        fit_values = fit_keys[name].reset_index(drop=True)
        valid_values = valid_keys[name].reset_index(drop=True)
        test_values = test_keys[name].reset_index(drop=True)
        frequencies = fit_values.value_counts(dropna=False, normalize=True).to_dict()
        fit_columns[f"frequency_{index}"] = fit_values.map(frequencies).fillna(0.0).to_numpy(dtype=np.float32)
        valid_columns[f"frequency_{index}"] = valid_values.map(frequencies).fillna(0.0).to_numpy(dtype=np.float32)
        test_columns[f"frequency_{index}"] = test_values.map(frequencies).fillna(0.0).to_numpy(dtype=np.float32)
        oof = np.zeros(len(fit_values), dtype=np.float32)
        for train_index, holdout_index in inner.split(np.zeros(len(y_fit)), y_fit):
            mapping, prior = _mapping(fit_values.iloc[train_index], y_fit[train_index])
            oof[holdout_index] = fit_values.iloc[holdout_index].map(mapping).fillna(prior).to_numpy(dtype=np.float32)
        mapping, prior = _mapping(fit_values, y_fit)
        fit_columns[f"target_mean_{index}"] = oof
        valid_columns[f"target_mean_{index}"] = valid_values.map(mapping).fillna(prior).to_numpy(dtype=np.float32)
        test_columns[f"target_mean_{index}"] = test_values.map(mapping).fillna(prior).to_numpy(dtype=np.float32)
    return pd.DataFrame(fit_columns, dtype=np.float32), pd.DataFrame(valid_columns, dtype=np.float32), pd.DataFrame(test_columns, dtype=np.float32)

def cv_purecat_tabprep(train, y, test, features, splitter):
    import lightgbm as lgb
    oof = np.zeros(len(train), dtype=float)
    prediction = np.zeros(len(test), dtype=float)
    selected_by_fold = []; started = time.time()
    for fold, (fit_index, valid_index) in enumerate(splitter.split(train, y)):
        fit = train.iloc[fit_index][features]; valid = train.iloc[valid_index][features]; test_features = test[features]
        y_fit = y[fit_index]
        selected = _select_columns(fit, features, y_fit, max_cols=7)
        selected_by_fold.append(selected)
        fit_matrix, valid_matrix, test_matrix = _encode(_keys(fit, selected), _keys(valid, selected), _keys(test_features, selected), y_fit)
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=900, learning_rate=0.028, num_leaves=64, min_child_samples=28,
            subsample=0.9923026236907, subsample_freq=1, colsample_bytree=0.5533919718605, reg_alpha=0.914411672958,
            reg_lambda=1.90439560009, extra_trees=True, random_state=SEED + fold, n_jobs=3, verbosity=-1,
        )
        model.fit(fit_matrix, y_fit, eval_set=[(valid_matrix, y[valid_index])], eval_metric="auc", callbacks=[lgb.early_stopping(70, verbose=False), lgb.log_evaluation(0)])
        oof[valid_index] = model.predict_proba(valid_matrix)[:, 1]
        prediction += model.predict_proba(test_matrix)[:, 1] / splitter.n_splits
        del model, fit_matrix, valid_matrix, test_matrix
        gc.collect()
    selected_count = min(len(features), 7)
    return oof, prediction, {"seconds": round(time.time() - started, 3), "selected_columns": selected_by_fold, "encoded_features": 2 * (selected_count + math.comb(selected_count, 2))}
