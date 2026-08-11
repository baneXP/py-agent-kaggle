#!/usr/bin/env python3
"""Leakage-safe automated feature generation.

- Infers numeric/categorical columns.
- Imputes missing values (imputers fit on TRAIN, applied to test).
- Adds safe row-wise aggregates over genuine numeric features only
  (ID-like / near-unique columns are excluded from aggregates but kept in data).
- Cleans infinities. Never touches the target when building features.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def _is_id_like(s: pd.Series) -> bool:
    n = len(s)
    return n > 0 and s.nunique(dropna=True) >= 0.95 * n


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate automated ML features.")
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--target", default="target")
    args = ap.parse_args()

    for p in (args.train, args.test):
        if not os.path.exists(p):
            print(f"Error: file '{p}' not found.")
            sys.exit(1)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    target = None
    if args.target in train.columns:
        target = train[args.target]
        train = train.drop(columns=[args.target])
    else:
        print(f"Warning: target '{args.target}' not in train; continuing without it.")

    # Leakage-safe alignment: keep only columns present in BOTH frames.
    common = [c for c in train.columns if c in test.columns]
    train, test = train[common].copy(), test[common].copy()
    print(f"Aligned: train={train.shape}, test={test.shape}")

    num = train.select_dtypes(include=[np.number]).columns.tolist()
    cat = train.select_dtypes(exclude=[np.number]).columns.tolist()

    if num:  # kill infinities before any statistic
        train[num] = train[num].replace([np.inf, -np.inf], np.nan)
        test[num] = test[num].replace([np.inf, -np.inf], np.nan)

    # Genuine numeric features for aggregates (drop ID-like from the *aggregate*).
    agg = [c for c in num if not _is_id_like(train[c])]

    # Missingness signal — computed BEFORE imputation.
    if agg:
        train["row_nan_count"] = train[agg].isna().sum(axis=1)
        test["row_nan_count"] = test[agg].isna().sum(axis=1)

    if num:
        imp = SimpleImputer(strategy="median")
        train[num] = imp.fit_transform(train[num])
        test[num] = imp.transform(test[num])
    if cat:
        imp = SimpleImputer(strategy="most_frequent")
        train[cat] = imp.fit_transform(train[cat])
        test[cat] = imp.transform(test[cat])

    if agg:
        for df in (train, test):
            df["row_mean"] = df[agg].mean(axis=1)
            df["row_std"] = df[agg].std(axis=1).fillna(0.0)

    if target is not None:
        train[args.target] = target.values

    train.to_csv("train_engineered.csv", index=False)
    test.to_csv("test_engineered.csv", index=False)
    print(f"Saved: train_engineered.csv {train.shape}, test_engineered.csv {test.shape}")


if __name__ == "__main__":
    main()
