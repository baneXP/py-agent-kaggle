---
name: feature-engineer
description: >-
  Robust Python script for leakage-safe automated feature generation:
  type inference, train-fitted imputation, and safe row-wise aggregates.
---

# Feature Engineer Skill

Pre-packaged CLI for automated, leakage-safe feature engineering.

## Scripts

### `generate_features.py`
Infers column types, imputes missing values (fit on train), and adds safe
row-wise aggregates (mean / std / NaN-count) computed over genuine numeric
features only (ID-like columns excluded from aggregates).

**Usage** (note the skill name matches this directory — `feature-engineer`):
```python
run_skill_script(
    skill_name="feature-engineer",
    script_name="generate_features.py",
    args="--train train.csv --test test.csv --target target",
)
```
**Arguments**: `--train` (default `train.csv`), `--test` (default `test.csv`),
`--target` (default `target`; read the real name from `target_col.txt`).

**Outputs**: `train_engineered.csv`, `test_engineered.csv`.

## Resources

### `leakage_checklist.md`
```python
load_skill_resource(
    skill_name="feature-engineer",
    resource_name="leakage_checklist.md",
)
```
