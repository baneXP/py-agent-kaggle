---
name: robust-tabular
description: Builds reliable binary-classification submissions with a proven diverse portfolio plus schema-gated categorical, interaction, neural, and temporal specialists.
---

# Robust Tabular Portfolio

Use the supplied scripts without editing them.

## `scripts/quick_baseline.py`
Discovers the current train, test, and sample-submission files under `/work`; infers the ID and binary target; writes a prior fallback first; then attempts a fast CatBoost model. It always leaves `/work/quick_baseline.csv` when the sample schema is available.

## `scripts/run_portfolio.py`
Runs five-fold CatBoost, native-categorical LightGBM, ExtraTrees, regularized logistic regression, and rank blends. On sufficiently large all-categorical tables it also evaluates one nested-OOF frequency/target-encoding LightGBM challenger using selected single-column and pair keys. At most one challenger is submitted alongside seven proven core candidates. The script prints one final `PORTFOLIO_MANIFEST` JSON line; submit only its paths.

## `scripts/run_neural_challenger.py`
Runs only after the proven portfolio. A cheap holdout probe and schema gate prevent wasted runtime. On mostly numeric tasks, it fits a fold-safe quantile-normalized MLP; only OOF AUC at or above 0.90 can produce a candidate. It can combine that model with the persisted CatBoost OOF predictions and a bounded TabPrep representation using a five-member Caruana-style rank ensemble. It emits at most one `SPECIALIST_MANIFEST` candidate, so at least one proven core submission remains protected in final selection.

## `scripts/run_zeroshot_lgb.py`
Runs three CPU zero-shot LightGBM configurations derived from the public AutoGluon CPU portfolio, stores their fold-safe OOF/test predictions, and emits at most two OOF-ranked candidates.


## `scripts/run_temporal_expert.py`
Detects genuine date/time columns conservatively, replaces raw date tokens with deterministic calendar and ordinal features, and trains a bounded LightGBM specialist. It persists the source for cross-fit stacking only when it remains close to the best core OOF model and emits a direct candidate only when a small predeclared rank-blend grid improves OOF AUC beyond the core by a safety margin.

## `scripts/meta_stack.py`
Combines the core and optional zero-shot prediction sources with fully cross-fit regularized logistic, simplex, and Caruana meta-models. It emits at most three candidates and never reads public feedback or hidden labels.

## `scripts/run_xgb_experts.py`
Runs two bounded CPU histogram-XGBoost configurations, stores fold-safe OOF/test predictions, and emits at most two candidates.

## `scripts/run_small_experts.py`
Runs bounded RBF-SVM, histogram boosting, KNN, and numeric QDA experts only on 300-3,500-row datasets. Their prediction state is consumed by the meta stage and is not submitted directly.

## `scripts/run_gam_experts.py`
Writes fold-safe spline-additive and quadratic-logistic prediction sources for the meta stage. These files are not submitted directly.

## `scripts/run_rsfc_expert.py`
Runs a fold-safe random-subset feature-compression specialist only on bounded mixed numeric/categorical schemas. Within every outer fold it selects features and bins numeric columns from the fit rows only, builds single/pair/triple interaction keys, and creates frequency plus nested-OOF target encodings. It emits at most one `RSFC_MANIFEST` candidate and is admitted only when its standalone OOF AUC remains within 0.04 of the strongest core source. Its state is also available to `meta_stack.py`.
