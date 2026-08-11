# Data Leakage Prevention Checklist

Leakage = information unavailable at true inference time bleeds into training,
giving optimistic local scores and collapse on the private leaderboard.

## Target leakage
- Never build features from the target (or anything derived from it).
- Drop post-outcome / proxy columns that would not exist at prediction time.

## Train/test contamination
- Fit ALL transformers (imputers, scalers, encoders, target statistics) on the
  TRAIN fold only, then apply to validation/test. Never fit on the full dataset.
- Compute categorical target-encodings inside CV folds, never on full train.

## Split hygiene
- Use stratified K-fold for classification to preserve class balance.
- If rows share an entity/time group, use grouped/temporal splits so the same
  group never appears in both train and validation.

## Sanity checks
- A feature nearly perfectly correlated with the target is almost always leakage.
- Validation score far above a plausible public score → suspect leakage first.
