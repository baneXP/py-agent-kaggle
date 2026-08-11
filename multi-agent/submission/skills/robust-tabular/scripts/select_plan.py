#!/usr/bin/env python3
"""Plan the final two submissions under multiple-testing control.

Why this exists. A session submits up to ~18 candidates and then keeps two for
private scoring. V23 kept the two with the highest *public* scores. That is an
argmax over many noisy estimates of the same underlying quantity, so the winning
public score is biased upward by roughly

    E[max of K noise draws] ~= sigma_pub * sqrt(2 * ln K)

With K = 16 and a public-split noise level of sigma_pub ~ 0.006 AUC, the
expected inflation is ~0.014 AUC - roughly thirty times the 5e-4 margin we use
to promote a challenger offline. Selecting on raw public feedback therefore
silently spends more private AUC than any modelling change in this pipeline is
likely to recover.

What this script does instead:
  1. Parses the dumped `get_status()` text into (submission_id, public_score).
  2. Joins it, in submission order, with `candidate_ledger.jsonl` so each public
     score is attributed to the model that produced it and to its OOF AUC.
  3. Estimates sigma_pub from the residuals of public ~ OOF across candidates
     and applies James-Stein / empirical-Bayes shrinkage to the public scores.
  4. Blends the shrunken public score with the OOF score, weighting OOF more
     heavily exactly when public noise dominates the between-candidate spread.
  5. Picks the runner-up under a Spearman de-correlation constraint on the
     actual prediction files, so the two retained bets are genuinely different.

Every step degrades gracefully. If the ledger, the CSVs or the status dump are
missing or unparseable, the script falls back to the V23 rule (top two public)
and says so in the manifest, so this stage can never become a new failure mode.
"""
from __future__ import annotations
import json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import LEDGER_NAME, runtime_workdir

FEEDBACK_NAMES = ("public_feedback.txt", "public_feedback.json", "status.txt")
DEFAULT_SIGMA = 0.006      # prior public-AUC noise when it cannot be estimated
MIN_SIGMA = 0.001
SELECT_CORR_CAP = 0.995    # the two kept submissions must differ at least this much
STOPWORDS = {"submission", "submissions", "public", "private", "score", "scores", "leaderboard",
             "status", "success", "successful", "succeeded", "failed", "failure", "error", "id",
             "auc", "roc", "metric", "rank", "value", "result", "results", "complete", "completed",
             "pending", "valid", "invalid", "file", "csv", "the", "and", "with", "of", "is", "at",
             "on", "for", "row", "rows", "predictions", "prediction", "remaining", "used", "budget"}


def emit(payload):
    print("SELECT_MANIFEST=" + json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _walk_json(node, out):
    if isinstance(node, dict):
        keys = {str(k).lower(): k for k in node}
        id_key = next((keys[k] for k in keys if k in ("id", "submission_id", "sub_id")), None)
        score_key = next((keys[k] for k in keys if "score" in k or k in ("public", "auc", "metric")), None)
        if id_key is not None and score_key is not None:
            try:
                out.append((str(node[id_key]), float(node[score_key])))
            except (TypeError, ValueError):
                pass
        for value in node.values():
            _walk_json(value, out)
    elif isinstance(node, list):
        for value in node:
            _walk_json(value, out)
    return out


def parse_feedback(text):
    """Return ordered, de-duplicated (submission_id, public_score) pairs."""
    pairs = []
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if 0 <= i < j:
            try:
                pairs = _walk_json(json.loads(text[i:j + 1]), [])
                break
            except Exception:
                pairs = []
    if not pairs:
        for line in text.splitlines():
            numbers = re.findall(r"[01]?\.\d{3,}", line)
            if not numbers:
                continue
            try:
                score = float(numbers[-1])
            except ValueError:
                continue
            tokens = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]{1,63}", line)
                      if t.lower() not in STOPWORDS and not re.fullmatch(r"\d+", t)]
            if tokens:
                pairs.append((tokens[0], score))
    ordered, seen = [], set()
    for sid, score in pairs:
        if not (0.0 <= score <= 1.0):
            continue
        if sid in seen:
            # Ambiguous identifiers mean the parse is untrustworthy; refuse to
            # guess and let the controller fall back to its own rule.
            return []
        seen.add(sid)
        ordered.append({"id": sid, "public": float(score)})
    return ordered


def read_ledger(work):
    path = work / LEDGER_NAME
    rows = []
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    rows.sort(key=lambda r: r.get("seq", 0))
    return rows


def load_prediction(path):
    try:
        frame = pd.read_csv(path)
        values = pd.to_numeric(frame[frame.columns[-1]], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or np.std(values) < 1e-12:
            return None
        return values
    except Exception:
        return None


def spearman(a, b):
    ra = pd.Series(a).rank(method="average", pct=True).to_numpy()
    rb = pd.Series(b).rank(method="average", pct=True).to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def estimate_sigma(public, oof):
    """Residual sd of the public score around its OOF-predicted level."""
    mask = np.isfinite(public) & np.isfinite(oof)
    if mask.sum() >= 6 and np.std(oof[mask]) > 1e-9:
        slope, intercept = np.polyfit(oof[mask], public[mask], 1)
        residual = public[mask] - (slope * oof[mask] + intercept)
        sigma = float(np.sqrt(np.sum(residual ** 2) / max(int(mask.sum()) - 2, 1)))
        return max(sigma, MIN_SIGMA), "regression"
    if mask.sum() >= 3:
        return max(float(np.std(public[mask])) * 0.7, MIN_SIGMA), "spread_proxy"
    return DEFAULT_SIGMA, "prior"


def expected_max_bias(sigma, k):
    """Gumbel approximation of E[max of k standard normals], scaled by sigma."""
    if k <= 1:
        return 0.0
    a = math.sqrt(2.0 * math.log(k))
    if k <= 2:
        return float(sigma * a)
    return float(sigma * (a - (math.log(math.log(k)) + math.log(4.0 * math.pi)) / (2.0 * a)))


def zscore(values):
    values = np.asarray(values, float)
    sd = float(np.std(values))
    if sd < 1e-12:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / sd


def main():
    work = runtime_workdir()
    diagnostics = {"fallback": None}
    try:
        text = ""
        for name in FEEDBACK_NAMES:
            candidate = work / name
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8", errors="ignore")
                diagnostics["feedback_file"] = str(candidate)
                break
        entries = parse_feedback(text) if text else []
        if not entries:
            emit({"select_ids": [], "reason": "no_parsable_feedback", **diagnostics})
            return

        ledger = read_ledger(work)
        joined = len(ledger) >= len(entries)
        for i, entry in enumerate(entries):
            record = ledger[i] if joined and i < len(ledger) else {}
            entry["stage"] = record.get("stage")
            entry["kind"] = record.get("kind")
            entry["path"] = record.get("path")
            entry["oof_auc"] = record.get("oof_auc")

        public = np.array([e["public"] for e in entries], float)
        oof = np.array([e["oof_auc"] if e["oof_auc"] is not None else np.nan for e in entries], float)
        k = len(entries)
        sigma, sigma_source = estimate_sigma(public, oof)
        bias = expected_max_bias(sigma, k)

        # Empirical-Bayes shrinkage of the public scores toward their common mean.
        tau2 = max(float(np.var(public)) - sigma ** 2, 0.25 * sigma ** 2)
        lam = float(sigma ** 2 / (sigma ** 2 + tau2))
        centre = float(np.mean(public))
        shrunk = centre + (1.0 - lam) * (public - centre)

        usable_oof = int(np.isfinite(oof).sum()) >= max(3, k // 2)
        if usable_oof:
            filled = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            score = (1.0 - lam) * zscore(shrunk) + lam * zscore(filled)
            rule = "shrunk_public_plus_oof"
        else:
            score = zscore(shrunk)
            rule = "shrunk_public_only"
        for entry, sh, sc in zip(entries, shrunk, score):
            entry["shrunk_public"] = float(sh)
            entry["score"] = float(sc)

        order = [int(i) for i in np.argsort(-score)]
        first = order[0]
        chosen = [first]
        predictions = {}
        for idx in order:
            path = entries[idx].get("path")
            if path and Path(path).is_file():
                values = load_prediction(path)
                if values is not None:
                    predictions[idx] = values
        for idx in order[1:]:
            if first in predictions and idx in predictions and len(predictions[idx]) == len(predictions[first]):
                rho = spearman(predictions[first], predictions[idx])
                entries[idx]["corr_to_first"] = rho
                if rho > SELECT_CORR_CAP:
                    continue
            chosen.append(idx)
            break
        if len(chosen) < 2 and len(order) > 1:
            chosen.append(order[1])
            diagnostics["fallback"] = "no_decorrelated_runner_up"

        fields = ("id", "stage", "kind", "public", "shrunk_public", "oof_auc", "score", "corr_to_first")
        emit({
            "select_ids": [entries[i]["id"] for i in chosen],
            "rule": rule,
            "k_candidates": k,
            "sigma_public": sigma,
            "sigma_source": sigma_source,
            "winners_curse_bias": bias,
            "tie_band_low": float(np.max(public) - bias),
            "shrinkage_lambda": lam,
            "joined_ledger": bool(joined),
            "naive_top2": [entries[int(i)]["id"] for i in list(np.argsort(-public))[:2]],
            "selected": [{f: entries[i].get(f) for f in fields} for i in chosen],
            "table": [{f: e.get(f) for f in fields} for e in entries],
            **diagnostics,
        })
    except Exception as exc:
        emit({"select_ids": [], "reason": "error", "error": f"{type(exc).__name__}:{exc}"})


if __name__ == "__main__":
    main()
