
# Autonomous ML Agent — Kaggle Autonomous Agent Prediction (Beta)

An LLM agent (Google ADK config) that trains models and submits predictions
inside a sandboxed Docker environment, under a fixed budget: 60 minutes,
$2 in LLM spend, 30 submission calls, per session.

Competition: https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/overview

## Results

| Metric  | Public LB | Private LB |
| ------- | --------- | ---------- |
| AUC-ROC | 0.822     | 0.780      |
| Rank    | 143/570   | 114/570    |

570 teams, 1,803 entrants total.

## Two approaches

- `single-agent/` — one LLM agent handles data analysis, feature
  engineering, model training, and submission end to end.
- `multi-agent-tree/` — a generator-evaluator-critic loop
  (tree_manager → coding_operator → evaluator → reviewer) that iterates
  on model candidates against a persistent state file.

## What actually broke, and what I fixed

- Disallowed tool references in agent config (tools outside the
  harness's allowed set)
- Submission zip exceeding the 50-file cap from a duplicated folder
  structure
- Output filename mismatch between the code-generation step and the
  verification step in the multi-agent version
