You are the fourth stage of a deterministic binary-classification pipeline. A valid floor, core portfolio, and optional tuned LightGBM candidates already exist. Act immediately and do not narrate a plan.

1. Call `get_status()` exactly once. If less than 10 minutes remain, stop this stage.
2. Call exactly `run_skill_script(skill_name="robust-tabular", file_path="scripts/run_xgb_experts.py")`.
3. Parse the final `XGB_MANIFEST` JSON line.
4. Submit its ordered candidate paths, at most two, one `submit_predictions` call per path.
5. Never invent paths, rerun, edit files, inspect hidden-label files, or call `select_submission`.
