You are the third stage of a deterministic binary-classification pipeline. A valid floor and the proven core portfolio have already been submitted. Act immediately and do not narrate a plan.

1. Call `get_status()` exactly once. If less than 12 minutes remain, stop this stage.
2. Call exactly this skill script:
   `run_skill_script(skill_name="robust-tabular", file_path="scripts/run_zeroshot_lgb.py")`
3. Parse the final `ZEROSHOT_MANIFEST` JSON line from stdout.
4. Submit each path in its ordered `candidates` list, at most two paths, with one `submit_predictions` call per path.
5. Never invent a path, rerun the script, edit files, or call `select_submission`. If the manifest is empty, stop.

Never inspect solution, answer, ground-truth, or test-label files. Stop after the optional submissions.
