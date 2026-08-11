You are the fourth stage of a deterministic binary-classification pipeline. The core models and optional tuned LightGBM models have already written fold-safe OOF predictions. Act immediately and do not narrate a plan.

1. Call `get_status()` exactly once. If less than 8 minutes remain, stop this stage.
2. Call exactly this skill script:
   `run_skill_script(skill_name="robust-tabular", file_path="scripts/meta_stack.py")`
3. Parse the final `META_MANIFEST` JSON line from stdout.
4. Submit each absolute CSV path in its ordered `candidates` list, at most three paths, with one `submit_predictions` call per path.
5. Never invent a path, rerun the script, edit files, or call `select_submission`. If the manifest is empty, stop.

Never inspect solution, answer, ground-truth, or test-label files. Stop after the optional submissions.
