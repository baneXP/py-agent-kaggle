You are the first stage of a deterministic binary-classification pipeline. Act immediately and do not narrate a plan.

1. Call `get_status()` once.
2. Call exactly this skill script:
   `run_skill_script(skill_name="robust-tabular", file_path="scripts/quick_baseline.py")`
3. The script always attempts to write `/work/quick_baseline.csv`. Call `submit_predictions("/work/quick_baseline.csv")` immediately after the script returns.
4. If the submission tool rejects the file, run the same skill script once more and retry the same absolute path once. Otherwise stop this stage.

Never inspect solution, answer, ground-truth, or test-label files. Never install packages or access the internet. Do not finish without attempting `submit_predictions`.
