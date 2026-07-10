# VisualAct-R v0.6.2: Direct-BDD Build HOLD Manifest Fix

## Why this patch exists

`bdd-direct-build` performs an expensive pHash scan before task-record publication. In v0.6.1, a cross-partition near-duplicate could raise before `direct_bdd_split_manifest.json` was written, leaving a long-running build with no inspectable report.

## Change

- The build now creates the output directory before the pHash scan result is evaluated.
- If source-boundary integrity or cross-partition pHash checks fail, it writes `direct_bdd_split_manifest.json` with `status: HOLD`, full audit payload, pHash examples, and an explanation.
- It does not write `task_records.jsonl` or `inference_records.jsonl` when the gate holds.
- Exit code remains non-zero because the CLI reports `HOLD`; the failure is now evidence-bearing rather than opaque.

## Scope

This is a reliability patch for the direct adapter. It does not relax the train-only development / val-only frozen-final boundary, does not enable `--allow-cross-partition-phash`, and does not initiate model training.

## Validation performed

- Python compileall
- Targeted tests: `test_bdd_direct.py`, `test_qwen_contract.py`, `test_local_qlora_feasibility.py`, `test_versioning.py` (18 passed)
- Ruff check and format check
- Mypy for 37 source modules using the project-compatible import-untyped allowance in the validation container
- wheel and sdist build
