# VisualAct-R v0.6.0 Local-First Validation Record

## Added scope

- Direct adapter for extracted BDD layout: `Images/{train,val}` and `Labels/{train,val}`.
- Hard source boundary: BDD train only feeds train/dev/calibration; BDD val only feeds frozen final.
- Exact image-label join audit, duplicate annotation-name detection, cross-source exact image-name overlap detection, and cross-partition pHash leakage audit.
- Local one-image 4-bit NF4 QLoRA forward/backward feasibility gate using default `Qwen/Qwen2.5-VL-3B-Instruct`.
- Peak allocated and reserved CUDA memory written as JSON evidence.

## Validation executed in the build environment

- `compileall src tests`: PASS.
- `ruff check src tests`: PASS.
- `ruff format --check src tests`: PASS.
- `mypy --python-version 3.11 --no-site-packages src`: PASS across 37 source modules.
- 50 tests passed through module-level invocations, including the four direct-BDD adapter tests, seven Qwen contract tests, and two local-QLoRA configuration tests.
- sdist and wheel build: PASS.
- wheel and sdist content inspection: PASS. Both include `bdd_direct.py`, `feasibility.py`, and `local_qlora_pilot.yaml`.

## Not executed here

- User's actual 70,000/10,000 extracted BDD layout audit.
- User's Windows runtime local patch test.
- Qwen weight download/model load or CUDA one-image QLoRA forward/backward.

Those remain explicit local gates and must not be inferred from source-level validation.
