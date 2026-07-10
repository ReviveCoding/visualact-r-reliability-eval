# VisualAct-R v0.6.3 Local pHash CUDA Compare Patch

## Purpose

v0.6.2 correctly blocked task-record publication when a cross-partition pHash gate detected a candidate near-duplicate, but the default build path had two operational issues on the local BDD dev5000 run:

1. Long-running pHash stages did not write a progress manifest until after the expensive stages completed.
2. The pHash comparison backend was CPU-only from the CLI, even though the 5,000 x 10,000 Hamming scan can be vectorized safely on CUDA.

v0.6.3 keeps CPU pHash image generation unchanged and adds a CUDA-only backend for the hash-comparison stage.

## New CLI options

```powershell
python -m visualact_r bdd-direct-build `
  --bdd-root <bdd-root> `
  --output <output-dir> `
  --max-train-records 5000 `
  --cross-partition-phash-threshold 3 `
  --phash-compare-backend cuda `
  --phash-chunk-size 256
```

Allowed backend values:

- `cpu`: existing BK-tree CPU path. This remains the default for deterministic compatibility.
- `cuda`: force PyTorch CUDA tensor comparison. Fails fast if CUDA is unavailable.
- `auto`: try CUDA and fall back to CPU if CUDA is not available.

## Safety contract

The patch does not change the pHash algorithm or split protocol. It changes only the backend used to compare already-computed pHash values.

- BDD train remains the only source for train/dev/calibration.
- BDD val remains the only source for frozen final.
- A HOLD manifest is still written when a cross-boundary pHash match is found.
- Manifest records backend requested, backend used, chunk size, elapsed seconds, and CUDA peak memory when CUDA is used.

## Validation performed before packaging

- `python -m compileall -q src tests`: PASS
- Targeted tests: `tests/test_bdd_direct.py`, `tests/test_qwen_contract.py`, `tests/test_local_qlora_feasibility.py`, `tests/test_versioning.py`: 20 PASS, 1 SKIP in this CPU-only packaging environment
- CLI help includes `--phash-compare-backend` and `--phash-chunk-size`

Ruff, mypy, wheel, and sdist were not run in this packaging container because those tools are not installed here. The user's local Phase A environment previously validated the toolchain; run local targeted tests after installing this patch.
