# Local Runbook

## CPU contract path

Use Python 3.11 for parity with GitHub CI.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
python scripts\verify_local.py --output runs\local_verify
```

`verify_local.py` creates an isolated staged source copy plus fresh source, wheel, and sdist environments beneath the requested output directory. It does not alter the source checkout with editable-install metadata or generated test artifacts.

## Configuration

```powershell
visualact-r write-config --template default --output configs\default.local.yaml
visualact-r write-config --template qwen_qlora --output configs\qwen_qlora.local.yaml
```

Pass an edited file with `--config`; the bundled resources remain the canonical defaults. The command requires `--force` to replace an existing template.

## Real BDD/GPU path

The archive-mode commands in the root [README](../README.md) are currently a contract path only. They must **not** be used to process the already-extracted user BDD folders because the current `build-data` command accepts one source pool and would not preserve the required frozen boundary.

The direct-data adapter is the next prerequisite:

```text
BDD train -> train / dev / calibration only
BDD val   -> untouched frozen final only
```

After that adapter is present, use `task_records.jsonl` only for offline task construction, SFT preparation, CEF training, RCTR fitting, and final offline evaluation. Use label-free `inference_records.jsonl` for every Qwen rollout. The loader rejects label-bearing records.

## Cleanup

Safe to delete after a completed run:

- `runs/`
- `.venv/`
- `build/`, `dist/`, and `*.egg-info/`
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, and `__pycache__/`

Do not delete external BDD media or model checkpoints unless their manifests and SHA-256 lineage are no longer needed.

For repeated clean-verification runs, set `VISUALACT_R_PIP_CACHE` to a user-owned external pip cache. For an already-prepared offline wheelhouse, set `VISUALACT_R_PIP_FIND_LINKS` to that directory; the verifier then uses that wheelhouse without reaching the package index. The verifier never writes package caches, build directories, or virtual environments into the repository checkout.
