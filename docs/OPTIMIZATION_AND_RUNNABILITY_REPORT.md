# v0.5.2 Optimization and Runnability Report

## Purpose

v0.5.2 removes avoidable repository drift and makes the CPU contract path easier to reproduce without changing the research target, candidate hierarchy, claim boundary, or real-data protocol.

## Structural optimizations

| Issue | Change | Protection against regression |
|---|---|---|
| Two editable YAML copies could diverge | Removed source-tree `configs/`; package resources are canonical; added `write-config` | Wheel/sdist resource smoke and overwrite-refusal test |
| Image records opened each image twice; DCT basis rebuilt per image | Reused the already-opened image and cached the DCT basis | pHash equivalence test |
| CEF development area could double-count overlap while final evaluation used union semantics | Switched development aggregation to exact rectangle-union semantics | CEF overlap test |
| Cache extraction could leave partial or stale mixed assets | Stage extraction, validate joins, then atomically publish; reject non-empty target unless `--overwrite` | Small archive atomic-publication test |
| The verifier could pollute the checkout | Stage a clean copy and use disposable source/wheel/sdist environments | staged source/wheel/sdist verifier design with provenance checks |
| Runtime version could drift from package metadata | `stages.py` imports `__version__`; test enforces `__version__ == pyproject.toml` | Version-contract test |
| Documentation instructed separate candidate adapters and label-bearing rollout input | Consolidated instructions around one shared C2 adapter and label-free `inference_records.jsonl` | README/runbook review and inference-contract tests |

## Removed or consolidated repository content

- Duplicate `configs/default.yaml` and `configs/qwen_qlora.yaml`
- Stale model/data template documents
- Redundant static root hardening, CI, pipeline, validation, and package-manifest snapshots
- Generated source-tree caches, build directories, `*.egg-info`, and bytecode before release packaging

Run artifacts, external BDD media, model weights, checkpoints, and cache extracts remain excluded from the repository.

## Connected pipeline contract

```text
preflight
-> atomic extract-bdd
-> build-data
-> cache-detector
-> train-cef
-> prepare-sft / train-vlm
-> sealed rollout-qwen
-> train-rctr
-> route-review
-> compare-final
-> qualify
```

The fixture pipeline exercises the same artifact contracts through evaluation, report generation, and qualification. It does not attempt to fabricate GPU or BDD performance.

## CI and local verification

The CI workflow has two Python 3.11 jobs:

- **Ubuntu:** quality checks, fixture qualification, sdist/wheel build, installed-wheel and installed-sdist fixture qualification.
- **Windows:** PowerShell parser validation, quality/fixture checks, sdist/wheel build, installed-wheel and installed-sdist fixture qualification.

Both jobs use least-privilege read-only repository permissions, timeout limits, cancellation of superseded runs, and artifact upload on failure or success. Actual GitHub-hosted execution remains a release prerequisite.

## Remaining non-code gates

The remaining risks require real assets or the user’s RTX/Windows environment rather than more synthetic fixtures:

1. Windows PowerShell execution with the user’s archive paths.
2. Qwen3-VL GPU load and 4-bit QLoRA memory/throughput validation.
3. Pinned detector raw pre-NMS extraction.
4. BDD image extraction and leakage-safe pHash split.
5. Sealed C3/C4/C5/C6 comparison, bootstrap confidence intervals, and counterfactual evidence tests.
