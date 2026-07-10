# v0.5.2 Validation Record

This record distinguishes executed CPU/package-contract checks from unexecuted GPU, real-data, Windows-host, and hosted-CI checks. It is not BDD-derived model-performance evidence.

## Executed in this validation environment

**Runtime:** Linux x86_64, CPython 3.13.5. The supported package contract is `>=3.11,<3.14`; static analysis was explicitly targeted to Python 3.11.

| Check | Result |
|---|---|
| `ruff format --check .` | PASS |
| `ruff check .` | PASS |
| `mypy --no-site-packages src --python-version 3.11` | PASS, 35 source modules |
| `pytest` | PASS, 41 tests, executed in two bounded batches with third-party plugin autoload disabled |
| Source `fixture-run -> qualify -> report` | PASS, `FIXTURE_ONLY_NO_PERFORMANCE_CLAIM` |
| `python -m build --sdist --wheel` | PASS |
| Wheel contents | PASS: canonical YAML and runtime-provenance module included; no cache payload |
| sdist contents | PASS: canonical YAML and runtime-provenance module included; no cache payload |
| Installed-wheel smoke | PASS: package provenance outside checkout, fixture, and qualification |
| Installed-sdist smoke | PASS: package provenance outside checkout, fixture, and qualification |

The installed wheel/sdist smoke tests used separate temporary virtual environments. To keep this constrained validation environment independent of network resolution, their runtime dependencies were supplied from its already-provisioned package directory; they prove distribution import isolation and functional fixture execution, not a newly resolved dependency lockfile.

## Pipeline evidence

The synthetic CPU path completed as:

```text
fixture-run
  -> data/split contracts
  -> detector and CEF fixture artifacts
  -> candidate evaluation and RCTR routing artifacts
  -> numeric outputs, figures, and reports
  -> qualify
```

`qualify` returned `PASS`, while the release decision remained `FIXTURE_ONLY_NO_PERFORMANCE_CLAIM` by design.

The full local verifier, `scripts/verify_local.py`, stages a repository copy, creates separate source/wheel/sdist environments, and removes those environments unless `--keep` is supplied. It accepts `VISUALACT_R_PIP_CACHE` for repeated dependency reuse and `VISUALACT_R_PIP_FIND_LINKS` for an already-prepared offline wheelhouse. It also disables third-party pytest plugin autoload, pins small-fixture numerical thread pools, and checks installed-package provenance. The full fresh-dependency verifier should be run on the user's Windows Python 3.11 environment before calling that path locally qualified.

## CI contract status

The workflow statically passed the repository CI-contract test. It defines Ubuntu and Windows Python 3.11 jobs that install the source package, run CPU fixture qualification, build both sdist and wheel, prove installed-package provenance, and install/qualify both distribution formats. The Windows job parses the PowerShell preflight script before dataset-dependent work.

**Not yet executed:** a GitHub-hosted Actions run. Do not claim a hosted-CI green status until the repository is pushed and the workflow completes on GitHub.

## Not validated by this record

- Windows execution on the user's host
- actual PowerShell execution against user paths
- fully fresh dependency resolution on the user's Python 3.11 environment
- Qwen3-VL GPU model loading or 4-bit QLoRA training
- a pinned Ultralytics checkpoint's raw pre-NMS extraction
- image-level BDD extraction and pHash grouping
- real C3/C4/C5 or C5/C6 final comparisons
- counterfactual crop-deletion study
- BDD-derived performance or safety claims
