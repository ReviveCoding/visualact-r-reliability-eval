# CI and Runtime-Provenance Hardening

## Purpose

The CPU fixture path validates package and artifact contracts, not BDD100K or Qwen model performance.
This release hardens that path against two common sources of false confidence:

1. external pytest plugins and unconstrained BLAS thread pools that make small tests noisy or slow;
2. package smoke tests that accidentally import the checkout source tree instead of the wheel or sdist under test.

## Controls

- CI and `scripts/verify_local.py` set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- CI and the local verifier pin OpenBLAS, OpenMP, MKL, and NumExpr to one thread for the small fixture contract.
- Wheel and sdist smoke paths run `visualact-r runtime-provenance --source-root <checkout>`. The command fails if `visualact_r.__file__` resolves under `<checkout>/src`.
- The provenance JSON is retained with the validation artifacts.

## Claim boundary

Passing these checks establishes source/wheel/sdist isolation and deterministic CPU fixture execution. It does not establish GPU compatibility, raw detector-candidate validity, real BDD-derived split quality, or C3/C4/C5/C6 performance.

## Operational errors

The CLI converts expected filesystem, configuration, provenance, and validation exceptions into a concise
message and exit code `2`. `VISUALACT_R_DEBUG=1` preserves the full traceback for local debugging.
