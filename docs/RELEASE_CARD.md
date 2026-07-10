# Release Card

## Current state

**v0.5.2 state:** implemented repository; CPU synthetic fixture validation; source fixture plus wheel/sdist installed-package smoke paths; package-safe configuration templates; external BDD source-archive join preflight; sealed final-comparison workflow; and controlled RCTR routing contract. It is **not** GPU-validated, BDD-derived performance-validated, hosted-CI-validated, or release-qualified for a model-improvement claim.

## Required before performance wording

1. Extract external BDD data to a new or explicitly replaced cache and retain its source manifest.
2. Build DCT-pHash-safe internal splits from actual extracted pixels.
3. Validate raw pre-NMS extraction with the pinned detector checkpoint.
4. Pass the train/dev CEF development gate using the exact union-area metric.
5. Train one shared C2 full-scene-only adapter from train/dev data and save the GPU manifest.
6. Run label-free C3/C4/C5 frozen-final rollouts without re-tuning.
7. Fit RCTR only from train C5 traces, then generate C5 confidence and C6 routes at matched capacity.
8. Execute `compare-final`, inspect paired-bootstrap/sample gates, run counterfactual crop deletion, and use the generated release decision.

## Residual risks

Qwen model/processor compatibility, 4-bit memory use, raw-candidate extraction compatibility, small hard-slice sizes, and crop-removal counterfactual behavior remain empirical questions until the local GPU and BDD-derived run completes.
