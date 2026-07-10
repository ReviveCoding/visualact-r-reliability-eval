# Model Card

## Candidates

- **C0:** Detector-zone sanity policy.
- **C1:** Calibrated logistic detector triage.
- **C2:** Full-scene Qwen3-VL QLoRA shared adapter, local GPU path only.
- **C3:** Shared Qwen adapter supplied with conventional NMS crops.
- **C4:** Shared Qwen adapter supplied with learned raw-candidate selector crops.
- **C5:** Shared Qwen adapter supplied with ABC-derived Critical Evidence Field crops.
- **C5_CONFIDENCE_GUARD:** C5 plus confidence-only fixed-capacity review routing.
- **C6:** C5 plus train-reference-conditioned fixed-capacity review routing.

## Intended use

Offline comparison of evidence policies, typed visual-agent output, evidence quality, and capacity-constrained review routing on a BDD100K-derived internal research protocol.

## Inference composition

For C3/C4/C5, the framework retrieves the candidate policy’s ranked crops and passes the full scene plus that bounded crop set to Qwen. It does not claim an unrestricted learned tool planner.

## Prohibited use

Vehicle control, collision prediction, safety certification, operational deployment, official BDD100K claims, or interpretation of fixture/mock behavior as trained VLM performance.

## Current evidence state

v0.5.2 has source, CPU tests, synthetic end-to-end validation, source/wheel/sdist package paths, a sealed final-comparison command, and Qwen local training/rollout code. It does not yet contain a completed local Qwen GPU run or BDD-derived frozen C3/C5 performance result.
