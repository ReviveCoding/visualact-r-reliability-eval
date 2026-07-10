# VisualAct-R v0.6.4 Detector Smoke Robustness Patch

Fixes the Ultralytics detector smoke path discovered on Windows local BDD samples.

Changes:
- cache-detector now always writes detector_cache_report.json on detector failures.
- Adds --confidence, --imgsz, --device, and --allow-post-nms-fallback CLI options.
- Keeps raw pre-NMS extraction as the default strict contract.
- Adds an explicitly reported Ultralytics post-NMS fallback for smoke tests only.
- Records fallback count and image ids in detector_cache_report.json.
- Adds regression tests for failure-report publication and explicit fallback behavior.

Claim boundary:
Post-NMS fallback is for detector smoke / model-load validation only. It should not be used for raw-pre-NMS CEF claims without documenting a claim-boundary exception.
