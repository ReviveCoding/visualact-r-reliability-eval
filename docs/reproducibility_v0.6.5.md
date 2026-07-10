# VisualAct-R v0.6.5 Reproducibility Notes

## Scope

This repository contains source code, tests, lightweight evidence summaries, and documentation for a reliability-gated visual action evaluation pipeline.

It intentionally does not include BDD100K images, labels, detector caches, crop records, agent traces, model weights, or full ML output directories.

## Final validated local artifacts

- Direct BDD build root: C:\Users\bjw-0\ML_Outputs\Apple01_visualact-r\bdd-direct-dev5000-v0.6.3-phash3-cuda-20260708T165619
- Detector cache root: C:\Users\bjw-0\ML_Outputs\Apple01_visualact-r\detector-cache-full-v0.6.5-yolo11n-conf0001-img960-retry1280-no-postnms-20260709T132420\detector-cache
- CEF root: C:\Users\bjw-0\ML_Outputs\Apple01_visualact-r\cef-full-v0.6.5-yolo11n-conf0001-img960-retry1280-no-postnms-20260709T174146
- Downstream eval root: C:\Users\bjw-0\ML_Outputs\Apple01_visualact-r\downstream-eval-v0.6.5-nms-constant-fallback-20260709T204225
- Final evidence root: C:\Users\bjw-0\ML_Outputs\Apple01_visualact-r\final-evidence-v0.6.5-20260709T211650

## Validated evidence summary

- Detector records: 15,000
- Raw candidates: 9,177,614
- NMS candidates: 2,154,855
- Detector misses: 0
- Detector retries: 1
- Post-NMS fallback count: 0
- CEF stage execution: PASS
- CEF development gate: HOLD
- CEF enabled: false
- Downstream C3/C4/C5 rollout and evaluation: PASS

## Claim boundary

Allowed claims:

- Full 15,000-record detector-cache validation.
- Zero detector misses after retry policy.
- CEF stage executed and generated artifacts, but its development gate held.
- Downstream C3/C4/C5 mock typed-agent evaluation completed.

Disallowed claims:

- Do not claim CEF improved over NMS.
- Do not claim high evidence precision/F1.
- Do not claim Qwen/VLM fine-tuning.
- Do not claim production autonomous-driving validation.