# VisualAct-R v0.6.5 Final Evidence Summary

## Decision

- Stop further model training for this branch.
- Use NMS / constant fallback downstream policy.
- Do not claim CEF improvement because the CEF development gate is HOLD and CefEnabled is false.

## Detector Cache

- Status: PASS
- Record count: 15000
- Raw candidates: 9177614
- NMS candidates: 2154855
- Detector retry count: 1
- Detector miss count: 0
- Post-NMS fallback count: 0

## CEF Stage

- Status: HOLD
- Stage execution status: PASS
- Development gate status: HOLD
- Gate reason: development_only_coverage_contamination_gate
- CefEnabled: False
- Train records: 3611
- Dev records: 833
- Usable records: 15000
- Detector miss excluded count: 0
- Ranker mode: constant_fallback
- Raw ranker mode: constant_fallback

## Downstream Evaluation

| Candidate | Meaning | TraceCount | GroundedActionSuccess | CriticalEvidenceRecall | EvidencePrecision | EvidenceF1 |
|---|---|---:|---:|---:|---:|---:|
| C3 | nms_baseline | 15000 | 0.0155 | 1.0 | 0.0 | 0.0 |
| C4 | raw_ranker_constant_fallback | 15000 | 0.0 | 1.0 | 0.0001 | 0.0001 |
| C5 | cef_constant_fallback_gate_hold | 15000 | 0.0149 | 1.0 | 0.0 | 0.0 |

## Claim Boundary

Allowed:
- Full 15,000-record detector cache validation.
- Zero detector misses after retry policy.
- 9.18M raw candidates and 2.15M NMS candidates, based on detector report.
- CEF stage executed and generated artifacts, but development gate held.
- Downstream C3/C4/C5 rollout and mock typed-agent evaluation completed.

Not allowed:
- Do not claim CEF improved over NMS.
- Do not claim Qwen/VLM fine-tuning.
- Do not claim production autonomous-driving validation.
- Do not claim high evidence precision or F1.
