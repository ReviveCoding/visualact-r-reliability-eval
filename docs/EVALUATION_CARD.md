# Evaluation Card

## Primary metric

**Grounded Action Success (GAS)** means the final action matches the deterministic offline triage label. For `REVIEW` and `ESCALATE`, selected evidence must also meet both Critical Evidence Recall@K ≥ 0.80 and evidence contamination ≤ 0.70.

## C5 evidence gate

C5 must exceed C3 by at least +3pp on frozen-hard GAS, improve Critical Evidence Recall by at least +3pp, remain within +2pp contamination, avoid any false-auto-clear deterioration, and retain clean GAS within -1pp. The C5-minus-C3 hard-GAS paired bootstrap lower 95% bound must be positive and the configured matched-final sample minimum must be reached.

## C6 routing gate

C6 is compared only with `C5_CONFIDENCE_GUARD` at the same review capacity. It must reduce false auto-clear by at least 10% relative, increase unsafe capture by at least +5pp, preserve clean Macro-F1 within -1pp, and retain complete trace/schema coverage. If the confidence guard leaves no residual false-auto-clear events, the project records `EVIDENCE_UPGRADE_C5_ONLY`; it does not infer RCTR superiority.

## Fail-closed behavior

Missing model traces become `REVIEW` at evaluation, count against trace coverage, and prevent a release claim. A malformed Qwen response becomes a typed `ROUTE_REVIEW` trace with `schema_valid=false`.

## Counterfactual evidence

Real crop-removal analysis requires repeated model inference and is not fabricated from stored traces. If not executed, the report writes `NOT_EXECUTED`; no causal-grounding wording is permitted.

## Scope limitation

Synthetic fixture metrics validate contracts only. They are not BDD100K-derived model-performance, GPU, production, or safety evidence.
