# Architecture and Pipeline Contract

```text
user-supplied BDD source archives
  -> preflight: archive/image-label join and source fingerprints
  -> extract-bdd: isolated staging extraction -> full join validation -> publish external cache
  -> build-data: deterministic triage labels + cached DCT pHash + complete-link clusters + frozen manifest
  -> cache-detector: fail-closed raw pre-NMS and conventional NMS evidence
  -> train-cef: train-only ABC-style expansion / directional correction / diversity-ranked crops
  -> CEF dev gate: exact union-area coverage/contamination; fallback to C3 if it fails
  -> QLoRA SFT: one C2 full-scene-only shared adapter trained on train/dev only
  -> rollout-qwen: label-free records + full scene + fixed C3/C4/C5 crop policy -> typed final trace
  -> train-rctr: train-only successful prefix reference library
  -> route-review: sealed C5 final traces -> matched-capacity confidence and RCTR routes, no labels read
  -> compare-final: immutable C3/C4/C5/C5-confidence/C6 evidence package
  -> qualify: artifact, split, trace, routing, and claim-boundary checks
```

## Serving boundary

Ground-truth boxes, policy labels, supporting-object IDs, hard-case metadata, and evaluation outcomes are available only for task construction, train/dev supervision, and offline evaluation. They are excluded from detector cache inputs, evidence construction at serving time, Qwen rollout, and RCTR routing.

## Attribution boundary

C3, C4, and C5 reuse the identical sealed C2 adapter. They differ only by the preselected evidence-crop policy. This makes the final comparison attributable to conventional NMS crops, raw-candidate selection, or ABC-derived CEF rather than separate fine-tunes.

## Artifact boundary

Every stage publishes machine-readable artifacts atomically. BDD extraction first writes a same-filesystem staging directory, checks the full image/label join, and only then publishes the cache. A non-empty existing target requires explicit `--overwrite`.

## Release boundary

`compare-final` consumes already-created final traces and crop artifacts. It does not train, tune thresholds, fit calibration, choose references, or regenerate labels. It records input hashes, final-ID fingerprints, split manifest information, and paired-comparison outputs.
