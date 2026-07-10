# VisualAct-R v0.6.0

## v0.6.0 local-first BDD and QLoRA gate

- Adds a direct adapter for the already-extracted `Images/{train,val}` and `Labels/{train,val}` BDD layout.
- Enforces `BDD train → train/dev/calibration` and `BDD val → frozen final` without archive-mode extraction.
- Adds an explicit cross-partition pHash audit and a one-image 4-bit NF4 QLoRA VRAM capacity gate for local 16GB development.
- See [the local-first protocol](docs/LOCAL_FIRST_BDD_QLORA_PROTOCOL.md).


## v0.5.3 Windows filename portability patch

SFT crop artifact filenames now sanitize both image and crop provenance identifiers for Windows-compatible output paths. The patch preserves deterministic identity with stable suffixes, rejects unsupported output suffixes in favor of PNG, and adds a regression test for colon-delimited crop identifiers.

**VisualAct-R** is an offline, BDD100K-derived urban-scene review research framework. It evaluates whether a visual assessment can be `AUTO_CLEAR`ed, should go to `REVIEW`, or must `ESCALATE` to human review. It is **not** a vehicle-control system, collision predictor, safety certification, official BDD100K leaderboard submission, or production deployment.

## What is being tested

The main comparison is deliberately narrow and falsifiable:

```text
same local VLM family + same sealed final image IDs + same shared full-scene-only adapter
C3: conventional NMS evidence crops
C4: learned raw-candidate selector
C5: ABC-derived Critical Evidence Field crops
```

C5 applies confidence-weighted asymmetric candidate expansion, learned directional correction, and diverse Top-K crop selection. C6 adds the PHT-inspired **Reference-Conditioned Trajectory Reliability** review gate, compared only against a matched-capacity C5 confidence-routing control.

## Candidate hierarchy

| ID | Candidate | Purpose |
|---|---|---|
| C0 | Detector-zone policy | Sanity baseline |
| C1 | Calibrated detector triage | Strong operational baseline |
| C2 | Full-scene Qwen3-VL QLoRA | Shared training adapter |
| C3 | Shared Qwen adapter + NMS crops | Conventional evidence baseline |
| C4 | Shared Qwen adapter + learned raw selector | Selector-only ablation |
| C5 | Shared Qwen adapter + ABC-derived CEF | Primary specialty candidate |
| C5_CONFIDENCE_GUARD | C5 + confidence-only review routing | Matched-capacity RCTR control |
| C6 | C5 + RCTR routing | Reliability-oriented candidate |

## Repository boundary and cleanup policy

The repository contains source, tests, configuration templates bundled as package resources, and documentation only. It does **not** contain BDD media, extracted cache data, trained model weights, checkpoints, run artifacts, or stale static package manifests.

- Editable YAML is created explicitly with `write-config`; there is no second source-tree `configs/` copy that can drift from package defaults.
- BDD extraction uses a staging directory, validates the image/label join, then publishes the cache. A non-empty target cache is rejected unless `--overwrite` is explicit.
- Local verification stages a clean source copy under its output directory, so editable installs, build artifacts, and caches do not pollute the working tree.

## CPU-only local validation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"

# Fast synthetic end-to-end contract path
visualact-r fixture-run --output runs\fixture_smoke
visualact-r qualify --run-dir runs\fixture_smoke

# Full clean staged-source verification: source + wheel + sdist
python scripts\verify_local.py --output runs\local_verify
```

The synthetic path validates schemas, split integrity, CEF gating/fallback, artifact lineage, report generation, fail-closed trace handling, and capacity-constrained RCTR routing. It is **not** VLM-performance evidence.

## Operational error behavior

Expected filesystem, configuration, provenance, and validation errors return exit code `2` with a compact
`ERROR [command]: ...` message. Set `VISUALACT_R_DEBUG=1` to re-raise the underlying exception with a traceback
when diagnosing a local implementation issue.

## CI and installed-package isolation

The clean verifier disables third-party pytest plugin auto-loading and pins small-fixture numerical
thread pools to one worker. Wheel and sdist smoke tests also prove that `visualact_r` is imported
from the installed distribution rather than the checkout `src/` tree. See
[`docs/CI_RUNTIME_HARDENING.md`](docs/CI_RUNTIME_HARDENING.md).

## Configuration without duplicate files

The packaged defaults are canonical. Materialize an editable override before changing QLoRA settings:

```powershell
visualact-r write-config --template qwen_qlora --output configs\qwen_qlora.local.yaml
# Review and edit configs\qwen_qlora.local.yaml, then pass it explicitly.
```

The command refuses to overwrite an existing file unless `--force` is supplied.

## Archive-mode contract workflow

> **Current local-data guard:** The commands below validate the archive-mode adapter only. They are **not** the approved real-data path for the currently extracted `BDD train` / `BDD val` folders. Do not run `extract-bdd`, `build-data`, CEF training, QLoRA training, or final comparison on the current BDD folders until the direct train/val adapter is added. The required boundary remains `BDD train -> train/dev/calibration` and `BDD val -> untouched frozen final`.

### 1. Verify user-supplied archive assets

```powershell
visualact-r preflight `
  --data-root "D:\bdd_assets" `
  --labels-zip "bdd100k_labels_images_val.zip" `
  --shard "BDD_first4000.7z" `
  --shard "BDD_second3000.7z" `
  --shard "BDD_third3000.7z" `
  --output runs\bdd_preflight

visualact-r extract-bdd `
  --data-root "D:\bdd_assets" `
  --labels-zip "bdd100k_labels_images_val.zip" `
  --shard "BDD_first4000.7z" `
  --shard "BDD_second3000.7z" `
  --shard "BDD_third3000.7z" `
  --output "D:\visualact_external_cache"
```

The extraction target must be new or empty. Use `--overwrite` only after checking that replacing the cache is intended.

### 2. Archive-mode commands reserved for synthetic/adapter-contract development

```powershell
visualact-r build-data `
  --annotations "D:\visualact_external_cache\labels\bdd100k_labels_images_val.json" `
  --images-dir "D:\visualact_external_cache\images" `
  --output runs\bdd_data

visualact-r cache-detector `
  --records runs\bdd_data\task_records.jsonl `
  --backend ultralytics `
  --weights "D:\models\pinned_detector.pt" `
  --output runs\detector

visualact-r train-cef `
  --records runs\bdd_data\task_records.jsonl `
  --raw-candidates runs\detector\detector_raw_candidates.jsonl `
  --nms-candidates runs\detector\detector_nms_candidates.jsonl `
  --output runs\cef
```

`train-cef` uses train/dev only. The CEF development gate and final evaluation both use exact union-area coverage/contamination semantics, avoiding overlap double-counting.

### 3. GPU commands reserved until the direct train/val adapter is available

```powershell
python -m pip install -e ".[gpu,vision]"
visualact-r write-config --template qwen_qlora --output configs\qwen_qlora.local.yaml

# Train exactly one C2 full-scene-only adapter.
visualact-r prepare-sft --records runs\bdd_data\task_records.jsonl --candidate-id C2 --split train --output runs\sft_shared_train
visualact-r prepare-sft --records runs\bdd_data\task_records.jsonl --candidate-id C2 --split dev --output runs\sft_shared_dev
visualact-r train-vlm --backend qwen --config configs\qwen_qlora.local.yaml `
  --sft-train runs\sft_shared_train\sft_train_C2.jsonl `
  --sft-dev runs\sft_shared_dev\sft_dev_C2.jsonl `
  --output runs\qwen_shared --execute-training

# Label-free serving records only. C3/C4/C5 reuse the same adapter.
visualact-r rollout-qwen --records runs\bdd_data\inference_records.jsonl --crops runs\cef\crop_records_C3.jsonl --candidate-id C3 --split final --adapter-dir runs\qwen_shared\adapter --output runs\rollout_c3_final
visualact-r rollout-qwen --records runs\bdd_data\inference_records.jsonl --crops runs\cef\crop_records_C4.jsonl --candidate-id C4 --split final --adapter-dir runs\qwen_shared\adapter --output runs\rollout_c4_final
visualact-r rollout-qwen --records runs\bdd_data\inference_records.jsonl --crops runs\cef\crop_records_C5.jsonl --candidate-id C5 --split final --adapter-dir runs\qwen_shared\adapter --output runs\rollout_c5_final
```

### 4. Final evaluation reserved until the direct train/val adapter is available

Use C5 **train** label-free rollout traces to fit RCTR. Then route only C5 final traces at fixed capacity and run `compare-final` with C3, C4, C5, C5 confidence guard, and C6. The detailed command signatures are available through `visualact-r --help`; the data and evaluation contracts are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/EVALUATION_CARD.md`](docs/EVALUATION_CARD.md), and [`docs/RELEASE_CARD.md`](docs/RELEASE_CARD.md).

## Claim boundary

| Evidence state | Permitted wording |
|---|---|
| Source, tests, fixture artifacts, wheel/sdist smoke | implemented; CPU contract validated |
| Real GPU training manifest | locally executed QLoRA training |
| Sealed BDD-derived comparison plus release gates | BDD100K-derived internal research evaluation |
| C5 gate passes with paired CI | improved the named metric versus C3 on the named internal frozen split |
| C6 matched-capacity gate passes | reduced false auto-clear at matched review capacity versus the named confidence control |

Never claim official BDD100K leaderboard performance, autonomous-driving deployment, safety certification, Apple data, Apple product validation, or causal grounding when counterfactual crop-removal execution is marked `NOT_EXECUTED`.

## Documents

- [Architecture and pipeline contract](docs/ARCHITECTURE.md)
- [Local runbook](docs/LOCAL_RUNBOOK.md)
- [Data card](docs/DATA_CARD.md)
- [Model card](docs/MODEL_CARD.md)
- [Evaluation card](docs/EVALUATION_CARD.md)
- [Release card](docs/RELEASE_CARD.md)
- [Failure taxonomy](docs/FAILURE_TAXONOMY.md)
- [Claim boundary](docs/CLAIM_BOUNDARY.md)
- [v0.5.2 validation evidence](docs/VALIDATION.md)
- [v0.5.2 pre-execution validation](docs/PREEXECUTION_VALIDATION.md)
