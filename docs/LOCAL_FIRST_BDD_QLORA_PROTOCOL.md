# Local-First BDD Adapter and 3B QLoRA Capacity Gate

## Scope

This v0.6.0 addition is designed for the validated local RTX 4090 Laptop GPU environment. It adds two deliberately separate operations:

1. `bdd-direct-audit` / `bdd-direct-build`: consume an already extracted BDD layout without archive-mode extraction.
2. `local-qlora-feasibility`: execute a single-image 4-bit NF4 QLoRA forward/backward and write peak VRAM evidence.

Neither operation is a frozen-final model-performance result.

## Required extracted BDD layout

```text
<BDD_ROOT>/Images/train
<BDD_ROOT>/Images/val
<BDD_ROOT>/Labels/train
<BDD_ROOT>/Labels/val
```

The direct adapter reads BDD train only for `train`, `dev`, and `calibration`. BDD val is assigned only to `final` and never enters the development splitter.

## Safe local sequence

First, run an audit. It is read-only and validates image-label joins, duplicate annotation names, missing image files, and train/val exact filename overlap.

```powershell
$RepoRoot = "C:\Users\bjw-0\Downloads\Apple01_visualact-r"
$BddRoot = "C:\Users\bjw-0\Downloads\Project_Data\bdd100k"
$OutputRoot = "$HOME\ML_Outputs\Apple01_visualact-r\bdd-direct-audit"

& "$RepoRoot\.venv\Scripts\python.exe" -m visualact_r bdd-direct-audit `
  --bdd-root $BddRoot `
  --output "$OutputRoot\bdd_direct_audit.json"
```

Only if this emits `PASS`, build the direct records. The full build opens every image to calculate a pHash, so it may be compute-intensive. It does not train a model.

```powershell
$BuildRoot = "$HOME\ML_Outputs\Apple01_visualact-r\bdd-direct-full"

& "$RepoRoot\.venv\Scripts\python.exe" -m visualact_r bdd-direct-build `
  --bdd-root $BddRoot `
  --output $BuildRoot
```

The command blocks by default if a pHash near-duplicate crosses BDD train-derived development and BDD val frozen final. Do not use `--allow-cross-partition-phash` unless the exception and its effect on final claims are documented.

For a non-claim development sample only, use `--max-train-records 5000`; BDD val remains complete but the resulting run is explicitly marked as a development sample.

## 3B VLM feasibility gate

The default model is `Qwen/Qwen2.5-VL-3B-Instruct`, with 4-bit NF4, BF16 compute, LoRA rank 16, one image, and a `262144` maximum image pixel budget. The command downloads the official model on first use, uses one supplied BDD **train** image, runs model construction plus one forward/backward, then writes a JSON report containing peak allocated and reserved VRAM.

```powershell
$PilotRoot = "$HOME\ML_Outputs\Apple01_visualact-r\qlora-feasibility"
$PilotImage = "$BddRoot\Images\train\0000f77c-6257be58.jpg"  # replace with an existing train image

& "$RepoRoot\.venv\Scripts\python.exe" -m visualact_r local-qlora-feasibility `
  --image $PilotImage `
  --output $PilotRoot
```

`PASS` means the one-image forward/backward completed and peak reserved VRAM was within 95% of device capacity. `HOLD_MEMORY_HEADROOM` means it ran but the headroom policy was not met. `OOM_HOLD` and `ERROR_HOLD` preserve an actionable JSON report and exit nonzero.

## Claim boundary

This protocol establishes local data-adapter and memory feasibility only. It does not justify an accuracy, safety, or production claim. Those require train-only fitting, dev and calibration selection, then one sealed final comparison on BDD val.
