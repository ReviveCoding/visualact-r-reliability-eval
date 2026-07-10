[CmdletBinding()]
param(
    [string]$RepoRoot,
    [string]$BddRoot = "C:\Users\bjw-0\Downloads\Project_Data\bdd100k",
    [string]$OutputRoot = "C:\Users\bjw-0\ML_Outputs\Apple01_visualact-r",
    [string]$HfCacheRoot = "C:\hf-visualact-r",
    [string]$PipCacheRoot = "C:\pip-cache\visualact-r",
    [switch]$SkipFullPackageVerifier,
    [switch]$ValidateInvocationOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# VisualAct-R Phase A
#
# This script validates the local execution environment before any real BDD training:
# - Windows Python 3.11 virtual environment and CUDA 12.8 PyTorch
# - repository dependency, editable-import provenance, and package consistency
# - NVIDIA runtime, public torchvision CUDA NMS, Qwen imports, and bitsandbytes NF4 CUDA smoke
# - repository PowerShell syntax, compile/Ruff/Mypy/Pytest, synthetic fixture qualification
# - optional clean source/wheel/sdist verifier
# - read-only BDD train/val layout inventory
#
# It intentionally does not download Qwen weights, modify BDD files, extract BDD archives,
# build real-data splits, train models, or evaluate BDD results.

$PhaseAScriptVersion = "0.5.6"

function Require-Path {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description,
        [switch]$Container
    )

    $PathType = if ($Container) { "Container" } else { "Any" }
    if (-not (Test-Path -LiteralPath $Path -PathType $PathType)) {
        throw "$Description not found: $Path"
    }
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Step,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter()][string[]]$Arguments = @()
    )

    Write-Host ""
    Write-Host "+ $FilePath $($Arguments -join ' ')"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "FAILED: $Step (exit code $LASTEXITCODE)"
    }
}

function Write-PhasePython {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )

    $Parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null
    Set-Content -LiteralPath $Path -Value $Content -Encoding UTF8
}

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = $PSScriptRoot
}
if ($RepoRoot.StartsWith("-")) {
    throw "Invalid RepoRoot argument: '$RepoRoot'. The launcher must pass a filesystem path, not a parameter token."
}

$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$BddRoot = [System.IO.Path]::GetFullPath($BddRoot)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$HfCacheRoot = [System.IO.Path]::GetFullPath($HfCacheRoot)
$PipCacheRoot = [System.IO.Path]::GetFullPath($PipCacheRoot)

$VenvRoot = Join-Path $RepoRoot ".venv"
$PythonExe = Join-Path $VenvRoot "Scripts\python.exe"
$VenvMarkerPath = Join-Path $VenvRoot ".visualact-r-phase-a-v0.5.6.json"
$VerifierScript = Join-Path $RepoRoot "scripts\verify_local.py"
$PhaseStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$PhaseRoot = Join-Path $OutputRoot "phase-a-$PhaseStamp"
$PythonScriptRoot = Join-Path $PhaseRoot "scripts"

# 0. Repository and data-layout prerequisites.
Require-Path -Path $RepoRoot -Description "Repository root" -Container
$RequiredPaths = @(
    "pyproject.toml",
    "src",
    "tests",
    "scripts\verify_local.py",
    "scripts\run_local_bdd_preflight.ps1",
    ".github\workflows\ci.yml"
)
foreach ($RelativePath in $RequiredPaths) {
    Require-Path -Path (Join-Path $RepoRoot $RelativePath) -Description "Required repository item"
}

$PyProjectPath = Join-Path $RepoRoot "pyproject.toml"
$PyProjectText = Get-Content -LiteralPath $PyProjectPath -Raw -Encoding UTF8
if ($PyProjectText -notmatch '(?m)^\s*version\s*=\s*"([^"]+)"\s*$') {
    throw "Unable to determine VisualAct-R project version from: $PyProjectPath"
}
$ExpectedProjectVersion = $Matches[1]

Write-Host ""
Write-Host "============================================================"
Write-Host "VisualAct-R Phase A v$PhaseAScriptVersion for project v$ExpectedProjectVersion"
Write-Host "Environment, package, GPU, fixture, and BDD-layout validation"
Write-Host "============================================================"

if ($ValidateInvocationOnly) {
    Write-Host "PHASE_A_PARAMETER_BINDING=PASS"
    Write-Host "MAIN_SCRIPT_REPO_ROOT=$RepoRoot"
    return
}

Require-Path -Path $BddRoot -Description "BDD100K root" -Container
foreach ($RelativePath in @("Images\train", "Images\val", "Labels\train", "Labels\val")) {
    Require-Path -Path (Join-Path $BddRoot $RelativePath) -Description "BDD directory" -Container
}

New-Item -ItemType Directory -Force -Path $OutputRoot, $PhaseRoot, $HfCacheRoot, $PipCacheRoot | Out-Null
Set-Location -LiteralPath $RepoRoot

$env:HF_HOME = $HfCacheRoot
$env:HF_HUB_CACHE = Join-Path $HfCacheRoot "hub"
$env:HF_ASSETS_CACHE = Join-Path $HfCacheRoot "assets"
$env:PIP_CACHE_DIR = $PipCacheRoot
$env:VISUALACT_R_PIP_CACHE = $PipCacheRoot
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PIP_NO_INPUT = "1"
$env:PYTHONUTF8 = "1"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

# 1. Create a clean, project-owned Python 3.11 venv.
# A project .venv is disposable. We never reuse an unmarked or dependency-inconsistent
# environment, because that can carry packages from an unrelated repository into pip check.
Write-Host ""
Write-Host "===== 1. Python 3.11 and clean project virtual environment ====="
if ($null -eq (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Windows Python Launcher 'py' was not found. Install/register Python 3.11, then rerun."
}
Invoke-NativeChecked -Step "Confirm Python 3.11" -FilePath "py" -Arguments @(
    "-3.11", "-c", "import sys; print(sys.version); raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 2)"
)

$RebuildVenv = $false
$VenvRebuildReasons = @()

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    $RebuildVenv = $true
    $VenvRebuildReasons += "missing_python_executable"
}
elseif (-not (Test-Path -LiteralPath $VenvMarkerPath -PathType Leaf)) {
    $RebuildVenv = $true
    $VenvRebuildReasons += "missing_phase_a_marker"
}
else {
    $VenvConfigPath = Join-Path $VenvRoot "pyvenv.cfg"
    if (-not (Test-Path -LiteralPath $VenvConfigPath -PathType Leaf)) {
        $RebuildVenv = $true
        $VenvRebuildReasons += "missing_pyvenv_cfg"
    }
    else {
        $VenvConfig = Get-Content -LiteralPath $VenvConfigPath -Raw
        if ($VenvConfig -match '(?im)^include-system-site-packages\s*=\s*true\s*$') {
            $RebuildVenv = $true
            $VenvRebuildReasons += "system_site_packages_enabled"
        }
    }

    if (-not $RebuildVenv) {
        $ExistingPipCheck = & $PythonExe -m pip check 2>&1
        $ExistingPipCheckExitCode = $LASTEXITCODE
        if ($ExistingPipCheckExitCode -ne 0) {
            $RebuildVenv = $true
            $VenvRebuildReasons += "existing_pip_check_failed"
            $ExistingPipCheck | ForEach-Object { Write-Host "EXISTING_VENV_PIP_CHECK: $_" }
        }
    }
}

if ($RebuildVenv) {
    Write-Host "PHASE_A_CLEAN_VENV_REBUILD=START"
    Write-Host "PHASE_A_CLEAN_VENV_REASON=$($VenvRebuildReasons -join ';')"
    if (Test-Path -LiteralPath $VenvRoot -PathType Container) {
        Remove-Item -LiteralPath $VenvRoot -Recurse -Force -ErrorAction Stop
    }
    Invoke-NativeChecked -Step "Create clean project .venv" -FilePath "py" -Arguments @("-3.11", "-m", "venv", $VenvRoot)
}
else {
    Write-Host "PHASE_A_CLEAN_VENV_REUSE=PASS"
}

Invoke-NativeChecked -Step "Verify .venv Python 3.11" -FilePath $PythonExe -Arguments @(
    "-c", "import sys; print('VENV_PYTHON=' + sys.executable); print('PYTHON_VERSION=' + sys.version); raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 2)"
)

# 2. Package tools and CUDA PyTorch.
Write-Host ""
Write-Host "===== 2. Packaging tools and CUDA 12.8 PyTorch ====="
# Keep setuptools below PyTorch's declared upper bound from the first install onward.
# This prevents pip from emitting a transient dependency-conflict ERROR before the
# CUDA PyTorch installation resolves the environment.
Invoke-NativeChecked -Step "Install compatible packaging tools" -FilePath $PythonExe -Arguments @(
    "-m", "pip", "install", "--upgrade",
    "pip>=24,<27", "setuptools>=70,<82", "wheel>=0.45,<1"
)
$PackagingBoundsCheck = "import setuptools, wheel; from importlib.metadata import version; assert int(setuptools.__version__.split('.')[0]) < 82; print('PIP=' + version('pip')); print('SETUPTOOLS=' + setuptools.__version__); print('WHEEL=' + wheel.__version__)"
Invoke-NativeChecked -Step "Verify compatible packaging-tool bounds" -FilePath $PythonExe -Arguments @(
    "-c", $PackagingBoundsCheck
)
Invoke-NativeChecked -Step "Install CUDA 12.8 PyTorch" -FilePath $PythonExe -Arguments @(
    "-m", "pip", "install", "--upgrade",
    "--index-url", "https://download.pytorch.org/whl/cu128",
    "torch", "torchvision"
)

# 3. Project and optional runtime dependencies. Do not install the gpu extra here because
# PyTorch was deliberately selected from the CUDA 12.8 index in the preceding step.
Write-Host ""
Write-Host "===== 3. VisualAct-R and QLoRA dependencies ====="
Invoke-NativeChecked -Step "Install VisualAct-R dev/vision/parquet extras" -FilePath $PythonExe -Arguments @(
    "-m", "pip", "install", "--upgrade", "--upgrade-strategy", "only-if-needed",
    "-e", "${RepoRoot}[dev,vision,parquet]"
)
Invoke-NativeChecked -Step "Install Qwen/QLoRA dependencies" -FilePath $PythonExe -Arguments @(
    "-m", "pip", "install", "--upgrade", "--upgrade-strategy", "only-if-needed",
    "transformers>=4.57.0,<5", "peft>=0.14", "accelerate>=1.2", "bitsandbytes", "qwen-vl-utils==0.0.14"
)
Invoke-NativeChecked -Step "pip dependency consistency" -FilePath $PythonExe -Arguments @("-m", "pip", "check")

$VenvMarker = [ordered]@{
    schema = "visualact-r-phase-a-environment"
    phase_a_script_version = $PhaseAScriptVersion
    project_version = $ExpectedProjectVersion
    repo_root = $RepoRoot
    python_executable = $PythonExe
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
}
$VenvMarker | ConvertTo-Json | Set-Content -LiteralPath $VenvMarkerPath -Encoding UTF8
Write-Host "PHASE_A_CLEAN_VENV=PASS"
Write-Host "PHASE_A_VENV_MARKER=$VenvMarkerPath"

$DependencySnapshot = & $PythonExe -m pip freeze
if ($LASTEXITCODE -ne 0) {
    throw "FAILED: Capture dependency snapshot (exit code $LASTEXITCODE)"
}
$DependencySnapshot | Set-Content -LiteralPath (Join-Path $PhaseRoot "requirements.installed.txt") -Encoding UTF8

# 4. Editable-import provenance and GPU dependency checks are scripts rather than fragile -c snippets.
$ProvenanceScript = Join-Path $PythonScriptRoot "check_editable_provenance.py"
Write-PhasePython -Path $ProvenanceScript -Content @'
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import visualact_r

repo = Path(sys.argv[1]).resolve()
expected = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
module_file = Path(visualact_r.__file__ or "").resolve()
source_root = (repo / "src").resolve()

print(f"VISUALACT_R_VERSION={visualact_r.__version__}")
print(f"VISUALACT_R_MODULE={module_file}")

if visualact_r.__version__ != expected:
    raise RuntimeError(f"Runtime version {visualact_r.__version__} does not match pyproject version {expected}.")
if source_root not in module_file.parents:
    raise RuntimeError(f"Editable install did not import from {source_root}; imported {module_file}.")

print("VISUALACT_R_EDITABLE_PROVENANCE=PASS")
'@
Invoke-NativeChecked -Step "Editable package provenance" -FilePath $PythonExe -Arguments @($ProvenanceScript, $RepoRoot)

Write-Host ""
Write-Host "===== 4. NVIDIA, torchvision, Qwen, and NF4 CUDA validation ====="
$NvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($null -ne $NvidiaSmi) {
    Invoke-NativeChecked -Step "nvidia-smi" -FilePath $NvidiaSmi.Path -Arguments @(
        "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
    )
}
else {
    Write-Host "WARNING: nvidia-smi is not on PATH. PyTorch CUDA checks below remain authoritative."
}

$GpuSmokeScript = Join-Path $PythonScriptRoot "gpu_smoke.py"
Write-PhasePython -Path $GpuSmokeScript -Content @'
from __future__ import annotations

import importlib.metadata as metadata

import torch
import torchvision
from packaging.version import Version

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Refusing CPU fallback.")
if str(torch.version.cuda) != "12.8":
    raise RuntimeError(f"Expected CUDA 12.8 PyTorch runtime, found {torch.version.cuda!r}.")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("GPU does not report bfloat16 support required by QLoRA configuration.")

properties = torch.cuda.get_device_properties(0)
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
print(f"TORCH={torch.__version__}")
print(f"TORCHVISION={torchvision.__version__}")
print(f"TORCH_CUDA={torch.version.cuda}")
print(f"GPU_NAME={torch.cuda.get_device_name(0)}")
print(f"GPU_COMPUTE_CAPABILITY={torch.cuda.get_device_capability(0)}")
print(f"GPU_TOTAL_GIB={properties.total_memory / 1024**3:.2f}")
print(f"GPU_FREE_GIB={free_bytes / 1024**3:.2f}")
print(f"GPU_RUNTIME_TOTAL_GIB={total_bytes / 1024**3:.2f}")
print("BF16_SUPPORT=PASS")

boxes = torch.tensor(
    [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]],
    device="cuda",
)
scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
kept = torchvision.ops.nms(boxes, scores, 0.5)
torch.cuda.synchronize()
if not kept.is_cuda or kept.numel() != 2:
    raise RuntimeError(f"torchvision.ops.nms CUDA smoke failed: {kept.detach().cpu().tolist()}")
print("TORCHVISION_NMS_CUDA_SMOKE=PASS")

import accelerate
import bitsandbytes
import peft
import qwen_vl_utils
import transformers
import ultralytics
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

if Version(transformers.__version__) < Version("4.57.0"):
    raise RuntimeError(f"Transformers >= 4.57.0 is required; found {transformers.__version__}.")
if not hasattr(qwen_vl_utils, "process_vision_info"):
    raise RuntimeError("qwen_vl_utils.process_vision_info is unavailable.")

print(f"TRANSFORMERS={transformers.__version__}")
print(f"PEFT={peft.__version__}")
print(f"ACCELERATE={accelerate.__version__}")
print(f"BITSANDBYTES={metadata.version('bitsandbytes')}")
print(f"QWEN_VL_UTILS={metadata.version('qwen-vl-utils')}")
print(f"ULTRALYTICS={ultralytics.__version__}")
print("QWEN_TRANSFORMERS_API=PASS")

from bitsandbytes.nn import Linear4bit

layer = Linear4bit(
    8,
    8,
    bias=False,
    compute_dtype=torch.bfloat16,
    compress_statistics=True,
    quant_type="nf4",
).to("cuda")
inputs = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    outputs = layer(inputs)
torch.cuda.synchronize()

if not outputs.is_cuda:
    raise RuntimeError("bitsandbytes NF4 output was not placed on CUDA.")
if tuple(outputs.shape) != (2, 8):
    raise RuntimeError(f"Unexpected NF4 output shape: {tuple(outputs.shape)}")
if not torch.isfinite(outputs).all().item():
    raise RuntimeError("NF4 CUDA output contains non-finite values.")

print("BITSANDBYTES_NF4_CUDA_SMOKE=PASS")

del layer, inputs, outputs, boxes, scores, kept
torch.cuda.empty_cache()
'@
Invoke-NativeChecked -Step "GPU, Qwen, and NF4 CUDA smoke" -FilePath $PythonExe -Arguments @($GpuSmokeScript)

# 5. Parse every repository-local PowerShell script, then test the source and fixture contracts.
Write-Host ""
Write-Host "===== 5. Repository PowerShell, source-quality, and fixture validation ====="
$PowerShellScripts = @(
    Get-ChildItem -LiteralPath $RepoRoot -Recurse -File -Filter "*.ps1" -ErrorAction Stop
)
if ($PowerShellScripts.Count -eq 0) {
    throw "No PowerShell scripts were found beneath repository root: $RepoRoot"
}

$ParseFailures = @()
foreach ($PowerShellScript in $PowerShellScripts) {
    $Tokens = $null
    $ParseErrors = $null
    $Ast = [System.Management.Automation.Language.Parser]::ParseFile(
        $PowerShellScript.FullName,
        [ref]$Tokens,
        [ref]$ParseErrors
    )
    if ($null -eq $Ast -or @($ParseErrors).Count -gt 0) {
        $ParseFailures += [PSCustomObject]@{
            File = $PowerShellScript.FullName
            Errors = (@($ParseErrors) | ForEach-Object { $_.Message }) -join " | "
        }
    }
}
if ($ParseFailures.Count -gt 0) {
    $ParseFailures | Format-List
    throw "Repository PowerShell parser validation failed."
}
Write-Host "REPOSITORY_POWERSHELL_PARSE=PASS"

Invoke-NativeChecked -Step "Compile Python source" -FilePath $PythonExe -Arguments @("-m", "compileall", "-q", "src")
Invoke-NativeChecked -Step "Ruff lint" -FilePath $PythonExe -Arguments @("-m", "ruff", "check", ".")
Invoke-NativeChecked -Step "Ruff formatting" -FilePath $PythonExe -Arguments @("-m", "ruff", "format", "--check", ".")
Invoke-NativeChecked -Step "Mypy" -FilePath $PythonExe -Arguments @("-m", "mypy", "--python-version", "3.11", "--no-site-packages", "src")
Invoke-NativeChecked -Step "Pytest" -FilePath $PythonExe -Arguments @("-m", "pytest", "-q")

$FixtureRoot = Join-Path $PhaseRoot "fixture"
Invoke-NativeChecked -Step "Synthetic fixture pipeline" -FilePath $PythonExe -Arguments @("-m", "visualact_r", "fixture-run", "--output", $FixtureRoot)
Invoke-NativeChecked -Step "Synthetic fixture qualification" -FilePath $PythonExe -Arguments @("-m", "visualact_r", "qualify", "--run-dir", $FixtureRoot)

if (-not $SkipFullPackageVerifier) {
    Write-Host ""
    Write-Host "===== 6. Clean source, wheel, and sdist package validation ====="
    $CleanVerifierRoot = Join-Path $PhaseRoot "clean_local_verify"
    Invoke-NativeChecked -Step "Clean source/wheel/sdist verification" -FilePath $PythonExe -Arguments @(
        $VerifierScript, "--output", $CleanVerifierRoot
    )
}
else {
    Write-Host "SKIPPED: full source/wheel/sdist verifier by explicit -SkipFullPackageVerifier."
}

# 7. Direct BDD layout inventory. It is strictly read-only and does not call archive-mode preflight.
Write-Host ""
Write-Host "===== 7. Read-only BDD train/val layout inventory ====="
$InventoryScript = Join-Path $PythonScriptRoot "inspect_bdd_layout.py"
$InventoryOutput = Join-Path $PhaseRoot "bdd_layout_inventory.json"
Write-PhasePython -Path $InventoryScript -Content @'
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ARCHIVE_EXTENSIONS = {".zip", ".7z", ".gz", ".tar", ".tgz"}


def count_images_and_sample(root: Path, sample_limit: int = 10) -> tuple[int, list[str]]:
    count = 0
    sample: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1
            if len(sample) < sample_limit:
                sample.append(path.name)
    return count, sample


def describe_payload(payload: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"payload_type": type(payload).__name__}
    if isinstance(payload, list):
        result["row_count"] = len(payload)
        if payload and isinstance(payload[0], dict):
            first = payload[0]
            result["first_row_keys"] = sorted(first.keys())
            result["first_row_name"] = first.get("name")
            labels = first.get("labels")
            result["first_row_label_count"] = len(labels) if isinstance(labels, list) else None
            if isinstance(labels, list) and labels and isinstance(labels[0], dict):
                result["first_label_keys"] = sorted(labels[0].keys())
                result["first_label_has_box2d"] = "box2d" in labels[0]
    elif isinstance(payload, dict):
        result["top_level_keys"] = sorted(payload.keys())[:100]
    return result


def inspect_split(name: str, images_dir: Path, labels_dir: Path) -> dict[str, Any]:
    image_count, sample_images = count_images_and_sample(images_dir)
    json_files = sorted(labels_dir.rglob("*.json"))
    jsonl_files = sorted(labels_dir.rglob("*.jsonl"))
    archives = sorted(
        path for path in labels_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in ARCHIVE_EXTENSIONS
    )
    report: dict[str, Any] = {
        "split": name,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "image_count": image_count,
        "sample_images": sample_images,
        "json_count": len(json_files),
        "jsonl_count": len(jsonl_files),
        "label_archives": [
            {"name": path.name, "suffix": path.suffix.lower(), "size_bytes": path.stat().st_size}
            for path in archives[:20]
        ],
        "json_samples": [],
    }
    for path in json_files[:3]:
        entry: dict[str, Any] = {"name": path.name, "size_bytes": path.stat().st_size}
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            entry["parse_status"] = "PASS"
            entry["schema"] = describe_payload(payload)
        except Exception as exc:
            entry["parse_status"] = "ERROR"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["json_samples"].append(entry)
    report["annotation_status"] = "JSON_DISCOVERED" if json_files else "NO_JSON_FOUND"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdd-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.bdd_root).resolve()
    report = {
        "bdd_root": str(root),
        "train": inspect_split("TRAIN", root / "Images" / "train", root / "Labels" / "train"),
        "val": inspect_split("VAL", root / "Images" / "val", root / "Labels" / "val"),
        "claim_boundary": "Read-only layout discovery. No BDD files are changed and no split, training, or evaluation is run.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split_name in ("train", "val"):
        split = report[split_name]
        print(f"[{split_name.upper()}]")
        print(f"image_count={split['image_count']}")
        print(f"json_count={split['json_count']}")
        print(f"jsonl_count={split['jsonl_count']}")
        print(f"annotation_status={split['annotation_status']}")
        for item in split["json_samples"]:
            print(f"json={item['name']}")
            print(f"parse_status={item['parse_status']}")
            if item["parse_status"] == "PASS":
                schema = item["schema"]
                print(f"first_row_keys={schema.get('first_row_keys')}")
                print(f"first_label_keys={schema.get('first_label_keys')}")
                print(f"first_label_has_box2d={schema.get('first_label_has_box2d')}")
    print(f"inventory_json={output}")


if __name__ == "__main__":
    main()
'@
Invoke-NativeChecked -Step "BDD direct-layout inventory" -FilePath $PythonExe -Arguments @(
    $InventoryScript, "--bdd-root", $BddRoot, "--output", $InventoryOutput
)

$InventoryText = Get-Content -LiteralPath $InventoryOutput -Raw -Encoding UTF8
$Inventory = $InventoryText | ConvertFrom-Json
foreach ($SplitName in @("train", "val")) {
    $Split = $Inventory.$SplitName
    if ($Split.image_count -le 0) {
        throw "BDD $SplitName image inventory is empty: $BddRoot"
    }
    if (($Split.json_count + $Split.jsonl_count) -le 0) {
        throw "BDD $SplitName annotation inventory contains no JSON or JSONL files: $BddRoot"
    }
    $ParseErrors = @($Split.json_samples | Where-Object { $_.parse_status -eq "ERROR" })
    if ($ParseErrors.Count -gt 0) {
        throw "BDD $SplitName sampled JSON annotation parsing failed. See inventory: $InventoryOutput"
    }
}
Write-Host "BDD_DIRECT_LAYOUT_GATE=PASS"

Write-Host ""
Write-Host "============================================================"
Write-Host "PHASE_A_PASS"
Write-Host "============================================================"
Write-Host "REPO_ROOT=$RepoRoot"
Write-Host "PHASE_OUTPUT=$PhaseRoot"
Write-Host "BDD_INVENTORY=$InventoryOutput"
Write-Host ""
Write-Host "STOP POINT: Do not run real BDD build-data, train-cef, train-vlm, rollout-qwen, train-rctr, route-review, or compare-final yet."
Write-Host "Required next implementation: direct BDD adapter that preserves BDD train for train/dev/calibration and BDD val as untouched frozen final."
