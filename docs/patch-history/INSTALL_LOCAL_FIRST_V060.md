# Install VisualAct-R v0.6.0 Local-First Patch

## Destination

Extract this flat ZIP directly into:

```text
C:\Users\bjw-0\Downloads\Apple01_visualact-r
```

It updates VisualAct-R source, tests, resources, and documentation. It does not contain `.venv` or the Phase A PowerShell files, so extraction does not delete them.

## Preflight after extraction

Open PowerShell and execute this entire block:

```powershell
& {
    Set-StrictMode -Version Latest
    $ErrorActionPreference = "Stop"

    $RepoRoot = "C:\Users\bjw-0\Downloads\Apple01_visualact-r"
    $PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

    foreach ($Path in @(
        (Join-Path $RepoRoot "pyproject.toml"),
        (Join-Path $RepoRoot "src\visualact_r\data\bdd_direct.py"),
        (Join-Path $RepoRoot "src\visualact_r\agents\feasibility.py"),
        (Join-Path $RepoRoot "src\visualact_r\resources\local_qlora_pilot.yaml"),
        $PythonExe
    )) {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "V060_REQUIRED_PATH_MISSING: $Path"
        }
    }

    Set-Location $RepoRoot

    & $PythonExe -m compileall -q src tests
    if ($LASTEXITCODE -ne 0) { throw "V060_COMPILE_FAILED" }

    & $PythonExe -m pytest -q tests/test_bdd_direct.py tests/test_qwen_contract.py tests/test_local_qlora_feasibility.py
    if ($LASTEXITCODE -ne 0) { throw "V060_TARGETED_TESTS_FAILED" }

    & $PythonExe -m visualact_r --help
    if ($LASTEXITCODE -ne 0) { throw "V060_CLI_HELP_FAILED" }

    Write-Host "V060_LOCAL_FIRST_PATCH=PASS"
}
```

## Read-only direct BDD audit

This reads annotation JSON and verifies exact image-label joins. It does not open every image, write task records, train, or download a model.

```powershell
$RepoRoot = "C:\Users\bjw-0\Downloads\Apple01_visualact-r"
$BddRoot = "C:\Users\bjw-0\Downloads\Project_Data\bdd100k"
$AuditRoot = "$HOME\ML_Outputs\Apple01_visualact-r\bdd-direct-audit"

& "$RepoRoot\.venv\Scripts\python.exe" -m visualact_r bdd-direct-audit `
  --bdd-root $BddRoot `
  --output "$AuditRoot\bdd_direct_audit.json"
```

Proceed only when the command prints `PASS`.

## Direct BDD build

This is CPU/image-I/O intensive because it opens images to calculate pHashes. It does not train any model.

```powershell
$BuildRoot = "$HOME\ML_Outputs\Apple01_visualact-r\bdd-direct-full"

& "$RepoRoot\.venv\Scripts\python.exe" -m visualact_r bdd-direct-build `
  --bdd-root $BddRoot `
  --output $BuildRoot
```

The default policy blocks on a pHash near-duplicate crossing BDD train-derived development and BDD val frozen-final. Do not bypass `--allow-cross-partition-phash` without documenting the exception.

## One-image 3B QLoRA VRAM feasibility gate

Use a **BDD train** image, never a BDD val image. First list a candidate safely:

```powershell
$PilotImage = Get-ChildItem -LiteralPath "$BddRoot\Images\train" -Recurse -File -Include *.jpg,*.jpeg,*.png |
    Select-Object -First 1 -ExpandProperty FullName
$PilotImage
```

Then run the capacity gate:

```powershell
$PilotRoot = "$HOME\ML_Outputs\Apple01_visualact-r\qlora-feasibility"

& "$RepoRoot\.venv\Scripts\python.exe" -m visualact_r local-qlora-feasibility `
  --image $PilotImage `
  --output $PilotRoot
```

The report is written to:

```text
$PilotRoot\local_qlora_feasibility_report.json
```

`PASS` means one image-conditioned 4-bit NF4 QLoRA forward/backward completed within the 95% VRAM reserve policy. It is a capacity result only, not a model-performance result.
