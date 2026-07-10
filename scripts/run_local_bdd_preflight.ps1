param(
  [Parameter(Mandatory=$true)][string]$RepoRoot,
  [Parameter(Mandatory=$true)][string]$DataRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Expected virtual environment Python at $Py" }

& $Py -m visualact_r preflight `
  --data-root $DataRoot `
  --labels-zip "bdd100k_labels_images_val.zip" `
  --shard "BDD_first4000.7z" `
  --shard "BDD_second3000.7z" `
  --shard "BDD_third3000.7z" `
  --output (Join-Path $RepoRoot "runs\bdd_preflight")

# PowerShell does not automatically treat a nonzero native-process exit as a terminating error.
# Preserve the CLI failure for both direct local invocation and the GitHub Windows runner.
if ($LASTEXITCODE -ne 0) {
  throw "VisualAct-R BDD preflight failed with exit code $LASTEXITCODE."
}

Write-Host "BDD_PREFLIGHT_WRAPPER=PASS"
