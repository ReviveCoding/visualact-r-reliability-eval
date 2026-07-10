[CmdletBinding()]
param(
    [string]$RepoRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ExpectedPhaseAVersion = "0.5.6"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = $PSScriptRoot
}
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$ManifestPath = Join-Path $RepoRoot "visualact-r-phase-a-manifest.json"

foreach ($Required in @(
    (Join-Path $RepoRoot "pyproject.toml"),
    (Join-Path $RepoRoot "run_visualact_phase_a.ps1"),
    (Join-Path $RepoRoot "Start-VisualActR-PhaseA.ps1"),
    $ManifestPath
)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "PHASE_A_INTEGRITY_REQUIRED_FILE_MISSING: $Required"
    }
}

$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($Manifest.phase_a_version -ne $ExpectedPhaseAVersion) {
    throw "PHASE_A_MANIFEST_VERSION_MISMATCH: expected $ExpectedPhaseAVersion, found $($Manifest.phase_a_version)."
}

foreach ($Entry in @($Manifest.files)) {
    $Path = Join-Path $RepoRoot $Entry.name
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "PHASE_A_MANIFEST_FILE_MISSING: $Path"
    }
    $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Entry.sha256) {
        throw "PHASE_A_MANIFEST_HASH_MISMATCH: $($Entry.name). Expected $($Entry.sha256), found $Actual."
    }
}

Write-Host "PHASE_A_ARTIFACT_INTEGRITY=PASS"
Write-Host "PHASE_A_VERSION=$ExpectedPhaseAVersion"
Write-Host "REPO_ROOT=$RepoRoot"
