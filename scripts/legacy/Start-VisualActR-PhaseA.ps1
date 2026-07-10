[CmdletBinding()]
param(
    [string]$RepoRoot,
    [switch]$ValidateInvocation,
    [switch]$Run,
    [switch]$SkipFullPackageVerifier
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PhaseALauncherVersion = "0.5.6"

if ($ValidateInvocation -and $Run) {
    throw "Choose either -ValidateInvocation or -Run, not both."
}

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = $PSScriptRoot
}
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$PhaseScript = Join-Path $RepoRoot "run_visualact_phase_a.ps1"

if (-not (Test-Path -LiteralPath $PhaseScript -PathType Leaf)) {
    throw "Phase A script not found: $PhaseScript. Extract this bundle into the repository root."
}

$Required = @(
    "pyproject.toml",
    "src",
    "tests",
    "scripts\verify_local.py",
    "scripts\run_local_bdd_preflight.ps1",
    ".github\workflows\ci.yml"
)
foreach ($RelativePath in $Required) {
    $Candidate = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $Candidate)) {
        throw "Required repository item not found: $Candidate"
    }
}

$ScriptsToParse = @(
    Get-ChildItem -LiteralPath $RepoRoot -Recurse -File -Filter "*.ps1" -ErrorAction Stop
)
if ($ScriptsToParse.Count -eq 0) {
    throw "No PowerShell scripts found beneath: $RepoRoot"
}

$ParseFailures = @()
foreach ($ScriptFile in $ScriptsToParse) {
    $Tokens = $null
    $ParseErrors = $null
    $Ast = [System.Management.Automation.Language.Parser]::ParseFile(
        $ScriptFile.FullName,
        [ref]$Tokens,
        [ref]$ParseErrors
    )
    if ($null -eq $Ast -or @($ParseErrors).Count -gt 0) {
        $ParseFailures += [PSCustomObject]@{
            File = $ScriptFile.FullName
            Errors = (@($ParseErrors) | ForEach-Object { $_.Message }) -join " | "
        }
    }
}

if ($ParseFailures.Count -gt 0) {
    $ParseFailures | Format-List
    throw "PHASE_A_POWERSHELL_PARSE_FAILED"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "REPOSITORY_DISCOVERY=PASS"
Write-Host "REQUIRED_REPOSITORY_ITEMS=PASS"
Write-Host "PHASE_A_POWERSHELL_PARSE=PASS"
Write-Host "============================================================"
Write-Host "REPO_ROOT=$RepoRoot"
Write-Host "POWERSHELL_SCRIPT_COUNT=$($ScriptsToParse.Count)"

if ($ValidateInvocation) {
    # Use explicit named arguments. Array splatting would pass an argument array positionally
    # and can turn the literal token '-RepoRoot' into the value received by the main script.
    & $PhaseScript -RepoRoot $RepoRoot -ValidateInvocationOnly
    Write-Host "PHASE_A_LAUNCHER_BINDING=PASS"
}
elseif ($Run) {
    # Use explicit named arguments, not an array splat, so the repository root binds by name.
    if ($SkipFullPackageVerifier) {
        & $PhaseScript -RepoRoot $RepoRoot -SkipFullPackageVerifier
    }
    else {
        & $PhaseScript -RepoRoot $RepoRoot
    }
    Write-Host "PHASE_A_RUNTIME=PASS"
}
else {
    Write-Host ""
    Write-Host "Phase A v$PhaseALauncherVersion parser preflight completed. Next, validate launcher-to-main-script parameter binding:"
    Write-Host ".\Start-VisualActR-PhaseA.ps1 -ValidateInvocation"
    Write-Host "Then run the full Phase A gate:"
    Write-Host ".\Start-VisualActR-PhaseA.ps1 -Run"
}
