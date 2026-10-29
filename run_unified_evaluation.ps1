$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
$Runner = Join-Path $Root "unified_evaluation.py"

if (-not (Test-Path $Runner)) {
    throw "Unified evaluator not found: $Runner"
}

Push-Location $Root
try {
    & $Python $Runner @args
    $ExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $ExitCode
