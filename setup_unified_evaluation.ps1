$ErrorActionPreference = "Stop"
param(
    [switch]$SkipInstall,
    [string]$PythonCommand = "python"
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Parent = Split-Path -Parent $Root
$Repositories = @(
    @{ Name = "behavior-bench"; Url = "https://github.com/eliem08/behavior-bench.git"; Path = (Join-Path $Parent "behavior-bench") },
    @{ Name = "safety-probe-lab"; Url = "https://github.com/eliem08/safety-probe-lab.git"; Path = (Join-Path $Parent "safety-probe-lab") }
)

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git was not found. Install Git for Windows, reopen PowerShell, and retry."
}

foreach ($Repo in $Repositories) {
    if (Test-Path (Join-Path $Repo.Path ".git")) {
        Write-Host "Found $($Repo.Name) at $($Repo.Path)"
        continue
    }
    if (Test-Path $Repo.Path) {
        throw "Path exists but is not a Git repository: $($Repo.Path). Move it or verify it manually before retrying."
    }
    Write-Host "Cloning $($Repo.Name)..."
    git clone $Repo.Url $Repo.Path
    if ($LASTEXITCODE -ne 0) {
        throw "Clone failed for $($Repo.Name) (exit $LASTEXITCODE)."
    }
}

if ($SkipInstall) {
    Write-Host "Repository setup complete. Dependency installation was skipped."
    exit 0
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating shared virtual environment..."
    & $PythonCommand -m venv (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the virtual environment. Confirm Python 3.10+ is installed."
    }
}

Write-Host "Installing Experiment Scientist dependencies. This may take several minutes."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $VenvPython -m pip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next: .\.venv\Scripts\python.exe unified_evaluation.py --model-id distilgpt2 --device cpu"
