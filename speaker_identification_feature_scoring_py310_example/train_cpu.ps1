$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Project environment not found. Run .\setup_venv.ps1 first."
}

& $Python (Join-Path $PSScriptRoot "scripts\train.py") @args
exit $LASTEXITCODE
