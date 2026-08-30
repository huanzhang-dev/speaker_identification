$ErrorActionPreference = "Stop"

$Venv = Join-Path $PSScriptRoot "venv"
python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"
python -m venv $Venv
$Python = Join-Path $Venv "Scripts\python.exe"

& $Python -m pip install --upgrade pip setuptools wheel
& $Python -m pip install -r (Join-Path $PSScriptRoot "requirements-cpu.txt")
& $Python -m pip install `
    numpy==1.26.4 `
    pandas==2.2.2 `
    scikit-learn==1.5.1 `
    matplotlib==3.8.4 `
    soundfile==0.12.1 `
    pytest==8.2.2

& $Python (Join-Path $PSScriptRoot "scripts\check_environment.py")
