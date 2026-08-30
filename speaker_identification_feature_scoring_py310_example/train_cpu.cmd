@echo off
setlocal
set "PROJECT_PYTHON=%~dp0venv\Scripts\python.exe"

if not exist "%PROJECT_PYTHON%" (
    echo Project environment not found. Run setup_venv.ps1 first. 1>&2
    exit /b 1
)

"%PROJECT_PYTHON%" "%~dp0scripts\train.py" %*
exit /b %errorlevel%
