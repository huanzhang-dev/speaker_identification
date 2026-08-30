$ErrorActionPreference = "Stop"

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-cuda121.txt
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.1 matplotlib==3.8.4 pytest==8.2.2

python scripts\check_environment.py
