$ErrorActionPreference = "Stop"

Write-Host "1/4 Training small example model..."
python scripts\train.py `
    --config example_data\config_example.json `
    --train-manifest example_data\train.csv `
    --dev-manifest example_data\dev.csv `
    --data-root example_data `
    --output-dir runs\example

Write-Host "2/4 Building enrollment gallery..."
python scripts\enroll.py `
    --checkpoint runs\example\best_encoder.pt `
    --manifest example_data\enroll.csv `
    --data-root example_data `
    --output runs\example\gallery.npz

Write-Host "3/4 Evaluating known + unknown probes..."
python scripts\evaluate.py `
    --checkpoint runs\example\best_encoder.pt `
    --enroll-manifest example_data\enroll.csv `
    --probe-manifest example_data\probes.csv `
    --data-root example_data `
    --output-dir runs\example\evaluation

Write-Host "4/4 Calibrating threshold for FAR=0.25 (tiny demo set)..."
python scripts\calibrate_threshold.py `
    --probe-scores runs\example\evaluation\probe_scores.csv `
    --target-far 0.25 `
    --output runs\example\threshold_demo.json

Write-Host ""
Write-Host "Example pipeline complete."
Write-Host "Model:      runs\example\best_encoder.pt"
Write-Host "Gallery:    runs\example\gallery.npz"
Write-Host "Evaluation: runs\example\evaluation\metrics.json"
