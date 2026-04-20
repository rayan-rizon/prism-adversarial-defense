$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"
$Log = "pipeline_run_log.txt"

function Log($msg) {
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $Log -Value $line
}

Set-Location "c:\Users\rayan\Desktop\Research\Prism\prism-adversarial-defense\prism"
"" | Set-Content $Log

Log "=== PHASE C1: Build reference profiles ==="
& $Python scripts/build_profile_testset.py
if ($LASTEXITCODE -ne 0) { Log "ERROR in C1"; exit 1 }
Log "C1 done"

Log "=== PHASE C2: Train ensemble scorer (37-dim DCT) ==="
& $Python scripts/train_ensemble_scorer.py --n-train 2000
if ($LASTEXITCODE -ne 0) { Log "ERROR in C2"; exit 1 }
Log "C2 done"

Log "=== PHASE C3: Calibrate ensemble thresholds ==="
& $Python scripts/calibrate_ensemble.py
if ($LASTEXITCODE -ne 0) { Log "ERROR in C3"; exit 1 }
Log "C3 done"

Log "=== PHASE C4: FPR gate check ==="
& $Python scripts/compute_ensemble_val_fpr.py
if ($LASTEXITCODE -ne 0) { Log "ERROR in C4"; exit 1 }
Log "C4 done"

Log "=== PHASE C5: Train MoE experts ==="
& $Python scripts/train_experts.py
if ($LASTEXITCODE -ne 0) { Log "ERROR in C5"; exit 1 }
Log "C5 done"

Log "=== PHASE D-GPU: n=500 evaluation on CUDA ==="
& $Python experiments/evaluation/run_evaluation_full.py `
    --n-test 500 `
    --attacks FGSM PGD Square `
    --seed 42 `
    --device cuda `
    --output experiments/evaluation/results_n500_cuda_20260419.json
if ($LASTEXITCODE -ne 0) { Log "ERROR in D-GPU"; exit 1 }
Log "D-GPU done"

Log "=== PHASE D-CPU: n=500 evaluation on CPU ==="
& $Python experiments/evaluation/run_evaluation_full.py `
    --n-test 500 `
    --attacks FGSM PGD Square `
    --seed 42 `
    --device cpu `
    --output experiments/evaluation/results_n500_cpu_20260419.json
if ($LASTEXITCODE -ne 0) { Log "ERROR in D-CPU"; exit 1 }
Log "D-CPU done"

Log "=== ALL PHASES COMPLETE ==="
Log "GPU results: experiments/evaluation/results_n500_cuda_20260419.json"
Log "CPU results: experiments/evaluation/results_n500_cpu_20260419.json"
