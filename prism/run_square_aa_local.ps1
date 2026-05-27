# =============================================================================
# PRISM Stretch A - Local Square / AutoAttack runner (resume-capable, per-attack)
# =============================================================================
# Runs Square and/or AutoAttack on WRN-28-10 for seeds 123 / 456 / 789 / 999.
# Per-attack, per-seed checkpointing - survives power loss, can be split into
# two phases (fast AA first, slow Square later).
#
# Resume logic:
#   For each (seed, attack) pair, if results_fast_wrn_seed{S}.json already
#   has a valid TPR for that attack, that pair is SKIPPED. Otherwise the
#   missing attack(s) are run and MERGED into the existing JSON without
#   touching the attacks that are already there.
#
# Hardware: GTX 1060 6GB or similar (small chunks chosen for 6GB VRAM).
#
# Typical workflow:
#   # Phase 1 - AutoAttack only (fast, ~40 min total for 4 seeds)
#   .\run_square_aa_local.ps1 -Attacks AutoAttack
#
#   # Phase 2 - Square (slow, ~13-15 hours total - run overnight)
#   .\run_square_aa_local.ps1 -Attacks Square
#
#   # Or both in one go (default):
#   .\run_square_aa_local.ps1
#
# Other usage:
#   .\run_square_aa_local.ps1 -DryRun           # print plan, do not run
#   .\run_square_aa_local.ps1 -Seeds 123,456    # only specific seeds
#   .\run_square_aa_local.ps1 -NTest 500        # smaller N (default 1000)
#   .\run_square_aa_local.ps1 -SquareIter 1000  # faster but lower-quality Square
#   .\run_square_aa_local.ps1 -Force            # re-run even if results exist
#
# After power loss / shutdown:
#   Re-run the same command. Completed (seed, attack) pairs are skipped.
# =============================================================================
param(
    [int[]]    $Seeds       = @(123, 456, 789, 999),
    [string[]] $Attacks     = @('AutoAttack', 'Square'),
    [int]      $NTest       = 1000,
    [int]      $SquareIter  = 5000,
    [int]      $AAChunk     = 16,
    [int]      $GenChunk    = 32,
    [switch]   $DryRun,
    [switch]   $Force
)

$ErrorActionPreference = 'Stop'

# Paths
$Root    = "C:\Users\rayan\Desktop\Research\Prism\prism-adversarial-defense\prism"
$VenvPy  = "$Root\.venv\Scripts\python.exe"
$EvalDir = "$Root\experiments\wrn\evaluation"
$LogDir  = "$Root\logs\local_finish"

if (-not (Test-Path $VenvPy)) {
    Write-Error "venv python not found at $VenvPy"
    exit 1
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Env
$env:PRISM_CONFIG    = "configs/wrn_cifar10.yaml"
$env:PYTHONPATH      = $Root
$env:PYTHONUNBUFFERED = "1"
Set-Location $Root

# Resume check: does this seed JSON already have valid results for an attack?
function Test-AttackDone {
    param([int]$Seed, [string]$Attack)
    $jsonPath = "$EvalDir\results_fast_wrn_seed$Seed.json"
    if (-not (Test-Path $jsonPath)) { return $false }
    try {
        $data = Get-Content $jsonPath -Raw | ConvertFrom-Json
        $names = $data.PSObject.Properties.Name
        if (-not ($names -contains $Attack)) { return $false }
        return ($null -ne $data.$Attack.TPR)
    } catch {
        return $false
    }
}

# Plan
Write-Output ""
Write-Output "============================================================"
Write-Output "PRISM Stretch A - Local runner"
Write-Output "Started:    $(Get-Date)"
Write-Output "Config:     $env:PRISM_CONFIG"
Write-Output "Attacks:    $($Attacks -join ', ')"
Write-Output "Seeds:      $($Seeds -join ', ')"
Write-Output "N per seed: $NTest"
Write-Output "Square max_iter: $SquareIter   AA chunk: $AAChunk   Gen chunk: $GenChunk"
Write-Output "Output dir: $EvalDir"
Write-Output "Log dir:    $LogDir"
Write-Output "============================================================"
Write-Output ""

# Build per-seed task list: for each seed, which attacks need running?
$tasks = @()
foreach ($seed in $Seeds) {
    $missing = @()
    foreach ($atk in $Attacks) {
        $done = Test-AttackDone -Seed $seed -Attack $atk
        if ($done -and -not $Force) {
            Write-Output ("  seed {0,-5}  {1,-12}  SKIP (already complete)" -f $seed, $atk)
        } else {
            if ($done -and $Force) {
                $tag = "RE-RUN (Force)"
            } else {
                $tag = "RUN"
            }
            Write-Output ("  seed {0,-5}  {1,-12}  {2}" -f $seed, $atk, $tag)
            $missing += $atk
        }
    }
    if ($missing.Count -gt 0) {
        $tasks += [PSCustomObject]@{ Seed = $seed; AttacksToRun = $missing }
    }
}

if ($tasks.Count -eq 0) {
    Write-Output ""
    Write-Output "Nothing to do. All requested (seed, attack) pairs already complete."
    exit 0
}

if ($DryRun) {
    Write-Output ""
    Write-Output "[DryRun] Would run $($tasks.Count) seed-batch(es). Exiting."
    exit 0
}

# Execute
$grandT0 = Get-Date
$failed = @()

foreach ($task in $tasks) {
    $seed = $task.Seed
    $atks = $task.AttacksToRun
    Write-Output ""
    Write-Output "-----------------------------------------------------------------"
    Write-Output ("Running seed=$seed   attacks=$($atks -join ',')   {0}" -f (Get-Date))
    Write-Output "-----------------------------------------------------------------"

    $atkTag = ($atks -join '_').ToLower()
    $log    = "$LogDir\${atkTag}_seed$seed.log"
    $final  = "$EvalDir\results_fast_wrn_seed$seed.json"
    $tmpOut = "$EvalDir\.results_${atkTag}_seed$seed.tmp.json"

    if (Test-Path $tmpOut) { Remove-Item $tmpOut -Force }

    $seedT0 = Get-Date
    try {
        & $VenvPy "$Root\experiments\evaluation\run_evaluation_full.py" `
            --attacks $atks `
            --n-test $NTest `
            --seed $seed `
            --aa-chunk $AAChunk `
            --gen-chunk $GenChunk `
            --square-max-iter $SquareIter `
            --output $tmpOut `
            --skip-latency 2>&1 | Tee-Object -FilePath $log
        $exitCode = $LASTEXITCODE
    } catch {
        Write-Output "EXCEPTION while running seed=${seed}: $_"
        $exitCode = -1
    }
    $seedDt = (Get-Date) - $seedT0

    if ($exitCode -eq 0 -and (Test-Path $tmpOut)) {
        # Merge tmp into final JSON. If final exists, preserve its existing
        # attack keys; tmp's keys for the just-run attacks overwrite/insert.
        # Atomic via write-to-side-file-then-rename.
        $mergeScript = @"
import json, sys
from pathlib import Path
final_path = Path(r'$final')
tmp_path   = Path(r'$tmpOut')
new_data   = json.loads(tmp_path.read_text())
if final_path.exists():
    try:
        merged = json.loads(final_path.read_text())
    except Exception:
        merged = {}
else:
    merged = {}
# Copy over only the requested-attack keys + always-update meta
for k, v in new_data.items():
    if k.startswith('_'):
        merged[k] = v       # meta keys always refreshed
        continue
    merged[k] = v           # attack key (overwrites if existed)
side = final_path.with_suffix('.json.new')
side.write_text(json.dumps(merged, indent=2))
side.replace(final_path)    # atomic rename
tmp_path.unlink()
print('MERGED', list(new_data.keys()), '->', str(final_path))
"@
        & $VenvPy -c $mergeScript
        $mergeExit = $LASTEXITCODE
        if ($mergeExit -eq 0) {
            Write-Output ("[OK] seed {0} {1} DONE in {2:N1} min  ->  {3}" -f $seed, ($atks -join '+'), $seedDt.TotalMinutes, $final)
        } else {
            Write-Output ("[FAIL] seed {0} merge failed (exit={1}) - tmp left at {2}" -f $seed, $mergeExit, $tmpOut)
            $failed += "seed${seed}-merge"
        }
    } else {
        Write-Output ("[FAIL] seed {0} run failed (exit={1}) after {2:N1} min - see {3}" -f $seed, $exitCode, $seedDt.TotalMinutes, $log)
        $failed += "seed${seed}-run"
    }
}

$grandDt = (Get-Date) - $grandT0
Write-Output ""
Write-Output "============================================================"
Write-Output ("Run complete.  Total wall-clock: {0:N1} min" -f $grandDt.TotalMinutes)
Write-Output ""

if ($failed.Count -gt 0) {
    Write-Output "FAILED: $($failed -join ', ')"
    Write-Output "Re-run the same command - completed pairs will be skipped."
    exit 1
}

# Aggregate across all seeds for paper table
Write-Output "Aggregating Square + AutoAttack across all seeds..."
$aggOut = "$EvalDir\results_fast_wrn_aggregate.json"
$aggScript = @"
import json, glob, statistics
from pathlib import Path
files = sorted(glob.glob(r'$EvalDir' + '/results_fast_wrn_seed*.json'))
agg = {}
for atk in ['Square', 'AutoAttack', 'FGSM', 'PGD']:
    tprs, fprs, sds = [], [], []
    for fp in files:
        d = json.loads(Path(fp).read_text())
        if atk in d and 'TPR' in d[atk]:
            tprs.append(d[atk]['TPR']); fprs.append(d[atk]['FPR'])
            sds.append(Path(fp).name)
    if tprs:
        agg[atk] = {
            'TPR_mean': round(statistics.mean(tprs), 4),
            'TPR_std':  round(statistics.pstdev(tprs) if len(tprs) > 1 else 0.0, 4),
            'FPR_mean': round(statistics.mean(fprs), 4),
            'FPR_std':  round(statistics.pstdev(fprs) if len(fprs) > 1 else 0.0, 4),
            'n_seeds':  len(tprs),
            'per_seed_tpr': tprs,
            'per_seed_fpr': fprs,
            'source_files': sds,
        }
out = {'aggregate': agg, 'all_source_files': files}
Path(r'$aggOut').write_text(json.dumps(out, indent=2))
print(json.dumps(agg, indent=2))
"@
& $VenvPy -c $aggScript
Write-Output ""
Write-Output "Aggregate written to: $aggOut"
Write-Output "============================================================"
