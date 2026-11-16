param([switch]$AuditOnly)
$ErrorActionPreference = 'Stop'
$mode = if ($AuditOnly) { @('--audit-only', '--no-dashboard') } else { @() }
& (Join-Path $PSScriptRoot 'train-temporal.ps1') -MotionTrain --final-tune --coverage-tune --paired-batch --frame-view full `
    --epochs 4 --patience 4 --videos-per-epoch 1000 --train-clips 2 --batch-size 4 --decode-workers 4 `
    --backbone-lr 0.000003 --head-lr 0.00002 --cache-gb 12 --reserve-gb 12 @mode
