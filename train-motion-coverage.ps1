param([switch]$AuditOnly)
$ErrorActionPreference = 'Stop'
$mode = if ($AuditOnly) { @('--audit-only', '--no-dashboard') } else { @() }
& (Join-Path $PSScriptRoot 'train-temporal.ps1') -MotionTrain --final-tune --coverage-tune --frame-view full `
    --epochs 12 --patience 5 --videos-per-epoch 1600 --train-clips 2 --batch-size 4 --decode-workers 4 `
    --backbone-lr 0.000005 --head-lr 0.00003 --cache-gb 12 --reserve-gb 12 @mode
