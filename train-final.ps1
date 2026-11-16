param(
    [ValidateSet('Both','Spatial','Motion')][string]$Stage = 'Both',
    [switch]$AuditOnly
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$spatial = Join-Path $root 'cache\artifact_spatial\runs\final_hybrid_v1'
$motion = Join-Path $root 'cache\temporal_v2\runs\final_motion_v1'
$spatialAudit = @()
$motionAudit = @()
if ($AuditOnly) {
    $spatialAudit += '--prepare-only'
    $motionAudit += '--audit-only'
}

if ($Stage -eq 'Motion' -and -not (Test-Path -LiteralPath (Join-Path $spatial 'config.json'))) {
    throw 'Prepare the shared snapshot first: run train-final.ps1 -Stage Spatial -AuditOnly.'
}
if ($Stage -ne 'Motion') {
    & (Join-Path $root 'train-artifacts.ps1') --recipe final-hybrid-v1 --run-dir $spatial `
        --source-run (Join-Path $root 'cache\artifact_spatial\runs\hybrid_v2') `
        --epochs 16 --patience 5 --videos-per-epoch 3000 --frames-per-video 2 `
        --batch-size 3 --decode-workers 4 --cache-gb 20 --reserve-gb 12 @spatialAudit
    if (-not $AuditOnly -and -not (Test-Path -LiteralPath (Join-Path $spatial 'final_report.json'))) {
        Write-Host 'Spatial run interrupted. Repeat the same command to resume before starting motion.'
        return
    }
}
if ($Stage -ne 'Spatial') {
    & (Join-Path $root 'train-temporal.ps1') -MotionTrain --final-tune --frame-view full --run-dir $motion `
        --source-run $spatial --epochs 16 --patience 5 --videos-per-epoch 800 --train-clips 2 `
        --max-fake-source-fraction 0.6 --batch-size 4 --decode-workers 4 --cache-gb 12 --reserve-gb 12 @motionAudit
}
