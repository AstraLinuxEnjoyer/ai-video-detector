param([switch]$FullFrame)
$TrainingArgs = $args
$ErrorActionPreference = 'Stop'
$run = Join-Path $PSScriptRoot 'cache\artifact_spatial\runs\hybrid_v2'
$parent = Join-Path $PSScriptRoot 'cache\artifact_spatial\runs\patch_v1'
$recipe = 'hybrid-v2'
if ($FullFrame) {
    $parent = $run
    $run = Join-Path $PSScriptRoot 'cache\artifact_spatial\runs\full_frame_v1'
    $recipe = 'full-frame-v1'
}
$isInference = ($TrainingArgs -contains '--predict-images') -or ($TrainingArgs -contains '--evaluate-only')
if ($isInference) {
    & (Join-Path $PSScriptRoot 'train-artifacts.ps1') --run-dir $run @TrainingArgs
} else {
    & (Join-Path $PSScriptRoot 'train-artifacts.ps1') --run-dir $run --source-run $parent --recipe $recipe @TrainingArgs
}
