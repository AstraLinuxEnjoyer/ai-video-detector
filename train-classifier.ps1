param([Parameter(ValueFromRemainingArguments=$true)][string[]]$TrainingArgs)
$ErrorActionPreference = 'Stop'
$run = Join-Path $PSScriptRoot 'cache\artifact_spatial\runs\hybrid_v2'
$parent = Join-Path $PSScriptRoot 'cache\artifact_spatial\runs\patch_v1'
$isInference = ($TrainingArgs -contains '--predict-images') -or ($TrainingArgs -contains '--evaluate-only')
if ($isInference) {
    & (Join-Path $PSScriptRoot 'train-artifacts.ps1') --run-dir $run @TrainingArgs
} else {
    & (Join-Path $PSScriptRoot 'train-artifacts.ps1') --run-dir $run --source-run $parent --recipe hybrid-v2 @TrainingArgs
}
