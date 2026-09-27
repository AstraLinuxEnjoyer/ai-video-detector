param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PhotoArgs)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'train-spatial.ps1') --predict-images @PhotoArgs
