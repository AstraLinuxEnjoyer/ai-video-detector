param(
    [switch]$DashboardOnly,
    [switch]$MotionAudit,
    [switch]$MotionTrain,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$TrainingArgs
)
$ErrorActionPreference = 'Stop'
if ($DashboardOnly -and $MotionAudit) { throw 'Choose either DashboardOnly or MotionAudit.' }
if ($MotionTrain -and ($DashboardOnly -or $MotionAudit)) { throw 'MotionTrain starts a new motion experiment; choose one mode.' }
$oldPythonPath = $env:PYTHONPATH
$projectPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$selectedPython = $null
Push-Location $PSScriptRoot
try {
    if (Test-Path -LiteralPath $projectPython) {
        try {
            & $projectPython -c 'import torch, torchvision; import sys; assert sys.version_info[:2] == (3, 12)' 2>$null
            if ($LASTEXITCODE -eq 0) { $selectedPython = $projectPython }
        } catch { Write-Host 'Trying compatible local Python runtime.' }
    }
    if (-not $selectedPython -and (Test-Path -LiteralPath $bundledPython)) {
        $selectedPython = $bundledPython
        $env:PYTHONPATH = "$(Join-Path $PSScriptRoot 'venv\Lib\site-packages');$PSScriptRoot;$oldPythonPath"
        & $selectedPython -c 'import torch, torchvision; import sys; assert sys.version_info[:2] == (3, 12)'
        if ($LASTEXITCODE -ne 0) { throw 'A working Python 3.12 + PyTorch environment is required.' }
    }
    if (-not $selectedPython) { throw 'Python runtime missing.' }
    $module = if ($DashboardOnly) { 'training.temporal_v2.dashboard' } elseif ($MotionAudit) { 'training.temporal_v2.motion_audit' } else { 'training.temporal_v2.run' }
    if ($MotionTrain) { $TrainingArgs = @('--motion') + $TrainingArgs }
    & $selectedPython -m $module @TrainingArgs
    if ($LASTEXITCODE -ne 0) { throw "Temporal command exited with code $LASTEXITCODE. See training.log." }
}
finally {
    $env:PYTHONPATH = $oldPythonPath
    Pop-Location
}
