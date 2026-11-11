$TrainingArgs = $args
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$oldPythonPath = $env:PYTHONPATH
$selectedPython = $null
Push-Location $PSScriptRoot
try {
    if (Test-Path -LiteralPath $projectPython) {
        try {
            & $projectPython -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>$null
            if ($LASTEXITCODE -eq 0) { $selectedPython = $projectPython }
        } catch { Write-Host 'Project Python unavailable; trying compatible local runtime.' }
    }
    if (-not $selectedPython -and (Test-Path -LiteralPath $bundledPython)) {
        $selectedPython = $bundledPython
        $env:PYTHONPATH = "$PSScriptRoot\venv\Lib\site-packages;$PSScriptRoot;$oldPythonPath"
    }
    if (-not $selectedPython) { throw 'A working Python 3.12 environment is required.' }
    & $selectedPython -m training.spectral.run @TrainingArgs
    if ($LASTEXITCODE -ne 0) { throw "Spectral command exited with code $LASTEXITCODE. See console and training.log." }
}
finally {
    $env:PYTHONPATH = $oldPythonPath
    Pop-Location
}
