$TrainingArgs = $args
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$projectPython = Join-Path $projectRoot 'venv\Scripts\python.exe'
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$oldPythonPath = $env:PYTHONPATH
$selectedPython = $null
Push-Location $projectRoot
try {
    if (Test-Path -LiteralPath $projectPython) {
        try {
            & $projectPython -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>$null
            if ($LASTEXITCODE -eq 0) { $selectedPython = $projectPython }
        } catch { Write-Host 'Project Python unavailable; trying compatible local runtime.' }
    }
    if (-not $selectedPython -and (Test-Path -LiteralPath $bundledPython)) {
        $selectedPython = $bundledPython
        $sitePackages = Join-Path $projectRoot 'venv\Lib\site-packages'
        $env:PYTHONPATH = "$sitePackages;$projectRoot;$oldPythonPath"
        & $selectedPython -c 'import sys; assert sys.version_info[:2] == (3, 12)'
        if ($LASTEXITCODE -ne 0) { throw 'Fallback runtime requires Python 3.12.' }
    }
    if (-not $selectedPython) { throw 'Python 3.12 runtime missing.' }
    & $selectedPython -m training.artifact_spatial.run @TrainingArgs
    if ($LASTEXITCODE -ne 0) { throw "Artifact command exited with code $LASTEXITCODE. See console and training.log." }
}
finally {
    $env:PYTHONPATH = $oldPythonPath
    Pop-Location
}
