$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    $python=$null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidate = & py -3.12 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0) { $python=$candidate.Trim() }
    }
    if (-not $python -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $candidate = & python -c 'import sys; assert sys.version_info[:2] == (3,12); print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0) { $python=$candidate.Trim() }
    }
    if (-not $python) { throw 'Install Python 3.12 (64-bit) from python.org, then run setup-demo.cmd again.' }
    $environment = Join-Path $PSScriptRoot '.demo-venv'
    if (-not (Test-Path -LiteralPath (Join-Path $environment 'Scripts\python.exe'))) {
        & $python -m venv $environment
        if ($LASTEXITCODE -ne 0) { throw 'Could not create demonstration environment.' }
    }
    $python = Join-Path $environment 'Scripts\python.exe'
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip installation failed. Check internet access.' }
    & $python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed. Check internet access.' }
    & $python -m pip install -r (Join-Path $PSScriptRoot 'demo-requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    if (-not (Test-Path -LiteralPath 'demo_weights\manifest.json')) { throw 'Weights missing. Transfer the complete TRACE-demo-windows.zip, not just the Git source.' }
    & $python -c 'from demo.inference import Detector; print(Detector().describe())'
    if ($LASTEXITCODE -ne 0) { throw 'Weight validation failed.' }
    Write-Host 'Installation complete. Double-click start-demo.cmd.'
} finally { Pop-Location }
