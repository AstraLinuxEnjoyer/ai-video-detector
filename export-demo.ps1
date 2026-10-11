$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
$old=$env:PYTHONPATH
try {
    $python=Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
    try { & $python -c 'import torch' 2>$null; if($LASTEXITCODE -ne 0){$python=$null} } catch {$python=$null}
    if(-not $python){
        $python=Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        $env:PYTHONPATH="$(Join-Path $PSScriptRoot 'venv\Lib\site-packages');$PSScriptRoot;$old"
    }
    & $python -m demo.package
    if($LASTEXITCODE -ne 0){throw 'Demo export failed.'}
} finally {$env:PYTHONPATH=$old;Pop-Location}
