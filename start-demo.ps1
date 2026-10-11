param([int]$Port = 8787, [ValidateSet('cpu','cuda')][string]$Device = 'cpu', [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = $null
$oldPythonPath = $env:PYTHONPATH
$oldPath = $env:PATH
$oldDevice = $env:TRACE_DEVICE
Push-Location $root
try {
    $candidates = @((Join-Path $root '.demo-venv\Scripts\python.exe'),
                    (Join-Path $root 'venv\Scripts\python.exe'))
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            try {
                & $candidate -c 'import torch, torchvision, fastapi, multipart, av' 2>$null
                if ($LASTEXITCODE -eq 0) { $python = $candidate; break }
            } catch { }
        }
    }
    $bundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (-not $python -and (Test-Path -LiteralPath $bundled) -and (Test-Path -LiteralPath (Join-Path $root 'venv\Lib\site-packages'))) {
        $python = $bundled
        $env:PYTHONPATH = "$(Join-Path $root 'venv\Lib\site-packages');$root;$oldPythonPath"
    }
    if (-not $python) { throw 'Run setup-demo.cmd first (Python 3.12 required).' }
    if (Test-Path -LiteralPath (Join-Path $root 'tools\ffmpeg.exe')) { $env:PATH = "$(Join-Path $root 'tools');$oldPath" }
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
        throw 'FFmpeg is missing. Use the portable ZIP, which includes FFmpeg.'
    }
    if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Port must be between 1024 and 65535.' }
    while (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        try {
            $health=Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 3
            if($health.models.spatial -and $health.models.temporal){
                Write-Host "TRACE is already running: http://127.0.0.1:$Port"
                if(-not $NoBrowser){Start-Process "http://127.0.0.1:$Port"}
                return
            }
        } catch { }
        $Port++
        if($Port -gt 65535){throw 'No free local port found.'}
    }
    $env:TRACE_DEVICE = $Device
    $url = "http://127.0.0.1:$Port"
    Write-Host "TRACE local demo: $url"
    Write-Host "Device: $Device. No training, datasets or external video uploads."
    Write-Host 'Stop the server with Ctrl+C.'
    if (-not $NoBrowser) { Start-Process $url }
    & $python -m uvicorn demo.app:app --host 127.0.0.1 --port $Port --log-level info
    if ($LASTEXITCODE -ne 0) { throw "Demo exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH=$oldPythonPath
    $env:PATH=$oldPath
    $env:TRACE_DEVICE=$oldDevice
    Pop-Location
}
