param([int]$Port = 8787, [ValidateSet('cpu','cuda')][string]$Device = 'cpu', [switch]$NoBrowser)
function Get-TraceInstanceAction {
    param($Health, [string]$ExpectedRevision, [bool]$OwnProcess, [bool]$ActiveJobs)
    if (-not $OwnProcess) { return 'other' }
    if ($Health.application_revision -eq $ExpectedRevision -and $Health.all_models_ready) { return 'reuse' }
    if ($ActiveJobs) { return 'busy' }
    return 'restart'
}

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = $null
$browserJob = $null
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
    & $python -B -m demo.check_runtime
    if ($LASTEXITCODE -ne 0) { throw 'Runtime validation failed. Read the mismatch above; restore matching tools/ and demo-requirements.txt.' }
    $revision = (& $python -B -c 'from demo.runtime import application_revision; print(application_revision())').Trim()
    if ($LASTEXITCODE -ne 0 -or -not $revision) { throw 'Could not determine TRACE application revision.' }
    while ($listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1) {
        $health = $null
        try { $health=Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 3 } catch { }
        $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" -ErrorAction SilentlyContinue
        $parent = $null
        if ($owner) { $parent=Get-CimInstance Win32_Process -Filter "ProcessId=$($owner.ParentProcessId)" -ErrorAction SilentlyContinue }
        $owned = $false
        if ($owner -and $owner.CommandLine -like '*uvicorn demo.app:app*') {
            foreach ($process in @($owner,$parent)) {
                if ($process -and $process.CommandLine -and $process.CommandLine.IndexOf($root+'\',[StringComparison]::OrdinalIgnoreCase) -ge 0) { $owned=$true }
            }
        }
        $active = [bool]$health.busy
        if ($owned -and -not $active) {
            Get-ChildItem -LiteralPath (Join-Path $root 'demo_runtime') -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^[a-f0-9]{32}$' } | ForEach-Object {
                $statusFile=Join-Path $_.FullName 'status.json'
                if (Test-Path -LiteralPath $statusFile) {
                    $state=Get-Content -LiteralPath $statusFile -Raw | ConvertFrom-Json
                    if ($state.status -in @('queued','running')) { $active=$true }
                }
            }
        }
        $action=Get-TraceInstanceAction -Health $health -ExpectedRevision $revision -OwnProcess $owned -ActiveJobs $active
        if ($action -eq 'reuse') {
            $url="http://127.0.0.1:$Port/?ui=$revision"
            Write-Host "TRACE is already running with current code: $url"
            if (-not $NoBrowser) { Start-Process $url }
            return
        }
        if ($action -eq 'busy') { throw 'TRACE has updated files, but analysis is active. Wait for it to finish, then run start-demo.cmd again.' }
        if ($action -eq 'restart') {
            Write-Host "TRACE files have changed. Restarting this project instance on port $Port."
            Stop-Process -Id $listener.OwningProcess -ErrorAction Stop
            for ($wait=0;$wait -lt 30;$wait++) {
                if (-not (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) { break }
                Start-Sleep -Milliseconds 100
            }
            if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw 'Old TRACE instance did not release the port. Close it and run start-demo.cmd again.' }
            continue
        }
        # A different checkout or another program owns the port; never stop it.
        $Port++
        if ($Port -gt 65535) { throw 'No free local port found.' }
    }
    $env:TRACE_DEVICE = $Device
    $url = "http://127.0.0.1:$Port/?ui=$revision"
    Write-Host "TRACE local demo: $url"
    Write-Host "Device: $Device. No training, datasets or external video uploads."
    Write-Host 'Stop the server with Ctrl+C.'
    if (-not $NoBrowser) {
        $browserJob=Start-Job -ArgumentList $url,$revision -ScriptBlock {
            param($Address,$ExpectedRevision)
            $base=($Address -split '\?')[0].TrimEnd('/')
            for ($attempt=0;$attempt -lt 90;$attempt++) {
                try {
                    $state=Invoke-RestMethod "$base/api/health" -TimeoutSec 10
                    if ($state.application_revision -eq $ExpectedRevision -and $state.status -eq 'ready') {
                        Start-Process $Address
                        return
                    }
                } catch { }
                Start-Sleep -Milliseconds 500
            }
        }
    }
    & $python -m uvicorn demo.app:app --host 127.0.0.1 --port $Port --log-level info
    if ($LASTEXITCODE -ne 0) { throw "Demo exited with code $LASTEXITCODE" }
} finally {
    if ($browserJob) {
        Stop-Job -Job $browserJob -ErrorAction SilentlyContinue
        Remove-Job -Job $browserJob -Force -ErrorAction SilentlyContinue
    }
    $env:PYTHONPATH=$oldPythonPath
    $env:PATH=$oldPath
    $env:TRACE_DEVICE=$oldDevice
    Pop-Location
}
