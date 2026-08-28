[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backendRoot = Join-Path $repoRoot "backend"
$frontendRoot = Join-Path $repoRoot "frontend"
$runtimeRoot = Join-Path $repoRoot "data\runtime"
$modeRoot = Join-Path $runtimeRoot "demo"
$pidFile = Join-Path $modeRoot "pids.json"
$pythonPath = Join-Path $backendRoot ".venv\Scripts\python.exe"
$npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "backend virtual environment is missing; create backend\.venv first"
}
if ($null -eq $npmCommand) {
    throw "npm.cmd is required to start the frontend"
}

function New-LauncherProcessRecord {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$CommandLineMarker
    )

    $Process.Refresh()
    $processPath = $null
    try { $processPath = $Process.MainModule.FileName } catch { }
    if ([string]::IsNullOrWhiteSpace($processPath)) {
        try { $processPath = $Process.Path } catch { }
    }
    if ([string]::IsNullOrWhiteSpace($processPath)) {
        throw "could not record launcher process path"
    }
    try {
        $utcStart = $Process.StartTime.ToUniversalTime()
        $startTime = $utcStart.ToString(
            "o",
            [Globalization.CultureInfo]::InvariantCulture
        )
        $startTimeTicks = [int64]$utcStart.Ticks
    } catch {
        throw "could not record launcher process start time"
    }
    [ordered]@{
        name = $Name
        process_id = [int]$Process.Id
        process_path = [IO.Path]::GetFullPath($processPath)
        command_line_marker = $CommandLineMarker
        start_time = $startTime
        start_time_ticks = $startTimeTicks
    }
}

New-Item -ItemType Directory -Force -Path $modeRoot | Out-Null

if (Test-Path -LiteralPath $pidFile -PathType Leaf) {
    $existing = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    foreach ($entry in @($existing)) {
        $existingId = $entry.process_id
        if ($null -eq $existingId) { $existingId = $entry.pid }
        try { $existingId = [int]$existingId } catch { $existingId = 0 }
        if ($existingId -gt 0) {
            try {
                Get-Process -Id $existingId -ErrorAction Stop | Out-Null
                throw "a demo process is already recorded; run scripts\stop_local.ps1 first"
            } catch [Microsoft.PowerShell.Commands.ProcessCommandException] {
                # The recorded process is stale; only the exact PID file is removed.
            }
        }
    }
    Remove-Item -LiteralPath $pidFile -Force
}

# Relative paths are resolved by backend/app/settings.py from the backend
# directory.  Every mutable demo artifact is therefore below data/runtime/.
$env:APP_RUNTIME_MODE = "demo"
$env:SOURCE_DOCUMENTS_DIR = "../demo/documents"
$env:DEMO_QUESTIONS_PATH = "../demo/questions.json"
$env:PRIVATE_DATA_DIR = "../data/runtime/demo/private"
$env:QDRANT_PATH = "../data/runtime/demo/qdrant"
$env:SQLITE_PATH = "../data/runtime/demo/app.sqlite3"
$env:HOST = "127.0.0.1"
$env:PORT = "8000"

& $pythonPath (Join-Path $repoRoot "scripts\build_demo_documents.py")
if ($LASTEXITCODE -ne 0) {
    throw "fictional demo document generation failed"
}

$backendProcess = $null
$frontendProcess = $null
try {
    $backendCommandLineMarker = "app.main:app --host 127.0.0.1 --port 8000"
    $frontendCommandLineMarker = "run dev -- --host 127.0.0.1 --port 5173"
    $backendProcess = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $backendRoot `
        -PassThru `
        -WindowStyle Hidden
    $frontendProcess = Start-Process `
        -FilePath $npmCommand.Source `
        -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "5173") `
        -WorkingDirectory $frontendRoot `
        -PassThru `
        -WindowStyle Hidden
    @(
        (New-LauncherProcessRecord $backendProcess "backend" $backendCommandLineMarker)
        (New-LauncherProcessRecord $frontendProcess "frontend" $frontendCommandLineMarker)
    ) | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding utf8
} catch {
    foreach ($started in @($frontendProcess, $backendProcess)) {
        if ($null -ne $started) {
            try { Stop-Process -Id ([int]$started.Id) -Force -ErrorAction SilentlyContinue } catch { }
        }
    }
    throw "demo processes could not be started"
}

Write-Host "Demo started: backend http://127.0.0.1:8000, frontend http://127.0.0.1:5173"
