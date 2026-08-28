[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeRoot = Join-Path $repoRoot "data\runtime"
$pidFiles = @(
    (Join-Path $runtimeRoot "demo\pids.json"),
    (Join-Path $runtimeRoot "cloud\pids.json")
)

function Test-LauncherProcessOwnership {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Entry
    )

    $processId = $Entry.process_id
    if ($null -eq $processId) { $processId = $Entry.pid }
    try { $processId = [int]$processId } catch { return $false }
    if ($processId -le 0) { return $false }

    $recordedPath = [string]$Entry.process_path
    $marker = [string]$Entry.command_line_marker
    $recordedStartText = [string]$Entry.start_time
    $recordedStartTicks = $Entry.start_time_ticks
    if (
        [string]::IsNullOrWhiteSpace($recordedPath) -or
        [string]::IsNullOrWhiteSpace($marker) -or
        [string]::IsNullOrWhiteSpace($recordedStartText) -or
        $null -eq $recordedStartTicks
    ) {
        return $false
    }

    try {
        $processInfo = @(
            Get-CimInstance `
                -ClassName Win32_Process `
                -Filter ("ProcessId = {0}" -f $processId) `
                -ErrorAction Stop
        )
        if ($processInfo.Count -ne 1) { return $false }
        $currentPath = [string]$processInfo[0].ExecutablePath
        $commandLine = [string]$processInfo[0].CommandLine
        if (
            [string]::IsNullOrWhiteSpace($currentPath) -or
            [string]::IsNullOrWhiteSpace($commandLine)
        ) {
            return $false
        }
        $currentPath = [IO.Path]::GetFullPath($currentPath)
        $recordedPath = [IO.Path]::GetFullPath($recordedPath)
        if (-not $currentPath.Equals($recordedPath, [StringComparison]::OrdinalIgnoreCase)) {
            return $false
        }
        if ($commandLine.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
            return $false
        }

        $process = Get-Process -Id $processId -ErrorAction Stop
        $recordedStart = [DateTime]::Parse(
            $recordedStartText,
            [Globalization.CultureInfo]::InvariantCulture,
            [Globalization.DateTimeStyles]::RoundtripKind
        ).ToUniversalTime()
        $currentStart = $process.StartTime.ToUniversalTime()
        if ($currentStart.Ticks -ne [int64]$recordedStartTicks) {
            return $false
        }
        return $true
    } catch {
        # Missing process metadata means ownership is unproven; never kill it.
        return $false
    }
}

$stopped = 0
$skipped = 0
foreach ($pidFile in $pidFiles) {
    if (-not (Test-Path -LiteralPath $pidFile -PathType Leaf)) {
        continue
    }
    try {
        $entries = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    } catch {
        $skipped += 1
        Remove-Item -LiteralPath $pidFile -Force
        continue
    }
    foreach ($entry in @($entries)) {
        $processId = $entry.process_id
        if ($null -eq $processId) { $processId = $entry.pid }
        try { $processId = [int]$processId } catch { $processId = 0 }
        if ($processId -le 0) {
            $skipped += 1
            continue
        }
        $owned = Test-LauncherProcessOwnership $entry
        if ($owned) {
            try {
                Stop-Process -Id $processId -Force -ErrorAction Stop
                $stopped += 1
            } catch [Microsoft.PowerShell.Commands.ProcessCommandException] {
                # The exact launcher-owned PID is already gone.
            }
        } else {
            $skipped += 1
        }
    }
    Remove-Item -LiteralPath $pidFile -Force
}

Write-Host "Stopped $stopped launcher-owned process(es); skipped $skipped unverified record(s)."
