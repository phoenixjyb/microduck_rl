# Read-only Windows GPU snapshots. No idle/training permission is inferred.
[CmdletBinding()]
param(
    [ValidateRange(1, 5)][int]$SampleCount = 3,
    [ValidateRange(1, 10)][int]$IntervalSeconds = 2
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$duckSmi = Join-Path $env:WINDIR 'System32\nvidia-smi.exe'
if (-not (Test-Path -LiteralPath $duckSmi -PathType Leaf)) {
    throw 'Windows NVIDIA telemetry tool is unavailable.'
}
$duckSamples = @()
for ($duckIndex = 0; $duckIndex -lt $SampleCount; $duckIndex++) {
    $duckGpuRows = @(& $duckSmi --query-gpu=uuid,name,temperature.gpu,memory.total,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0 -or $duckGpuRows.Count -eq 0) {
        throw 'Windows NVIDIA GPU query failed.'
    }
    $duckComputeRows = @(& $duckSmi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0) { throw 'Windows compute-process query failed.' }
    $duckNames = @{}
    Get-Process | ForEach-Object { $duckNames[[int]$_.Id] = $_.ProcessName }
    $duckMemory = @()
    Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory |
        Where-Object DedicatedUsage -GT 8388608 |
        Sort-Object DedicatedUsage -Descending |
        Select-Object -First 8 |
        ForEach-Object {
            $duckMatch = [regex]::Match($_.Name, '^pid_(\d+)_')
            if (-not $duckMatch.Success) { throw 'Unexpected Windows GPU process instance.' }
            $duckProcId = [int]$duckMatch.Groups[1].Value
            $duckMemory += [ordered]@{
                instance = $_.Name
                pid = $duckProcId
                process_name = $duckNames[$duckProcId]
                dedicated_bytes = [long]$_.DedicatedUsage
                shared_bytes = [long]$_.SharedUsage
            }
        }
    $duckEngines = @(
        Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine |
            Where-Object UtilizationPercentage -GT 0 |
            Sort-Object UtilizationPercentage -Descending |
            Select-Object -First 8 Name, UtilizationPercentage
    )
    $duckSamples += [ordered]@{
        sampled_at = (Get-Date -Format o)
        gpu_csv = @($duckGpuRows | Select-Object -First 4)
        compute_process_csv = @($duckComputeRows | Select-Object -First 32)
        dedicated_memory_top8 = $duckMemory
        active_engines_top8 = $duckEngines
    }
    if ($duckIndex + 1 -lt $SampleCount) { Start-Sleep -Seconds $IntervalSeconds }
}
[ordered]@{
    protocol = 'microduck-windows-gpu-observation-oct8-v1'
    machine = $env:COMPUTERNAME
    samples = $duckSamples
    scope = 'Windows telemetry only; CIM engine instances may cover multiple adapters and snapshots are not simultaneous.'
    idle_gpu_proven = $false
    training_authorized = $false
    workloads_changed = $false
} | ConvertTo-Json -Depth 8 -Compress
