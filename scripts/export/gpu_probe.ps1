# Sonda de GPU por processo (Windows): memoria dedicada e utilizacao por adaptador (LUID). Saida JSON em stdout.
# uso: powershell -NoProfile -File gpu_probe.ps1 -ProcId <pid> [-Samples 2] [-Interval 1]
param([int]$ProcId = 0, [int]$Samples = 2, [int]$Interval = 1)
$ErrorActionPreference = 'Stop'
$adapters = @{}
(Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples | ForEach-Object {
    if ($_.InstanceName -match 'luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)') { $adapters[$Matches[1]] = [math]::Round($_.CookedValue / 1MB, 1) }
}
$proc = @{}
(Get-Counter '\GPU Process Memory(*)\Dedicated Usage').CounterSamples | ForEach-Object {
    if ($_.InstanceName -match "^pid_$ProcId`_luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)") {
        $k = $Matches[1]; $proc[$k] = [math]::Round(($proc[$k] + 0) + $_.CookedValue / 1MB, 1)
    }
}
$procShared = @{}
(Get-Counter '\GPU Process Memory(*)\Shared Usage').CounterSamples | ForEach-Object {
    if ($_.InstanceName -match "^pid_$ProcId`_luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)") {
        $k = $Matches[1]; $procShared[$k] = [math]::Round(($procShared[$k] + 0) + $_.CookedValue / 1MB, 1)
    }
}
$util = @{}
$cs = Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval $Interval -MaxSamples $Samples
$last = $cs | Select-Object -Last 1
$last.CounterSamples | ForEach-Object {
    if ($_.InstanceName -match "^pid_$ProcId`_luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)") {
        $k = $Matches[1]; $util[$k] = [math]::Round(($util[$k] + 0) + $_.CookedValue, 1)
    }
}
$allutil = @{}
$last.CounterSamples | ForEach-Object {
    if ($_.InstanceName -match 'luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)') { $k = $Matches[1]; $allutil[$k] = [math]::Round(($allutil[$k] + 0) + $_.CookedValue, 1) }
}
@{ adapters_dedicated_mb = $adapters; proc_dedicated_mb = $proc; proc_shared_mb = $procShared; proc_engine_util_pct = $util; system_engine_util_pct = $allutil } | ConvertTo-Json -Compress -Depth 4
