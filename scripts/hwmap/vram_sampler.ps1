# Amostrador de GPU por processo: a cada ~1 s, memoria dedicada/compartilhada e utilizacao de engine por LUID para um pid. Uma linha JSON por amostra.
# uso: powershell -NoProfile -File vram_sampler.ps1 -ProcId <pid> -Seconds 15
param([int]$ProcId = 0, [int]$Seconds = 15)
$ErrorActionPreference = 'Continue'
$paths = @('\GPU Process Memory(*)\Dedicated Usage', '\GPU Process Memory(*)\Shared Usage', '\GPU Engine(*)\Utilization Percentage')
$end = (Get-Date).AddSeconds($Seconds)
while ((Get-Date) -lt $end) {
    $cs = (Get-Counter -Counter $paths -SampleInterval 1 -MaxSamples 1 -ErrorAction SilentlyContinue).CounterSamples
    $ded = @{}; $sh = @{}; $ut = @{}
    foreach ($s in $cs) {
        if ($s.InstanceName -notmatch "^pid_$ProcId`_luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)") { continue }
        $k = $Matches[1]
        if ($s.Path -like '*dedicated usage') { $ded[$k] = [math]::Round(($ded[$k] + 0) + $s.CookedValue / 1MB, 1) }
        elseif ($s.Path -like '*shared usage') { $sh[$k] = [math]::Round(($sh[$k] + 0) + $s.CookedValue / 1MB, 1) }
        elseif ($s.Path -like '*utilization percentage') { $ut[$k] = [math]::Round(($ut[$k] + 0) + $s.CookedValue, 1) }
    }
    @{ t = (Get-Date).ToString('o'); ded = $ded; sh = $sh; util = $ut } | ConvertTo-Json -Compress -Depth 4
}
