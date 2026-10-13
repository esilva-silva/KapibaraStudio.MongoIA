<#
.SYNOPSIS
  Fila da GPU local (docs/fase-05-treinamento.md §4.2 e §9.6): um job por vez, lock em tmp/gpu.lock, watchdog e retomada.

.DESCRIPTION
  Le configs/gpu_queue.yaml (job, comando, dependencias, janela, max_restarts, watchdog). Detem tmp/gpu.lock (o filho recebe
  KAPI_GPU_QUEUE_PID e respeita o lock). Watchdog: se o processo morrer ou nenhum arquivo de `watch_files` avancar por
  `watchdog_minutes` (15), mata a arvore, espera a verificacao de ambiente passar e relanca (o comando ja traz --resume auto) ate
  `max_restarts` (3) vezes; na ultima falha o job vira `failed`, vai para reports/gpu_queue/incidents.md e a fila segue para o
  proximo job que nao dependa dele. Pausa cooperativa: -Pause cria tmp/gpu.pause (o treino salva checkpoint no passo seguinte e sai
  com codigo 75); -Resume remove o arquivo e a fila relanca.

.EXAMPLE
  .\scripts\gpu_queue.ps1 -Status
  .\scripts\gpu_queue.ps1 -Start -DryRun
  .\scripts\gpu_queue.ps1 -Start
  .\scripts\gpu_queue.ps1 -Start -Only J09-teacher-sft-v2c
  .\scripts\gpu_queue.ps1 -Pause ; .\scripts\gpu_queue.ps1 -Resume
#>
[CmdletBinding()]
param(
    [switch]$Start,
    [switch]$Status,
    [switch]$Pause,
    [switch]$Resume,
    [switch]$DryRun,
    [string]$Only = "",
    [string]$Config = "configs/gpu_queue.yaml"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Get-Prop($obj, [string]$name, $default = $null) {
    if ($null -ne $obj -and $obj.PSObject.Properties.Name -contains $name -and $null -ne $obj.$name) { return $obj.$name }
    return $default
}

function Read-Config {
    $py = if ($env:KAPI_PYTHON) { $env:KAPI_PYTHON } else { "C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe" }
    $cfgPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $Root $Config }
    $json = & $py -I -c "import yaml,json,sys;print(json.dumps(yaml.safe_load(open(sys.argv[1],encoding='utf-8').read())))" $cfgPath
    if ($LASTEXITCODE -ne 0) { throw "falha ao ler $Config com $py" }
    return ($json | ConvertFrom-Json)
}

$cfg = Read-Config
$Python = if ($env:KAPI_PYTHON) { $env:KAPI_PYTHON } else { $cfg.python }
$LockFile = Join-Path $Root $cfg.lock_file
$PauseFile = Join-Path $Root $cfg.pause_file
$StateFile = Join-Path $Root $cfg.state_file
$IncidentsFile = Join-Path $Root $cfg.incidents_file
$LogDir = Join-Path $Root $cfg.log_dir
$Poll = [int](Get-Prop $cfg "poll_seconds" 30)
$QueueDir = Join-Path $Root "tmp/gpu_queue"
foreach ($d in @($LogDir, $QueueDir, (Split-Path $StateFile), (Split-Path $LockFile), (Split-Path $IncidentsFile))) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

# ---------------------------------------------------------------- estado
function Get-State {
    if (Test-Path $StateFile) { return (Get-Content $StateFile -Raw | ConvertFrom-Json -AsHashtable) }
    return @{}
}
function Set-JobState([string]$id, [hashtable]$patch) {
    $state = Get-State
    if (-not $state.ContainsKey($id)) { $state[$id] = @{} }
    foreach ($k in $patch.Keys) { $state[$id][$k] = $patch[$k] }
    $state[$id]["updated"] = (Get-Date).ToString("s")
    $state | ConvertTo-Json -Depth 6 | Set-Content -Path $StateFile -Encoding utf8
}
function Get-JobStatus($job) {
    $state = Get-State
    if ($state.ContainsKey($job.id) -and $state[$job.id].ContainsKey("status")) { return $state[$job.id]["status"] }
    return $job.status
}
function Write-Incident([string]$jobId, [string]$kind, [string]$detail) {
    if (-not (Test-Path $IncidentsFile)) {
        "# Incidentes da fila da GPU`n`n| quando | job | tipo | detalhe |`n|---|---|---|---|" | Set-Content $IncidentsFile -Encoding utf8
    }
    $line = "| {0} | {1} | {2} | {3} |" -f (Get-Date).ToString("s"), $jobId, $kind, ($detail -replace "\|", "/" -replace "\r?\n", " ")
    Add-Content -Path $IncidentsFile -Value $line -Encoding utf8
    Write-Host "[incidente] $jobId $kind $detail" -ForegroundColor Yellow
    try { [System.Media.SystemSounds]::Exclamation.Play() } catch { }
    Set-Content -Path (Join-Path (Split-Path $IncidentsFile) "ALERT.txt") -Value "$((Get-Date).ToString('s')) $jobId $kind $detail" -Encoding utf8
}

# ---------------------------------------------------------------- utilitarios
function Test-Window([string]$spec) {
    if (-not $spec -or $spec -eq "any") { return $true }
    $a, $b = $spec.Split("-")
    $now = (Get-Date).TimeOfDay
    $winStart = [TimeSpan]::Parse($a); $winEnd = [TimeSpan]::Parse($b)
    if ($winStart -le $winEnd) { return ($now -ge $winStart -and $now -le $winEnd) }
    return ($now -ge $winStart -or $now -le $winEnd)
}
function Get-LatestWrite([string[]]$files, [datetime]$floor) {
    $latest = $floor
    foreach ($f in $files) {
        $p = Join-Path $Root $f
        if (Test-Path $p) { $t = (Get-Item $p).LastWriteTime; if ($t -gt $latest) { $latest = $t } }
    }
    return $latest
}
function Stop-Tree([int]$procId) {
    & taskkill.exe /F /T /PID $procId 2>&1 | Out-Null
}
function Test-EnvCheck {
    $retries = [int](Get-Prop $cfg "env_check_retries" 10)
    for ($i = 0; $i -lt $retries; $i++) {
        $out = & $Python @($cfg.env_check) 2>&1
        if ($LASTEXITCODE -eq 0) { return $true }
        Write-Host "env_check falhou ($($i + 1)/$retries): $($out | Select-Object -Last 1)"
        Start-Sleep -Seconds ($Poll * 2)
    }
    return $false
}
function Set-QueueLock([string]$jobId) {
    if (Test-Path $LockFile) {
        $holder = $null
        try { $holder = Get-Content $LockFile -Raw | ConvertFrom-Json } catch { }
        if ($holder -and $holder.pid -and (Get-Process -Id $holder.pid -ErrorAction SilentlyContinue) -and [int]$holder.pid -ne $PID) {
            throw "GPU ocupada: $LockFile detido por pid $($holder.pid) job '$($holder.job)'"
        }
    }
    @{ pid = $PID; job = $jobId; queue = $true; started = (Get-Date).ToString("s") } | ConvertTo-Json | Set-Content $LockFile -Encoding utf8
}
function Expand-Command($job) {
    return @($job.command | ForEach-Object { $_.Replace("{python}", $Python) })
}

# ---------------------------------------------------------------- execucao de um job
function Invoke-Job($job) {
    $id = $job.id
    $max = [int](Get-Prop $job "max_restarts" (Get-Prop $cfg.defaults "max_restarts" 3))
    $wd = [double](Get-Prop $job "watchdog_minutes" (Get-Prop $cfg.defaults "watchdog_minutes" 15))
    $window = [string](Get-Prop $job "window" (Get-Prop $cfg.defaults "window" "any"))
    $enforce = [bool](Get-Prop $job "window_enforce" (Get-Prop $cfg.defaults "window_enforce" $false))
    $watch = @(Get-Prop $job "watch_files" @())
    $pauseCodes = @(Get-Prop $job "pause_exit_codes" @(75))
    $argv = Expand-Command $job
    $exe = $argv[0]; $rest = @($argv | Select-Object -Skip 1)
    $log = Join-Path $LogDir "$id.log"
    $attempt = 0
    Set-JobState $id @{ status = "running"; started = (Get-Date).ToString("s"); restarts = 0 }
    while ($true) {
        while (-not (Test-Window $window)) { Write-Host "[$id] fora da janela $window; aguardando"; Start-Sleep -Seconds 60 }
        while (Test-Path $PauseFile) { Set-JobState $id @{ status = "paused" }; Write-Host "[$id] pausado (tmp/gpu.pause)"; Start-Sleep -Seconds 15 }
        if (-not (Test-EnvCheck)) { Write-Incident $id "env_check" "ambiente/GPU nao respondeu; job abandonado"; Set-JobState $id @{ status = "failed"; reason = "env_check" }; return }
        Set-QueueLock $id
        $env:KAPI_GPU_QUEUE_PID = "$PID"
        $env:KAPI_GPU_QUEUE_JOB = $id
        $launch = Get-Date
        Add-Content -Path $log -Value "=== $(Get-Date -Format s) lancamento (tentativa $($attempt + 1)): $exe $($rest -join ' ')" -Encoding utf8
        $proc = Start-Process -FilePath $exe -ArgumentList $rest -WorkingDirectory $Root -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err"
        Set-JobState $id @{ status = "running"; pid = $proc.Id; restarts = $attempt }
        $code = $null; $stalled = $false
        while ($true) {
            Start-Sleep -Seconds $Poll
            @{ t = (Get-Date).ToString("s"); job = $id; pid = $proc.Id; attempt = $attempt } | ConvertTo-Json | Set-Content (Join-Path $QueueDir "heartbeat.json") -Encoding utf8
            if ($proc.HasExited) { $code = $proc.ExitCode; break }
            if ($watch.Count -gt 0) {
                $last = Get-LatestWrite $watch $launch
                if (((Get-Date) - $last).TotalMinutes -gt $wd) {
                    Write-Incident $id "watchdog" "sem avanco ha $([int]((Get-Date) - $last).TotalMinutes) min (limite $wd); matando pid $($proc.Id)"
                    Stop-Tree $proc.Id; $stalled = $true; $code = -1; break
                }
            }
            if ($enforce -and -not (Test-Window $window) -and -not (Test-Path $PauseFile)) {
                New-Item -ItemType File -Force -Path $PauseFile | Out-Null
                Write-Host "[$id] janela encerrada: pausa cooperativa solicitada"
            }
        }
        foreach ($ext in @("out", "err")) { if (Test-Path "$log.$ext") { Add-Content -Path $log -Value (Get-Content "$log.$ext" -Raw) -Encoding utf8; Remove-Item "$log.$ext" -Force } }
        if ($code -eq 0) {
            $marker = Get-Prop $job "done_marker" $null
            if ($marker -and -not (Test-Path (Join-Path $Root $marker))) {
                Write-Incident $id "sem_marcador" "exit 0 mas $marker nao existe (parada por orcamento/--max-steps?)"
            }
            Set-JobState $id @{ status = "done"; finished = (Get-Date).ToString("s"); exit_code = 0 }
            return
        }
        if ($pauseCodes -contains $code) { Set-JobState $id @{ status = "paused"; exit_code = $code }; Write-Host "[$id] saiu pausado (codigo $code)"; continue }
        $attempt++
        if (-not $stalled) { Write-Incident $id "falha" "exit code $code (tentativa $attempt de $($max + 1))" }
        if ($attempt -gt $max) {
            Write-Incident $id "abandonado" "$max relancamentos esgotados; seguindo para o proximo job sem dependencia"
            Set-JobState $id @{ status = "failed"; finished = (Get-Date).ToString("s"); exit_code = $code }
            return
        }
        Write-Host "[$id] relancando com --resume auto ($attempt/$max)"
    }
}

# ---------------------------------------------------------------- comandos
if ($Pause) { New-Item -ItemType File -Force -Path $PauseFile | Out-Null; Write-Host "pausa solicitada: $PauseFile"; return }
if ($Resume) { Remove-Item $PauseFile -Force -ErrorAction SilentlyContinue; Write-Host "pausa removida"; return }

if ($Status -or -not $Start) {
    Write-Host "Fila da GPU ($Config) | lock: $(if (Test-Path $LockFile) { Get-Content $LockFile -Raw } else { 'livre' }) | pausa: $(Test-Path $PauseFile)"
    foreach ($job in $cfg.jobs) {
        $jobStatus = Get-JobStatus $job
        $extra = ""
        foreach ($w in @(Get-Prop $job "watch_files" @())) {
            if ($w -like "*metrics.jsonl" -and (Test-Path (Join-Path $Root $w))) {
                $extra = " | ultimo passo: " + ((Get-Content (Join-Path $Root $w) -Tail 1) -replace '^(.{0,160}).*$', '$1'); break
            }
        }
        Write-Host ("{0,-28} {1,-9} deps=[{2}]{3}" -f $job.id, $jobStatus, ((Get-Prop $job "depends_on" @()) -join ","), $extra)
    }
    if (-not $Start) { return }
}

# ---- -Start
$pending = @($cfg.jobs | Where-Object { -not $Only -or $_.id -eq $Only })
Write-Host "Iniciando fila ($($pending.Count) jobs candidatos); python=$Python"
$ranAny = $true
try {
    while ($ranAny) {
        $ranAny = $false
        foreach ($job in $pending) {
            $jobStatus = Get-JobStatus $job
            if ($jobStatus -notin @("ready", "waiting", "paused")) { continue }
            $deps = @(Get-Prop $job "depends_on" @())
            $unmet = @($deps | Where-Object { ((Get-State)[$_] -eq $null) -or ((Get-State)[$_]["status"] -ne "done") })
            if ($unmet.Count -gt 0) {
                $failedDep = @($deps | Where-Object { (Get-State).ContainsKey($_) -and (Get-State)[$_]["status"] -eq "failed" })
                if ($failedDep.Count -gt 0) { Set-JobState $job.id @{ status = "blocked"; reason = "dependencia falhou: $($failedDep -join ',')" } }
                continue
            }
            $missing = @(Get-Prop $job "requires_files" @() | Where-Object { -not (Test-Path (Join-Path $Root $_)) })
            if ($missing.Count -gt 0) { Write-Incident $job.id "pre_requisito" "arquivos ausentes: $($missing -join ', ')"; Set-JobState $job.id @{ status = "blocked"; reason = "requires_files" }; continue }
            if ($DryRun) {
                Write-Host "[dry-run] $($job.id): $((Expand-Command $job) -join ' ')"
                continue
            }
            Invoke-Job $job
            $ranAny = $true
        }
        if ($DryRun) { break }
    }
}
finally {
    if (-not $DryRun -and (Test-Path $LockFile)) {
        try { $h = Get-Content $LockFile -Raw | ConvertFrom-Json; if ([int]$h.pid -eq $PID) { Remove-Item $LockFile -Force } } catch { }
    }
}
Write-Host "Fila encerrada."
