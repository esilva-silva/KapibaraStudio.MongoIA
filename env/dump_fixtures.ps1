<#
 Dump de cada banco kapi_fx_* via mongodump (WSL) para data\fixtures\dumps\<db>.archive.gz.
 Restaurar: wsl -d Ubuntu -e mongorestore --gzip --archive=<path /mnt/c/...> --drop
 Uso: powershell -File env\dump_fixtures.ps1 [-Distro Ubuntu] [-Uri mongodb://127.0.0.1:27017]
#>
param([string]$Distro = "Ubuntu", [string]$Uri = "mongodb://127.0.0.1:27017")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "data\fixtures\dumps"
New-Item -ItemType Directory -Force $out | Out-Null
$manifest = Get-Content (Join-Path $root "data\fixtures\manifest.json") -Raw | ConvertFrom-Json
$dbs = $manifest.entries | ForEach-Object { $_.db } | Sort-Object -Unique
foreach ($db in $dbs) {
    if ($db -notmatch '^kapi_fx_[a-z0-9_]+$') { throw "banco fora do prefixo kapi_fx_: $db" }
    $win = Join-Path $out "$db.archive.gz"
    $lin = (wsl -d $Distro -e wslpath -a ($win.Replace([string][char]92, '/'))).Trim()
    wsl -d $Distro -e mongodump --quiet --uri=$Uri --db=$db --gzip "--archive=$lin"
    if ($LASTEXITCODE -ne 0) { throw "mongodump falhou: $db" }
    "{0,-48} {1,10:N0} bytes" -f $db, (Get-Item $win).Length
}
