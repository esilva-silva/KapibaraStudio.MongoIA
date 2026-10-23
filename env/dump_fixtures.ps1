<#
 Dump de cada banco kapi_fx_* via mongodump (WSL) para data\fixtures\dumps\<db>.archive.gz.
 Restaurar: wsl -d Ubuntu -e mongorestore --gzip --archive=<path /mnt/c/...> --drop
 Uso: powershell -File env\dump_fixtures.ps1 [-Distro Ubuntu] [-Uri mongodb://127.0.0.1:27017] [-SchemaPrefix v3_]
 -SchemaPrefix: so bancos cujo schema_id comeca com o prefixo (ex.: v3_ = lote v3, sem refazer os 110 antigos).
#>
param([string]$Distro = "Ubuntu", [string]$Uri = "mongodb://127.0.0.1:27017", [string]$SchemaPrefix = "")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "data\fixtures\dumps"
New-Item -ItemType Directory -Force $out | Out-Null
$manifest = Get-Content (Join-Path $root "data\fixtures\manifest.json") -Raw | ConvertFrom-Json
$ents = $manifest.entries
if ($SchemaPrefix) { $ents = $ents | Where-Object { $_.schema_id.StartsWith($SchemaPrefix) } }
$dbs = $ents | ForEach-Object { $_.db } | Sort-Object -Unique
foreach ($db in $dbs) {
    if ($db -notmatch '^kapi_fx_[a-z0-9_]+$') { throw "banco fora do prefixo kapi_fx_: $db" }
    $win = Join-Path $out "$db.archive.gz"
    $lin = (wsl -d $Distro -e wslpath -a ($win.Replace([string][char]92, '/'))).Trim()
    wsl -d $Distro -e mongodump --quiet --uri=$Uri --db=$db --gzip "--archive=$lin"
    if ($LASTEXITCODE -ne 0) { throw "mongodump falhou: $db" }
    "{0,-48} {1,10:N0} bytes" -f $db, (Get-Item $win).Length
}
