#!/usr/bin/env bash
# Roda `kapilab bench autocomplete` (apphost CPU publicado, somente leitura) nos pacotes CPU de T0 e T1 com 203 registros do val v2c
# (1 aquecimento + 203 medidos). Saidas no workspace C:\...\tmp\kapilab_ws\tmp (o KapiLab so grava em data/lab, reports/lab, tmp).
# uso: bash scripts/export/kapilab_bench_cpu.sh [t0:int4 t1:int8 ...]
set -u
ROOT=/c/IA/models/KapibaraStudio.MongoIA
K='F:\source\esilva-silva\EsilvaSoft.KapibaraStudio\artifacts\f7b\publish\Cpu\win-x64\EsilvaSoft.KapibaraStudio.KapiLab.exe'
WS='C:\IA\models\KapibaraStudio.MongoIA\tmp\kapilab_ws'
cd "$ROOT/tmp/kapilab_ws" || exit 1
LIST=${*:-"t0:int4 t0:int4-rtn t0:int4-kquantmixed t0:int8 t0:int8-emb4 t1:int8 t1:int4 t1:int4-rtn"}
for item in $LIST; do
  t=${item%%:*}; v=${item##*:}
  case $t in t0) fam=Nano;; t1) fam=Small;; esac
  pkg="C:\\IA\\models\\KapibaraStudio.MongoIA\\Repos\\KapiCoder-Mongo-$fam\\exports\\KapiCoder-Mongo-$fam-ONNX-$v"
  "$K" --workspace "$WS" bench autocomplete --package "$pkg" --in "tmp/records_${t}_203.jsonl" --device cpu --warmup 1 --iterations 1 \
       --out "tmp/bench_${t}_${v}_cpu.jsonl" > "$ROOT/logs/export/kapilab_bench_${t}_${v}.log" 2>&1
  echo "$t $v exit $?"
done
echo done
