#!/usr/bin/env bash
# Bake-off Fase 01 — bancada FIM zero-shot (contrato editor-context-v1 da IDE, benchmark v1, 1.000 casos fim+completion).
# Students em bf16; candidatos a professor (DeepSeek 6.7B, Qwen 7B, Qwen 14B) em NF4 (7B/14B não cabem em bf16 nos 16 GB).
# Um modelo por vez na GPU; --resume permite retomar após interrupção.
set -u
PY=/c/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe
ROOT=/c/IA/models/KapibaraStudio.MongoIA
BASE=C:/IA/models/KapibaraStudio.MongoIA/BaseRepos
cd "$ROOT/bakeoff/v1"

run() {  # name model load
  echo "=== $(date '+%F %T') $1 ($3)"
  "$PY" -I scripts/benchmark.py run --backend hf --model "$2" --load "$3" --types fim completion --limit 1000 \
    --run-name "$1" --resume || echo "FAILED $1"
}

run qwen2.5-coder-0.5b-bf16  "$BASE/Qwen2.5-Coder-0.5B"        bf16
run qwen2.5-coder-1.5b-bf16  "$BASE/Qwen2.5-Coder-1.5B"        bf16
run deepseek-6.7b-nf4        "$BASE/deepseek-coder-6.7b-base"  nf4
run qwen2.5-coder-7b-nf4     "$BASE/Qwen2.5-Coder-7B"          nf4
run qwen2.5-coder-14b-nf4    "$BASE/Qwen2.5-Coder-14B"         nf4

"$PY" -I scripts/benchmark.py compare --runs qwen2.5-coder-0.5b-bf16 qwen2.5-coder-1.5b-bf16 deepseek-6.7b-nf4 qwen2.5-coder-7b-nf4 \
  qwen2.5-coder-14b-nf4 --report-dir "$ROOT/bakeoff/reports/2026-10-09-fim" --out "$ROOT/bakeoff/reports/2026-10-09-fim/comparison.md"
echo "=== $(date '+%F %T') done"
