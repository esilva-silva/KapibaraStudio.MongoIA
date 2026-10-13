#!/usr/bin/env bash
# Baseline zero-shot FIM (Fase 02 §3.2): 1 modelo por vez na GPU (tmp/gpu.lock). Venv ROCm do bake-off.
set -u
PY=/c/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe
cd "$(dirname "$0")/../.."
for m in Qwen2.5-Coder-0.5B Qwen2.5-Coder-1.5B Qwen2.5-Coder-7B deepseek-coder-6.7b-base; do
  echo "=== $(date '+%F %T') $m"
  timeout 10800 "$PY" -I scripts/teacher/eval_teacher.py --model "BaseRepos/$m" --split val --contract fim-raw --n 1000 --resume \
    || echo "FAILED $m"
done
echo "=== $(date '+%F %T') done"
