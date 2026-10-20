#!/usr/bin/env python3
"""Orquestra as medidas de todos os pacotes de um tier (um processo por pacote). Fases:
  package  : package_variant.py (tokenizer byte a byte, metadata kapi, manifestos)
  quality  : tokenizer nativo x HF, 500 prompts greedy (ONNX) + score (eval_teacher: FIM acc, exec-valid, campos, sintaxe)
  latency  : PROXY do harness da IDE (maquina ociosa; DML verificado na RX 7800 XT por contadores; aborta se detectar a iGPU)
  compare  : paridade x HF bf16 (200 prompts), delta pareado/McNemar nos 500, tabela JSON em reports/export/results_<tier>.json

  python -I scripts/export/measure_all.py --tier t0 --phase quality [--variant dml-fp16 ...] [--force]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from build_variants import TIERS, VARIANTS, package_dir  # noqa: E402

PY_ROCM = "C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe"
PY_WINML = "C:/IA/models/SlopStudio.MongoAI-Compact/.venv-winml/Scripts/python.exe"
WORK = ROOT / "reports" / "export" / "work"
IDLE_WAIT_S = 180


def run_py(provider: str, script: str, *args: str) -> int:
    cmd = ([PY_ROCM, "-I"] if provider == "cpu" else [PY_WINML]) + [str(ROOT / "scripts/export" / script), *args]   # WinML: sem -I (sitecustomize)
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def wait_idle(max_s: int = 180, limit: float = 5.0) -> float:
    import psutil
    load = 100.0
    t = time.time()
    while time.time() - t < max_s:
        load = psutil.cpu_percent(interval=8.0)
        if load <= limit:
            return load
        print(f"[idle] carga {load}% > {limit}%, aguardando", flush=True)
        time.sleep(10)
    return load


def quality(tier: str, variant: str, force: bool) -> None:
    pkg = package_dir(tier, variant)
    ep = VARIANTS[variant][1]
    q = WORK / f"{tier}_{variant}_quality.json"
    if force or not q.exists():
        run_py(ep, "onnx_run.py", "--pkg", str(pkg), "--provider", ep, "--tier", tier, "--mode", "tokenizer", "--out", str(WORK / f"{tier}_{variant}_tokenizer.json"))
        if ep == "dml":
            rc = run_py(ep, "onnx_run.py", "--pkg", str(pkg), "--provider", ep, "--tier", tier, "--mode", "verify", "--out", str(WORK / f"{tier}_{variant}_verify.json"))
            if rc:
                raise SystemExit(f"{variant}: DML nao esta na RX 7800 XT (rc={rc}); abortado")
        rc = run_py(ep, "onnx_run.py", "--pkg", str(pkg), "--provider", ep, "--tier", tier, "--mode", "quality", "--out", str(q))
        if rc:
            raise SystemExit(f"{variant}: quality falhou rc={rc}")
    if force or not (ROOT / "reports/export/eval" / tier / variant / "summary.json").exists():
        rc = run_py("cpu", "score_onnx.py", "--tier", tier, "--variant", variant, "--run", str(q))
        if rc:
            raise SystemExit(f"{variant}: score falhou rc={rc}")


def latency(tier: str, variant: str, force: bool, n: int) -> None:
    pkg = package_dir(tier, variant)
    ep = VARIANTS[variant][1]
    out = WORK / f"{tier}_{variant}_latency.json"
    if out.exists() and not force:
        return
    load = wait_idle(max_s=IDLE_WAIT_S)
    print(f"[idle] carga antes: {load}%", flush=True)
    rc = run_py(ep, "onnx_run.py", "--pkg", str(pkg), "--provider", ep, "--tier", tier, "--mode", "latency", "--n", str(n), "--out", str(out))
    if rc:
        print(f"!! {variant}: latency rc={rc} (DML fora da 7800 XT => medida invalida)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS), required=True)
    ap.add_argument("--phase", choices=["package", "quality", "latency"], required=True)
    ap.add_argument("--variant", action="append", choices=list(VARIANTS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--idle-wait", type=int, default=180, help="segundos esperando CPU <= 5%% antes de cada medida de latencia")
    a = ap.parse_args()
    global IDLE_WAIT_S
    IDLE_WAIT_S = a.idle_wait
    variants = a.variant or [v for v in VARIANTS if (package_dir(a.tier, v) / "genai_config.json").exists()]
    WORK.mkdir(parents=True, exist_ok=True)
    for v in variants:
        if a.phase == "package":
            run_py("cpu", "package_variant.py", "--tier", a.tier, "--variant", v)
        elif a.phase == "quality":
            quality(a.tier, v, a.force)
        else:
            latency(a.tier, v, a.force, a.n)


if __name__ == "__main__":
    main()
