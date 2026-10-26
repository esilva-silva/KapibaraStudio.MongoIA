#!/usr/bin/env python3
"""Orquestrador da fase 2 do mapa de GPU: executa as baterias A-E, UM processo por medida, com a porta de ociosidade da maquina.

Antes de cada medida mede a carga de CPU por 10 s (psutil) e so segue com <= 3%; senao espera e repete (ate `tries`), e depois segue marcando a medida
como `noisy` (NAO encerra processos do usuario). Cada medida grava reports/hwmap/phase2_raw/<nome>.json e uma linha em phase2_runs.jsonl
(carga observada, tentativas, ruidosa). Retomavel: pula o que ja existe. Repete ate 3 vezes quando a variacao entre execucoes passa de 10%.

  python scripts/hwmap/run_phase2.py --stage verify|lat-dml|lat-cpu|parity|ctx|micro|all
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "reports" / "hwmap" / "phase2_raw"
RUNS = ROOT / "reports" / "hwmap" / "phase2_runs.jsonl"
PY_CPU = ["C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe", "-I"]
PY_DML = ["C:/IA/models/SlopStudio.MongoAI-Compact/.venv-winml/Scripts/python.exe"]
NANO = ROOT / "Repos/KapiCoder-Mongo-Nano/exports"
SMALL = ROOT / "Repos/KapiCoder-Mongo-Small/exports"
PK = {
    "t0-dml-fp16": ("t0", NANO / "KapiCoder-Mongo-Nano-v3-ONNX-dml-fp16"),
    "t0-dml-fp32": ("t0", NANO / "KapiCoder-Mongo-Nano-v3-ONNX-dml-fp32-exp"),
    "t0-dml-fp16-mha": ("t0", NANO / "KapiCoder-Mongo-Nano-v3-ONNX-dml-fp16-mha-exp"),
    "t0-dml-int4": ("t0", NANO / "KapiCoder-Mongo-Nano-v3-ONNX-dml-int4-exp"),
    "t0-cpu-int4": ("t0", NANO / "KapiCoder-Mongo-Nano-v3-ONNX-int4"),
    "t1-dml-fp16": ("t1", SMALL / "KapiCoder-Mongo-Small-v3-ONNX-dml-fp16"),
    "t1-cpu-int8": ("t1", SMALL / "KapiCoder-Mongo-Small-v3-ONNX-int8"),
}
AD = {"7800xt": "1002:747E", "igpu": "1002:164E"}
IDLE_MAX = 3.0
_tries = [6]


def gate(label: str) -> dict:
    """carga de CPU em 10 s; <= 3% para prosseguir. Registra tudo."""
    import psutil
    obs = []
    for k in range(_tries[0]):
        v = psutil.cpu_percent(interval=10.0)
        obs.append(round(v, 1))
        if v <= IDLE_MAX:
            return {"idle_obs": obs, "noisy": False}
        print(f"[idle] {label}: {v}% > {IDLE_MAX}% (tentativa {k + 1})", flush=True)
        time.sleep(15)
    _tries[0] = 2   # a maquina nao fica abaixo do limite (apps do usuario): nao gastar horas esperando; registra ruidosa
    return {"idle_obs": obs, "noisy": True}


def log(rec: dict) -> None:
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def run(name: str, py: list[str], script: str, args: list[str], timeout: int = 7200, do_gate: bool = True) -> dict | None:
    out = RAW / f"{name}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    g = gate(name) if do_gate else {"idle_obs": [], "noisy": None}
    cmd = [*py, str(ROOT / "scripts/hwmap" / script), *args, "--out", str(out)]
    t = time.time()
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, timeout=timeout, capture_output=True, text=True)
    dur = time.time() - t
    rec = {"name": name, "rc": r.returncode, "wall_s": round(dur, 1), **g, "cmd": cmd[len(py):]}
    if r.returncode and not out.exists():
        rec["stderr_tail"] = (r.stderr or r.stdout)[-500:]
    log(rec)
    print(f"  rc={r.returncode} {dur:.0f}s noisy={g['noisy']} idle={g['idle_obs']}", flush=True)
    if out.exists():
        d = json.loads(out.read_text(encoding="utf-8"))
        d["_gate"] = g
        d["_wall_s"] = round(dur, 1)
        out.write_text(json.dumps(d), encoding="utf-8")
        return d
    return None


def bench(name: str, pk: str, mode: str, *, adapter: str | None = None, threads: int = 0, n: int = 100, forced: int = 20, extra: list[str] | None = None) -> dict | None:
    tier, pkg = PK[pk]
    dml = adapter is not None
    args = ["--pkg", str(pkg), "--provider", "dml" if dml else "cpu", "--tier", tier, "--mode", mode, "--n", str(n), "--forced", str(forced)]
    if dml:
        args += ["--adapter", AD[adapter]] + (["--allow-igpu"] if adapter == "igpu" else [])
    if threads:
        args += ["--threads", str(threads)]
    return run(name, PY_DML if dml else PY_CPU, "phase2_bench.py", args + (extra or []))


def metric(d: dict) -> tuple[float, float]:
    return d["ttft_mean"], d.get("dec_forced_mean") or d.get("dec_tps_mean") or 0.0


def with_repeats(base: str, pk: str, **kw) -> list[dict]:
    """r1, r2; se TTFT media ou tok/s diferirem > 10% entre elas, r3."""
    runs = [bench(f"{base}_r{i}", pk, "latency", **kw) for i in (1, 2)]
    runs = [r for r in runs if r]
    if len(runs) == 2:
        (t1, s1), (t2, s2) = metric(runs[0]), metric(runs[1])
        if abs(t1 - t2) / ((t1 + t2) / 2) > 0.10 or (s1 and s2 and abs(s1 - s2) / ((s1 + s2) / 2) > 0.10):
            r3 = bench(f"{base}_r3", pk, "latency", **kw)
            if r3:
                runs.append(r3)
    return runs


def stage_verify() -> None:
    for pk in ("t0-dml-fp16", "t1-dml-fp16", "t0-dml-fp32", "t0-dml-int4"):
        bench(f"verify_{pk}_7800xt", pk, "verify", adapter="7800xt")
    for pk in ("t0-dml-fp16", "t1-dml-fp16"):
        bench(f"verify_{pk}_igpu", pk, "verify", adapter="igpu")


def stage_lat_dml() -> None:
    for pk in ("t0-dml-fp16", "t1-dml-fp16", "t0-dml-fp32", "t0-dml-int4"):
        with_repeats(f"lat_{pk}_7800xt", pk, adapter="7800xt")
    with_repeats("lat_t0-dml-fp16_igpu", "t0-dml-fp16", adapter="igpu")
    with_repeats("lat_t1-dml-fp16_igpu", "t1-dml-fp16", adapter="igpu", extra=["--max-seconds", "1500"])


def stage_lat_cpu() -> None:
    for pk in ("t0-cpu-int4", "t1-cpu-int8"):
        for th in (24, 12, 16, 20, 8, 6, 4, 0):
            if pk == "t1-cpu-int8" and th in (4, 6):
                bench(f"lat_{pk}_th{th}_r1", pk, "latency", threads=th, forced=10)   # lentissimo: 1 execucao
                continue
            with_repeats(f"lat_{pk}_th{th}", pk, threads=th, forced=10 if pk.startswith("t1") else 20)


def stage_parity() -> None:
    for pk in ("t0-dml-fp16", "t0-dml-fp32", "t0-dml-int4"):
        bench(f"parity_{pk}_7800xt", pk, "parity", adapter="7800xt", n=100)


def stage_ctx() -> None:
    for pk in ("t0-dml-fp16", "t1-dml-fp16"):
        bench(f"ctx_{pk}_7800xt", pk, "ctx", adapter="7800xt")


def stage_exp() -> None:
    """dml-fp32 (MHA, sem graph capture) e controle dml-fp16-mha na 7800 XT: verify, paridade, latencia."""
    for pk in ("t0-dml-fp32", "t0-dml-fp16-mha"):
        bench(f"verify_{pk}_7800xt", pk, "verify", adapter="7800xt")
        bench(f"parity_{pk}_7800xt", pk, "parity", adapter="7800xt", n=100)
        with_repeats(f"lat_{pk}_7800xt", pk, adapter="7800xt", forced=10)


def stage_evidence() -> None:
    """re-execucao curta (n=20) de cada configuracao DML com os contadores ligados durante a medida (adapter_ok medido)."""
    for pk, ad in (("t0-dml-fp16", "7800xt"), ("t1-dml-fp16", "7800xt"), ("t0-dml-int4", "7800xt"), ("t0-dml-fp32", "7800xt"), ("t0-dml-fp16-mha", "7800xt"), ("t0-dml-fp16", "igpu"), ("t1-dml-fp16", "igpu")):
        sec = 200 if (pk.startswith("t1") and ad == "igpu") else (120 if ad == "igpu" else 60)
        bench(f"evid_lat_{pk}_{ad}", pk, "latency", adapter=ad, n=20, forced=3, extra=["--evidence", "--evidence-seconds", str(sec)])
    for pk in ("t0-dml-fp16", "t0-dml-int4", "t0-dml-fp32", "t0-dml-fp16-mha"):
        bench(f"evid_parity_{pk}_7800xt", pk, "parity", adapter="7800xt", n=20, extra=["--evidence", "--evidence-seconds", "30"])


def stage_micro() -> None:
    cands = [("t1-dml-fp16", "7800xt"), ("t0-dml-fp16", "7800xt"), ("t0-dml-fp16", "igpu"), ("t1-dml-fp16", "igpu"), ("t0-cpu-int4", None), ("t1-cpu-int8", None)]
    for pk, ad in cands:
        tier, pkg = PK[pk]
        for i in range(1, 11):
            name = f"micro_{pk}_{ad or 'cpu'}_{i:02d}"
            args = ["--pkg", str(pkg), "--provider", "dml" if ad else "cpu", "--tier", tier]
            if ad:
                args += ["--adapter", AD[ad]] + (["--allow-igpu"] if ad == "igpu" else [])
            run(name, PY_DML if ad else PY_CPU, "microbench.py", args)


STAGES = {"exp": stage_exp, "evidence": stage_evidence, "verify": stage_verify, "parity": stage_parity, "ctx": stage_ctx, "lat-dml": stage_lat_dml, "micro": stage_micro, "lat-cpu": stage_lat_cpu}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=[*STAGES, "all"])
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for s in (list(STAGES) if a.stage == "all" else [a.stage]):
        print("=== stage", s, flush=True)
        STAGES[s]()


if __name__ == "__main__":
    main()
