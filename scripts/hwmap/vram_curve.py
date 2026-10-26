#!/usr/bin/env python3
"""Curva VRAM de processo x comprimento do prompt (D1), 7800 XT, T0 e T1 dml-fp16 v3. UM processo por execucao.

Cada execucao: carrega o pacote (adaptador fixado por dxgi_select em 1002:747E), liga o amostrador de contadores do Windows
(vram_sampler.ps1: GPU Process Memory dedicada/compartilhada + GPU Engine por LUID, por pid) e gera repetidamente (prefill de L tokens +
32 tokens, max_length = L + 32; em L=2048 o prompt tem 2016 para o contexto TOTAL ser 2048) durante o tempo do amostrador. Pico = maior amostra de 1 s.
`adapter_ok` = pico > 100 MB no LUID pedido e <= 50 MB / <= 1% de engine no outro (mesmo criterio de phase2_bench.py).
Porta de ociosidade: CPU <= 3% em 10 s antes de cada execucao (run_phase2.gate); registra a carga e `noisy`.

  python scripts/hwmap/vram_curve.py            # orquestra tudo (retomavel; grava reports/hwmap/vram_curve_raw/*.json)
  python scripts/hwmap/vram_curve.py --child ...  # interno
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
sys.path.insert(0, str(ROOT / "scripts" / "hwmap"))
RAW = ROOT / "reports" / "hwmap" / "vram_curve_raw"
PY_DML = "C:/IA/models/SlopStudio.MongoAI-Compact/.venv-winml/Scripts/python.exe"
PKG = {"t0": ROOT / "Repos/KapiCoder-Mongo-Nano/exports/KapiCoder-Mongo-Nano-v3-ONNX-dml-fp16",
       "t1": ROOT / "Repos/KapiCoder-Mongo-Small/exports/KapiCoder-Mongo-Small-v3-ONNX-dml-fp16"}
LENGTHS = [270, 512, 768, 1024, 1536, 2048]
DECODE = 32
N = 10
ADAPTER = "1002:747E"


def prompt_len(L: int) -> int:
    return L if L + DECODE <= 2048 else 2048 - DECODE


def child(a) -> None:
    import numpy as np
    import onnxruntime_genai as og
    import psutil
    import dxgi_select
    from phase2_bench import Runner, resolve_adapter, start_sampler, summarize_sampler, probe
    prompts = json.loads((ROOT / "reports/export/work_v3" / f"{a.tier}_prompts.json").read_text(encoding="utf-8"))
    items = prompts["items"]
    ids = []
    for it in items:
        ids += it["prompt_ids"]
        if len(ids) >= 2100:
            break
    P = prompt_len(a.length)
    ids = ids[:P]
    adapter = resolve_adapter(ADAPTER)
    from detect_gpus import enumerate_dxgi
    allad = enumerate_dxgi()
    other = {x["luid"]: x["name"] for x in allad if x["luid"] != adapter["luid"]}
    res = {"tier": a.tier, "length_req": a.length, "prompt_tokens": P, "max_length": P + DECODE, "run": a.run, "pid": os.getpid(),
           "adapter": {k: adapter.get(k) for k in ("name", "luid", "vendor_id", "device_id", "dedicated_vram_mb")}, "other_luids": other,
           "machine_load_pct_before": psutil.cpu_percent(interval=1.0)}
    res["dxgi_hook"] = dxgi_select.install(*(int(x, 16) for x in ADAPTER.split(":")))
    cfg = og.Config(str(PKG[a.tier]))
    model = og.Model(cfg)
    runner = Runner(og, model, None, prompts["stops"])
    my = adapter["luid"]
    runner.gen(ids[:64], 4, "", force=True)   # aquece a sessao DML (nao conta; prompt curto)
    sam = start_sampler(os.getpid(), a.seconds)
    ttfts, k = [], 0
    t_end = time.time() + a.seconds - 1.5
    while time.time() < t_end or k < 3:
        r = runner.gen(ids, DECODE, "", force=True, max_len=P + DECODE)
        ttfts.append(r["ttft"] * 1000)
        k += 1
        if k >= 40:
            break
    pk = summarize_sampler(sam, timeout=a.seconds + 90)
    pm, pu = pk["proc_dedicated_mb_max"], pk["proc_engine_util_pct_max"]
    mine = pm.get(my, 0) or 0
    om = max([pm.get(l, 0) or 0 for l in other] + [0])
    ou = max([pu.get(l, 0) or 0 for l in other] + [0])
    res.update(gens=k, ttft_ms_median=round(st.median(ttfts), 1), ttft_ms_all=[round(x, 1) for x in ttfts[:12]],
               vram_dedicated_mb_peak=mine, vram_shared_mb_peak=pk["proc_shared_mb_max"].get(my), sampler_samples=pk["samples"],
               confirm={"mine_mb": mine, "others_mb_max": om, "mine_util": pu.get(my, 0), "others_util_max": ou},
               adapter_ok=bool(mine > 100 and om <= 50 and ou <= 1.0))
    mi = psutil.Process().memory_info()
    res.update(rss_gb=round(mi.rss / 2**30, 3), machine_load_pct_after=psutil.cpu_percent(interval=1.0))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("tier", "length_req", "run", "vram_dedicated_mb_peak", "ttft_ms_median", "adapter_ok", "gens")}))
    raise SystemExit(0 if res["adapter_ok"] else 3)


def orchestrate(a) -> None:
    import psutil

    def gate(label: str) -> dict:
        """CPU <= 3% em 10 s (2 tentativas curtas; o PC tem apps do usuario que nao sao encerrados): senao segue marcada `noisy`."""
        obs = []
        for _ in range(2):
            v = psutil.cpu_percent(interval=10.0)
            obs.append(round(v, 1))
            if v <= 3.0:
                return {"idle_obs": obs, "noisy": False}
            time.sleep(2)
        return {"idle_obs": obs, "noisy": True}

    RAW.mkdir(parents=True, exist_ok=True)
    for L in LENGTHS:
        for tier in ("t0", "t1"):
            for i in range(1, N + 1):
                out = RAW / f"{tier}_{L:04d}_{i:02d}.json"
                if out.exists():
                    continue
                g = gate(f"{tier}-{L}-{i}")
                cmd = [PY_DML, str(Path(__file__).resolve()), "--child", "--tier", tier, "--length", str(L), "--run", str(i), "--seconds", str(a.seconds), "--out", str(out)]
                t = time.time()
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
                if out.exists():
                    d = json.loads(out.read_text(encoding="utf-8"))
                    d["_gate"] = g
                    out.write_text(json.dumps(d), encoding="utf-8")
                print(f"{tier} L={L} run={i} rc={r.returncode} {time.time() - t:.0f}s idle={g['idle_obs']} :: {(r.stdout or r.stderr).strip()[-200:]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--tier")
    ap.add_argument("--length", type=int)
    ap.add_argument("--run", type=int)
    ap.add_argument("--seconds", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    child(a) if a.child else orchestrate(a)
