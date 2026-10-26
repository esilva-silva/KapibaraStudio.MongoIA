#!/usr/bin/env python3
"""Micro-benchmark de primeira execucao (docs/mapa-compatibilidade-gpu.md §6): prefill de ~270 tokens + 32 tokens de decodificacao.

STATUS: implementacao de referencia. As funcoes puras (`cache_key`, `evaluate`) sao testadas; o executor (`run_candidate`) NAO foi executado
nesta fase (restricao: outro agente mede o PC). Um processo por candidato; nao use o indice DXGI: confirme o adaptador por LUID/contadores
(scripts/export/gpu_probe.ps1) como o onnx_run.py --mode verify.

  python scripts/hwmap/microbench.py --pkg <pasta do pacote ONNX> --provider dml|cpu --tier t0|t1 --out saida.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED_TEXT = ('db.getCollection("orders").aggregate([{ $match: { status: "paid", createdAt: { $gte: ISODate("2026-01-01") } } },'
             ' { $group: { _id: "$customerId", total: { $sum: "$amount" } } }, { $sort: { total: -1 } }, { $limit: 10 }])\n')


def cache_key(adapter: dict, driver_version: str, genai_version: str, package_hash: str, windows_build: str) -> str:
    """Chave de cache: muda com GPU (vendor/device/subsys/revision), driver, GenAI, pacote ou build do Windows."""
    raw = "|".join(str(x) for x in (adapter.get("vendor_id"), adapter.get("device_id"), adapter.get("subsys_id"), adapter.get("revision"),
                                    driver_version, genai_version, package_hash, windows_build))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def evaluate(tier: str, gpu: dict, cpu: dict | None, cfg_mb: dict) -> dict:
    """gpu/cpu: {"ttft_ms_p50", "decode_tok_s", "ok"}. -> {"accept": bool, "reasons": [...]}  (criterios [E] do YAML)."""
    reasons = []
    if not gpu or not gpu.get("ok", True):
        return {"accept": False, "reasons": ["falha na execucao da GPU"]}
    ttft_max = cfg_mb["t1_accept_ttft_ms_p50_max"] if tier == "t1" else cfg_mb["accept_ttft_ms_p50_max"]
    tok_min = cfg_mb["t1_accept_decode_tok_s_min"] if tier == "t1" else cfg_mb["accept_decode_tok_s_min"]
    if gpu["ttft_ms_p50"] > ttft_max:
        reasons.append(f"TTFT {gpu['ttft_ms_p50']:.0f} ms > {ttft_max}")
    if gpu["decode_tok_s"] < tok_min:
        reasons.append(f"decode {gpu['decode_tok_s']:.1f} tok/s < {tok_min}")
    if cpu and cpu.get("ttft_ms_p50"):
        ratio = gpu["ttft_ms_p50"] / cpu["ttft_ms_p50"]
        if ratio > cfg_mb["accept_ttft_vs_cpu_ratio_max"]:
            reasons.append(f"TTFT GPU/CPU {ratio:.2f} > {cfg_mb['accept_ttft_vs_cpu_ratio_max']}")
    return {"accept": not reasons, "reasons": reasons or ["ok"]}


def run_candidate(pkg: str, provider: str, prompt_tokens: int = 270, decode_tokens: int = 32, repeats: int = 3) -> dict:
    import numpy as np
    import onnxruntime_genai as og
    t_load = time.perf_counter()
    cfg = og.Config(pkg)
    cfg.clear_providers()
    if provider != "cpu":
        cfg.append_provider(provider)  # adaptador: o GenAI-WinML escolhe; CONFIRMAR por contador/LUID (device_id nao e confiavel)
    model = og.Model(cfg)
    tok = og.Tokenizer(model)
    ids = []
    while len(ids) < prompt_tokens:
        ids += [int(x) for x in tok.encode(SEED_TEXT)]
    ids = ids[:prompt_tokens]
    load_s = time.perf_counter() - t_load
    ttfts, rates = [], []
    for k in range(repeats + 1):  # a 1a e aquecimento (descartada)
        p = og.GeneratorParams(model)
        p.set_search_options(do_sample=False, max_length=len(ids) + decode_tokens, min_length=len(ids) + decode_tokens, batch_size=1)
        g = og.Generator(model, p)
        t = time.perf_counter()
        g.append_tokens(np.array(ids, dtype=np.int32))
        g.generate_next_token()
        ttft = time.perf_counter() - t
        t1 = time.perf_counter()
        n = 1
        while n < decode_tokens and not g.is_done():
            g.generate_next_token()
            n += 1
        dec = time.perf_counter() - t1
        del g
        if k:
            ttfts.append(ttft * 1000)
            rates.append((n - 1) / dec if dec > 0 else 0.0)
    return {"ok": True, "provider": provider, "load_s": round(load_s, 2), "ttft_ms_p50": round(statistics.median(ttfts), 1),
            "ttft_ms_all": [round(x, 1) for x in ttfts], "decode_tok_s": round(statistics.median(rates), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--provider", choices=["cpu", "dml"], required=True)
    ap.add_argument("--tier", choices=["t0", "t1"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        res = run_candidate(a.pkg, a.provider)
    except Exception as e:  # noqa: BLE001  - qualquer falha => candidato reprovado, o orquestrador cai no CPU
        res = {"ok": False, "provider": a.provider, "error": f"{type(e).__name__}: {e}"[:400]}
    res["tier"] = a.tier
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res))
    return 0 if res.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
