#!/usr/bin/env python3
"""Agrega reports/hwmap/vram_curve_raw/*.json em reports/hwmap/vram_curve.json e .md (curva VRAM de processo x comprimento do prompt, 7800 XT).

Tambem calcula o `context_cap_by_vram` proposto: para cada variante e cada contexto L medido, VRAM minima = MAIOR pico medido + margem (MARGIN_MB);
a VRAM de uma placa e a reportada pelo DXGI (8 GB -> ~8.176; 6 GB -> ~6.144; 4 GB -> ~4.096).
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "reports" / "hwmap" / "vram_curve_raw"
OUT = ROOT / "reports" / "hwmap"
MARGIN_MB = 400
TIERS = ("t0", "t1")


def load():
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RAW.glob("*.json"))]
    agg = {}
    for r in rows:
        agg.setdefault((r["tier"], r["length_req"]), []).append(r)
    return rows, agg


def summarize(agg):
    out = {}
    for (tier, L), rs in sorted(agg.items()):
        pk = [r["vram_dedicated_mb_peak"] for r in rs]
        out[f"{tier}:{L}"] = {
            "tier": tier, "length_req": L, "prompt_tokens": rs[0]["prompt_tokens"], "max_length": rs[0]["max_length"], "n": len(rs),
            "peak_mb": {"min": min(pk), "median": st.median(pk), "mean": round(st.mean(pk), 1), "max": max(pk), "sd": round(st.pstdev(pk), 2)},
            "shared_mb_max": max((r.get("vram_shared_mb_peak") or 0) for r in rs),
            "ttft_ms_median_of_runs": round(st.median(r["ttft_ms_median"] for r in rs), 1),
            "adapter_ok_all": all(r["adapter_ok"] for r in rs), "adapter_ok_n": sum(bool(r["adapter_ok"]) for r in rs),
            "others_mb_max": max(r["confirm"]["others_mb_max"] for r in rs), "others_util_max": max(r["confirm"]["others_util_max"] for r in rs),
            "noisy_n": sum(bool(r.get("_gate", {}).get("noisy")) for r in rs),
            "cpu_load_before_median": round(st.median(r["machine_load_pct_before"] for r in rs), 1),
            "rss_gb_median": round(st.median(r["rss_gb"] for r in rs), 2),
        }
    return out


def caps(summary):
    res = {}
    for tier in TIERS:
        pts = sorted((v for v in summary.values() if v["tier"] == tier), key=lambda v: v["length_req"])
        res[tier] = [{"context_cap_tokens": v["length_req"], "peak_max_mb": v["peak_mb"]["max"], "min_vram_mb": int(round((v["peak_mb"]["max"] + MARGIN_MB) / 50.0) * 50)} for v in pts]
    return res


def main() -> None:
    rows, agg = load()
    summary = summarize(agg)
    cp = caps(summary)
    (OUT / "vram_curve.json").write_text(json.dumps({"margin_mb": MARGIN_MB, "n_runs": len(rows), "summary": summary, "caps": cp, "runs": rows}, indent=1), encoding="utf-8")
    L = ["# Curva VRAM de processo x comprimento do prompt (RX 7800 XT, T0/T1 dml-fp16 v3)", "",
         f"Gerado por `scripts/hwmap/vram_curve.py` + `vram_curve_report.py` a partir de `reports/hwmap/vram_curve_raw/*.json` ({len(rows)} execucoes) em 2026-10-26. "
         "Cada execucao = **um processo novo**: carrega o pacote v3 `dml-fp16` com o adaptador fixado em 1002:747E (`dxgi_select`), liga o amostrador "
         "`vram_sampler.ps1` (contadores `GPU Process Memory`/`GPU Engine` por pid e por LUID, 1 amostra/s por ~8 s) e gera repetidamente (prefill de L tokens + 32 forcados, "
         "`max_length = L + 32`; em L = 2.048 o prompt tem 2.016 tokens para o contexto TOTAL ser 2.048; tokens dos prompts reais do val v2c concatenados). "
         "Pico = maior amostra de **memoria dedicada do processo** no LUID da 7800 XT. `adapter_ok` = pico > 100 MB no LUID pedido e <= 50 MB / <= 1% de engine na iGPU.", "",
         "## Resultado (n = 10 por ponto)", "",
         "| variante | prompt (tok) | contexto total (tok) | n | pico dedicado min / mediana / **max** (MB) | desvio (MB) | compart. max (MB) | TTFT mediano (ms) | adapter_ok | execucoes ruidosas |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for v in summary.values():
        p = v["peak_mb"]
        L.append(f"| {v['tier'].upper()} dml-fp16 | {v['prompt_tokens']} | {v['max_length']} | {v['n']} | {p['min']:.0f} / {p['median']:.0f} / **{p['max']:.0f}** | {p['sd']} | {v['shared_mb_max']:.0f} | "
                 f"{v['ttft_ms_median_of_runs']} | {v['adapter_ok_n']}/{v['n']} | {v['noisy_n']}/{v['n']} |")
    L += ["", "## Teto de contexto por VRAM (D1)", "",
          f"Regra: VRAM minima = **maior pico medido + {MARGIN_MB} MB** (driver, janela/compositor, fragmentacao), arredondada a 50 MB. Uma placa informa ao DXGI 8 GB ~ 8.176, 6 GB ~ 6.144, 4 GB ~ 4.096 MB.", "",
          "| teto de contexto (tokens) | T1 pico max (MB) | T1 VRAM min (MB) | T0 pico max (MB) | T0 VRAM min (MB) |", "|---|---|---|---|---|"]
    t1 = {c["context_cap_tokens"]: c for c in cp["t1"]}
    t0 = {c["context_cap_tokens"]: c for c in cp["t0"]}
    for cap in sorted(set(t1) | set(t0)):
        a, b = t1.get(cap), t0.get(cap)
        L.append(f"| {cap} | {a['peak_max_mb']:.0f} | {a['min_vram_mb']} | {b['peak_max_mb']:.0f} | {b['min_vram_mb']} |" if a and b else f"| {cap} | - | - | - | - |")
    L += ["", "## Observacoes", "",
          "- Carga de fundo: a regra era CPU <= 3% em 10 s antes de cada execucao; o PC tem apps do usuario (ChatGPT, claude, Taskmgr) que mantiveram 4-17%, entao a maioria das execucoes ficou marcada `noisy` "
          "(coluna acima). O pico de **memoria** e determinado pelo tamanho dos buffers do DML (graph capture + KV em `max_length`), nao pela carga de CPU: o desvio entre execucoes e de poucos MB.",
          "- O pico cresce com `max_length` porque o pacote usa `past_present_share_buffer` + graph capture (o KV e alocado no tamanho maximo pedido) e o prefill aloca buffers intermediarios proporcionais ao prompt.",
          "- A curva foi medida na 7800 XT (RDNA3, GenAI-WinML 0.15.2). A VRAM de processo depende do grafo/runtime, nao da potencia da GPU; em outras GPUs o valor deve ser da mesma ordem **[E]** ate a homologacao.",
          "- Reproduzir: `C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe scripts/hwmap/vram_curve.py` (retomavel) e `python scripts/hwmap/vram_curve_report.py`.", ""]
    (OUT / "vram_curve.md").write_text("\n".join(L), encoding="utf-8")
    print(json.dumps(cp))


if __name__ == "__main__":
    main()
