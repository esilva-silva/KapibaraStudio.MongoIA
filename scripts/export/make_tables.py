#!/usr/bin/env python3
"""Imprime as tabelas markdown (tier x variante) do relatorio a partir de reports/export/results_<tier>.json, gates/<tier>/summary.json e kapilab_bench_summary.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from build_variants import GATES_DIR, V3, results_path  # noqa: E402

# bench do KapiLab so existe para a geracao legada (v3: KapiLab real nao rodado nesta etapa)
bench = {} if V3 else json.loads((ROOT / "reports/export/kapilab_bench_summary.json").read_text(encoding="utf-8"))
for tier in ("t0", "t1"):
    res = json.loads(results_path(tier).read_text(encoding="utf-8"))
    gates = json.loads((GATES_DIR / tier / "summary.json").read_text(encoding="utf-8"))
    hf = res["hf_bf16"]
    print(f"\n### {tier.upper()} (HF bf16 n=500 do mesmo recorte: FIM {hf['n500']['fim']:.3f}, exec {hf['n500']['exec']:.3f}, campos inexist. {hf['n500']['fields']:.3f}, sintaxe {hf['n500']['syntax']:.3f}; n=2000: {hf['n2000_reference']})\n")
    print("| variante | MB | tok.nativo=HF | greedy idêntico % | texto idêntico % | prefixo aceito % | FIM acc (Δpp; p) | exec-valid (Δpp; p) | campos inex. (Δpp) | sintaxe | TTFT méd/p95 ms | tok/s | RAM pico GB | VRAM MB | KapiLab TTFT méd/p95 | gates proxy |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for v, o in res["variants"].items():
        q, p, l = o["quality"], o["parity"], o.get("latency_proxy")
        g = gates.get(v, {})
        gs = "-"
        if "gates" in g:
            fails = [k.split(".")[-1] for k, x in g["gates"].items() if x["result"] == "failed"]
            oks = [k.split(".")[-1] for k, x in g["gates"].items() if x["result"] == "passed"]
            gs = ("FALHA: " + ",".join(fails) if fails else "") + (" | ok: " + ",".join(oks) if oks else "")
            if g.get("scenario_is_informative"):
                gs += " (cenário informativo)"
        b = bench.get(f"{tier}_{v}")
        print(f"| {v} | {o['size_mb']:.0f} | {o.get('tokenizer_identical_pct', '-')}% | {p['greedy_tokens_identical_pct']} | {p['text_identical_pct']} | {p['accepted_prefix_frac_mean_pct']} | "
              f"{q['fim_acc']:.3f} ({q['delta_pp_vs_hf']['fim_acc']:+.1f}; {q['mcnemar']['fim']['p']}) | {q['exec_valid']:.3f} ({q['delta_pp_vs_hf']['exec_valid']:+.1f}; {q['mcnemar']['exec']['p']}) | "
              f"{q['nonexistent_fields_rate']:.3f} ({q['delta_pp_vs_hf']['nonexistent_fields']:+.1f}) | {q['syntax_valid']:.3f} | "
              + (f"{l['ttft_mean_ms']:.0f}/{l['ttft_p95_ms']:.0f} | {l['decode_tok_s_forced64']} | {l['ram_peak_gb']} | {l.get('vram_dedicated_mb_7800xt', '-')} | " if l else "- | - | - | - | ")
              + (f"{b['ttft_mean_ms']:.0f}/{b['ttft_p95_ms']:.0f} | " if b else "- | ") + f"{gs} |")
