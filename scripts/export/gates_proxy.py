#!/usr/bin/env python3
"""Gera as medidas PROXY no formato do check_gates (owner kapilab) a partir de reports/export/results_<tier>.json e roda scripts/eval/check_gates.py.

  python -I scripts/export/gates_proxy.py --tier t0

Saidas: reports/export/gates/<tier>/<variante>.{kapilab,python}.json e <variante>.gates.json; resumo em reports/export/gates/<tier>/summary.json.
As medidas do KapiLab sao marcadas `proxy: true` e `qualified: false`: o check_gates devolve status `unqualified` e preserva o resultado cru em
`raw_result` (passed/failed) - NUNCA ha gate "qualificado" aqui; o gate oficial vem do KapiLab/harness da IDE em maquina ociosa.
Cenarios: T0 -> T0.cpu.* (variantes CPU); T1 -> T1.cpu_int8.* (CPU) e T1.dml_fp16.* (DML). Variantes sem gate proprio (ex.: T1 dml-int4,
T0 dml-*) sao avaliadas contra o cenario mais proximo APENAS como informacao (`scenario_is_informative`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "export"))
import check_gates  # noqa: E402
from build_variants import VARIANTS  # noqa: E402

GATE_TIER = {"t0": "T0", "t1": "T1"}


def scenario(tier: str, variant: str) -> tuple[str | None, bool]:
    prec, ep, _ = VARIANTS[variant]
    if tier == "t0":
        return ("T0.cpu", False) if ep == "cpu" and prec != "fp32" else ("T0.cpu", True)
    if ep == "dml":
        return "T1.dml_fp16", prec != "fp16"
    return "T1.cpu_int8", prec != "int8"


def kapilab_measures(tier: str, variant: str, o: dict, sc: str) -> dict:
    lat = o["latency_proxy"]
    m: dict = {}
    if sc == "T0.cpu":
        m.update({"T0.cpu.ttft_mean_ms": lat["ttft_mean_ms"], "T0.cpu.ttft_p95_ms": lat["ttft_p95_ms"], "T0.cpu.ram_peak_gb": lat["ram_peak_gb"],
                  "T0.cpu.disk_mb": o["size_mb"]})
    elif sc == "T1.cpu_int8":
        # o gate usa a metrica de CHAT (ttft_chat_*); aqui so ha prompts de autocomplete (~270 tokens) -> proxy imperfeito
        m.update({"T1.cpu_int8.ttft_mean_ms": lat["ttft_mean_ms"], "T1.cpu_int8.ttft_p95_ms": lat["ttft_p95_ms"], "T1.cpu_int8.ram_gb": lat["ram_peak_gb"]})
    else:
        m.update({"T1.dml_fp16.ttft_mean_ms": lat["ttft_mean_ms"], "T1.dml_fp16.ttft_p95_ms": lat["ttft_p95_ms"],
                  "T1.dml_fp16.decode_tok_s": lat["decode_tok_s_forced64"] or lat["decode_tok_s_stop_aware"]})
    return {"run_id": f"proxy-onnx-{tier}-{variant}", "proxy": True, "qualified": False,
            "env": {"used_fallback": False, "allow_gpu_parallel": False, "machine_load_pct": lat["machine_load_pct_before"],
                    "genai": lat.get("genai") or "0.15.2", "package_min_genai": "0.15.2"},
            "notes": ["PROXY do harness da IDE: onnxruntime-genai Python 0.15.2, ids pre-tokenizados pelo HF, 1 processo por pacote; NAO e o KapiLab/IDE"],
            "metrics": m}


def python_measures(tier: str, variant: str, o: dict) -> dict:
    prec, ep, _ = VARIANTS[variant]
    q, p = o["quality"], o["parity"]
    m: dict = {}
    if prec == "fp32":
        m["common.parity.fp32_cpu.greedy"] = p["greedy_tokens_identical_pct"]
    if prec == "fp16" and ep == "dml":
        m["common.parity.fp16_dml.greedy"] = p["greedy_tokens_identical_pct"]
    if prec == "int8":
        m["common.parity.int8_cpu.delta_apt"] = q["delta_apt_vs_hf"]["delta"]
    if prec == "int4":
        m["common.parity.int4.delta_apt"] = q["delta_apt_vs_hf"]["delta"]
        m["common.parity.int4.exec_match"] = q["delta_pp_vs_hf"]["exec_valid"]
    return {"run_id": f"proxy-onnx-{tier}-{variant}-parity", "env": {"genai": "0.15.2", "package_min_genai": "0.15.2"}, "metrics": m}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["t0", "t1"], required=True)
    a = ap.parse_args()
    res = json.loads((ROOT / "reports/export" / f"results_{a.tier}.json").read_text(encoding="utf-8"))
    out_dir = ROOT / "reports" / "export" / "gates" / a.tier
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for v, o in res["variants"].items():
        if "latency_proxy" not in o:
            continue
        if o["provider"] == "dml" and not o["latency_proxy"].get("on_7800xt"):
            summary[v] = {"invalid": "DML fora da RX 7800 XT - medida descartada"}
            continue
        sc, informative = scenario(a.tier, v)
        kl = kapilab_measures(a.tier, v, o, sc)
        py = python_measures(a.tier, v, o)
        pk, pp = out_dir / f"{v}.kapilab.json", out_dir / f"{v}.python.json"
        pk.write_text(json.dumps(kl, indent=2), encoding="utf-8")
        pp.write_text(json.dumps(py, indent=2), encoding="utf-8")
        doc = check_gates.run_check(GATE_TIER[a.tier], ROOT / "configs" / "gates.yaml", check_gates.load_measures(pp), check_gates.load_measures(pk),
                                    ("python", "kapilab"))
        (out_dir / f"{v}.gates.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        rows = {}
        for g in doc["gates"]:
            if g["status"] in ("inactive", "skipped", "not_applicable", "unevaluated"):
                continue
            if g["id"].startswith(sc + ".") or g["id"].startswith("common.parity."):
                rows[g["id"]] = {"value": g["value"], "op": g["op"], "threshold": g["threshold"], "result": g.get("raw_result") or g["status"]}
        summary[v] = {"scenario": sc, "scenario_is_informative": informative, "qualified": doc["qualified"], "qualified_reasons": doc["qualified_reasons"],
                      "gates": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for v, s in summary.items():
        if "gates" not in s:
            print(v, s)
            continue
        print(f"== {v} [{s['scenario']}{' (informativo)' if s['scenario_is_informative'] else ''}] qualified={s['qualified']}")
        for gid, g in s["gates"].items():
            print(f"   {gid:42s} {g['value']!s:>10} {g['op']} {g['threshold']!s:<6} -> {g['result']}")


if __name__ == "__main__":
    main()
