#!/usr/bin/env python3
"""Paridade e qualidade dos pacotes ONNX x HF bf16 do mesmo modelo, latencia (PROXY) e gates -> reports/export/results_<tier>.json.

  python -I scripts/export/compare_onnx.py --tier t0

Paridade (200 primeiros prompts): greedy token a token (lista de tokens identica, inclusive o token de parada), texto final identico apos
`postprocess`, primeiro token identico, prefixo aceito = fracao dos tokens do HF reproduzidos antes da 1a divergencia.
Qualidade (500 ids do val v2c): FIM acc, exec-valid (estrito), campos inexistentes, sintaxe do ONNX x as MESMAS linhas do HF bf16 (predictions.jsonl
do eval_student, n=2000) -> delta pareado e McNemar exato (scripts/eval/stats.py).
Latencia: onnx_run.py --mode latency (PROXY do harness da IDE; marcado qualified=false).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "teacher"))
sys.path.insert(0, str(ROOT / "bakeoff" / "v1" / "scripts"))
import stats  # noqa: E402
from build_variants import TIERS, VARIANTS, package_dir  # noqa: E402

WORK = ROOT / "reports" / "export" / "work"
HF_EVAL = {"t0": ROOT / "Repos/KapiCoder-Mongo-Nano/reports/eval/sft_distill_v1", "t1": ROOT / "Repos/KapiCoder-Mongo-Small/reports/eval/sft_distill_v1"}
HF_REF = {"t0": {"fim": 0.403, "exec": 0.582}, "t1": {"fim": 0.4105, "exec": 0.595}}


def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def strict_exec(r: dict) -> bool:
    return bool(r["exec"]["exec_valid"]) and not r["metrics"]["empty"]


def p95(v):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))] if v else None


def latency_summary(tier: str, variant: str) -> dict | None:
    f = WORK / f"{tier}_{variant}_latency.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    tt = [r["ttft_ms"] for r in d["rows"]]
    tps = [r["dec_tps"] for r in d["rows"] if r["dec_tps"]]
    out = {"n": len(tt), "ttft_mean_ms": round(st.mean(tt), 1), "ttft_median_ms": round(st.median(tt), 1), "ttft_p95_ms": round(p95(tt), 1),
           "ttft_max_ms": round(max(tt), 1), "prompt_tokens_median": st.median(r["prompt_tokens"] for r in d["rows"]),
           "prompt_tokens_mean": round(st.mean(r["prompt_tokens"] for r in d["rows"]), 1),
           "decode_tok_s_stop_aware": round(st.mean(tps), 1) if tps else None,
           "decode_tok_s_forced64": round(st.mean(d["decode_forced_tps"]), 1) if d["decode_forced_tps"] else None,
           "ram_peak_gb": d["peak_wset_gb"], "ram_private_gb": d["private_gb"], "ram_after_load_gb": d["rss_after_load_gb"], "load_s": d["load_s"],
           "machine_load_pct_before": d["machine_load_pct_before"], "machine_load_pct_after": d.get("machine_load_pct_after"), "genai": d.get("genai")}
    if d["provider"] == "dml":
        out.update(vram_dedicated_mb_7800xt=d.get("vram_dedicated_mb_7800xt"), vram_dedicated_mb_igpu=d.get("vram_dedicated_mb_igpu"),
                   on_7800xt=d.get("on_7800xt"), probe_end=d.get("probe_end"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS), required=True)
    a = ap.parse_args()
    tier = a.tier
    prompts = json.loads((WORK / f"{tier}_prompts.json").read_text(encoding="utf-8"))
    n_par = prompts["n_parity"]
    ids500 = [p["id"] for p in prompts["items"]]
    hf_tok = {p["id"]: p["hf_tokens"] for p in prompts["items"][:n_par]}
    hf_rows_all = {r["id"]: r for r in jl(HF_EVAL[tier] / "predictions.jsonl")}
    hf_rows = [hf_rows_all[i] for i in ids500]
    stops = set(prompts["stops"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(prompts["model"])
    from compact.contracts import postprocess
    import eval_teacher as et
    et.DATA_VERSION = "v2c"
    rows_src = {r["id"]: r for r in et.read_rows("editor-context-v1", "val")}

    def text_of(tokens, rid):
        kept = [t for t in tokens if t not in stops]
        return postprocess(tok.decode(kept, skip_special_tokens=False, clean_up_tokenization_spaces=False), rows_src[rid])

    # sanidade: HF regenerado (GPU) x predicoes armazenadas do eval_student
    san = sum(text_of(hf_tok[i], i) == hf_rows_all[i]["prediction"] for i in hf_tok) / len(hf_tok)
    hf_mean = {"fim": st.mean(r["metrics"]["normalized_match"] for r in hf_rows), "exec": st.mean(strict_exec(r) for r in hf_rows),
               "fields": st.mean(bool(r["exec"]["field_hallucination"]) for r in hf_rows), "syntax": st.mean(bool(r["metrics"]["syntax_valid"]) for r in hf_rows),
               "apt": st.mean(r["metrics"]["accepted_prefix_tokens"] for r in hf_rows)}
    res = {"tier": tier, "n_quality": len(ids500), "n_parity": n_par, "hf_bf16": {"n500": hf_mean, "n2000_reference": HF_REF[tier],
           "regenerated_text_equals_stored_pct": round(100 * san, 2)}, "variants": {}}
    for v in VARIANTS:
        pkg = package_dir(tier, v)
        qf = WORK / f"{tier}_{v}_quality.json"
        sf = ROOT / "reports/export/eval" / tier / v / "predictions.jsonl"
        if not (qf.exists() and sf.exists()):
            continue
        q = json.loads(qf.read_text(encoding="utf-8"))
        out: dict = {"provider": VARIANTS[v][1], "precision": VARIANTS[v][0], "size_mb": round(sum(p.stat().st_size for p in pkg.iterdir() if p.is_file()) / 2**20, 1),
                     "size_mb_ex_aux": round(sum(p.stat().st_size for p in pkg.iterdir() if p.name in ("model.onnx", "model.onnx.data")) / 2**20, 1)}
        tf = WORK / f"{tier}_{v}_tokenizer.json"
        if tf.exists():
            out["tokenizer_identical_pct"] = json.loads(tf.read_text(encoding="utf-8"))["identical_pct"]
        vf = WORK / f"{tier}_{v}_verify.json"
        if vf.exists():
            vd = json.loads(vf.read_text(encoding="utf-8"))
            out["dml_verify"] = {"on_7800xt": vd["on_7800xt"], "proc_dedicated_mb": vd["probe"].get("proc_dedicated_mb"), "proc_engine_util_pct": vd["probe"].get("proc_engine_util_pct")}
        items = {i["id"]: i for i in q["items"]}
        exact = text_eq = first = 0
        fracs = []
        for rid, ref in hf_tok.items():
            o = items[rid]["tokens"]
            exact += o == ref
            text_eq += text_of(o, rid) == text_of(ref, rid)
            first += o[:1] == ref[:1]
            k = 0
            for x, y in zip(o, ref):
                if x != y:
                    break
                k += 1
            fracs.append(k / len(ref))
        n = len(hf_tok)
        out["parity"] = {"greedy_tokens_identical_pct": round(100 * exact / n, 1), "text_identical_pct": round(100 * text_eq / n, 1),
                         "first_token_identical_pct": round(100 * first / n, 1), "accepted_prefix_frac_mean_pct": round(100 * st.mean(fracs), 1),
                         "padding_token_outputs": q.get("padding_token_outputs"), "n": n}
        rows = jl(sf)
        by = {r["id"]: r for r in rows}
        ids = [i for i in ids500 if i in by]
        mine = [by[i] for i in ids]
        hf_pair = [hf_rows_all[i] for i in ids]
        fim = [bool(r["metrics"]["normalized_match"]) for r in mine]
        fim_h = [bool(r["metrics"]["normalized_match"]) for r in hf_pair]
        ex = [strict_exec(r) for r in mine]
        ex_h = [strict_exec(r) for r in hf_pair]
        fld = [not r["exec"]["field_hallucination"] for r in mine]
        fld_h = [not r["exec"]["field_hallucination"] for r in hf_pair]
        syn = [bool(r["metrics"]["syntax_valid"]) for r in mine]
        syn_h = [bool(r["metrics"]["syntax_valid"]) for r in hf_pair]
        apt = [float(r["metrics"]["accepted_prefix_tokens"]) for r in mine]
        apt_h = [float(r["metrics"]["accepted_prefix_tokens"]) for r in hf_pair]
        mc = {k: stats.mcnemar_exact(h, o) for k, (h, o) in {"fim": (fim_h, fim), "exec": (ex_h, ex), "no_nonexistent_field": (fld_h, fld), "syntax": (syn_h, syn)}.items()}
        boot = stats.paired_bootstrap(apt_h, apt)
        out["quality"] = {"n": len(ids), "fim_acc": round(st.mean(fim), 4), "exec_valid": round(st.mean(ex), 4), "nonexistent_fields_rate": round(1 - st.mean(fld), 4),
                          "syntax_valid": round(st.mean(syn), 4), "apt_mean": round(st.mean(apt), 4),
                          "delta_pp_vs_hf": {"fim_acc": round(100 * (st.mean(fim) - st.mean(fim_h)), 2), "exec_valid": round(100 * (st.mean(ex) - st.mean(ex_h)), 2),
                                             "nonexistent_fields": round(100 * ((1 - st.mean(fld)) - (1 - st.mean(fld_h))), 2),
                                             "syntax_valid": round(100 * (st.mean(syn) - st.mean(syn_h)), 2)},
                          "delta_apt_vs_hf": {"delta": round(boot["delta"], 4), "ci_low": round(boot["ci_low"], 4), "ci_high": round(boot["ci_high"], 4)},
                          "mcnemar": {k: {"better": m["better"], "worse": m["worse"], "p": round(m["p"], 4)} for k, m in mc.items()}}
        lat = latency_summary(tier, v)
        if lat:
            out["latency_proxy"] = lat
        res["variants"][v] = out
    dest = ROOT / "reports" / "export" / f"results_{tier}.json"
    dest.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", dest)
    for v, o in res["variants"].items():
        q, p = o["quality"], o["parity"]
        print(f"{v:18s} {o['size_mb']:8.1f}MB par={p['greedy_tokens_identical_pct']:5.1f}% txt={p['text_identical_pct']:5.1f}% pref={p['accepted_prefix_frac_mean_pct']:5.1f}% "
              f"FIM={q['fim_acc']:.3f}({q['delta_pp_vs_hf']['fim_acc']:+.1f}pp p={q['mcnemar']['fim']['p']}) exec={q['exec_valid']:.3f}({q['delta_pp_vs_hf']['exec_valid']:+.1f}pp p={q['mcnemar']['exec']['p']})")


if __name__ == "__main__":
    main()
