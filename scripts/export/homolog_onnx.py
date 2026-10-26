#!/usr/bin/env python3
"""Qualidade dos pacotes ONNX na HOMOLOGACAO v1 (amostra de n itens) contra o HF bf16 do MESMO modelo, so pela API autorizada
(`homolog_guard.authorized` via scripts/eval/eval_homolog.py; cada leitura/escrita em data/homolog deixa linha em access.log.jsonl).

Fases (a amostra = os primeiros N itens na ordem sha1(id) de eval_homolog.load_items, a mesma do `--limit N`):
  prep    (venv ROCm, SO CPU)  : prompts (ids do contrato editor-context-v1) -> data/homolog/results/_onnx_prompts_<tier>.json   [uma vez por tier]
  run     (venv ROCm p/ CPU; venv WinML p/ DML) : geracao greedy <= 96 tokens no runtime ONNX -> data/homolog/results/onnx_<tier>_<variante>/tokens.json
  post    (venv ROCm)          : tokens -> predicoes no formato do eval_homolog (metricas de texto, sintaxe) -> .../predictions.jsonl
  (exec)  env/.venv            : scripts/eval/eval_homolog.py score --name onnx_<tier>_<variante>      (execucao do texto no Mongo RO; teto ja existe)
  compare (venv ROCm)          : agregados ONNX x HF (mesmos ids), Delta e McNemar exato -> reports/export/homolog_onnx_v3_<tier>.json (SO agregados)

Nada do texto dos itens sai de data/homolog/results: o relatorio em reports/ tem apenas contagens e taxas.
Variavel KAPI_EXPORT_GEN=v3 escolhe os pacotes v3. HF de referencia: execucoes `t0_v3` / `t1_v3` do eval_homolog (bf16 completo, 6.030 itens).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in ("scripts/export", "scripts/eval", "scripts/teacher", "scripts/data", "scripts/data/llm", "bakeoff/v1/scripts"):
    sys.path.insert(0, str(ROOT / _p))
import homolog_guard as hg  # noqa: E402
from build_variants import TIERS, V3, VARIANTS, package_dir  # noqa: E402

HF_NAME = {"t0": "t0_v3", "t1": "t1_v3"}
N_DEFAULT = 500
MAX_NEW = 96
OUT_REPORT = ROOT / "reports" / "export"


def tag(tier: str, variant: str) -> str:
    return f"onnx_{'v3_' if V3 else ''}{tier}_{variant}"


def eh_mod():
    import eval_homolog as eh
    return eh


def sample_items(eh, n: int, who: str) -> list[dict]:
    return eh.load_items(f"amostra de {n} itens para a paridade ONNX x HF ({who})", f"homolog_onnx:{who}", n)


def prompts_file(eh, tier: str) -> Path:
    return eh.RESULTS / f"_onnx_prompts_{tier}.json"


def phase_prep(a) -> int:
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"       # somente CPU: so tokeniza
    hg.enforce()
    eh = eh_mod()
    import eval_teacher as et
    from transformers import AutoTokenizer
    from compact.contracts import contract_for
    merged = ROOT / TIERS[a.tier]["merged"]
    tok = AutoTokenizer.from_pretrained(merged)
    contract = contract_for(tok)
    items = sample_items(eh, a.n, "prep")
    out, sanity = [], 0
    for r in items:
        ids = et.build_ids(contract, r, eh.CONTRACT)
        sanity += int(tok.decode(ids, skip_special_tokens=False) == r.get("prompt"))
        out.append({"id": r["id"], "prompt_ids": ids, "suffix": r["suffix"][:1024]})
    eh.write_json(prompts_file(eh, a.tier), {"tier": a.tier, "model": str(merged), "stops": sorted(contract.stops), "max_new": MAX_NEW, "n": len(out),
                                              "prompt_equals_sealed_prompt": sanity, "items": out})
    print(json.dumps({"n": len(out), "prompt_equals_sealed_prompt": sanity}))
    return 0


def phase_run(a) -> int:
    hg.enforce()
    eh = eh_mod()
    import onnx_run
    import psutil
    from transformers import AutoTokenizer
    P = eh.read_json(prompts_file(eh, a.tier))
    pkg = package_dir(a.tier, a.variant)
    prov = VARIANTS[a.variant][1]
    tok = AutoTokenizer.from_pretrained(P["model"])
    og, model = onnx_run.open_model(str(pkg), prov, None, None)
    runner = onnx_run.Runner(og, model, tok, P["stops"])
    res = {"pkg": str(pkg), "provider": prov, "genai": getattr(og, "__version__", ""), "pid": os.getpid(), "items": []}
    t0 = time.perf_counter()
    on_gpu = None
    for k, it in enumerate(P["items"]):
        r = runner.gen(it["prompt_ids"], P["max_new"], it["suffix"])
        res["items"].append({"id": it["id"], "tokens": r["tokens"], "stop": r["stop"], "ttft_ms": round(r["ttft"] * 1000, 1)})
        if prov == "dml" and k == 60:       # confirma a 7800 XT com contadores do Windows (nunca a iGPU)
            probe = onnx_run.gpu_probe(os.getpid(), samples=1)
            pd, pu = probe.get("proc_dedicated_mb", {}), probe.get("proc_engine_util_pct", {})
            on_gpu = bool(pd.get(onnx_run.RTX, 0) > 100 and pd.get(onnx_run.IGPU, 0) < 100)
            res["gpu_probe"] = {"proc_dedicated_mb": pd, "proc_engine_util_pct": pu, "on_7800xt": on_gpu}
            if not on_gpu:
                print("ERRO: DML nao esta na RX 7800 XT; abortando", file=sys.stderr)
                return 3
        if (k + 1) % 100 == 0:
            print(k + 1, f"{time.perf_counter() - t0:.0f}s", flush=True)
    res["seconds"] = round(time.perf_counter() - t0, 1)
    eh.write_json(eh.run_dir(tag(a.tier, a.variant)) / "tokens.json", res)
    print(json.dumps({"n": len(res["items"]), "seconds": res["seconds"], "on_7800xt": on_gpu}))
    return 0


def phase_post(a) -> int:
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    hg.enforce()
    eh = eh_mod()
    import eval_teacher as et
    from compact import syntax
    from compact.contracts import contract_for, ide_rejects, postprocess
    from compact.mongo_validate import validate
    from transformers import AutoTokenizer
    P = eh.read_json(prompts_file(eh, a.tier))
    run = eh.read_json(eh.run_dir(tag(a.tier, a.variant)) / "tokens.json")
    tok = AutoTokenizer.from_pretrained(P["model"])
    stops = contract_for(tok).stops
    by_id = {r["id"]: r for r in sample_items(eh, P["n"], "post")}
    plen = {p["id"]: len(p["prompt_ids"]) for p in P["items"]}
    rows = []
    for it in run["items"]:
        src = by_id[it["id"]]
        toks = it["tokens"]
        kept = [t for t in toks if t not in stops]
        pred = postprocess(tok.decode(kept, skip_special_tokens=False, clean_up_tokenization_spaces=False), src)
        ref = src["middle"]
        rows.append({"id": src["id"], "cut_kind": src["cut_kind"], "context_kind": src["context_kind"], "domain": src["domain"], "author": src["author"],
                     "source": src["source"], "dialect": src["dialect"], "kind": src.get("kind"), "schema_id": src["schema_id"],
                     "collection": src["collection"], "prompt_tokens": plen[src["id"]], "generated_tokens": len(toks), "stop": it["stop"],
                     "ttft_ms": it["ttft_ms"], "ide_rejected": ide_rejects(pred, it["stop"] in ("eos", "suffix")), "prediction": pred, "reference": ref,
                     "metrics": et.score_text(pred, ref, tok)})
    jobs = []
    for r in rows:
        src = by_id[r["id"]]
        for key, text in (("pred", r["prediction"]), ("ref", r["reference"])):
            jobs.append((f"{r['id']}|{key}", (src["prefix"] + text + src["suffix"]).replace("\r\n", "\n"), syntax.kind_for(src["dialect"])))
    sres = syntax.check_many(jobs)
    texts = {j[0]: j[1] for j in jobs}
    for r in rows:
        for key, target in (("pred", "metrics"), ("ref", "reference_checks")):
            job = f"{r['id']}|{key}"
            ok = sres[job][0]
            chk = validate(texts[job], r["dialect"]) if ok else None
            mv = chk.valid if chk and chk.validatable else (False if not ok else None)
            if key == "pred":
                r["metrics"]["syntax_valid"], r["metrics"]["mongo_valid"] = ok, mv
            else:
                r["reference_checks"] = {"syntax_valid": ok, "mongo_valid": mv}
    d = eh.run_dir(tag(a.tier, a.variant))
    eh.write_jsonl(d / "predictions.jsonl", rows)
    eh.write_json(d / "gen_meta.json", {"name": tag(a.tier, a.variant), "pkg": run["pkg"], "provider": run["provider"], "genai": run.get("genai"), "n": len(rows),
                                         "gpu_probe": run.get("gpu_probe"), "seconds": run.get("seconds"), "max_new_tokens": MAX_NEW,
                                         "sample": f"primeiros {len(rows)} itens por sha1(id) (= --limit {len(rows)})", "finished": time.strftime("%FT%T")})
    print(json.dumps({"n": len(rows)}))
    return 0


def phase_compare(a) -> int:
    hg.enforce()
    eh = eh_mod()
    import stats
    hf_rows_all = {r["id"]: r for r in eh.load_model_rows(HF_NAME[a.tier])}
    res: dict = {"tier": a.tier, "generation": "v3" if V3 else "v2", "hf_run": HF_NAME[a.tier], "sample": f"primeiros {a.n} itens por sha1(id) da homologacao v1",
                 "note": "so agregados; textos dos itens ficam em data/homolog/results", "variants": {}}
    hf_agg_done = False
    for v in VARIANTS:
        if not (eh.exists(eh.run_dir(tag(a.tier, v)) / "exec.jsonl") and eh.exists(eh.run_dir(tag(a.tier, v)) / "predictions.jsonl")):
            continue
        mine = {r["id"]: r for r in eh.load_model_rows(tag(a.tier, v))}
        ids = sorted(set(mine) & set(hf_rows_all))
        m = [mine[i] for i in ids]
        h = [hf_rows_all[i] for i in ids]
        if not hf_agg_done:
            res["hf_bf16"] = eh.aggregate(h)
            res["n"] = len(ids)
            hf_agg_done = True
        agg = eh.aggregate(m)
        mc = {}
        for k, (fh, fm) in {"fim": ([r["normalized_match"] for r in h], [r["normalized_match"] for r in m]),
                            "exec": ([r["exec_valid"] for r in h], [r["exec_valid"] for r in m]),
                            "no_nonexistent_field": ([not r["field_hallucination"] for r in h], [not r["field_hallucination"] for r in m]),
                            "syntax": ([r["syntax_valid"] for r in h], [r["syntax_valid"] for r in m])}.items():
            x = stats.mcnemar_exact(fh, fm)
            mc[k] = {"better": x["better"], "worse": x["worse"], "p": round(x["p"], 4)}
        hagg = res["hf_bf16"]
        d = lambda key: round(100 * ((agg[key] or 0) - (hagg[key] or 0)), 2)   # noqa: E731
        res["variants"][v] = {"n": len(ids), **{k: agg[k] for k in ("normalized_match", "exec_valid", "ceiling_exec_valid", "exec_over_ceiling",
                                                                    "nonexistent_fields_rate", "syntax_valid", "edit_similarity")},
                              "delta_pp_vs_hf": {"fim_acc": d("normalized_match"), "exec_valid": d("exec_valid"), "exec_over_ceiling": d("exec_over_ceiling"),
                                                 "nonexistent_fields": d("nonexistent_fields_rate"), "syntax_valid": d("syntax_valid")},
                              "mcnemar": mc}
    out = OUT_REPORT / f"homolog_onnx_{'v3_' if V3 else ''}{a.tier}.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}")
    h = res.get("hf_bf16", {})
    if h:
        print(f"HF bf16 n={res['n']}: FIM {h['normalized_match']:.3f} exec {h['exec_valid']:.3f} teto {h['ceiling_exec_valid']:.3f} exec/teto {h['exec_over_ceiling']} campos {h['nonexistent_fields_rate']:.3f}")
    for v, o in res["variants"].items():
        print(f"{v:12s} FIM {o['normalized_match']:.3f} ({o['delta_pp_vs_hf']['fim_acc']:+.1f}pp p={o['mcnemar']['fim']['p']}) exec {o['exec_valid']:.3f} "
              f"({o['delta_pp_vs_hf']['exec_valid']:+.1f}pp p={o['mcnemar']['exec']['p']}) exec/teto {o['exec_over_ceiling']} campos {o['nonexistent_fields_rate']:.3f} "
              f"({o['delta_pp_vs_hf']['nonexistent_fields']:+.1f}pp p={o['mcnemar']['no_nonexistent_field']['p']})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "run", "post", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--tier", choices=list(TIERS), required=True)
        p.add_argument("--n", type=int, default=N_DEFAULT)
        if name in ("run", "post"):
            p.add_argument("--variant", choices=list(VARIANTS), required=True)
    a = ap.parse_args()
    return {"prep": phase_prep, "run": phase_run, "post": phase_post, "compare": phase_compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
