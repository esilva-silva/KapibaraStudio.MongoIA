#!/usr/bin/env python3
"""Dados de GRAMATICA dos schemas v3 (papel `train` -> treino; papel `eval` -> so val/test/bench de avaliacao) com a pipeline de 9 etapas.

  env/.venv/Scripts/python.exe scripts/data/llm/run_guarded.py scripts/data/build_grammar_v3.py --role eval  --out data/generated/v3/grammar
  env/.venv/Scripts/python.exe scripts/data/llm/run_guarded.py scripts/data/build_grammar_v3.py --role train --out data/generated/v3/grammar

Fluxo (cada etapa e retomavel: pula se a saida existe; `--force` refaz):
  0 stage      copia os schemas v3 do papel pedido para <out>/_schemas_<role>/ (assert: nenhum `hom`)
  1 snippets   gen_snippets.py --schemas-dir <stage> --seeded-only (seed fixa; split vem do papel: train->train; eval->val/test/bench)
  2 exec       validate_snippets.py (execucao somente-leitura em kapi_fx_*)  -> taxa exec-valid dos snippets
  3 prefilter  snippets que seriam removidos pela decontaminacao OFICIAL (ngram_min_content 2) contra val/test/bench v2c, v3-eval e o
               conjunto de homologacao selado (leitura SO pela API autorizada de llm_to_fim.homolog_docs; log em access.log.jsonl)
  4 fim        gen_fim.py (cortes, 9 cut_kinds, 96 tokens Qwen) com config derivada (sem eval_counts; categorias com mais atlas/json/dsl)
  5 contexto   editor-context-v1 com context_kind CONSERVADOR (>= 25% none/irrelevant so onde field_need <= 1, como a v2d e o gerador LLM)
  6 pipeline   validate_pipeline.py (9 etapas) com decontaminacao etapa 9 tambem contra v2c val/test/bench, v3-eval e homologacao
Saidas: <out>/{snippets,validation,fim,fim_ctx}*.jsonl, data/validated/v3g[/eval]/, data/splits/v3g[_eval]/, <out>/<role>.funnel.json.
Rode SEMPRE via run_guarded.py (bloqueia leitura de data/homolog fora da API autorizada). Nunca usa --include-hom.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, HERE / "llm"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import yaml  # noqa: E402

import homolog_guard as hg  # noqa: E402
import hom_guard  # noqa: E402

hg.enforce()          # idempotente; a leitura de data/homolog so passa por hg.authorized (llm_to_fim.homolog_docs)

import blind_guard  # noqa: E402
import decontaminate as dc  # noqa: E402
import gen_editor_context as gec  # noqa: E402
import gen_fim as gf  # noqa: E402
import gen_snippets as gs  # noqa: E402
import par  # noqa: E402
import render_contract as rc  # noqa: E402
import validate_pipeline as vp  # noqa: E402
import validate_snippets as vsn  # noqa: E402
import llm_to_fim as l2f  # noqa: E402
import homolog_refs  # noqa: E402
from jsonl_shards import iter_jsonl  # noqa: E402

SEEDS = {"train": 20261024, "eval": 20261025}      # fixas (documentadas no relatorio)
RAW_SNIPPETS = {"train": 150000, "eval": 24000}
CUTS = {"train": 152000, "eval": 15000}
SOURCE_TAG = "grammar-v3"
CONTEXT_SHARE = 0.31                                  # none+irrelevant alvo (>= 25% exigido; so cortes com field_need <= 1)
CONTEXT_RATIOS = {"none": 0.5, "irrelevant": 0.5}


def jwrite(path: Path, rows) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def derived_config(role: str, out: Path) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "dataset.yaml").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["version"] = "v3g"
    cfg["seed"] = SEEDS[role]
    # mais cobertura estrutural: atlas_search / json_ejson / dsl / indexes (as categorias que o LLM nao cobre)
    cfg["snippets"]["extra_categories"] = {"atlas_search": 0.12, "json_ejson": 0.20, "dsl": 0.14, "indexes": 0.06}
    fim = cfg["fim"]
    fim.pop("eval_counts", None)                      # amostragem unica; o split vem do papel do schema
    fim["decontam_prefilter"] = False                 # o prefiltro e feito aqui (etapa 3) com v2c + v3-eval + homologacao
    fim["max_cuts_per_snippet"] = 5
    fim["category_targets"] = {"aggregation": 0.18, "crud": 0.10, "operators": 0.10, "atlas_search": 0.20, "json_ejson": 0.18,
                               "dsl": 0.16, "indexes": 0.08}
    fim["category_survival"] = {k: 1.0 for k in fim["category_targets"]}
    p = out / f"dataset_v3g_{role}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def stage_schemas(role: str, out: Path) -> Path:
    dst = out / f"_schemas_{role}"
    if dst.exists():
        shutil.rmtree(dst)
    n = 0
    for p in sorted((ROOT / "data" / "schemas").glob("*/v3_*.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        r = raw.get("split_role")
        if r != role:
            continue
        if hom_guard.is_hom_schema(raw.get("schema_id") or p.stem, raw.get("domain") or p.parent.name, r):
            raise SystemExit(f"[hom-guard] schema hom no stage: {p}")
        (dst / p.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / p.parent.name / p.name)
        n += 1
    if not n:
        raise SystemExit(f"nenhum schema v3 com papel {role}")
    return dst


def eval_references(role: str, out: Path) -> list[tuple[str, dict]]:
    """Documentos de avaliacao contra os quais o treino e decontaminado: v2c val/test/bench, homologacao selada (API autorizada)
    e, no treino, os cortes v3-eval ja gerados."""
    ev = l2f.eval_docs_v2c()
    ev = list(ev) + homolog_refs.homolog_docs()
    if role == "train":
        p = out / "_eval_docs_v3g.jsonl"
        if p.exists():
            for r in iter_jsonl(p):
                ev.append((r["split"], r))
    return ev


def run_step_snippets(role, out, stage, cfgp, count, force):
    snip = out / f"snippets_{role}.raw.jsonl"
    if force or not snip.exists():
        t = time.time()
        gs.main(["--config", str(cfgp), "--schemas-dir", str(stage), "--seed", str(SEEDS[role]), "--count", str(count), "--out", str(snip),
                 "--seeded-only"])
        print(f"[1] snippets {role}: {time.time() - t:.0f}s", flush=True)
    return snip


def run_step_exec(role, out, snip, cfgp, workers, force):
    val = out / f"validation_{role}.jsonl"
    if force or not val.exists():
        t = time.time()
        vsn.main(["--config", str(cfgp), "--snippets", str(snip), "--out", str(val), "--workers", str(workers)])
        sm = val.parent / "validation.summary.json"
        if sm.exists():
            shutil.move(str(sm), str(val.parent / f"validation_{role}.summary.json"))
        print(f"[2] exec snippets {role}: {time.time() - t:.0f}s", flush=True)
    c = Counter()
    for r in iter_jsonl(val):
        c[r.get("exec")] += 1
    return val, dict(c)


def run_step_prefilter(role, out, snip, val, refs, workers, force):
    surv = out / f"snippets_{role}.jsonl"
    rep = out / f"prefilter_{role}.json"
    if surv.exists() and rep.exists() and not force:
        return surv, json.loads(rep.read_text(encoding="utf-8"))
    t = time.time()
    status = {}
    for r in iter_jsonl(val):
        status[r["id"]] = r.get("exec")
    idx_docs = refs
    seed = SEEDS[role]
    keep, stats = [], Counter()
    pending: list = []

    def blocks():
        for ch in par.chunked(iter_jsonl(snip), 2000):
            pending.append(ch)
            yield [(x["text"], x.get("collection")) for x in ch]

    for flags in par.ordered_map(gf._hit_run, blocks(), workers, gf._hit_init, (idx_docs, seed, 2), window=max(2, workers * 2)):  # noqa: SLF001
        chunk = pending.pop(0)
        for r, h in zip(chunk, flags):
            stats["in"] += 1
            st = status.get(r["id"])
            if "<|" in r["text"] or "```" in r["text"]:
                stats["special_token"] += 1
                continue
            if not (st == "ok" or (st or "").startswith("skipped")):
                stats[f"exec_{st}"] += 1
                continue
            if role == "train" and h:
                stats["decontam_hit"] += 1
                continue
            keep.append(r)
    # roles proibidos nunca passam (defesa extra)
    hom_guard.assert_no_hom(keep, "snippets gramatica v3")
    if role == "train" and any(r.get("split") != "train" for r in keep):
        raise SystemExit("snippets de treino com split != train")
    n = jwrite(surv, keep)
    rep_d = {"in": stats["in"], "kept": n, "dropped": {k: v for k, v in stats.items() if k != "in"}, "eval_reference_docs": len(idx_docs),
             "by_source": dict(Counter(s for s, _ in idx_docs)), "seconds": round(time.time() - t, 1)}
    rep.write_text(json.dumps(rep_d, indent=1), encoding="utf-8")
    print(f"[3] prefilter {role}: {rep_d}", flush=True)
    return surv, rep_d


def run_step_fim(role, out, surv, val, cfgp, workers, count, force):
    fim = out / f"fim_{role}.jsonl"
    if force or not fim.exists():
        t = time.time()
        gf.main(["--config", str(cfgp), "--seed", str(SEEDS[role]), "--count", str(count), "--snippets", str(surv), "--validation", str(val),
                 "--out", str(fim), "--workers", str(workers)])
        print(f"[4] fim {role}: {time.time() - t:.0f}s", flush=True)
    return fim


def run_step_context(role, out, stage, surv, fim, cfgp, force):
    ctx = out / f"fim_ctx_{role}.jsonl"
    man = out / f"fim_ctx_{role}.manifest.json"
    if ctx.exists() and man.exists() and not force:
        return ctx, json.loads(man.read_text(encoding="utf-8"))
    t = time.time()
    dscfg = yaml.safe_load(cfgp.read_text(encoding="utf-8"))
    seed = SEEDS[role]
    tok = rc.Tokenizer(ROOT / dscfg["editor_context"]["tokenizer"])
    schemas = gs.load_schemas(stage)
    names = {}
    for s in schemas:
        shim = type("S", (), {"schema_id": s.schema_id})
        for c in s.colls:
            names[(s.full_id, c.name)] = l2f.schema_names(shim, c)
    snips = list(iter_jsonl(surv))
    cuts = []
    for r in iter_jsonl(fim):
        if hom_guard.record_is_hom(r):
            hom_guard.assert_no_hom([r], "cortes gramatica v3")
        cuts.append(r)
    kinds, kst = l2f.assign_context_kinds(cuts, names, CONTEXT_SHARE, seed, CONTEXT_RATIOS)
    # distratores (known_names/result_fields de OUTRO schema no contexto irrelevant): so schemas `train` (nunca eval/hom/holdout)
    dist = [s for s in gs.load_schemas(ROOT / "data" / "schemas") if (s.raw_role in ("train", None)) and not s.holdout]
    dist = [s for s in dist if not hom_guard.is_hom_schema(s.schema_id, s.domain, s.raw_role)]
    pools = l2f.LlmPools([{"id": s["id"], "op": s["op"], "style": s["style"], "text": s["text"], "schema_id": s["schema_id"],
                           "collection": s["collection"], "split": s["split"], "group_id": s["group_id"]} for s in snips],
                         list({s.full_id: s for s in schemas + dist}.values()), dist)
    recs = l2f.contextualize(cuts, kinds, pools, dscfg, tok, seed)
    for r in recs:
        r["source"] = SOURCE_TAG
        r["author"] = "grammar"
        r["label_source"] = "gold"
    n = jwrite(ctx, recs)
    st = {"role": role, "cuts_in": len(cuts), "cuts_out": n, "context_assignment": kst, "seconds": round(time.time() - t, 1),
          "by_context_kind": dict(Counter(r["context_kind"] for r in recs)), "by_cut_kind": dict(Counter(r["cut_kind"] for r in recs)),
          "by_category": dict(Counter(r["category"] for r in recs)), "none_or_irrelevant_share":
              round(sum(1 for r in recs if r["context_kind"] != "full") / max(1, n), 4)}
    man.write_text(json.dumps(st, indent=1), encoding="utf-8")
    print(f"[5] contexto {role}: {st['by_context_kind']} none+irr={st['none_or_irrelevant_share']}", flush=True)
    return ctx, st


def run_step_pipeline(role, out, snip_surv, ctx, cfgp, refs_files, workers):
    vdir = ROOT / "data" / "validated" / "v3g" / ("eval" if role == "eval" else "")
    vdir = vdir if role == "eval" else ROOT / "data" / "validated" / "v3g"
    splits = ROOT / "data" / "splits" / ("v3g_eval" if role == "eval" else "v3g")
    report = ROOT / "reports" / "datasets" / ("v3g-eval-validation-report.md" if role == "eval" else "v3g-validation-report.md")
    # a decontaminacao da etapa 9 contra a homologacao usa a API autorizada (o dc.load_homolog padrao leria o diretorio de logs)
    dc.load_homolog = lambda *a, **k: [(s, d) for s, d in homolog_refs.homolog_docs()]   # noqa: E731
    _orig_render = vp.render_report

    def _safe_render(path, ver, info):
        """O relatorio padrao assume treino + avaliacao no mesmo lote (gates G3.3 com 'actual' numerico); no lote so-eval/so-treino falha ao formatar."""
        try:
            _orig_render(path, ver, info)
        except Exception as e:     # noqa: BLE001
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(f"# Validacao v3g ({role}) - relatorio padrao indisponivel ({type(e).__name__}); ver {role}.funnel.json e manifest.json\n"
                                  f"\nfunil: {json.dumps(info.get('funnel'), ensure_ascii=False)}\n", encoding="utf-8")
    vp.render_report = _safe_render
    argv = ["--config", str(cfgp), "--ver", "v3g", "--fim", str(ctx), "--chat", str(out / "_no_chat.jsonl"), "--snippets", str(snip_surv),
            "--out-dir", str(vdir), "--splits-dir", str(splits), "--report", str(report), "--workers", str(workers), "--shard-size", "50000",
            "--legacy", str(ROOT / "bakeoff" / "data" / "v1_benchmark.jsonl"), str(ROOT / "bakeoff" / "data" / "v1_validation.jsonl"),
            *[str(p) for p in refs_files]]
    t = time.time()
    res = vp.main(argv)
    print(f"[6] pipeline {role}: {time.time() - t:.0f}s", flush=True)
    return res, vdir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--role", choices=["train", "eval"], required=True)
    ap.add_argument("--out", default="data/generated/v3/grammar")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--snippets", type=int, help="snippets brutos (padrao por papel)")
    ap.add_argument("--cuts", type=int, help="cortes FIM (padrao por papel)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-after", choices=["snippets", "exec", "prefilter", "fim", "context"])
    a = ap.parse_args(argv)
    role = a.role
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    funnel: dict = {"role": role, "seed": SEEDS[role], "source": SOURCE_TAG}
    if role == "train" and not (out / "_eval_docs_v3g.jsonl").exists():
        raise SystemExit("rode --role eval antes (os documentos v3-eval entram na decontaminacao do treino)")
    stage = stage_schemas(role, out)
    funnel["schemas"] = sum(1 for _ in stage.glob("*/*.json"))
    cfgp = derived_config(role, out)
    snip = run_step_snippets(role, out, stage, cfgp, a.snippets or RAW_SNIPPETS[role], a.force)
    funnel["snippets_raw"] = sum(1 for _ in iter_jsonl(snip))
    if a.stop_after == "snippets":
        return 0
    val, exec_counts = run_step_exec(role, out, snip, cfgp, a.workers, a.force)
    funnel["snippet_exec"] = exec_counts
    ok = exec_counts.get("ok", 0)
    ran = ok + exec_counts.get("error", 0)
    funnel["snippet_exec_valid_rate_live"] = round(ok / ran, 4) if ran else None
    if a.stop_after == "exec":
        return 0
    refs = eval_references(role, out)
    surv, prerep = run_step_prefilter(role, out, snip, val, refs, a.workers, a.force)
    funnel["prefilter"] = prerep
    if a.stop_after == "prefilter":
        return 0
    fim = run_step_fim(role, out, surv, val, cfgp, a.workers, a.cuts or CUTS[role], a.force)
    funnel["fim_cuts"] = sum(1 for _ in iter_jsonl(fim))
    if a.stop_after == "fim":
        return 0
    ctx, cst = run_step_context(role, out, stage, surv, fim, cfgp, a.force)
    funnel["context"] = cst
    if a.stop_after == "context":
        return 0
    # referencias extras da etapa 9 (formato jsonl de documentos): cache v2c; no treino, tambem o v3-eval validado
    refs_files = [l2f.EVAL_CACHE] + ([out / "_eval_docs_v3g.jsonl"] if role == "train" else [])
    _res, vdir = run_step_pipeline(role, out, surv, ctx, cfgp, refs_files, a.workers)
    mf = json.loads((vdir / "manifest.json").read_text(encoding="utf-8"))
    if role == "eval":     # documentos de avaliacao v3 (um por snippet) para decontaminar o treino
        seen, docs = set(), []
        for q in sorted(vdir.glob("fim*.jsonl")):
            for r in iter_jsonl(q):
                if r.get("split") in ("val", "test", "bench") and r["snippet_id"] not in seen:
                    seen.add(r["snippet_id"])
                    docs.append({k: r.get(k) for k in ("id", "type", "kind", "prefix", "middle", "suffix", "snippet_id", "collection", "split", "cut_kind")})
        jwrite(out / "_eval_docs_v3g.jsonl", docs)
    funnel["pipeline"] = {"counts": mf["counts"], "rejections": mf["rejections"], "exec_mode": mf.get("exec_mode"), "gates": mf.get("gates"),
                          "decontam": mf.get("decontam"), "inputs_records": mf.get("inputs_records")}
    funnel["seconds_total"] = round(time.time() - t0, 1)
    (out / f"{role}.funnel.json").write_text(json.dumps(funnel, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: funnel[k] for k in ("role", "snippets_raw", "fim_cuts")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
