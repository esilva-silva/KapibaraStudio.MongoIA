#!/usr/bin/env python3
"""Conjunto de HOMOLOGACAO LLM v1 (data/homolog/v1; nao versionado): cortes FIM de consultas LLM validadas (+ cortes de gramatica) dos schemas `hom`.

  env/.venv/Scripts/python.exe scripts/data/llm/build_homolog.py --queries data/generated/v3/hom/queries.validated.jsonl [--grammar 4000] [--n 5000]
  ... --dry-run   # ensaio (schemas hom, saida em tmp/homolog_dryrun, NUNCA em data/homolog, sem selar); --train-schemas troca por schemas train

Contrato: so schemas com split_role == "hom" (nunca mistura com train); `author: llm`, `split: hom`; balanceado por cut_kind x context_kind
(full/irrelevant/none, >= 25% sem schema visivel, alvo conservador como na v2d); n >= `homolog.min_per_group` por grupo; decontaminado (ngram_min_content: 2)
contra val/test/bench do v2c; selado por hash por item e do conjunto (homolog_tool.py seal). LIMITACAO documentada no README gerado: mesma familia de
modelo (Qwen) dos alunos => mede generalizacao a estilo/estrutura novos, NAO a consultas humanas reais.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from common import ROOT, load_cfg, load_schema_infos, read_jsonl, rel, sha, write_jsonl
import llm_to_fim as L
import homolog_guard as hg
import homolog_tool as ht
import gen_fim as gf
from gen_snippets import load_config as load_dataset_cfg, load_schemas as _ls  # noqa: F401
from jsonl_shards import iter_jsonl  # noqa: F401

README = """# Conjunto de homologacao LLM v1 (`author: llm`)

- Itens: cortes FIM (`prefix`/`suffix`/`middle`) com contexto de editor `editor-context-v1`, de consultas escritas por **Qwen3-14B (Q4_K_M)** e validadas
  por **execucao do texto real** no MongoDB (mongosh atras de proxy somente-leitura) + cortes dos geradores de gramatica, so de schemas `split_role: hom`.
- Selado: `sealed/manifest.json` (sha256 por item e do conjunto). Leitura so por `scripts/data/llm/homolog_tool.py read --reason ...` (registra em `access.log.jsonl`).
- **Limitacao**: o autor e uma LLM da mesma familia (Qwen) dos modelos treinados. Mede generalizacao a **estilo e estrutura novos** (schemas nunca vistos, consultas
  mais naturais e variadas) e NAO a consultas humanas reais; nao substitui o conjunto cego humano (descartado por decisao do usuario em 2026-10-22).
- Isolamento: nenhum schema `hom` no treino; o treino LLM e decontaminado contra estes itens; `homolog_guard.py` impede os geradores de treino de ler este diretorio.
"""


def grammar_snippets(infos, count: int, dscfg: dict, cfg: dict, seed: int) -> list[dict]:
    """Snippets dos geradores de gramatica para os schemas `hom`, validados por execucao do plano (validate_snippets) -> formato do gen_fim."""
    import gen_snippets as gs
    import validate_snippets as vs
    from pymongo import MongoClient
    dscfg = json.loads(json.dumps(dscfg))
    dscfg["seed"] = seed
    schemas = [i.schema for i in infos]
    by_full = {i.full_id: i for i in infos}
    client, _g = vs.make_client("mongodb://" + cfg["execution"]["mongo_host"])
    out = []
    for r in gs.generate_iter(dscfg, schemas, count):
        info = by_full.get(f"{r['domain']}/{r['schema_id']}") or by_full.get(r["schema_id"])
        if info is None:
            continue
        v = vs.validate_record(lambda db: client[db], {**r, "db": info.db}, 2000, 101)
        if v["exec"] != "ok" or r.get("kind") != "read":
            continue
        group_id = f"{info.full_id}::{r['template_id']}"
        out.append({**r, "id": "snip-g-" + sha(r["id"], 12), "schema_id": info.full_id, "split": "hom", "split_role": "hom", "group_id": group_id,
                    "split_key": gs.split_key(group_id), "grammar": True, "style": r.get("style") or {"root_form": "dot"}})
    client.close()
    return out


def run(a) -> int:
    cfg = load_cfg(a.config)
    seed = int(a.seed if a.seed is not None else cfg["seed"])
    hc = cfg["homolog"]
    dry = a.dry_run
    role = "train" if a.train_schemas else "hom"          # --train-schemas: ensaio da maquinaria sem schemas hom (so com --dry-run)
    if a.train_schemas and not dry:
        raise SystemExit("--train-schemas so com --dry-run")
    infos = load_schema_infos(roles=(role,))
    if not infos:
        print("[erro] nenhum schema com papel `hom` e fixture semeada (aguarde o outro agente / releia data/fixtures/manifest.json). "
              "Use --dry-run para ensaiar a maquinaria com schemas train.", file=sys.stderr)
        return 2
    by_full = {i.full_id: i for i in infos}
    out_root = ROOT / ("tmp/homolog_dryrun" if dry else cfg["homolog_dir"])
    dscfg = load_dataset_cfg()
    tc = gf.middle_tokens_for(dscfg)
    tok = L.make_tokenizer(dscfg)
    t0 = time.time()
    # ---- snippets: LLM + gramatica
    qs = [q for p in a.queries for q in read_jsonl(ROOT / p)]
    qs = [q for q in qs if q["schema_id"] in by_full]
    snips = []
    for q in qs:
        s = L.snippet_from_query(q, by_full[q["schema_id"]], seed)
        s["split"], s["split_role"] = "hom", "hom"
        snips.append(s)
    n_llm = len(snips)
    if a.grammar:
        for g in grammar_snippets(infos, a.grammar, dscfg, cfg, seed):
            snips.append(g)
    print(f"[homolog] snippets: llm={n_llm} gramatica={len(snips) - n_llm}", flush=True)
    # ---- cortes candidatos (kinds equilibrados para cobrir os raros)
    flat = {k: 1.0 for k in L.CUT_KINDS}
    cuts = []
    for s in snips:
        cs = L.cuts_for_snippet(s, dscfg, tc, seed, a.cuts_per_snippet, flat, per_kind_cap=6)
        for c in cs:
            L.finish_fim_record(c, s, None, cfg)
            c["author"] = "llm"
            if s.get("grammar"):
                c["source"] = "grammar-hom"
                c["author"] = "grammar"
            c["split"], c["split_role"] = "hom", "hom"
        cuts += cs
    print(f"[homolog] cortes candidatos: {len(cuts)}", flush=True)
    # ---- decontaminacao contra val/test/bench (documento inteiro)
    evals = L.eval_docs_v2c()
    cuts, drep = L.decontaminate(cuts, evals, int(cfg["fim"]["decontam"]["ngram_min_content"]), seed)
    print(f"[homolog] apos decontaminacao: {len(cuts)} (docs removidos={drep['removed_docs']})", flush=True)
    # ---- need por corte
    names = {}
    for info in infos:
        for coll in info.schema.colls:
            names[(info.full_id, coll.name)] = L.schema_names(info, coll)
    for c in cuts:
        c["_need"] = L.field_need(c["middle"], c["prefix"] + c["suffix"], names[(c["schema_id"], c["collection"])])
    # ---- quotas por (cut_kind, context_kind)
    N = int(a.n or hc["min_cuts"])
    floor = int(hc["min_per_group"]) + 10
    ratios = hc["context_ratios"]
    ct = dscfg["fim"]["cut_targets"]
    kinds = [k for k in hc["main_cut_kinds"]]
    quota = {}
    for k in kinds:
        kk = max(round(N * ct.get(k, 0.05)), floor * 3)
        for ck, r in ratios.items():
            quota[(k, ck)] = max(round(kk * r), floor)
    by_cell = defaultdict(list)
    for c in sorted(cuts, key=lambda c: sha(f"{seed}:hsel:{c['id']}")):
        by_cell[c["cut_kind"]].append(c)
    chosen, ctxkind, per_q, per_schema, short = [], {}, Counter(), Counter(), {}
    cap_q = int(a.max_per_query)
    for k in kinds:
        pool = by_cell.get(k, [])
        used = set()
        for ck in ("none", "irrelevant", "full"):          # os mais restritos primeiro (elegibilidade need <= 1)
            want = quota[(k, ck)]
            got = 0
            for c in pool:
                if c["id"] in used or got >= want:
                    continue
                if ck != "full" and c["_need"] == 2:
                    continue
                qk = c["snippet_id"]
                if per_q[qk] >= cap_q:
                    continue
                used.add(c["id"])
                per_q[qk] += 1
                per_schema[c["schema_id"]] += 1
                chosen.append(c)
                ctxkind[c["id"]] = ck
                got += 1
            if got < want:
                short[f"{k}/{ck}"] = {"want": want, "got": got}
    print(f"[homolog] selecionados: {len(chosen)} (alvo {N}); celulas com falta: {len(short)}", flush=True)
    # ---- contexto + prompt
    dist = list({s.full_id: s.schema for s in load_schema_infos(roles=("train", "hom"))}.values())
    pools = L.LlmPools([{"id": s["id"], "op": s["op"], "style": s.get("style") or {"root_form": "dot"}, "text": s["text"], "schema_id": s["schema_id"],
                         "collection": s["collection"], "split": "hom", "group_id": s["group_id"]} for s in snips],
                       list({i.full_id: i.schema for i in infos}.values()) + dist, [i.schema for i in infos] + dist)
    recs = L.contextualize(chosen, ctxkind, pools, dscfg, tok, seed)
    for r in recs:
        r["homolog_version"] = "v1"
        r["dry_run"] = bool(dry)
    need_by_id = {c["id"]: c["_need"] for c in chosen}
    for r in recs:
        r["field_need"] = need_by_id.get(r["id"])
    # ---- grava e sela
    d = out_root / "v1" if not dry else out_root
    (d / "items").mkdir(parents=True, exist_ok=True)
    ip = d / "items" / "items.jsonl"
    if ip.exists():
        import os
        import stat
        os.chmod(ip, stat.S_IWRITE | stat.S_IREAD)
    with hg.authorized("build_homolog grava o conjunto", "build_homolog", hdir=None if not dry else out_root, detail={"cmd": "build"}):
        n = write_jsonl(ip, recs)
    (d / "README.md").write_text(README, encoding="utf-8")
    cell = Counter((r["cut_kind"], r["context_kind"]) for r in recs)
    groups_ok = {f"{k}/{c}": v for (k, c), v in sorted(cell.items())}
    min_cell = min((cell.get((k, c), 0) for k in kinds for c in ratios), default=0)
    stats = {"n_items": n, "dry_run": dry, "queries_llm": n_llm, "snippets_grammar": len(snips) - n_llm, "cuts_candidates": len(cuts) + drep["removed_docs"],
             "decontam": drep, "cells": groups_ok, "min_cell": min_cell, "shortfalls": short, "min_per_group_target": int(hc["min_per_group"]),
             "none_or_irrelevant_share": round(sum(1 for r in recs if r["context_kind"] != "full") / max(1, n), 4),
             "by_cut_kind": dict(Counter(r["cut_kind"] for r in recs)), "by_context_kind": dict(Counter(r["context_kind"] for r in recs)),
             "by_source": dict(Counter(r["source"] for r in recs)), "by_domain": dict(Counter(r["domain"] for r in recs)),
             "schemas": len({r["schema_id"] for r in recs}), "seconds": round(time.time() - t0, 1)}
    (d / "build_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    if not dry:
        man = ht.seal(d, quiet=True)
        stats["set_sha256"] = man["set_sha256"]
        ok = ht.verify(d)
        stats["verify"] = ok
    else:
        print("[dry-run] nao selado")
    print(json.dumps({k: v for k, v in stats.items() if k not in ("cells", "decontam")}, indent=1, ensure_ascii=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--queries", nargs="+", required=True, help="queries.validated.jsonl de schemas hom")
    ap.add_argument("--grammar", type=int, default=0, help="no. de snippets de gramatica por geracao (validados por execucao)")
    ap.add_argument("--n", type=int)
    ap.add_argument("--cuts-per-snippet", type=int, default=30)
    ap.add_argument("--max-per-query", type=int, default=14)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--dry-run", action="store_true", help="escreve em tmp/homolog_dryrun, sem selar e sem tocar data/homolog")
    ap.add_argument("--train-schemas", action="store_true")
    a = ap.parse_args(argv)
    sys.exit(run(a))


if __name__ == "__main__":
    main()
