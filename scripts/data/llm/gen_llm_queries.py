#!/usr/bin/env python3
"""Gerador de consultas Mongo por LLM local independente (Qwen3-14B Q4_K_M via llama-server), por (schema, colecao).

  env/.venv/Scripts/python.exe scripts/data/llm/gen_llm_queries.py --role train --target 300 --out data/generated/v3/pilot [--resume]
  ... --role hom --target 8000 --out data/homolog/v1/_work/raw   (SO schemas `hom`; nunca mistura com train)

Cada chamada pede N consultas (JSON restrito por json_schema) com persona/tarefa/estilo SORTEADOS aqui (seed determinista por
(seed, schema, colecao, indice)). Saida append-only e retomavel: <out>/queries.raw.jsonl (uma linha por consulta) + calls.jsonl (uma linha por
chamada: tokens, tempo, erro) + gen_summary.json. Bloqueio estatico de escrita/ddl/execucao arbitraria ja nesta etapa (`static`).
O validador (validate_text.py) executa o TEXTO no MongoDB real depois.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

from common import (ROOT, append_jsonl, assert_roles_disjoint, load_cfg, load_schema_infos, query_key, read_jsonl, rel, sha, static_check)
import prompts as P


def build_plan(infos, target: int, cfg: dict, seed: int, task_ids=None, per_call: int | None = None, prefix: str = "") -> list[dict]:
    """Chamadas deterministicas: distribui `target` consultas entre (schema, colecao) com peso ~ sqrt(no. de campos) e cobertura minima.
    Cada chamada tem `group` = idx // group_calls: chamadas do mesmo grupo compartilham o PREFIXO do prompt (schema, amostras, regras)."""
    per_call = per_call or int(cfg["queries_per_call"])
    prng = random.Random(f"{seed}:pairs")
    per_schema = []
    for i in sorted(infos, key=lambda x: x.full_id):
        cs = list(i.schema.colls)
        prng.shuffle(cs)
        per_schema.append([(i, c) for c in cs])
    prng.shuffle(per_schema)
    pairs = []                                      # intercala schemas (1 colecao de cada por vez): alvos pequenos ja cobrem muitos schemas/dominios
    for k in range(max((len(x) for x in per_schema), default=0)):
        pairs += [x[k] for x in per_schema if k < len(x)]
    if not pairs:
        return []
    w = [math.sqrt(max(4, len(c.fields))) for _, c in pairs]
    tot = sum(w)
    ncalls = max(1, math.ceil(target / per_call))
    # grupo = cadeia de chamadas com o mesmo prefixo; para alvos pequenos reduz o grupo para manter >= 2 cadeias por slot
    gcalls = max(1, min(int(cfg.get("group_calls", 6)), ncalls // (2 * int(cfg["server"]["slots"]))))
    alloc = [max(1, round(ncalls * x / tot)) for x in w] if ncalls >= len(pairs) else [1 if k < ncalls else 0 for k in range(len(pairs))]
    plan = []
    for (info, coll), n in zip(pairs, alloc):
        for k in range(n):
            plan.append({"call_id": f"{prefix}{info.schema_id}.{coll.name}.{k:04d}", "info": info, "coll": coll, "idx": k,
                         "group": f"{prefix}{info.schema_id}.{coll.name}.g{k // gcalls}"})
    rng = random.Random(f"{seed}:plan")
    # embaralha os GRUPOS (cadeias), mantendo as chamadas de um grupo juntas e em ordem
    groups: dict = {}
    for c in plan:
        groups.setdefault(c["group"], []).append(c)
    keys = sorted(groups)
    rng.shuffle(keys)
    ordered = [c for k in keys for c in groups[k]]
    return ordered[:ncalls] if len(ordered) > ncalls else ordered


def make_job(call: dict, cfg: dict, seed: int, mongo, task_ids=None, per_call: int | None = None) -> dict:
    info, coll = call["info"], call["coll"]
    rng = random.Random(f"{seed}:{call['call_id']}")                     # fichas (tarefa/persona/estilo): uma por chamada
    grng = random.Random(f"{seed}:{call.get('group', call['call_id'])}")      # amostras e colecoes relacionadas: uma por GRUPO
    n = per_call or int(cfg["queries_per_call"])
    want_intent = bool(cfg.get("want_intent", False))
    specs = P.make_specs(rng, n, float(cfg["short_incomplete_fraction"]), task_ids)
    docs = P.sample_docs(mongo, info.db, coll.name, int(cfg["max_prompt_docs"]), int(cfg["doc_truncate_chars"]), grng) if mongo else []
    msgs = P.build_messages(info, coll, specs, docs, grng, int(cfg["max_related"]), want_intent=want_intent)
    schema = (P.QUERY_SCHEMA if want_intent else P.QUERY_SCHEMA_NOINTENT) if cfg.get("constrained_json", True) else None
    return {"call": call, "specs": specs, "messages": msgs, "seed": rng.randrange(1, 2 ** 31), "schema": schema}


def to_records(job: dict, res: dict, cfg: dict, role: str) -> tuple[list[dict], dict]:
    call, specs = job["call"], job["specs"]
    info, coll = call["info"], call["coll"]
    items = P.parse_response(res["text"]) if not res.get("error") else []
    call_rec = {"call_id": call["call_id"], "schema_id": info.schema_id, "collection": coll.name, "n_requested": len(specs),
                "n_returned": len(items), "prompt_tokens": res.get("prompt_tokens"), "completion_tokens": res.get("completion_tokens"),
                "seconds": round(res.get("seconds", 0.0), 2), "finish": res.get("finish"), "error": res.get("error")}
    recs = []
    for k, it in enumerate(items[:len(specs)]):
        sp = specs[k]
        text = it["query"]
        why = static_check(text, {info.schema_id, info.db})
        recs.append({
            "id": "llmq-" + sha(f"{call['call_id']}:{k}", 14), "call_id": call["call_id"], "k": k,
            "schema_id": info.full_id, "schema_short": info.schema_id, "domain": info.domain, "db": info.db, "collection": coll.name,
            "split_role": role, "source": cfg["source_tag"], "author": "llm", "intent": it["intent"], "text": text,
            "persona": sp["persona"], "task": sp["task"], "layout": "short" if sp["short"] else sp["layout"], "quote": sp["quote"],
            "comments": sp["comments"], "short": sp["short"], "static_block": why, "qkey": query_key(text),
        })
    return recs, call_rec


def run(args, srv=None) -> int:
    cfg = load_cfg(args.config)
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    role = args.role
    infos = load_schema_infos(roles=(role,), require_fixture=True)
    if args.schemas:
        keep = set(args.schemas)
        infos = [i for i in infos if i.schema_id in keep or i.full_id in keep]
    if args.limit_schemas:
        infos = infos[:args.limit_schemas]
    if not infos:
        print(f"[erro] nenhum schema com papel {role!r} e fixture semeada (data/schemas + data/fixtures/manifest.json)", file=sys.stderr)
        return 2
    assert_roles_disjoint(infos if role == "train" else [], infos if role == "hom" else [])
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    prefix = getattr(args, "call_prefix", "") or ""
    plan = build_plan(infos, args.target, cfg, seed, args.tasks, args.per_call, prefix)
    done = {r["call_id"] for r in read_jsonl(out / "calls.jsonl")} if (out / "calls.jsonl").exists() and args.resume else set()
    if not args.resume:
        for f in ("queries.raw.jsonl", "calls.jsonl"):
            (out / f).unlink(missing_ok=True)
    todo = [c for c in plan if c["call_id"] not in done]
    print(f"[plano] papel={role} schemas={len(infos)} chamadas={len(plan)} pendentes={len(todo)} (consultas pedidas ~{len(todo) * (args.per_call or cfg['queries_per_call'])})", flush=True)
    from llm_client import LlmServer, gpu_confirm
    import pymongo
    mongo = pymongo.MongoClient("mongodb://" + cfg["execution"]["mongo_host"], serverSelectionTimeoutMS=4000, appname="kapi-llm-gen")
    own_srv = srv is None
    srv = srv or LlmServer(cfg, url=args.server_url)
    t_start = time.time()
    stats = Counter()
    gpu = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    try:
        gpu = srv.start() if own_srv else srv.gpu

        def work(call):
            job = make_job(call, cfg, seed, mongo, args.tasks, args.per_call)
            try:
                res = srv.chat(job["messages"], seed=job["seed"], schema=job["schema"])
                res["error"] = None
            except Exception as e:  # noqa: BLE001
                res = {"text": "", "error": f"{type(e).__name__}: {e}"[:300], "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
            return job, res

        box = {}
        if srv.proc is not None and not getattr(srv, "_confirmed", False):          # confirmacao independente (contadores do Windows) com a GPU sob carga
            def confirm():
                time.sleep(9)
                box.update(gpu_confirm(srv.proc.pid, 4))
            srv._confirmed = True
            th = threading.Thread(target=confirm, daemon=True)
            th.start()
        lock = threading.Lock()
        chains: dict = {}
        for c in todo:
            chains.setdefault(c.get("group", c["call_id"]), []).append(c)
        chain_list = list(chains.values())
        aborted = []

        def handle(job, res):
            with lock:
                recs, crec = to_records(job, res, cfg, role)
                append_jsonl(out / "queries.raw.jsonl", recs)
                append_jsonl(out / "calls.jsonl", [crec])
                stats["calls"] += 1
                stats["queries"] += len(recs)
                stats["prompt_tokens"] += res.get("prompt_tokens") or 0
                stats["completion_tokens"] += res.get("completion_tokens") or 0
                stats["errors"] += 1 if res.get("error") else 0
                stats["static_blocked"] += sum(1 for r in recs if r["static_block"])
                tm = res.get("timings") or {}
                stats["cached_prompt_tokens"] += tm.get("cache_n") or 0
                if box.get("verdict") == "SUSPEITO":
                    aborted.append(box)
                cb = getattr(args, "on_progress", None)
                if cb:
                    cb(dict(stats))
                if stats["calls"] % 8 == 0:
                    el = time.time() - t_start
                    print(f"[gen] chamadas={stats['calls']} consultas={stats['queries']} {stats['queries'] / el * 3600:.0f}/h "
                          f"tok_saida={stats['completion_tokens']} ({stats['completion_tokens'] / el:.0f} tok/s) erros={stats['errors']} "
                          f"bloq_estatico={stats['static_blocked']} prompt_em_cache={stats['cached_prompt_tokens']}/{stats['prompt_tokens']}", flush=True)

        def run_chain(chain):
            for c in chain:
                if aborted:
                    return
                handle(*work(c))

        with ThreadPoolExecutor(srv.slots) as ex:
            list(ex.map(run_chain, chain_list))
        if aborted:
            raise SystemExit(f"contadores do Windows nao confirmam a 7800 XT: {aborted[0]}")
        if box:
            gpu["confirm"] = box
            print(f"[gpu] confirmacao por contadores do Windows: {box.get('verdict')} (7800 XT {box.get('engine_util_on_big_adapter')}% vs outros {box.get('engine_util_elsewhere')}%)", flush=True)
    finally:
        if own_srv:
            srv.close()
        mongo.close()
    el = time.time() - t_start
    summ = {"role": role, "schemas": len(infos), "seed": seed, "seconds": round(el, 1), **stats, "queries_per_hour": round(stats["queries"] / el * 3600) if el else 0,
            "completion_tok_per_s": round(stats["completion_tokens"] / el, 1) if el else 0, "gpu": {k: v for k, v in gpu.items() if k != "all_devices"},
            "model": cfg["server"]["gguf"], "slots": cfg["server"]["slots"]}
    (out / "gen_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--role", choices=["train", "hom"], required=True)
    ap.add_argument("--target", type=int, required=True, help="consultas BRUTAS a pedir (aceitas serao menos)")
    ap.add_argument("--out", required=True, help="pasta de saida relativa a raiz do repo")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--schemas", nargs="*")
    ap.add_argument("--limit-schemas", type=int)
    ap.add_argument("--tasks", nargs="*", help="restringe aos ids de tarefa")
    ap.add_argument("--per-call", type=int)
    ap.add_argument("--server-url", help="llama-server ja em execucao (nao inicia outro; a guarda de velocidade roda mesmo assim)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    sys.exit(run(a))


if __name__ == "__main__":
    main()
