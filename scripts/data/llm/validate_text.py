#!/usr/bin/env python3
"""Validador por EXECUCAO DO TEXTO real das consultas geradas por LLM (nao do plano estruturado).

  env/.venv/Scripts/python.exe scripts/data/llm/validate_text.py --in data/generated/v3/pilot/queries.raw.jsonl --out data/generated/v3/pilot

Funil (cada etapa grava o motivo da rejeicao):
  1 static     bloqueio estatico de escrita/ddl/execucao arbitraria (common.static_check) -- ja feito na geracao, refeito aqui
  2 syntax     node vm.Script (syntax_check.js)
  3 dup        hash normalizado do texto (comentarios/espacos/aspas) -> duplicata exata
  4 exec       mongosh (WSL) executa o TEXTO atras do proxy somente-leitura (ro_proxy.py): erro de execucao, timeout, nenhum comando de leitura
               enviado (`no_exec`: cursor nunca iterado), tentativa de escrita recusada pelo proxy (`blocked`)
  5 fields     os comandos de leitura REAIS capturados pelo proxy (find/aggregate/count/distinct, com variaveis ja resolvidas) sao conferidos
               contra o schema por plan_fields.py (campo inexistente, colecao inexistente em $lookup/$unionWith, aninhados incluidos)
  6 empty      resultado vazio mantido ate `execution.empty_max_fraction` (25%) das aceitas
  7 shape      forma estrutural repetida (literais trocados) limitada a 2 por (schema, colecao)
Leitura apenas: prova = (a) proxy conta e RECUSA qualquer comando fora da allowlist (proxy_stats), (b) `dbHash` de cada banco tocado antes e
depois e identico, (c) `maxTimeMS` injetado em todo comando de dados.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import (HERE, ROOT, load_cfg, load_schema_infos, query_key, read_jsonl, rel, shape_key, sha, static_check, write_jsonl)
import plan_fields

DISTRO_PATH = lambda p: "/mnt/" + str(p)[0].lower() + str(p)[2:].replace("\\", "/")  # noqa: E731  C:\x\y -> /mnt/c/x/y
DATA_CMDS = {"find", "aggregate", "count", "distinct"}


def syntax_check(items: list[dict]) -> dict:
    """{id: (ok, err)} via node vm.Script."""
    if not items:
        return {}
    payload = "\n".join(json.dumps({"id": it["id"], "text": it["text"]}, ensure_ascii=False) for it in items) + "\n"
    res = subprocess.run(["node", str(HERE / "syntax_check.js")], input=payload, capture_output=True, text=True, encoding="utf-8", timeout=600)
    out = {}
    for line in res.stdout.splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = (r["ok"], r["err"])
    return out


def rewrite_db(text: str, short: str, real: str) -> str:
    """getSiblingDB("<schema_id>") -> getSiblingDB("<kapi_fx_...>") (so no texto enviado ao servidor; o texto original e preservado)."""
    return re.sub(r"(getSiblingDB\s*\(\s*)([\"'`])" + re.escape(short) + r"\2", lambda m: f"{m.group(1)}{m.group(2)}{real}{m.group(2)}", text)


def fingerprint(mongo, dbs) -> dict:
    out = {}
    for d in sorted(dbs):
        try:
            r = mongo[d].command("dbHash")
            out[d] = r["md5"]
        except Exception as e:  # noqa: BLE001
            out[d] = f"erro:{type(e).__name__}"
    return out


def opcounters(mongo) -> dict:
    return dict(mongo.admin.command("serverStatus")["opcounters"])


def run_batch(args) -> dict:
    bdir, items, cfg = args
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "batch.jsonl").write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n", encoding="utf-8")
    ex = cfg["execution"]
    cmd = ["wsl", "-d", ex["wsl_distro"], "-e", "python3", DISTRO_PATH(HERE / "wsl_exec_host.py"), "--batch", DISTRO_PATH(bdir / "batch.jsonl"),
           "--out", DISTRO_PATH(bdir), "--timeout", str(ex["per_query_timeout_s"]), "--max-time-ms", str(ex["max_time_ms"]),
           "--mongo", ex["mongo_host"]]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                         timeout=max(300, int(ex["per_query_timeout_s"]) * len(items) + 120))
    out = {"dir": bdir, "rc": res.returncode, "stderr": res.stderr[-500:], "seconds": time.time() - t0, "results": {}}
    rp = bdir / "results.jsonl"
    if rp.exists():
        for line in rp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out["results"][r["id"]] = r
    sp = bdir / "proxy_stats.json"
    out["proxy"] = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    return out


def analyze(rec: dict, res: dict, info) -> dict:
    """Aplica plan_fields aos comandos reais capturados."""
    from bson import decode
    unknown, ucoll, stages, ops, cmds, unsup = set(), set(), [], set(), [], []
    for c in res.get("commands") or []:
        if c["name"].lower() not in DATA_CMDS:
            continue
        doc = decode(base64.b64decode(c["bson"]))
        r = plan_fields.check_command(doc, info)
        unknown |= set(r["unknown_fields"])
        ucoll |= set(r["unknown_collections"])
        stages += r["stages"]
        ops |= set(r["ops"])
        unsup += r["unsupported"]
        cmds.append(c["name"])
    return {"unknown_fields": sorted(unknown), "unknown_collections": sorted(ucoll), "stages": stages, "ops": sorted(ops), "commands": cmds,
            "unsupported": unsup}


def validate(raw_path: Path, out_dir: Path, cfg: dict, *, workers: int | None = None, limit: int | None = None, seen_keys=None,
             infos=None, skip_exec=False, keep_exec_dirs=False, shape_seen=None) -> dict:
    rows = read_jsonl(raw_path)
    if limit:
        rows = rows[:limit]
    ex = cfg["execution"]
    workers = workers or int(ex["workers"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if infos is None:
        infos = load_schema_infos(roles=("train", "hom"))
    by_schema = {i.full_id: i for i in infos}
    rejected: list[dict] = []
    funnel = Counter()
    funnel["raw"] = len(rows)

    def rej(r, stage, reason):
        rejected.append({"id": r["id"], "schema_id": r["schema_id"], "collection": r["collection"], "task": r.get("task"), "stage": stage,
                         "reason": str(reason)[:300], "text": r["text"], "persona": r.get("persona")})
        funnel["rej_" + stage] += 1

    # 1 static
    cur = []
    for r in rows:
        info = by_schema.get(r["schema_id"])
        if info is None:
            rej(r, "schema", "schema sem papel train/hom ou sem fixture")
            continue
        why = static_check(r["text"], {info.schema_id, info.db})
        if why:
            rej(r, "static", ",".join(why))
        else:
            cur.append(r)
    funnel["after_static"] = len(cur)
    # 2 syntax
    sx = syntax_check(cur)
    nxt = []
    for r in cur:
        ok, err = sx.get(r["id"], (False, "sem resposta do node"))
        if ok:
            nxt.append(r)
        else:
            rej(r, "syntax", err)
    cur = nxt
    funnel["after_syntax"] = len(cur)
    # 3 dup exata
    seen = set(seen_keys or ())
    nxt = []
    for r in cur:
        k = query_key(r["text"])
        if k in seen:
            rej(r, "dup", "texto normalizado repetido")
        else:
            seen.add(k)
            nxt.append(r)
    cur = nxt
    funnel["after_dup"] = len(cur)
    # 4 exec
    results: dict = {}
    proxy_total: dict = {"allowed": Counter(), "blocked": Counter(), "server_errors": 0, "queries": 0, "seconds": 0.0}
    fp_before = fp_after = op_before = op_after = {}
    mongo = None
    dbs = sorted({r["db"] for r in cur})
    if not skip_exec and cur:
        import pymongo
        mongo = pymongo.MongoClient("mongodb://" + ex["mongo_host"], serverSelectionTimeoutMS=4000, appname="kapi-llm-validate-fp")
        fp_before = fingerprint(mongo, dbs)
        op_before = opcounters(mongo)
        work = out_dir / "_exec"
        items = []
        for r in cur:
            info = by_schema[r["schema_id"]]
            items.append({"id": r["id"], "db": info.db, "text": rewrite_db(r["text"], info.schema_id, info.db)})
        bs = int(ex["batch_size"])
        batches = [(work / f"b{n:05d}", items[i:i + bs], cfg) for n, i in enumerate(range(0, len(items), bs))]
        t0 = time.time()
        with ThreadPoolExecutor(workers) as pool:
            for k, out in enumerate(pool.map(run_batch, batches), 1):
                results.update(out["results"])
                for key in ("allowed", "blocked"):
                    for c, v in (out["proxy"].get(key) or {}).items():
                        proxy_total[key][c] += v
                proxy_total["server_errors"] += out["proxy"].get("server_errors", 0)
                proxy_total["queries"] += out["proxy"].get("queries", 0)
                proxy_total["seconds"] += out["proxy"].get("seconds", 0.0)
                if out["rc"] != 0:
                    print(f"[exec] lote {k}: rc={out['rc']} {out['stderr'][-200:]}", flush=True)
                print(f"[exec] lote {k}/{len(batches)} ok  ({(time.time() - t0):.0f}s)", flush=True)
        fp_after = fingerprint(mongo, dbs)
        op_after = opcounters(mongo)
        mongo.close()
    elif skip_exec:
        for r in cur:
            results[r["id"]] = {"status": "ok", "n_docs": 1, "commands": [], "ms": 0}
    nxt = []
    for r in cur:
        res = results.get(r["id"])
        if res is None:
            rej(r, "exec", "sem resultado (falha do host WSL)")
            continue
        if res["status"] != "ok":
            rej(r, "exec", f"{res['status']}: {res.get('error') or ''}")
            continue
        nxt.append((r, res))
    funnel["after_exec"] = len(nxt)
    # 5 fields
    ok_rows = []
    for r, res in nxt:
        info = by_schema[r["schema_id"]]
        a = analyze(r, res, info) if not skip_exec else {"unknown_fields": [], "unknown_collections": [], "stages": [], "ops": [], "commands": [], "unsupported": []}
        if a["unknown_fields"]:
            rej(r, "fields", "campo inexistente: " + ",".join(a["unknown_fields"][:6]))
        elif a["unknown_collections"]:
            rej(r, "fields", "colecao inexistente: " + ",".join(a["unknown_collections"][:4]))
        else:
            ok_rows.append((r, res, a))
    funnel["after_fields"] = len(ok_rows)
    # 7 shape (antes do vazio, para o limite de vazio valer sobre o conjunto final)
    cnt = shape_seen if shape_seen is not None else Counter()
    nxt = []
    for r, res, a in sorted(ok_rows, key=lambda x: x[0]["id"]):
        k = (r["schema_id"], r["collection"], shape_key(r["text"]))
        cnt[k] += 1
        if cnt[k] > 2:
            rej(r, "shape", "forma estrutural repetida")
        else:
            nxt.append((r, res, a))
    funnel["after_shape"] = len(nxt)
    # 6 vazio <= f
    f = float(ex["empty_max_fraction"])
    nonempty = [x for x in nxt if x[1]["n_docs"] > 0]
    empty = sorted([x for x in nxt if x[1]["n_docs"] <= 0], key=lambda x: sha(x[0]["id"]))
    allowed_empty = int(f / (1 - f) * len(nonempty))
    keep = nonempty + empty[:allowed_empty]
    for r, res, a in empty[allowed_empty:]:
        rej(r, "empty", "resultado vazio acima do limite")
    funnel["after_empty"] = len(keep)
    accepted = []
    for r, res, a in sorted(keep, key=lambda x: x[0]["id"]):
        rec = dict(r)
        rec["exec"] = "ok"
        rec["validation"] = {"exec": "ok", "n_docs": res["n_docs"], "empty": res["n_docs"] <= 0, "ms": res.get("ms"), "commands": a["commands"],
                             "stages": a["stages"], "ops": a["ops"], "unsupported": a["unsupported"], "syntax": "ok", "fields": "ok",
                             "validator": "mongosh-text+ro-proxy"}
        accepted.append(rec)
    write_jsonl(out_dir / "queries.validated.jsonl", accepted)
    write_jsonl(out_dir / "queries.rejected.jsonl", rejected)
    unchanged = fp_before == fp_after
    deltas = {k: op_after.get(k, 0) - op_before.get(k, 0) for k in ("insert", "update", "delete", "query", "command", "getmore")} if op_before else {}
    summ = {
        "input": rel(raw_path), "funnel": dict(funnel), "accepted": len(accepted),
        "acceptance_rate": round(len(accepted) / max(1, len(rows)), 4),
        "empty_in_accepted": sum(1 for r in accepted if r["validation"]["empty"]),
        "readonly_proof": {
            "dbHash_before_equals_after": unchanged, "dbs_checked": len(dbs), "fingerprints_before": fp_before if len(dbs) <= 10 else f"{len(dbs)} bancos",
            "server_opcounters_delta_during_run": deltas,
            "proxy_allowed_commands": dict(proxy_total["allowed"]), "proxy_blocked_commands": dict(proxy_total["blocked"]),
            "proxy_write_commands_forwarded": {k: v for k, v in proxy_total["allowed"].items() if k in
                                               ("insert", "update", "delete", "findandmodify", "createindexes", "drop", "dropdatabase", "create")},
            "maxTimeMS_injected": ex["max_time_ms"],
        },
        "exec_seconds_sum": round(proxy_total["seconds"], 1),
    }
    (out_dir / "validation.summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    if not keep_exec_dirs:
        import shutil
        shutil.rmtree(out_dir / "_exec", ignore_errors=True)
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--keep-exec-dirs", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    summ = validate(ROOT / a.inp, ROOT / a.out, cfg, workers=a.workers, limit=a.limit, keep_exec_dirs=a.keep_exec_dirs)
    print(json.dumps(summ, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
