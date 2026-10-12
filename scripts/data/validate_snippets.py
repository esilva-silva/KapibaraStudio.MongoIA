#!/usr/bin/env python3
"""Valida snippets de LEITURA executando-os no MongoDB real (spec fase-03 §4.5, etapa 5).

  python scripts/data/validate_snippets.py --ver v0 [--limit N] [--uri mongodb://127.0.0.1:27017]

Seguranca (defesa em profundidade):
  * so bancos com prefixo kapi_fx_ (e o nome do banco vem do snippet, validado por regex);
  * so find/findOne/aggregate/countDocuments/distinct; escritas sao `skipped` e nunca chegam ao driver;
  * aggregate: allowlist de estagios; $out/$merge/$function/$accumulator/$where bloqueados em qualquer nivel;
  * maxTimeMS em todas as chamadas; resultado limitado (result_cap);
  * listener de comandos conta qualquer comando de escrita visto pelo servidor (esperado: 0).
Nao parseia JS: executa o plano estruturado (EJSON proprio) gravado por gen_snippets.py.
Saida: data/generated/<ver>/validation.jsonl com {id, exec: ok|error|skipped|blocked|unseeded, n_docs, empty, ms, error}.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid as uuidlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_snippets import CONFIG_PATH, ROOT, load_config  # noqa: E402

DB_RE = re.compile(r"^kapi_fx_[a-z0-9_]{1,40}$")
READ_OPS = {"find", "findOne", "aggregate", "countDocuments", "distinct"}
WRITE_OPS = {"updateOne", "updateMany", "deleteOne", "deleteMany", "insertOne", "insertMany", "replaceOne"}
ALLOWED_STAGES = {"$match", "$group", "$project", "$sort", "$limit", "$skip", "$unwind", "$lookup", "$addFields", "$set",
                  "$unset", "$count", "$sortByCount", "$facet", "$bucket", "$bucketAuto", "$replaceRoot", "$replaceWith",
                  "$sample", "$redact", "$densify", "$fill", "$setWindowFields", "$unionWith"}
FORBIDDEN_KEYS = {"$out", "$merge", "$function", "$accumulator", "$where", "$currentOp", "$listSessions", "$collStats",
                  "$indexStats", "$planCacheStats", "$changeStream", "$search", "$searchMeta", "$vectorSearch"}
# comandos de servidor aceitos (qualquer outro e registrado como violacao)
SAFE_COMMANDS = {"find", "aggregate", "count", "distinct", "getMore", "killCursors", "hello", "isMaster", "ismaster", "ping",
                 "buildInfo", "buildinfo", "endSessions", "saslStart", "saslContinue", "listDatabases", "listCollections",
                 "authenticate", "getnonce", "logout", "connectionStatus"}


class Blocked(Exception):
    pass


def scan_forbidden(v, depth=0):
    if depth > 40:
        raise Blocked("profundidade")
    if isinstance(v, dict):
        for k, x in v.items():
            if k in FORBIDDEN_KEYS:
                raise Blocked(f"operador proibido {k}")
            scan_forbidden(x, depth + 1)
    elif isinstance(v, list):
        for x in v:
            scan_forbidden(x, depth + 1)


def check_pipeline(stages):
    for s in stages:
        if not isinstance(s, dict) or len(s) != 1:
            raise Blocked("estagio invalido")
        (name, body), = s.items()
        if name not in ALLOWED_STAGES:
            raise Blocked(f"estagio nao permitido {name}")
        if name == "$lookup" and isinstance(body, dict) and "pipeline" in body:
            check_pipeline(body["pipeline"])
        if name == "$facet" and isinstance(body, dict):
            for sub in body.values():
                check_pipeline(sub)
    scan_forbidden(stages)


def from_ej(v):
    """EJSON proprio -> tipos BSON do pymongo."""
    from bson import Binary, ObjectId
    from bson.binary import UuidRepresentation
    from bson.decimal128 import Decimal128
    from bson.int64 import Int64
    if isinstance(v, dict):
        if len(v) == 1:
            (k, x), = v.items()
            if k == "$oid":
                return ObjectId(x)
            if k == "$date":
                return datetime.fromisoformat(x.replace("Z", "+00:00")).astimezone(timezone.utc)
            if k == "$numberLong":
                return Int64(int(x))
            if k == "$numberDecimal":
                return Decimal128(x)
            if k == "$uuid":
                return Binary.from_uuid(uuidlib.UUID(x), UuidRepresentation.STANDARD)
        return {k: from_ej(x) for k, x in v.items()}
    if isinstance(v, list):
        return [from_ej(x) for x in v]
    return v


def execute_plan(coll, plan: dict, max_time_ms: int, cap: int) -> int:
    """Executa um plano de LEITURA em `coll`. Retorna n_docs (limitado a `cap`).
    Levanta Blocked para qualquer operacao fora da allowlist (antes de tocar o driver)."""
    op = plan.get("op")
    if op not in READ_OPS:
        raise Blocked(f"op nao permitida: {op}")
    scan_forbidden(plan)
    if op in ("find", "findOne"):
        cur = coll.find(from_ej(plan.get("filter") or {}), from_ej(plan["projection"]) if plan.get("projection") else None,
                        max_time_ms=max_time_ms)
        if plan.get("sort"):
            cur = cur.sort(list(plan["sort"].items()))
        if plan.get("skip"):
            cur = cur.skip(int(plan["skip"]))
        lim = 1 if op == "findOne" else min(int(plan.get("limit") or cap), cap)
        return len(list(cur.limit(lim)))
    if op == "aggregate":
        check_pipeline(plan["pipeline"])
        pipe = from_ej(plan["pipeline"]) + [{"$limit": cap}]
        return len(list(coll.aggregate(pipe, maxTimeMS=max_time_ms, allowDiskUse=False)))
    if op == "countDocuments":
        return int(coll.count_documents(from_ej(plan.get("filter") or {}), maxTimeMS=max_time_ms))
    if op == "distinct":
        return len(coll.distinct(plan["field"], from_ej(plan.get("filter") or {}), maxTimeMS=max_time_ms))
    raise Blocked(op)


def validate_record(client_db_getter, rec: dict, max_time_ms: int, cap: int) -> dict:
    out = {"id": rec["id"], "db": rec.get("db"), "op": rec["op"]}
    plan = rec["plan"]
    if rec["op"] in WRITE_OPS or rec.get("kind") == "write":
        return {**out, "exec": "skipped", "reason": "write", "n_docs": None, "ms": 0}
    db = rec.get("db", "")
    if not DB_RE.match(db):
        return {**out, "exec": "blocked", "reason": f"banco fora do prefixo kapi_fx_: {db}", "n_docs": None, "ms": 0}
    t0 = time.perf_counter()
    try:
        coll = client_db_getter(db)[rec["collection"]]
        n = execute_plan(coll, plan, max_time_ms, cap)
        return {**out, "exec": "ok", "n_docs": n, "empty": n == 0, "ms": round((time.perf_counter() - t0) * 1000, 1)}
    except Blocked as e:
        return {**out, "exec": "blocked", "reason": str(e), "n_docs": None, "ms": 0}
    except Exception as e:  # erro do servidor/driver => snippet invalido
        return {**out, "exec": "error", "error": f"{type(e).__name__}: {e}"[:300], "n_docs": None,
                "ms": round((time.perf_counter() - t0) * 1000, 1)}


def make_client(uri: str):
    from pymongo import MongoClient, monitoring

    class Guard(monitoring.CommandListener):
        def __init__(self):
            self.violations: list[str] = []
            self.reads = 0

        def started(self, event):
            if event.command_name not in SAFE_COMMANDS:
                self.violations.append(event.command_name)
            elif event.command_name in ("find", "aggregate", "count", "distinct"):
                self.reads += 1

        def succeeded(self, event):
            pass

        def failed(self, event):
            pass

    g = Guard()
    return MongoClient(uri, serverSelectionTimeoutMS=3000, event_listeners=[g], appname="kapi-validate-snippets"), g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--ver")
    ap.add_argument("--snippets")
    ap.add_argument("--out")
    ap.add_argument("--uri")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    vc = cfg.get("validation", {})
    ver = a.ver or cfg.get("version", "v0")
    gdir = ROOT / cfg.get("out_dir", "data/generated") / ver
    src = Path(a.snippets) if a.snippets else gdir / "snippets.jsonl"
    recs = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    uri = a.uri or vc.get("uri", "mongodb://127.0.0.1:27017")
    if not re.match(r"^mongodb://(127\.0\.0\.1|localhost)(:\d+)?/?$", uri):
        raise SystemExit("RECUSADO: apenas MongoDB local (127.0.0.1/localhost) sem credenciais")
    client, guard = make_client(uri)
    client.admin.command("ping")
    existing = {n for n in client.list_database_names() if n.startswith(vc.get("db_prefix", "kapi_fx_"))}
    getter = lambda name: client[name]
    max_ms, cap = int(vc.get("max_time_ms", 2000)), int(vc.get("result_cap", 101))

    def work(r):
        if r.get("db") not in existing and r["op"] in READ_OPS:
            return {"id": r["id"], "db": r.get("db"), "op": r["op"], "exec": "unseeded", "reason": "banco nao semeado", "n_docs": None, "ms": 0}
        return validate_record(getter, r, max_ms, cap)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.workers or int(vc.get("workers", 8))) as ex:
        res = list(ex.map(work, recs))
    out = Path(a.out) if a.out else gdir / "validation.jsonl"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    cnt: dict = {}
    for r in res:
        cnt[r["exec"]] = cnt.get(r["exec"], 0) + 1
    reads = cnt.get("ok", 0) + cnt.get("error", 0)
    ok_nonempty = sum(1 for r in res if r["exec"] == "ok" and not r.get("empty"))
    summary = {"total": len(res), **cnt, "exec_valid_rate_reads": round(cnt.get("ok", 0) / reads, 4) if reads else None,
               "ok_nonempty_rate": round(ok_nonempty / cnt["ok"], 4) if cnt.get("ok") else None,
               "write_commands_seen": len(guard.violations), "violations": sorted(set(guard.violations)),
               "read_commands": guard.reads, "elapsed_s": round(time.perf_counter() - t0, 1)}
    (out.parent / "validation.summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))
    if guard.violations:
        sys.exit(3)


if __name__ == "__main__":
    main()
