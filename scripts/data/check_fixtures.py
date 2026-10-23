#!/usr/bin/env python3
"""Valida data/fixtures/manifest.json x servidor: contagens, índices, validators e hash do conteúdo.

  python scripts/data/check_fixtures.py [--uri URI] [--no-hash]   # exit 1 se divergir
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import bson
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_fixtures import canon  # noqa: E402  (hash canonico p/ time-series: o servidor reordena campos)

ROOT = Path(__file__).resolve().parents[2]


def norm_index(ix):
    """(keys, unique, ttl, partial) comparável entre manifest e servidor."""
    keys = tuple((k, d) for k, d in ix["key"].items()) if "key" in ix else tuple((k, d) for k, d in ix["keys"])
    return (keys, bool(ix.get("unique")), ix.get("expireAfterSeconds"),
            json.dumps(ix.get("partialFilterExpression"), sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="mongodb://127.0.0.1:27017")
    ap.add_argument("--manifest", default=str(ROOT / "data" / "fixtures" / "manifest.json"))
    ap.add_argument("--no-hash", action="store_true")
    a = ap.parse_args()
    m = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    client = MongoClient(a.uri, serverSelectionTimeoutMS=5000)
    errs, ok = [], 0
    ver = client.server_info()["version"]
    if ver != m["server_version"]:
        print(f"AVISO: servidor {ver} != manifest {m['server_version']}")
    for e in m["entries"]:
        tag = f"{e['db']}.{e['collection']}"
        if not e["db"].startswith("kapi_fx_"):
            errs.append(f"{tag}: db fora do prefixo kapi_fx_")
            continue
        coll = client[e["db"]][e["collection"]]
        n = coll.count_documents({})
        if n != e["count"]:
            errs.append(f"{tag}: count {n} != {e['count']}")
        got = {norm_index(ix) for ix in coll.list_indexes() if ix["name"] != "_id_"}
        exp = {norm_index(ix) for ix in e["indexes"]}
        # índice text aparece no servidor como _fts/_ftsx; compara só os não-text
        got = {g for g in got if not any(k in ("_fts", "_ftsx") for k, _ in g[0])}
        exp = {x for x in exp if not any(d == "text" for _, d in x[0])}
        tsopt = e.get("options", {}).get("timeseries")
        if tsopt:  # o servidor cria sozinho o indice (metaField, timeField) em colecoes time-series
            got.discard(((tsopt["metaField"], 1), (tsopt["timeField"], 1)))
            got = {g for g in got if g[0] != ((tsopt["metaField"], 1), (tsopt["timeField"], 1))}
        if got != exp:
            errs.append(f"{tag}: índices divergem (faltam={len(exp - got)}, extras={len(got - exp)})")
        info = client[e["db"]].list_collections(filter={"name": e["collection"]}).next()
        opts = info.get("options", {})
        if bool(opts.get("validator")) != e["validator"]:
            errs.append(f"{tag}: validator presente={bool(opts.get('validator'))} esperado={e['validator']}")
        eo = e.get("options", {})  # v3: capped | timeseries
        if eo.get("capped") and not opts.get("capped"):
            errs.append(f"{tag}: deveria ser capped")
        if eo.get("timeseries"):
            ts = opts.get("timeseries", {})
            if info.get("type") != "timeseries" or ts.get("timeField") != eo["timeseries"]["timeField"]                     or ts.get("metaField") != eo["timeseries"]["metaField"]:
                errs.append(f"{tag}: opcoes time-series divergem ({ts})")
        if not eo and (info.get("type") == "timeseries" or opts.get("capped")):
            errs.append(f"{tag}: colecao especial inesperada (capped/timeseries)")
        if not a.no_hash:
            h = hashlib.sha256()
            canonical = "timeseries" in e.get("options", {})
            for d in coll.find({}).sort("_id", 1):
                h.update(bson.encode(canon(d) if canonical else d))
            if h.hexdigest() != e["docs_sha256"]:
                errs.append(f"{tag}: hash de documentos diverge")
        ok += 1
    for x in errs:
        print("ERRO", x)
    print(f"{'FALHOU' if errs else 'OK'}: {len(m['entries'])} colecoes verificadas, {len(errs)} divergencias")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
