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
        if got != exp:
            errs.append(f"{tag}: índices divergem (faltam={len(exp - got)}, extras={len(got - exp)})")
        opts = client[e["db"]].list_collections(filter={"name": e["collection"]}).next().get("options", {})
        if bool(opts.get("validator")) != e["validator"]:
            errs.append(f"{tag}: validator presente={bool(opts.get('validator'))} esperado={e['validator']}")
        if not a.no_hash:
            h = hashlib.sha256()
            for d in coll.find({}).sort("_id", 1):
                h.update(bson.encode(d))
            if h.hexdigest() != e["docs_sha256"]:
                errs.append(f"{tag}: hash de documentos diverge")
        ok += 1
    for x in errs:
        print("ERRO", x)
    print(f"{'FALHOU' if errs else 'OK'}: {len(m['entries'])} colecoes verificadas, {len(errs)} divergencias")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
