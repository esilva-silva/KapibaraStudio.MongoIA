#!/usr/bin/env python3
"""Popula fixtures MongoDB a partir de data/schemas (spec fase-03 §4.2).

  python scripts/data/seed_fixtures.py --seed 20261008 [--uri URI] [--domains a,b] [--schemas x,y] [--drop]

Cria bancos kapi_fx_<domain>_<schema_id>. Recusa qualquer nome fora do prefixo kapi_fx_.
Determinístico: cada documento usa random.Random(f"{seed}:{schema_id}:{coll}:{i}") e datas
ancoradas em intervalos fixos (nunca "agora"); o hash de conteúdo (docs_sha256 no manifest)
é sha256 do BSON de todos os docs ordenados por _id.

v3 (format_version 2, seed por schema = schema["seed"], ids `v3_*`): ADITIVO. Novos construtos do spec de campo (nenhum
muda o fluxo de RNG dos schemas antigos):
  alt   - heterogeneidade: lista de variantes {aw, ...spec} (aw = probabilidade da variante) alternativas ao tipo primario (ex.: ObjectId 88% / string hex 12%);
  map   - objeto de chaves dinamicas: {"map": {"keys": pool, "min", "max", "value": spec}};
  geo   - GeoJSON Point: {"geo": {"lat": [a,b], "lng": [c,d]}};
  sk    - assimetria numerica (valor = lo + (hi-lo) * u**sk); g=isodate|numstr em strings;
  _id   - por colecao: {"t": uuid|string|int64, ...}; options - capped / timeseries; hash canonico p/ time-series.
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import bson
from bson import Binary, ObjectId
from bson.decimal128 import Decimal128
from bson.int64 import Int64
from bson.binary import UuidRepresentation
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_PREFIX = "kapi_fx_"
DB_RE = re.compile(r"^kapi_fx_[a-z0-9_]{1,40}$")
DEFAULT_URI = "mongodb://127.0.0.1:27017"

WORDS = ("alpha beta gamma delta sigma omega river stone cloud ember falcon harbor island jungle kernel lantern marble nectar "
         "orbit pixel quartz raven summit tundra umbra velvet willow xenon yonder zephyr amber bronze cedar dune echo flint "
         "grove haze iris jade kite lotus mesa nova opal pine quill reef sage thorn urn vale wren yarrow zinc").split()
FIRST = "Ana Bruno Carla Diego Elisa Felipe Gabi Hugo Iara Joao Karen Lucas Marina Nuno Olivia Pedro Quiteria Rafael Sofia Tiago Ursula Vitor".split()
LAST = "Silva Souza Costa Lima Alves Rocha Dias Melo Nunes Pinto Ramos Cardoso Moreira Barros Teixeira Gomes Freitas".split()
CITIES = ["Sao Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba", "Porto Alegre", "Salvador", "Recife", "Fortaleza", "Brasilia",
          "Lisboa", "Madrid", "Austin", "Denver", "Toronto", "Osaka", "Pune"]
COUNTRIES = ["BR", "PT", "ES", "US", "CA", "JP", "IN", "AR", "MX", "DE"]
DOMAINS = ["example.com", "mail.test", "corp.example", "demo.invalid"]
MAP_POOLS = {
    "word": WORDS,
    "lang": ["en", "pt", "es", "fr", "de", "ja", "zh", "it"],
    "metric": ["cpu", "mem", "disk", "net_in", "net_out", "temp", "load1", "load5", "iops", "errors", "latency_p50", "latency_p99"],
    "sku": [f"SKU-{n:04d}" for n in range(1, 400)],
    "day": [f"2026-0{m}-{d:02d}" for m in range(1, 9) for d in range(1, 29)],
    "cur": ["BRL", "USD", "EUR", "GBP", "JPY", "ARS", "CLP", "MXN"],
    "region": ["N", "NE", "CO", "SE", "S", "us-east-1", "eu-west-1", "sa-east-1"],
    "hour": [f"h{h:02d}" for h in range(24)],
    "slot": [f"slot_{n}" for n in range(1, 40)],
    "hex": None,  # gerado por rng
}


def safe_db(name):
    if not name.startswith(ALLOWED_PREFIX) or not DB_RE.match(name):
        raise SystemExit(f"RECUSADO: banco '{name}' fora do prefixo {ALLOWED_PREFIX} / nome inseguro")
    return name


def db_name(domain, schema_id):
    return safe_db(f"{ALLOWED_PREFIX}{domain.replace('-', '_')}_{schema_id}"[:60].lower())


# ---------------------------------------------------------------- values ----
def gen_string(spec, rng, i):
    g = spec.get("g", "word")
    if g == "enum":
        w = spec.get("w")
        return rng.choices(spec["values"], weights=w)[0] if w else rng.choice(spec["values"])
    if g == "word":
        return rng.choice(WORDS)
    if g == "name":
        return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    if g == "email":
        return f"{rng.choice(FIRST).lower()}.{rng.choice(LAST).lower()}{i}@{rng.choice(DOMAINS)}"
    if g == "city":
        return rng.choice(CITIES)
    if g == "country":
        return rng.choice(COUNTRIES)
    if g == "sku":
        return f"SKU-{i:05d}"
    if g == "code":
        return f"{spec.get('prefix', '')}{i:06d}" if spec.get("prefix") != "" else f"{rng.randrange(10000):04d}"
    if g == "phone":
        return f"+55 11 9{rng.randrange(10**8):08d}"
    if g == "url":
        return f"https://{rng.choice(WORDS)}.example.com/{rng.choice(WORDS)}"
    if g == "ip":
        return f"10.{rng.randrange(256)}.{rng.randrange(256)}.{rng.randrange(1, 255)}"
    if g == "slug":
        return f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{i}"
    if g == "title":
        return " ".join(w.capitalize() for w in rng.sample(WORDS, rng.randint(2, 4)))
    if g == "sentence":
        return " ".join(rng.sample(WORDS, rng.randint(3, 8))).capitalize() + "."
    if g == "hex":
        return f"{rng.getrandbits(64):016x}"
    if g == "isodate":  # v3: data legada como string ISO-8601 ("iso" com hora, "date" so dia)
        d = _dt(spec["lo"], spec["hi"], rng)
        return d.strftime("%Y-%m-%dT%H:%M:%SZ") if spec.get("fmt", "iso") == "iso" else d.strftime("%Y-%m-%d")
    if g == "numstr":  # v3: numero legado como string
        v = rng.uniform(spec["lo"], spec["hi"])
        return str(int(v)) if spec.get("dec", 0) == 0 else f"{v:.{spec['dec']}f}"
    raise ValueError(f"gerador string desconhecido: {g}")


def _dt(lo, hi, rng):
    a = datetime.fromisoformat(lo)
    ms = int((datetime.fromisoformat(hi) - a).total_seconds() * 1000 * rng.random())
    return a + timedelta(milliseconds=ms)  # sem fromtimestamp: datas < 1970 quebram no Windows


def _num(spec, rng):
    u = rng.random() ** spec["sk"]
    return spec["lo"] + (spec["hi"] - spec["lo"]) * u


def gen_map(m, rng, i, ids):
    pool = MAP_POOLS[m["keys"]]
    n = rng.randint(m["min"], m["max"])
    if pool is None:
        keys = [f"{rng.getrandbits(32):08x}" for _ in range(n)]
    else:
        keys = rng.sample(pool, min(n, len(pool)))
    return {k: gen_value(m["value"], rng, i, ids) for k in keys}


def gen_value(spec, rng, i, ids, uniq=False):
    if spec["n"] and rng.random() < spec["n"]:
        return None
    alt = spec.get("alt")
    if alt:  # v3: heterogeneidade - variante alternativa com probabilidade w (o resto segue o tipo primario)
        r, acc = rng.random(), 0.0
        for v in alt:
            acc += v["aw"]
            if r < acc:
                return gen_value({**v, "n": 0.0, "p": 1.0}, rng, i, ids)
    t = spec["t"]
    ref = spec.get("ref")
    if ref:  # referencia a outra colecao: o tipo segue o _id alvo (objectid | uuid | string | int64)
        lst = ids[ref]
        v = lst[int(len(lst) * rng.random() ** 1.6)]
        if spec.get("as") == "string":
            return str(v.as_uuid()) if isinstance(v, Binary) else str(v)
        return v
    if t == "objectid":
        return ObjectId(rng.randbytes(12))
    if t == "uuid":
        u = uuid.UUID(int=rng.getrandbits(128), version=4)
        rep = UuidRepresentation.STANDARD if spec.get("v", 4) == 4 else UuidRepresentation.PYTHON_LEGACY
        return Binary.from_uuid(u, rep)
    if t == "string":
        v = gen_string(spec, rng, i)
        return f"{v}-{i}" if uniq and spec.get("g") != "enum" else v
    if t == "int32":
        if "sk" in spec:
            return int(_num(spec, rng))
        return rng.randint(spec["lo"], spec["hi"])
    if t == "int64":
        if "sk" in spec:
            return Int64(int(_num(spec, rng)))
        return Int64(rng.randint(spec["lo"], spec["hi"]))
    if t == "double":
        if "sk" in spec:
            return round(_num(spec, rng), 4)
        return round(rng.uniform(spec["lo"], spec["hi"]), 4)
    if t == "decimal128":
        if "sk" in spec:
            return Decimal128(f"{_num(spec, rng):.2f}")
        return Decimal128(f"{rng.uniform(spec['lo'], spec['hi']):.2f}")
    if t == "bool":
        return rng.random() < spec.get("pt", 0.5)
    if t == "date":
        return _dt(spec["lo"], spec["hi"], rng)
    if t == "object":
        if "map" in spec:
            return gen_map(spec["map"], rng, i, ids)
        if "geo" in spec:
            g = spec["geo"]
            return {"type": "Point", "coordinates": [round(rng.uniform(*g["lng"]), 5), round(rng.uniform(*g["lat"]), 5)]}
        return gen_object(spec["f"], rng, i, ids)
    if t == "array":
        n = rng.randint(spec["min"], spec["max"])
        return [gen_value(spec["item"], rng, i * 10 + k, ids) for k in range(n)]
    raise ValueError(t)


def gen_object(fields, rng, i, ids, uniq=()):
    out = {}
    for k, spec in fields.items():
        if spec["p"] < 1.0 and rng.random() >= spec["p"]:
            continue
        out[k] = gen_value(spec, rng, i, ids, uniq=k in uniq)
    return out


def unique_string_fields(col):
    return {e["keys"][0][0] for e in col["indexes"]
            if e.get("unique") and len(e["keys"]) == 1 and e["keys"][0][0] in col["fields"]
            and col["fields"][e["keys"][0][0]]["t"] == "string"}


def make_ids(c, r):
    """_id por colecao: ObjectId (padrao; fluxo de RNG inalterado) ou, no v3, uuid | string | int64."""
    spec, n = c.get("_id"), c["doc_count"]
    if not spec or spec["t"] == "objectid":
        return [ObjectId(r.randbytes(12)) for _ in range(n)]
    if spec["t"] == "uuid":
        return [Binary.from_uuid(uuid.UUID(int=r.getrandbits(128), version=4)) for _ in range(n)]
    if spec["t"] == "string":
        return [f"{spec.get('prefix', '')}{i:06d}-{r.getrandbits(20):05x}" for i in range(n)]
    if spec["t"] == "int64":
        return [Int64(spec.get("base", 100000) + i * 3 + r.randrange(3)) for i in range(n)]
    raise ValueError(spec)


def gen_collection(schema, seed):
    sid = schema["schema_id"]
    ids = {}
    for c in schema["collections"]:
        r = random.Random(f"{seed}:{sid}:{c['key']}:ids")
        ids[c["key"]] = make_ids(c, r)
    for c in schema["collections"]:
        uniq = unique_string_fields(c)
        docs = []
        for i in range(c["doc_count"]):
            rng = random.Random(f"{seed}:{sid}:{c['key']}:{i}")
            d = {"_id": ids[c["key"]][i]}
            d.update(gen_object(c["fields"], rng, i, ids, uniq))
            docs.append(d)
        yield c, docs


def canon(v):
    """Ordena chaves recursivamente (time-series reordena campos ao devolver os documentos)."""
    if isinstance(v, dict):
        return {k: canon(v[k]) for k in sorted(v)}
    if isinstance(v, list):
        return [canon(x) for x in v]
    return v


def docs_hash(docs, canonical=False):
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda x: x["_id"]):
        h.update(bson.encode(canon(d) if canonical else d))
    return h.hexdigest()


INDEX_OPTS = ("unique", "expireAfterSeconds", "partialFilterExpression", "sparse", "collation")


def all_indexes(col):
    """indexes + extra_indexes (v3: wildcard `campo.$**`, fora de `indexes` p/ nao confundir leitores que mapeiam chaves->campos)."""
    return list(col["indexes"]) + list(col.get("extra_indexes", []))


def index_specs(col):
    out = []
    for e in all_indexes(col):
        o = {"keys": [[k, d] for k, d in e["keys"]]}
        for opt in INDEX_OPTS:
            if opt in e:
                o[opt] = e[opt]
        out.append(o)
    return out


def create_indexes(coll, col):
    for e in all_indexes(col):
        opts = {k: e[k] for k in INDEX_OPTS if k in e}
        coll.create_index([(k, d) for k, d in e["keys"]], **opts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--uri", default=DEFAULT_URI)
    ap.add_argument("--domains", default="")
    ap.add_argument("--schemas", default="")
    ap.add_argument("--schema-prefix", default="", help="so schemas cujo schema_id comeca com este prefixo (ex.: v3_)")
    ap.add_argument("--drop", action="store_true",
                    help="dropa so bancos kapi_fx_* listados no manifest (execucao completa) ou os alvos (--domains/--schemas)")
    ap.add_argument("--schemas-dir", default=str(ROOT / "data" / "schemas"))
    ap.add_argument("--manifest", default=str(ROOT / "data" / "fixtures" / "manifest.json"))
    a = ap.parse_args()
    t0 = time.time()
    want_d = {x for x in a.domains.split(",") if x}
    want_s = {x for x in a.schemas.split(",") if x}
    files = sorted(Path(a.schemas_dir).glob("*/*.json"))
    client = MongoClient(a.uri, serverSelectionTimeoutMS=5000)
    version = client.server_info()["version"]
    existing = set(client.list_database_names())
    entries, summary = [], []
    mp = Path(a.manifest)
    filtered = bool(want_d or want_s or a.schema_prefix)
    if a.drop and not filtered and mp.exists():
        # execucao completa: dropa todos os kapi_fx_* do manifest anterior (nunca outros bancos)
        for e in json.loads(mp.read_text(encoding="utf-8"))["entries"]:
            n = e["db"]
            if n in existing:
                client.drop_database(safe_db(n))
                existing.discard(n)
    for f in files:
        raw = f.read_bytes()
        schema = json.loads(raw)
        if want_d and schema["domain"] not in want_d:
            continue
        if want_s and schema["schema_id"] not in want_s:
            continue
        if a.schema_prefix and not schema["schema_id"].startswith(a.schema_prefix):
            continue
        v3 = schema.get("format_version", 1) >= 2
        eff_seed = schema["seed"] if v3 else a.seed  # v3: seed por schema (20261022)
        if not v3 and schema["seed"] != a.seed:
            print(f"AVISO: {f.name} gerado com seed {schema['seed']} != {a.seed}", file=sys.stderr)
        name = db_name(schema["domain"], schema["schema_id"])
        if name in existing:
            if not a.drop:
                raise SystemExit(f"banco {name} já existe; use --drop para recriar")
            client.drop_database(safe_db(name))
            existing.discard(name)
        db = client[name]
        schema_sha = hashlib.sha256(raw).hexdigest()
        for col, docs in gen_collection(schema, eff_seed):
            kw = {}
            if "validator" in col:
                kw = {"validator": col["validator"], "validationAction": col.get("validationAction", "error")}
                if "validationLevel" in col:
                    kw["validationLevel"] = col["validationLevel"]
            kw.update(col.get("options", {}))  # v3: capped/size/max | timeseries
            db.create_collection(col["name"], **kw)
            coll = db[col["name"]]
            for b in range(0, len(docs), 500):
                coll.insert_many(docs[b:b + 500], ordered=False)
            create_indexes(coll, col)
            n = coll.count_documents({})
            assert n == len(docs), (name, col["name"], n, len(docs))
            ent = {
                "schema_id": schema["schema_id"], "domain": schema["domain"], "holdout": schema["holdout"],
                "split_role": schema.get("split_role"),
                "db": name, "collection": col["name"], "count": n,
                "docs_sha256": docs_hash(docs, "timeseries" in col.get("options", {})), "schema_sha256": schema_sha,
                "indexes": index_specs(col), "validator": "validator" in col,
                "search_indexes": [x["name"] for x in col.get("search_indexes", [])],  # static_only: nao criados
            }
            if v3:  # chaves extras so no v3: entradas antigas ficam byte a byte iguais
                ent["seed"] = eff_seed
                if col.get("options"):
                    ent["options"] = col["options"]
            entries.append(ent)
        summary.append((name, len(schema["collections"]), sum(e["count"] for e in entries if e["db"] == name)))
    manifest = {"format_version": 1, "seed": a.seed, "server_version": version, "uri_host": "127.0.0.1:27017",
                "entries": entries}
    mp.parent.mkdir(parents=True, exist_ok=True)
    if filtered and mp.exists():  # execução parcial (--domains/--schemas): mescla com entradas de outros bancos
        old = json.loads(mp.read_text(encoding="utf-8"))["entries"]
        done = {e["db"] for e in entries}
        manifest["entries"] = sorted([e for e in old if e["db"] not in done] + entries,
                                     key=lambda e: (e["db"], e["collection"]))
    manifest["entries"].sort(key=lambda e: (e["db"], e["collection"]))
    mp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    for name, nc, nd in summary:
        print(f"{name:48s} colls={nc} docs={nd}")
    print(f"TOTAL bancos={len(summary)} colecoes={len(entries)} docs={sum(e['count'] for e in entries)} "
          f"server={version} {time.time() - t0:.1f}s manifest={mp}")


if __name__ == "__main__":
    main()
