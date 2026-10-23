#!/usr/bin/env python3
"""Gera o lote de schemas v3 (mais realistas e dificeis): data/schemas/<domain>/v3_<marca>_<dom>.json.

ADITIVO: nao toca nos 110 schemas de build_schemas.py (ids sem prefixo `v3_`), nem em seus bancos/hashes.
Determinístico: seed 20261022. Saida so depende deste arquivo, de schemas_v3_lib.py, de build_schemas.py (ENT/DSL) e da seed.

  python scripts/data/build_schemas_v3.py [--seed 20261022] [--out data/schemas]

Formato (format_version 2) = o de build_schemas.py + construtos opcionais (documentados em seed_fixtures.py):
  alt (heterogeneidade de tipo), map (chaves dinamicas), geo (GeoJSON Point), sk (assimetria), _id por colecao
  (uuid|string|int64), options (capped | timeseries), validationLevel, indices com sparse/collation/hashed/2dsphere/wildcard,
  e `realism_notes` (lista de strings) por schema + `v3` (estatisticas) por colecao.
Indices wildcard (`campo.$**`) ficam em `extra_indexes` (criados pelo seed e listados no manifest). Construtos novos sempre usam `t` = tipo primario valido para leitores antigos (leitura defensiva).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_schemas as B  # noqa: E402
import schemas_v3_lib as L  # noqa: E402
from split_roles import assign_roles_v3  # noqa: E402

SEED = 20261022
ROOT = Path(__file__).resolve().parents[2]
TTL_10Y = B.TTL_10Y

DS3 = dict(B.DS, telecom="tel", adtech="adt", **{"retail-pos": "pos"}, manufacturing="mfg")
PLAN = {  # dominio -> n de schemas v3 (>= 80 no total)
    "telecom": 14, "adtech": 14, "retail-pos": 8, "manufacturing": 8,
    "commerce": 4, "content": 4, "education": 2, "energy": 4, "finance": 4, "gaming": 6, "healthcare-synthetic": 4,
    "hr": 4, "iot": 6, "legal": 2, "logistics": 6, "saas": 6, "travel": 4, "workspace": 4,
}
NEW_DOMAINS = ("telecom", "adtech", "retail-pos", "manufacturing")
HOLDOUT = ("education", "legal")
BRANDS = list(dict.fromkeys(B.BRANDS + ("argon bastion corvid dynamo echelon flux gambit helix ion jasper krypton lattice monolith nexus orbital "
          "pylon quartz relay sable talon uplink vector warden xenial yarrow zephyr bolt crest dune ember flint grove").split()))
# hints de time-series/capped para entidades da biblioteca antiga (ENT)
ENT_HINT = {("iot", "readings"): "ts:at:sensor_id", ("saas", "audit_events"): "cap", ("logistics", "tracking_events"): "cap",
            ("workspace", "notifications"): "cap", ("finance", "fx_rates"): "ts:at:pair"}

# ---------------------------------------------------------------- DSL v3


def parse_fields3(s, sep):
    out = {}
    for it in B.split_top(s, sep):
        name, _, ty = it.partition("=")
        out[name.strip()] = parse_type3(ty.strip())
    return out


def parse_type3(s):
    p, n = 1.0, 0.0
    if s.endswith("??"):
        s, p, n = s[:-2], "o", 0.1
    elif s.endswith("?"):
        s, p = s[:-1], "o"
    elif s.endswith("^"):
        s, p = s[:-1], "r"
    if s.startswith("o("):
        return B.obj(parse_fields3(s[2:-1], ","), p, n)
    if s.startswith("a("):
        return B.arr(B.obj(parse_fields3(s[2:-1], ",")), 1, 4, p, n)
    if s.startswith("aa:"):
        return B.arr(B.arr(parse_type3(s[3:]), 2, 4), 1, 3, p, n)
    if s.startswith("as:"):
        return B.arr(parse_type3(s[3:]), 0, 4, p, n)
    if s.startswith("mp."):
        i = s.index("(")
        d = B.obj({}, p, n)
        d["map"] = {"keys": s[3:i], "min": 2, "max": 6, "value": parse_type3(s[i + 1:-1])}
        return d
    k, _, a = s.partition(":")
    if k == "refu":
        d = B.oid(a, p, n)
        d["alt"] = [{"t": "objectid", "ref": a, "as": "string", "aw": 0.12}]
        return d
    if k == "tu":
        d = B.date(p=p, n=n)
        d["alt"] = [{"t": "string", "g": "isodate", "lo": d["lo"], "hi": d["hi"], "fmt": "iso", "aw": 0.10}]
        return d
    if k == "ds":
        return B._f("string", p, n, g="isodate", lo="2025-01-01", hi="2026-09-30", fmt="iso")
    if k == "nu":
        lo, hi = B._rng_num(a, int)
        d = B.i32(lo, hi, p, n)
        d["alt"] = [{"t": "string", "g": "numstr", "lo": lo, "hi": hi, "dec": 0, "aw": 0.08}]
        return d
    if k == "geo":
        return B._f("object", p, 0.0, f={"type": B.enum(["Point"]), "coordinates": B.arr(B.dbl(-180, 180), 2, 2)},
                    geo={"lat": [-33.0, 5.0], "lng": [-73.0, -35.0]})
    r = B.parse_type(s)
    r["p"], r["n"] = p, n
    return r


def parse_idx3(spec):
    out = []
    for it in [x.strip() for x in spec.split(";") if x.strip()]:
        if it.startswith("!"):
            out.append(dict(keys=it[1:], unique=True))
        elif it.startswith("ttl:"):
            out.append(dict(keys=it[4:], ttl=TTL_10Y))
        elif it.startswith("text:"):
            out.append(dict(keys=it[5:], text=True))
        elif it.startswith("sp:"):
            out.append(dict(keys=it[3:], sparse=True))
        elif it.startswith("ci:"):
            out.append(dict(keys=it[3:], ci=True))
        elif it.startswith("hash:"):
            out.append(dict(keys=it[5:], hashed=True))
        elif it.startswith("wild:"):
            out.append(dict(keys=it[5:], wild=True))
        elif it.startswith("geo:"):
            out.append(dict(keys=it[4:], geo=True))
        elif it.startswith("~"):
            ks, _, fl = it[1:].partition("@")
            out.append(dict(keys=ks, partial_flag=fl))
        else:
            out.append(dict(keys=it))
    return out


# ---------------------------------------------------------------- utilidades de spec
def walk(fields, prefix="", in_arr=False):
    """(caminho, spec, dentro_de_array) de todos os campos (objetos e arrays de objetos, recursivamente)."""
    for k, s in fields.items():
        pp = prefix + k
        yield pp, s, in_arr
        if s["t"] == "object" and "map" not in s and "geo" not in s:
            yield from walk(s["f"], pp + ".", in_arr)
        elif s["t"] == "array":
            it = s["item"]
            while it["t"] == "array":
                it = it["item"]
            if it["t"] == "object":
                yield from walk(it["f"], pp + ".", True)


def get_spec(fields, path):
    f, spec = fields, None
    for part in path.split("."):
        spec = f.get(part)
        if spec is None:
            return None
        if spec["t"] == "object" and "map" not in spec and "geo" not in spec:
            f = spec["f"]
        elif spec["t"] == "array" and spec["item"]["t"] == "object":
            f = spec["item"]["f"]
        else:
            f = {}
    return spec


def array_free(fields, path):
    f = fields
    for part in path.split("."):
        s = f.get(part)
        if s is None or s["t"] == "array":
            return False
        f = s["f"] if (s["t"] == "object" and "map" not in s and "geo" not in s) else {}
    return True


def find_refs(spec):
    out = []
    if spec.get("ref"):
        out.append(spec["ref"])
    if spec["t"] == "object" and "map" not in spec and "geo" not in spec:
        for v in spec["f"].values():
            out += find_refs(v)
    if spec["t"] == "array":
        out += find_refs(spec["item"])
    return out


def prune_refs(fields, keys):
    for k in list(fields):
        s = fields[k]
        if s["t"] == "objectid" and s.get("ref") and s["ref"] not in keys:
            del fields[k]
        elif s["t"] == "array" and not s["item"].get("f") and s["item"].get("ref") and s["item"]["ref"] not in keys:
            del fields[k]
        elif s["t"] == "object" and "map" not in s and "geo" not in s:
            prune_refs(s["f"], keys)
        elif s["t"] == "array":
            it = s["item"]
            while it["t"] == "array":
                it = it["item"]
            if it["t"] == "object":
                prune_refs(it["f"], keys)


def walk_apply(fields, fn):
    """aplica fn(spec) em todo spec (inclui itens de array, variantes alt e valores de mapa)."""
    for s in fields.values():
        _apply(s, fn)


def _apply(s, fn):
    fn(s)
    if s["t"] == "object" and "map" not in s and "geo" not in s:
        walk_apply(s["f"], fn)
    elif s["t"] == "object" and "map" in s:
        _apply(s["map"]["value"], fn)
    elif s["t"] == "array":
        _apply(s["item"], fn)
    for v in s.get("alt", []):
        fn(v)


def tr_fields3(fields):
    """traducao pt-BR de nomes (como B.tr_fields) sem tocar em GeoJSON/mapas; recursiva em arrays de arrays."""
    out = {}
    for k, s in fields.items():
        s = dict(s)
        if s["t"] == "object" and "map" not in s and "geo" not in s:
            s["f"] = tr_fields3(s["f"])
        elif s["t"] == "array":
            s["item"] = _tr_item(s["item"])
        out[B.tr_name(k, True)] = s
    return out


def _tr_item(it):
    it = dict(it)
    if it["t"] == "object" and "map" not in it and "geo" not in it:
        it["f"] = tr_fields3(it["f"])
    elif it["t"] == "array":
        it["item"] = _tr_item(it["item"])
    return it


# ---------------------------------------------------------------- transformacoes de realismo
RARE_POOL = [
    ("notes", lambda: B.text("sentence")), ("deleted_at", lambda: B.date("2025-06-01", "2026-09-30")),
    ("import_batch", lambda: B.text("code", prefix="IMP-")), ("migrated_from", lambda: B.enum(["v1", "mysql", "csv", "erp"])),
    ("override", lambda: B.obj({"by": B.text("name"), "reason": B.text("sentence"), "at": B.date()})),
    ("beta_flag", lambda: B.flag(0.5)), ("external_ref", lambda: B.text("hex")), ("review_state", lambda: B.enum(["pending", "ok", "hold"])),
]
DEEP_TOP = ["audit", "meta", "ctx", "trace", "extra", "origin", "ext", "info"]
DEEP_MID = ["source", "system", "client", "device", "runtime", "network", "route", "policy", "rule", "limits", "profile",
            "region", "zone", "tenant", "config", "options", "flags", "stack"]
DEEP_LEAF = [("version", lambda: B.text("code", prefix="v")), ("enabled", lambda: B.flag(0.6)), ("count", lambda: B.i32(0, 900)),
             ("name", lambda: B.text("word")), ("seen_at", lambda: B.date()), ("score", lambda: B.dbl(0, 100)),
             ("level", lambda: B.enum(["low", "mid", "high"]))]


def inject_rare(fields, rng, notes, cname):
    added = []
    for nm, mk in rng.sample(RARE_POOL, rng.randint(1, 2)):
        if nm in fields:
            continue
        s = mk()
        s["p"] = "r"
        fields[nm] = s
        added.append(nm)
    if added:
        notes.append(f"{cname}: campos raros (<10% dos docs): {', '.join(added)}")
    return added


def inject_legacy(fields, protected, rng, notes, cname):
    cands = [k for k, s in fields.items() if k not in protected and s["t"] in ("string", "int32", "int64", "double", "decimal128", "date", "bool")
             and not s.get("alt") and s.get("g") != "enum" and s["p"] != "r"]
    rng.shuffle(cands)
    out = []
    for k in cands[:rng.randint(0, 2)]:
        twin = copy.deepcopy(fields[k])
        twin["p"] = rng.choice([0.15, 0.25, 0.4])
        if twin["t"] == "date":
            twin = B._f("string", twin["p"], 0.0, g="isodate", lo=twin["lo"], hi=twin["hi"], fmt=rng.choice(["iso", "date"]))
        elif twin["t"] in ("int32", "int64", "double", "decimal128"):
            twin = B._f("string", twin["p"], 0.0, g="numstr", lo=twin["lo"], hi=twin["hi"], dec=0 if twin["t"] in ("int32", "int64") else 2)
        nm = rng.choice([f"{k}_v2", f"{k}_old", f"legacy_{k}", f"old_{k}"])
        if nm not in fields:
            fields[nm] = twin
            out.append(nm)
    if out:
        notes.append(f"{cname}: campos legados/duplicados em formato antigo: {', '.join(out)}")
    return out


def deep_chain(rng):
    k = rng.choice([4, 4, 5, 5, 6])  # niveis de objeto aninhados abaixo do documento
    names = [rng.choice(DEEP_TOP)] + rng.sample(DEEP_MID, k - 1)
    ln, mk = rng.choice(DEEP_LEAF)
    leaf = {ln: mk()}
    if rng.random() < 0.6:
        n2, mk2 = rng.choice(DEEP_LEAF)
        leaf[n2] = mk2()
    node = leaf
    for lvl in range(k, 0, -1):
        o = B.obj(node, "o" if lvl == 1 else rng.choice([1.0, "o"]))
        if lvl > 1 and rng.random() < 0.5:
            sn, smk = rng.choice(DEEP_LEAF)
            if sn != names[lvl - 1]:
                o["f"].setdefault(sn, smk())
        node = {names[lvl - 1]: o}
    return names, node


def inject_deep(fields, rng, notes, cname):
    names, node = deep_chain(rng)
    top = names[0]
    if top in fields:
        return None
    fields[top] = node[top]
    notes.append(f"{cname}: aninhamento profundo {'.'.join(names)}.* ({len(names)} niveis de objeto)")
    return names


def calibrate(fields, rng, protected, prefix=""):
    """pesos assimetricos em enums, assimetria numerica e proporcao de booleanos calibradas (filtros devolvem alguns-nao-todos)."""
    for k, s in fields.items():
        pp = prefix + k
        t = s["t"]
        if t == "string" and s.get("g") == "enum" and not s.get("w"):
            n = len(s["values"])
            w = [round(100.0 / (i + 1) ** rng.uniform(0.6, 1.3), 1) for i in range(n)]
            rng.shuffle(w)
            s["w"] = w
        elif t in ("int32", "int64", "double", "decimal128") and "sk" not in s and pp not in protected and rng.random() < 0.45:
            if s["hi"] - s["lo"] >= 20:
                s["sk"] = round(rng.uniform(1.4, 2.6), 2)
        elif t == "bool":
            s["pt"] = round(min(0.95, max(0.05, s.get("pt", 0.5) + rng.uniform(-0.2, 0.2))), 2)
        elif t == "object" and "map" not in s and "geo" not in s:
            calibrate(s["f"], rng, protected, pp + ".")
        elif t == "array":
            it = s["item"]
            if it["t"] == "object":
                calibrate(it["f"], rng, protected, pp + ".")
        if t == "object" and "map" in s:
            v = s["map"]["value"]
            if v["t"] == "bool":
                v["pt"] = round(rng.uniform(0.1, 0.9), 2)


def hetero(fields, rng, protected, stats, prefix=""):
    """mesmo campo com 2 tipos em partes da colecao (alt)."""
    for k, s in fields.items():
        pp = prefix + k
        t = s["t"]
        if pp in protected or s.get("alt") or "map" in s or "geo" in s:
            if t == "object" and "map" not in s and "geo" not in s:
                hetero(s["f"], rng, protected, stats, pp + ".")
            continue
        r = rng.random()
        if t == "objectid" and s.get("ref") and r < 0.12:
            s["alt"] = [{"t": "objectid", "ref": s["ref"], "as": "string", "aw": round(rng.uniform(0.05, 0.2), 2)}]
            stats["ref_string"] += 1
        elif t == "date" and r < 0.14:
            s["alt"] = [{"t": "string", "g": "isodate", "lo": s["lo"], "hi": s["hi"], "fmt": rng.choice(["iso", "date"]),
                         "aw": round(rng.uniform(0.05, 0.2), 2)}]
            stats["date_string"] += 1
        elif t in ("int32", "int64") and r < 0.10:
            if rng.random() < 0.5:
                s["alt"] = [{"t": "double", "lo": s["lo"], "hi": s["hi"], "aw": round(rng.uniform(0.05, 0.2), 2)}]
            else:
                s["alt"] = [{"t": "string", "g": "numstr", "lo": s["lo"], "hi": s["hi"], "dec": 0, "aw": round(rng.uniform(0.04, 0.12), 2)}]
            stats["num_mixed"] += 1
        elif t == "decimal128" and r < 0.10:
            s["alt"] = [{"t": "double", "lo": s["lo"], "hi": s["hi"], "aw": round(rng.uniform(0.05, 0.2), 2)}]
            stats["num_mixed"] += 1
        elif t == "double" and r < 0.07:
            s["alt"] = [{"t": "int32", "lo": int(s["lo"]), "hi": int(s["hi"]) + 1, "aw": round(rng.uniform(0.1, 0.25), 2)}]
            stats["num_mixed"] += 1
        elif t == "bool" and r < 0.06:
            s["alt"] = [{"t": "string", "g": "enum", "values": ["Y", "N", "true", "false", "1", "0"], "aw": 0.06}]
            stats["bool_string"] += 1
        elif t == "array" and s["item"]["t"] == "string" and s["item"].get("g") not in ("enum",) and r < 0.12:
            s["alt"] = [{"t": "string", "g": "word", "aw": round(rng.uniform(0.08, 0.2), 2)}]
            stats["scalar_or_array"] += 1
        elif t == "object":
            hetero(s["f"], rng, protected, stats, pp + ".")
        elif t == "array" and s["item"]["t"] == "object":
            hetero(s["item"]["f"], rng, protected, stats, pp + ".")


# ---------------------------------------------------------------- nomes bagunçados
ABBR = dict(customer="cust", quantity="qty", address="addr", amount="amt", number="num", description="descr", created="crt",
            updated="upd", reference="ref", message="msg", transaction="txn", information="info", configuration="cfg",
            category="cat", department="dept", employee="emp", product="prod", document="doc", organization="org",
            account="acct", balance="bal", maximum="max", minimum="min", temperature="temp", location="loc", payment="pmt",
            invoice="inv", shipment="shp", vehicle="veh", supplier="supp", price="prc", subscriber="sub", campaign="cmp",
            advertiser="adv", impression="imp", machine="mach", operator="oper", inspection="insp", terminal="term",
            status="st", total="tot", customer_id="cid", timestamp="ts", identifier="ident", percent="pct", session="sess",
            device="dev", duration="dur", latitude="lat", longitude="lng", priority="prio", source="src", target="tgt")
HUNG = {"date": ["dt", "d"], "bool": ["is", "fl"], "int32": ["n", "cnt"], "int64": ["n", "cnt"], "string": ["s", "str"],
        "objectid": ["fk", "ref"], "double": ["f", "dbl"], "decimal128": ["vl", "dec"]}
SUFFIX = ["v2", "new", "old", "legacy", "x", "2"]


def style_words(words, style):
    if style == "snake":
        return "_".join(words)
    if style == "camel":
        return words[0] + "".join(w[:1].upper() + w[1:] for w in words[1:])
    return "".join(w[:1].upper() + w[1:] for w in words)


def mess_name(name, spec, style, rng, top, stats):
    if name == "_id":
        return name
    words = [w for w in name.split("_") if w]
    if not words:
        return name
    tags = []
    if rng.random() < 0.04:
        alt = rng.choice([x for x in ("snake", "camel", "pascal") if x != style])
        style_used = alt
        tags.append("style")
    else:
        style_used = style
    if rng.random() < 0.35:
        ab = [i for i, w in enumerate(words) if w in ABBR]
        if ab:
            i = rng.choice(ab)
            words[i] = ABBR[words[i]]
            tags.append("abbr")
    if top and rng.random() < 0.025 and spec["t"] in HUNG:
        words = [rng.choice(HUNG[spec["t"]])] + words
        tags.append("hung")
    elif rng.random() < 0.025:
        words = words + [rng.choice(SUFFIX)]
        tags.append("suffix")
    out = style_words(words, style_used)
    if "id" in words and rng.random() < 0.15 and style_used != "snake":
        out = out[:-2] + "ID" if out.endswith("Id") else out
        tags.append("ID")
    if rng.random() < 0.012 and top:
        out = "_" + out
        tags.append("underscore")
    if rng.random() < 0.008 and spec["t"] in ("string", "bool", "int32"):
        out = out.upper()
        tags.append("upper")
    for t in tags:
        stats[f"name_{t}"] += 1
    return out


def rename_fields(fields, rng, style, prefix_old, prefix_new, mp, stats, top=True, resolve=True):
    out = {}
    used = set()
    for k, s in fields.items():
        s = dict(s)
        if resolve:
            if s["p"] == "o":
                s["p"] = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
            elif s["p"] == "r":
                s["p"] = rng.choice([0.03, 0.05, 0.08])
        nk = mess_name(k, s, style, rng, top, stats)
        if nk in used or nk == "_id" or "." in nk:
            nk = B.style_name(k, style)
        if nk in used:
            nk = nk + "2"
        used.add(nk)
        mp[prefix_old + k] = prefix_new + nk
        sub_style = style if rng.random() > 0.15 else rng.choice([x for x in ("snake", "camel", "pascal") if x != style])
        if s["t"] == "object" and "map" not in s and "geo" not in s:
            s["f"] = rename_fields(s["f"], rng, sub_style, prefix_old + k + ".", prefix_new + nk + ".", mp, stats, False, resolve)
        elif s["t"] == "object" and "map" in s:
            m = dict(s["map"])
            m["value"] = _resolve_nested(m["value"], rng, sub_style, stats, resolve)
            s["map"] = m
        elif s["t"] == "array":
            s["item"] = _rename_item(s["item"], rng, sub_style, prefix_old + k + ".", prefix_new + nk + ".", mp, stats, resolve)
        out[nk] = s
    return out


def _rename_item(it, rng, style, po, pn, mp, stats, resolve):
    it = dict(it)
    it["p"] = 1.0
    if it["t"] == "object" and "map" not in it and "geo" not in it:
        it["f"] = rename_fields(it["f"], rng, style, po, pn, mp, stats, False, resolve)
    elif it["t"] == "array":
        it["item"] = _rename_item(it["item"], rng, style, po, pn, mp, stats, resolve)
    elif it["t"] == "object" and "map" in it:
        m = dict(it["map"])
        m["value"] = _resolve_nested(m["value"], rng, style, stats, resolve)
        it["map"] = m
    return it


def _resolve_nested(v, rng, style, stats, resolve):
    v = dict(v)
    if v["t"] == "object" and "map" not in v and "geo" not in v:
        v["f"] = rename_fields(v["f"], rng, style, "#", "#", {}, stats, False, resolve)
    elif v["t"] == "array":
        v["item"] = _rename_item(v["item"], rng, style, "#", "#", {}, stats, resolve)
    return v


def tr_path(path, mp):
    if path in mp:
        return mp[path]
    parts = path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        pre = ".".join(parts[:i])
        if pre in mp:
            return mp[pre] + "." + ".".join(parts[i:])
    return path


def coll_name(key, pt_key, pt, style, rng, used, stats):
    base = pt_key if pt else key
    words = base.split("_")
    name = style_words(words, style) if style != "pascal" else style_words(words, "pascal")
    r = rng.random()
    if r < 0.10:
        name = rng.choice(["tbl_", "col_", "t_"]) + "_".join(words)
        stats["coll_prefix"] += 1
    elif r < 0.22:
        name = name + rng.choice(["_v2", "_old", "_bkp", "_2025", "_new"])
        stats["coll_suffix"] += 1
    elif r < 0.28 and len(words) > 1:
        name = "-".join(words)
        stats["coll_hyphen"] += 1
    elif r < 0.31 and len(words) > 1:
        name = " ".join(w.capitalize() for w in words)
        stats["coll_space"] += 1
    if name in used:
        name = name + "_2"
    used.add(name)
    return name


# ---------------------------------------------------------------- indices
def valid_index(fields, ix):
    keys = [k.lstrip("-") for k in ix["keys"].split(",")]
    for k in keys:
        if get_spec(fields, k) is None:
            return False
    if ix.get("geo"):
        s = get_spec(fields, keys[0])
        return len(keys) == 1 and "geo" in s and array_free(fields, keys[0])
    if ix.get("wild"):
        s = get_spec(fields, keys[0])
        return len(keys) == 1 and s["t"] == "object" and array_free(fields, keys[0])
    if ix.get("hashed"):
        s = get_spec(fields, keys[0])
        return len(keys) == 1 and s["t"] not in ("object", "array") and array_free(fields, keys[0])
    if ix.get("ci"):
        s = get_spec(fields, keys[0])
        return len(keys) == 1 and s["t"] == "string" and array_free(fields, keys[0])
    base = {k: v for k, v in ix.items() if k in ("keys", "unique", "ttl", "text")}
    if not B.idx_ok(fields, base):
        return False
    if (ix.get("unique") or ix.get("ttl") or ix.get("text")) and any(get_spec(fields, k).get("alt") for k in keys):
        return False
    for k in keys:  # indice comum nao pode cobrir mapas/GeoJSON sem ser wild/geo
        s = get_spec(fields, k)
        if "map" in s or "geo" in s:
            return False
    return True


def build_indexes(fields, idx_in, key, hint, rng, notes, cname):
    """-> (lista de indices em caminhos ORIGINAIS, protegidos)."""
    fin = []
    for ix in idx_in:
        ix = dict(ix)
        if "partial_flag" in ix:
            fl = ix.pop("partial_flag")
            s = fl and get_spec(fields, fl)
            if s and s["t"] == "bool":
                ix["partial"] = {fl: True}
            else:
                ix["partial"] = {ix["keys"].split(",")[0].lstrip("-"): {"$exists": True}}
        if valid_index(fields, ix):
            fin.append(ix)
    seen, out, has_text = {}, [], False
    for ix in fin:
        kk = (ix["keys"], bool(ix.get("geo")), bool(ix.get("wild")), bool(ix.get("hashed")))
        if kk not in seen or ix.get("ttl"):
            seen[kk] = ix
    for ix in seen.values():
        if ix.get("text"):
            if has_text:
                continue
            has_text = True
        out.append(ix)
    # extras de variedade
    used = {ix["keys"].lstrip("-") for ix in out}
    maps = [p for p, s, a in walk(fields) if s["t"] == "object" and "map" in s and not a and array_free(fields, p)]
    geos = [p for p, s, a in walk(fields) if "geo" in s and not a and array_free(fields, p)]
    refs = [k for k, s in fields.items() if s["t"] == "objectid" and s.get("ref") and not s.get("alt")]
    cis = [k for k, s in fields.items() if s["t"] == "string" and s.get("g") in ("name", "email", "title")]
    rares = [k for k, s in fields.items() if s["p"] == "r" and s["t"] not in ("object", "array")]
    dts = [k for k, s in fields.items() if s["t"] == "date" and s["lo"] >= "2025-01-01" and not s.get("alt")]
    extras = []
    if maps and rng.random() < 0.6 and not any(i.get("wild") for i in out):
        extras.append(dict(keys=rng.choice(maps), wild=True))
    if geos and rng.random() < 0.7 and not any(i.get("geo") for i in out):
        extras.append(dict(keys=rng.choice(geos), geo=True))
    if refs and rng.random() < 0.22 and not any(i.get("hashed") for i in out):
        extras.append(dict(keys=rng.choice(refs), hashed=True))
    if rares and rng.random() < 0.5:
        extras.append(dict(keys=rng.choice(rares), sparse=True))
    if cis and rng.random() < 0.3 and not any(i.get("ci") for i in out):
        extras.append(dict(keys=rng.choice(cis), ci=True))
    if dts and rng.random() < 0.18 and not any(i.get("ttl") for i in out):
        extras.append(dict(keys=rng.choice(dts), ttl=TTL_10Y))
    for ix in extras:
        k0 = ix["keys"]
        if k0 not in used and valid_index(fields, ix):
            out.append(ix)
            used.add(k0)
    return out


def index_protected(idx):
    prot = set()
    for ix in idx:
        keys = [k.lstrip("-") for k in ix["keys"].split(",")]
        if any(ix.get(f) for f in ("unique", "ttl", "text", "hashed", "geo", "wild", "ci")):
            prot.update(keys)
        if ix.get("partial"):
            prot.update(k for k in ix["partial"] if not k.startswith("$"))
    return prot


def final_index(ix, mp, fields_final):
    keys = []
    for part in ix["keys"].split(","):
        d = -1 if part.startswith("-") else 1
        path = tr_path(part.lstrip("-"), mp)
        if ix.get("wild"):
            keys.append([path + ".$**", 1])
        elif ix.get("geo"):
            keys.append([path, "2dsphere"])
        elif ix.get("hashed"):
            keys.append([path, "hashed"])
        elif ix.get("text"):
            keys.append([path, "text"])
        else:
            keys.append([path, d])
    e = {"keys": keys}
    if ix.get("unique"):
        e["unique"] = True
    if ix.get("ttl"):
        e["expireAfterSeconds"] = ix["ttl"]
    if ix.get("partial"):
        e["partialFilterExpression"] = {(k if k.startswith("$") else tr_path(k, mp)): v for k, v in ix["partial"].items()}
    if ix.get("sparse"):
        e["sparse"] = True
    if ix.get("ci"):
        e["collation"] = {"locale": "en", "strength": 2}
    # unique em campo opcional/nulo => partial implicito (evita colisao de null)
    if e.get("unique") and "partialFilterExpression" not in e and len(e["keys"]) == 1:
        s = get_spec(fields_final, e["keys"][0][0])
        if s and (s["p"] < 1.0 or s["n"]):
            e["partialFilterExpression"] = {e["keys"][0][0]: {"$exists": True}}
    return e


# ---------------------------------------------------------------- validator
IDBT = B.BSON_TYPE


def _bt(spec):
    if spec.get("as") == "string":
        return "string"
    return IDBT[spec["t"]]


def prop_schema(spec):
    types = [_bt(spec)]
    for v in spec.get("alt", []):
        b = _bt(v)
        if b not in types:
            types.append(b)
    if spec["n"]:
        types.append("null")
    p = {"bsonType": types[0] if len(types) == 1 else types}
    if spec["t"] == "string" and spec.get("g") == "enum" and not spec.get("alt"):
        p["enum"] = list(spec["values"]) + ([None] if spec["n"] else [])
    if spec["t"] == "object":
        if "map" in spec:
            p["additionalProperties"] = prop_schema(spec["map"]["value"])
        elif "geo" in spec:
            p["properties"] = {"type": {"enum": ["Point"]}, "coordinates": {"bsonType": "array", "items": {"bsonType": "double"}}}
        else:
            sub = obj_schema(spec["f"])
            p.update({k: v for k, v in sub.items() if k != "bsonType"})
    if spec["t"] == "array":
        p["items"] = prop_schema({**spec["item"], "n": 0.0})
        # item de array ja e sempre presente; alt do array (escalar) vale no proprio campo
    return p


def obj_schema(fields):
    props, req = {}, []
    for k, s in fields.items():
        props[k] = prop_schema(s)
        if s["p"] == 1.0:
            req.append(k)
    out = {"bsonType": "object", "properties": props}
    if req:
        out["required"] = req
    return out


def retarget(fields, idkind):
    def fn(s):
        ref = s.get("ref")
        if not ref:
            return
        kind = idkind.get(ref, "objectid")
        if kind == "uuid":
            s["t"] = "uuid"
            s["v"] = 4
        elif kind == "string":
            s["t"] = "string"
            s["g"] = "word"
        elif kind == "int64":
            s["t"] = "int64"
            s["lo"], s["hi"] = 0, 1
    walk_apply(fields, fn)
    # alt redundante: referencia a _id string nao tem variante "as string"
    def fn2(s):
        if s.get("alt"):
            s["alt"] = [v for v in s["alt"] if not (v.get("as") == "string" and idkind.get(v.get("ref"), "objectid") == "string")]
            if not s["alt"]:
                del s["alt"]
    walk_apply(fields, fn2)


# ---------------------------------------------------------------- montagem
def pool_for(dom):
    if dom == "telecom":
        return list(L.TELECOM)
    if dom == "adtech":
        return list(L.ADTECH)
    if dom == "retail-pos":
        return list(L.RETAIL_POS)
    if dom == "manufacturing":
        return list(L.MANUFACTURING)
    base = [(k, pk, f, ix, ENT_HINT.get((dom, k), "")) for k, pk, f, ix in B.ENT[dom]]
    return base + list(L.EXTRA.get(dom, []))


def doc_count(key, hint, rng):
    big = re.search(r"event|log|reading|cdr|impression|click|transaction|tick|record|ping|view|metric|usage|swipe|stream|interval|"
                    r"sale|line|bid|downtime|activity|search", key)
    small = re.search(r"plan|categor|store|plant|supplier|tool|court|role|region|dealer|destination|facilit|department|position|"
                      r"publisher|placement|advertiser|carrier|tariff", key)
    if hint:
        return rng.randrange(1500, 3001, 50)
    if big:
        return rng.randrange(1000, 2801, 50)
    if small:
        return rng.randrange(200, 701, 50)
    return rng.randrange(300, 1601, 50)


def build_collection(ent, dom, pt, style, keys, rng, srng, notes, hint_budget, stats_total):
    key, pt_key, fstr, istr, hint = ent
    fields = parse_fields3(fstr, ";")
    idx_in = parse_idx3(istr)
    cname_notes = key
    if pt:
        fields = tr_fields3(fields)
        # tr_fields so entra em object/array-de-objeto; traduz tambem o caminho dos indices e hints
        def trk(path):
            return ".".join(B.tr_name(p, True) for p in path.split("."))
        for ix in idx_in:
            ix["keys"] = ",".join(("-" if x.startswith("-") else "") + trk(x.lstrip("-")) for x in ix["keys"].split(","))
            if ix.get("partial_flag"):
                ix["partial_flag"] = trk(ix["partial_flag"])
        if hint.startswith("ts:"):
            _, tf, mf = hint.split(":")
            hint = f"ts:{trk(tf)}:{trk(mf)}"
    prune_refs(fields, keys)
    # hint (time-series / capped) so se couber no orcamento do schema
    use_hint = ""
    if hint and hint_budget[0] > 0 and rng.random() < 0.7:
        if hint == "cap":
            use_hint = "cap"
        else:
            _, tf, mf = hint.split(":")
            if tf in fields and mf in fields and fields[tf]["t"] == "date":
                use_hint = hint
        if use_hint:
            hint_budget[0] -= 1
    # campos extras de realismo
    cnotes = []
    inject_rare(fields, rng, cnotes, key)
    # legado/profundidade antes de indices: protegidos dependem so dos indices
    base_idx = build_indexes(fields, idx_in, key, use_hint, rng, cnotes, key)
    protected = index_protected(base_idx)
    if use_hint.startswith("ts:"):
        _, tf, mf = use_hint.split(":")
        protected |= {tf, mf}
        fields[tf]["p"], fields[tf]["n"] = 1.0, 0.0
        fields[mf]["p"], fields[mf]["n"] = 1.0, 0.0
        for k in (tf, mf):
            fields[k].pop("alt", None)
        # time-series: sem unique/text/ttl/parcial/multikey/sparse especiais
        keep = []
        for ix in base_idx:
            ks = [k.lstrip("-") for k in ix["keys"].split(",")]
            if ix.get("unique") or ix.get("text") or ix.get("ttl") or ix.get("partial") or ix.get("wild") or ix.get("hashed") \
                    or ix.get("ci") or ix.get("geo") or ix.get("sparse") or not all(array_free(fields, k) for k in ks):
                continue
            keep.append(ix)
        base_idx = keep
        protected = index_protected(base_idx) | {tf, mf}
        for p_, s_, _ in walk(fields):
            pass
    if use_hint == "cap":
        base_idx = [ix for ix in base_idx if not ix.get("ttl")]
        protected = index_protected(base_idx)
    legacy = [] if use_hint.startswith("ts:") else inject_legacy(fields, protected, rng, cnotes, key)
    deep = None
    if rng.random() < 0.5 and not use_hint.startswith("ts:"):
        deep = inject_deep(fields, rng, cnotes, key)
    calibrate(fields, rng, protected)
    hs = Counter()
    if not use_hint.startswith("ts:"):
        hetero(fields, rng, protected, hs, "")
    for kk, v in hs.items():
        stats_total[kk] += v
    if hs:
        cnotes.append(f"{key}: heterogeneidade de tipo no mesmo campo: " + ", ".join(f"{k}={v}" for k, v in sorted(hs.items())))
    # embedding (vector search, so estatico)
    if not use_hint and srng.random() < 0.06 and not any(k.lower() == "embedding" for k in fields):
        fields["embedding"] = B.arr(B.dbl(-1, 1), 8, 8, p=0.5)
    return dict(key=key, pt_key=pt_key, fields=fields, idx=base_idx, hint=use_hint, notes=cnotes, hs=hs, deep=deep, legacy=legacy)


def build_schema(dom, brand, k_in_dom, seed):
    sid = f"v3_{brand}_{DS3[dom]}"
    rng = random.Random(f"{seed}:{sid}")
    srng = random.Random(f"{seed}:{sid}:search")
    pt = rng.random() < 0.25
    style = rng.choices(["snake", "camel", "pascal"], [0.45, 0.42, 0.13])[0]
    pool = pool_for(dom)
    n_target = rng.randint(4, 14)
    n_target = min(n_target, len(pool))
    pick = rng.sample(pool, n_target)
    keys = {e[0] for e in pick}
    changed = True
    while changed:
        changed = False
        for e in list(pick):
            for m in re.findall(r"refu?:(\w+)", e[2]):
                if m not in keys and len(pick) < 14 and any(x[0] == m for x in pool):
                    pick.append(next(x for x in pool if x[0] == m))
                    keys.add(m)
                    changed = True
    # remove coleções que ficaram sem utilidade? (mantem todas; refs ausentes sao podadas)
    notes = []
    stats_total = Counter()
    hint_budget = [2]
    built = []
    for ent in pick:
        r = random.Random(f"{seed}:{sid}:{ent[0]}")
        built.append(build_collection(ent, dom, pt, style, keys, r, srng, notes, hint_budget, stats_total))
    # ids por colecao
    idkind = {}
    for b in built:
        r = random.Random(f"{seed}:{sid}:{b['key']}:idkind")
        referenced = any(b["key"] in find_refs(s) for c in built for s in c["fields"].values())
        kind = "objectid"
        if not b["hint"].startswith("ts:") and b["hint"] != "cap":
            x = r.random()
            if x < 0.11:
                kind = "uuid"
            elif x < 0.20:
                kind = "string"
            elif x < 0.23:
                kind = "int64"
        idkind[b["key"]] = kind
    used_names = set()
    cols = []
    mstats = Counter()
    for b in built:
        r = random.Random(f"{seed}:{sid}:{b['key']}:names")
        mp = {}
        fields = rename_fields(b["fields"], r, style, "", "", mp, mstats)
        # indices (caminhos traduzidos)
        idxs = [final_index(ix, mp, fields) for ix in b["idx"]]
        retarget(fields, idkind)
        cname = coll_name(b["key"], b["pt_key"], pt, style, r, used_names, mstats)
        wild = [e for e in idxs if any(k.endswith(".$**") for k, _ in e["keys"])]
        idxs = [e for e in idxs if e not in wild]
        col = {"key": b["key"], "name": cname, "doc_count": doc_count(b["key"], b["hint"], r), "fields": fields, "indexes": idxs}
        if wild:  # wildcard fora de `indexes`: leitores antigos mapeiam chaves de indice -> campos
            col["extra_indexes"] = wild
        if idkind[b["key"]] != "objectid":
            kd = idkind[b["key"]]
            col["_id"] = {"t": kd, **({"prefix": b["key"][:3].upper() + "-"} if kd == "string" else {"base": 100000} if kd == "int64" else {})}
        opts_note = None
        if b["hint"].startswith("ts:"):
            _, tf, mf = b["hint"].split(":")
            col["options"] = {"timeseries": {"timeField": mp[tf], "metaField": mp[mf], "granularity": "hours"}}
            opts_note = f"{b['key']}: colecao time-series (timeField={mp[tf]}, metaField={mp[mf]})"
        elif b["hint"] == "cap":
            col["options"] = {"capped": True, "size": max(1 << 20, col["doc_count"] * 16384), "max": col["doc_count"] + 500}
            opts_note = f"{b['key']}: colecao capped (max={col['options']['max']} docs)"
        if opts_note:
            b["notes"].append(opts_note)
        if "options" not in col and r.random() < 0.55:
            col["validator"] = {"$jsonSchema": obj_schema(fields)}
            col["validationAction"] = "warn" if r.random() < 0.2 else "error"
            if r.random() < 0.15:
                col["validationLevel"] = "moderate"
        sr = random.Random(f"{seed}:{sid}:{b['key']}:search")
        sf = {k: s for k, s in fields.items() if not (s["t"] == "string" and s.get("g") in ("isodate", "numstr"))}
        has_emb = any(k.lower() == "embedding" for k in fields)
        if "options" not in col and (has_emb or sr.random() < 0.18):
            sd = B.search_def(sr, cname, sf)
            if sd:
                col["search_indexes"] = sd
        col["max_depth"] = B.depth(fields)
        col["v3"] = {
            "alt_fields": sum(1 for _, s, _a in walk(fields) if s.get("alt")), "map_fields": sum(1 for _, s, _a in walk(fields) if "map" in s),
            "geo_fields": sum(1 for _, s, _a in walk(fields) if "geo" in s), "rare_fields": sum(1 for _, s, _a in walk(fields) if s["p"] < 0.1),
            "array_of_arrays": sum(1 for _, s, _a in walk(fields) if s["t"] == "array" and s["item"]["t"] == "array"),
            "id_type": idkind[b["key"]], "kind": ("timeseries" if b["hint"].startswith("ts:") else "capped" if b["hint"] == "cap" else "collection"),
        }
        col["_notes"] = b["notes"]
        cols.append(col)
    # notas do schema
    allnotes = []
    for c in cols:
        allnotes += [f"{c['name']} ({n})" if False else n for n in c.pop("_notes")]
    alt_total = sum(c["v3"]["alt_fields"] for c in cols)
    kinds = Counter(k["v3"]["id_type"] for k in cols)
    allnotes.append("_id mistos: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) + " (referencias seguem o tipo do _id alvo)")
    names_msg = [f"{k}={v}" for k, v in sorted(mstats.items()) if v]
    allnotes.append(f"nomes de producao bagunçados (estilo dominante {style}; mistura de estilos, abreviacoes, prefixos de tipo, sufixos _v2/_old): "
                    + (", ".join(names_msg) or "sem ocorrencias extras"))
    allnotes.append(f"campos heterogeneos (alt) no schema: {alt_total}; mapas dinamicos: {sum(c['v3']['map_fields'] for c in cols)}; "
                    f"max_depth={max(c['max_depth'] for c in cols)}")
    allnotes.append(f"dominio {dom}: " + {
        "telecom": "CDR/assinantes/faturas com chaves de referencia ObjectId|string e mapas por plano/dia",
        "adtech": "eventos de leilao/impressao/clique em alto volume, relatorios com mapas de metricas",
        "retail-pos": "vendas PDV com linhas, pagamentos divididos e fechamentos de caixa",
        "manufacturing": "BOM aninhado, ordens de producao, leituras de maquina e paradas (time-series)",
    }.get(dom, "entidades do dominio com extensoes v3 (uniao de tipos, mapas, aninhamento profundo)"))
    schema = {
        "schema_id": sid, "domain": dom, "holdout": dom in HOLDOUT,
        "description": f"{dom} v3 realista {'pt-BR sem acento' if pt else 'EN'}, {style}",
        "name_style": style, "seed": seed, "format_version": 2, "generation": "v3",
        "language": "pt-BR" if pt else "en",
        "realism_notes": allnotes,
        "collections": cols,
    }
    return schema


def plan_ids(seed):
    out = []
    for dom, n in PLAN.items():
        r = random.Random(f"{seed}:brands:{dom}")
        brands = r.sample(BRANDS, n)
        for k, br in enumerate(brands):
            out.append((dom, br, k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(ROOT / "data" / "schemas"))
    a = ap.parse_args()
    out = Path(a.out)
    built = [build_schema(dom, br, k, a.seed) for dom, br, k in plan_ids(a.seed)]
    roles = assign_roles_v3([{k: s[k] for k in ("schema_id", "domain", "holdout", "name_style", "language")} for s in built], a.seed)
    nc = 0
    for s in built:
        s = {**{k: v for k, v in s.items() if k != "collections"}, "split_role": roles[s["schema_id"]], "collections": s["collections"]}
        # ordem estavel de chaves: split_role antes de realism_notes nao importa; json ordenado por insercao
        p = out / s["domain"] / f"{s['schema_id']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(s, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        nc += len(s["collections"])
        print(f"{s['domain']:22s} {s['schema_id']:28s} {s['split_role']:5s} colls={len(s['collections']):2d} "
              f"docs={sum(c['doc_count'] for c in s['collections']):6d} depth<={max(c['max_depth'] for c in s['collections'])}")
    print(f"schemas v3={len(built)} colecoes={nc} docs_planejados={sum(c['doc_count'] for s in built for c in s['collections'])}")


if __name__ == "__main__":
    sys.exit(main())
