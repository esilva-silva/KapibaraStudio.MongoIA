#!/usr/bin/env python3
"""Gerador de snippets Mongo Console tipado pelo schema (spec fase-03 §4.3 `gen_snippets.py`).

  python scripts/data/gen_snippets.py --seed 20261008 --count 2000 --ver v0

Saida: data/generated/<ver>/snippets.jsonl (+ snippets.manifest.json), deterministica por --seed.
Cada registro carrega o texto Console, os *spans* sintaticos (para os cortes FIM) e o plano
estruturado de execucao (EJSON proprio), de modo que a validacao nao precisa parsear JS.

Le data/schemas/<dominio>/<schema_id>.json defensivamente: usa so t/p/n/lo/hi/values/g/f/item/ref/
min/max/prefix; ignora chaves desconhecidas. Python 3.12, so stdlib + PyYAML.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "dataset.yaml"

# --------------------------------------------------------------------------- config

def load_config(path: Path | str | None = None) -> dict:
    p = Path(path) if path else CONFIG_PATH
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:12], 16)


def split_key(group_id: str) -> str:
    return hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:16]


def assign_split(group_id: str, domain: str, template_id: str, cfg: dict, holdout: bool = False) -> str:
    """Split por GRUPO (nunca por exemplo). Holdout so test/bench; templates reservados -> test."""
    sp = cfg.get("splits", {})
    h = stable_hash("split:" + group_id) % 10000 / 10000.0
    if holdout or domain in set(cfg.get("holdout_domains", [])):
        return "bench" if h < sp.get("holdout_bench_fraction", 0.4) else "test"
    if stable_hash("tpl:" + template_id) % 1000 / 1000.0 < sp.get("reserved_template_fraction", 0.10):
        return "test"
    edges = []
    acc = 0.0
    for name in ("train", "val", "test", "bench"):
        acc += sp.get(name, 0)
        edges.append((acc, name))
    for edge, name in edges:
        if h < edge:
            return name
    return "train"


# --------------------------------------------------------------------------- schema model

TYPE_ALIASES = {
    "string": "string", "str": "string", "text": "string",
    "int32": "int32", "int": "int32", "integer": "int32",
    "int64": "int64", "long": "int64",
    "double": "double", "float": "double", "number": "double",
    "decimal128": "decimal128", "decimal": "decimal128",
    "bool": "bool", "boolean": "bool",
    "date": "date", "datetime": "date", "timestamp": "date",
    "objectid": "objectid", "oid": "objectid",
    "uuid": "uuid",
    "object": "object", "document": "object",
    "array": "array",
}
NUMERIC = {"int32", "int64", "double", "decimal128"}
SCALAR = NUMERIC | {"string", "bool", "date", "objectid", "uuid"}

WORDS = ("alpha beta gamma delta sigma omega river stone cloud ember falcon harbor island jungle kernel lantern marble nectar "
         "orbit pixel quartz raven summit tundra umbra velvet willow xenon yonder zephyr amber bronze cedar dune echo flint "
         "grove haze iris jade kite lotus mesa nova opal pine quill reef sage thorn urn vale wren yarrow zinc").split()
FIRST = "Ana Bruno Carla Diego Elisa Felipe Gabi Hugo Iara Joao Karen Lucas Marina Nuno Olivia Pedro Quiteria Rafael Sofia Tiago Ursula Vitor".split()
LAST = "Silva Souza Costa Lima Alves Rocha Dias Melo Nunes Pinto Ramos Cardoso Moreira Barros Teixeira Gomes Freitas".split()
CITIES = ["Sao Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba", "Porto Alegre", "Salvador", "Recife", "Fortaleza", "Brasilia",
          "Lisboa", "Madrid", "Austin", "Denver", "Toronto", "Osaka", "Pune"]
COUNTRIES = ["BR", "PT", "ES", "US", "CA", "JP", "IN", "AR", "MX", "DE"]
try:  # alinha vocabulario com o seeder quando disponivel (somente leitura)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib
    _sf = importlib.import_module("seed_fixtures")
    WORDS = list(getattr(_sf, "WORDS", WORDS))
    FIRST = list(getattr(_sf, "FIRST", FIRST))
    LAST = list(getattr(_sf, "LAST", LAST))
    CITIES = list(getattr(_sf, "CITIES", CITIES))
    COUNTRIES = list(getattr(_sf, "COUNTRIES", COUNTRIES))
except Exception:
    pass


def norm_type(t) -> str | None:
    if not isinstance(t, str):
        return None
    return TYPE_ALIASES.get(t.strip().lower())


class F:
    """Campo achatado (caminho pontilhado)."""
    __slots__ = ("path", "t", "item_t", "spec", "item_spec", "via_array", "ref", "nullable", "presence")

    def __init__(self, path, t, spec, item_t=None, item_spec=None, via_array=False):
        self.path, self.t, self.spec = path, t, spec
        self.item_t, self.item_spec, self.via_array = item_t, item_spec, via_array
        self.ref = spec.get("ref") if isinstance(spec, dict) else None
        self.nullable = bool(spec.get("n", 0)) if isinstance(spec, dict) else False
        self.presence = spec.get("p", 1.0) if isinstance(spec, dict) else 1.0

    @property
    def leaf_t(self):
        return self.item_t if self.t == "array" else self.t


class Coll:
    def __init__(self, raw: dict, sch: "Schema"):
        self.key = str(raw.get("key") or raw.get("name"))
        self.name = str(raw.get("name") or self.key)
        self.doc_count = int(raw.get("doc_count") or 100)
        self.fields: dict[str, F] = {"_id": F("_id", "objectid", {"t": "objectid", "p": 1.0, "n": 0.0})}
        self.indexes = raw.get("indexes") or []
        self.sch = sch
        self._flatten(raw.get("fields") or {}, "", False)

    def _flatten(self, fields: dict, prefix: str, via_array: bool):
        if not isinstance(fields, dict):
            return
        for name, spec in fields.items():
            if not isinstance(spec, dict) or not isinstance(name, str) or not name or "." in name or name.startswith("$"):
                continue
            t = norm_type(spec.get("t"))
            if t is None:
                continue
            path = prefix + name
            if t == "object":
                self.fields[path] = F(path, t, spec, via_array=via_array)
                self._flatten(spec.get("f") or {}, path + ".", via_array)
            elif t == "array":
                item = spec.get("item") if isinstance(spec.get("item"), dict) else {}
                it = norm_type(item.get("t"))
                self.fields[path] = F(path, t, spec, item_t=it, item_spec=item, via_array=via_array)
                if it == "object":
                    self._flatten(item.get("f") or {}, path + ".", True)
            else:
                self.fields[path] = F(path, t, spec, via_array=via_array)


class Schema:
    def __init__(self, raw: dict, path: Path):
        self.domain = str(raw.get("domain") or path.parent.name)
        self.schema_id = str(raw.get("schema_id") or path.stem)
        self.holdout = bool(raw.get("holdout", False))
        self.name_style = str(raw.get("name_style") or "camel")
        self.seed = raw.get("seed")
        self.description = raw.get("description", "")
        self.colls = [Coll(c, self) for c in (raw.get("collections") or []) if isinstance(c, dict)]
        self.by_key = {c.key: c for c in self.colls}
        for c in self.colls:
            self.by_key.setdefault(c.name, c)

    @property
    def full_id(self):
        return f"{self.domain}/{self.schema_id}"

    @property
    def db_name(self):
        return f"kapi_fx_{self.domain.replace('-', '_')}_{self.schema_id}"[:60].lower()


def load_schemas(schemas_dir: Path) -> list[Schema]:
    out = []
    for p in sorted(Path(schemas_dir).glob("*/*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            sch = Schema(raw, p)
            if sch.colls:
                out.append(sch)
        except Exception as e:  # leitura defensiva: um schema quebrado nao derruba o resto
            print(f"[warn] schema ignorado {p}: {e}", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- value classes

class Ctor:
    """Construtor BSON do Console (ObjectId, ISODate, NumberLong, NumberDecimal, UUID)."""
    __slots__ = ("kind", "val")

    def __init__(self, kind, val):
        self.kind, self.val = kind, val

    def ejson(self):
        k = self.kind
        if k == "oid":
            return {"$oid": self.val}
        if k == "date":
            return {"$date": self.val}
        if k == "i64":
            return {"$numberLong": self.val}
        if k == "dec":
            return {"$numberDecimal": self.val}
        return {"$uuid": self.val}

    def js(self, q, date_form):
        if self.kind == "date" and date_form == "new_date":
            return f"new Date({q}{self.val}{q})"
        name = {"oid": "ObjectId", "date": "ISODate", "i64": "NumberLong", "dec": "NumberDecimal", "uuid": "UUID"}[self.kind]
        return f"{name}({q}{self.val}{q})"


class Ident:
    """Referencia a variavel (const filter = ...)."""
    __slots__ = ("name", "value")

    def __init__(self, name, value):
        self.name, self.value = name, value


EJSON_KEYS = {"$oid", "$date", "$numberLong", "$numberDecimal", "$uuid"}


def to_ej(v):
    if isinstance(v, Ctor):
        return v.ejson()
    if isinstance(v, Ident):
        return to_ej(v.value)
    if isinstance(v, dict):
        return {k: to_ej(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_ej(x) for x in v]
    return v


def style_name(words: list[str], style: str) -> str:
    ws = [w for w in (x.lower() for x in words) if w]
    if not ws:
        return "value"
    if style == "snake":
        return "_".join(ws)
    if style == "pascal":
        return "".join(w.capitalize() for w in ws)
    return ws[0] + "".join(w.capitalize() for w in ws[1:])


def split_words(name: str) -> list[str]:
    name = name.replace("_", " ")
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [w.lower() for w in name.split() if w]


IDENT_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
RESERVED_COLL = {"stats", "count", "find", "drop", "help", "constructor", "name", "length"}


# --------------------------------------------------------------------------- generator

class Builder:
    """Gera um plano estruturado + tags de template, tipado pelo schema."""

    def __init__(self, rng: random.Random, sch: Schema, coll: Coll, cfg: dict):
        self.rng, self.sch, self.coll, self.cfg = rng, sch, coll, cfg
        self.tags: list[str] = []
        self.var_counter = 0

    # ---- scope -------------------------------------------------------------
    def root_scope(self) -> dict[str, F]:
        return dict(self.coll.fields)

    @staticmethod
    def scalars(scope, exclude_arr=False):
        return [p for p, f in scope.items() if f.t in SCALAR and not (exclude_arr and f.via_array)]

    def pick(self, seq, k=1):
        seq = list(seq)
        if not seq:
            return []
        return self.rng.sample(seq, min(k, len(seq)))

    # ---- values ------------------------------------------------------------
    def doc_idx(self, coll: Coll):
        return self.rng.randrange(max(1, coll.doc_count))

    def oid_for(self, coll: Coll) -> str:
        """Replica a derivacao de _id do seeder para acertar documentos reais (best effort)."""
        seed = self.sch.seed if isinstance(self.sch.seed, int) else self.cfg.get("seed", 20261008)
        i = self.doc_idx(coll)
        r = random.Random(f"{seed}:{self.sch.schema_id}:{coll.key}:ids")
        data = b""
        for _ in range(i + 1):
            data = r.randbytes(12)
        return data.hex()

    def sample(self, f: F, coll: Coll | None = None):
        rng = self.rng
        t = f.leaf_t if f.t == "array" else f.t
        spec = f.item_spec if f.t == "array" else f.spec
        spec = spec if isinstance(spec, dict) else {}
        if t == "string":
            return self.sample_string(spec, coll or self.coll)
        if t == "bool":
            return rng.random() < 0.5
        if t in ("int32", "int64"):
            lo, hi = int(spec.get("lo", 0)), int(spec.get("hi", 1000))
            hi = max(hi, lo)
            v = rng.randint(lo, hi)
            if t == "int64" and rng.random() < 0.3:
                return Ctor("i64", str(v))
            return v
        if t in ("double", "decimal128"):
            lo, hi = float(spec.get("lo", 0)), float(spec.get("hi", 1000))
            hi = max(hi, lo)
            v = rng.uniform(lo, hi)
            if t == "decimal128" and rng.random() < 0.3:
                return Ctor("dec", f"{v:.2f}")
            return round(v, 1) if hi - lo > 20 else round(v, 2)
        if t == "date":
            return Ctor("date", self.sample_date(spec) + "T00:00:00Z")
        if t == "objectid":
            ref = f.ref
            if ref and ref in self.sch.by_key:
                return Ctor("oid", self.oid_for(self.sch.by_key[ref]))
            return Ctor("oid", rng.randbytes(12).hex())
        if t == "uuid":
            return Ctor("uuid", str(__import__("uuid").UUID(int=rng.getrandbits(128), version=4)))
        return None

    def sample_date(self, spec) -> str:
        lo = str(spec.get("lo", "2025-01-01"))[:10]
        hi = str(spec.get("hi", "2026-09-30"))[:10]
        try:
            a, b = date.fromisoformat(lo), date.fromisoformat(hi)
        except ValueError:
            a, b = date(2025, 1, 1), date(2026, 9, 30)
        days = max(0, (b - a).days)
        return (a + timedelta(days=self.rng.randint(0, days))).isoformat()

    def sample_string(self, spec, coll: Coll):
        rng = self.rng
        g = spec.get("g")
        if g == "enum" and spec.get("values"):
            return rng.choice(spec["values"])
        if g == "sku":
            return f"SKU-{rng.randrange(coll.doc_count):05d}"
        if g == "name":
            return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if g == "city":
            return rng.choice(CITIES)
        if g == "country":
            return rng.choice(COUNTRIES)
        if g == "code":
            pre = spec.get("prefix")
            return f"{pre}{rng.randrange(coll.doc_count):06d}" if pre else f"{rng.randrange(10000):04d}"
        if g == "email":
            return f"{rng.choice(FIRST).lower()}.{rng.choice(LAST).lower()}{rng.randrange(coll.doc_count)}@example.com"
        if g == "title":
            return " ".join(w.capitalize() for w in rng.sample(WORDS, 2))
        return rng.choice(WORDS)

    def regex_for(self, f: F):
        s = self.sample(f)
        if not isinstance(s, str) or len(s) < 2:
            s = self.rng.choice(WORDS)
        n = self.rng.randint(1, min(4, len(s)))
        return "^" + re.escape(s[:n]) if self.rng.random() < 0.6 else re.escape(s[:max(2, n)])

    # ---- condicoes por tipo --------------------------------------------------
    def cond(self, p: str, f: F):
        """Retorna (valor_da_condicao, tag). Operadores compativeis com o tipo do campo."""
        rng = self.rng
        t = f.t
        r = rng.random()
        if f.nullable and r < 0.06:
            return None, "null"
        if t == "string":
            if r < 0.30:
                return self.sample(f), "eq"
            if r < 0.48:
                return {"$in": [self.sample(f) for _ in range(rng.randint(2, 4))]}, "in"
            if r < 0.66:
                op = {"$regex": self.regex_for(f)}
                if rng.random() < 0.4:
                    op["$options"] = "i"
                return op, "regex"
            if r < 0.76:
                return {"$ne": self.sample(f)}, "ne"
            if r < 0.84:
                return {"$nin": [self.sample(f) for _ in range(rng.randint(1, 3))]}, "nin"
            if r < 0.92:
                return {"$exists": rng.random() < 0.8}, "exists"
            return {rng.choice(["$gte", "$lt"]): self.sample(f)}, "range"
        if t in NUMERIC:
            if r < 0.15:
                return self.sample(f), "eq"
            if r < 0.50:
                return {rng.choice(["$gt", "$gte", "$lt", "$lte"]): self.sample(f)}, "cmp"
            if r < 0.72:
                a, b = sorted([self.sample(f), self.sample(f)], key=lambda x: float(x.val) if isinstance(x, Ctor) else x)
                lo_op, hi_op = rng.choice([("$gte", "$lte"), ("$gt", "$lt"), ("$gte", "$lt")])
                return {lo_op: a, hi_op: b}, "range"
            if r < 0.82:
                return {"$in": [self.sample(f) for _ in range(rng.randint(2, 4))]}, "in"
            if r < 0.88:
                return {"$ne": self.sample(f)}, "ne"
            if r < 0.93 and t in ("int32", "int64"):
                return {"$mod": [rng.choice([2, 3, 5, 10]), 0]}, "mod"
            if r < 0.97:
                return {"$not": {rng.choice(["$gt", "$lt"]): self.sample(f)}}, "not"
            return {"$exists": True}, "exists"
        if t == "date":
            if r < 0.40:
                return {rng.choice(["$gte", "$gt"]): self.sample(f)}, "cmp"
            if r < 0.55:
                return {rng.choice(["$lt", "$lte"]): self.sample(f)}, "cmp"
            if r < 0.88:
                a, b = sorted([self.sample(f), self.sample(f)], key=lambda x: x.val)
                return {"$gte": a, "$lt": b}, "range"
            if r < 0.94:
                return {"$exists": True}, "exists"
            return {"$type": "date"}, "type"
        if t == "bool":
            if r < 0.80:
                return rng.random() < 0.5, "eq"
            if r < 0.92:
                return {"$ne": rng.random() < 0.5}, "ne"
            return {"$exists": True}, "exists"
        if t in ("objectid", "uuid"):
            if r < 0.55:
                return self.sample(f), "eq"
            if r < 0.80:
                return {"$in": [self.sample(f) for _ in range(rng.randint(2, 3))]}, "in"
            if r < 0.92:
                return {"$ne": self.sample(f)}, "ne"
            return {"$exists": rng.random() < 0.85}, "exists"
        if t == "array":
            it = f.item_t
            if it in SCALAR:
                if r < 0.25:
                    return self.sample(f), "eq"
                if r < 0.45:
                    return {"$in": [self.sample(f) for _ in range(rng.randint(1, 3))]}, "in"
                if r < 0.60:
                    return {"$all": [self.sample(f) for _ in range(rng.randint(1, 2))]}, "all"
                if r < 0.75:
                    return {"$size": rng.randint(0, 4)}, "size"
                if r < 0.88 and it in NUMERIC:
                    return {"$elemMatch": {"$gt": self.sample(f)}}, "elem"
                return {"$exists": True}, "exists"
            if r < 0.5:
                return {"$size": rng.randint(0, 4)}, "size"
            return {"$exists": rng.random() < 0.8}, "exists"
        if t == "object":
            return {"$exists": rng.random() < 0.8}, "exists"
        return {"$exists": True}, "exists"

    def elem_children(self, scope, arr_path: str):
        pre = arr_path + "."
        return [p for p, f in scope.items() if p.startswith(pre) and f.t in SCALAR and "." not in p[len(pre):]]

    def gen_filter(self, scope, depth=0, n=None, allow_complex=True, allow_empty=False):
        rng = self.rng
        cand = [p for p, f in scope.items() if f.t in SCALAR or f.t == "array" or (f.t == "object")]
        cand = [p for p in cand if not (p == "_id" and rng.random() < 0.7)]
        if not cand:
            return {}
        if n is None:
            n = rng.choices([1, 2, 3], weights=[40, 40, 20])[0]
        if allow_empty and rng.random() < 0.05:
            self.tags.append("empty")
            return {}
        # evita pai+filho no mesmo filtro (conflito semantico, nao erro, mas confuso)
        chosen: list[str] = []
        for p in rng.sample(cand, len(cand)):
            if any(p.startswith(c + ".") or c.startswith(p + ".") for c in chosen):
                continue
            chosen.append(p)
            if len(chosen) >= n:
                break
        flt: dict = {}
        for p in chosen:
            f = scope[p]
            if f.t == "array" and f.item_t == "object" and rng.random() < 0.5:
                kids = self.elem_children(scope, p)
                if kids:
                    sub = {}
                    for kp in self.pick(kids, rng.randint(1, 2)):
                        v, tag = self.cond(kp, scope[kp])
                        sub[kp[len(p) + 1:]] = v
                        self.tags.append("elem_" + tag)
                    flt[p] = {"$elemMatch": sub}
                    continue
            v, tag = self.cond(p, f)
            flt[p] = v
            self.tags.append(tag)
        if allow_complex and depth == 0 and len(cand) >= 4:
            r = rng.random()
            if r < 0.14:
                a = self.gen_filter(scope, 1, n=1, allow_complex=False)
                b = self.gen_filter(scope, 1, n=1, allow_complex=False)
                if a and b and a != b:
                    self.tags.append("or")
                    flt = {"$or": [a, b]} if not flt or rng.random() < 0.4 else {**flt, "$or": [a, b]}
            elif r < 0.19:
                a = self.gen_filter(scope, 1, n=1, allow_complex=False)
                b = self.gen_filter(scope, 1, n=1, allow_complex=False)
                if a and b and a != b:
                    self.tags.append("and")
                    flt = {"$and": [a, b]}
            elif r < 0.23:
                nums = [p for p in cand if scope[p].t in NUMERIC]
                if len(nums) >= 2:
                    a, b = self.pick(nums, 2)
                    self.tags.append("expr")
                    flt["$expr"] = {rng.choice(["$gt", "$lt", "$gte"]): ["$" + a, "$" + b]}
        return flt

    # ---- projecao / sort ------------------------------------------------------
    def gen_projection(self, scope):
        rng = self.rng
        leaves = [p for p, f in scope.items() if f.t in SCALAR or f.t in ("array", "object")]
        leaves = [p for p in leaves if p != "_id"]
        if not leaves:
            return None
        picks: list[str] = []
        for p in rng.sample(leaves, len(leaves)):
            if any(p.startswith(c + ".") or c.startswith(p + ".") for c in picks):
                continue
            picks.append(p)
            if len(picks) >= rng.randint(1, 4):
                break
        if rng.random() < 0.82:
            proj = {p: 1 for p in picks}
            if rng.random() < 0.3:
                proj["_id"] = 0
            self.tags.append("proj")
        else:
            proj = {p: 0 for p in picks[:2]}
            self.tags.append("proj_excl")
        return proj

    def gen_sort(self, scope, n=None):
        rng = self.rng
        cand = [p for p, f in scope.items() if f.t in SCALAR and f.t != "bool"]
        if not cand:
            return None
        n = n or rng.choices([1, 2], weights=[75, 25])[0]
        keys = self.pick(cand, n)
        # "cannot sort with keys that are parallel arrays": no maximo uma chave que atravesse array
        keys = [k for i, k in enumerate(keys) if not (scope[k].via_array and any(scope[x].via_array for x in keys[:i]))]
        return {k: rng.choice([1, -1]) for k in keys}

    # ---- expressoes tipadas ---------------------------------------------------
    def out_name(self, hint_words, taken: set[str]):
        base = style_name(hint_words, self.sch.name_style)
        name, i = base, 2
        while name in taken or name in ("_id",) or not name:
            name = f"{base}{i}" if self.sch.name_style != "snake" else f"{base}_{i}"
            i += 1
        taken.add(name)
        return name

    def gen_expr(self, scope):
        """Expressao para $project/$addFields. Retorna (expr, tipo_saida, palavras_nome)."""
        rng = self.rng
        by = lambda pred: [p for p, f in scope.items() if pred(f)]
        nums = by(lambda f: f.t in NUMERIC and not f.via_array)
        strs = by(lambda f: f.t == "string" and not f.via_array)
        dates = by(lambda f: f.t == "date" and not f.via_array)
        arrs = by(lambda f: f.t == "array" and not f.via_array)
        options = []
        if nums:
            options += ["mul", "round", "cond"] * 2
        if len(nums) >= 2:
            options += ["add", "sub"]
        if strs:
            options += ["upper", "ifnull_s"]
        if len(strs) >= 2:
            options += ["concat"]
        if dates:
            options += ["year", "dts"]
        if arrs:
            options += ["size"]
        if not options:
            return None
        k = rng.choice(options)
        if k == "mul":
            p = rng.choice(nums)
            return {"$multiply": ["$" + p, rng.choice([2, 10, 1.1, 0.5])]}, "double", split_words(p.split(".")[-1]) + ["x"]
        if k == "round":
            p = rng.choice(nums)
            return {"$round": ["$" + p, rng.choice([0, 1, 2])]}, "double", split_words(p.split(".")[-1]) + ["rounded"]
        if k == "cond":
            p = rng.choice(nums)
            thr = self.sample(scope[p])
            if isinstance(thr, Ctor):
                thr = float(thr.val)
            return {"$cond": [{"$gte": ["$" + p, thr]}, "high", "low"]}, "string", split_words(p.split(".")[-1]) + ["band"]
        if k in ("add", "sub"):
            a, b = self.pick(nums, 2)
            return {"$add" if k == "add" else "$subtract": ["$" + a, "$" + b]}, "double", ["diff" if k == "sub" else "sum", "calc"]
        if k == "upper":
            p = rng.choice(strs)
            return {"$toUpper": "$" + p}, "string", split_words(p.split(".")[-1]) + ["upper"]
        if k == "ifnull_s":
            p = rng.choice(strs)
            return {"$ifNull": ["$" + p, "n/a"]}, "string", split_words(p.split(".")[-1]) + ["safe"]
        if k == "concat":
            a, b = self.pick(strs, 2)
            return {"$concat": ["$" + a, " - ", "$" + b]}, "string", ["label"]
        if k == "year":
            p = rng.choice(dates)
            return {"$year": "$" + p}, "int32", split_words(p.split(".")[-1]) + ["year"]
        if k == "dts":
            p = rng.choice(dates)
            return {"$dateToString": {"format": rng.choice(["%Y-%m-%d", "%Y-%m"]), "date": "$" + p}}, "string", split_words(p.split(".")[-1]) + ["day"]
        p = rng.choice(arrs)
        return {"$size": {"$ifNull": ["$" + p, []]}}, "int32", split_words(p.split(".")[-1]) + ["count"]

    @staticmethod
    def mk(t, spec=None, **kw):
        fl = F("", t, spec or {"t": t, "p": 1.0, "n": 0.0})
        return fl

    def with_path(self, path, f: F, via_array=None) -> F:
        nf = F(path, f.t, f.spec, f.item_t, f.item_spec, f.via_array if via_array is None else via_array)
        return nf

    def copy_subtree(self, scope, new_scope, src: str, dst: str):
        for p, f in scope.items():
            if p == src or p.startswith(src + "."):
                np_ = dst + p[len(src):]
                new_scope[np_] = self.with_path(np_, f)

    # ---- estagios --------------------------------------------------------------
    def stage_match(self, scope):
        flt = self.gen_filter(scope, n=self.rng.choices([1, 2], weights=[65, 35])[0], allow_empty=False)
        if not flt:
            return None, scope
        self.tags.append("match")
        return {"$match": flt}, scope

    def stage_group(self, scope):
        rng = self.rng
        keys = [p for p, f in scope.items() if f.t in ("string", "bool", "int32", "objectid", "date") and not f.via_array
                and f.t != "uuid" and (p != "_id" or rng.random() < 0.1)]
        datep = [p for p, f in scope.items() if f.t == "date" and not f.via_array]
        nums = [p for p, f in scope.items() if f.t in NUMERIC and not f.via_array]
        new_scope: dict[str, F] = {}
        taken: set[str] = {"_id"}
        r = rng.random()
        if r < 0.10 or not keys:
            gid, gtag = None, "all"
            new_scope["_id"] = self.mk("object")
        elif r < 0.30 and len(keys) >= 2:
            a, b = self.pick(keys, 2)
            na, nb = style_name(split_words(a.split(".")[-1]), self.sch.name_style), style_name(split_words(b.split(".")[-1]), self.sch.name_style)
            if na == nb:
                nb += "2"
            gid = {na: "$" + a, nb: "$" + b}
            gtag = "key2"
            new_scope["_id"] = self.mk("object")
            new_scope["_id." + na] = self.with_path("_id." + na, scope[a], False)
            new_scope["_id." + nb] = self.with_path("_id." + nb, scope[b], False)
        elif r < 0.45 and datep:
            d = rng.choice(datep)
            if rng.random() < 0.5:
                gid = {"$dateToString": {"format": rng.choice(["%Y-%m", "%Y-%m-%d"]), "date": "$" + d}}
                new_scope["_id"] = self.mk("string")
            else:
                gid = {"$year": "$" + d}
                new_scope["_id"] = self.mk("int32")
            gtag = "keydate"
        else:
            k = rng.choice(keys)
            gid, gtag = "$" + k, "key"
            new_scope["_id"] = self.with_path("_id", scope[k], False)
        accs: dict = {}
        acc_tags = []
        n_acc = rng.randint(1, 3)
        for _ in range(n_acc):
            kind = rng.choice(["sum1", "sumf", "avg", "min", "max", "first", "push", "addset"] if nums else ["sum1", "first", "push", "addset", "min", "max"])
            if kind == "sum1":
                nm = self.out_name(rng.choice([["count"], ["total"], ["n"], ["orders", "count"]]), taken)
                accs[nm] = {"$sum": 1}
                new_scope[nm] = self.mk("int32")
                acc_tags.append("count")
            elif kind in ("sumf", "avg", "min", "max") and nums:
                p = rng.choice(nums)
                op = {"sumf": "$sum", "avg": "$avg", "min": "$min", "max": "$max"}[kind]
                word = {"sumf": "total", "avg": "avg", "min": "min", "max": "max"}[kind]
                nm = self.out_name([word] + split_words(p.split(".")[-1]), taken)
                accs[nm] = {op: "$" + p}
                new_scope[nm] = self.with_path(nm, scope[p], False) if kind in ("min", "max") else self.mk("double")
                acc_tags.append({"sumf": "sum", "avg": "avg", "min": "min", "max": "max"}[kind])
            else:
                cand = [p for p, f in scope.items() if f.t in ("string", "int32", "objectid", "date") and not f.via_array]
                if not cand:
                    continue
                p = rng.choice(cand)
                op = {"first": "$first", "push": "$push", "addset": "$addToSet"}.get(kind, "$first")
                if kind in ("min", "max"):
                    op = "$" + kind
                nm = self.out_name(([kind if kind != "addset" else "unique"] if kind not in ("min", "max") else [kind]) + split_words(p.split(".")[-1]), taken)
                accs[nm] = {op: "$" + p}
                if op in ("$push", "$addToSet"):
                    nf = F(nm, "array", {"t": "array", "p": 1.0, "n": 0.0}, item_t=scope[p].t, item_spec=scope[p].spec)
                    new_scope[nm] = nf
                else:
                    new_scope[nm] = self.with_path(nm, scope[p], False)
                acc_tags.append(op[1:])
        if not accs:
            nm = self.out_name(["count"], taken)
            accs[nm] = {"$sum": 1}
            new_scope[nm] = self.mk("int32")
            acc_tags.append("count")
        self.tags.append("group_" + "_".join(sorted(set(acc_tags))) + ("" if gtag == "key" else "_" + gtag))
        return {"$group": {"_id": gid, **accs}}, new_scope

    def stage_project(self, scope):
        rng = self.rng
        taken: set[str] = set(scope)
        new_scope: dict[str, F] = {}
        leaves = [p for p, f in scope.items() if (f.t in SCALAR or f.t in ("array",)) and p != "_id"]
        spec: dict = {}
        chosen: list[str] = []
        for p in rng.sample(leaves, len(leaves)):
            if any(p.startswith(c + ".") or c.startswith(p + ".") for c in chosen):
                continue
            chosen.append(p)
            if len(chosen) >= rng.randint(1, 3):
                break
        for p in chosen:
            spec[p] = 1
            new_scope[p] = scope[p]
        tags = ["incl"]
        for _ in range(rng.choices([0, 1, 2], weights=[45, 40, 15])[0]):
            e = self.gen_expr(scope)
            if not e:
                break
            expr, t, words = e
            nm = self.out_name(words, taken)
            if nm in spec:
                continue
            spec[nm] = expr
            new_scope[nm] = self.mk(t)
            tags.append("expr")
        if rng.random() < 0.3 and rng.random() < 0.6 and "_id" in scope:
            spec["_id"] = 0
        elif "_id" in scope:
            new_scope["_id"] = scope["_id"]
        if not spec or all(v == 0 for v in spec.values()):
            return None, scope
        if rng.random() < 0.15 and chosen:  # rename via alias
            p = chosen[0]
            if "." not in p and len(spec) > 1:
                pass
        self.tags.append("project" if "expr" not in tags else "project_expr")
        return {"$project": spec}, new_scope

    def stage_sort(self, scope):
        s = self.gen_sort(scope)
        if not s:
            return None, scope
        self.tags.append("sort")
        return {"$sort": s}, scope

    def stage_limit(self, scope):
        self.tags.append("limit")
        return {"$limit": self.rng.choice([5, 10, 20, 50, 100])}, scope

    def stage_skip(self, scope):
        self.tags.append("skip")
        return {"$skip": self.rng.choice([5, 10, 20])}, scope

    def stage_unwind(self, scope):
        arrs = [p for p, f in scope.items() if f.t == "array" and not f.via_array and p != "_id"]
        if not arrs:
            return None, scope
        p = self.rng.choice(arrs)
        f = scope[p]
        new = dict(scope)
        nf = F(p, f.item_t or "object", f.item_spec if isinstance(f.item_spec, dict) else {"t": f.item_t or "object"}, via_array=False)
        new[p] = nf
        # filhos deixam de ser "via array" (cada documento agora tem um unico elemento)
        for q, g in list(new.items()):
            if q.startswith(p + "."):
                new[q] = self.with_path(q, g, False)
        if self.rng.random() < 0.25:
            self.tags.append("unwind_keep")
            return {"$unwind": {"path": "$" + p, "preserveNullAndEmptyArrays": True}}, new
        self.tags.append("unwind")
        return {"$unwind": "$" + p}, new

    def stage_lookup(self, scope):
        rng = self.rng
        refs = [p for p, f in scope.items() if f.ref and f.ref in self.sch.by_key and f.t == "objectid" and not f.via_array]
        if not refs:
            return None, scope
        lp = rng.choice(refs)
        f = scope[lp]
        fc = self.sch.by_key[f.ref]
        taken = set(scope)
        as_name = self.out_name([fc.name.replace("-", " ").replace("_", " ").split()[-1].rstrip("s") or "ref", "doc"] if rng.random() < 0.5 else [f.ref, "info"], taken)
        as_name = re.sub(r"[^A-Za-z0-9_]", "", as_name) or "joined"
        if as_name in scope:
            as_name += "X"
        new = dict(scope)
        new[as_name] = F(as_name, "array", {"t": "array", "p": 1.0, "n": 0.0}, item_t="object", item_spec={"t": "object"})
        if rng.random() < 0.7:
            body = {"from": fc.name, "localField": lp, "foreignField": "_id", "as": as_name}
            self.tags.append("lookup")
        else:
            var = style_name(["v"] + split_words(lp.split(".")[-1]), "camel")
            inner = [{"$match": {"$expr": {"$eq": ["$_id", "$$" + var]}}}]
            fsc = {p: ff for p, ff in fc.fields.items()}
            if rng.random() < 0.5:
                sub = self.gen_projection({p: ff for p, ff in fsc.items()})
                if sub and all(v == 1 for k, v in sub.items() if k != "_id"):
                    inner.append({"$project": sub})
            body = {"from": fc.name, "let": {var: "$" + lp}, "pipeline": inner, "as": as_name}
            self.tags.append("lookup_pipe")
        for p, ff in fc.fields.items():
            np_ = f"{as_name}.{p}"
            new[np_] = F(np_, ff.t, ff.spec, ff.item_t, ff.item_spec, via_array=True)
        return {"$lookup": body}, new

    def stage_addfields(self, scope):
        e = self.gen_expr(scope)
        if not e:
            return None, scope
        expr, t, words = e
        nm = self.out_name(words, set(scope))
        new = dict(scope)
        new[nm] = self.mk(t)
        self.tags.append("addfields")
        return {self.rng.choice(["$addFields", "$set"]): {nm: expr}}, new

    def stage_count(self, scope):
        nm = self.out_name(self.rng.choice([["total"], ["count"], ["n", "docs"]]), set())
        self.tags.append("count")
        return {"$count": nm}, {nm: self.mk("int32")}

    def stage_sortbycount(self, scope):
        keys = [p for p, f in scope.items() if f.t in ("string", "bool", "int32") and not f.via_array]
        if not keys:
            return None, scope
        k = self.rng.choice(keys)
        self.tags.append("sortbycount")
        return {"$sortByCount": "$" + k}, {"_id": self.with_path("_id", scope[k], False), "count": self.mk("int32")}

    def stage_unset(self, scope):
        leaves = [p for p, f in scope.items() if f.t in SCALAR and not f.via_array and p != "_id" and "." not in p]
        if len(leaves) < 3:
            return None, scope
        ps = self.pick(leaves, self.rng.randint(1, 2))
        new = {p: f for p, f in scope.items() if p not in ps}
        self.tags.append("unset")
        return {"$unset": ps[0] if len(ps) == 1 else ps}, new

    def gen_pipeline(self):
        rng = self.rng
        scope = self.root_scope()
        stages = []
        grouped = False
        projected = False
        terminal = False
        if rng.random() < 0.85:
            st, scope = self.stage_match(scope)
            if st:
                stages.append(st)
        target = rng.choices([2, 3, 4, 5], weights=[25, 35, 28, 12])[0]
        guard = 0
        while len(stages) < target and not terminal and guard < 20:
            guard += 1
            opts = {"group": 0 if grouped else 30, "project": 0 if (projected or grouped) else 16, "sort": 14, "limit": 8, "skip": 2,
                    "unwind": 10, "lookup": 12, "addfields": 8, "count": 0 if grouped else 3, "sortbycount": 0 if grouped else 3,
                    "match": 8, "unset": 3}
            name = rng.choices(list(opts), weights=list(opts.values()))[0]
            if opts[name] == 0:
                continue
            if stages and name in ("limit", "skip", "sort", "match") and list(stages[-1])[0] == "$" + name:
                continue
            fn = getattr(self, "stage_" + name)
            st, new_scope = fn(scope)
            if not st:
                continue
            if name == "group":
                grouped = True
                # depois de $group, projecoes sobre saidas viram raras
            if name in ("project", "unset"):
                projected = True
            if name in ("count", "sortbycount"):
                terminal = True
            stages.append(st)
            scope = new_scope
        if not stages:
            stages = [{"$match": self.gen_filter(self.root_scope(), n=1) or {"_id": {"$exists": True}}}]
        if rng.random() < 0.25 and not terminal and stages[-1].get("$limit") is None:
            stages.append({"$limit": rng.choice([5, 10, 20, 50])})
            self.tags.append("limit")
        return stages

    # ---- statements --------------------------------------------------------------
    def build(self, op: str) -> dict:
        rng = self.rng
        scope = self.root_scope()
        if op in ("find", "findOne"):
            flt = self.gen_filter(scope, allow_empty=True)
            plan = {"op": op, "filter": flt}
            if rng.random() < 0.45:
                pr = self.gen_projection(scope)
                if pr:
                    plan["projection"] = pr
            if op == "find":
                if rng.random() < 0.5:
                    s = self.gen_sort(scope)
                    if s:
                        plan["sort"] = s
                        self.tags.append("sort")
                if rng.random() < 0.15:
                    plan["skip"] = rng.choice([5, 10, 20])
                    self.tags.append("skip")
                if rng.random() < 0.5:
                    plan["limit"] = rng.choice([1, 5, 10, 20, 50])
                    self.tags.append("limit")
            else:
                if rng.random() < 0.3 and flt:
                    pass
            return plan
        if op == "aggregate":
            return {"op": "aggregate", "pipeline": self.gen_pipeline()}
        if op == "countDocuments":
            return {"op": "countDocuments", "filter": self.gen_filter(scope, allow_empty=True)}
        if op == "distinct":
            cand = [p for p, f in scope.items() if f.t in ("string", "int32", "bool", "objectid", "date") and p != "_id" and not f.via_array]
            if not cand:
                cand = [p for p, f in scope.items() if f.t in SCALAR]
            plan = {"op": "distinct", "field": rng.choice(cand)}
            if rng.random() < 0.5:
                flt = self.gen_filter(scope, n=1, allow_complex=False)
                if flt:
                    plan["filter"] = flt
            return plan
        if op in ("update", "delete"):
            flt = self.gen_filter(scope, n=rng.choices([1, 2], weights=[60, 40])[0], allow_complex=False)
            if not flt:
                flt = {"_id": self.sample(scope["_id"])}
            if op == "delete":
                self.tags.append("delete")
                return {"op": rng.choice(["deleteOne", "deleteMany"]), "filter": flt}
            return self.build_update(scope, flt)
        raise ValueError(op)

    def build_update(self, scope, flt):
        rng = self.rng
        settable = [p for p, f in scope.items() if p != "_id" and not f.via_array and f.t in SCALAR]
        nums = [p for p in settable if scope[p].t in NUMERIC]
        dates = [p for p in settable if scope[p].t == "date"]
        arrs = [p for p, f in scope.items() if f.t == "array" and f.item_t in SCALAR and not f.via_array]
        used: list[str] = []

        def free(cands):
            return [p for p in cands if not any(p == u or p.startswith(u + ".") or u.startswith(p + ".") for u in used)]

        upd: dict = {}
        kinds = ["set"] * 4 + (["inc"] * 3 if nums else []) + (["unset"] if settable else []) + (["currentDate"] if dates else []) + \
                (["push", "addToSet", "pull"] if arrs else []) + (["mul"] if nums else []) + (["minmax"] if nums else [])
        for _ in range(rng.choices([1, 2, 3], weights=[55, 35, 10])[0]):
            k = rng.choice(kinds)
            pool = {"set": settable, "inc": nums, "unset": settable, "currentDate": dates, "push": arrs, "addToSet": arrs,
                    "pull": arrs, "mul": nums, "minmax": nums}[k]
            pool = free(pool)
            if not pool:
                continue
            p = rng.choice(pool)
            used.append(p)
            f = scope[p]
            if k == "set":
                upd.setdefault("$set", {})[p] = self.sample(f)
            elif k == "inc":
                upd.setdefault("$inc", {})[p] = rng.choice([1, -1, 5, 10]) if f.t in ("int32", "int64") else rng.choice([1, 0.5, -2])
            elif k == "unset":
                upd.setdefault("$unset", {})[p] = ""
            elif k == "currentDate":
                upd.setdefault("$currentDate", {})[p] = True
            elif k in ("push", "addToSet"):
                upd.setdefault("$" + k, {})[p] = self.sample(f)
            elif k == "pull":
                upd.setdefault("$pull", {})[p] = self.sample(f)
            elif k == "mul":
                upd.setdefault("$mul", {})[p] = rng.choice([2, 1.1, 0.9])
            else:
                upd.setdefault(rng.choice(["$min", "$max"]), {})[p] = self.sample(f)
        if not upd:
            p = rng.choice(settable or ["_id"])
            upd = {"$set": {p: self.sample(scope[p])}}
        self.tags.append("update_" + "_".join(sorted(k[1:] for k in upd)))
        return {"op": rng.choice(["updateOne", "updateMany"]), "filter": flt, "update": upd}


# --------------------------------------------------------------------------- rendering

class Style:
    def __init__(self, rng: random.Random, cfg: dict):
        sc = cfg["snippets"]

        def pick(d):
            ks = list(d)
            return rng.choices(ks, weights=[d[k] for k in ks])[0]
        self.layout = pick(sc["layouts"])
        self.q = '"' if pick(sc["quotes"]) == "double" else "'"
        self.indent = " " * int(pick({str(k): v for k, v in sc["indent"].items()}))
        self.nl = "\r\n" if rng.random() < sc["crlf_ratio"] else "\n"
        self.unquoted = rng.random() < sc["unquoted_keys"]
        self.semicolon = rng.random() < sc["semicolon"]
        self.to_array = rng.random() < sc["to_array"]
        self.root_form = pick(sc["root_forms"])
        self.date_form = pick(sc["date_forms"])
        self.brace_space = rng.random() < 0.85
        self.chain_lines = rng.random() < 0.4
        self.const_extract = rng.random() < sc["const_extract"]


class Writer:
    def __init__(self):
        self.buf: list[str] = []
        self.pos = 0
        self.spans: list[list] = []  # [start, end, role, unit_end]

    def t(self, s: str):
        self.buf.append(s)
        self.pos += len(s)

    def open(self, role: str) -> int:
        self.spans.append([self.pos, -1, role, -1])
        return len(self.spans) - 1

    def close(self, i: int, unit_end: int | None = None):
        self.spans[i][1] = self.pos
        self.spans[i][3] = self.pos if unit_end is None else unit_end

    def text(self):
        return "".join(self.buf)


def js_str(s: str, q: str) -> str:
    s = s.replace("\\", "\\\\").replace(q, "\\" + q).replace("\n", "\\n").replace("\r", "\\r")
    return q + s + q


def fmt_num(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    s = repr(round(float(v), 6))
    if "e" in s or "E" in s:
        s = f"{float(v):.6f}".rstrip("0").rstrip(".")
    return s


def nest_depth(v) -> int:
    if isinstance(v, dict):
        return 1 + max((nest_depth(x) for x in v.values()), default=0)
    if isinstance(v, (list, tuple)):
        return 1 + max((nest_depth(x) for x in v), default=0)
    return 0


def is_op_obj(d: dict) -> bool:
    return bool(d) and all(k.startswith("$") for k in d) and not any(isinstance(v, (dict, list)) and not isinstance(v, Ctor) for v in d.values())


class Renderer:
    def __init__(self, st: Style, w: Writer):
        self.st, self.w = st, w

    # chaves
    def key(self, k: str):
        st, w = self.st, self.w
        role = "op" if k.startswith("$") else "key"
        quoted = not (st.unquoted and IDENT_RE.match(k) and "." not in k) and not (k.startswith("$") and st.unquoted)
        if k.startswith("$") and not quoted:
            quoted = False
        if quoted:
            w.t(st.q)
        i = w.open(role)
        w.t(k)
        w.close(i)
        if quoted:
            w.t(st.q)
        return i

    def scalar(self, v, role_hint="value"):
        st, w = self.st, self.w
        if isinstance(v, Ident):
            i = w.open("ident")
            w.t(v.name)
            w.close(i)
            return
        if isinstance(v, str):
            if v.startswith("$") and not v.startswith("$$") and len(v) > 1 and role_hint != "literal":
                w.t(st.q + "$")
                i = w.open("fieldref")
                w.t(v[1:])
                w.close(i)
                w.t(st.q)
                return
            i = w.open("value")
            w.t(js_str(v, st.q))
            w.close(i)
            return
        i = w.open("value")
        if isinstance(v, Ctor):
            w.t(v.js(st.q, st.date_form))
        elif v is None:
            w.t("null")
        else:
            w.t(fmt_num(v))
        w.close(i)

    def value(self, v, lvl, inline, role_hint="value"):
        if isinstance(v, Ident):
            self.scalar(v)
        elif isinstance(v, dict):
            self.obj(v, lvl, inline)
        elif isinstance(v, (list, tuple)):
            self.arr(v, lvl, inline)
        else:
            self.scalar(v, role_hint)

    def obj(self, d: dict, lvl: int, inline: bool, span_role="obj"):
        st, w = self.st, self.w
        si = w.open(span_role)
        if not d:
            w.t("{}")
            w.close(si)
            return
        expand = (not inline and st.layout == "pretty" and not is_op_obj(d) and (len(d) >= 2 or nest_depth(d) >= 3))
        sp = " " if st.brace_space else ""
        if not expand:
            w.t("{" + sp)
            first = True
            for k, v in d.items():
                if not first:
                    w.t(", ")
                first = False
                ki = self.key(k)
                w.t(": ")
                self.value(v, lvl, True, "literal" if k in ("$regex", "$options", "$type", "format") else "value")
                w.spans[ki][3] = w.pos
            w.t(sp + "}")
        else:
            w.t("{" + st.nl)
            items = list(d.items())
            for n, (k, v) in enumerate(items):
                w.t(st.indent * (lvl + 1))
                ki = self.key(k)
                w.t(": ")
                self.value(v, lvl + 1, False, "literal" if k in ("$regex", "$options", "$type", "format") else "value")
                w.spans[ki][3] = w.pos
                if n < len(items) - 1:
                    w.t(",")
                w.t(st.nl)
            w.t(st.indent * lvl + "}")
        w.close(si)

    def arr(self, a, lvl, inline, stages=False):
        st, w = self.st, self.w
        si = w.open("arr")
        if not a:
            w.t("[]")
            w.close(si)
            return
        has_dict = any(isinstance(x, dict) for x in a)
        expand = False
        if not inline and has_dict:
            if stages and st.layout in ("lines", "pretty"):
                expand = True
            elif st.layout == "pretty" and any(isinstance(x, dict) and len(x) >= 2 for x in a):
                expand = True
        if not expand:
            w.t("[" + (" " if st.brace_space and has_dict else ""))
            for n, x in enumerate(a):
                if n:
                    w.t(", ")
                self.elem(x, lvl, True, stages)
            w.t((" " if st.brace_space and has_dict else "") + "]")
        else:
            w.t("[" + st.nl)
            for n, x in enumerate(a):
                w.t(st.indent * (lvl + 1))
                self.elem(x, lvl + 1, st.layout == "lines", stages)
                if n < len(a) - 1:
                    w.t(",")
                w.t(st.nl)
            w.t(st.indent * lvl + "]")
        w.close(si)

    def elem(self, x, lvl, inline, stages):
        w = self.w
        if stages and isinstance(x, dict):
            i = w.open("stage")
            self.obj(x, lvl, inline)
            w.close(i)
        else:
            self.value(x, lvl, inline)


def render_snippet(plan: dict, sch: Schema, coll: Coll, st: Style, rng: random.Random) -> tuple[str, list, dict]:
    """Renderiza o plano em texto Console + spans. Retorna (texto, spans, info)."""
    w = Writer()
    R = Renderer(st, w)
    nl = st.nl
    q = st.q
    op = plan["op"]
    name = coll.name
    preamble_vars: list[str] = []
    stmts_start: list[int] = []

    root_form = st.root_form
    if root_form == "auto":
        root_form = "dot" if (IDENT_RE.match(name) and name not in RESERVED_COLL and "$" not in name) else rng.choice(["getCollection", "bracket"])
    elif not IDENT_RE.match(name) and root_form == "dot":
        root_form = "getCollection"

    var_name = None
    if root_form == "var":
        var_name = style_name(split_words(re.sub(r"[^A-Za-z0-9]+", " ", name)), "camel") or "col"
        if not IDENT_RE.match(var_name) or var_name in ("db",):
            var_name = "col"
        # declaracao: const orders = db.getCollection("x");
        i = w.open("stmt")
        w.t("const " + var_name + " = db")
        if rng.random() < 0.5:
            w.t(".")
            m = w.open("method"); w.t("getCollection"); w.close(m)
            w.t("(" + q)
            s = w.open("member"); w.t(name); w.close(s)
            w.t(q + ")")
        else:
            w.t("[" + q)
            s = w.open("member"); w.t(name); w.close(s)
            w.t(q + "]")
        w.t(";" + nl)
        w.close(i)

    # const extraction (filtro/pipeline)
    extract_name = None
    key_arg = {"find": "filter", "findOne": "filter", "countDocuments": "filter", "aggregate": "pipeline"}.get(op)
    if st.const_extract and key_arg and (plan.get("filter") or plan.get("pipeline")):
        extract_name = {"filter": rng.choice(["filter", "query"]), "pipeline": rng.choice(["pipeline", "stages"])}[key_arg]
        i = w.open("stmt")
        w.t("const " + extract_name + " = ")
        if key_arg == "pipeline":
            R.arr(plan["pipeline"], 0, False, stages=True)
        else:
            R.obj(plan["filter"], 0, False)
        w.t(";" + nl)
        w.close(i)

    # ---- statement principal
    mi = w.open("stmt")
    if root_form == "var":
        w.t(var_name)
    elif root_form == "pool":
        w.t("ConnectionPool")
        for part in ("local", sch.schema_id, name):
            if IDENT_RE.match(part):
                w.t(".")
                s = w.open("member"); w.t(part); w.close(s)
            else:
                w.t("[" + q)
                s = w.open("member"); w.t(part); w.close(s)
                w.t(q + "]")
    else:
        w.t("db")
        if root_form == "dot":
            w.t(".")
            s = w.open("member"); w.t(name); w.close(s)
        elif root_form == "getCollection":
            w.t(".")
            m = w.open("method"); w.t("getCollection"); w.close(m)
            w.t("(" + q)
            s = w.open("member"); w.t(name); w.close(s)
            w.t(q + ")")
        else:
            w.t("[" + q)
            s = w.open("member"); w.t(name); w.close(s)
            w.t(q + "]")

    def call(method, args_fn, first=False, newline=False):
        ci = w.open("chain")
        if newline and st.chain_lines and st.layout != "compact":
            w.t(nl + st.indent)
            w.spans[ci][0] = w.pos
        w.t(".")
        m = w.open("method"); w.t(method); w.close(m)
        w.t("(")
        args_fn()
        w.t(")")
        w.close(ci)
        w.spans[m][3] = w.pos

    def arg_obj(v, ref_name=None, is_stages=False):
        if ref_name:
            R.scalar(Ident(ref_name, v))
        elif is_stages:
            R.arr(v, 0, False, stages=True)
        else:
            R.obj(v, 0, False)

    if op in ("find", "findOne"):
        def args():
            if extract_name == "filter" or extract_name == "query":
                arg_obj(plan["filter"], extract_name)
            else:
                arg_obj(plan.get("filter", {}))
            if plan.get("projection"):
                w.t(", ")
                R.obj(plan["projection"], 0, False)
        call(op, args)
        if op == "find":
            if plan.get("sort"):
                call("sort", lambda: R.obj(plan["sort"], 0, False), newline=True)
            if plan.get("skip") is not None:
                call("skip", lambda: R.scalar(plan["skip"]), newline=True)
            if plan.get("limit") is not None:
                call("limit", lambda: R.scalar(plan["limit"]), newline=True)
            if st.to_array:
                call("toArray", lambda: None, newline=True)
    elif op == "aggregate":
        def args():
            if extract_name in ("pipeline", "stages"):
                arg_obj(plan["pipeline"], extract_name)
            else:
                R.arr(plan["pipeline"], 0, False, stages=True)
        call("aggregate", args)
        if st.to_array:
            call("toArray", lambda: None, newline=True)
    elif op == "countDocuments":
        def args():
            if extract_name:
                arg_obj(plan["filter"], extract_name)
            else:
                R.obj(plan.get("filter", {}), 0, False)
        call("countDocuments", args)
    elif op == "distinct":
        def args():
            w.t(q)
            s = w.open("strarg"); w.t(plan["field"]); w.close(s)
            w.t(q)
            if plan.get("filter"):
                w.t(", ")
                R.obj(plan["filter"], 0, False)
        call("distinct", args)
    elif op in ("updateOne", "updateMany"):
        def args():
            R.obj(plan["filter"], 0, False)
            w.t(", ")
            R.obj(plan["update"], 0, False)
        call(op, args)
    elif op in ("deleteOne", "deleteMany"):
        call(op, lambda: R.obj(plan["filter"], 0, False))
    else:
        raise ValueError(op)
    if st.semicolon:
        w.t(";")
    w.close(mi)
    text = w.text()
    return text, w.spans, {"root_form": root_form, "extracted": extract_name}


# --------------------------------------------------------------------------- field checker (independente do gerador)

def _subtree(scope: dict, src: str):
    return {p: o for p, o in scope.items() if p == src or p.startswith(src + ".")}


def analyze_plan(plan: dict, sch: Schema, coll: Coll) -> dict:
    """Caminha o plano EJSON com escopo de campos por estagio.
    Retorna {used, created, missing}. `used` so contem caminhos que resolvem para campos do schema
    (colecao raiz: 'a.b'; colecao estrangeira: 'coll::a.b')."""
    used: set[str] = set()
    missing: list[str] = []
    created: list[str] = []

    def root_scope(c: Coll, prefix=""):
        return {p: (prefix + p) for p in c.fields}

    def ref(path: str, scope: dict):
        if path in scope:
            o = scope[path]
            if o:
                used.add(o)
            return
        missing.append(path)

    def expr(e, scope):
        if isinstance(e, str):
            if e.startswith("$$"):
                return
            if e.startswith("$") and len(e) > 1:
                ref(e[1:], scope)
            return
        if isinstance(e, list):
            for x in e:
                expr(x, scope)
        elif isinstance(e, dict):
            if set(e) & EJSON_KEYS:
                return
            for k, v in e.items():
                if k.startswith("$") and k not in ("$literal",):
                    expr(v, scope)
                elif not k.startswith("$"):
                    expr(v, scope)

    def filt(f, scope, prefix=""):
        if not isinstance(f, dict):
            return
        for k, v in f.items():
            if k in ("$and", "$or", "$nor"):
                for sub in v:
                    filt(sub, scope, prefix)
            elif k == "$expr":
                expr(v, scope)
            elif k.startswith("$"):
                missing.append("op:" + k)
            else:
                full = prefix + k if prefix else k
                ref(full, scope)
                if isinstance(v, dict) and not (set(v) & EJSON_KEYS):
                    for op, arg in v.items():
                        if op == "$elemMatch" and isinstance(arg, dict):
                            if any(not kk.startswith("$") for kk in arg):
                                filt(arg, scope, full + ".")
                        elif op == "$not" and isinstance(arg, dict):
                            pass
                        elif not op.startswith("$"):
                            missing.append("op:" + op)

    def pipeline(stages, scope):
        for s in stages:
            if not isinstance(s, dict) or len(s) != 1:
                missing.append("stage:?")
                continue
            (name, body), = s.items()
            if name == "$match":
                filt(body, scope)
            elif name == "$group":
                new = {}
                for k, v in body.items():
                    if k == "_id":
                        expr(v, scope)
                        new["_id"] = None
                        if isinstance(v, dict) and not (set(v) & EJSON_KEYS) and all(not kk.startswith("$") for kk in v):
                            for kk in v:
                                new["_id." + kk] = None
                        elif isinstance(v, str) and v.startswith("$") and v[1:] in scope:
                            for p, o in _subtree(scope, v[1:]).items():
                                new["_id" + p[len(v) - 1:]] = o
                    else:
                        expr(v, scope)
                        new[k] = None
                        created.append(k)
                scope = new
            elif name == "$project":
                incl = {}
                excl = set()
                computed = {}
                for k, v in body.items():
                    if v in (0, False) and not isinstance(v, (dict, str)):
                        if k != "_id":
                            ref(k, scope)
                        excl.add(k)
                    elif v in (1, True) and not isinstance(v, (dict, str)):
                        ref(k, scope)
                        incl.update(_subtree(scope, k))
                    else:
                        expr(v, scope)
                        computed[k] = None
                        if isinstance(v, str) and v.startswith("$") and v[1:] in scope:
                            for p, o in _subtree(scope, v[1:]).items():
                                computed[k + p[len(v) - 1:]] = o
                        created.append(k)
                if incl or computed:
                    new = dict(incl)
                    new.update(computed)
                    if "_id" in scope and "_id" not in excl:
                        new["_id"] = scope["_id"]
                    scope = new
                else:
                    scope = {p: o for p, o in scope.items() if not any(p == e or p.startswith(e + ".") for e in excl)}
            elif name in ("$addFields", "$set"):
                new = dict(scope)
                for k, v in body.items():
                    expr(v, scope)
                    new[k] = None
                    created.append(k)
                scope = new
            elif name == "$unset":
                ps = [body] if isinstance(body, str) else list(body)
                for p in ps:
                    ref(p, scope)
                scope = {p: o for p, o in scope.items() if not any(p == e or p.startswith(e + ".") for e in ps)}
            elif name == "$sort":
                for k in body:
                    ref(k, scope)
            elif name in ("$limit", "$skip"):
                pass
            elif name == "$unwind":
                p = body if isinstance(body, str) else body.get("path", "")
                if isinstance(p, str) and p.startswith("$"):
                    ref(p[1:], scope)
                else:
                    missing.append("unwind:" + str(p))
            elif name == "$lookup":
                fc = sch.by_key.get(body.get("from"))
                if fc is None:
                    missing.append("from:" + str(body.get("from")))
                    continue
                fscope = root_scope(fc, fc.key + "::")
                if "localField" in body:
                    ref(body["localField"], scope)
                    if body.get("foreignField") not in fscope:
                        missing.append(fc.key + "::" + str(body.get("foreignField")))
                if "let" in body:
                    for v in body["let"].values():
                        expr(v, scope)
                if "pipeline" in body:
                    pipeline(body["pipeline"], dict(fscope))
                new = dict(scope)
                a = body.get("as")
                new[a] = None
                for p, o in fscope.items():
                    new[f"{a}.{p}"] = o
                created.append(a)
                scope = new
            elif name == "$count":
                scope = {body: None}
                created.append(body)
            elif name == "$sortByCount":
                expr(body, scope)
                scope = {"_id": None, "count": None}
            else:
                missing.append("stage:" + name)

    op = plan.get("op")
    base = root_scope(coll)
    if op in ("find", "findOne", "countDocuments", "deleteOne", "deleteMany", "updateOne", "updateMany", "distinct"):
        if plan.get("filter"):
            filt(plan["filter"], base)
        if op == "distinct":
            ref(plan["field"], base)
        for k in (plan.get("projection") or {}):
            ref(k, base)
        for k in (plan.get("sort") or {}):
            ref(k, base)
        if op.startswith("update"):
            for opn, fields in plan["update"].items():
                for k in fields:
                    ref(k, base)
    elif op == "aggregate":
        pipeline(plan["pipeline"], base)
    else:
        missing.append("op:" + str(op))
    return {"used": sorted(used), "created": sorted(set(created)), "missing": missing}


# --------------------------------------------------------------------------- driver

def template_id_for(op: str, tags: list[str], plan: dict) -> str:
    uniq = []
    for t in tags:
        if t not in uniq:
            uniq.append(t)
    if op == "aggregate":
        stages = [list(s)[0].lstrip("$") for s in plan["pipeline"]]
        gtags = [t for t in uniq if t.startswith("group_")]
        body = []
        for s in stages:
            if s == "group" and gtags:
                body.append(gtags.pop(0))
            else:
                body.append(s.lower())
        return "agg." + ".".join(body)
    if op in ("find", "findOne", "countDocuments", "distinct"):
        cond = sorted({t for t in uniq if t in {"eq", "in", "regex", "ne", "nin", "range", "cmp", "exists", "null", "all", "size",
                                                "elem", "not", "or", "and", "expr", "mod", "type", "empty"} or t.startswith("elem_")})
        mods = [m for m in ("proj", "proj_excl", "sort", "skip", "limit") if m in uniq]
        head = {"find": "find", "findOne": "findone", "countDocuments": "count", "distinct": "distinct"}[op]
        return head + "." + "+".join(cond[:4] or ["all"]) + ("." + ".".join(mods) if mods else "")
    if op in ("updateOne", "updateMany"):
        t = next((t for t in uniq if t.startswith("update_")), "update")
        return op.lower().replace("update", "update.") + t.replace("update_", "")
    return "delete." + op.lower().replace("delete", "")


READ_OPS = {"find", "findOne", "aggregate", "countDocuments", "distinct"}


def gen_one(index: int, schemas: list[Schema], cfg: dict, seen: set) -> dict | None:
    sc = cfg["snippets"]
    seed = cfg.get("seed", 20261008)
    rej = {"field_check": 0, "dup": 0, "long": 0}
    for attempt in range(12):
        rng = random.Random(f"{seed}:snip:{index}:{attempt}")
        sch = schemas[rng.randrange(len(schemas))]
        coll = sch.colls[rng.randrange(len(sch.colls))]
        kinds = list(sc["kinds"])
        kind = rng.choices(kinds, weights=[sc["kinds"][k] for k in kinds])[0]
        b = Builder(rng, sch, coll, cfg)
        try:
            plan = b.build(kind)
        except Exception:
            continue
        # plano EJSON para analise e execucao
        ej = to_ej(plan)
        an = analyze_plan(ej, sch, coll)
        if an["missing"]:
            rej["field_check"] += 1
            continue
        st = Style(rng, cfg)
        text, spans, info = render_snippet(plan, sch, coll, st, rng)
        if len(text) > sc.get("max_chars", 1800):
            rej["long"] += 1
            continue
        norm = (sch.full_id, coll.key, re.sub(r"\s+", "", text.replace("'", '"')))
        if norm in seen:
            rej["dup"] += 1
            continue
        seen.add(norm)
        tid = template_id_for(plan["op"], b.tags, plan)
        group_id = f"{sch.full_id}::{tid}"
        holdout = sch.holdout or sch.domain in set(cfg.get("holdout_domains", []))
        return {
            "id": f"snip-{sch.domain}.{sch.schema_id}.{coll.key.replace(' ', '_')}-{index:06d}",
            "kind": "read" if plan["op"] in READ_OPS else "write",
            "op": plan["op"],
            "dialect": "console",
            "domain": sch.domain,
            "schema_id": sch.full_id,
            "db": sch.db_name,
            "collection": coll.name,
            "template_id": tid,
            "group_id": group_id,
            "split_key": split_key(group_id),
            "split": assign_split(group_id, sch.domain, tid, cfg, holdout),
            "holdout": holdout,
            "text": text,
            "spans": spans,
            "plan": ej,
            "fields_used": an["used"],
            "fields_created": an["created"],
            "style": {"layout": st.layout, "quote": st.q, "indent": len(st.indent), "crlf": st.nl == "\r\n",
                      "root_form": info["root_form"], "extracted": info["extracted"], "date_form": st.date_form},
        }
    return None


def blind_clean(record: dict, text: str) -> bool:
    """False se o registro/texto coincide (hash) com item do conjunto cego. So usa a API do guard."""
    import blind_guard
    try:
        blind_guard.check_record(record)
        blind_guard.check_text(text)
    except blind_guard.BlindAccessError:
        return False
    return True


def generate(cfg: dict, schemas: list[Schema], count: int | None = None) -> list[dict]:
    count = count or cfg["snippets"]["count"]
    seen: set = set()
    out = []
    for i in range(count):
        r = gen_one(i, schemas, cfg, seen)
        if r and blind_clean(r, r["text"]):
            out.append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--count", type=int)
    ap.add_argument("--ver", help="subpasta em data/generated/")
    ap.add_argument("--schemas-dir")
    ap.add_argument("--out")
    ap.add_argument("--seeded-only", action="store_true",
                    help="so schemas presentes em data/fixtures/manifest.json (bancos kapi_fx_* semeados)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg["seed"] = a.seed
    ver = a.ver or cfg.get("version", "v0")
    schemas = load_schemas(Path(a.schemas_dir) if a.schemas_dir else ROOT / cfg.get("schemas_dir", "data/schemas"))
    if a.seeded_only:
        mf = json.loads((ROOT / "data" / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
        seeded = {(e.get("domain"), e.get("schema_id")) for e in mf.get("entries", [])}
        schemas = [s for s in schemas if (s.domain, s.schema_id) in seeded]
    if not schemas:
        raise SystemExit("nenhum schema em data/schemas")
    recs = generate(cfg, schemas, a.count)
    out = Path(a.out) if a.out else ROOT / cfg.get("out_dir", "data/generated") / ver / "snippets.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            line = json.dumps(r, ensure_ascii=False, separators=(",", ":"))
            f.write(line + "\n")
            h.update(line.encode("utf-8"))
    by_op: dict = {}
    for r in recs:
        by_op[r["op"]] = by_op.get(r["op"], 0) + 1
    manifest = {"seed": cfg["seed"], "count": len(recs), "requested": a.count or cfg["snippets"]["count"], "sha256": h.hexdigest(),
                "by_op": by_op, "schemas": len(schemas), "splits": {s: sum(1 for r in recs if r["split"] == s) for s in ("train", "val", "test", "bench")}}
    (out.parent / "snippets.manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
