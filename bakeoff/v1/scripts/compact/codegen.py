"""Gerador de snippets MongoDB completos e sintéticos: código + contexto de IDE + intenções (pt/en).

Cada snippet é código completo e válido (sintaxe e estrutura MongoDB), usado por prepare_fim.py para recortes.
Dialetos: ``console`` (API real do Console da IDE), ``mongosh`` e ``json`` (modo Agregação: array de estágios).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any

from . import schemas as S
from .jsrender import IDENT, DateV, Money, Oid, Raw, Regex, Style, UuidV, call, number, quote, render, render_inline
from .contracts import LANGUAGE_AGGREGATION, LANGUAGE_CONSOLE

CATEGORIES = ("crud", "aggregation", "filters", "atlas_search", "indexes", "mongosh_js", "dsl_ide", "complex_queries")
STRINGY = ("enum", "text", "word", "code", "name", "city", "state", "country", "email")


def pick(rng: random.Random, weights: dict) -> Any:
    items = [(k, w) for k, w in weights.items() if w > 0]
    return rng.choices([k for k, _ in items], weights=[w for _, w in items])[0]


@dataclass
class Stmt:
    code: str
    pt: str = ""
    en: str = ""
    main: bool = True


def display(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return number(value)
    if isinstance(value, str):
        return value
    if isinstance(value, DateV):
        return value.iso[:10]
    if isinstance(value, Oid):
        return "ObjectId"
    if isinstance(value, UuidV):
        return "UUID"
    if isinstance(value, Regex):
        return "/" + value.pattern + "/"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(display(v) for v in value[:3]) + "]"
    if isinstance(value, Raw):
        return value.text
    return "{...}"


def singular(name: str) -> str:
    if name.endswith("ies"):
        return name[:-3] + "y"
    if name.endswith("sses"):
        return name[:-2]
    return name[:-1] if name.endswith("s") else name


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


class SnippetGenerator:
    def __init__(self, rng: random.Random, domain: str, dialect: str, category: str, context_cfg: dict | None = None) -> None:
        self.rng = r = rng
        self.domain = domain
        self.dialect = dialect
        self.category = category
        self.context_cfg = context_cfg or {}
        self.colls = S.load_domain(domain)
        self.json = dialect == "json"
        self.name_style = pick(r, {"camel": .5, "pascal": .35, "snake": .15})
        self.coll_style = self.name_style if r.random() < .9 else "kebab"
        self.style = Style(
            indent=r.choice([2, 2, 4]), width=r.choice([60, 80, 80, 100, 120]),
            key_quotes="all" if (self.json or r.random() < .12) else "minimal",
            string_quote='"' if (self.json or r.random() < .8) else "'",
            brace_space=r.random() < .8, json=self.json, semicolons=(not self.json) and r.random() < .35,
            chain_break=r.random() < .3)
        self.date_form = "newdate" if dialect == "mongosh" and r.random() < .5 else "isodate"
        self.coll_key = r.choice(list(self.colls))
        self.coll = self.colls[self.coll_key]
        self.database = r.choice(S.DOMAINS[domain]["databases"])
        self.connection = r.choice(S.CONNECTION_NAMES)
        self.prelude: list[Stmt] = []
        self.bound: dict[str, str] = {}
        self.subcategory = category

    # -- nomes e valores ----------------------------------------------------------------------------------------------
    def cname(self, key: str) -> str:
        return S.style_collection(key, self.coll_style)

    def fpath(self, f: S.Field) -> str:
        return S.style_path(f.path, self.name_style)

    def ppath(self, path: str) -> str:
        return S.style_path(path, self.name_style)

    def out(self, name: str) -> str:
        return S.style_segment(name, self.name_style)

    def q(self, text: str) -> str:
        return quote(text, self.style)

    def semi(self) -> str:
        return ";" if self.style.semicolons else ""

    def value(self, f: S.Field) -> Any:
        v = S.sample_value(f, self.rng, self.date_form)
        if isinstance(v, UuidV) and self.dialect == "console" and self.rng.random() < .3:
            v.ctor = self.rng.choice(["CGUUID", "JUUID", "GUUID"])
        return v

    def distinct_values(self, f: S.Field, n: int) -> list:
        if f.enum:
            return self.rng.sample(f.enum, min(n, len(f.enum)))
        seen: dict[str, Any] = {}
        for _ in range(n * 3):
            v = self.value(f)
            seen.setdefault(render_inline(v, Style(json=False)), v)
            if len(seen) == n:
                break
        return list(seen.values())

    def plain_fields(self, *types: str) -> list[S.Field]:
        return [f for f in self.coll.fields if not f.array_parent and (not types or f.type in types)]

    def text_term(self, f: S.Field) -> str:
        words = [w for w in re.split(r"[^\wÀ-ÿ]+", str(self.value(f))) if len(w) >= 3]
        return self.rng.choice(words) if words else self.rng.choice(S.SEARCH_TERMS)

    # -- acesso à coleção e cadeias --------------------------------------------------------------------------------------
    def target_segments(self, key: str | None = None, allow_bind: bool = True, force: str | None = None) -> list[str]:
        key = key or self.coll_key
        if key in self.bound:
            return [self.bound[key]]
        r = self.rng
        name = self.cname(key)
        ident = bool(IDENT.match(name))
        if force:
            kind = force
        elif self.dialect == "console":
            kind = pick(r, {"dot": .35 if ident else 0, "get": .27, "bracket": .03, "conn": .17, "pool": .12, "bind": .06 if allow_bind else 0})
        else:
            kind = pick(r, {"dot": .45 if ident else 0, "get": .3, "sibling": .1, "sibling_dot": .05 if ident else 0, "bind": .1 if allow_bind else 0})
        q = self.q
        if kind == "dot":
            return ["db", "." + name]
        if kind == "get":
            return ["db", f".getCollection({q(name)})"]
        if kind == "bracket":
            return ["db", f"[{q(name)}]"]
        if kind == "sibling":
            return ["db", f".getSiblingDB({q(self.database)})", f".getCollection({q(name)})"]
        if kind == "sibling_dot":
            return ["db", f".getSiblingDB({q(self.database)})", "." + name]
        if kind == "conn":
            return [f"getConnection({q(self.connection)})", f".getDatabase({q(self.database)})", f".getCollection({q(name)})"]
        if kind == "pool":
            segs = [self.connection, self.database, name]
            if all(IDENT.match(s) for s in segs) and r.random() < .75:
                return ["ConnectionPool." + ".".join(segs)]
            return ["ConnectionPool" + "".join(f"[{q(s)}]" for s in segs)]
        var = self.var_name(key)
        inner = "".join(self.target_segments(key, allow_bind=False))
        self.prelude.append(Stmt(f"const {var} = {inner}{self.semi()}", f"Referência à coleção {name}",
                                 f"Reference to the {name} collection", main=False))
        self.bound[key] = var
        return [var]

    def var_name(self, key: str) -> str:
        base = re.sub(r"[^A-Za-z0-9]", "", S.style_collection(key, "camel"))
        name = self.rng.choice([base, base + "Collection", "coll", "col", singular(base) + "Coll"])
        return name if name not in ("db", "print", "console") else name + "Coll"

    def chain(self, segments: list[str], calls: list[tuple[str, list]]) -> str:
        """Renderiza alvo + chamadas; cadeias longas podem quebrar linhas iniciando com '.'."""
        st = self.style
        parts = len(segments) + len(calls)
        breaking = parts >= 3 and (st.chain_break or (segments[0].startswith("getConnection") and self.rng.random() < .5))
        if breaking:
            pad = " " * st.indent
            head = segments[0]
            rest: list[str] = []
            for segment in segments[1:]:
                if "(" in segment:
                    rest.append(segment)
                elif rest:
                    rest[-1] += segment
                else:
                    head += segment  # db.orders fica junto; quebra só antes de chamadas
            text = head + "".join("\n" + pad + s for s in rest)
            for method, args in calls:
                text += "\n" + pad + call("." + method, args, st, level=1, used=len(pad))
            return text
        text = "".join(segments)
        for method, args in calls:
            used = len(text) - (text.rfind("\n") + 1)
            text += call("." + method, args, st, level=0, used=used)
        return text

    def statement(self, segments: list[str], calls: list[tuple[str, list]], pt: str, en: str, main: bool = True) -> Stmt:
        return Stmt(self.chain(segments, calls) + self.semi(), pt, en, main)

    # -- filtros ------------------------------------------------------------------------------------------------------------
    def condition(self, f: S.Field) -> tuple[Any, str, str]:
        r = self.rng
        p = self.fpath(f)
        t = f.type
        if t in STRINGY:
            k = pick(r, {"eq": .45, "in": .2, "ne": .1, "nin": .06, "regex": .12 if t in ("text", "word", "code", "name", "email", "city") else 0, "exists": .07})
            v = self.value(f)
            if k == "eq":
                return v, f"{p} igual a {display(v)}", f"{p} equals {display(v)}"
            if k in ("in", "nin"):
                vals = self.distinct_values(f, r.randint(2, 3))
                return ({"$" + k: vals}, f"{p} {'em' if k == 'in' else 'fora de'} {display(vals)}",
                        f"{p} {'in' if k == 'in' else 'not in'} {display(vals)}")
            if k == "ne":
                return {"$ne": v}, f"{p} diferente de {display(v)}", f"{p} not equal to {display(v)}"
            if k == "regex":
                prefix = re.sub(r"[^A-Za-z0-9]", "", str(v))[: r.randint(2, 4)] or "a"
                pattern = "^" + prefix
                flags = "i" if r.random() < .7 else ""
                if self.json or r.random() < .4:
                    value = {"$regex": pattern, "$options": flags} if flags else {"$regex": pattern}
                else:
                    value = Regex(pattern, flags)
                return value, f"{p} começando com {prefix}", f"{p} starting with {prefix}"
            exists = r.random() < .75
            return ({"$exists": exists}, f"{'com' if exists else 'sem'} {p}", f"{'with' if exists else 'without'} {p}")
        if f.numeric:
            a, b = sorted([self.value(f), self.value(f)])
            k = pick(r, {"gte": .3, "gt": .15, "lt": .15, "lte": .1, "range": .2, "eq": .1})
            if k == "range" and a != b:
                lo, hi = r.choice(["$gte", "$gt"]), r.choice(["$lte", "$lt"])
                return {lo: a, hi: b}, f"{p} entre {display(a)} e {display(b)}", f"{p} between {display(a)} and {display(b)}"
            if k in ("eq", "range"):
                return a, f"{p} igual a {display(a)}", f"{p} equals {display(a)}"
            sym = {"gte": ">=", "gt": ">", "lt": "<", "lte": "<="}[k]
            return {"$" + k: a}, f"{p} {sym} {display(a)}", f"{p} {sym} {display(a)}"
        if t == "date":
            d1, d2 = sorted([self.value(f), self.value(f)], key=lambda d: d.iso)
            k = pick(r, {"gte": .45, "range": .35, "lt": .2})
            if k == "gte":
                return {"$gte": d1}, f"{p} a partir de {display(d1)}", f"{p} since {display(d1)}"
            if k == "lt":
                return {"$lt": d2}, f"{p} antes de {display(d2)}", f"{p} before {display(d2)}"
            return ({"$gte": d1, "$lt": d2}, f"{p} entre {display(d1)} e {display(d2)}",
                    f"{p} between {display(d1)} and {display(d2)}")
        if t == "bool":
            v = r.random() < .6
            if r.random() < .8:
                return v, f"{p} = {display(v)}", f"{p} is {display(v)}"
            return {"$ne": v}, f"{p} diferente de {display(v)}", f"{p} not {display(v)}"
        if t in ("ref", "oid"):
            if r.random() < .75:
                return self.value(f), f"{p} com id específico", f"{p} with a specific id"
            return {"$in": [self.value(f), self.value(f)]}, f"{p} em uma lista de ids", f"{p} in a list of ids"
        if t == "uuid":
            return self.value(f), f"{p} com UUID específico", f"{p} with a specific UUID"
        if t == "tags":
            k = pick(r, {"eq": .4, "all": .2, "in": .3, "size": .1})
            if k == "eq":
                v = self.value(f)
                return v, f"{p} contendo {v}", f"{p} containing {v}"
            if k == "size":
                n = r.randint(1, 3)
                return {"$size": n}, f"{p} com {n} itens", f"{p} with {n} items"
            vals = self.distinct_values(f, 2)
            return ({"$" + k: vals}, f"{p} com {'todas' if k == 'all' else 'alguma'} de {display(vals)}",
                    f"{p} with {'all' if k == 'all' else 'any'} of {display(vals)}")
        if t == "geo":
            lng, lat = round(r.uniform(-60, -34), 3), round(r.uniform(-30, -3), 3)
            km = r.choice([5, 10, 25, 50])
            return ({"$geoWithin": {"$centerSphere": [[lng, lat], round(km / 6378.1, 5)]}},
                    f"{p} num raio de {km} km", f"{p} within {km} km")
        v = self.value(f)
        return v, f"{p} igual a {display(v)}", f"{p} equals {display(v)}"

    def filter(self, n: int | None = None, logical: float = .2) -> tuple[dict, str, str]:
        r = self.rng
        fields = [f for f in self.coll.fields if f.type != "geo" or r.random() < .1]
        n = n or pick(r, {1: .35, 2: .4, 3: .2, 4: .05})
        chosen = r.sample(fields, min(n, len(fields)))
        conds: dict[str, Any] = {}
        pts, ens = [], []
        for f in chosen:
            value, pt, en = self.condition(f)
            conds[self.fpath(f)] = value
            pts.append(pt)
            ens.append(en)
        roll = r.random()
        if roll < logical and len(conds) >= 2:
            items = list(conds.items())
            op = pick(r, {"$or": .7, "$and": .15, "$nor": .15})
            head = dict(items[2:])
            branch = [{k: v} for k, v in items[:2]]
            joiner = {"$or": (" ou ", " or "), "$and": (" e ", " and "), "$nor": (" nem ", " nor ")}[op]
            conds = {**head, op: branch} if r.random() < .5 else {op: branch, **head}
            pts = [f"({pts[0]}{joiner[0]}{pts[1]})"] + pts[2:]
            ens = [f"({ens[0]}{joiner[1]}{ens[1]})"] + ens[2:]
        arrays = sorted({f.array_parent for f in self.coll.fields if f.array_parent})
        if arrays and r.random() < logical * .6:
            parent = r.choice(arrays)
            subs = [f for f in self.coll.fields if f.array_parent == parent and f.type != "date"]
            inner = {}
            for f in r.sample(subs, min(len(subs), r.randint(1, 2))):
                value, _, _ = self.condition(f)
                inner[S.style_segment(f.path.split(".")[-1], self.name_style)] = value
            conds[self.ppath(parent)] = {"$elemMatch": inner}
            pts.append(f"algum item de {self.ppath(parent)} atendendo condições")
            ens.append(f"an element of {self.ppath(parent)} matching conditions")
        numeric = self.plain_fields("int", "money", "float")
        if len(numeric) >= 2 and r.random() < logical * .25:
            a, b = r.sample(numeric, 2)
            conds["$expr"] = {r.choice(["$gt", "$lt", "$gte"]): ["$" + self.fpath(a), "$" + self.fpath(b)]}
            pts.append(f"comparando {self.fpath(a)} com {self.fpath(b)}")
            ens.append(f"comparing {self.fpath(a)} to {self.fpath(b)}")
        return conds, " e ".join(pts), " and ".join(ens)

    def projection(self) -> dict:
        r = self.rng
        fields = r.sample(self.plain_fields(), min(len(self.plain_fields()), r.randint(1, 4)))
        proj: dict[str, Any] = {}
        if r.random() < .3:
            proj["_id"] = 0
        for f in fields:
            proj[self.fpath(f)] = 1 if r.random() < .85 else True
        return proj

    def sort_spec(self) -> dict:
        r = self.rng
        candidates = self.plain_fields("date", "int", "money", "float", "name", "enum", "code")
        fields = r.sample(candidates, min(len(candidates), pick(r, {1: .75, 2: .25}))) or [self.coll.fields[0]]
        return {self.fpath(f): (-1 if f.type in ("date", "money", "int") and r.random() < .7 else r.choice([1, -1])) for f in fields}

    # -- updates e documentos ------------------------------------------------------------------------------------------
    def update(self) -> tuple[dict, str, str, dict | None]:
        r = self.rng
        update: dict[str, Any] = {}
        pts, ens = [], []
        options = None
        plain = [f for f in self.plain_fields() if f.type not in ("geo", "tags")]
        ops = r.sample(["$set", "$inc", "$unset", "$push", "$addToSet", "$pull", "$currentDate", "$max", "$arrayFilters"],
                       k=pick(r, {1: .6, 2: .32, 3: .08}))
        if "$set" not in ops and r.random() < .5:
            ops.insert(0, "$set")
        for op in ops:
            if op == "$set" and plain:
                body = {self.fpath(f): self.value(f) for f in r.sample(plain, min(len(plain), r.randint(1, 3)))}
                update["$set"] = body
                pts.append("define " + ", ".join(body))
                ens.append("set " + ", ".join(body))
            elif op == "$inc" and self.plain_fields("int"):
                f = r.choice(self.plain_fields("int"))
                step = r.choice([1, 1, 2, 5, 10, -1])
                update["$inc"] = {self.fpath(f): step}
                pts.append(f"incrementa {self.fpath(f)} em {step}")
                ens.append(f"increment {self.fpath(f)} by {step}")
            elif op == "$unset" and plain:
                f = r.choice(plain)
                update["$unset"] = {self.fpath(f): ""}
                pts.append(f"remove {self.fpath(f)}")
                ens.append(f"unset {self.fpath(f)}")
            elif op in ("$push", "$addToSet", "$pull") and self.plain_fields("tags"):
                f = r.choice(self.plain_fields("tags"))
                v = self.value(f)
                if op == "$push" and r.random() < .3:
                    update[op] = {self.fpath(f): {"$each": self.distinct_values(f, 2), "$slice": -r.choice([10, 20, 50])}}
                else:
                    update[op] = {self.fpath(f): v}
                verb = {"$push": ("adiciona", "push"), "$addToSet": ("adiciona sem duplicar", "add to set"), "$pull": ("remove", "pull")}[op]
                pts.append(f"{verb[0]} {v} em {self.fpath(f)}")
                ens.append(f"{verb[1]} {v} {'into' if op != '$pull' else 'from'} {self.fpath(f)}")
            elif op == "$currentDate" and self.plain_fields("date"):
                f = r.choice(self.plain_fields("date"))
                update["$currentDate"] = {self.fpath(f): True}
                pts.append(f"atualiza {self.fpath(f)} para agora")
                ens.append(f"set {self.fpath(f)} to now")
            elif op == "$max" and self.plain_fields("int", "money", "float"):
                f = r.choice(self.plain_fields("int", "money", "float"))
                update["$max"] = {self.fpath(f): self.value(f)}
                pts.append(f"mantém o maior {self.fpath(f)}")
                ens.append(f"keep the max {self.fpath(f)}")
            elif op == "$arrayFilters" and self.dialect != "json":
                arrays = sorted({f.array_parent for f in self.coll.fields if f.array_parent})
                if not arrays:
                    continue
                parent = r.choice(arrays)
                subs = [f for f in self.coll.fields if f.array_parent == parent]
                numeric = [f for f in subs if f.numeric] or subs
                target, key_field = r.choice(numeric), r.choice(subs)
                leaf = S.style_segment(target.path.split(".")[-1], self.name_style)
                key_leaf = S.style_segment(key_field.path.split(".")[-1], self.name_style)
                update.setdefault("$set", {})[f"{self.ppath(parent)}.$[item].{leaf}"] = self.value(target)
                options = {"arrayFilters": [{f"item.{key_leaf}": self.value(key_field)}]}
                pts.append(f"atualiza {leaf} dos itens filtrados de {self.ppath(parent)}")
                ens.append(f"update {leaf} of filtered {self.ppath(parent)} elements")
        if not update:
            f = r.choice(plain)
            update["$set"] = {self.fpath(f): self.value(f)}
            pts.append("define " + self.fpath(f))
            ens.append("set " + self.fpath(f))
        if options is None and r.random() < .15:
            options = {"upsert": True}
        return update, "; ".join(pts), "; ".join(ens), options

    def document(self) -> dict:
        r = self.rng
        fields = [f for f in self.coll.fields if f.type != "geo" or r.random() < .2]
        chosen = r.sample(fields, min(len(fields), r.randint(3, 7)))
        doc: dict[str, Any] = {}
        arrays: dict[str, dict] = {}
        for f in chosen:
            if f.array_parent:
                leaf = S.style_segment(f.path.split(".")[-1], self.name_style)
                arrays.setdefault(self.ppath(f.array_parent), {})[leaf] = self.value(f)
                continue
            value = self.distinct_values(f, r.randint(1, 3)) if f.type == "tags" else self.value(f)
            node = doc
            segs = self.fpath(f).split(".")
            for seg in segs[:-1]:
                node = node.setdefault(seg, {})
                if not isinstance(node, dict):
                    break
            else:
                node[segs[-1]] = value
        for parent, item in arrays.items():
            doc[parent] = [item]
        return doc

    # -- agregação -----------------------------------------------------------------------------------------------------------
    def group_stage(self, unwound: str | None) -> tuple[dict, list[str], str, str]:
        r = self.rng
        groupable = self.plain_fields("enum", "state", "city", "country", "ref", "code")
        if unwound:
            groupable += [f for f in self.coll.fields if f.array_parent == unwound and f.type in ("code", "enum", "text")]
        dates = self.plain_fields("date")
        kind = pick(r, {"field": .6 if groupable else 0, "compound": .15 if len(groupable) >= 2 else 0, "date": .12 if dates else 0, "null": .13})
        if kind == "field":
            f = r.choice(groupable)
            gid: Any = "$" + self.fpath(f)
            label = self.fpath(f)
        elif kind == "compound":
            a, b = r.sample(groupable, 2)
            gid = {self.out(a.path.split(".")[-1]): "$" + self.fpath(a), self.out(b.path.split(".")[-1]): "$" + self.fpath(b)}
            label = f"{self.fpath(a)} e {self.fpath(b)}"
        elif kind == "date":
            d = r.choice(dates)
            if r.random() < .5:
                gid = {self.out("year"): {"$year": "$" + self.fpath(d)}, self.out("month"): {"$month": "$" + self.fpath(d)}}
            else:
                gid = {"$dateToString": {"format": r.choice(["%Y-%m", "%Y-%m-%d"]), "date": "$" + self.fpath(d)}}
            label = f"período de {self.fpath(d)}"
        else:
            gid = None
            label = "todos"
        group: dict[str, Any] = {"_id": gid}
        outputs: list[str] = []
        numeric = self.plain_fields("int", "money", "float")
        if unwound:
            numeric += [f for f in self.coll.fields if f.array_parent == unwound and f.numeric]
        if r.random() < .6 or not numeric:
            name = self.out("count")
            group[name] = {"$sum": 1} if r.random() < .8 else {"$count": {}}
            outputs.append(name)
        for f in r.sample(numeric, min(len(numeric), pick(r, {0: .2, 1: .5, 2: .3}))):
            op = r.choice(["$sum", "$sum", "$avg", "$max", "$min"])
            base = {"$sum": "total", "$avg": "avg", "$max": "max", "$min": "min"}[op]
            leaf = f.path.split(".")[-1]
            name = self.out(leaf if leaf.lower().startswith(base) else base + cap(leaf))
            if name in group:
                continue
            group[name] = {op: "$" + self.fpath(f)}
            outputs.append(name)
        if unwound:
            subs = [f for f in self.coll.fields if f.array_parent == unwound and f.numeric]
            if len(subs) >= 2 and r.random() < .5:
                name = self.out("revenue")
                group[name] = {"$sum": {"$multiply": ["$" + self.fpath(subs[0]), "$" + self.fpath(subs[1])]}}
                outputs.append(name)
        strings = self.plain_fields("name", "code", "text")
        if strings and r.random() < .15:
            f = r.choice(strings)
            name = self.out(singular(f.path.split(".")[-1]) + "List")
            group[name] = {r.choice(["$addToSet", "$push"]): "$" + self.fpath(f)}
        return {"$group": group}, outputs, f"agrupa por {label}", f"group by {label}"

    def lookup_stages(self, ref: S.Field) -> tuple[list[dict], str]:
        r = self.rng
        target = ref.arg
        tname = self.cname(target)
        as_name = self.out(singular(re.sub(r"[^A-Za-z]", "", target)))
        local = self.fpath(ref)
        if r.random() < .7:
            stages: list[dict] = [{"$lookup": {"from": tname, "localField": local, "foreignField": "_id", "as": as_name}}]
        else:
            tcoll = self.colls[target]
            fields = [f for f in tcoll.fields if not f.array_parent and f.type != "geo"]
            proj = {S.style_path(f.path, self.name_style): 1 for f in r.sample(fields, min(len(fields), r.randint(1, 3)))}
            stages = [{"$lookup": {"from": tname, "let": {"refId": "$" + local}, "pipeline": [
                {"$match": {"$expr": {"$eq": ["$_id", "$$refId"]}}}, {"$project": proj}], "as": as_name}}]
        if r.random() < .7:
            stages.append({"$unwind": "$" + as_name} if r.random() < .6 else
                          {"$unwind": {"path": "$" + as_name, "preserveNullAndEmptyArrays": True}})
        return stages, tname

    def set_stage(self) -> tuple[dict, str, str]:
        r = self.rng
        numeric = self.plain_fields("int", "money", "float")
        dates = self.plain_fields("date")
        strings = self.plain_fields("enum", "text", "name", "code", "city", "word")
        tags = self.plain_fields("tags")
        options = []
        if len(numeric) >= 2:
            a, b = r.sample(numeric, 2)
            options.append(({self.out("score"): {"$multiply": ["$" + self.fpath(a), "$" + self.fpath(b)]}}, "multiplica valores", "multiply values"))
        if numeric:
            a = r.choice(numeric)
            threshold = int(a.range()[1] / 2) or 1
            options.append(({self.out("tier"): {"$cond": {"if": {"$gte": ["$" + self.fpath(a), threshold]}, "then": "high", "else": "normal"}}},
                            "classifica por faixa", "classify by threshold"))
            options.append(({self.out("rounded" + cap(a.path.split(".")[-1])): {"$round": [{"$multiply": ["$" + self.fpath(a), 1.1]}, 2]}},
                            "calcula valor ajustado", "compute adjusted value"))
            options.append(({self.fpath(a): {"$ifNull": ["$" + self.fpath(a), 0]}}, "preenche valores nulos", "fill null values"))
        if dates:
            d = r.choice(dates)
            options.append(({self.out("year"): {"$year": "$" + self.fpath(d)}}, "extrai o ano", "extract the year"))
            options.append(({self.out("day"): {"$dateToString": {"format": "%Y-%m-%d", "date": "$" + self.fpath(d)}}}, "formata a data", "format the date"))
            options.append(({self.out("ageDays"): {"$dateDiff": {"startDate": "$" + self.fpath(d), "endDate": "$$NOW", "unit": "day"}}},
                            "calcula dias decorridos", "compute elapsed days"))
        if len(strings) >= 2:
            a, b = r.sample(strings, 2)
            options.append(({self.out("label"): {"$concat": ["$" + self.fpath(a), " - ", "$" + self.fpath(b)]}}, "concatena campos", "concatenate fields"))
        if strings:
            a = r.choice(strings)
            options.append(({self.out("upper" + cap(a.path.split(".")[-1])): {"$toUpper": "$" + self.fpath(a)}}, "converte para maiúsculas", "uppercase"))
        if tags:
            t = r.choice(tags)
            options.append(({self.out(singular(t.path) + "Count"): {"$size": {"$ifNull": ["$" + self.fpath(t), []]}}}, "conta tags", "count tags"))
        body, pt, en = r.choice(options) if options else ({self.out("flag"): True}, "marca documentos", "flag documents")
        return {r.choice(["$set", "$addFields"]): body}, pt, en

    def pipeline(self, complex_: bool = False) -> tuple[list[dict], str, str]:
        r = self.rng
        stages: list[dict] = []
        pts, ens = [], []
        arrays = sorted({f.array_parent for f in self.coll.fields if f.array_parent})
        refs = [f for f in self.plain_fields("ref") if f.arg in self.colls]
        numeric = self.plain_fields("int", "money", "float")
        if r.random() < (.95 if complex_ else .85):
            flt, pt, en = self.filter(n=pick(r, {1: .5, 2: .4, 3: .1}))
            stages.append({"$match": flt})
            pts.append("filtra " + pt)
            ens.append("match " + en)
        unwound = None
        if arrays and r.random() < (.5 if complex_ else .2):
            unwound = r.choice(arrays)
            path = "$" + self.ppath(unwound)
            stages.append({"$unwind": path} if r.random() < .6 else {"$unwind": {"path": path, "preserveNullAndEmptyArrays": r.random() < .5}})
            pts.append("desfaz " + self.ppath(unwound))
            ens.append("unwind " + self.ppath(unwound))
        if refs and r.random() < (.6 if complex_ else .2):
            more, tname = self.lookup_stages(r.choice(refs))
            stages += more
            pts.append("junta com " + tname)
            ens.append("join " + tname)
        if r.random() < (.5 if complex_ else .15):
            stage, pt, en = self.set_stage()
            stages.append(stage)
            pts.append(pt)
            ens.append(en)
        ending = pick(r, {"group": .55, "project": .2, "facet": .12 if complex_ else .03, "count": .08, "bucket": .07 if numeric else 0})
        if ending == "group":
            stage, outputs, pt, en = self.group_stage(unwound)
            stages.append(stage)
            pts.append(pt)
            ens.append(en)
            if outputs and r.random() < .8:
                stages.append({"$sort": {r.choice(outputs): -1}})
            if r.random() < .6:
                stages.append({"$limit": r.choice([5, 10, 10, 20, 50, 100])})
            if r.random() < .3:
                group_id = stage["$group"]["_id"]
                proj: dict[str, Any] = {"_id": 0}
                if isinstance(group_id, str):
                    proj[group_id[1:].split(".")[-1]] = "$_id"
                for name in outputs:
                    proj[name] = 1
                stages.append({"$project": proj})
        elif ending == "project":
            proj = self.projection()
            stages.append({"$project": proj})
            stages.append({"$sort": self.sort_spec()})
            if r.random() < .3:
                stages.append({"$skip": r.choice([10, 20, 50])})
            stages.append({"$limit": r.choice([10, 20, 25, 50])})
            pts.append("projeta e ordena")
            ens.append("project and sort")
        elif ending == "facet":
            group, outputs, _, _ = self.group_stage(None)
            sort_field = r.choice(numeric) if numeric else None
            facet: dict[str, Any] = {self.out("byGroup"): [group, {"$sort": {outputs[0]: -1}}] if outputs else [group]}
            if sort_field:
                facet[self.out("top")] = [{"$sort": {self.fpath(sort_field): -1}}, {"$limit": r.choice([3, 5, 10])}]
            facet[self.out("total")] = [{"$count": self.out("count")}]
            stages.append({"$facet": facet})
            pts.append("resume em facetas")
            ens.append("summarize with facets")
        elif ending == "count":
            groupable = self.plain_fields("enum", "state", "city")
            if groupable and r.random() < .5:
                f = r.choice(groupable)
                stages.append({"$sortByCount": "$" + self.fpath(f)})
                pts.append(f"conta por {self.fpath(f)}")
                ens.append(f"count by {self.fpath(f)}")
            else:
                stages.append({"$count": self.out("total")})
                pts.append("conta documentos")
                ens.append("count documents")
        else:
            f = r.choice(numeric)
            lo, hi = f.range()
            step = max(1, int((hi - lo) / 4))
            bounds = sorted({int(lo), int(lo) + step, int(lo) + 2 * step, int(hi) + 1})
            stages.append({"$bucket": {"groupBy": "$" + self.fpath(f), "boundaries": bounds, "default": "other",
                                       "output": {self.out("count"): {"$sum": 1}}}})
            pts.append(f"distribui {self.fpath(f)} em faixas")
            ens.append(f"bucket {self.fpath(f)} into ranges")
        if not stages or all("$match" in s for s in stages):
            stages.append({"$limit": 20})
        return stages, "; ".join(pts), "; ".join(ens)

    def search_pipeline(self) -> tuple[list[dict], str, str]:
        r = self.rng
        texts = self.plain_fields("text", "name") or self.plain_fields("enum", "word", "code", "city")
        numeric_dates = self.plain_fields("int", "money", "float", "date")
        exacts = self.plain_fields("bool", "ref")
        index = r.choice(S.DOMAINS[self.domain]["search_index"])
        search: dict[str, Any] = {}
        if r.random() < .85:
            search["index"] = index
        f = r.choice(texts)
        path = self.fpath(f)

        def text_op(field: S.Field) -> dict:
            op: dict[str, Any] = {"query": self.text_term(field)}
            others = [t for t in texts if t is not field]
            op["path"] = [self.fpath(field), self.fpath(r.choice(others))] if others and r.random() < .2 else self.fpath(field)
            if r.random() < .3:
                op["fuzzy"] = {"maxEdits": r.choice([1, 2])}
            return {"text": op}

        def filter_op() -> dict:
            if numeric_dates and (r.random() < .6 or not exacts):
                nf = r.choice(numeric_dates)
                a, b = sorted([self.value(nf), self.value(nf)], key=lambda v: v.iso if isinstance(v, DateV) else v)
                body = {"path": self.fpath(nf), "gte": a}
                if r.random() < .5:
                    body["lte"] = b
                return {"range": body}
            ef = r.choice(exacts)
            return {"equals": {"path": self.fpath(ef), "value": self.value(ef)}}

        kind = pick(r, {"text": .3, "autocomplete": .2, "compound": .3, "phrase": .08, "range": .06 if numeric_dates else 0, "equals": .06 if exacts else 0})
        if kind == "text":
            search.update(text_op(f))
            pt, en = f"busca textual em {path}", f"full-text search on {path}"
        elif kind == "autocomplete":
            term = self.text_term(f)
            body = {"query": term[: r.randint(2, 4)], "path": path}
            if r.random() < .3:
                body["tokenOrder"] = r.choice(["sequential", "any"])
            if r.random() < .2:
                body["fuzzy"] = {"maxEdits": 1}
            search["autocomplete"] = body
            pt, en = f"autocomplete em {path}", f"autocomplete on {path}"
        elif kind == "phrase":
            body = {"query": str(self.value(f)), "path": path}
            if r.random() < .3:
                body["slop"] = 1
            search["phrase"] = body
            pt, en = f"busca frase em {path}", f"phrase search on {path}"
        elif kind == "range":
            search.update(filter_op() if numeric_dates else text_op(f))
            pt, en = "busca por intervalo", "range search"
        elif kind == "equals":
            ef = r.choice(exacts)
            search["equals"] = {"path": self.fpath(ef), "value": self.value(ef)}
            pt, en = f"busca exata em {self.fpath(ef)}", f"exact match on {self.fpath(ef)}"
        else:
            compound: dict[str, Any] = {"must": [text_op(f)]}
            if r.random() < .5:
                other = r.choice(texts)
                compound["should"] = [{"phrase": {"query": str(self.value(other)), "path": self.fpath(other)}}
                                      if r.random() < .5 else text_op(other)]
                if r.random() < .5:
                    compound["minimumShouldMatch"] = 1
            if r.random() < .6 and (numeric_dates or exacts):
                compound["filter"] = [filter_op()]
            enums = self.plain_fields("enum")
            if enums and r.random() < .2:
                ef = r.choice(enums)
                compound["mustNot"] = [{"equals": {"path": self.fpath(ef), "value": self.value(ef)}}]
            search["compound"] = compound
            pt, en = f"busca composta em {path}", f"compound search on {path}"
        highlight = r.random() < .15 and kind in ("text", "compound")
        if highlight:
            search["highlight"] = {"path": path}
        enums = self.plain_fields("enum")
        if r.random() < .08 and enums:
            ef = r.choice(enums)
            meta = {"index": index, "facet": {"operator": text_op(f), "facets": {self.out("facet" + cap(ef.path.split(".")[-1])): {"type": "string", "path": self.fpath(ef)}}}}
            return [{"$searchMeta": meta}], f"facetas de busca por {self.fpath(ef)}", f"search facets by {self.fpath(ef)}"
        stages: list[dict] = [{"$search": search}]
        if r.random() < .25:
            flt, _, _ = self.filter(n=1, logical=0)
            stages.append({"$match": flt})
        if r.random() < .8:
            stages.append({"$limit": r.choice([5, 10, 10, 20, 50])})
        if r.random() < .7:
            proj = {self.fpath(x): 1 for x in r.sample(self.plain_fields(), min(len(self.plain_fields()), r.randint(1, 3)))}
            proj[self.out("score")] = {"$meta": "searchScore"}
            if highlight:
                proj[self.out("highlights")] = {"$meta": "searchHighlights"}
            stages.append({"$project": proj})
        return stages, pt, en

    # -- categorias ----------------------------------------------------------------------------------------------------------
    def aggregate_statement(self, stages: list[dict], pt: str, en: str) -> Stmt:
        if self.json:
            return Stmt(render(stages, self.style), "Pipeline: " + pt, "Pipeline: " + en)
        return self.statement(self.target_segments(), [("aggregate", [stages])], f"Agrega {self.cname(self.coll_key)}: {pt}",
                              f"Aggregate {self.cname(self.coll_key)}: {en}")

    def cursor_calls(self, allow_project: bool = True) -> list[tuple[str, list]]:
        r = self.rng
        calls: list[tuple[str, list]] = []
        if r.random() < .5:
            calls.append(("sort", [self.sort_spec()]))
        if r.random() < .15:
            calls.append(("skip", [r.choice([10, 20, 50, 100])]))
        if r.random() < .6:
            calls.append(("limit", [r.choice([1, 5, 10, 20, 50, 100])]))
        if self.dialect == "console" and allow_project and r.random() < .08:
            calls.append(("project", [self.projection()]))
        if r.random() < .04:
            calls.append(("maxTimeMS", [r.choice([1000, 5000, 30000])]))
        if self.dialect == "console" and r.random() < .03:
            calls.append(("collation", [{"locale": "pt", "strength": 2}]))
        if r.random() < .08:
            calls.append(("toArray", []))
        return calls

    def cat_crud(self) -> list[Stmt]:
        r = self.rng
        ops = {"find": .3, "findOne": .1, "countDocuments": .08, "estimatedDocumentCount": .02, "distinct": .05, "insertOne": .08,
               "insertMany": .05, "updateOne": .08, "updateMany": .07, "replaceOne": .03, "deleteOne": .05, "deleteMany": .04}
        if self.dialect == "mongosh":
            ops.update({"bulkWrite": .04, "findOneAndUpdate": .03})
        op = pick(r, ops)
        self.subcategory = op
        name = self.cname(self.coll_key)
        target = self.target_segments()
        if op in ("find", "findOne"):
            flt, pt, en = self.filter()
            args: list = [flt]
            if r.random() < .3:
                args.append(self.projection())
            calls = [(op, args)] + (self.cursor_calls() if op == "find" else [])
            return [self.statement(target, calls, f"Busca {name} onde {pt}", f"Find {name} where {en}")]
        if op == "countDocuments":
            flt, pt, en = self.filter()
            return [self.statement(target, [(op, [flt])], f"Conta {name} onde {pt}", f"Count {name} where {en}")]
        if op == "estimatedDocumentCount":
            return [self.statement(target, [(op, [])], f"Estima o total de {name}", f"Estimate the {name} count")]
        if op == "distinct":
            f = r.choice(self.plain_fields("enum", "city", "state", "code", "ref", "tags") or self.plain_fields())
            args = [self.fpath(f)]
            if r.random() < .5:
                args.append(self.filter(n=1)[0])
            return [self.statement(target, [(op, args)], f"Valores distintos de {self.fpath(f)} em {name}", f"Distinct {self.fpath(f)} values in {name}")]
        if op == "insertOne":
            return [self.statement(target, [(op, [self.document()])], f"Insere um documento em {name}", f"Insert one document into {name}")]
        if op == "insertMany":
            docs = [self.document() for _ in range(r.randint(2, 3))]
            args = [docs] + ([{"ordered": False}] if r.random() < .15 else [])
            return [self.statement(target, [(op, args)], f"Insere {len(docs)} documentos em {name}", f"Insert {len(docs)} documents into {name}")]
        if op in ("updateOne", "updateMany", "findOneAndUpdate"):
            flt, fpt, fen = self.filter(n=pick(r, {1: .6, 2: .4}))
            update, upt, uen, options = self.update()
            args = [flt, update]
            if op == "findOneAndUpdate":
                options = {**(options or {}), "returnDocument": "after"}
            if options:
                args.append(options)
            return [self.statement(target, [(op, args)], f"Atualiza {name} onde {fpt}: {upt}", f"Update {name} where {fen}: {uen}")]
        if op == "replaceOne":
            flt, pt, en = self.filter(n=1)
            return [self.statement(target, [(op, [flt, self.document()])], f"Substitui um documento de {name} onde {pt}", f"Replace a {name} document where {en}")]
        if op in ("deleteOne", "deleteMany"):
            flt, pt, en = self.filter(n=pick(r, {1: .6, 2: .4}))
            return [self.statement(target, [(op, [flt])], f"Remove {name} onde {pt}", f"Delete {name} where {en}")]
        # bulkWrite
        ops_list = []
        for _ in range(r.randint(2, 3)):
            kind = r.choice(["insertOne", "updateOne", "deleteOne", "updateMany"])
            if kind == "insertOne":
                ops_list.append({"insertOne": {"document": self.document()}})
            elif kind in ("updateOne", "updateMany"):
                update, _, _, _ = self.update()
                ops_list.append({kind: {"filter": self.filter(n=1)[0], "update": update}})
            else:
                ops_list.append({"deleteOne": {"filter": self.filter(n=1)[0]}})
        return [self.statement(target, [("bulkWrite", [ops_list])], f"Executa operações em lote em {name}", f"Run bulk operations on {name}")]

    def cat_filters(self) -> list[Stmt]:
        r = self.rng
        op = pick(r, {"find": .55, "countDocuments": .2, "deleteMany": .08, "updateMany": .07, "findOne": .1})
        self.subcategory = "filter_" + op
        flt, pt, en = self.filter(n=pick(r, {2: .45, 3: .4, 4: .15}), logical=.45)
        name = self.cname(self.coll_key)
        target = self.target_segments()
        if op == "updateMany":
            update, upt, uen, options = self.update()
            args = [flt, update] + ([options] if options else [])
            return [self.statement(target, [(op, args)], f"Atualiza {name} onde {pt}: {upt}", f"Update {name} where {en}: {uen}")]
        calls = [(op, [flt])] + (self.cursor_calls() if op == "find" else [])
        verb = {"find": ("Busca", "Find"), "countDocuments": ("Conta", "Count"), "deleteMany": ("Remove", "Delete"), "findOne": ("Busca um", "Find one")}[op]
        return [self.statement(target, calls, f"{verb[0]} {name} onde {pt}", f"{verb[1]} {name} where {en}")]

    def cat_aggregation(self) -> list[Stmt]:
        stages, pt, en = self.pipeline(False)
        self.subcategory = "pipeline_" + "_".join(next(iter(s)).lstrip("$") for s in stages[:3])
        return [self.aggregate_statement(stages, pt, en)]

    def cat_complex_queries(self) -> list[Stmt]:
        stages, pt, en = self.pipeline(True)
        while len(stages) < 5:
            extra, _, _ = self.set_stage()
            stages.insert(1, extra)
        self.subcategory = f"complex_{len(stages)}_stages"
        return [self.aggregate_statement(stages, pt, en)]

    def cat_atlas_search(self) -> list[Stmt]:
        r = self.rng
        name = self.cname(self.coll_key)
        if self.dialect == "mongosh" and r.random() < .15:
            texts = self.plain_fields("text", "name", "enum")
            if r.random() < .6 and texts:
                fields = {self.fpath(f): {"type": r.choice(["string", "autocomplete"]) if f.type != "enum" else "token"} for f in r.sample(texts, min(len(texts), 2))}
                definition = {"mappings": {"dynamic": r.random() < .3, "fields": fields}}
                index = r.choice(S.DOMAINS[self.domain]["search_index"])
                self.subcategory = "search_index_create"
                return [self.statement(self.target_segments(), [("createSearchIndex", [index, definition])],
                                       f"Cria índice Atlas Search em {name}", f"Create an Atlas Search index on {name}")]
            self.subcategory = "search_index_list"
            return [self.statement(self.target_segments(), [("getSearchIndexes", [])], f"Lista índices Atlas Search de {name}", f"List Atlas Search indexes of {name}")]
        stages, pt, en = self.search_pipeline()
        self.subcategory = "search_" + next(k for k in (next(iter(stages[0].values())) or {}) if k not in ("index", "highlight")) if "$search" in stages[0] else "search_meta"
        return [self.aggregate_statement(stages, pt, en)]

    def cat_indexes(self) -> list[Stmt]:
        r = self.rng
        name = self.cname(self.coll_key)
        target = self.target_segments()
        kinds = {"compound": .25, "unique": .12 if self.plain_fields("email", "code") else 0, "ttl": .1 if self.plain_fields("date") else 0,
                 "partial": .1 if self.plain_fields("bool", "enum") else 0, "text": .1 if self.plain_fields("text") else 0,
                 "geo": .06 if self.plain_fields("geo") else 0, "hashed": .04, "wildcard": .05, "drop": .06, "hint": .07}
        if self.dialect == "mongosh":
            kinds.update({"list": .08, "many": .05, "explain": .05})
        kind = pick(r, kinds)
        self.subcategory = "index_" + kind

        def keys(n: int) -> dict:
            fields = self.plain_fields("enum", "date", "int", "money", "code", "ref", "city", "bool", "name", "email")
            return {self.fpath(f): r.choice([1, 1, -1]) for f in r.sample(fields, min(len(fields), n))}

        def auto_name(spec: dict) -> str:
            return "_".join(f"{k}_{v}" for k, v in spec.items())

        if kind in ("compound", "hashed", "wildcard"):
            if kind == "compound":
                spec = keys(r.randint(1, 3))
            elif kind == "hashed":
                spec = {self.fpath(r.choice(self.plain_fields("code", "ref", "enum", "name"))): "hashed"}
            else:
                objects = sorted({f.path.split(".")[0] for f in self.coll.fields if "." in f.path and not f.array_parent})
                spec = {(objects and r.random() < .7 and self.ppath(r.choice(objects)) + ".$**") or "$**": 1}
            args: list = [spec] + ([{"name": auto_name(spec)}] if r.random() < .3 and kind == "compound" else [])
            return [self.statement(target, [("createIndex", args)], f"Cria índice em {name} para {', '.join(spec)}", f"Create an index on {name} for {', '.join(spec)}")]
        if kind == "unique":
            f = r.choice(self.plain_fields("email", "code"))
            return [self.statement(target, [("createIndex", [{self.fpath(f): 1}, {"unique": True}])], f"Garante {self.fpath(f)} único em {name}", f"Enforce unique {self.fpath(f)} on {name}")]
        if kind == "ttl":
            f = r.choice(self.plain_fields("date"))
            seconds = r.choice([3600, 86400, 604800, 2592000])
            return [self.statement(target, [("createIndex", [{self.fpath(f): 1}, {"expireAfterSeconds": seconds}])], f"Índice TTL em {self.fpath(f)} de {name}", f"TTL index on {self.fpath(f)} of {name}")]
        if kind == "partial":
            cond_field = r.choice(self.plain_fields("bool", "enum"))
            value, _, _ = self.condition(cond_field)
            spec = keys(r.randint(1, 2))
            return [self.statement(target, [("createIndex", [spec, {"partialFilterExpression": {self.fpath(cond_field): value}}])],
                                   f"Índice parcial em {name}", f"Partial index on {name}")]
        if kind == "text":
            texts = self.plain_fields("text")
            spec = {self.fpath(f): "text" for f in r.sample(texts, min(len(texts), 2))}
            options = {"name": r.choice(["text_idx", "search_text", auto_name(spec)])}
            if r.random() < .5:
                options["weights"] = {next(iter(spec)): r.choice([5, 10])}
            return [self.statement(target, [("createIndex", [spec, options])], f"Índice de texto em {name}", f"Text index on {name}")]
        if kind == "geo":
            f = r.choice(self.plain_fields("geo"))
            return [self.statement(target, [("createIndex", [{self.fpath(f): "2dsphere"}])], f"Índice geoespacial em {self.fpath(f)}", f"Geospatial index on {self.fpath(f)}")]
        if kind == "drop":
            return [self.statement(target, [("dropIndex", [auto_name(keys(r.randint(1, 2)))])], f"Remove índice de {name}", f"Drop an index from {name}")]
        if kind == "hint":
            spec = keys(r.randint(1, 2))
            flt, pt, en = self.filter(n=1)
            return [self.statement(target, [("find", [flt]), ("hint", [spec])] + [("limit", [r.choice([10, 20, 50])])],
                                   f"Busca {name} forçando índice", f"Find {name} forcing an index")]
        if kind == "list":
            return [self.statement(target, [("getIndexes", [])], f"Lista índices de {name}", f"List indexes of {name}")]
        if kind == "many":
            specs = [keys(1), keys(2)]
            return [self.statement(target, [("createIndexes", [specs])], f"Cria vários índices em {name}", f"Create several indexes on {name}")]
        flt, pt, en = self.filter(n=1)
        return [self.statement(target, [("find", [flt]), ("explain", ["executionStats"])], f"Explica plano da busca em {name}", f"Explain the query plan on {name}")]

    def out_call(self, expr: str, json_output: bool = False) -> str:
        if self.dialect == "console":
            return f"console.log({expr})"
        return f"{'printjson' if json_output else 'print'}({expr})"

    def cat_mongosh_js(self) -> list[Stmt]:
        r = self.rng
        s = self.style
        ind = " " * s.indent
        semi = self.semi()
        name = self.cname(self.coll_key)
        tgt = "".join(self.target_segments(allow_bind=False))
        dates = self.plain_fields("date")
        numeric = [f for f in self.plain_fields("int", "money", "float") if "." not in f.path]
        enums = self.plain_fields("enum")
        refs = [f for f in self.plain_fields("ref") if f.arg in self.colls]
        templates = {"recent": .12 if dates else 0, "update_each": .12, "distinct_in": .1 if refs else 0, "sum": .12 if numeric else 0,
                     "function": .1 if enums else 0, "try": .1, "map": .12, "if_count": .1, "agg_rows": .07, "update_result": .05}
        kind = pick(r, templates)
        self.subcategory = "js_" + kind
        inline = lambda v: render_inline(v, s)  # noqa: E731
        if kind == "recent":
            d = self.fpath(r.choice(dates))
            days = r.choice([1, 7, 30, 90])
            n = r.choice([10, 20, 50, 100])
            code = (f"const since = new Date(Date.now() - {days} * 24 * 60 * 60 * 1000){semi}\n"
                    f"const recent = {tgt}.find({inline({d: {'$gte': Raw('since')}})}).sort({inline({d: -1})}).limit({n}).toArray(){semi}\n"
                    f"{self.out_call('recent.length')}{semi}")
            return [Stmt(code, f"Lista {name} dos últimos {days} dias e mostra a quantidade", f"List {name} from the last {days} days and print the count")]
        if kind == "update_each":
            flt, pt, en = self.filter(n=1)
            f = r.choice([x for x in self.plain_fields() if x.type not in ("geo", "tags")])
            iterate = ".forEach(" if self.dialect == "mongosh" else ".toArray().forEach("
            code = (f"{tgt}.find({inline(flt)}){iterate}doc => {{\n"
                    f"{ind}{tgt}.updateOne({inline({'_id': Raw('doc._id')})}, {inline({'$set': {self.fpath(f): self.value(f)}})}){semi}\n"
                    f"}}){semi}")
            return [Stmt(code, f"Atualiza {self.fpath(f)} de cada documento de {name} onde {pt}", f"Update {self.fpath(f)} for each {name} document where {en}")]
        if kind == "distinct_in":
            ref = r.choice(refs)
            other = self.cname(ref.arg)
            other_fields = [x for x in self.colls[ref.arg].fields if not x.array_parent and x.type != "geo"]
            proj = {S.style_path(x.path, self.name_style): 1 for x in r.sample(other_fields, min(len(other_fields), 2))}
            flt, pt, en = self.filter(n=1)
            other_target = "".join(self.target_segments(ref.arg, allow_bind=False))
            code = (f"const ids = {tgt}.distinct({self.q(self.fpath(ref))}, {inline(flt)}){semi}\n"
                    f"const docs = {other_target}.find({inline({'_id': {'$in': Raw('ids')}})}, {inline(proj)}).toArray(){semi}\n"
                    f"{self.out_call('docs', json_output=True)}{semi}")
            return [Stmt(code, f"Busca {other} referenciados por {name} onde {pt}", f"Fetch {other} referenced by {name} where {en}")]
        if kind == "sum":
            f = r.choice(numeric)
            flt, pt, en = self.filter(n=1)
            leaf = self.fpath(f)
            code = (f"let total = 0{semi}\n"
                    f"for (const doc of {tgt}.find({inline(flt)}).toArray()) {{\n"
                    f"{ind}total += doc.{leaf}{semi}\n"
                    f"}}\n"
                    f"{self.out_call('total')}{semi}")
            return [Stmt(code, f"Soma {leaf} de {name} onde {pt}", f"Sum {leaf} of {name} where {en}")]
        if kind == "function":
            f = r.choice(enums)
            fn = "count" + cap(re.sub(r"[^A-Za-z0-9]", "", S.style_collection(self.coll_key, "camel"))) + "By" + cap(f.path.split(".")[-1].replace("_", ""))
            code = (f"function {fn}(value) {{\n"
                    f"{ind}return {tgt}.countDocuments({inline({self.fpath(f): Raw('value')})}){semi}\n"
                    f"}}\n"
                    f"{self.out_call(f'{fn}({self.q(self.value(f))})')}{semi}")
            return [Stmt(code, f"Função que conta {name} por {self.fpath(f)}", f"Function that counts {name} by {self.fpath(f)}")]
        if kind == "try":
            code = (f"try {{\n"
                    f"{ind}{tgt}.insertOne({inline(self.document())}){semi}\n"
                    f"}} catch (e) {{\n"
                    f"{ind}{self.out_call('e.message')}{semi}\n"
                    f"}}")
            return [Stmt(code, f"Insere em {name} tratando erro", f"Insert into {name} handling errors")]
        if kind == "map":
            f = r.choice([x for x in self.plain_fields() if IDENT.match(self.fpath(x).replace(".", "_"))])
            flt, pt, en = self.filter(n=1)
            code = (f"const values = {tgt}.find({inline(flt)}, {inline({self.fpath(f): 1})}).toArray().map(d => d.{self.fpath(f)}){semi}\n"
                    f"{self.out_call('values')}{semi}")
            return [Stmt(code, f"Extrai {self.fpath(f)} de {name} onde {pt}", f"Extract {self.fpath(f)} from {name} where {en}")]
        if kind == "if_count":
            flt, pt, en = self.filter(n=1)
            n = r.choice([10, 100, 1000])
            code = (f"const pending = {tgt}.countDocuments({inline(flt)}){semi}\n"
                    f"if (pending > {n}) {{\n"
                    f"{ind}{self.out_call(self.q(name + ': ') + ' + pending')}{semi}\n"
                    f"}}")
            return [Stmt(code, f"Avisa quando {name} onde {pt} passar de {n}", f"Warn when {name} where {en} exceeds {n}")]
        if kind == "agg_rows":
            group, outputs, pt, en = self.group_stage(None)
            stages = [group, {"$sort": {outputs[0]: -1}}, {"$limit": 10}]
            code = (f"const rows = {tgt}.aggregate({render_inline(stages, s)}).toArray(){semi}\n"
                    f"rows.forEach(row => {self.out_call('row._id + ' + self.q(': ') + ' + row.' + outputs[0])}){semi}")
            return [Stmt(code, f"Mostra ranking de {name}: {pt}", f"Print a {name} ranking: {en}")]
        flt, pt, en = self.filter(n=1)
        update, upt, uen, _ = self.update()
        code = (f"const result = {tgt}.updateMany({inline(flt)}, {inline(update)}){semi}\n"
                f"{self.out_call('result.modifiedCount')}{semi}")
        return [Stmt(code, f"Atualiza {name} onde {pt} e mostra modificados", f"Update {name} where {en} and print modified count")]

    def cat_dsl_ide(self) -> list[Stmt]:
        r = self.rng
        s = self.style
        semi = self.semi()
        q = self.q
        name = self.cname(self.coll_key)
        kind = pick(r, {"conn_chain": .2, "compare": .12, "pool": .16, "env": .12, "bson": .12, "ejson": .07, "db_api": .07,
                        "numbers": .05, "cursor_options": .09})
        self.subcategory = "dsl_" + kind
        inline = lambda v: render_inline(v, s)  # noqa: E731
        if kind == "conn_chain":
            flt, pt, en = self.filter()
            calls = [("find", [flt])] + self.cursor_calls() if r.random() < .75 else [("aggregate", [self.pipeline()[0]])]
            return [self.statement(self.target_segments(force="conn"), calls, f"Consulta {name} na conexão {self.connection}", f"Query {name} on connection {self.connection}")]
        if kind == "compare":
            other = r.choice([c for c in S.CONNECTION_NAMES if c != self.connection])
            flt, pt, en = self.filter(n=1)
            a, b = r.choice([("dev", "prod"), ("source", "target"), ("staging", "production"), ("left", "right")])
            db_name = q(self.database)
            code = (f"const {a} = getConnection({q(self.connection)}).getDatabase({db_name}).getCollection({q(name)}){semi}\n"
                    f"const {b} = getConnection({q(other)}).getDatabase({db_name}).getCollection({q(name)}){semi}\n"
                    f"const {a}Count = {a}.countDocuments({inline(flt)}){semi}\n"
                    f"const {b}Count = {b}.countDocuments({inline(flt)}){semi}\n"
                    f"console.log({a}Count, {b}Count){semi}")
            return [Stmt(code, f"Compara contagem de {name} entre {self.connection} e {other}", f"Compare {name} counts between {self.connection} and {other}")]
        if kind == "pool":
            flt, pt, en = self.filter()
            calls = [(r.choice(["find", "find", "findOne", "countDocuments"]), [flt])]
            if calls[0][0] == "find":
                calls += self.cursor_calls()
            return [self.statement(self.target_segments(force="pool"), calls, f"Consulta {name} via ConnectionPool", f"Query {name} through ConnectionPool")]
        if kind == "env":
            key = r.choice(S.ENV_KEYS)
            var = re.sub(r"_([a-z])", lambda m: m[1].upper(), key.lower())
            f = r.choice([x for x in self.plain_fields() if x.type in STRINGY + ("ref",)] or self.plain_fields())
            tgt = self.chain(self.target_segments(allow_bind=False), [("find", [{self.fpath(f): Raw(var)}])] + [("limit", [r.choice([10, 20, 50])])])
            code = f"const {var} = ENV.get({q(key)}){semi}\n{tgt}{semi}"
            return [Stmt(code, f"Filtra {name} por {self.fpath(f)} vindo do ambiente ({key})", f"Filter {name} by {self.fpath(f)} from the environment ({key})")]
        if kind == "bson":
            tgt = self.target_segments()
            oid = S.oid(r)
            op = r.choice(["deleteOne", "findOne", "updateOne"])
            args: list = [{"_id": oid}]
            if op == "updateOne":
                update, _, _, _ = self.update()
                args.append(update)
            uuid_fields = self.plain_fields("uuid")
            if uuid_fields and r.random() < .4:
                f = r.choice(uuid_fields)
                args = [{self.fpath(f): self.value(f)}] + args[1:]
            return [self.statement(tgt, [(op, args)], f"{op} em {name} por identificador", f"{op} on {name} by identifier")]
        if kind == "ejson":
            f = r.choice(self.plain_fields("int", "money", "float") or self.plain_fields())
            v = self.value(f)
            payload = json.dumps({self.fpath(f): {"$gte": v if isinstance(v, (int, float)) else str(display(v))}}, ensure_ascii=False)
            tgt = self.chain(self.target_segments(allow_bind=False), [("find", [Raw("filter")]), ("limit", [20])])
            code = f"const filter = EJSON.parse('{payload}'){semi}\n{tgt}{semi}"
            return [Stmt(code, f"Monta filtro Extended JSON para {name}", f"Build an Extended JSON filter for {name}")]
        if kind == "db_api":
            choice = r.choice(["name", "create", "stats", "sibling"])
            if choice == "name":
                return [Stmt(f"console.log(db.getName()){semi}", "Mostra o banco atual", "Print the current database")]
            if choice == "create":
                return [Stmt(f"db.createCollection({q(name + '_archive')}){semi}", f"Cria coleção de arquivo para {name}", f"Create an archive collection for {name}")]
            if choice == "stats":
                return [Stmt(f"db.getCollection({q(name)}).stats(){semi}", f"Estatísticas de {name}", f"Statistics of {name}")]
            other_db = r.choice(S.DOMAINS[self.domain]["databases"])
            return [Stmt(f"db.getSiblingDB({q(other_db)}).getCollection({q(name)}).estimatedDocumentCount(){semi}",
                         f"Conta {name} no banco {other_db}", f"Count {name} in database {other_db}")]
        if kind == "numbers":
            money = self.plain_fields("money") or self.plain_fields("int", "float")
            ints = self.plain_fields("int")
            doc = self.document()
            if money:
                doc[self.fpath(r.choice(money)).split(".")[0]] = Raw(f"NumberDecimal({q(number(round(r.uniform(1, 999), 2)))})")
            if ints:
                doc[self.fpath(r.choice(ints)).split(".")[0]] = Raw(f"NumberLong({q(str(r.randint(10**9, 10**12)))})")
            return [self.statement(self.target_segments(), [("insertOne", [doc])], f"Insere em {name} com tipos numéricos BSON", f"Insert into {name} with BSON numeric types")]
        flt, pt, en = self.filter(n=1)
        calls = [("find", [flt])]
        extras = [("hint", [{self.fpath(r.choice(self.plain_fields())): 1}]), ("collation", [{"locale": "pt", "strength": 2}]),
                  ("maxTimeMS", [5000]), ("batchSize", [100]), ("comment", ["relatorio"]), ("project", [self.projection()])]
        calls += r.sample(extras, r.randint(1, 3)) + [("limit", [r.choice([10, 50, 100])])]
        return [self.statement(self.target_segments(), calls, f"Busca {name} com opções de cursor", f"Find {name} with cursor options")]

    # -- montagem ------------------------------------------------------------------------------------------------------------
    def simple_statement(self) -> Stmt:
        r = self.rng
        saved = self.coll_key, self.coll
        self.coll_key = r.choice(list(self.colls))
        self.coll = self.colls[self.coll_key]
        name = self.cname(self.coll_key)
        target = self.target_segments(allow_bind=False)
        flt, pt, en = self.filter(n=1)
        op = r.choice(["find", "countDocuments", "findOne"])
        calls = [(op, [flt])] + ([("limit", [r.choice([5, 10, 20])])] if op == "find" else [])
        stmt = self.statement(target, calls, f"{op} em {name} onde {pt}", f"{op} on {name} where {en}", main=False)
        self.coll_key, self.coll = saved
        return stmt

    def build(self) -> dict:
        r = self.rng
        if self.json and self.category not in ("aggregation", "complex_queries", "atlas_search"):
            self.category = "aggregation"
        before = [self.simple_statement() for _ in range(pick(r, {0: .75, 1: .2, 2: .05}))] if not self.json else []
        retries = 0
        while True:
            try:
                main = getattr(self, "cat_" + self.category)()
                break
            except (IndexError, ValueError):
                # Template sem campos compatíveis nesta coleção: troca a coleção (registrado em "retries").
                retries += 1
                if retries > 8:
                    raise
                self.prelude, self.bound = [], {}
                self.coll_key = r.choice(list(self.colls))
                self.coll = self.colls[self.coll_key]
        self.retries = retries
        after = [self.simple_statement() for _ in range(pick(r, {0: .75, 1: .2, 2: .05}))] if not self.json else []
        statements = self.prelude + before + main + after
        code = ""
        spans = []
        for index, stmt in enumerate(statements):
            if index:
                code += "\n\n" if r.random() < .25 else "\n"
            start = len(code)
            code += stmt.code
            spans.append({"start": start, "end": len(code), "pt": stmt.pt, "en": stmt.en, "main": stmt.main})
        if r.random() < .5:
            code += "\n"
        return {
            "domain": self.domain, "category": self.category, "subcategory": self.subcategory, "dialect": self.dialect,
            "code": code, "statements": spans, "context": self.context(), "template_retries": retries,
            "style": {"names": self.name_style, "collections": self.coll_style, "indent": self.style.indent,
                      "width": self.style.width, "semicolons": self.style.semicolons},
        }

    def context(self) -> dict | None:
        r = self.rng
        cfg = self.context_cfg
        if r.random() < cfg.get("none_probability", .08):
            return None
        conns = list(dict.fromkeys([self.connection] + r.sample(S.CONNECTION_NAMES, r.randint(0, 3))))
        dbs = list(dict.fromkeys([self.database] + r.sample(S.DOMAINS[self.domain]["databases"], r.randint(0, 2))))
        colls = [self.cname(k) for k in self.colls]
        other_domain = r.choice([d for d in S.DOMAINS if d != self.domain and d != "education"])
        distractors = [self.cname(k) for k in r.sample(list(S.DOMAINS[other_domain]["collections"]), r.randint(0, 2))]
        known = list(dict.fromkeys(conns + dbs + colls + distractors))
        current = self.cname(self.coll_key) if r.random() < .7 else ""
        known += [self.connection, self.database, current]
        fields: list[str] = []
        if r.random() < cfg.get("result_fields_probability", .6):
            fields = ["_id"]
            for f in self.coll.fields:
                segs = self.fpath(f).split(".")
                for i in range(1, len(segs) + 1):
                    path = ".".join(segs[:i])
                    if path not in fields:
                        fields.append(path)
                if f.type == "geo":
                    fields += [self.fpath(f) + ".type", self.fpath(f) + ".coordinates"]
            fields = [p for p in fields if p == "_id" or r.random() > .12][:128]
        roll = r.random()
        if roll < .72:
            input_panel = "{}"
        elif roll < .87:
            sample_fields = r.sample(self.plain_fields(*STRINGY, "int", "bool"), min(2, len(self.plain_fields(*STRINGY, "int", "bool"))))
            input_panel = json.dumps({self.out(f.path.split(".")[-1]): S.sample_value(f, r) for f in sample_fields}, ensure_ascii=False,
                                     default=str, separators=(",", ":") if r.random() < .5 else (", ", ": "))
        else:
            input_panel = ""
        recent: list[str] = []
        if r.random() < cfg.get("recent_commands_probability", .35) and not self.json:
            helper = SnippetGenerator(random.Random(r.random()), self.domain, "console" if self.dialect == "json" else self.dialect, "crud")
            helper.name_style, helper.coll_style = self.name_style, self.coll_style
            recent = [helper.simple_statement().code[:256] for _ in range(r.randint(1, 3))]
        return {
            "language": LANGUAGE_AGGREGATION if self.json else LANGUAGE_CONSOLE,
            "known_names": known[:128],
            "result_fields": fields,
            "input_panel": input_panel,
            "recent_commands": recent,
            "newline": "\r\n" if r.random() < cfg.get("crlf_probability", .7) else "\n",
        }


def generate_snippet(rng: random.Random, domain: str, dialect: str, category: str, context_cfg: dict | None = None) -> dict:
    return SnippetGenerator(rng, domain, dialect, category, context_cfg).build()
