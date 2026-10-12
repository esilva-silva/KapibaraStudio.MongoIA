"""Extensões do gerador v1 para o pipeline compacto.

- Taxonomia de categorias do pedido: crud, aggregation, operators, atlas_search, json_ejson, indexes, slop_dsl, console_scripts.
- Cabeçalho de contexto com a linguagem atual da IDE por modo (Console, Script = "JavaScript (mongosh)", Agregação = json).
- JSON/Extended JSON: EJSON.parse, construtores BSON, $type/$convert e pipelines JSON com literais canônicos.
- Atlas Search ampliado (wildcard, regex, exists, in, near, count/sort, $vectorSearch) e operadores de expressão/array/update.
- DSL do Slop Studio: navegação por proxy, comandos gerados pela IDE (ConsoleScripts.Find/FromQuery), múltiplas conexões.
Somente APIs existentes em ConsoleBootstrap.js / docs/20-console.md (ConnectionPool, getConnection/getDatabase/getCollection).
"""
from __future__ import annotations

import random
import re
from typing import Any

from . import schemas as S
from .codegen import SnippetGenerator, Stmt, cap, pick, singular
from .contracts import DIALECT_LANGUAGE, stj_string
from .jsrender import IDENT, DateV, DecimalV, LongV, Money, Oid, Raw, Style, UuidV, number, render, render_inline

CATEGORIES = ("crud", "aggregation", "operators", "atlas_search", "json_ejson", "indexes", "slop_dsl", "console_scripts")
JSON_CATEGORIES = ("aggregation", "operators", "atlas_search", "json_ejson")
V1_CATEGORY = {"crud": "crud", "aggregation": "aggregation", "complex_queries": "aggregation", "filters": "operators",
               "atlas_search": "atlas_search", "indexes": "indexes", "mongosh_js": "console_scripts", "dsl_ide": "slop_dsl"}
TYPE_NAMES = {"date": ["date"], "money": ["decimal", "double"], "float": ["double"], "int": ["int", "long"], "ref": ["objectId"],
              "oid": ["objectId"], "bool": ["bool"], "tags": ["array"], "uuid": ["binData"]}


class CompactSnippetGenerator(SnippetGenerator):
    # -- montagem -------------------------------------------------------------------------------------------------------------
    def build(self) -> dict:
        r = self.rng
        if self.json and self.category not in JSON_CATEGORIES:
            self.category = "aggregation"
        before = [self.simple_statement() for _ in range(pick(r, {0: .75, 1: .2, 2: .05}))] if not self.json else []
        retries = 0
        while True:
            try:
                main = getattr(self, "cat_" + self.category)()
                break
            except (IndexError, ValueError, KeyError):
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
        ctx = super().context()
        if ctx:
            ctx["language"] = DIALECT_LANGUAGE[self.dialect]
        return ctx

    def inline(self, value: Any) -> str:
        return render_inline(value, self.style)

    # -- valores Extended JSON ---------------------------------------------------------------------------------------------------
    def ejson_value(self, f: S.Field, allow_uuid: bool = True) -> Any:
        r = self.rng
        value = self.value(f)
        if f.type == "money":
            return DecimalV(number(Money(round(float(value), 2)))) if r.random() < .6 else value
        if f.type == "int" and r.random() < .3:
            return LongV(str(value))
        if isinstance(value, UuidV) and not allow_uuid:
            return value.value
        return value

    def ejsonify(self, value: Any, allow_uuid: bool) -> Any:
        if isinstance(value, dict):
            return {k: self.ejsonify(v, allow_uuid) for k, v in value.items()}
        if isinstance(value, list):
            return [self.ejsonify(v, allow_uuid) for v in value]
        if isinstance(value, Money) and self.rng.random() < .6:
            return DecimalV(number(value))
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value > 1000 and self.rng.random() < .3:
            return LongV(str(value))
        if isinstance(value, UuidV) and not allow_uuid:
            return value.value
        return value

    def ejson_text(self, value: Any) -> str:
        text = render_inline(value, Style(json=True, brace_space=self.rng.random() < .4))
        if "'" in text or "\\" in text:
            raise ValueError("texto EJSON exige escape em aspas simples")
        return text

    # -- categorias novas ----------------------------------------------------------------------------------------------------------
    def cat_aggregation(self) -> list[Stmt]:
        return self.cat_complex_queries() if self.rng.random() < .3 else super().cat_aggregation()

    def cat_console_scripts(self) -> list[Stmt]:
        if self.dialect == "mongosh" and self.rng.random() < .12:
            return self.slop_script()
        return self.cat_mongosh_js()

    def slop_script(self) -> list[Stmt]:
        """Modo Script da IDE: entrada JSON em slop.input e resultados estruturados por slop.results (docs/14-guia-de-uso.md)."""
        r = self.rng
        semi = self.semi()
        name = self.cname(self.coll_key)
        target = "".join(self.target_segments(allow_bind=False, force="get"))
        kind = pick(r, {"stream_query": .45, "stream_params": .35, "emit_one": .2})
        self.subcategory = "slop_script_" + kind
        if kind == "stream_query":
            code = (f"const filtro = slop.input.query{semi}\n"
                    f"const limite = slop.input.parameters.limite ?? {r.choice([50, 100, 500])}{semi}\n"
                    f"const cursor = {target}.find(filtro).limit(limite){semi}\n"
                    f"await slop.results.stream(cursor){semi}")
            return [Stmt(code, f"Consulta {name} com a query do painel de entrada", f"Query {name} with the input panel query")]
        fields = [f for f in self.plain_fields("enum", "code", "city", "bool") if IDENT.match(f.path.split(".")[-1])]
        f = r.choice(fields)
        param = f.path.split(".")[-1]
        if kind == "stream_params":
            code = (f"const {param} = slop.input.parameters.{param}{semi}\n"
                    f"const cursor = {target}.find({{ {self.q(self.fpath(f)) if '.' in self.fpath(f) else self.fpath(f)}: {param} }})"
                    f".sort({self.inline(self.sort_spec())}).limit({r.choice([20, 50, 100])}){semi}\n"
                    f"await slop.results.stream(cursor){semi}")
            return [Stmt(code, f"Filtra {name} por {self.fpath(f)} vindo dos parâmetros", f"Filter {name} by {self.fpath(f)} from the parameters")]
        code = (f"const doc = {target}.findOne({{ {self.q(self.fpath(f)) if '.' in self.fpath(f) else self.fpath(f)}: slop.input.parameters.{param} }}){semi}\n"
                f"slop.results.emit(doc){semi}")
        return [Stmt(code, f"Emite um documento de {name} como resultado estruturado", f"Emit a {name} document as a structured result")]

    def cat_json_ejson(self) -> list[Stmt]:
        r = self.rng
        name = self.cname(self.coll_key)
        semi = self.semi()
        if self.json:
            return self.json_ejson_pipeline()
        kind = pick(r, {"parse_filter": .22, "parse_insert": .14, "bson_ctors": .22, "type_query": .16, "convert": .16, "stringify": .10})
        self.subcategory = "ejson_" + kind
        if kind == "parse_filter":
            fields = [f for f in self.plain_fields() if f.type in ("date", "ref", "money", "int", "enum", "bool", "code")]
            body: dict[str, Any] = {}
            for f in r.sample(fields, min(len(fields), r.randint(1, 2))):
                value = self.ejson_value(f, allow_uuid=False)
                if f.type == "date":
                    body[self.fpath(f)] = {r.choice(["$gte", "$lt"]): value}
                elif f.type in ("money", "int"):
                    body[self.fpath(f)] = {r.choice(["$gte", "$lte"]): value}
                else:
                    body[self.fpath(f)] = value
            var = r.choice(["filter", "query", "criteria"])
            calls = [("find", [Raw(var)])] + ([("limit", [r.choice([10, 20, 50])])] if r.random() < .7 else [])
            code = f"const {var} = EJSON.parse('{self.ejson_text(body)}'){semi}\n" + self.chain(self.target_segments(allow_bind=False), calls) + semi
            return [Stmt(code, f"Filtra {name} com Extended JSON em {', '.join(body)}", f"Filter {name} with Extended JSON on {', '.join(body)}")]
        if kind == "parse_insert":
            doc = self.ejsonify(self.document(), allow_uuid=False)
            call = Raw(f"EJSON.parse('{self.ejson_text(doc)}')")
            return [self.statement(self.target_segments(), [("insertOne", [call])], f"Insere em {name} a partir de Extended JSON",
                                   f"Insert into {name} from Extended JSON")]
        if kind == "bson_ctors":
            op = pick(r, {"insertOne": .45, "updateOne": .3, "find": .25})
            if op == "insertOne":
                doc = self.ejsonify(self.document(), allow_uuid=True)
                return [self.statement(self.target_segments(), [("insertOne", [doc])], f"Insere em {name} com tipos BSON explícitos",
                                       f"Insert into {name} with explicit BSON types")]
            numeric = self.plain_fields("money", "int")
            dates = self.plain_fields("date")
            if op == "updateOne":
                body = {}
                for f in (r.sample(numeric, 1) if numeric else []) + (r.sample(dates, 1) if dates else []):
                    body[self.fpath(f)] = self.ejson_value(f)
                if not body:
                    raise IndexError
                return [self.statement(self.target_segments(), [("updateOne", [{"_id": S.oid(r)}, {"$set": body}])],
                                       f"Atualiza {name} por ObjectId com tipos BSON", f"Update {name} by ObjectId with BSON types")]
            f = r.choice(numeric or dates)
            value = self.ejson_value(f)
            return [self.statement(self.target_segments(), [("find", [{self.fpath(f): {"$gte": value}}]), ("limit", [r.choice([10, 20])])],
                                   f"Busca {name} comparando {self.fpath(f)} com tipo BSON", f"Find {name} comparing {self.fpath(f)} with a BSON type")]
        if kind == "type_query":
            candidates = [f for f in self.plain_fields() if f.type in TYPE_NAMES or f.type in ("text", "name", "enum", "code")]
            f = r.choice(candidates)
            names = TYPE_NAMES.get(f.type, ["string"])
            type_value: Any = r.choice(names) if r.random() < .8 or len(names) < 2 else names
            condition: Any = {"$type": type_value}
            negated = r.random() < .25
            if negated:
                condition = {"$not": condition}
            op = r.choice(["find", "countDocuments"])
            calls = [(op, [{self.fpath(f): condition}])] + ([("limit", [20])] if op == "find" else [])
            label = type_value if isinstance(type_value, str) else "/".join(type_value)
            return [self.statement(self.target_segments(), calls, f"{op} em {name} onde {self.fpath(f)} {'não ' if negated else ''}é {label}",
                                   f"{op} on {name} where {self.fpath(f)} is {'not ' if negated else ''}{label}")]
        if kind == "convert":
            fields = [f for f in self.plain_fields() if f.type in ("money", "int", "float", "date", "code", "ref")]
            f = r.choice(fields)
            path = "$" + self.fpath(f)
            leaf = cap(f.path.split(".")[-1])
            options = {
                "money": [{"$toDecimal": path}, {"$convert": {"input": path, "to": "decimal", "onError": None, "onNull": None}}],
                "int": [{"$toLong": path}, {"$toString": path}], "float": [{"$toDouble": path}, {"$round": [path, 2]}],
                "date": [{"$toString": path}, {"$toLong": path}], "code": [{"$toString": path}, {"$toInt": path}],
                "ref": [{"$toString": path}, {"$toObjectId": {"$toString": path}}],
            }[f.type]
            expr = r.choice(options)
            stages: list[dict] = []
            if r.random() < .4:
                flt, _, _ = self.filter(n=1)
                stages.append({"$match": flt})
            stages.append({r.choice(["$set", "$addFields"]): {self.out("converted" + leaf): expr}})
            stages.append({"$limit": r.choice([10, 20, 50])})
            return [self.aggregate_statement(stages, f"converte {self.fpath(f)} com {next(iter(expr))}", f"convert {self.fpath(f)} with {next(iter(expr))}")]
        flt, pt, en = self.filter(n=1)
        target = "".join(self.target_segments(allow_bind=False))
        code = (f"const doc = {target}.findOne({self.inline(flt)}){semi}\n"
                + (f"console.log(EJSON.stringify(doc)){semi}" if self.dialect == "console" else f"print(EJSON.stringify(doc)){semi}"))
        return [Stmt(code, f"Mostra em Extended JSON um documento de {name} onde {pt}", f"Print a {name} document where {en} as Extended JSON")]

    def json_ejson_pipeline(self) -> list[Stmt]:
        r = self.rng
        kind = pick(r, {"match_ejson": .5, "type": .2, "convert": .3})
        self.subcategory = "ejson_pipeline_" + kind
        if kind == "convert":
            return self.cat_json_ejson_convert_json()
        fields = [f for f in self.plain_fields() if f.type in ("date", "ref", "money", "int", "uuid", "enum")]
        if kind == "type":
            f = r.choice([x for x in self.plain_fields() if x.type in TYPE_NAMES] or self.plain_fields())
            match: dict[str, Any] = {self.fpath(f): {"$type": r.choice(TYPE_NAMES.get(f.type, ["string"]))}}
        else:
            match = {}
            for f in r.sample(fields, min(len(fields), r.randint(1, 3))):
                value = self.ejson_value(f)
                match[self.fpath(f)] = {r.choice(["$gte", "$lt"]): value} if f.type in ("date", "money", "int") else value
        stages: list[dict] = [{"$match": match}]
        if r.random() < .5:
            stages.append({"$project": self.projection()})
        if r.random() < .5:
            stages.append({"$sort": self.sort_spec()})
        stages.append({"$limit": r.choice([10, 20, 50, 100])})
        return [Stmt(render(stages, self.style), "Pipeline com Extended JSON em " + ", ".join(match), "Extended JSON pipeline on " + ", ".join(match))]

    def cat_json_ejson_convert_json(self) -> list[Stmt]:
        r = self.rng
        f = r.choice([x for x in self.plain_fields() if x.type in ("money", "int", "date", "code")])
        path = "$" + self.fpath(f)
        expr = r.choice([{"$toDecimal": path}, {"$toString": path}, {"$convert": {"input": path, "to": "string", "onError": None, "onNull": None}}])
        stages = [{"$set": {self.out("as" + cap(f.path.split(".")[-1])): expr}}, {"$limit": r.choice([10, 20])}]
        return [Stmt(render(stages, self.style), f"Pipeline JSON convertendo {self.fpath(f)}", f"JSON pipeline converting {self.fpath(f)}")]

    def expression_options(self) -> list[tuple[dict, str, str]]:
        r = self.rng
        numeric = self.plain_fields("int", "money", "float")
        dates = self.plain_fields("date")
        strings = self.plain_fields("enum", "text", "name", "code", "city", "word")
        enums = self.plain_fields("enum")
        bools = self.plain_fields("bool")
        out: list[tuple[dict, str, str]] = []
        if numeric:
            a = r.choice(numeric)
            pa = self.fpath(a)
            hi = a.range()[1]
            low, high = max(1, int(hi * .3)), max(2, int(hi * .7))
            leaf = cap(a.path.split(".")[-1])
            out.append(({self.out("band"): {"$switch": {"branches": [{"case": {"$lt": ["$" + pa, low]}, "then": "low"},
                                                                    {"case": {"$lt": ["$" + pa, high]}, "then": "medium"}], "default": "high"}}},
                        f"classifica {pa} em faixas com $switch", f"bucket {pa} with $switch"))
            out.append(({self.out("tier"): {"$cond": {"if": {"$gte": ["$" + pa, high]}, "then": "high", "else": "normal"}}},
                        f"classifica {pa} com $cond", f"classify {pa} with $cond"))
            out.append(({self.out("rounded" + leaf): {"$round": [{"$multiply": ["$" + pa, 1.1]}, 2]}}, f"arredonda {pa} ajustado", f"round adjusted {pa}"))
            out.append(({self.out("safe" + leaf): {"$ifNull": ["$" + pa, 0]}}, f"trata {pa} nulo", f"default null {pa}"))
            if len(numeric) >= 2:
                b = r.choice([x for x in numeric if x is not a])
                pb = self.fpath(b)
                out.append(({self.out("ratio"): {"$cond": [{"$eq": ["$" + pb, 0]}, None, {"$divide": ["$" + pa, "$" + pb]}]}},
                            f"razão entre {pa} e {pb}", f"ratio between {pa} and {pb}"))
        if dates:
            d = self.fpath(r.choice(dates))
            out.append(({self.out("month"): {"$dateTrunc": {"date": "$" + d, "unit": "month"}}}, f"trunca {d} por mês", f"truncate {d} to month"))
            out.append(({self.out("ageDays"): {"$dateDiff": {"startDate": "$" + d, "endDate": "$$NOW", "unit": "day"}}}, f"dias desde {d}", f"days since {d}"))
            out.append(({self.out("day"): {"$dateToString": {"format": "%Y-%m-%d", "date": "$" + d}}}, f"formata {d}", f"format {d}"))
            out.append(({self.out("weekday"): {"$dayOfWeek": "$" + d}}, f"dia da semana de {d}", f"weekday of {d}"))
        if strings:
            s = r.choice(strings)
            ps = self.fpath(s)
            prefix = re.sub(r"[^A-Za-z0-9]", "", str(self.value(s)))[:3] or "a"
            out.append(({self.out("matches"): {"$regexMatch": {"input": "$" + ps, "regex": "^" + prefix, "options": "i"}}},
                        f"testa regex em {ps}", f"regex test on {ps}"))
            out.append(({self.out("upper" + cap(s.path.split(".")[-1])): {"$toUpper": "$" + ps}}, f"maiúsculas de {ps}", f"uppercase {ps}"))
            if len(strings) >= 2:
                t = r.choice([x for x in strings if x is not s])
                out.append(({self.out("label"): {"$concat": ["$" + ps, " - ", {"$toString": "$" + self.fpath(t)}]}},
                            f"concatena {ps} e {self.fpath(t)}", f"concatenate {ps} and {self.fpath(t)}"))
        if enums:
            e = r.choice(enums)
            values = self.distinct_values(e, 2)
            out.append(({self.out("isSelected"): {"$in": ["$" + self.fpath(e), values]}}, f"{self.fpath(e)} em {values}", f"{self.fpath(e)} in {values}"))
        if bools:
            b = self.fpath(r.choice(bools))
            out.append(({self.out("flag"): {"$cond": ["$" + b, 1, 0]}}, f"converte {b} em número", f"convert {b} to a number"))
        return out

    def cat_operators(self) -> list[Stmt]:
        r = self.rng
        name = self.cname(self.coll_key)
        kind = pick(r, {"query": .45, "expr": .28, "update": 0 if self.json else .15, "array": .12})
        self.subcategory = "operators_" + kind
        if kind == "query":
            if not self.json:
                return self.cat_filters()
            flt, pt, en = self.filter(n=pick(r, {2: .5, 3: .4, 4: .1}), logical=.45)
            stages = [{"$match": flt}] + ([{"$sort": self.sort_spec()}] if r.random() < .4 else []) + [{"$limit": r.choice([10, 20, 50])}]
            return [Stmt(render(stages, self.style), "Pipeline: filtra " + pt, "Pipeline: match " + en)]
        if kind == "expr":
            options = self.expression_options()
            if not options:
                raise IndexError
            chosen = r.sample(options, min(len(options), pick(r, {1: .5, 2: .35, 3: .15})))
            stage_name = r.choice(["$project", "$project", "$set", "$addFields"])
            body: dict[str, Any] = {}
            if stage_name == "$project" and r.random() < .6:
                for f in r.sample(self.plain_fields(), min(2, len(self.plain_fields()))):
                    body[self.fpath(f)] = 1
            for expr, _, _ in chosen:
                body.update(expr)
            stages: list[dict] = []
            if r.random() < .5:
                stages.append({"$match": self.filter(n=1)[0]})
            stages.append({stage_name: body})
            if r.random() < .3:
                stages.append({"$limit": r.choice([10, 20, 50])})
            return [self.aggregate_statement(stages, "; ".join(c[1] for c in chosen), "; ".join(c[2] for c in chosen))]
        if kind == "update":
            flt, fpt, fen = self.filter(n=pick(r, {1: .6, 2: .4}), logical=.3)
            update, upt, uen, options = self.update_extended()
            op = r.choice(["updateMany", "updateOne"])
            args = [flt, update] + ([options] if options else [])
            return [self.statement(self.target_segments(), [(op, args)], f"Atualiza {name} onde {fpt}: {upt}", f"Update {name} where {fen}: {uen}")]
        return self.array_operators()

    def update_extended(self) -> tuple[dict, str, str, dict | None]:
        r = self.rng
        update, pt, en, options = self.update()
        used = {path.split(".$[")[0] for body in update.values() if isinstance(body, dict) for path in body}
        numeric = [f for f in self.plain_fields("int", "money", "float") if self.fpath(f) not in used]
        tags = [f for f in self.plain_fields("tags") if self.fpath(f) not in used]
        plain = [f for f in self.plain_fields() if f.type not in ("geo", "tags") and self.fpath(f) not in used]
        extra = pick(r, {"none": .45, "$mul": .12 if numeric else 0, "$rename": .1 if plain else 0, "$setOnInsert": .12 if plain else 0,
                         "$min": .08 if numeric else 0, "$pop": .06 if tags else 0, "$pullAll": .07 if tags else 0})
        if extra == "$mul":
            f = r.choice(numeric)
            update["$mul"] = {self.fpath(f): r.choice([1.1, 0.9, 2])}
            pt, en = pt + f"; multiplica {self.fpath(f)}", en + f"; multiply {self.fpath(f)}"
        elif extra == "$rename":
            f = r.choice(plain)
            update["$rename"] = {self.fpath(f): self.fpath(f) + r.choice(["Old", "Legacy", "_v1"])}
            pt, en = pt + f"; renomeia {self.fpath(f)}", en + f"; rename {self.fpath(f)}"
        elif extra == "$setOnInsert":
            f = r.choice(plain)
            update["$setOnInsert"] = {self.fpath(f): self.value(f)}
            options = {**(options or {}), "upsert": True}
            pt, en = pt + "; upsert com $setOnInsert", en + "; upsert with $setOnInsert"
        elif extra == "$min":
            f = r.choice(numeric)
            update["$min"] = {self.fpath(f): self.value(f)}
            pt, en = pt + f"; mantém o menor {self.fpath(f)}", en + f"; keep the min {self.fpath(f)}"
        elif extra == "$pop":
            f = r.choice(tags)
            update["$pop"] = {self.fpath(f): r.choice([1, -1])}
            pt, en = pt + f"; remove extremo de {self.fpath(f)}", en + f"; pop from {self.fpath(f)}"
        elif extra == "$pullAll":
            f = r.choice(tags)
            update["$pullAll"] = {self.fpath(f): self.distinct_values(f, 2)}
            pt, en = pt + f"; remove valores de {self.fpath(f)}", en + f"; pull all from {self.fpath(f)}"
        return update, pt, en, options

    def array_operators(self) -> list[Stmt]:
        r = self.rng
        name = self.cname(self.coll_key)
        arrays = sorted({f.array_parent for f in self.coll.fields if f.array_parent})
        tags = self.plain_fields("tags")
        options: list[tuple[dict, str, str]] = []
        match: dict[str, Any] | None = None
        if arrays:
            parent = r.choice(arrays)
            path = self.ppath(parent)
            subs = [f for f in self.coll.fields if f.array_parent == parent]
            numeric = [f for f in subs if f.numeric]
            labels = [f for f in subs if f.type in ("code", "enum", "text", "ref", "name")]
            if numeric:
                s = r.choice(numeric)
                leaf = S.style_segment(s.path.split(".")[-1], self.name_style)
                options.append(({self.out("filtered" + cap(parent)): {"$filter": {"input": "$" + path, "as": "item",
                                                                                  "cond": {"$gte": ["$$item." + leaf, self.value(s)]}}}},
                                f"filtra {path} com $filter", f"filter {path} with $filter"))
                options.append(({self.out(s.path.split(".")[-1] + "Total"): {"$reduce": {"input": "$" + path, "initialValue": 0,
                                                                                         "in": {"$add": ["$$value", "$$this." + leaf]}}}},
                                f"soma {leaf} com $reduce", f"sum {leaf} with $reduce"))
            if labels:
                c = r.choice(labels)
                leaf = S.style_segment(c.path.split(".")[-1], self.name_style)
                options.append(({self.out(singular(c.path.split(".")[-1]) + "List"): {"$map": {"input": "$" + path, "as": "item", "in": "$$item." + leaf}}},
                                f"lista {leaf} com $map", f"list {leaf} with $map"))
                match = {path: {"$elemMatch": {leaf: self.value(c)}}}
            options.append(({self.out("first" + cap(singular(parent))): {"$arrayElemAt": ["$" + path, 0]}}, f"primeiro de {path}", f"first of {path}"))
            options.append(({self.out(singular(parent) + "Count"): {"$size": {"$ifNull": ["$" + path, []]}}}, f"conta {path}", f"count {path}"))
        if tags:
            t = r.choice(tags)
            tpath = self.fpath(t)
            values = self.distinct_values(t, 2)
            options.append(({self.out("hasTag"): {"$in": [values[0], {"$ifNull": ["$" + tpath, []]}]}}, f"verifica {values[0]} em {tpath}", f"check {values[0]} in {tpath}"))
            options.append(({self.out("sharedTags"): {"$setIntersection": [{"$ifNull": ["$" + tpath, []]}, values]}},
                            f"interseção de {tpath}", f"intersection of {tpath}"))
            if match is None or r.random() < .5:
                match = {tpath: r.choice([{"$all": values}, {"$in": values}, {"$size": r.randint(1, 3)}])}
        if not options:
            raise IndexError("coleção sem arrays")
        if not self.json and match is not None and r.random() < .35:
            calls = [("find", [match])] + ([("limit", [r.choice([10, 20])])] if r.random() < .7 else [])
            return [self.statement(self.target_segments(), calls, f"Busca {name} por condição em array", f"Find {name} by array condition")]
        chosen = r.sample(options, min(len(options), r.randint(1, 2)))
        body: dict[str, Any] = {}
        for expr, _, _ in chosen:
            body.update(expr)
        stages: list[dict] = ([{"$match": match}] if match is not None and r.random() < .5 else []) + [{"$project": body}]
        if r.random() < .3:
            stages.append({"$limit": r.choice([10, 20])})
        return [self.aggregate_statement(stages, "; ".join(c[1] for c in chosen), "; ".join(c[2] for c in chosen))]

    # -- Atlas Search ampliado ----------------------------------------------------------------------------------------------------------
    def cat_atlas_search(self) -> list[Stmt]:
        if self.rng.random() < .35:
            stages, pt, en = self.search_extra()
            first = stages[0]
            stage = next(iter(first))
            self.subcategory = "vector_search" if stage == "$vectorSearch" else "search_" + next(k for k in first[stage] if k not in ("index", "count", "sort", "returnStoredSource"))
            return [self.aggregate_statement(stages, pt, en)]
        return super().cat_atlas_search()

    def search_extra(self) -> tuple[list[dict], str, str]:
        r = self.rng
        texts = self.plain_fields("text", "name") or self.plain_fields("enum", "word", "code", "city")
        numeric_dates = self.plain_fields("int", "money", "float", "date")
        enums = self.plain_fields("enum")
        index = r.choice(S.DOMAINS[self.domain]["search_index"])
        kind = pick(r, {"wildcard": .18, "regex": .1, "exists": .08, "in": .14 if enums else 0, "near": .14 if numeric_dates else 0,
                        "vector": .22, "count_sort": .14})
        f = r.choice(texts)
        path = self.fpath(f)
        term = self.text_term(f)
        if kind == "vector":
            dims = [round(r.uniform(-1, 1), 3) for _ in range(r.choice([4, 6, 8]))]
            body: dict[str, Any] = {"index": r.choice(["vector_index", "embeddings_index"]), "path": self.out("embedding"),
                                    "queryVector": dims if self.json else Raw("queryVector"), "numCandidates": r.choice([50, 100, 200]),
                                    "limit": r.choice([5, 10, 20])}
            if enums and r.random() < .4:
                e = r.choice(enums)
                body["filter"] = {self.fpath(e): self.value(e)}
            project = {self.fpath(x): 1 for x in r.sample(self.plain_fields(), min(2, len(self.plain_fields())))}
            project[self.out("score")] = {"$meta": "vectorSearchScore"}
            if not self.json:
                self.prelude.append(Stmt(f"const queryVector = {self.inline(dims)}{self.semi()}", "Vetor de consulta", "Query vector", main=False))
            return [{"$vectorSearch": body}, {"$project": project}], "busca vetorial com $vectorSearch", "vector search with $vectorSearch"
        search: dict[str, Any] = {"index": index} if r.random() < .85 else {}
        if kind == "wildcard":
            search["wildcard"] = {"query": term[:3] + "*", "path": path, "allowAnalyzedField": True}
            pt, en = f"wildcard em {path}", f"wildcard on {path}"
        elif kind == "regex":
            search["regex"] = {"query": term[:3] + ".*", "path": path, "allowAnalyzedField": True}
            pt, en = f"regex Atlas Search em {path}", f"Atlas Search regex on {path}"
        elif kind == "exists":
            search["exists"] = {"path": path}
            pt, en = f"documentos com {path}", f"documents with {path}"
        elif kind == "in":
            e = r.choice(enums)
            search["in"] = {"path": self.fpath(e), "value": self.distinct_values(e, 2)}
            pt, en = f"in em {self.fpath(e)}", f"in on {self.fpath(e)}"
        elif kind == "near":
            nf = r.choice(numeric_dates)
            origin = self.value(nf)
            pivot = 7 * 24 * 60 * 60 * 1000 if nf.type == "date" else max(1, int(nf.range()[1] / 10))
            search["near"] = {"path": self.fpath(nf), "origin": origin, "pivot": pivot}
            pt, en = f"proximidade em {self.fpath(nf)}", f"near on {self.fpath(nf)}"
        else:
            search["text"] = {"query": term, "path": path}
            if r.random() < .5:
                search["count"] = {"type": r.choice(["total", "lowerBound"])}
            if numeric_dates and r.random() < .6:
                search["sort"] = {self.fpath(r.choice(numeric_dates)): -1}
            if r.random() < .2:
                search["returnStoredSource"] = True
            pt, en = f"busca textual ordenada em {path}", f"sorted text search on {path}"
        stages: list[dict] = [{"$search": search}]
        if r.random() < .8:
            stages.append({"$limit": r.choice([5, 10, 20])})
        if r.random() < .6:
            project = {self.fpath(x): 1 for x in r.sample(self.plain_fields(), min(2, len(self.plain_fields())))}
            project[self.out("score")] = {"$meta": "searchScore"}
            stages.append({"$project": project})
        return stages, pt, en

    # -- DSL do Slop Studio ---------------------------------------------------------------------------------------------------------------
    def cat_slop_dsl(self) -> list[Stmt]:
        r = self.rng
        if self.dialect != "console":
            return self.cat_dsl_ide()
        if r.random() < .55:
            return self.cat_dsl_ide()
        kind = pick(r, {"proxy_nav": .22, "ide_generated": .24, "default_open": .04, "multi_result": .12, "cross_copy": .12,
                        "sibling_prop": .12, "console_levels": .14})
        self.subcategory = "dsl_" + kind
        name = self.cname(self.coll_key)
        q = self.q
        semi = self.semi()
        ind = " " * self.style.indent
        if kind == "proxy_nav":
            head = f"getConnection({q(self.connection)})"
            segments = ([head, "." + self.database, "." + name] if IDENT.match(self.database) and IDENT.match(name)
                        else [head, f"[{q(self.database)}]", f"[{q(name)}]"])
            flt, pt, en = self.filter()
            return [self.statement(segments, [("find", [flt])] + self.cursor_calls(), f"Consulta {name} navegando pela conexão {self.connection}",
                                   f"Query {name} navigating connection {self.connection}")]
        if kind in ("ide_generated", "default_open"):
            if kind == "default_open":
                return [Stmt("db.getCollection(" + stj_string(name) + ").find({}).limit(100)", f"Abre {name} pelo Explorer", f"Open {name} from the Explorer")]
            js = Style(json=True, brace_space=r.random() < .5)
            flt = {} if r.random() < .15 else self.filter()[0]
            script = "db.getCollection(" + stj_string(name) + ").find(" + render_inline(flt, js) + ")"
            if r.random() < .3:
                script += ".project(" + render_inline(self.projection(), js) + ")"
            if r.random() < .5:
                script += ".sort(" + render_inline(self.sort_spec(), js) + ")"
            if r.random() < .08:
                script += ".hint(" + render_inline({self.fpath(r.choice(self.plain_fields())): 1}, js) + ")"
            if r.random() < .06:
                script += ".collation(" + render_inline({"locale": "pt", "strength": 2}, js) + ")"
            if r.random() < .06:
                script += ".comment(" + stj_string(r.choice(["relatorio", "auditoria", "painel diario"])) + ")"
            if r.random() < .05:
                script += f".batchSize({r.choice([100, 500])})"
            if r.random() < .05:
                script += f".maxTimeMS({r.choice([5000, 30000])})"
            script += f".skip({r.choice([0, 0, 0, 20, 100])}).limit({r.choice([20, 50, 100, 100])})"
            return [Stmt(script, f"Consulta gerada pela IDE para {name}", f"IDE-generated query for {name}")]
        if kind == "multi_result":
            flt, pt, en = self.filter(n=1)
            target = "".join(self.target_segments(allow_bind=False))
            lines = [f"{target}.countDocuments({self.inline(flt)}){semi}", f"{target}.find({self.inline(flt)}).limit({r.choice([5, 10])}){semi}"]
            labels = self.plain_fields("enum", "city", "code") or self.plain_fields()
            if r.random() < .4:
                lines.append(f"{target}.distinct({q(self.fpath(r.choice(labels)))}){semi}")
            return [Stmt("\n".join(lines), f"Conta e lista {name} onde {pt}", f"Count and list {name} where {en}")]
        if kind == "cross_copy":
            other = r.choice([c for c in S.CONNECTION_NAMES if c != self.connection])
            flt, pt, en = self.filter(n=1)
            code = (f"const docs = getConnection({q(self.connection)}).getDatabase({q(self.database)}).getCollection({q(name)})"
                    f".find({self.inline(flt)}).limit({r.choice([10, 50, 100])}).toArray(){semi}\n"
                    f"getConnection({q(other)}).getDatabase({q(self.database)}).getCollection({q(name)}).insertMany(docs){semi}")
            return [Stmt(code, f"Copia {name} de {self.connection} para {other}", f"Copy {name} from {self.connection} to {other}")]
        if kind == "sibling_prop":
            other_db = r.choice(S.DOMAINS[self.domain]["databases"])
            segments = ["db", f".getSiblingDB({q(other_db)})", "." + name if IDENT.match(name) else f".getCollection({q(name)})"]
            flt, pt, en = self.filter(n=1)
            calls = [("find", [flt])] + self.cursor_calls() if r.random() < .7 else [("countDocuments", [flt])]
            return [self.statement(segments, calls, f"Consulta {name} no banco {other_db}", f"Query {name} in database {other_db}")]
        flt, pt, en = self.filter(n=1)
        target = "".join(self.target_segments(allow_bind=False))
        limit = r.choice([0, 10, 100])
        code = (f"const total = {target}.countDocuments({self.inline(flt)}){semi}\n"
                f"if (total > {limit}) {{\n{ind}console.warn({q(name + ' acima do limite')}, total){semi}\n}} else {{\n{ind}console.log(total){semi}\n}}")
        return [Stmt(code, f"Avisa quando {name} onde {pt} passar de {limit}", f"Warn when {name} where {en} exceeds {limit}")]


def generate_snippet(rng: random.Random, domain: str, dialect: str, category: str, context_cfg: dict | None = None) -> dict:
    return CompactSnippetGenerator(rng, domain, dialect, category, context_cfg).build()
