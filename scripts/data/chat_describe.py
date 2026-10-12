#!/usr/bin/env python3
"""Plano estruturado (EJSON) -> pedido em linguagem natural (pt-BR / en) para o chat curto.

Usado por gen_chat.py. Sem rede, sem modelo: templates parametrizados pelo plano, com varias
parafrases por operacao, registro (informal / tecnico / neutro) e referencia ambigua a colecao.
Valores e nomes de campo aparecem LITERALMENTE no texto (protegidos por marcadores \\x01..\\x02
ate a pos-edicao: remocao de acento e erro de digitacao so tocam o texto de template).

Qualquer construcao nao coberta levanta `Unsupported` (o snippet e descartado do chat).
"""
from __future__ import annotations

import random
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_snippets import EJSON_KEYS, fmt_num  # noqa: E402

P0, P1 = "\x01", "\x02"          # marcadores de trecho protegido (valor/campo literal)
NOUN_RE = re.compile(r"^[A-Za-z0-9 ]*$")


class Unsupported(Exception):
    pass


def deaccent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def strip_marks(s: str) -> str:
    return s.replace(P0, "").replace(P1, "")


def map_unprotected(s: str, fn) -> str:
    """Aplica fn so fora de P0..P1."""
    out, i = [], 0
    while i < len(s):
        a = s.find(P0, i)
        if a < 0:
            out.append(fn(s[i:]))
            break
        out.append(fn(s[i:a]))
        b = s.find(P1, a)
        if b < 0:
            out.append(s[a:])
            break
        out.append(s[a:b + 1])
        i = b + 1
    return "".join(out)


def swap_typo(s: str, rng: random.Random) -> str:
    """Troca duas letras adjacentes numa palavra longa nao protegida."""
    def fn(seg):
        words = [m for m in re.finditer(r"[A-Za-z]{5,}", seg)]
        if not words:
            return seg
        m = rng.choice(words)
        w = m.group(0)
        i = rng.randrange(1, len(w) - 2)
        w2 = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        return seg[:m.start()] + w2 + seg[m.end():]
    parts = []
    # aplica em um unico segmento nao protegido (o primeiro que tiver palavra longa, em ordem aleatoria)
    segs = re.split(f"({P0}[^{P1}]*{P1})", s)
    idx = [k for k, p in enumerate(segs) if not p.startswith(P0) and re.search(r"[A-Za-z]{5,}", p)]
    if not idx:
        return s
    k = rng.choice(idx)
    segs[k] = fn(segs[k])
    return "".join(segs)


def F(path: str) -> str:
    return P0 + path + P1


def lit(v) -> str:
    """Texto literal (como aparece no codigo) de um valor EJSON."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return fmt_num(v)
    if isinstance(v, str):
        if "\n" in v or "\r" in v or "\\" in v or ('"' in v and "'" in v):
            raise Unsupported("string")
        return f"'{v}'" if '"' in v else f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(lit(x) for x in v) + "]"
    if isinstance(v, dict) and len(v) == 1:
        (k, x), = v.items()
        if k == "$date":
            return x[:10] if x.endswith("T00:00:00Z") else x
        if k == "$oid":
            return f'ObjectId("{x}")'
        if k == "$numberLong":
            return f'NumberLong("{x}")'
        if k == "$numberDecimal":
            return f'NumberDecimal("{x}")'
        if k == "$uuid":
            return f'UUID("{x}")'
    raise Unsupported("valor")


def V(v) -> str:
    return P0 + lit(v) + P1


def is_ej(v) -> bool:
    return isinstance(v, dict) and len(v) == 1 and next(iter(v)) in EJSON_KEYS


def is_date(v) -> bool:
    return isinstance(v, dict) and "$date" in v


def is_opdict(v) -> bool:
    return isinstance(v, dict) and bool(v) and not is_ej(v) and all(k.startswith("$") for k in v)


class Describer:
    def __init__(self, lang: str, rng: random.Random, reg: str = "n", ambiguous_coll: bool = False):
        assert lang in ("pt", "en")
        self.pt = lang == "pt"
        self.rng = rng
        self.reg = reg                      # n | i | t
        self.amb = ambiguous_coll

    # ---- utilidades ----
    def ch(self, pt, en=None):
        opts = pt if self.pt else en
        return self.rng.choice(opts)

    def and_(self):
        return " e " if self.pt else " and "

    def or_(self):
        return " ou " if self.pt else " or "

    def join(self, items: list[str]) -> str:
        items = [i for i in items if i]
        if len(items) <= 1:
            return "".join(items)
        return ", ".join(items[:-1]) + self.and_() + items[-1]

    def lst(self, items: list[str]) -> str:
        return ", ".join(items)

    def coll_in(self, name: str) -> str:
        if self.amb:
            return self.ch(["nesta coleção", "na coleção atual", "aqui", "nessa coleção"],
                           ["in this collection", "in the current collection", "here"])
        return ("em " if self.pt else "in ") + F(name)

    def coll_of(self, name: str) -> str:
        if self.amb:
            return self.ch(["desta coleção", "da coleção atual", "dessa coleção"],
                           ["from this collection", "from the current collection"])
        return ("de " if self.pt else "from ") + F(name)

    # ---- condicoes ----
    def cmp_text(self, f: str, op: str, v) -> str:
        d = is_date(v)
        fv = F(f)
        vv = V(v)
        pt = {
            "$eq": ["{f} igual a {v}", "{f} = {v}", "{f} é {v}"],
            "$ne": ["{f} diferente de {v}", "{f} != {v}", "{f} que não seja {v}"],
            "$gt": ["{f} maior que {v}", "{f} > {v}", "{f} acima de {v}"],
            "$gte": ["{f} maior ou igual a {v}", "{f} >= {v}", "{f} de {v} para cima"],
            "$lt": ["{f} menor que {v}", "{f} < {v}", "{f} abaixo de {v}"],
            "$lte": ["{f} menor ou igual a {v}", "{f} <= {v}", "{f} até {v} (inclusive)"],
        }
        pt_d = {
            "$gt": ["{f} depois de {v}", "{f} posterior a {v}", "{f} > {v}"],
            "$gte": ["{f} a partir de {v} (inclusive)", "{f} >= {v}", "{f} em {v} ou depois"],
            "$lt": ["{f} antes de {v}", "{f} anterior a {v}", "{f} < {v}"],
            "$lte": ["{f} até {v} (inclusive)", "{f} <= {v}", "{f} em {v} ou antes"],
        }
        en = {
            "$eq": ["{f} equals {v}", "{f} = {v}", "{f} is {v}"],
            "$ne": ["{f} not equal to {v}", "{f} != {v}", "{f} is not {v}"],
            "$gt": ["{f} greater than {v}", "{f} > {v}", "{f} above {v}"],
            "$gte": ["{f} greater than or equal to {v}", "{f} >= {v}", "{f} at least {v}"],
            "$lt": ["{f} less than {v}", "{f} < {v}", "{f} below {v}"],
            "$lte": ["{f} less than or equal to {v}", "{f} <= {v}", "{f} at most {v}"],
        }
        en_d = {
            "$gt": ["{f} after {v}", "{f} > {v}"],
            "$gte": ["{f} on or after {v}", "{f} >= {v}"],
            "$lt": ["{f} before {v}", "{f} < {v}"],
            "$lte": ["{f} up to and including {v}", "{f} <= {v}"],
        }
        if op not in pt:
            raise Unsupported(op)
        table = (pt_d if (self.pt and d and op in pt_d) else pt) if self.pt else (en_d if (d and op in en_d) else en)
        t = self.rng.choice(table[op])
        if self.reg == "t":
            sym = {"$eq": "=", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}[op]
            t = "{f} " + sym + " {v}"
        return t.format(f=fv, v=vv)

    def range_text(self, f: str, ops: dict) -> str | None:
        lo = "$gte" if "$gte" in ops else ("$gt" if "$gt" in ops else None)
        hi = "$lte" if "$lte" in ops else ("$lt" if "$lt" in ops else None)
        if not (lo and hi) or set(ops) != {lo, hi}:
            return None
        a, b = V(ops[lo]), V(ops[hi])
        fv = F(f)
        sym_lo = "<=" if lo == "$gte" else "<"
        sym_hi = "<=" if hi == "$lte" else "<"
        math = f"{a} {sym_lo} {fv} {sym_hi} {b}"
        if self.reg == "t" and self.rng.random() < 0.6:
            return math
        incl = "inclusive" if (lo, hi) == ("$gte", "$lte") else None
        if (lo, hi) == ("$gte", "$lte"):
            return self.rng.choice([math, (f"{fv} entre {a} e {b} (inclusive)" if self.pt else f"{fv} between {a} and {b} (inclusive)")])
        if (lo, hi) == ("$gte", "$lt"):
            return self.rng.choice([math, (f"{fv} de {a} (inclusive) até {b} (exclusivo)" if self.pt
                                           else f"{fv} from {a} (inclusive) to {b} (exclusive)")])
        if (lo, hi) == ("$gt", "$lt"):
            return self.rng.choice([math, (f"{fv} entre {a} e {b} (exclusivo)" if self.pt else f"{fv} between {a} and {b} (exclusive)")])
        return math

    def field_cond(self, f: str, spec, prefix: str = "") -> str:
        fp = prefix + f
        if is_opdict(spec):
            ops = dict(spec)
            parts = []
            r = self.range_text(fp, ops)
            if r:
                return r
            opts = ops.pop("$options", None)
            for op, arg in ops.items():
                parts.append(self.op_text(fp, op, arg, opts))
            if opts is not None and "$regex" not in spec:
                raise Unsupported("options")
            return self.join_and(parts)
        if isinstance(spec, dict) and not is_ej(spec):
            raise Unsupported("embedded doc")
        return self.eq_text(fp, spec)

    def join_and(self, parts):
        return parts[0] if len(parts) == 1 else self.and_().join(parts)

    def eq_text(self, f: str, v) -> str:
        if v is True or v is False:
            if self.reg != "t" and self.rng.random() < 0.6:
                return self.ch([f"{F(f)} é verdadeiro" if v else f"{F(f)} é falso", f"{F(f)} igual a {lit(v)}"],
                               [f"{F(f)} is true" if v else f"{F(f)} is false", f"{F(f)} equals {lit(v)}"])
        if v is None:
            return self.ch([f"{F(f)} nulo", f"{F(f)} é null"], [f"{F(f)} is null", f"{F(f)} is empty (null)"])
        return self.cmp_text(f, "$eq", v)

    def op_text(self, f: str, op: str, arg, opts=None) -> str:
        fv = F(f)
        if op in ("$eq", "$ne", "$gt", "$gte", "$lt", "$lte"):
            return self.cmp_text(f, op, arg)
        if op in ("$in", "$nin", "$all"):
            if not isinstance(arg, list) or not arg:
                raise Unsupported("in")
            L = P0 + "[" + ", ".join(lit(x) for x in arg) + "]" + P1
            if op == "$in":
                return self.ch([f"{fv} em {L}", f"{fv} seja um destes: {L}", f"{fv} dentre {L}"],
                               [f"{fv} in {L}", f"{fv} is one of {L}", f"{fv} among {L}"])
            if op == "$nin":
                return self.ch([f"{fv} fora de {L}", f"{fv} não esteja em {L}", f"{fv} diferente de todos de {L}"],
                               [f"{fv} not in {L}", f"{fv} is none of {L}"])
            return self.ch([f"{fv} contenha todos de {L}", f"{fv} com todos os itens {L}"],
                           [f"{fv} contains all of {L}", f"{fv} has every item of {L}"])
        if op == "$exists":
            if getattr(self, "strict", False):
                if arg in (True, False):
                    return (f"{fv} existe" if arg else f"{fv} não existe") if self.pt else (f"{fv} exists" if arg else f"{fv} does not exist")
            if arg is True:
                return self.ch([f"{fv} existe", f"com o campo {fv}", f"{fv} presente"],
                               [f"{fv} exists", f"with the {fv} field", f"{fv} is present"])
            if arg is False:
                return self.ch([f"{fv} não existe", f"sem o campo {fv}", f"{fv} ausente"],
                               [f"{fv} does not exist", f"without the {fv} field", f"{fv} is missing"])
            raise Unsupported("exists")
        if op == "$regex":
            if not isinstance(arg, str) or (opts not in (None, "i")):
                raise Unsupported("regex")
            ci = opts == "i"
            m = re.fullmatch(r"\^([A-Za-z0-9 ]+)", arg)
            c = re.fullmatch(r"[A-Za-z0-9 ]+", arg)
            if m:
                base = self.ch([f"{fv} começa com {V(m.group(1))}", f"{fv} iniciando com {V(m.group(1))}"],
                               [f"{fv} starts with {V(m.group(1))}", f"{fv} begins with {V(m.group(1))}"])
            elif c:
                base = self.ch([f"{fv} contém {V(arg)}", f"{fv} que tenha {V(arg)}"],
                               [f"{fv} contains {V(arg)}", f"{fv} that includes {V(arg)}"])
            else:
                base = self.ch([f"{fv} casa com a regex {V(arg)}", f"{fv} com regex {V(arg)}"],
                               [f"{fv} matches the regex {V(arg)}", f"{fv} with regex {V(arg)}"])
            if ci:
                base += self.ch([" (sem diferenciar maiúsculas)", " ignorando caixa", " (case-insensitive)"],
                                [" (case-insensitive)", " ignoring case"])
            return base
        if op == "$size":
            if not isinstance(arg, int) or isinstance(arg, bool):
                raise Unsupported("size")
            return self.ch([f"{fv} com {arg} elementos", f"tamanho de {fv} igual a {arg}"],
                           [f"{fv} with {arg} elements", f"size of {fv} equals {arg}"])
        if op == "$type":
            if not isinstance(arg, str):
                raise Unsupported("type")
            return self.ch([f"{fv} do tipo {V(arg)}", f"{fv} com tipo BSON {V(arg)}"],
                           [f"{fv} of type {V(arg)}", f"{fv} with BSON type {V(arg)}"])
        if op == "$mod":
            if not (isinstance(arg, list) and len(arg) == 2 and all(isinstance(x, int) for x in arg)):
                raise Unsupported("mod")
            d, r = arg
            return self.ch([f"{fv} % {d} == {r}", f"{fv} dividido por {d} deixa resto {r}"],
                           [f"{fv} % {d} == {r}", f"{fv} divided by {d} leaves remainder {r}"])
        if op == "$not":
            if not is_opdict(arg):
                raise Unsupported("not")
            inner = self.join_and([self.op_text(f, o, a, None) for o, a in arg.items() if o != "$options"])
            return self.ch([f"não ({inner})"], [f"not ({inner})"])
        if op == "$elemMatch":
            if not isinstance(arg, dict) or not arg or any(k.startswith("$") for k in arg):
                raise Unsupported("elemMatch")
            inner = self.join_and([self.field_cond(k, v) for k, v in arg.items()])
            return self.ch([f"{fv} tem algum item em que {inner}", f"em {fv} existe um item com {inner}"],
                           [f"{fv} has an item where {inner}", f"some item of {fv} satisfies {inner}"])
        raise Unsupported(op)

    def filter_text(self, flt: dict, top: bool = True) -> str:
        parts = []
        for k, v in flt.items():
            if k in ("$or", "$and"):
                if not isinstance(v, list) or not v or not all(isinstance(x, dict) for x in v):
                    raise Unsupported(k)
                subs = [self.filter_text(x, False) for x in v]
                sep = self.or_() if k == "$or" else self.and_()
                parts.append("(" + sep.join(subs) + ")")
            elif k == "$expr":
                parts.append(self.expr_cmp(v))
            elif k.startswith("$"):
                raise Unsupported(k)
            else:
                parts.append(self.field_cond(k, v))
        if not parts:
            return ""
        return parts[0] if len(parts) == 1 else (self.join(parts) if top else self.and_().join(parts))

    def expr_cmp(self, e) -> str:
        if isinstance(e, dict) and len(e) == 1:
            (op, args), = e.items()
            sym = {"$eq": "=", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}.get(op)
            if sym and isinstance(args, list) and len(args) == 2:
                a, b = (self.ex(x) for x in args)
                return f"{a} {sym} {b}"
        raise Unsupported("expr")

    # ---- expressoes de agregacao ----
    def ex(self, e) -> str:
        if isinstance(e, str):
            if e.startswith("$$"):
                raise Unsupported("var")
            if e.startswith("$") and len(e) > 1:
                return F(e[1:])
            return V(e)
        if e is None or isinstance(e, (bool, int, float)) or is_ej(e):
            return V(e)
        if isinstance(e, dict) and len(e) == 1:
            (op, a), = e.items()
            if op in ("$multiply", "$add", "$subtract", "$divide") and isinstance(a, list) and len(a) >= 2:
                sym = {"$multiply": "*", "$add": "+", "$subtract": "-", "$divide": "/"}[op]
                return "(" + f" {sym} ".join(self.ex(x) for x in a) + ")"
            if op == "$round" and isinstance(a, list) and len(a) == 2:
                return f"round({self.ex(a[0])}, {a[1]})"
            if op in ("$toUpper", "$toLower", "$size", "$year", "$month", "$dayOfMonth"):
                nm = {"$toUpper": "upper", "$toLower": "lower", "$size": "size", "$year": "year", "$month": "month",
                      "$dayOfMonth": "day"}[op]
                return f"{nm}({self.ex(a)})"
            if op == "$concat" and isinstance(a, list):
                return "concat(" + ", ".join(self.ex(x) for x in a) + ")"
            if op == "$ifNull" and isinstance(a, list) and len(a) == 2:
                return f"ifNull({self.ex(a[0])}, {self.ex(a[1])})"
            if op == "$cond" and isinstance(a, list) and len(a) == 3:
                c = self.expr_cmp(a[0])
                t, f_ = self.ex(a[1]), self.ex(a[2])
                return self.ch([f"se {c} então {t} senão {f_}", f"{c} ? {t} : {f_}"], [f"if {c} then {t} else {f_}", f"{c} ? {t} : {f_}"])
            if op == "$dateToString" and isinstance(a, dict) and set(a) == {"format", "date"}:
                return self.ch([f"{self.ex(a['date'])} formatado como {V(a['format'])}"],
                               [f"{self.ex(a['date'])} formatted as {V(a['format'])}"])
        raise Unsupported("expr")

    def accum(self, name: str, spec) -> str:
        if not (isinstance(spec, dict) and len(spec) == 1):
            raise Unsupported("acc")
        (op, a), = spec.items()
        nm = F(name)
        if op == "$sum" and a == 1:
            return self.ch([f"{nm} = contagem", f"{nm} com a quantidade de documentos"], [f"{nm} = count", f"{nm} as the number of documents"])
        w = {
            "$sum": ("soma de", "sum of"), "$avg": ("média de", "average of"), "$min": ("mínimo de", "minimum of"),
            "$max": ("máximo de", "maximum of"), "$first": ("primeiro valor de", "first value of"),
            "$push": ("lista de", "list of"), "$addToSet": ("valores únicos de", "unique values of"),
        }.get(op)
        if not w:
            raise Unsupported(op)
        return f"{nm} = {w[0] if self.pt else w[1]} {self.ex(a)}"

    # ---- estagios ----
    def sort_text(self, srt: dict) -> str:
        items = []
        for k, d in srt.items():
            if d not in (1, -1):
                raise Unsupported("sort")
            if self.reg == "t" and self.rng.random() < 0.5:
                items.append(f"{F(k)} {'asc' if d == 1 else 'desc'}")
            elif self.pt:
                items.append(f"{F(k)} {self.rng.choice(['crescente', 'ascendente']) if d == 1 else self.rng.choice(['decrescente', 'descendente'])}")
            else:
                items.append(f"{F(k)} {'ascending' if d == 1 else 'descending'}")
        return self.lst(items)

    def proj_text(self, p: dict) -> str:
        inc = [k for k, v in p.items() if v in (1, True) and not isinstance(v, str)]
        exc = [k for k, v in p.items() if v in (0, False) and not isinstance(v, str)]
        if len(inc) + len(exc) != len(p):
            raise Unsupported("proj")
        parts = []
        inc_n = [k for k in inc]
        if inc_n:
            parts.append(self.ch([f"mostrando só {self.lst(F(k) for k in inc_n)}", f"retornando apenas {self.lst(F(k) for k in inc_n)}",
                                  f"com os campos {self.lst(F(k) for k in inc_n)}"],
                                 [f"showing only {self.lst(F(k) for k in inc_n)}", f"returning just {self.lst(F(k) for k in inc_n)}",
                                  f"projecting {self.lst(F(k) for k in inc_n)}"]))
        if exc:
            if exc == ["_id"] and inc_n:
                parts.append(self.ch(["sem o _id", "omitindo o _id"], ["without the _id", "excluding _id"]))
            else:
                parts.append(self.ch([f"sem os campos {self.lst(F(k) for k in exc)}", f"excluindo {self.lst(F(k) for k in exc)}"],
                                     [f"without the fields {self.lst(F(k) for k in exc)}", f"excluding {self.lst(F(k) for k in exc)}"]))
        return self.join(parts) if len(parts) < 2 else parts[0] + ", " + parts[1]

    def stage_text(self, st: dict) -> str:
        (name, b), = st.items()
        if name == "$match":
            t = self.filter_text(b)
            return self.ch([f"filtrar onde {t}", f"filtrar por {t}", f"match: {t}"],
                           [f"filter where {t}", f"keep only rows with {t}", f"match: {t}"])
        if name == "$group":
            if "_id" not in b:
                raise Unsupported("group")
            _id = b["_id"]
            if _id is None:
                key = self.ch(["agrupar tudo em um único grupo"], ["group everything into a single group"])
            elif isinstance(_id, str) and _id.startswith("$") and not _id.startswith("$$"):
                key = self.ch([f"agrupar por {F(_id[1:])}"], [f"group by {F(_id[1:])}"])
            elif isinstance(_id, dict) and not is_ej(_id) and all(not k.startswith("$") for k in _id):
                inner = ", ".join(f"{F(k)}: {self.ex(v)}" for k, v in _id.items())
                key = self.ch([f"agrupar por {{ {inner} }}"], [f"group by {{ {inner} }}"])
            else:
                key = self.ch([f"agrupar por {self.ex(_id)}"], [f"group by {self.ex(_id)}"])
            accs = [self.accum(k, v) for k, v in b.items() if k != "_id"]
            if accs:
                key += self.ch([", calculando ", ", com "], [", computing ", ", with "]) + self.lst(accs)
            return key
        if name == "$project":
            comp = [(k, v) for k, v in b.items() if not (v in (0, 1, True, False) and not isinstance(v, str))]
            simple = {k: v for k, v in b.items() if (k, v) not in comp}
            parts = []
            if simple:
                parts.append(self.proj_text(simple))
            for k, v in comp:
                parts.append(f"{F(k)} = {self.ex(v)}")
            return self.ch(["projetar: ", "projeção: "], ["project: ", "projection: "]) + ", ".join(parts)
        if name in ("$addFields", "$set"):
            parts = [f"{F(k)} = {self.ex(v)}" for k, v in b.items()]
            return self.ch(["adicionar o campo ", "criar o campo "] if len(parts) == 1 else ["adicionar os campos ", "criar os campos "],
                           ["add the field "] if len(parts) == 1 else ["add the fields "]) + ", ".join(parts)
        if name == "$unset":
            ps = [b] if isinstance(b, str) else list(b)
            return self.ch([f"remover os campos {self.lst(F(p) for p in ps)}"], [f"remove the fields {self.lst(F(p) for p in ps)}"])
        if name == "$sort":
            return self.ch([f"ordenar por {self.sort_text(b)}"], [f"sort by {self.sort_text(b)}"])
        if name == "$limit":
            return self.ch([f"limitar a {b}"], [f"limit to {b}"])
        if name == "$skip":
            return self.ch([f"pular {b}"], [f"skip {b}"])
        if name == "$count":
            return self.ch([f"contar e guardar em {F(b)}"], [f"count into {F(b)}"])
        if name == "$sortByCount":
            return self.ch([f"contar por {self.ex(b)} em ordem decrescente de contagem"],
                           [f"count by {self.ex(b)} sorted by count descending"])
        if name == "$unwind":
            if isinstance(b, str) and b.startswith("$"):
                return self.ch([f"desmembrar (unwind) {F(b[1:])}"], [f"unwind {F(b[1:])}"])
            if isinstance(b, dict) and isinstance(b.get("path"), str) and set(b) <= {"path", "preserveNullAndEmptyArrays"}:
                t = self.ch([f"desmembrar (unwind) {F(b['path'][1:])}"], [f"unwind {F(b['path'][1:])}"])
                if b.get("preserveNullAndEmptyArrays"):
                    t += self.ch([", preservando vazios"], [", preserving null and empty arrays"])
                return t
            raise Unsupported("unwind")
        if name == "$lookup":
            if set(b) == {"from", "localField", "foreignField", "as"}:
                return self.ch([f"fazer lookup em {F(b['from'])} com {F(b['localField'])} = {F(b['foreignField'])}, guardando em {F(b['as'])}"],
                               [f"lookup {F(b['from'])} joining {F(b['localField'])} = {F(b['foreignField'])} into {F(b['as'])}"])
            raise Unsupported("lookup pipeline")
        raise Unsupported(name)

    # ---- pedido completo ----
    def describe(self, plan: dict, coll: str) -> str:
        op = plan["op"]
        pt = self.pt
        cin, cof = self.coll_in(coll), self.coll_of(coll)
        flt = plan.get("filter") or {}
        where = self.filter_text(flt) if flt else ""
        w_pt = (" " + self.rng.choice(["onde", "em que", "com"]) + " " + where) if where else ""
        w_en = (" " + self.rng.choice(["where", "with"]) + " " + where) if where else ""
        mods: list[str] = []
        if op in ("find", "findOne"):
            if plan.get("sort") and op == "find":
                mods.append(self.ch([f"ordenados por {self.sort_text(plan['sort'])}", f"ordenando por {self.sort_text(plan['sort'])}"],
                                    [f"sorted by {self.sort_text(plan['sort'])}", f"ordered by {self.sort_text(plan['sort'])}"]))
            if plan.get("skip") is not None and op == "find":
                mods.append(self.ch([f"pulando os {plan['skip']} primeiros", f"ignorando os {plan['skip']} primeiros"],
                                    [f"skipping the first {plan['skip']}", f"after skipping {plan['skip']}"]))
            if plan.get("limit") is not None and op == "find":
                n = plan["limit"]
                mods.append(self.ch([f"limitado a {n}", f"só {n}", f"no máximo {n}"], [f"limited to {n}", f"only {n}", f"at most {n}"]))
            if plan.get("projection"):
                mods.append(self.proj_text(plan["projection"]))
            self.rng.shuffle(mods)
            tail = (", " + ", ".join(mods)) if mods else ""
            if op == "find":
                if pt:
                    heads = {"n": [f"Busque {cin} os documentos{w_pt}", f"Liste os documentos {cof}{w_pt}",
                                   f"Consulte {cin} os registros{w_pt}"],
                             "i": [f"me mostra os registros {cof}{w_pt}", f"traz os documentos {cof}{w_pt}",
                                   f"lista os itens {cof}{w_pt}"],
                             "t": [f"find {cin}{(' com filtro: ' + where) if where else ''}", f"Consulta {cin}{(' filtrando ' + where) if where else ''}"]}
                else:
                    heads = {"n": [f"Find the documents {cin}{w_en}", f"List the documents {cof}{w_en}",
                                   f"Get the records {cin}{w_en}"],
                             "i": [f"show me the docs {cof}{w_en}", f"get me the records {cof}{w_en}", f"list the stuff {cin}{w_en}"],
                             "t": [f"find {cin}{(' with filter: ' + where) if where else ''}", f"Query {cin}{(' filtering ' + where) if where else ''}"]}
                head = self.rng.choice(heads.get(self.reg) or heads["n"])
                return head + tail
            # findOne
            if pt:
                h = {"n": [f"Busque um documento {cof}{w_pt}", f"Traga um registro {cof}{w_pt}"],
                     "i": [f"traz só um documento {cof}{w_pt}", f"me dá um registro {cof}{w_pt}"],
                     "t": [f"findOne {cin}{w_pt}", f"findOne {cin}{(' com filtro: ' + where) if where else ''}"]}
            else:
                h = {"n": [f"Find one document {cof}{w_en}", f"Get a single record {cof}{w_en}"],
                     "i": [f"grab one doc {cof}{w_en}", f"give me one record {cof}{w_en}"],
                     "t": [f"findOne {cin}{w_en}", f"findOne {cin}{(' with filter: ' + where) if where else ''}"]}
            return self.rng.choice(h.get(self.reg) or h["n"]) + tail
        if op == "countDocuments":
            if pt:
                h = {"n": [f"Quantos documentos há {cin}{w_pt}?", f"Conte os documentos {cof}{w_pt}", f"Qual o total de registros {cof}{w_pt}?"],
                     "i": [f"quantos tem {cin}{w_pt}?", f"conta quantos documentos {cof}{w_pt}"],
                     "t": [f"countDocuments {cin}{w_pt}", f"countDocuments {cin}{(' com filtro: ' + where) if where else ''}"]}
            else:
                h = {"n": [f"How many documents are there {cin}{w_en}?", f"Count the documents {cof}{w_en}",
                           f"What is the total number of records {cof}{w_en}?"],
                     "i": [f"how many are there {cin}{w_en}?", f"count how many docs {cof}{w_en}"],
                     "t": [f"countDocuments {cin}{w_en}", f"countDocuments {cin}{(' with filter: ' + where) if where else ''}"]}
            return self.rng.choice(h.get(self.reg) or h["n"])
        if op == "distinct":
            f = plan["field"]
            if pt:
                h = {"n": [f"Quais os valores distintos de {F(f)} {cin}{w_pt}?", f"Liste os valores únicos de {F(f)} {cof}{w_pt}"],
                     "i": [f"quais valores diferentes de {F(f)} {cin}{w_pt}?", f"me lista os {F(f)} únicos {cof}{w_pt}"],
                     "t": [f"distinct de {F(f)} {cin}{w_pt}", f"distinct({F(f)}) {cin}{(' com filtro: ' + where) if where else ''}"]}
            else:
                h = {"n": [f"What are the distinct values of {F(f)} {cin}{w_en}?", f"List the unique values of {F(f)} {cof}{w_en}"],
                     "i": [f"what different {F(f)} values are there {cin}{w_en}?", f"list the unique {F(f)} {cof}{w_en}"],
                     "t": [f"distinct {F(f)} {cin}{w_en}", f"distinct({F(f)}) {cin}{(' with filter: ' + where) if where else ''}"]}
            return self.rng.choice(h.get(self.reg) or h["n"])
        if op == "aggregate":
            stages = plan["pipeline"]
            if len(stages) > 5:
                raise Unsupported("long pipeline")
            steps = [self.stage_text(s) for s in stages]
            if self.rng.random() < 0.5:
                conn = self.ch([", depois ", "; em seguida "], [", then ", "; next "])
                body = steps[0] + "".join(conn + s for s in steps[1:])
            else:
                body = " ".join(f"({i + 1}) {s}" for i, s in enumerate(steps))
            if pt:
                head = self.rng.choice([f"Faça uma agregação {cin}: ", f"Monte um aggregate {cin}: ", f"Aggregate {cin}: ", f"No pipeline {cin}: "])
            else:
                head = self.rng.choice([f"Run an aggregation {cin}: ", f"Build an aggregate {cin}: ", f"Aggregate {cin}: ", f"Pipeline {cin}: "])
            return head + body
        raise Unsupported(op)

    def explain(self, plan: dict, coll: str) -> str:
        """Frase curta (3a pessoa) do que a consulta faz, para comentario `//`."""
        op = plan["op"]
        self.strict = True
        flt = plan.get("filter") or {}
        where = self.filter_text(flt) if flt else ""
        c = F(coll)
        if op in ("find", "findOne"):
            bits = []
            if plan.get("sort") and op == "find":
                bits.append((f"ordenados por {self.sort_text(plan['sort'])}" if self.pt else f"sorted by {self.sort_text(plan['sort'])}"))
            if plan.get("limit") is not None and op == "find":
                bits.append(f"limite {plan['limit']}" if self.pt else f"limit {plan['limit']}")
            if plan.get("projection"):
                bits.append(self.proj_text(plan["projection"]))
            s = (f"Retorna {'o primeiro documento' if op == 'findOne' else 'documentos'} de {c}" if self.pt
                 else f"Returns {'the first document' if op == 'findOne' else 'documents'} from {c}")
            if where:
                s += (f" onde {where}" if self.pt else f" where {where}")
            if bits:
                s += ", " + ", ".join(bits)
            return s
        if op == "countDocuments":
            return (f"Conta os documentos de {c}" + (f" onde {where}" if where else "")) if self.pt else \
                   (f"Counts the documents in {c}" + (f" where {where}" if where else ""))
        if op == "distinct":
            return (f"Lista os valores distintos de {F(plan['field'])} em {c}" + (f" onde {where}" if where else "")) if self.pt else \
                   (f"Lists the distinct values of {F(plan['field'])} in {c}" + (f" where {where}" if where else ""))
        if op == "aggregate":
            steps = [self.stage_text(s) for s in plan["pipeline"]]
            return (f"Pipeline em {c}: " if self.pt else f"Pipeline on {c}: ") + "; ".join(steps)
        raise Unsupported(op)

    def describe_write(self, plan: dict, coll: str) -> str:
        """Pedido de ESCRITA (usado so em negativos)."""
        op = plan["op"]
        cin = self.coll_in(coll)
        where = self.filter_text(plan.get("filter") or {}) if plan.get("filter") else ""
        if op in ("deleteOne", "deleteMany"):
            one = op == "deleteOne"
            if self.pt:
                v = self.rng.choice(["Apague", "Exclua", "Remova", "Delete"])
                tgt = "um documento" if one else "todos os documentos"
                return f"{v} {tgt} {cin}" + (f" onde {where}" if where else "")
            v = self.rng.choice(["Delete", "Remove", "Erase"])
            tgt = "one document" if one else "all the documents"
            return f"{v} {tgt} {cin}" + (f" where {where}" if where else "")
        if op in ("updateOne", "updateMany"):
            ups = []
            for uop, fields in plan["update"].items():
                for k, v in fields.items():
                    if uop == "$set":
                        ups.append(self.ch([f"definir {F(k)} como {V(v)}"], [f"set {F(k)} to {V(v)}"]))
                    elif uop == "$inc":
                        ups.append(self.ch([f"incrementar {F(k)} em {V(v)}"], [f"increment {F(k)} by {V(v)}"]))
                    else:
                        raise Unsupported(uop)
            one = op == "updateOne"
            if self.pt:
                return (f"Atualize {'um documento' if one else 'os documentos'} {cin}" + (f" onde {where}" if where else "")
                        + ": " + ", ".join(ups))
            return (f"Update {'one document' if one else 'the documents'} {cin}" + (f" where {where}" if where else "")
                    + ": " + ", ".join(ups))
        raise Unsupported(op)


def finalize(text: str, lang: str, rng: random.Random, reg: str, deaccent_p: float = 0.0, typo_p: float = 0.0,
             info: dict | None = None) -> str:
    """Pos-edicao: pontuacao/caixa por registro, sem acento, erro de digitacao; remove marcadores."""
    t = re.sub(r"\s+", " ", text).strip()
    if reg == "i":
        if rng.random() < 0.5 and t and t[0] not in (P0,):
            t = t[0].lower() + t[1:]
        t = t.rstrip(".")
        if rng.random() < 0.15:
            t += rng.choice([" pfv", " valeu", " por favor"] if lang == "pt" else [" pls", " thanks", " please"])
    elif reg == "t":
        if t and t[0] != P0:
            t = t[0].upper() + t[1:]
    else:
        if t and t[0] != P0:
            t = t[0].upper() + t[1:]
        if not t.endswith(("?", ".")) and rng.random() < 0.3:
            t += "."
    if lang == "pt" and rng.random() < deaccent_p:
        t2 = map_unprotected(t, deaccent)
        if info is not None:
            info["deaccent"] = t2 != t
        t = t2
    if rng.random() < typo_p:
        t2 = swap_typo(t, rng)
        if info is not None:
            info["typo"] = t2 != t
        t = t2
    return strip_marks(t)
