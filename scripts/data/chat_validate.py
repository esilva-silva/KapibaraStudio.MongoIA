#!/usr/bin/env python3
"""Validacao das saidas de chat/correcao (spec fase-03 §4.5, etapas 2-6) sem o ContractProbe.

Etapas, na ordem; a primeira que falha define `stage`:
  syntax  - sintaxe JS (node `vm.Script`, em lote; se node faltar, so o parser abaixo)
  api     - metodo/global/operador fora do Console (parser do subconjunto Console + lista de operadores)
  fields  - campos inexistentes no schema (analyze_plan do gen_snippets), `$` faltando, estrutura
  type    - valor de tipo incompativel com o tipo BSON do campo
  exec    - executa o plano no MongoDB (kapi_fx_*, somente leitura, maxTimeMS) via validate_snippets
Retorna o plano reconstruido do texto, para compara-lo com o plano que originou o `output`.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_snippets as vs  # noqa: E402
from gen_snippets import EJSON_KEYS, analyze_plan  # noqa: E402

READ_METHODS = {"find", "findOne", "aggregate", "countDocuments", "distinct"}
OK_COLL_METHODS = READ_METHODS | {"estimatedDocumentCount"}
WRITE_METHODS = {"insertOne", "insertMany", "updateOne", "updateMany", "replaceOne", "deleteOne", "deleteMany", "drop",
                 "createIndex", "dropIndex", "bulkWrite", "createCollection", "dropDatabase", "findOneAndUpdate",
                 "findOneAndDelete", "findOneAndReplace", "remove", "update", "insert", "save", "renameCollection"}
CURSOR_METHODS = {"sort", "skip", "limit", "project", "toArray"}
WRITE_RE = re.compile(r"\.(?:" + "|".join(sorted(WRITE_METHODS, key=len, reverse=True)) + r")\s*\(|\$out\b|\$merge\b|\brunCommand\b"
                      r"|\bdropDatabase\b|\beval\s*\(")

QUERY_OPS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$and", "$or", "$nor", "$not", "$exists", "$type",
             "$regex", "$options", "$expr", "$mod", "$size", "$all", "$elemMatch", "$text", "$jsonSchema"}
ACC_OPS = {"$sum", "$avg", "$min", "$max", "$first", "$last", "$push", "$addToSet", "$count", "$stdDevPop", "$stdDevSamp"}
EXPR_OPS = {"$multiply", "$add", "$subtract", "$divide", "$round", "$toUpper", "$toLower", "$concat", "$ifNull", "$cond",
            "$year", "$month", "$dayOfMonth", "$dateToString", "$literal", "$toString", "$toInt", "$arrayElemAt", "$abs",
            "$ceil", "$floor", "$mod", "$switch", "$isArray", "$dayOfWeek", "$hour", "$substr", "$trim", "$split",
            "$filter", "$map", "$reduce", "$toDate", "$toDouble", "$strLenCP", "$max", "$min"}
KNOWN_OPS = QUERY_OPS | ACC_OPS | EXPR_OPS | vs.ALLOWED_STAGES | EJSON_KEYS | {"$expr", "$group", "$unwind"}
# chaves de estagio com argumentos nao-operador (nao sao operadores)
NUM_T = {"int32", "int64", "double", "decimal128"}


class ChatParseError(Exception):
    def __init__(self, stage: str, msg: str):
        super().__init__(msg)
        self.stage, self.msg = stage, msg


# --------------------------------------------------------------------------- tokenizer / parser

TOKEN_RE = re.compile(r"""
    (?P<ws>\s+) | (?P<lc>//[^\n]*) | (?P<bc>/\*.*?\*/) |
    (?P<str>"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*') |
    (?P<num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) |
    (?P<id>[A-Za-z_$][\w$]*) |
    (?P<p>[{}\[\](),:;.\-=])
""", re.X | re.S)

ESC = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"', "'": "'", "0": "\0", "b": "\b", "f": "\f", "v": "\v"}


def unescape(s: str) -> str:
    out, i = [], 1
    end = len(s) - 1
    while i < end:
        c = s[i]
        if c == "\\" and i + 1 < end:
            n = s[i + 1]
            if n == "u" and i + 5 < len(s) + 1:
                out.append(chr(int(s[i + 2:i + 6], 16)))
                i += 6
                continue
            out.append(ESC.get(n, n))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def tokenize(text: str):
    toks, pos = [], 0
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m:
            raise ChatParseError("syntax", f"token inesperado em {pos}: {text[pos:pos + 12]!r}")
        pos = m.end()
        k = m.lastgroup
        if k in ("ws", "lc", "bc"):
            continue
        toks.append((k, m.group()))
    return toks


ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d{1,3})?(Z|[+-]\d{2}:?\d{2})?)?$")
OID_RE = re.compile(r"^[0-9a-fA-F]{24}$")
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
NUM_STR_RE = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")


class Db:
    pass


class Pool:
    def __init__(self, parts):
        self.parts = parts


class Coll:
    def __init__(self, name, via, conn=None, database=None):
        self.name, self.via, self.conn, self.database = name, via, conn, database


class Cur:
    def __init__(self, plan, terminal=False):
        self.plan, self.terminal = plan, terminal


class Bound:
    def __init__(self, obj, name):
        self.obj, self.name = obj, name


class Parser:
    def __init__(self, text: str):
        self.t = tokenize(text)
        self.i = 0
        self.env: dict = {}
        self.meta: dict = {}

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else (None, None)

    def take(self):
        if self.i >= len(self.t):
            raise ChatParseError("syntax", "fim inesperado do texto")
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, p):
        k, v = self.take()
        if v != p or k not in ("p", "id"):
            raise ChatParseError("syntax", f"esperado {p!r}, encontrado {v!r}")

    def at(self, p):
        k, v = self.peek()
        return v == p and k == "p"

    # ---- valores ----
    def value(self):
        k, v = self.take()
        if k == "str":
            return unescape(v)
        if k == "num":
            return float(v) if re.search(r"[.eE]", v) else int(v)
        if k == "p" and v == "-":
            k2, v2 = self.take()
            if k2 != "num":
                raise ChatParseError("syntax", "numero esperado apos '-'")
            n = float(v2) if re.search(r"[.eE]", v2) else int(v2)
            return -n
        if k == "p" and v == "{":
            return self.obj()
        if k == "p" and v == "[":
            return self.arr()
        if k == "id":
            if v == "true":
                return True
            if v == "false":
                return False
            if v == "null":
                return None
            if v == "new":
                k2, v2 = self.take()
                if v2 != "Date":
                    raise ChatParseError("api", f"new {v2} nao suportado")
                return self.ctor("Date")
            if self.at("(") and v in ("ObjectId", "ISODate", "NumberLong", "NumberDecimal", "UUID", "Date"):
                return self.ctor(v)
            if v in self.env and self.env[v][0] == "val":
                return self.env[v][1]
            raise ChatParseError("api", f"identificador desconhecido: {v}")
        raise ChatParseError("syntax", f"valor inesperado: {v!r}")

    def ctor(self, name):
        self.expect("(")
        if self.at(")"):
            self.take()
            raise ChatParseError("api", f"{name}() exige argumento de texto")
        arg = self.value()
        self.expect(")")
        if not isinstance(arg, str):
            raise ChatParseError("api", f"{name} exige texto (recebeu {type(arg).__name__})")
        if name == "ObjectId":
            if not OID_RE.match(arg):
                raise ChatParseError("api", f"ObjectId invalido: {arg!r}")
            return {"$oid": arg.lower()}
        if name in ("ISODate", "Date"):
            if not ISO_RE.match(arg):
                raise ChatParseError("api", f"data invalida: {arg!r}")
            return {"$date": arg}
        if name == "NumberLong":
            if not re.match(r"^-?\d+$", arg):
                raise ChatParseError("api", f"NumberLong invalido: {arg!r}")
            return {"$numberLong": arg}
        if name == "NumberDecimal":
            if not NUM_STR_RE.match(arg):
                raise ChatParseError("api", f"NumberDecimal invalido: {arg!r}")
            return {"$numberDecimal": arg}
        if not UUID_RE.match(arg):
            raise ChatParseError("api", f"UUID invalido: {arg!r}")
        return {"$uuid": arg.lower()}

    def obj(self):
        d = {}
        while True:
            if self.at("}"):
                self.take()
                return d
            k, v = self.take()
            if k == "str":
                key = unescape(v)
            elif k in ("id", "num"):
                key = v
            else:
                raise ChatParseError("syntax", f"chave de objeto invalida: {v!r}")
            self.expect(":")
            d[key] = self.value()
            if self.at(","):
                self.take()
                continue
            if self.at("}"):
                continue
            raise ChatParseError("syntax", f"esperado ',' ou '}}' em objeto, encontrado {self.peek()[1]!r}")

    def arr(self):
        a = []
        while True:
            if self.at("]"):
                self.take()
                return a
            a.append(self.value())
            if self.at(","):
                self.take()
                continue
            if self.at("]"):
                continue
            raise ChatParseError("syntax", f"esperado ',' ou ']' em array, encontrado {self.peek()[1]!r}")

    def args(self):
        self.expect("(")
        out = []
        while True:
            if self.at(")"):
                self.take()
                return out
            out.append(self.value())
            if self.at(","):
                self.take()
                continue
            if self.at(")"):
                continue
            raise ChatParseError("syntax", f"esperado ',' ou ')' em chamada, encontrado {self.peek()[1]!r}")

    # ---- cadeia de statement ----
    def primary(self):
        k, v = self.take()
        if k != "id":
            raise ChatParseError("syntax", f"statement inesperado: {v!r}")
        if v == "db":
            return Db()
        if v == "ConnectionPool":
            return Pool([])
        if v in self.env:
            kind, val = self.env[v]
            if kind == "coll":
                return val
            raise ChatParseError("api", f"{v} nao e uma colecao")
        raise ChatParseError("api", f"identificador desconhecido: {v}")

    def chain(self):
        obj = self.primary()
        while True:
            k, v = self.peek()
            if k == "p" and v == ".":
                self.take()
                k2, name = self.take()
                if k2 != "id":
                    raise ChatParseError("syntax", f"nome de membro esperado, encontrado {name!r}")
                obj = self.member(obj, name, via_dot=True)
            elif k == "p" and v == "[":
                self.take()
                k2, s = self.take()
                if k2 != "str":
                    raise ChatParseError("syntax", "indexador deve ser texto")
                self.expect("]")
                obj = self.member(obj, unescape(s), via_dot=False)
            elif k == "p" and v == "(":
                if not isinstance(obj, Bound):
                    raise ChatParseError("syntax", "chamada sobre valor que nao e funcao")
                obj = self.call(obj, self.args())
            else:
                break
        return obj

    def member(self, obj, name, via_dot):
        if isinstance(obj, Db):
            if via_dot and name == "getCollection" and self.at("("):
                return Bound(obj, name)
            if via_dot and self.at("(") :
                raise ChatParseError("api", f"metodo de banco nao suportado no chat: {name}")
            return Coll(name, "dot" if via_dot else "bracket")
        if isinstance(obj, Pool):
            parts = obj.parts + [name]
            if len(parts) == 3:
                return Coll(parts[2], "pool", parts[0], parts[1])
            if len(parts) > 3:
                raise ChatParseError("api", "ConnectionPool.<conexao>.<banco>.<colecao>")
            return Pool(parts)
        if isinstance(obj, Coll):
            if not via_dot:
                raise ChatParseError("api", "indexador sobre colecao")
            return Bound(obj, name)
        if isinstance(obj, Cur):
            return Bound(obj, name)
        raise ChatParseError("api", f"membro {name!r} indisponivel")

    def call(self, b: Bound, args):
        o, name = b.obj, b.name
        if isinstance(o, Db):
            if name == "getCollection" and len(args) == 1 and isinstance(args[0], str):
                return Coll(args[0], "getCollection")
            raise ChatParseError("api", "getCollection exige texto")
        if isinstance(o, Coll):
            self.meta.update(collection=o.name, root_form=o.via, connection=o.conn, database=o.database)
            if name in WRITE_METHODS:
                raise ChatParseError("api", f"metodo de escrita nao permitido no chat: {name}")
            if name not in READ_METHODS:
                raise ChatParseError("api", f"metodo inexistente no Console: {name}")
            return Cur(self.build(name, args), terminal=name not in ("find", "aggregate"))
        if isinstance(o, Cur):
            if name not in CURSOR_METHODS:
                raise ChatParseError("api", f"metodo de cursor inexistente no Console: {name}")
            if o.terminal:
                raise ChatParseError("api", f"{name} nao se aplica ao resultado")
            p = o.plan
            if name == "toArray":
                return o
            if p["op"] != "find" and name in ("sort", "skip", "limit", "project"):
                raise ChatParseError("api", f"{name} so existe em cursor de find")
            if name in ("sort", "project"):
                if len(args) != 1 or not isinstance(args[0], dict):
                    raise ChatParseError("api", f"{name} exige um objeto")
                p["sort" if name == "sort" else "projection"] = args[0]
            else:
                if len(args) != 1 or isinstance(args[0], bool) or not isinstance(args[0], int):
                    raise ChatParseError("api", f"{name} exige inteiro")
                p[name] = args[0]
            return o
        raise ChatParseError("api", "chamada invalida")

    @staticmethod
    def build(name, args):
        if name in ("find", "findOne"):
            if len(args) > 2:
                raise ChatParseError("api", f"{name} aceita (filtro, projecao)")
            flt = args[0] if args else {}
            if not isinstance(flt, dict):
                raise ChatParseError("api", "filtro deve ser objeto")
            p = {"op": name, "filter": flt}
            if len(args) == 2:
                if not isinstance(args[1], dict):
                    raise ChatParseError("api", "projecao deve ser objeto")
                p["projection"] = args[1]
            return p
        if name == "aggregate":
            if len(args) < 1 or not isinstance(args[0], list):
                raise ChatParseError("api", "aggregate exige array de estagios")
            return {"op": "aggregate", "pipeline": args[0]}
        if name == "countDocuments":
            if len(args) > 1 or (args and not isinstance(args[0], dict)):
                raise ChatParseError("api", "countDocuments(filtro)")
            return {"op": "countDocuments", "filter": args[0] if args else {}}
        if name == "distinct":
            if not args or not isinstance(args[0], str):
                raise ChatParseError("api", "distinct(campo, filtro)")
            p = {"op": "distinct", "field": args[0], "filter": args[1] if len(args) > 1 and isinstance(args[1], dict) else {}}
            return p
        raise ChatParseError("api", name)

    def program(self):
        result = None
        n = 0
        while self.i < len(self.t):
            k, v = self.peek()
            if k == "id" and v in ("const", "let", "var"):
                self.take()
                k2, name = self.take()
                if k2 != "id":
                    raise ChatParseError("syntax", "nome de variavel esperado")
                self.expect("=")
                save = self.i
                k3, v3 = self.peek()
                if k3 == "id" and (v3 in ("db", "ConnectionPool") or (v3 in self.env and self.env[v3][0] == "coll")):
                    c = self.chain()
                    if not isinstance(c, Coll):
                        raise ChatParseError("api", "variavel deve guardar uma colecao")
                    self.env[name] = ("coll", c)
                else:
                    self.env[name] = ("val", self.value())
            else:
                result = self.chain()
                n += 1
            if self.at(";"):
                self.take()
        if result is None or not isinstance(result, Cur):
            raise ChatParseError("api", "nenhuma consulta (find/aggregate/...) no texto")
        if n != 1:
            raise ChatParseError("api", "esperado exatamente um statement de consulta")
        self.meta["root_form"] = self.meta.get("root_form")
        return result.plan, self.meta


def parse_console(text: str) -> tuple[dict, dict]:
    return Parser(text).program()


def norm_plan(p: dict) -> dict:
    q = dict(p)
    if q["op"] in ("find", "findOne", "countDocuments", "distinct"):
        q["filter"] = q.get("filter") or {}
    for k in ("projection", "sort"):
        if k in q and not q[k]:
            del q[k]
    return q


def plans_equal(a: dict, b: dict) -> bool:
    a, b = norm_plan(a), norm_plan(b)
    if a != b:
        return False
    if "sort" in a and list(a["sort"].items()) != list(b["sort"].items()):
        return False
    return True


# --------------------------------------------------------------------------- JS syntax (node)

NODE_SCRIPT = ("const vm=require('vm');let s='';process.stdin.setEncoding('utf8');process.stdin.on('data',d=>s+=d).on('end',()=>{"
               "const a=JSON.parse(s);process.stdout.write(JSON.stringify(a.map(t=>{try{new vm.Script(t);return null}"
               "catch(e){return e.name+': '+e.message}})))})")


def node_available() -> bool:
    return shutil.which("node") is not None


def node_syntax(texts: list[str]) -> list[str | None]:
    """Mensagem de erro por texto (None = sintaxe JS valida). Sem node: tudo None."""
    if not texts or not node_available():
        return [None] * len(texts)
    out: list[str | None] = []
    for i in range(0, len(texts), 4000):
        chunk = texts[i:i + 4000]
        r = subprocess.run(["node", "-e", NODE_SCRIPT], input=json.dumps(chunk).encode("utf-8"), capture_output=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError("node: " + r.stderr.decode("utf-8", "replace")[:300])
        out.extend(json.loads(r.stdout.decode("utf-8")))
    return out



class NodeChecker:
    """Processo node persistente (uma linha JSON por texto -> uma linha de resposta). Sem node: tudo valido."""

    SERVER = ("const vm=require('vm');const rl=require('readline').createInterface({input:process.stdin});"
              "rl.on('line',l=>{let r=null;try{new vm.Script(JSON.parse(l))}catch(e){r=e.name+': '+e.message}"
              "process.stdout.write(JSON.stringify(r)+String.fromCharCode(10))})")

    def __init__(self):
        self.proc = None
        if node_available():
            self.proc = subprocess.Popen(["node", "-e", self.SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL)

    def check(self, text: str) -> str | None:
        if self.proc is None:
            return None
        self.proc.stdin.write((json.dumps(text) + chr(10)).encode("utf-8"))
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline().decode("utf-8"))

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None

# --------------------------------------------------------------------------- checagens estaticas

def walk_keys(v, fn, depth=0):
    if depth > 40:
        return
    if isinstance(v, dict):
        for k, x in v.items():
            fn(k)
            walk_keys(x, fn, depth + 1)
    elif isinstance(v, list):
        for x in v:
            walk_keys(x, fn, depth + 1)


def check_ops(plan: dict):
    bad = []

    def fn(k):
        if k.startswith("$") and k not in KNOWN_OPS:
            bad.append(k)
    walk_keys(plan.get("filter"), fn)
    walk_keys(plan.get("pipeline"), fn)
    if bad:
        return "api", f"operador/estagio inexistente: {bad[0]}"
    return None


def vtype(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, dict) and len(v) == 1:
        k = next(iter(v))
        return {"$date": "date", "$oid": "objectid", "$uuid": "uuid", "$numberLong": "number", "$numberDecimal": "number"}.get(k)
    return None


LEAF = {"string": "string", "int32": "number", "int64": "number", "double": "number", "decimal128": "number", "bool": "bool",
        "date": "date", "objectid": "objectid", "uuid": "uuid"}


def _check_filter_types(flt, coll, errs):
    if not isinstance(flt, dict):
        return
    for k, v in flt.items():
        if k in ("$and", "$or", "$nor"):
            for s in v:
                _check_filter_types(s, coll, errs)
            continue
        if k.startswith("$"):
            continue
        f = coll.fields.get(k)
        want = LEAF.get(f.leaf_t) if f else None
        if want is None:
            continue
        vals, regex = [], False
        if isinstance(v, dict) and v and not (set(v) & EJSON_KEYS) and all(x.startswith("$") for x in v):
            for op, a in v.items():
                if op in ("$eq", "$ne", "$gt", "$gte", "$lt", "$lte"):
                    vals.append(a)
                elif op in ("$in", "$nin", "$all") and isinstance(a, list):
                    vals.extend(a)
                elif op == "$regex":
                    regex = True
                elif op == "$not" and isinstance(a, dict):
                    for o2, a2 in a.items():
                        if o2 in ("$gt", "$gte", "$lt", "$lte", "$eq", "$ne"):
                            vals.append(a2)
        else:
            vals.append(v)
        if regex and want != "string":
            errs.append(f"$regex em campo nao-string: {k} ({f.leaf_t})")
        for x in vals:
            t = vtype(x)
            if t is None or t == "null":
                continue
            if t != want:
                errs.append(f"tipo incompativel: {k} e {f.leaf_t} mas o valor e {t}")
                return


def check_types(plan: dict, coll):
    errs: list[str] = []
    if plan["op"] in ("find", "findOne", "countDocuments", "distinct"):
        _check_filter_types(plan.get("filter"), coll, errs)
    elif plan["op"] == "aggregate":
        for st in plan["pipeline"]:
            if not isinstance(st, dict) or len(st) != 1:
                break
            (name, b), = st.items()
            if name == "$match":
                _check_filter_types(b, coll, errs)
            elif name not in ("$sort", "$limit", "$skip"):
                break
    if errs:
        return "type", errs[0]
    return None


def check_dollar(plan: dict, coll):
    """`$` faltando em referencia a campo dentro de $group/$sortByCount/$unwind (vira literal silencioso)."""
    if plan["op"] != "aggregate":
        return None
    names = set(coll.fields)
    for st in plan["pipeline"]:
        if not (isinstance(st, dict) and len(st) == 1):
            continue
        (name, b), = st.items()
        if name == "$group" and isinstance(b, dict):
            for k, v in b.items():
                arg = v if k == "_id" else (next(iter(v.values())) if isinstance(v, dict) and len(v) == 1 else None)
                if isinstance(arg, str) and not arg.startswith("$") and arg in names and arg != "_id":
                    return "fields", f"possivel '$' faltando: {arg!r} em $group"
                if k != "_id" and isinstance(v, dict):
                    for o in v:
                        if not o.startswith("$"):
                            return "fields", f"operador de acumulador sem '$': {o}"
        elif name == "$sortByCount" and isinstance(b, str) and not b.startswith("$"):
            return "fields", "possivel '$' faltando em $sortByCount"
        elif name == "$unwind":
            p = b if isinstance(b, str) else (b.get("path") if isinstance(b, dict) else None)
            if isinstance(p, str) and not p.startswith("$"):
                return "fields", "possivel '$' faltando em $unwind"
    return None


class Executor:
    """Executa planos de LEITURA em kapi_fx_* (somente leitura; ver validate_snippets)."""

    def __init__(self, cfg: dict, uri: str | None = None):
        vc = cfg.get("validation", {})
        uri = uri or vc.get("uri", "mongodb://127.0.0.1:27017")
        if not re.match(r"^mongodb://(127\.0\.0\.1|localhost)(:\d+)?/?$", uri):
            raise SystemExit("RECUSADO: apenas MongoDB local")
        self.client, self.guard = vs.make_client(uri)
        self.client.admin.command("ping")
        self.existing = {n for n in self.client.list_database_names() if n.startswith(vc.get("db_prefix", "kapi_fx_"))}
        self.max_ms = int(vc.get("max_time_ms", 2000))
        self.cap = int(vc.get("result_cap", 101))

    def run(self, db: str, collection: str, plan: dict) -> dict:
        if db not in self.existing:
            return {"exec": "unseeded", "n_docs": None, "error": f"banco {db} nao semeado"}
        r = vs.validate_record(lambda n: self.client[n], {"id": "chat", "db": db, "op": plan["op"], "plan": plan,
                                                           "collection": collection}, self.max_ms, self.cap)
        return r

    @property
    def violations(self):
        return list(self.guard.violations)


def validate_text(text: str, sch, coll, executor: Executor | None, syntax_msg: str | None = None, db: str | None = None) -> dict:
    """Pipeline completo de um texto. `stage` = ok | syntax | api | fields | type | exec."""
    res: dict = {"stage": "ok", "error": None, "n_docs": None, "plan": None}
    if syntax_msg:
        return {**res, "stage": "syntax", "error": syntax_msg}
    try:
        plan, meta = parse_console(text)
    except ChatParseError as e:
        return {**res, "stage": e.stage, "error": e.msg}
    except Exception as e:  # defensivo
        return {**res, "stage": "syntax", "error": f"{type(e).__name__}: {e}"}
    res["plan"], res["meta"] = plan, meta
    if meta.get("collection") != coll.name:
        return {**res, "stage": "fields", "error": f"colecao {meta.get('collection')!r} nao e {coll.name!r}"}
    if meta.get("root_form") == "pool" and meta.get("database") != sch.schema_id:
        return {**res, "stage": "fields", "error": f"banco {meta.get('database')!r} desconhecido"}
    for chk in (check_ops(plan),):
        if chk:
            return {**res, "stage": chk[0], "error": chk[1]}
    try:
        an = analyze_plan(plan, sch, coll)
    except Exception as e:
        return {**res, "stage": "fields", "error": f"plano malformado: {type(e).__name__}: {e}"}
    if an["missing"]:
        return {**res, "stage": "fields", "error": "campo/estrutura invalido: " + str(an["missing"][0])}
    res["fields_used"] = an["used"]
    for chk in (check_types(plan, coll), check_dollar(plan, coll)):
        if chk:
            return {**res, "stage": chk[0], "error": chk[1]}
    if executor is not None:
        t0 = time.perf_counter()
        r = executor.run(db or sch.db_name, coll.name, plan)
        res["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if r["exec"] == "ok":
            res["n_docs"] = r["n_docs"]
        else:
            return {**res, "stage": "exec", "error": r.get("error") or r.get("reason") or r["exec"]}
    return res


def has_executable_write(text: str) -> bool:
    """True se alguma linha de CODIGO (fora de comentario //) contem escrita/DDL/$out/$merge."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("//"):
            continue
        s = re.sub(r"//.*$", "", s) if "//" in s and "://" not in s else s
        if WRITE_RE.search(s):
            return True
    return False
