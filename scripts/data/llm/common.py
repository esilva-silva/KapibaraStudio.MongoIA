#!/usr/bin/env python3
"""Utilitarios comuns da geracao por LLM: configuracao, schemas por PAPEL, fixtures (amostras), checagem estatica de escrita,
normalizacao/hash de consultas. Somente leitura sobre data/schemas e data/fixtures."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent
ROOT = HERE.parents[2]
for _p in (str(HERE), str(DATA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jsscan  # noqa: E402
from gen_snippets import Schema  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "llm_gen.yaml"
SCHEMAS_DIR = ROOT / "data" / "schemas"
FIXTURES_MANIFEST = ROOT / "data" / "fixtures" / "manifest.json"
ROLES_OK = ("train", "hom")      # `eval` (val/test/bench do corpus v2c) nunca e usado aqui


def load_cfg(path=None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha(s: str, n: int = 16) -> str:
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()[:n]


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


# ----------------------------------------------------------------------------------------------------------------------------
# schemas por papel (le `split_role` do JSON cru: gen_snippets.Schema ignora `hom`)
# ----------------------------------------------------------------------------------------------------------------------------
@dataclass
class SchemaInfo:
    schema: Schema
    raw: dict
    role: str | None
    path: Path
    db: str = ""                           # banco real kapi_fx_* (fixtures/manifest.json)
    coll_counts: dict = field(default_factory=dict)

    @property
    def schema_id(self) -> str:             # id curto (nome do "banco" no editor)
        return self.schema.schema_id

    @property
    def full_id(self) -> str:
        return self.schema.full_id

    @property
    def domain(self) -> str:
        return self.schema.domain

    def coll(self, name: str):
        return next(c for c in self.schema.colls if c.name == name)


def load_fixture_dbs() -> dict:
    """{(schema_id, colecao): {db, count}} e {schema_id: db} a partir de data/fixtures/manifest.json (releia: o outro agente semeia v3_*)."""
    try:
        m = json.loads(FIXTURES_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for e in m.get("entries", []):
        out[(e["schema_id"], e["collection"])] = {"db": e["db"], "count": e.get("count"), "role": e.get("split_role")}
    return out


def load_schema_infos(roles=ROLES_OK, schemas_dir=SCHEMAS_DIR, require_fixture: bool = True) -> list[SchemaInfo]:
    """Schemas cujo `split_role` esta em `roles` (train e/ou hom). `eval` e ausente sao ignorados. Com `require_fixture`, so entram os
    semeados (presentes no manifesto de fixtures)."""
    fx = load_fixture_dbs()
    out = []
    for p in sorted(Path(schemas_dir).glob("*/*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        role = raw.get("split_role")
        if role not in roles or role not in ROLES_OK:
            continue
        try:
            sch = Schema(raw, p)
        except Exception:
            continue
        if not sch.colls:
            continue
        sid = sch.schema_id
        dbs = {fx[(sid, c.name)]["db"] for c in sch.colls if (sid, c.name) in fx}
        if require_fixture and not dbs:
            continue
        db = sorted(dbs)[0] if dbs else sch.db_name
        info = SchemaInfo(sch, raw, role, p, db, {c.name: (fx.get((sid, c.name)) or {}).get("count") for c in sch.colls})
        out.append(info)
    return out


def assert_roles_disjoint(train: list[SchemaInfo], hom: list[SchemaInfo]) -> None:
    """Isolamento hom x train: nenhum schema (id ou banco) nos dois lados."""
    a = {s.schema_id for s in train}
    b = {s.schema_id for s in hom}
    if a & b:
        raise SystemExit(f"schemas em train e hom ao mesmo tempo: {sorted(a & b)}")
    bad = [s.schema_id for s in train if s.role != "train"] + [s.schema_id for s in hom if s.role != "hom"]
    if bad:
        raise SystemExit(f"papel incompativel: {bad}")


# ----------------------------------------------------------------------------------------------------------------------------
# checagem ESTATICA de escrita/ddl/execucao arbitraria (sobre o texto com strings e comentarios mascarados)
# ----------------------------------------------------------------------------------------------------------------------------
FORBIDDEN_IDENTS = {
    # escrita / DDL
    "insert", "insertOne", "insertMany", "update", "updateOne", "updateMany", "replaceOne", "deleteOne", "deleteMany", "remove",
    "save", "bulkWrite", "initializeOrderedBulkOp", "initializeUnorderedBulkOp", "findOneAndUpdate", "findOneAndDelete",
    "findOneAndReplace", "findAndModify", "drop", "dropDatabase", "dropIndex", "dropIndexes", "createIndex", "createIndexes",
    "ensureIndex", "reIndex", "createCollection", "createView", "createSearchIndex", "createSearchIndexes", "dropSearchIndex",
    "updateSearchIndex", "renameCollection", "collMod", "convertToCapped", "compact", "hideIndex", "unhideIndex",
    "createUser", "updateUser", "dropUser", "createRole", "grantRolesToUser", "revokeRolesFromUser", "changeUserPassword",
    "setProfilingLevel", "setLogLevel", "setParameter", "setFeatureCompatibilityVersion", "shutdownServer", "killOp", "fsync",
    "fsyncLock", "currentOp", "cloneCollection", "copyDatabase", "mapReduce", "startSession", "withSession", "withTransaction",
    "startTransaction", "commitTransaction", "watch", "enableSharding", "shardCollection", "moveChunk", "sh", "rs",
    "runCommand", "adminCommand", "getMongo", "Mongo", "connect", "getSiblingDBs",
    # execucao arbitraria / ambiente
    "eval", "load", "require", "import", "process", "child_process", "fs", "fetch", "sleep", "exit", "quit", "Function",
    "constructor", "globalThis", "window", "global", "setTimeout", "setInterval", "XMLHttpRequest", "WebSocket", "passwordPrompt",
    "cat", "ls", "pwd", "cd", "run", "runProgram", "_srand", "_rand", "hostname", "use", "show", "db_eval", "getWriteConcern",
    "setWriteConcern", "setReadPref", "setSecondaryOk", "snapshot", "notebook", "prompt",
}
FORBIDDEN_DOLLAR = re.compile(r"\$(out|merge|function|accumulator|where|currentOp|listSessions|listLocalSessions|changeStream|"
                              r"planCacheStats|collStats|indexStats|listSearchIndexes|search|searchMeta|vectorSearch)\b")
_WORD = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")


def static_check(text: str, allowed_dbs: set[str] | None = None) -> list[str]:
    """Motivos de bloqueio (lista vazia = ok). Atua nos identificadores do codigo (strings/comentarios mascarados) e na presenca de
    operadores proibidos em qualquer lugar do texto (nome de operador dentro de string tambem e recusado)."""
    reasons: list[str] = []
    if not text or not text.strip():
        return ["vazio"]
    if len(text) > 3500:
        reasons.append("longo_demais")
    masked = jsscan.mask(text)
    for m in _WORD.finditer(masked):
        w = m.group(0)
        if w in FORBIDDEN_IDENTS:
            after = masked[m.end():m.end() + 8].lstrip(" \t\r\n")
            before = masked[max(0, m.start() - 8):m.start()].rstrip(" \t\r\n")
            if after.startswith(":") and before[-1:] in ("{", ",", ""):      # nome de campo (chave de objeto), nao codigo
                continue
            reasons.append(f"identificador_proibido:{w}")
    for m in FORBIDDEN_DOLLAR.finditer(text):
        reasons.append(f"operador_proibido:${m.group(1)}")
    if re.search(r"\bmaxTimeMS\b|\bsetReadPref\b|\bwriteConcern\b", masked):
        reasons.append("opcao_de_execucao")
    if re.search(r"\bwhile\s*\(\s*(true|1)\s*\)|\bfor\s*\(\s*;\s*;\s*\)", masked):
        reasons.append("loop_infinito")
    if allowed_dbs is not None:
        for m in re.finditer(r"getSiblingDB\s*\(\s*([\"'`])(.*?)\1\s*\)", text):
            if m.group(2) not in allowed_dbs:
                reasons.append(f"getSiblingDB_fora_do_schema:{m.group(2)}")
        if re.search(r"getSiblingDB\s*\(\s*[^\"'`\s)]", text):
            reasons.append("getSiblingDB_dinamico")
    return sorted(set(reasons))


# ----------------------------------------------------------------------------------------------------------------------------
# normalizacao / hash
# ----------------------------------------------------------------------------------------------------------------------------
def strip_comments(text: str) -> str:
    toks, comments = jsscan.tokenize(text)
    chars = list(text)
    for a, b in comments:
        for k in range(a, b):
            chars[k] = " "
    return "".join(chars)


def normalized(text: str) -> str:
    """texto sem comentarios, aspas simples->duplas, sem espacos em volta de pontuacao, sem ; final: igualdade de CONSULTA."""
    t = strip_comments(text)
    toks, _ = jsscan.tokenize(t)
    out = []
    for kind, a, b in toks:
        s = t[a:b]
        if kind == "str" and s[0] == "'":
            s = '"' + s[1:-1].replace('"', '\\"') + '"'
        out.append(s)
    while out and out[-1] == ";":
        out.pop()
    return " ".join(out)


def shape(text: str) -> str:
    """forma estrutural: literais -> placeholders (numeros, strings que nao comecam com $, datas) e variaveis nao reservadas ficam;
    duas consultas que so diferem em valores tem a mesma forma."""
    t = strip_comments(text)
    toks, _ = jsscan.tokenize(t)
    out = []
    for i, (kind, a, b) in enumerate(toks):
        s = t[a:b]
        if kind == "num":
            s = "N"
        elif kind == "str":
            inner = s[1:-1]
            nxt = t[toks[i + 1][1]:toks[i + 1][2]] if i + 1 < len(toks) else ""
            s = '"' + inner + '"' if (nxt == ":" or inner.startswith("$")) else "S"
        elif kind in ("tpl", "regex"):
            s = "S"
        out.append(s)
    while out and out[-1] == ";":
        out.pop()
    return " ".join(out)


def query_key(text: str) -> str:
    return sha(normalized(text), 20)


def shape_key(text: str) -> str:
    return sha(shape(text), 20)


def read_jsonl(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path, rows) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path, rows) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
