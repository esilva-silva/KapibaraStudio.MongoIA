#!/usr/bin/env python3
"""Pipeline de validacao de 9 etapas (spec fase-03 §4.5) sobre fim_ctx.jsonl e chat.jsonl (+ snippets opcional).

  1 sanitize  2 syntax  3 api  4 fields  5 exec  6 semantic  7 budget  8 dedup  9 decontam

Reusa os validadores existentes (nao duplica): `chat_validate` (parser do Console, ops, campos, tipos, `$` faltando, escrita
executavel, node vm.Script), `gen_snippets.analyze_plan`, `validate_snippets` (execucao somente leitura em kapi_fx_*),
`render_contract` (prompt editor-context-v1 + tokenizer Qwen), `blind_guard` e `decontaminate` (etapa 9, o MESMO da Fase 06).

Saidas (por --out-dir, padrao data/validated/<ver>):
  fim.jsonl, chat.jsonl (+snippets.jsonl)  registros aprovados com bloco `validation` (cada etapa: ok|fail|skipped + motivo)
  rejected.jsonl                           {id, stage, reason, ...}
  manifest.json                            congelamento (sha256, contagens, hash de configs/dataset.yaml, commit git, seed)
  ../../splits/<ver>/{train,val,test,bench}.ids.txt
  reports/datasets/<ver>-validation-report.md  e  reports/decontam/<dataset-hash>.json

Execucao (etapa 5): `--exec recorded` usa o resultado gravado pelos geradores; `live` reexecuta no MongoDB local (somente leitura);
`auto` (padrao) = live se o servidor responder. Orcamento (etapa 7): tokenizer Qwen se disponivel (pacote `tokenizers`; tenta
tambem $KAPI_TOKENIZERS_SITE / venv ROCm do bake-off), senao aproximacao `chars/3` marcada como `approx`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blind_guard as bg  # noqa: E402
import chat_validate as cv  # noqa: E402
from jsonl_shards import ShardWriter, iter_jsonl, resolve_inputs  # noqa: E402
import decontaminate as dc  # noqa: E402
import render_contract as rc  # noqa: E402
import validate_snippets as vs  # noqa: E402
from gen_snippets import CONFIG_PATH, ROOT, analyze_plan, load_config, load_schemas  # noqa: E402

PIPELINE_VERSION = "validate-pipeline/1"
STAGES = ["sanitize", "syntax", "api", "fields", "exec", "semantic", "budget", "dedup", "decontam"]
EVAL = ("val", "test", "bench")
SPLITS = ("train", "val", "test", "bench")
WRITE4 = {"updateOne", "updateMany", "deleteOne", "deleteMany"}
INDEX_OPS = {"createIndex", "dropIndex"}
WRITE_OPS = WRITE4 | INDEX_OPS            # nunca executadas (exec_mode skipped_write)
STATIC_EXEC = ("skipped_static", "skipped_write")
UPDATE_OPS = {"$set", "$inc", "$unset", "$currentDate", "$push", "$addToSet", "$pull", "$mul", "$min", "$max", "$rename", "$pop",
              "$setOnInsert", "$each", "$pullAll", "$slice", "$position", "$sort"}
RESP_MAX = {"fim": 96, "chat": 256}
PROMPT_MAX = 2048
DEFAULT_TOK_SITE = r"C:\SlopStudio.MongoAI-artifacts\.venv\Lib\site-packages"


# =========================================================================== etapa 1: sanitizacao
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+\.)+[A-Za-z]{2,}")
_CPF = re.compile(r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)")
_CNPJ = re.compile(r"(?<!\d)\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}(?!\d)")
_CARD = re.compile(r"(?<![\w.])(?:[3-6]\d{12,18})(?![\w.])|(?<!\d)\d{4}([ -])\d{4}\1\d{4}\1\d{3,4}(?!\d)")
_TOKENS = re.compile(r"\b(?:sk|pk)-[A-Za-z0-9]{20,}|\bAKIA[0-9A-Z]{16}\b|\bghp_[A-Za-z0-9]{30,}|\bxox[baprs]-[A-Za-z0-9-]{10,}"
                     r"|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}")


def _luhn(s: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", s)]
    if not 13 <= len(d) <= 19:
        return False
    tot = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x *= 2
            if x > 9:
                x -= 9
        tot += x
    return tot % 10 == 0


def sanitize_text(text: str) -> str | None:
    """Motivo de rejeicao (padrao proibido) ou None. Regras v1: URIs/credenciais/tokens, CPF/CNPJ, cartao, e-mail so example.*,
    marcadores `<|`/`<｜` e cercas ```."""
    if not text:
        return None
    if rc.contains_sensitive(text):
        return "credencial/URI mongodb:// (padrao proibido)"
    if "<|" in text or "<｜" in text:
        return "marcador especial <| no texto"
    if "```" in text:
        return "cerca markdown ```"
    if _TOKENS.search(text):
        return "token/chave de API"
    if _CPF.search(text) or _CNPJ.search(text):
        return "CPF/CNPJ"
    for m in _CARD.finditer(text):
        if _luhn(m.group()):
            return "numero de cartao (Luhn)"
    for m in _EMAIL.finditer(text):
        dom = m.group().split("@", 1)[1].lower()
        if not re.match(r"^(?:[a-z0-9-]+\.)*example\.[a-z]+$", dom):
            return "e-mail fora de example.*"
    return None


def fim_texts(r: dict) -> list[str]:
    ed = r.get("editor") or {}
    out = [r.get("prefix", ""), r.get("middle", ""), r.get("suffix", ""), ed.get("input_panel") or ""]
    for k in ("known_names", "result_fields", "recent_commands"):
        out += [str(x) for x in (ed.get(k) or [])]
    return out


def chat_texts(r: dict) -> list[str]:
    c = r.get("context") or {}
    out = [r.get("instruction", ""), r.get("output", ""), c.get("editor") or "", c.get("selection") or ""]
    out += [str(c.get(k) or "") for k in ("connection", "database", "collection")]
    out += [str(x) for x in c.get("known_collections") or []]
    return out


# =========================================================================== analise de documento (etapas 2-4, 6)
class WriteParser(cv.Parser):
    """Parser do Console que tambem aceita updateOne/updateMany/deleteOne/deleteMany (so para validar texto; nunca executa)."""

    def call(self, b, args):
        o, name = b.obj, b.name
        if isinstance(o, cv.Coll) and name in INDEX_OPS:
            self.meta.update(collection=o.name, root_form=o.via, connection=o.conn, database=o.database)
            if name == "createIndex":
                if not args or len(args) > 2 or not isinstance(args[0], dict) or not args[0]                         or (len(args) == 2 and not isinstance(args[1], dict)):
                    raise cv.ChatParseError("api", "createIndex(chaves, opcoes)")
                plan = {"op": name, "keys": args[0]}
                if len(args) == 2 and args[1]:
                    plan["options"] = args[1]
                return cv.Cur(plan, terminal=True)
            if len(args) != 1 or not isinstance(args[0], (str, dict)) or not args[0]:
                raise cv.ChatParseError("api", "dropIndex(nome | chaves)")
            return cv.Cur({"op": name, "index": args[0]}, terminal=True)
        if isinstance(o, cv.Coll) and name in WRITE4:
            self.meta.update(collection=o.name, root_form=o.via, connection=o.conn, database=o.database)
            if not args or not isinstance(args[0], dict):
                raise cv.ChatParseError("api", f"{name} exige filtro (objeto)")
            if name.startswith("delete"):
                if len(args) > 2:
                    raise cv.ChatParseError("api", f"{name}(filtro)")
                return cv.Cur({"op": name, "filter": args[0]}, terminal=True)
            if len(args) < 2 or not isinstance(args[1], dict):
                raise cv.ChatParseError("api", f"{name}(filtro, atualizacao)")
            return cv.Cur({"op": name, "filter": args[0], "update": args[1]}, terminal=True)
        return super().call(b, args)


def parse_doc(text: str, dialect: str = "console") -> tuple[dict, dict]:
    if dialect == "json":
        return cv.parse_json_pipeline(text)
    return WriteParser(text).program()


def check_update_ops(plan: dict):
    if "update" in plan:
        for k, v in plan["update"].items():
            if k not in UPDATE_OPS:
                return "api", f"operador de update inexistente: {k}"
            if not isinstance(v, dict):
                return "api", f"{k} exige objeto"
    return None


def analyze_text(text: str, sch, coll, node_msg: str | None, *, allow_write: bool = True, expect_plan: dict | None = None,
                 dialect: str = "console") -> dict:
    """Etapas 2,3,4 e 6 (estatica) para um texto de codigo. Retorna {syntax,api,fields,semantic: (status, reason), plan}.
    `allow_write` (documentos FIM) tambem habilita Atlas Search (so estatico), createIndex/dropIndex e o dialeto `json`."""
    ok = ("ok", None)
    res = {"syntax": ok, "api": ok, "fields": ok, "semantic": ok, "plan": None}
    if node_msg:
        res["syntax"] = ("fail", node_msg)
        return res
    try:
        plan, meta = parse_doc(text, dialect) if allow_write else cv.parse_console(text)
    except cv.ChatParseError as e:
        key = "syntax" if e.stage == "syntax" else "api"
        res[key] = ("fail", e.msg)
        return res
    except Exception as e:  # defensivo
        res["syntax"] = ("fail", f"{type(e).__name__}: {e}")
        return res
    res["plan"] = plan
    if dialect == "json":
        meta["collection"] = coll.name           # o modo Agregacao nao carrega a colecao no texto (vem do contexto)
        if re.search(r"\$(?:out|merge)\b", text):
            res["api"] = ("fail", "$out/$merge fora da allowlist")
            return res
    bad = cv.check_ops(plan, allow_atlas=allow_write) or check_update_ops(plan) or cv.check_atlas(plan)
    if bad:
        res["api"] = ("fail", bad[1])
        return res
    if dialect != "json" and cv.WRITE_RE.search(re.sub(r"//.*", "", text)) and plan["op"] not in WRITE_OPS:
        res["api"] = ("fail", "escrita/DDL/$out/$merge fora da allowlist")
        return res
    if meta.get("collection") != coll.name:
        res["fields"] = ("fail", f"colecao {meta.get('collection')!r} nao e {coll.name!r}")
        return res
    if meta.get("database") is not None and meta.get("database") != sch.schema_id:
        res["fields"] = ("fail", f"banco {meta.get('database')!r} desconhecido")
        return res
    try:
        an = analyze_plan(plan, sch, coll)
    except Exception as e:
        res["fields"] = ("fail", f"plano malformado: {type(e).__name__}: {e}")
        return res
    if an["missing"]:
        res["fields"] = ("fail", "campo/estrutura invalido: " + str(an["missing"][0]))
        return res
    chk = None if plan["op"] in INDEX_OPS else (cv.check_types(plan, coll) or cv.check_dollar(plan, coll))
    if chk:
        res["fields"] = ("fail", chk[1])
        return res
    res["fields_used"] = an["used"]
    if expect_plan is not None and not cv.plans_equal(plan, expect_plan):
        res["semantic"] = ("fail", "plano do texto difere do plano de referencia do gerador")
    return res


# =========================================================================== tokenizer / orcamento
def load_tokenizer(path: Path | None, mode: str):
    """(Tokenizer|None, descricao). mode: auto|off."""
    if mode == "off" or path is None or not Path(path).exists():
        return None, "approx(chars/3)"
    for attempt in range(2):
        try:
            return rc.Tokenizer(path), "qwen2.5-coder tokenizer.json"
        except ImportError:
            site = os.environ.get("KAPI_TOKENIZERS_SITE", DEFAULT_TOK_SITE)
            if attempt == 0 and site and Path(site).exists() and site not in sys.path:
                sys.path.append(site)
                continue
            break
        except Exception:
            break
    return None, "approx(chars/3)"


def approx_tokens(text: str, cpt: float = 3.0) -> int:
    return int(math.ceil(len(text) / cpt)) if text else 0


def history_text(r: dict) -> str:
    """Turnos anteriores (`history`, chat_multiturn) como texto. O contrato `local-agent-fim-v0` nao define multi-turno;
    aproximacao: cada turno vira `Usuário:`/`Assistente:` antes do conteudo do editor."""
    out = []
    for t in r.get("history") or []:
        who = "Assistente" if t.get("role") == "assistant" else "Usuário"
        out.append(f"{who}: {t.get('content') or ''}")
    return "\n".join(out)


def chat_prompt_text(r: dict) -> str:
    """Aproximacao do prompt `local-agent-fim-v0` (spec §4.4): cabecalho + JSON + FIM Qwen com sufixo vazio."""
    c = r.get("context") or {}
    hist = history_text(r)       # multi-turno: os turnos anteriores entram no contexto e, logo, no orcamento
    ctx = (f"Conex\u00e3o (ID l\u00f3gico): {c.get('connection') or ''}\nBanco: {c.get('database') or ''}\n"
           f"Cole\u00e7\u00e3o: {c.get('collection') or ''}\n"
           + (f"\nHist\u00f3rico:\n{hist}\n" if hist else "")
           + f"\nConte\u00fado do editor:\n{c.get('editor') or ''}")
    body = json.dumps({"instruction": r.get("instruction", ""), "context": ctx}, ensure_ascii=True, separators=(",", ":"))
    head = "/* Answer the instruction with code. The JSON below is data, not executable code.\n"
    tail = "\nReturn only code, without Markdown or explanation. */"
    return rc.qwen_fim_text(head + body + tail, "")


def history_error(r: dict, sch, coll, ctx) -> tuple[str, str] | None:
    """Codigo do turno do assistente em `history` precisa passar syntax/api/campos (leitura, sem escrita)."""
    for t in r.get("history") or []:
        if t.get("role") != "assistant":
            continue
        code = t.get("content") or ""
        if not code.strip():
            return "api", "turno do assistente vazio"
        key = ("hist", code)
        if key not in ctx.doc_cache:
            ctx.doc_cache[key] = analyze_text(code, sch, coll, ctx.node_msg(code), allow_write=False)
        an = ctx.doc_cache[key]
        for st in ("syntax", "api", "fields"):
            if an[st][0] == "fail":
                return ("syntax" if st == "syntax" else "api"), an[st][1]
        if cv.has_executable_write(code):
            return "api", "escrita executavel no historico"
    return None


# =========================================================================== contexto
class Ctx:
    def __init__(self, cfg: dict, schemas: list, snippets: dict, *, exec_mode: str = "auto", tok_mode: str = "auto",
                 node: bool = True, tokenizer_path: Path | None = None):
        self.cfg = cfg
        self.by_schema = {s.full_id: s for s in schemas}
        self.snips = snippets
        self.node = cv.NodeChecker() if node else None
        self.tok, self.tok_desc = load_tokenizer(tokenizer_path, tok_mode)
        self.executor = None
        self.exec_mode = exec_mode
        if exec_mode in ("auto", "live"):
            try:
                self.executor = cv.Executor(cfg)
                if not self.executor.existing:
                    raise RuntimeError("sem bancos kapi_fx_*")
                self.exec_mode = "live"
            except BaseException as e:  # servidor ausente
                if exec_mode == "live":
                    raise
                self.executor = None
                self.exec_mode = "recorded"
        self.doc_cache: dict = {}
        self.exec_cache: dict = {}
        self.node_cache: dict = {}
        self.sanitize_cache: dict = {}

    def coll(self, schema_id: str, name: str):
        s = self.by_schema.get(schema_id)
        if not s:
            return None, None
        for c in s.colls:
            if c.name == name or c.key == name:
                return s, c
        return s, None

    def node_msg(self, text: str):
        if self.node is None:
            return None
        if text not in self.node_cache:
            self.node_cache[text] = self.node.check(text)
        return self.node_cache[text]

    def close(self):
        if self.node:
            self.node.close()


_ST_CACHE: dict = {}


def _st(status, reason=None):
    """Dicts de status compartilhados (imutaveis por convencao): economiza memoria com ~1,5 M de registros."""
    k = (status, reason)
    v = _ST_CACHE.get(k)
    if v is None:
        v = {"status": status, "reason": reason}
        if len(_ST_CACHE) < 4096:
            _ST_CACHE[k] = v
    return v


# =========================================================================== registros
class Entry:
    __slots__ = ("id", "type", "split", "kind", "domain", "holdout", "group_id", "doc", "val", "failed", "lite", "extra")

    def __init__(self, r: dict, typ: str):
        self.id, self.type, self.split = r["id"], typ, r.get("split")
        self.kind = r.get("kind") if typ == "chat" else (r.get("cut_kind") or r.get("kind"))
        self.domain, self.holdout, self.group_id = r.get("domain"), bool(r.get("holdout")), r.get("group_id") or r.get("group")
        self.doc = r.get("snippet_id") if typ == "fim" else r["id"]
        self.val: dict = {}
        self.failed: tuple | None = None
        self.lite: dict = {}
        self.extra: dict = {}


def fail(e: Entry, stage: str, reason: str):
    e.val[stage] = _st("fail", reason)
    e.failed = (stage, reason)


def stages_1_to_7(r: dict, typ: str, ctx: Ctx) -> Entry:
    e = Entry(r, typ)
    # --- 1 sanitize
    texts = fim_texts(r) if typ == "fim" else chat_texts(r)
    why = None
    for t in texts:
        why = sanitize_text(t)
        if why:
            break
    if why:
        fail(e, "sanitize", why)
        return e
    e.val["sanitize"] = _st("ok")
    return _stages_2_7(e, r, typ, ctx)


def _stages_2_7(e: Entry, r: dict, typ: str, ctx: Ctx) -> Entry:
    sch, coll = ctx.coll(r.get("schema_id"), r.get("collection"))
    if sch is None or coll is None:
        fail(e, "fields", f"schema/colecao desconhecidos: {r.get('schema_id')}/{r.get('collection')}")
        e.val["syntax"] = _st("skipped", "schema desconhecido")
        e.val["api"] = _st("skipped", "schema desconhecido")
        return e
    snip = ctx.snips.get(r.get("snippet_id")) if r.get("snippet_id") else None
    if typ == "fim":
        text = r["prefix"] + r["middle"] + r["suffix"]
        if snip is not None and snip["text"] != text:
            fail(e, "syntax", "documento reconstruido (prefixo+resposta+sufixo) difere do snippet")
            return e
        dialect = (snip or {}).get("dialect") or r.get("dialect") or "console"
        key = ("fim", dialect, text)
        if key not in ctx.doc_cache:
            ctx.doc_cache[key] = analyze_text(text, sch, coll, ctx.node_msg(text), allow_write=True,
                                              expect_plan=snip["plan"] if snip else None, dialect=dialect)
        an = ctx.doc_cache[key]
        has_code = True
        e.extra["dialect"] = dialect
        e.extra["exec_mode"] = (snip or {}).get("exec_mode") or (r.get("exec") if r.get("exec") in STATIC_EXEC else None)
    else:
        text = r.get("output") or ""
        code = "\n".join(l for l in text.splitlines() if not l.strip().startswith("//")).strip()
        has_code = bool(code)
        if has_code:
            key = ("chat", text)
            if key not in ctx.doc_cache:
                ctx.doc_cache[key] = analyze_text(text, sch, coll, ctx.node_msg(text), allow_write=False,
                                                  expect_plan=snip["plan"] if snip and r.get("kind") != "negative" else None)
            an = ctx.doc_cache[key]
        else:
            an = {"syntax": ("ok", None), "api": ("ok", None), "fields": ("ok", None), "semantic": ("ok", None), "plan": None}
            msg = ctx.node_msg(text)
            if msg:
                an["syntax"] = ("fail", msg)
        if cv.has_executable_write(text):
            an = dict(an)
            an["api"] = ("fail", "escrita executavel no output")
        if an["api"][0] == "ok" and r.get("kind") == "chat_multiturn":
            why = history_error(r, sch, coll, ctx)
            if why:
                an = dict(an)
                an["syntax" if why[0] == "syntax" else "api"] = ("fail", "historico: " + why[1])
    e.lite["has_code"] = has_code
    # --- 2 syntax
    if an["syntax"][0] == "fail":
        fail(e, "syntax", an["syntax"][1])
        return e
    e.val["syntax"] = _st("ok")
    # --- 3 api
    if an["api"][0] == "fail":
        fail(e, "api", an["api"][1])
        return e
    e.val["api"] = _st("ok") if has_code else _st("skipped", "sem codigo (resposta so comentario)")
    # --- 4 fields
    if an["fields"][0] == "fail":
        fail(e, "fields", an["fields"][1])
        return e
    e.val["fields"] = _st("ok") if has_code else _st("skipped", "sem codigo (resposta so comentario)")
    e.extra["fields_used"] = an.get("fields_used")
    # --- 5 exec
    plan = an["plan"]
    is_write = bool(plan and plan["op"] in WRITE_OPS)
    # defesa em profundidade: busca Atlas nunca vai ao driver, mesmo sem `exec_mode` gravado (o driver tambem a bloqueia)
    is_static = bool(plan and plan["op"] == "aggregate" and any(
        isinstance(st, dict) and (set(st) & cv.ATLAS_STAGES) for st in plan.get("pipeline") or []))         or e.extra.get("exec_mode") == "skipped_static"
    if not has_code:
        e.val["exec"] = _st("skipped", "sem codigo")
        n_docs = None
    elif is_write or e.extra.get("exec_mode") == "skipped_write":
        e.val["exec"] = _st("skipped", "skipped_write: escrita/DDL nunca executada")
        n_docs = None
    elif is_static:
        e.val["exec"] = _st("skipped", "skipped_static: Atlas Search so tem validacao estatica")
        n_docs = None
    else:
        gv = r.get("validation") or {}
        if ctx.exec_mode == "live" and plan is not None:
            db = (snip or {}).get("db") or gv.get("fixture") or ""
            ck = (db, r["collection"], json.dumps(plan, sort_keys=True, default=str))
            if ck not in ctx.exec_cache:
                ctx.exec_cache[ck] = ctx.executor.run(db, r["collection"], plan)
            ex = ctx.exec_cache[ck]
            ok, n_docs = ex["exec"] == "ok", ex.get("n_docs")
            why = ex.get("error") or ex.get("reason") or ex["exec"]
            src = "live"
        else:
            rec_ok = (r.get("exec") == "ok") if typ == "fim" else (gv.get("exec") == "ok")
            ok, n_docs = rec_ok, (None if gv.get("nonempty") is None else (1 if gv.get("nonempty") else 0))
            why = f"exec gravado pelo gerador = {r.get('exec') if typ == 'fim' else gv.get('exec')}"
            src = "recorded"
        if not ok:
            fail(e, "exec", why)
            return e
        e.val["exec"] = _st("ok", f"{src}")
    e.extra["n_docs"] = n_docs
    # --- 6 semantic
    if typ == "fim":
        if not r["prefix"].strip() and not r["suffix"].strip():
            fail(e, "semantic", "sem codigo antes e depois do cursor")
            return e
        if not r["middle"].strip():
            fail(e, "semantic", "resposta vazia")
            return e
        if an["semantic"][0] == "fail":
            fail(e, "semantic", an["semantic"][1])
            return e
        e.val["semantic"] = _st("ok", "documento reconstruido == plano do snippet")
    else:
        k = r.get("kind")
        gv = r.get("validation") or {}
        if k == "negative":
            if cv.has_executable_write(text):
                fail(e, "semantic", "escrita executavel")
                return e
            e.val["semantic"] = _st("ok", "negativo: sem escrita executavel")
        else:
            if gv.get("plan_equal") is False:
                fail(e, "semantic", "plan_equal=false gravado pelo gerador")
                return e
            if an["semantic"][0] == "fail":
                fail(e, "semantic", an["semantic"][1])
                return e
            if k == "correction":
                ed = (r.get("context") or {}).get("editor") or ""
                if ed.strip() == (r.get("output") or "").strip():
                    fail(e, "semantic", "correcao identica ao editor")
                    return e
            if ctx.exec_mode == "live" and plan is not None and plan["op"] not in WRITE_OPS and n_docs == 0 and gv.get("nonempty"):
                fail(e, "semantic", "referencia retorna vazio (nao discriminativa)")
                return e
            if gv.get("nonempty") is False and k in ("chat", "chat_multiturn", "explain", "correction"):
                fail(e, "semantic", "referencia retorna vazio (nao discriminativa)")
                return e
            e.val["semantic"] = _st("ok", "plano do texto == plano do snippet; resultado nao vazio")
    # --- 7 budget
    tok = ctx.tok
    if typ == "fim":
        if not r.get("prompt_truncated"):
            want = rc.render_prompt(r)["prompt"]
            if want != r.get("prompt"):
                fail(e, "budget", "prompt gravado difere do renderer editor-context-v1")
                return e
        pt = r.get("prompt_tokens")
        if tok is not None:
            pt = len(tok.encode(r["prompt"]))
            if r.get("prompt_tokens") is not None and pt != r["prompt_tokens"]:
                fail(e, "budget", f"prompt_tokens divergente (gravado {r['prompt_tokens']}, tokenizer {pt})")
                return e
            rt = len(tok.encode(r["middle"]))
            src = "tokenizer"
        else:
            rt = approx_tokens(r["middle"])
            src = "approx" if pt is None else "recorded-prompt+approx-response"
    else:
        ptxt = chat_prompt_text(r)
        if tok is not None:
            pt, rt, src = len(tok.encode(ptxt)), len(tok.encode(r.get("output") or "")), "tokenizer(prompt aproximado)"
        else:
            pt, rt, src = approx_tokens(ptxt), approx_tokens(r.get("output") or ""), "approx"
    limit = RESP_MAX[typ]
    if pt is not None and pt > PROMPT_MAX:
        fail(e, "budget", f"prompt com {pt} tokens > {PROMPT_MAX}")
        return e
    if rt > limit:
        fail(e, "budget", f"resposta com {rt} tokens > {limit}")
        return e
    e.val["budget"] = {"status": "ok", "reason": None, "prompt_tokens": pt, "response_tokens": rt, "source": src}
    return e


# =========================================================================== etapas 8 e 9
def dedup_key(r: dict, typ: str) -> str:
    if typ == "fim":
        parts = [" ".join(dc.code_tokens(r["prefix"], "keep")), " ".join(dc.code_tokens(r["middle"], "keep")),
                 " ".join(dc.code_tokens(r["suffix"], "keep"))]
    else:
        c = r.get("context") or {}
        parts = [" ".join(dc.instr_tokens(r.get("instruction", ""), "keep")), " ".join(dc.code_tokens(r.get("output") or "", "keep")),
                 " ".join(dc.code_tokens(c.get("editor") or "", "keep"))]
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()


def stage_dedup(entries: list[Entry], lite: dict, seed: int, thr: float) -> dict:
    """Exato por split (hash normalizado, mantem o primeiro) + near-dup MinHash >= thr dentro do treino (decisao por documento)."""
    stats = Counter()
    seen: dict = {}
    mh = dc.MinHasher(dc.NUM_PERM, seed)
    lsh = {"fim": dc.Lsh(thr), "chat": dc.Lsh(thr)}
    doc_dropped: dict = {}
    for e in entries:
        if e.failed:
            continue
        r = lite[e.id]
        k = (e.type, e.split, dedup_key(r, e.type))
        if k in seen:
            fail(e, "dedup", f"duplicata exata de {seen[k]} (hash normalizado, split {e.split})")
            stats["exact"] += 1
            continue
        seen[k] = e.id
    for e in entries:
        if e.failed or e.split != "train":
            continue
        dk = (e.type, e.doc)
        if dk in doc_dropped:
            fail(e, "dedup", doc_dropped[dk])
            stats["near_dup"] += 1
            continue
        r = lite[e.id]
        if e.type == "fim":
            toks = dc.code_tokens(r["prefix"] + r["middle"] + r["suffix"], "keep")
        else:
            c = r.get("context") or {}
            toks = dc.instr_tokens(r.get("instruction", ""), "keep") + ["\x00"] + dc.code_tokens(r.get("output") or "", "keep") \
                + ["\x00"] + dc.code_tokens(c.get("editor") or "", "keep")
        sig = mh.signature(toks)
        if sig is None:
            continue
        key = (e.type, e.doc)
        if key in lsh[e.type].sigs:
            continue                      # outro corte do mesmo documento ja mantido
        hit = lsh[e.type].query(sig, limit=1)
        if hit:
            msg = f"near-dup MinHash J~{hit[0][1]:.2f} >= {thr} de {hit[0][0][1]}"
            fail(e, "dedup", msg)
            stats["near_dup"] += 1
            if e.type == "fim":
                doc_dropped[dk] = msg
        else:
            lsh[e.type].add(key, sig)
    return dict(stats)


def lite_record(r: dict, typ: str) -> dict:
    if typ == "fim":
        return {"id": r["id"], "split": r["split"], "snippet_id": r.get("snippet_id"), "collection": r.get("collection"),
                "prefix": r["prefix"], "middle": r["middle"], "suffix": r["suffix"], "type": "fim", "domain": r.get("domain"),
                "holdout": r.get("holdout"), "cut_kind": r.get("cut_kind"), "kind": r.get("kind"), "group_id": r.get("group_id"),
                "schema_id": r.get("schema_id"), "cursor_role": r.get("cursor_role"), "category": r.get("category"),
                "dialect": r.get("dialect")}
    c = r.get("context") or {}
    return {"id": r["id"], "split": r["split"], "snippet_id": r.get("snippet_id"), "collection": r.get("collection"),
            "instruction": r.get("instruction", ""), "output": r.get("output", ""), "context": {"editor": c.get("editor")},
            "type": "chat", "kind": r.get("kind"), "domain": r.get("domain"), "holdout": r.get("holdout"),
            "group_id": r.get("group_id"), "language": r.get("language"), "schema_id": r.get("schema_id")}


def read_jsonl(path):
    """Gera registros de um arquivo ou de shards `x-00000.jsonl` (streaming)."""
    files = resolve_inputs(path)
    for p in files:
        bg.check_path(p)
    yield from iter_jsonl(files)


# =========================================================================== relatorio / gates
def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


OP_RE = re.compile(r"\$[A-Za-z][A-Za-z0-9]*")


MIX_COMPONENT = {"chat": "chat", "chat_multiturn": "chat", "explain": "chat", "correction": "correction", "negative": "negative"}


def mixture_component(kind) -> str:
    """Componente da mistura (`chat.mixture`) de um `kind` de chat; multi-turno e explain contam como `chat`."""
    return MIX_COMPONENT.get(kind, "chat")


def compute_gates(kept: list[Entry], lite: dict, cfg: dict, snips: dict, final_leak: dict, ctx: Ctx, entries_all: list[Entry],
                  decon: dict, out_split_ids: dict) -> dict:
    g: dict = {}
    train = [e for e in kept if e.split == "train"]
    fim_t = [e for e in train if e.type == "fim"]
    chat_t = [e for e in train if e.type == "chat"]
    # G3.2 validade
    def status(e, s):
        return e.val.get(s, {}).get("status")
    tot = len(kept)
    g["G3.2"] = {"records": tot, "all_static_ok": sum(1 for e in kept if all(status(e, s) in ("ok", "skipped") for s in STAGES[:7])),
                 "exec_ok": sum(1 for e in kept if status(e, "exec") == "ok"),
                 "exec_skipped": sum(1 for e in kept if status(e, "exec") == "skipped"),
                 "exec_valid_rate_reads": None}
    inp_reads = [e for e in entries_all if e.val.get("exec", {}).get("status") in ("ok", "fail")]
    g["G3.2"]["exec_valid_rate_reads"] = (sum(1 for e in inp_reads if e.val["exec"]["status"] == "ok") / len(inp_reads)) if inp_reads else None
    g["G3.2"]["pass"] = g["G3.2"]["all_static_ok"] == tot and all(status(e, "exec") != "fail" for e in kept)
    # G3.3 volume e distribuicao
    targets = (cfg.get("fim") or {}).get("cut_targets") or {}
    dist = Counter(e.kind for e in fim_t)
    n = len(fim_t)
    cut = {k: {"target": t, "actual": round(dist.get(k, 0) / n, 4) if n else None,
               "ok": (abs(dist.get(k, 0) / n - t) <= 0.02) if n else False} for k, t in targets.items()}
    lang = Counter(lite[e.id].get("language") for e in chat_t)
    nl = len(chat_t)
    langd = {k: {"target": 0.5, "actual": round(v / nl, 4), "ok": abs(v / nl - 0.5) <= 0.02} for k, v in lang.items()} if nl else {}
    ops = Counter()
    for e in fim_t:
        pass
    cats = Counter(lite[e.id].get("category") or "?" for e in fim_t)
    cat_t = (cfg.get("fim") or {}).get("category_targets") or {}
    floor = float((cfg.get("fim") or {}).get("category_floor", 0.04))
    cat_tot = sum(cats.values()) or 1
    # alvo renormalizado sobre as categorias presentes (o piso 4% ja esta embutido em `category_targets`)
    tsum = sum(cat_t.values()) or 1.0
    catd = {c: {"target": round(cat_t.get(c, 0.0) / tsum, 4), "actual": round(cats.get(c, 0) / cat_tot, 4),
                "ok": abs(cats.get(c, 0) / cat_tot - cat_t.get(c, 0.0) / tsum) <= 0.02, "floor_ok": cats.get(c, 0) / cat_tot >= floor}
            for c in sorted(set(cat_t) | set(cats))}
    dialects = Counter(lite[e.id].get("dialect") or "console" for e in fim_t)
    spec_cats = ["aggregation", "crud", "operators", "atlas_search", "json_ejson", "dsl", "indexes"]
    g["G3.3"] = {"train_validated": len(train), "train_fim": len(fim_t), "train_chat": len(chat_t), "required_min": 1_000_000,
                 "volume_ok": len(train) >= 1_000_000,
                 "cut_kind": cut, "cut_kind_ok": all(v["ok"] for v in cut.values()) if cut else None, "language": langd,
                 "language_ok": all(v["ok"] for v in langd.values()) if langd else None,
                 "dialects": dict(dialects), "categories": catd,
                 "categories_ok": all(v["ok"] for v in catd.values()) if catd else None,
                 "floors_ok": all(catd.get(c, {}).get("floor_ok", False) for c in ("atlas_search", "json_ejson", "dsl", "indexes")),
                 "spec_categories_missing": [c for c in spec_cats if not cats.get(c)],
                 "pass": False}
    g["G3.3"]["pass"] = bool(g["G3.3"]["volume_ok"] and g["G3.3"]["cut_kind_ok"] and g["G3.3"]["language_ok"]
                             and g["G3.3"]["categories_ok"] and g["G3.3"]["floors_ok"])
    # G3.4 vazamento
    gsplit: dict = defaultdict(set)
    for e in kept:
        if e.group_id:
            gsplit[e.group_id].add(e.split)
    shared = {k: sorted(v) for k, v in gsplit.items() if len(v) > 1}
    g["G3.4"] = {"groups_shared_across_splits": len(shared), "examples": list(shared)[:10],
                 "train_x_eval_hits_after_decontam": final_leak["train_removed"], "blind": decon["blind"],
                 "legacy_checked": decon["eval_docs_by_source"].get("legacy-v1", 0),
                 "pass": len(shared) == 0 and final_leak["train_removed"] == 0}
    # G3.5 privacidade (reescaneia os mantidos)
    bad = 0
    for e in kept:
        r = lite[e.id]
        tx = [r.get("prefix", ""), r.get("middle", ""), r.get("suffix", "")] if e.type == "fim" else [r.get("instruction", ""), r.get("output", "")]
        if any(sanitize_text(t) for t in tx):
            bad += 1
    g["G3.5"] = {"occurrences_in_output": bad, "pass": bad == 0}
    # mistura
    comp = Counter()
    for e in train:
        if e.type == "fim":
            comp["autocomplete"] += 1
        else:
            comp[mixture_component(e.kind)] += 1
    mix = (cfg.get("chat") or {}).get("mixture") or {}
    nmax = min((comp.get(k, 0) / v for k, v in mix.items() if v), default=0)
    binding = min(mix, key=lambda k: comp.get(k, 0) / mix[k]) if mix else None
    ntrain = len(train)
    pool_ratio = (comp.get("chat", 0) + comp.get("correction", 0)) / ntrain if ntrain else 0
    tgt_ratio = mix.get("chat", 0) + mix.get("correction", 0)
    g["mixture"] = {"pool": dict(comp), "target": mix, "pool_chat_plus_correction_ratio": round(pool_ratio, 4),
                    "target_chat_plus_correction_ratio": round(tgt_ratio, 4), "limit": 0.25,
                    "max_total_without_repetition": int(nmax), "binding_component": binding,
                    "pass": pool_ratio <= 0.25 and tgt_ratio <= 0.25}
    # grupos pequenos (piso 4%)
    kinds = Counter(lite[e.id].get("cut_kind") for e in fim_t)
    small = {k: round(v / len(fim_t), 4) for k, v in kinds.items()} if fim_t else {}
    g["floors"] = {"applicable_groups": ["indexes", "json_ejson", "dsl", "atlas_search"],
                   "category_share": {c: v["actual"] for c, v in catd.items()},
                   "below_4pct": [c for c, v in catd.items() if v["actual"] < 0.04],
                   "cut_kind_share": small, "cut_kind_below_4pct": [k for k, v in small.items() if v < 0.04],
                   "pass": g["G3.3"]["floors_ok"]}
    # cobertura
    cov_dom = {s: Counter(e.domain for e in kept if e.split == s) for s in SPLITS}
    opc = Counter()
    seen_docs = set()
    for e in train:
        r = lite[e.id]
        key = (e.type, e.doc)
        if key in seen_docs:
            continue
        seen_docs.add(key)
        code = (r["prefix"] + r["middle"] + r["suffix"]) if e.type == "fim" else r.get("output", "")
        for o in set(OP_RE.findall(code)):
            if o in dc._OPS:                      # `$campo` (referencia a campo) nao e operador
                opc[o] += 1
    g["coverage"] = {"domains_train": dict(cov_dom["train"]), "domains_val": dict(cov_dom["val"]), "domains_test": dict(cov_dom["test"]),
                     "domains_bench": dict(cov_dom["bench"]), "operators_train_docs": dict(opc.most_common()),
                     "operators_rare_lt20": sorted(o for o, c in opc.items() if c < 20)}
    # G2.7 / holdout / escritas
    ids = {s: set(out_split_ids[s]) for s in SPLITS}
    inter = {f"{a}&{b}": len(ids[a] & ids[b]) for a in SPLITS for b in SPLITS if a < b}
    train_bad = [e.id for e in train if e.split != "train" or e.holdout]
    hold_bad = [e.id for e in kept if e.holdout and e.split in ("train", "val")]
    g["G2.7"] = {"eval_ids_in_train": len(ids["train"] & (ids["val"] | ids["test"] | ids["bench"])), "split_intersections": inter,
                 "holdout_in_train_or_val": len(hold_bad), "train_records_with_holdout_flag": len(train_bad),
                 "pass": len(ids["train"] & (ids["val"] | ids["test"] | ids["bench"])) == 0 and not hold_bad and not train_bad}
    chat_w = sum(1 for e in kept if e.type == "chat" and cv.has_executable_write(lite[e.id].get("output", "")))
    fim_w = sum(1 for e in kept if e.type == "fim" and status(e, "exec") == "skipped"
                and str(e.val["exec"]["reason"]).startswith(("escrita", "skipped_write")))
    viol = getattr(ctx, "violations_total", None)
    if viol is None:
        viol = len(ctx.executor.violations) if ctx.executor else None
    g["writes"] = {"chat_outputs_with_executable_write": chat_w, "fim_docs_that_are_writes_not_executed": fim_w,
                   "server_write_commands_seen": viol, "pass": chat_w == 0 and (viol in (None, 0))}
    return g


def gates_g3(info: dict) -> dict:
    return info["gates"]["G3.3"]


def render_report(path: Path, ver: str, info: dict):
    L: list[str] = []
    A = L.append
    A(f"# Dataset {ver} - relatorio de validacao (pipeline de 9 etapas)")
    A("")
    A(f"Gerado por `scripts/data/validate_pipeline.py` (`{PIPELINE_VERSION}`), seed {info['seed']}, commit `{info['commit'][:12]}`"
      f"{' (arvore com alteracoes)' if info['dirty'] else ''}. Execucao (etapa 5): `{info['exec_mode']}`; orcamento (etapa 7): "
      f"`{info['tok_desc']}`; sintaxe JS: `{'node vm.Script' if info['node'] else 'parser interno (node ausente)'}`.")
    A("")
    A("## 1. Funil por etapa")
    A("")
    A("| etapa | entrada | aprovados | rejeitados | fim (ent->saida) | chat (ent->saida) |")
    A("|---|---:|---:|---:|---|---|")
    for s in info["funnel"]:
        A(f"| {s['stage']} | {s['in']} | {s['out']} | {s['rejected']} | {s['fim_in']}->{s['fim_out']} | {s['chat_in']}->{s['chat_out']} |")
    A("")
    A(f"Entrada: {info['n_in']} registros ({info['n_in_fim']} fim_ctx + {info['n_in_chat']} chat). Saida: {info['n_out']} "
      f"(train {info['final']['train']}, val {info['final']['val']}, test {info['final']['test']}, bench {info['final']['bench']}).")
    A("")
    A("### Principais motivos de rejeicao")
    A("")
    A("| etapa | motivo | n |")
    A("|---|---|---:|")
    for (st, why), n in info["reasons"][:30]:
        A(f"| {st} | {why[:110]} | {n} |")
    if not info["reasons"]:
        A("| - | nenhum | 0 |")
    A("")
    A("## 2. Decontaminacao (etapa 9)")
    A("")
    d = info["decon"]
    A(f"- Treino de entrada: {d['train_in']} registros; removidos: **{d['train_removed']}** ({pct(d['train_removed'], d['train_in'])}); mantidos {d['train_kept']}.")
    A(f"- Documentos de avaliacao indexados: {d['eval_docs']} {d['eval_docs_by_source']}. Conjunto cego: **{d['blind']}**.")
    A("")
    A("| regra | removidos (qualquer coincidencia) | removidos (1a regra) |")
    A("|---|---:|---:|")
    for r in dc.RULES:
        A(f"| {r} | {d['by_rule'][r]} | {d['by_rule_first_match'][r]} |")
    A("")
    sv = d["ngram13_sensitivity_min_content"]
    A(f"Sensibilidade do 13-grama (registros de treino removidos se o 13-grama exigisse >= k tokens de conteudo): "
      + ", ".join(f"k={k}: {v}" for k, v in sv.items()) + f" (APLICADO: k={d['params']['ngram_min_content']}; 0 = regra literal da spec).")
    A("")
    A(f"Parametros: {d['params']}. Relatorio completo: `reports/decontam/{info['dataset_hash'][:16]}.json`. Verificacao pos-remocao "
      f"(2a passada treino x avaliacao): {info['final_leak']} registros ainda coincidentes.")
    A("")
    A("### Sobrevivencia por cut_kind (train FIM, regra oficial) - base para `fim.decontam_survival`")
    A("")
    A("| cut_kind | entrada (etapas 1-8) | mantidos | sobrevivencia |")
    A("|---|---:|---:|---:|")
    for k, v in info["survival"].items():
        A(f"| {k} | {info['survival_in'].get(k, 0)} | {info['survival_out'].get(k, 0)} | {v:.3f} |")
    A("")
    A("### Sobrevivencia por categoria (train FIM, regra oficial) - base para `fim.category_survival`")
    A("")
    A("| categoria | entrada (etapas 1-8) | mantidos | sobrevivencia |")
    A("|---|---:|---:|---:|")
    for k, v in info.get("survival_cat", {}).items():
        A(f"| {k} | {info['survival_cat_in'].get(k, 0)} | {info['survival_cat_out'].get(k, 0)} | {v:.3f} |")
    A("")
    an = d.get("analysis") or {}
    for k, a in an.items():
        tin = d["train_in"]
        mc = d["params"]["ngram_min_content"]
        what = "regra LITERAL da spec (qualquer 13-grama comum)" if int(k) == 0 else f"13-grama com >= {k} tokens de conteudo"
        A(f"### Auditoria paralela (NAO aplicada ao treino): {what}")
        A("")
        A(f"Removeria {a['removed']} de {tin} registros de treino ({pct(a['removed'], tin)}) contra {d['train_removed']} "
          f"({pct(d['train_removed'], tin)}) na regra OFICIAL (13-grama com >= {mc} tokens de conteudo); mantidos {a['kept']} contra "
          f"{d['train_kept']}. A diferenca e estrutura da linguagem (sem conteudo de schema); os schemas sao disjuntos por split "
          f"(`split_role`), entao nao ha vazamento de conteudo. exact, approx_key, MinHash e blind valem nas duas.")
        A("")
        A("| cut_kind | entrada | removidos (auditoria) | sobrevivencia (auditoria) | sobrevivencia (oficial) |")
        A("|---|---:|---:|---:|---:|")
        for ck, nin in sorted(a["in_by_cut_kind"].items()):
            rem = a["removed_by_cut_kind"].get(ck, 0)
            lit = info["survival"].get(ck)
            A(f"| {ck} | {nin} | {rem} | {1 - rem / nin:.3f} | {'-' if lit is None else format(lit, '.3f')} |")
        A("")
    A("## 3. Por split, tipo e domino")
    A("")
    A("| split | tipo | entrada | saida | rejeitados |")
    A("|---|---|---:|---:|---:|")
    for (sp, ty), v in sorted(info["by_split_type"].items()):
        A(f"| {sp} | {ty} | {v['in']} | {v['out']} | {v['in'] - v['out']} |")
    A("")
    A("| dominio | holdout | entrada | saida | train | val | test | bench |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for dm, v in sorted(info["by_domain"].items()):
        A(f"| {dm} | {'sim' if v['holdout'] else 'nao'} | {v['in']} | {v['out']} | {v['train']} | {v['val']} | {v['test']} | {v['bench']} |")
    A("")
    A("| kind (fim: cut_kind; chat: kind) | tipo | entrada | saida |")
    A("|---|---|---:|---:|")
    for (ty, k), v in sorted(info["by_kind"].items(), key=lambda x: (x[0][0], str(x[0][1]))):
        A(f"| {k} | {ty} | {v['in']} | {v['out']} |")
    A("")
    g = info["gates"]
    A("## 4. Gates de dataset (spec §6) medidos sobre a saida")
    A("")
    A("| gate | resultado | medicao |")
    A("|---|---|---|")
    A(f"| G3.1 paridade de contrato | NAO MEDIDO aqui | depende de ContractProbe/IDE (`data/contracts/parity_inputs.jsonl`) |")
    g2 = g["G3.2"]
    A(f"| G3.2 validade | {'PASSA' if g2['pass'] else 'FALHA'} | {g2['all_static_ok']}/{g2['records']} com syntax/api/fields ok; exec ok {g2['exec_ok']}, "
      f"skipped {g2['exec_skipped']} (escritas e respostas sem codigo); exec-valid das leituras na entrada do pipeline: "
      f"{pct(g2['exec_valid_rate_reads'] or 0, 1) if g2['exec_valid_rate_reads'] is not None else 'n/a'}; chat/correcao com equivalencia ok |")
    g3 = g["G3.3"]
    A(f"| G3.3 volume e distribuicao | {'PASSA' if g3['pass'] else 'FALHA'} | train validado {g3['train_validated']} "
      f"(fim {g3['train_fim']} + chat {g3['train_chat']}; alvo >= 1.000.000: {'ok' if g3['volume_ok'] else 'FALHA'}); "
      f"cut_kind +-2pp: {'ok' if g3['cut_kind_ok'] else 'FALHA'}; categorias +-2pp: {'ok' if g3['categories_ok'] else 'FALHA'}; "
      f"piso 4% (atlas/json/dsl/indexes): {'ok' if g3['floors_ok'] else 'FALHA'}; idiomas +-2pp: "
      f"{'ok' if g3['language_ok'] else 'FALHA'}; dialetos {g3['dialects']}; categorias ausentes: "
      f"{', '.join(g3['spec_categories_missing']) or 'nenhuma'} |")
    g4 = g["G3.4"]
    A(f"| G3.4 vazamento | {'PASSA' if g4['pass'] else 'FALHA'} | grupos compartilhados entre splits: {g4['groups_shared_across_splits']}; "
      f"treino x (val/test/bench/legacy-v1) restantes: {g4['train_x_eval_hits_after_decontam']}; blind: {g4['blind']}; legacy-v1 conferido: {g4['legacy_checked']} docs |")
    A(f"| G3.5 privacidade | {'PASSA' if g['G3.5']['pass'] else 'FALHA'} | {g['G3.5']['occurrences_in_output']} ocorrencias de padroes proibidos na saida |")
    A("| G3.6 blind | PENDENTE | conjunto cego nao selado (`blind: not_sealed`); 400 itens escritos/revisados por pessoa |")
    A("| G3.7 revisao humana | PENDENTE | 300 exemplos aleatorios do train (amostrar de `data/splits/" + ver + "/train.ids.txt`) |")
    A(f"| G3.8 reproducibilidade | {info['repro']} | sha256 de saidas no `manifest.json`; mesmo seed/config/commit => mesmos hashes |")
    A(f"| G3.9 licencas | {info['licenses']} | fontes geradas (`source_license: generated`); `data/sources/licenses.md` |")
    A("")
    A("### Gates adicionais da tarefa")
    A("")
    mx = g["mixture"]
    A(f"- **Chat+correcao <= 25%** (pool de treino validado): {mx['pool_chat_plus_correction_ratio']*100:.1f}%; na mistura-alvo da config: "
      f"{mx['target_chat_plus_correction_ratio']*100:.1f}% -> {'PASSA' if mx['pass'] else 'FALHA'}. Pool por componente: {mx['pool']}; "
      f"alvo: {mx['target']}; maximo sem repeticao: {mx['max_total_without_repetition']} exemplos (limitante: `{mx['binding_component']}`).")
    fl = g["floors"]
    A(f"- **Piso de grupos pequenos (4%)** ({', '.join(fl['applicable_groups'])}): categorias abaixo de 4%: "
      f"{fl['below_4pct'] or 'nenhuma'}; cut_kind abaixo de 4%: {fl['cut_kind_below_4pct'] or 'nenhum'} -> "
      f"{'PASSA' if fl['pass'] else 'FALHA'}.")
    A("- **Categorias (train FIM; alvo = `fim.category_targets`, +-2pp)**: " + ", ".join(
        f"{c} {v['actual']*100:.1f}% (alvo {v['target']*100:.1f}%)" for c, v in g3['categories'].items()) + ".")
    cv_ = g["coverage"]
    A(f"- **Cobertura**: dominios no train {len(cv_['domains_train'])}; operadores distintos no train {len(cv_['operators_train_docs'])}; "
      f"operadores raros (<20 docs): {cv_['operators_rare_lt20'] or 'nenhum'}. cut_kind (train, alvo +-2pp): "
      + ", ".join(f"{k} {v['actual']*100:.1f}% (alvo {v['target']*100:.0f}%)" for k, v in g3['cut_kind'].items()) + ".")
    g27 = g["G2.7"]
    A(f"- **G2.7 (consulta ao manifesto)**: ids de val/test/bench no train = {g27['eval_ids_in_train']}; holdout em train/val = "
      f"{g27['holdout_in_train_or_val']} -> {'PASSA' if g27['pass'] else 'FALHA'}.")
    w = g["writes"]
    A(f"- **0 escritas executaveis**: outputs de chat com escrita executavel = {w['chat_outputs_with_executable_write']}; comandos de escrita "
      f"vistos pelo servidor = {w['server_write_commands_seen']}; documentos FIM que sao escritas (sem execucao) = "
      f"{w['fim_docs_that_are_writes_not_executed']} -> {'PASSA' if w['pass'] else 'FALHA'}.")
    A("")
    A("## 5. Falhas e ajustes propostos (sem alterar geradores)")
    A("")
    for p in info["proposals"]:
        A(f"- {p}")
    A("")
    cmpv = info.get("compare")
    if cmpv:
        A("## 5b. Comparacao v1 x v2")
        A("")
        c1 = cmpv.get("counts") or {}
        A("| metrica | v1 | v2 |")
        A("|---|---:|---:|")
        for sp in SPLITS:
            A(f"| {sp} (fim+chat) | {c1.get(sp)} | {info['final'][sp]} |")
        A(f"| train fim | {cmpv.get('train_fim')} | {gates_g3(info)['train_fim']} |")
        A(f"| train chat | {cmpv.get('train_chat')} | {gates_g3(info)['train_chat']} |")
        r1 = cmpv.get("rejections") or {}
        for st in STAGES:
            n2 = next((x["rejected"] for x in info["funnel"] if x["stage"] == st), 0)
            A(f"| rejeitados em {st} | {r1.get(st)} | {n2} |")
        d1 = cmpv.get("decontam") or {}
        A(f"| removidos na decontaminacao (treino) | {d1.get('removed')} | {info['decon']['train_removed']} |")
        A("")
        A("| cut_kind (train FIM) | v1 | v2 |")
        A("|---|---:|---:|")
        n1 = sum((cmpv.get("cut_kind") or {}).values()) or 1
        for k, v in gates_g3(info)["cut_kind"].items():
            A(f"| {k} | {100.0 * (cmpv.get('cut_kind') or {}).get(k, 0) / n1:.1f}% | {v['actual'] * 100:.1f}% |")
        A("")
        A("| categoria (train FIM) | v1 | v2 |")
        A("|---|---:|---:|")
        cats1 = cmpv.get("category") or {}
        for c, v in gates_g3(info)["categories"].items():
            A(f"| {c} | {100.0 * cats1.get(c, 0) / n1:.1f}% | {v['actual'] * 100:.1f}% |")
        A("")
    A("## 6. Limites")
    A("")
    A("- Orcamento de chat usa renderizacao aproximada do `local-agent-fim-v0` (spec §4.4), nao o ContractProbe.")
    A("- Etapa 5 em FIM executa o plano do snippet (o documento reconstruido e igual ao texto e ao plano); escritas nunca sao executadas.")
    A("- Dedup exato por split usa tokens com literais preservados; near-dup (Jaccard >= 0,85) so no treino, decidido por documento "
      "(cortes distintos do mesmo snippet nao sao duplicatas).")
    A("- `limite por grupo` (max_per_group) e nomes de pessoas (regras v1) nao estao na config atual e nao foram aplicados.")
    A("- Atlas Search (`$search`/`$searchMeta`/`$vectorSearch`) so tem validacao ESTATICA (sem atlas-local): `exec: skipped_static`; "
      "createIndex/dropIndex e update/delete sao `skipped_write` (0 escritas chegam ao driver).")
    A("- `.hint()` NAO e metodo de cursor do Console (spec §2: sort, skip, limit, project, toArray): continua rejeitado na etapa API.")
    A("- Multi-turno (`history`) nao tem renderizacao no contrato `local-agent-fim-v0`: o orcamento de tokens (etapa 7) usa uma "
      "aproximacao (turnos `Usuario:`/`Assistente:` antes do editor).")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# =========================================================================== orquestracao
def git_info() -> tuple[str, bool]:
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        return c, bool(d)
    except Exception:
        return "unknown", False


_PW: dict = {}
_SNIP_ID_RE = re.compile(r'"snippet_id":"([^"]+)"')


def _pw_init(cfg, snippets, exec_mode, tok_mode, node, tokenizer_path, schemas_dir):
    """Estado de um worker: Ctx proprio (node, conexao pymongo somente leitura, tokenizer)."""
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    schemas = load_schemas(Path(schemas_dir))
    _PW["ctx"] = Ctx(cfg, schemas, snippets, exec_mode=exec_mode, tok_mode=tok_mode, node=node, tokenizer_path=tokenizer_path)


def _pw_run(chunk):
    ctx = _PW["ctx"]
    out = []
    for typ, r in chunk:
        e = stages_1_to_7(r, typ, ctx)
        out.append((e, None if e.failed else lite_record(r, typ)))
    if len(ctx.doc_cache) > 150000:
        ctx.doc_cache.clear()
    if len(ctx.exec_cache) > 150000:
        ctx.exec_cache.clear()
    viol = list(ctx.executor.violations) if ctx.executor is not None else []
    return out, os_pid(), viol


def os_pid() -> int:
    import os
    return os.getpid()


def needed_snippet_ids(files) -> set:
    """Ids de snippet referenciados pelos registros (regex por linha: sem parse JSON completo)."""
    ids = set()
    for p in files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                m = _SNIP_ID_RE.search(line)
                if m:
                    ids.add(m.group(1))
    return ids


def summarize_validated(vdir: Path) -> dict | None:
    """Resumo de um dataset validado anterior (fim.jsonl/chat.jsonl + manifest.json) para a comparacao v1 x v2."""
    mf = vdir / "manifest.json"
    if not mf.exists():
        return None
    m = json.loads(mf.read_text(encoding="utf-8"))
    out: dict = {"version": m.get("version"), "counts": m.get("counts"), "rejections": m.get("rejections"),
                 "decontam": m.get("decontam"), "generator_commit": m.get("generator_commit")}
    cut, cat, lang, kinds = Counter(), Counter(), Counter(), Counter()
    n_fim_tr = n_chat_tr = 0
    for f in resolve_inputs(vdir / "fim.jsonl"):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if '"split":"train"' not in line:
                    continue
                r = json.loads(line)
                if r.get("split") == "train":
                    n_fim_tr += 1
                    cut[r.get("cut_kind")] += 1
                    cat[r.get("category") or ("aggregation" if r.get("kind") == "aggregate" else "crud")] += 1
    for f in resolve_inputs(vdir / "chat.jsonl"):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("split") == "train":
                    n_chat_tr += 1
                    lang[r.get("language")] += 1
                    kinds[r.get("kind")] += 1
    out.update({"train_fim": n_fim_tr, "train_chat": n_chat_tr, "cut_kind": dict(cut), "category": dict(cat),
                "language": dict(lang), "chat_kinds": dict(kinds)})
    return out


def run_pipeline(args) -> dict:
    t0 = time.time()
    cfg = load_config(args.config)
    ver = args.ver or cfg.get("version", "v1")
    seed = int(cfg.get("seed", 20261008))
    gdir = ROOT / cfg.get("out_dir", "data/generated") / ver
    fim_in = Path(args.fim) if args.fim else gdir / "fim_ctx.jsonl"
    chat_in = Path(args.chat) if args.chat else gdir / "chat.jsonl"
    snip_in = Path(args.snippets) if args.snippets else gdir / "snippets.jsonl"
    fim_files = resolve_inputs(fim_in)
    chat_files = resolve_inputs(chat_in)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "data" / "validated" / ver
    split_dir = Path(args.splits_dir) if args.splits_dir else ROOT / "data" / "splits" / ver
    rep_md = Path(args.report) if args.report else ROOT / "reports" / "datasets" / f"{ver}-validation-report.md"
    dec_dir = Path(args.decontam_dir) if args.decontam_dir else ROOT / "reports" / "decontam"
    for d in (out_dir, split_dir, rep_md.parent, dec_dir):
        d.mkdir(parents=True, exist_ok=True)
    shard_size = int(getattr(args, "shard_size", 0) or 0)
    dcfg = cfg.get("decontam") or {}
    min_content = args.ngram_min_content if args.ngram_min_content is not None else int(dcfg.get("ngram_min_content", dc.DEFAULT_NGRAM_MIN_CONTENT))
    args.ngram_min_content = min_content
    # auditoria paralela (nunca aplicada): a regra LITERAL (k=0) quando a oficial exige conteudo; senao, k=2 se pedido
    if getattr(args, "audit_literal", True) and min_content > 0:
        analysis_ks = (0,)
    else:
        a_k = int(getattr(args, "analysis_ngram_min_content", 0) or 0)
        analysis_ks = (a_k,) if a_k and a_k != min_content else ()

    schemas = load_schemas(ROOT / cfg.get("schemas_dir", "data/schemas"))
    snippets = {}
    if resolve_inputs(snip_in):
        # so o que o pipeline le do snippet (texto, plano de referencia, banco, dialeto, exec_mode): ~1 M de snippets cabem em memoria
        keys = ("id", "text", "plan", "db", "dialect", "exec_mode", "op", "kind")
        want = needed_snippet_ids(list(fim_files) + list(chat_files))
        for r in read_jsonl(snip_in):
            if r["id"] in want:
                snippets[r["id"]] = r if args.include_snippets else {k: r[k] for k in keys if k in r}
    tok_path = ROOT / (cfg.get("editor_context") or {}).get("tokenizer", "BaseRepos/Qwen2.5-Coder-0.5B/tokenizer.json")
    ctx = Ctx(cfg, schemas, snippets, exec_mode=args.exec, tok_mode=args.tokenizer, node=not args.no_node, tokenizer_path=tok_path)

    entries: list[Entry] = []
    lite: dict = {}
    limit = args.limit
    srcs = [("fim", fim_files, fim_in), ("chat", chat_files, chat_in)]
    n_in = Counter()
    try:
        import par

        def stream():
            for typ, files, _ in srcs:
                for p in files:
                    bg.check_path(p)
                for i, r in enumerate(iter_jsonl(files)):
                    if limit and i >= limit:
                        break
                    n_in[typ] += 1
                    yield (typ, r)

        workers = args.workers if getattr(args, "workers", None) is not None else par.default_workers(8)
        viol_by_pid: dict = {}
        # etapas 1-7 por registro (independentes): processos em paralelo; resultados na ORDEM de entrada (saida deterministica)
        for res_chunk, pid, viol in par.ordered_map(
                _pw_run, par.chunked(stream(), 1000), workers, _pw_init,
                (cfg, snippets, ctx.exec_mode if ctx.exec_mode != "live" else "live", args.tokenizer, not args.no_node, tok_path,
                 str(ROOT / cfg.get("schemas_dir", "data/schemas")))):
            viol_by_pid[pid] = viol
            for e, lr in res_chunk:
                entries.append(e)
                if lr is not None:                    # so quem segue para dedup/decontaminacao precisa do texto
                    lite[e.id] = lr
            if len(entries) % 100000 < 1000:
                print(f"  etapas 1-7: {len(entries)} registros ({time.time() - t0:.0f}s)", flush=True)
        ctx.violations_total = sum(len(v) for v in viol_by_pid.values()) + (len(ctx.executor.violations) if ctx.executor else 0)
        ctx.doc_cache.clear()
        ctx.exec_cache.clear()
        # ---- etapa 8
        ded = stage_dedup(entries, lite, seed, args.dedup_jaccard)
        for e in entries:
            if not e.failed and "dedup" not in e.val:
                e.val["dedup"] = _st("ok")
        print(f"  etapa 8 (dedup) ok ({time.time() - t0:.0f}s)", flush=True)
        # ---- etapa 9
        surv = [e for e in entries if not e.failed]
        evals = [(e.split, lite[e.id]) for e in surv if e.split in EVAL]
        legacy = [Path(p) for p in (args.legacy if args.legacy is not None else
                  [str(ROOT / "bakeoff" / "data" / "v1_benchmark.jsonl"), str(ROOT / "bakeoff" / "data" / "v1_validation.jsonl")])]
        legacy = [p for p in legacy if p.exists()]
        for p in legacy:
            evals += [("legacy-v1", r) for r in read_jsonl(p)]
        train = [lite[e.id] for e in surv if e.split == "train"]
        res = dc.run(train, evals, seed=seed, min_content=min_content, analysis_k=analysis_ks)
        for e in surv:
            if e.split == "train" and e.id in res["removed"]:
                fail(e, "decontam", "coincide com avaliacao: " + ",".join(res["removed"][e.id]))
            elif "decontam" not in e.val:
                e.val["decontam"] = _st("ok" if e.split == "train" else "skipped",
                                        None if e.split == "train" else "split de avaliacao (referencia da decontaminacao)")
        print(f"  etapa 9 (decontam) ok ({time.time() - t0:.0f}s)", flush=True)
        # sobrevivencia por cut_kind (train FIM; entrada = apos etapas 1-8; regra literal) -> recalibra fim.decontam_survival
        surv_in, surv_out = Counter(), Counter()
        for e in surv:
            if e.split == "train" and e.type == "fim":
                surv_in[e.kind] += 1
                surv_out[e.kind] += 0 if e.failed else 1
        survival = {k: round(surv_out[k] / surv_in[k], 4) for k in sorted(surv_in) if surv_in[k]}
        cs_in, cs_out = Counter(), Counter()
        for e in surv:
            if e.split == "train" and e.type == "fim":
                c = lite[e.id].get("category") or "?"
                cs_in[c] += 1
                cs_out[c] += 0 if e.failed else 1
        survival_cat = {k: round(cs_out[k] / cs_in[k], 4) for k in sorted(cs_in) if cs_in[k]}
        # verificacao: 2a passada sobre o treino mantido
        kept = [e for e in entries if not e.failed]
        leak = dc.run([lite[e.id] for e in kept if e.split == "train"], evals, seed=seed, min_content=min_content)
        final_leak = {"train_removed": leak["train_removed"]}
        del leak

        # ---- saidas (shards jsonl em streaming)
        splits_ids = {s: [e.id for e in kept if e.split == s] for s in SPLITS}
        fates = {e.id: e for e in entries}
        counts = Counter()
        files = {}
        for typ, ifiles, _ in srcs:
            stem = "fim" if typ == "fim" else "chat"
            w = ShardWriter(out_dir, stem, shard_size if typ == "fim" else 0)
            for i, r in enumerate(iter_jsonl(ifiles)):
                if limit and i >= limit:
                    break
                e = fates[r["id"]]
                if e.failed:
                    continue
                if typ == "chat" and "validation" in r:
                    r["generator_validation"] = r.pop("validation")
                r["validation"] = {s: e.val.get(s, _st("skipped", "nao aplicavel")) for s in STAGES}
                w.write(r)
                counts[(typ, e.split)] += 1
            wr = w.close()
            for fi in wr["files"]:
                files[Path(fi["path"]).name] = {"path": fi["path"], "sha256": fi["sha256"], "records": fi["records"]}
        if args.include_snippets and snippets:
            keep_snips = {e.doc for e in kept if e.type == "fim"}
            w = ShardWriter(out_dir, "snippets", 0)
            for sid, r in snippets.items():
                if sid in keep_snips:
                    w.write(r)
            for fi in w.close()["files"]:
                files[Path(fi["path"]).name] = {"path": fi["path"], "sha256": fi["sha256"], "records": fi["records"]}
        rej = out_dir / "rejected.jsonl"
        reasons = Counter()
        with open(rej, "w", encoding="utf-8", newline="\n") as fo:
            for e in entries:
                if e.failed:
                    st, why = e.failed
                    fo.write(json.dumps({"id": e.id, "type": e.type, "split": e.split, "kind": e.kind, "domain": e.domain,
                                         "stage": st, "reason": why}, ensure_ascii=False, separators=(",", ":")) + "\n")
                    rk = why.split(" de ", 1)[0] if st == "dedup" else why
                    reasons[(st, re.sub(r"[0-9a-f]{24}|\d+(\.\d+)?", "#", rk)[:110])] += 1
        files["rejected.jsonl"] = {"path": str(rej), "sha256": sha256_file(rej), "records": None}
        for s in SPLITS:
            p = split_dir / f"{s}.ids.txt"
            p.write_text("".join(i + "\n" for i in sorted(splits_ids[s])), encoding="utf-8", newline="\n")
            files[f"splits/{s}.ids.txt"] = {"path": str(p), "sha256": sha256_file(p), "records": len(splits_ids[s])}

        # ---- funil
        funnel = []
        alive = list(entries)
        for st in STAGES:
            ins = [e for e in alive]
            outs = [e for e in ins if not (e.failed and e.failed[0] == st)]
            funnel.append({"stage": st, "in": len(ins), "out": len(outs), "rejected": len(ins) - len(outs),
                           "fim_in": sum(1 for e in ins if e.type == "fim"), "fim_out": sum(1 for e in outs if e.type == "fim"),
                           "chat_in": sum(1 for e in ins if e.type == "chat"), "chat_out": sum(1 for e in outs if e.type == "chat")})
            alive = outs
        del alive
        by_split_type, by_domain, by_kind = {}, {}, {}
        for e in entries:
            v = by_split_type.setdefault((e.split, e.type), {"in": 0, "out": 0})
            v["in"] += 1
            v["out"] += 0 if e.failed else 1
            d = by_domain.setdefault(e.domain, {"in": 0, "out": 0, "holdout": e.holdout, "train": 0, "val": 0, "test": 0, "bench": 0})
            d["in"] += 1
            if not e.failed:
                d["out"] += 1
                d[e.split] += 1
            k = by_kind.setdefault((e.type, e.kind), {"in": 0, "out": 0})
            k["in"] += 1
            k["out"] += 0 if e.failed else 1

        decon_report = dc.public_report(res, {})
        all_inputs = [p for p in list(fim_files) + list(chat_files) if p.exists()]
        dhash = dc.dataset_hash(all_inputs, {"seed": seed, "params": res["params"], "legacy": [p.name for p in legacy], "limit": limit})
        decon_report.update({"dataset_hash": dhash, "inputs": [str(p) for p in all_inputs], "legacy": [str(p) for p in legacy],
                             "second_pass_removed": final_leak["train_removed"], "pipeline": PIPELINE_VERSION,
                             "survival_by_cut_kind": survival, "survival_by_category": survival_cat})
        (dec_dir / f"{dhash[:16]}.json").write_text(json.dumps(decon_report, indent=1, ensure_ascii=False), encoding="utf-8")

        gates = compute_gates(kept, lite, cfg, snippets, final_leak, ctx, entries, res, splits_ids)
        commit, dirty = git_info()
        licenses = "OK (geradas)" if (ROOT / "data" / "sources" / "licenses.md").exists() else "PENDENTE (licenses.md ausente)"
        props = build_proposals(gates, res, funnel, reasons)
        final = {s: len(splits_ids[s]) for s in SPLITS}
        compare = None
        if getattr(args, "compare", None):
            compare = summarize_validated(Path(args.compare))
        info = {"seed": seed, "commit": commit, "dirty": dirty, "exec_mode": ctx.exec_mode, "tok_desc": ctx.tok_desc,
                "node": ctx.node is not None and cv.node_available(), "funnel": funnel, "n_in": sum(n_in.values()),
                "n_in_fim": n_in["fim"], "n_in_chat": n_in["chat"], "n_out": len(kept), "final": final,
                "reasons": reasons.most_common(), "decon": res, "dataset_hash": dhash, "final_leak": final_leak["train_removed"],
                "by_split_type": by_split_type, "by_domain": by_domain, "by_kind": by_kind, "gates": gates,
                "repro": "COBERTO POR TESTE (2 execucoes => mesmos sha256)", "licenses": licenses, "proposals": props,
                "survival": survival, "survival_in": dict(surv_in), "survival_out": dict(surv_out), "compare": compare,
                "survival_cat": survival_cat, "survival_cat_in": dict(cs_in), "survival_cat_out": dict(cs_out),
                "analysis_ks": list(analysis_ks), "min_content": min_content, "dedup": ded, "seconds": round(time.time() - t0, 1), "limit": limit}
        render_report(rep_md, ver, info)
        files["report.md"] = {"path": str(rep_md), "sha256": sha256_file(rep_md), "records": None}
        files["decontam.json"] = {"path": str(dec_dir / f"{dhash[:16]}.json"), "sha256": sha256_file(dec_dir / f"{dhash[:16]}.json"),
                                  "records": None}

        # ---- manifesto (congelamento)
        cfg_path = Path(args.config)
        mf = {
            "dataset": "kapi-mongo-ds", "version": ver, "pipeline": PIPELINE_VERSION, "seed": seed,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator_commit": commit, "git_dirty": dirty,
            "config": {"path": str(cfg_path.relative_to(ROOT)) if cfg_path.is_absolute() and ROOT in cfg_path.parents else str(cfg_path),
                       "sha256": sha256_file(cfg_path)},
            "inputs": {p.name: {"sha256": sha256_file(p), "records": None} for p in all_inputs},
            "inputs_records": {"fim": n_in.get("fim"), "chat": n_in.get("chat")},
            "tokenizer": {"mode": ctx.tok_desc, "sha256": sha256_file(tok_path) if ctx.tok is not None else None},
            "exec_mode": ctx.exec_mode,
            "files": {k: v for k, v in sorted(files.items())},
            "counts": {"train": final["train"], "val": final["val"], "test": final["test"], "bench": final["bench"],
                       "by_type_split": {f"{t}.{s}": c for (t, s), c in sorted(counts.items())}, "blind": 0},
            "rejections": {st: sum(1 for e in entries if e.failed and e.failed[0] == st) for st in STAGES},
            "decontam": {"dataset_hash": dhash, "blind": res["blind"], "removed": res["train_removed"], "by_rule": res["by_rule"],
                         "survival_by_cut_kind": survival, "survival_by_category": survival_cat, "analysis": res.get("analysis")},
            "gates": {k: (v.get("pass") if isinstance(v, dict) else v) for k, v in gates.items()},
            "limit": limit, "seconds": info["seconds"],
        }
        (out_dir / "manifest.json").write_text(json.dumps(mf, indent=1, ensure_ascii=False), encoding="utf-8")
        return {"manifest": mf, "info": info, "seconds": round(time.time() - t0, 1)}
    finally:
        ctx.close()


def build_proposals(g: dict, decon: dict, funnel: list, reasons: Counter) -> list[str]:
    """Falhas medidas e ajustes propostos (o pipeline nao altera geradores)."""
    P = []
    g3 = g["G3.3"]
    tin = max(1, decon["train_in"])
    if not g3["volume_ok"]:
        P.append(f"G3.3 volume: train validado {g3['train_validated']} (fim {g3['train_fim']} + chat {g3['train_chat']}) < 1.000.000. "
                 f"Gargalo medido: a decontaminacao literal remove {100 * decon['train_removed'] / tin:.0f}% do treino que chega a etapa 9 "
                 "(13-gramas de estrutura da linguagem sao compartilhados com a avaliacao). Ajuste: mais volume bruto (a sobrevivencia e "
                 "~constante por documento) ou mais planos distintos; ver `Sobrevivencia por cut_kind`.")
    if g3["cut_kind_ok"] is False:
        bad = [k for k, v in g3["cut_kind"].items() if not v["ok"]]
        P.append(f"G3.3 cut_kind fora de +-2pp apos o pipeline: {bad}. Ajuste: recalibrar `fim.decontam_survival` com a "
                 "sobrevivencia medida e reamostrar (a decontaminacao remove desproporcionalmente alguns cut_kind).")
    if g3.get("categories_ok") is False:
        bad = [c for c, v in g3["categories"].items() if not v["ok"]]
        P.append(f"G3.3 categorias fora de +-2pp apos o pipeline: {bad}. Ajuste: `fim.category_survival` com a sobrevivencia medida.")
    if g3.get("floors_ok") is False:
        P.append("G3.3 piso de 4%: alguma categoria pequena (atlas_search/json_ejson/dsl/indexes) ficou abaixo do piso apos a decontaminacao.")
    if g3["spec_categories_missing"]:
        P.append(f"G3.3 categorias ausentes no train: {g3['spec_categories_missing']}.")
    if decon["train_removed"] > 0.2 * tin:
        P.append(f"G3.4: a decontaminacao removeu {decon['train_removed']} de {decon['train_in']} itens do treino "
                 f"({100 * decon['train_removed'] / tin:.0f}%): templates compartilham 13-gramas/chave aproximada entre schemas. "
                 "Ver a analise paralela (>= k tokens de conteudo) para separar estrutura de conteudo.")
    if g["mixture"]["max_total_without_repetition"] < g3["train_validated"]:
        P.append(f"Mistura: a componente `{g['mixture']['binding_component']}` limita a mistura-alvo a "
                 f"{g['mixture']['max_total_without_repetition']} exemplos sem repeticao (pool: {g['mixture']['pool']}).")
    n_san = sum(n for (st, _), n in reasons.items() if st == "sanitize")
    if n_san:
        P.append(f"Etapa 1: {n_san} registros rejeitados por padrao proibido (campos `secret`/`token`/`password` nos schemas).")
    n_bud = sum(n for (st, _), n in reasons.items() if st == "budget")
    if n_bud:
        P.append(f"Etapa 7: {n_bud} registros acima do limite de tokens (prompt 2048 / resposta 96 fim, 256 chat).")
    if g["G3.4"]["blind"] == "not_sealed":
        P.append("G3.6: conjunto cego nao selado; a decontaminacao contra blind so passa a valer apos `blind_tool.py seal`.")
    if not g["G3.2"]["pass"]:
        P.append("G3.2 falhou: ver motivos de rejeicao na secao 1.")
    return P


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--ver")
    ap.add_argument("--fim")
    ap.add_argument("--chat")
    ap.add_argument("--snippets")
    ap.add_argument("--include-snippets", action="store_true", help="grava tambem validated/<ver>/snippets.jsonl (snippets dos FIM mantidos)")
    ap.add_argument("--out-dir")
    ap.add_argument("--splits-dir")
    ap.add_argument("--report")
    ap.add_argument("--decontam-dir")
    ap.add_argument("--exec", choices=["auto", "recorded", "live"], default="auto")
    ap.add_argument("--tokenizer", choices=["auto", "off"], default="auto")
    ap.add_argument("--no-node", action="store_true")
    ap.add_argument("--workers", type=int, help="processos das etapas 1-7 (padrao: nucleos - 2, max 8); saida identica para qualquer N")
    ap.add_argument("--legacy", nargs="*", default=None, help="jsonl legacy-v1 (padrao: bakeoff/data/v1_*.jsonl se existirem)")
    ap.add_argument("--ngram-min-content", type=int, default=None,
                    help="regra OFICIAL: tokens de conteudo minimos no 13-grama (padrao: decontam.ngram_min_content da config = 2; 0 = literal)")
    ap.add_argument("--no-audit-literal", dest="audit_literal", action="store_false",
                    help="desliga a auditoria paralela da regra literal (k=0), calculada por padrao e nunca aplicada")
    ap.add_argument("--analysis-ngram-min-content", type=int, default=0,
                    help="ANALISE paralela (nao aplicada): remocao se o 13-grama exigisse >= N tokens de conteudo (0 = desligada)")
    ap.add_argument("--shard-size", type=int, default=0, help="linhas por shard de fim.jsonl na saida (0 = arquivo unico)")
    ap.add_argument("--compare", help="diretorio validado anterior (ex.: data/validated/v1) para a secao v1 x v2 do relatorio")
    ap.add_argument("--dedup-jaccard", type=float, default=dc.DEDUP_JACCARD)
    ap.add_argument("--limit", type=int, help="primeiros N registros de cada arquivo (testes)")
    a = ap.parse_args(argv)
    out = run_pipeline(a)
    mf = out["manifest"]
    print(json.dumps({"counts": mf["counts"], "rejections": mf["rejections"], "decontam": mf["decontam"], "gates": mf["gates"],
                      "exec_mode": mf["exec_mode"], "tokenizer": mf["tokenizer"]["mode"], "seconds": out["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
