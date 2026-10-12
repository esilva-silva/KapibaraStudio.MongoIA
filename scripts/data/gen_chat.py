#!/usr/bin/env python3
"""Gerador de chat curto / correcao / negativos / explain (spec fase-03 §4.3 `gen_chat.py`, §4.4, §4.5).

  python scripts/data/gen_chat.py --ver v1 [--count 6000] [--seed N] [--report]

Entrada : data/generated/<ver>/snippets.jsonl (+ validation.jsonl) - so snippets de LEITURA validos viram
          chat/correction/explain; snippets de ESCRITA viram apenas NEGATIVOS (recusa; nunca saem como codigo).
Saida   : data/generated/<ver>/chat.jsonl (+ chat.manifest.json); --report grava reports/datasets/<ver>-chat-report.md.

Registro estruturado, independente de contrato (nao renderiza para local-agent-fim-v0):
  {id, schema_version, type:"chat", kind: chat|correction|negative|explain, language: pt-BR|en, domain, schema_id,
   collection, group, group_id, split_key, split, holdout, template_id, snippet_id, instruction,
   context:{connection, database, collection, selection, editor, known_collections}, output, intent, defect?, validation,...}
`output` e o codigo que substitui o editor (ou, em negativos/explain, comentarios `//`, pois o contrato pede so codigo).
Todo `output` de codigo e validado: sintaxe JS (node), API do Console, campos/tipos do schema, execucao em kapi_fx_*
(somente leitura) e igualdade com o plano de origem. Em correction o `editor` defeituoso tem de FALHAR e o corrigido passar.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind_guard  # noqa: E402
import chat_describe as cd  # noqa: E402
import chat_validate as cv  # noqa: E402
from gen_snippets import CONFIG_PATH, ROOT, Ctor, Style, load_config, load_schemas, render_snippet  # noqa: E402

SCHEMA_VERSION = "kapi-ds-record-v2"
LANG_CODE = {"pt": "pt-BR", "en": "en"}
CONNECTIONS = ["dev-cluster", "staging", "prod-readonly", "atlas-dev", "homolog", "analytics", "local"]
KINDS = ("chat", "correction", "negative", "explain")


# --------------------------------------------------------------------------- utilidades

def ej_to_py(v):
    """EJSON proprio -> plano com Ctor (aceito por render_snippet)."""
    if isinstance(v, dict):
        if len(v) == 1:
            (k, x), = v.items()
            kinds = {"$oid": "oid", "$date": "date", "$numberLong": "i64", "$numberDecimal": "dec", "$uuid": "uuid"}
            if k in kinds:
                return Ctor(kinds[k], x)
        return {k: ej_to_py(x) for k, x in v.items()}
    if isinstance(v, list):
        return [ej_to_py(x) for x in v]
    return v


def pick(rng: random.Random, d: dict):
    ks = list(d)
    return rng.choices(ks, weights=[d[k] for k in ks])[0]


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", cd.deaccent(s)).strip().lower()


def leaves(v, out=None):
    """Valores escalares 'de usuario' do plano (nao `$campo`, nao chaves)."""
    out = [] if out is None else out
    if isinstance(v, dict):
        if len(v) == 1 and next(iter(v)) in cd.EJSON_KEYS:
            out.append(v)
        else:
            for k, x in v.items():
                if k in ("$options", "format", "as", "from", "localField", "foreignField", "path", "preserveNullAndEmptyArrays"):
                    continue
                if k == "$sum" and x == 1:
                    continue
                leaves(x, out)
    elif isinstance(v, list):
        for x in v:
            leaves(x, out)
    elif isinstance(v, str):
        if not v.startswith("$"):
            out.append(v)
    elif isinstance(v, bool) or v is None:
        pass
    elif isinstance(v, (int, float)):
        out.append(v)
    return out


def literals_present(plan: dict, instr_raw: str) -> bool:
    """Todo valor citado no plano aparece literalmente na instrucao (valores aprendiveis)."""
    body = {k: v for k, v in plan.items() if k in ("filter", "pipeline", "update")}
    if "pipeline" in body:   # direcao de sort e 0/1 de projecao nao sao "valores citados"
        stages = []
        for st in body["pipeline"]:
            (nm, b), = st.items()
            if nm == "$sort":
                continue
            if nm == "$project" and isinstance(b, dict):
                b = {k: v for k, v in b.items() if not (isinstance(v, (int, bool)) and v in (0, 1))}
            stages.append({nm: b})
        body = {**body, "pipeline": stages}
    flat = cd.strip_marks(instr_raw)
    for x in leaves(body):
        try:
            s = cd.lit(x)
        except cd.Unsupported:
            return False
        if isinstance(x, str):
            m = re.fullmatch(r"\^([A-Za-z0-9 ]+)", x)
            if m:
                s = m.group(1)
            s = s.strip("\"'")
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            # inteiros pequenos de $limit/$round/$mod aparecem como numero; so exige quando for valor de filtro/grupo
            pass
        if s not in flat:
            return False
    return True


def order_dependent(plan: dict) -> bool:
    """True se o resultado depende da ordem instavel de $group/$sortByCount (ex.: $group + $limit sem $sort)."""
    if plan.get("op") != "aggregate":
        return False
    unstable = False
    for st in plan["pipeline"]:
        (name, _b), = st.items()
        if name in ("$group", "$sortByCount", "$bucket", "$bucketAuto", "$facet"):
            unstable = True
        elif name == "$sort":
            unstable = False
        elif name in ("$limit", "$skip") and unstable:
            return True
    return False


def style_cfg(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["snippets"].update(cfg["chat"].get("style", {}))
    return c


def render_text(plan_ej: dict, sch, coll, cfg_style: dict, rng: random.Random):
    st = Style(rng, cfg_style)
    text, _spans, info = render_snippet(ej_to_py(plan_ej), sch, coll, st, rng)
    return text, info


def refusal(lines: list[str]) -> str:
    return "\n".join("// " + l for l in lines)


# --------------------------------------------------------------------------- defeitos (correction)

INVENTED = {
    "$regex": ["$like", "$contains", "$matches"], "$nin": ["$notIn", "$not_in"], "$in": ["$any", "$inside", "$inn"],
    "$gte": ["$ge", "$gteq", "$gtee"], "$lte": ["$le", "$lteq", "$ltee"], "$gt": ["$greater", "$gtt"],
    "$lt": ["$less", "$lessThan", "$ltt"], "$ne": ["$neq", "$notEqual", "$nee"], "$exists": ["$exist", "$existe"],
    "$sum": ["$total", "$summ"], "$avg": ["$average", "$mean"], "$match": ["$mach", "$filter", "$matchh"],
    "$group": ["$grup", "$groupBy", "$grouup"], "$sort": ["$order", "$sortt"], "$limit": ["$take", "$limite"],
    "$project": ["$projection", "$projet"], "$unwind": ["$unwinds", "$flatten"], "$lookup": ["$join", "$lookp"],
    "$all": ["$every"], "$size": ["$length", "$len"], "$or": ["$orr", "$either"], "$and": ["$andd", "$both"],
}
OP_TOKEN = re.compile(r"\$(" + "|".join(sorted((k[1:] for k in INVENTED), key=len, reverse=True)) + r")\b")
NOT_FIELDS = {"_id"}


def tokens_with_pos(text: str):
    pos = 0
    out = []
    while pos < len(text):
        m = cv.TOKEN_RE.match(text, pos)
        if not m:
            break
        out.append((m.lastgroup, m.start(), m.end(), m.group()))
        pos = m.end()
    return out


def wrong_name(path: str, coll, rng: random.Random) -> str | None:
    seg = path.split(".")
    last = seg[-1]
    opts = []
    if len(last) >= 4:
        i = rng.randrange(1, len(last) - 1)
        opts.append(last[:i] + last[i + 1:])                         # letra faltando
        opts.append(last[:i] + last[i + 1] + last[i] + last[i + 2:] if i + 2 <= len(last) else last)  # trocadas
    opts.append(last[0].swapcase() + last[1:])                       # caixa
    opts.append(last + "s" if not last.endswith("s") else last[:-1])  # plural
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", last).lower()
    if snake != last:
        opts.append(snake)
    rng.shuffle(opts)
    for o in opts:
        np = ".".join(seg[:-1] + [o])
        if o and o != last and np not in coll.fields and o not in {x.split(".")[-1] for x in coll.fields}:
            return np
    return None


def inject_defect(text: str, plan: dict, used: list[str], coll, rng: random.Random, weights: dict, kind: str | None = None):
    """Retorna (texto_defeituoso, tipo, detalhe) ou None. Edicoes textuais; a validacao decide se de fato falha."""
    kind = kind or pick(rng, weights)
    toks = tokens_with_pos(text)
    if kind == "wrong_field":
        cands = [p for p in used if "::" not in p and p not in NOT_FIELDS and p in coll.fields]
        rng.shuffle(cands)
        for p in cands:
            bad = wrong_name(p, coll, rng)
            if not bad:
                continue
            pat = re.compile(r"(?<![\w.])" + re.escape(p) + r"(?!\w)")
            if pat.search(text):
                return pat.sub(bad, text), kind, {"from": bad, "to": p}
        return None
    if kind == "wrong_operator":
        ms = list(OP_TOKEN.finditer(text))
        if not ms:
            return None
        op = "$" + rng.choice(ms).group(1)
        inv = rng.choice(INVENTED[op])
        return re.sub(re.escape(op) + r"\b", inv.replace("$", "$"), text), kind, {"from": inv, "to": op}
    if kind == "missing_dollar":
        ms = list(OP_TOKEN.finditer(text))
        refs = list(re.finditer(r"([\"'])\$([A-Za-z_][\w.]*)\1", text))
        if refs and (not ms or rng.random() < 0.5):
            m = rng.choice(refs)
            return text[:m.start()] + m.group(1) + m.group(2) + m.group(1) + text[m.end():], kind, {"where": "field_ref", "name": m.group(2)}
        if ms:
            m = rng.choice(ms)
            return text[:m.start()] + text[m.start() + 1:], kind, {"where": "operator", "name": "$" + m.group(1)}
        return None
    if kind == "syntax_comma":
        commas = [t for t in toks if t[0] == "p" and t[3] == ","]
        if not commas:
            return None
        t = rng.choice(commas)
        return text[:t[1]] + text[t[2]:], kind, {"removed": "comma"}
    if kind == "syntax_quote":
        strs = [t for t in toks if t[0] == "str" and len(t[3]) >= 3]
        if not strs:
            return None
        t = rng.choice(strs)
        return text[:t[2] - 1] + text[t[2]:], kind, {"removed": "closing_quote"}
    if kind == "syntax_bracket":
        cl = [t for t in toks if t[0] == "p" and t[3] in ")]}"]
        if not cl:
            return None
        t = rng.choice(cl)
        return text[:t[1]] + text[t[2]:], kind, {"removed": t[3]}
    if kind == "wrong_type":
        opts = []
        for m in re.finditer(r"ObjectId\(([\"'])([0-9a-f]{24})\1\)", text):
            opts.append((m.start(), m.end(), m.group(1) + m.group(2) + m.group(1), "objectid_as_string"))
            opts.append((m.start(), m.end(), "ObjectId(" + m.group(1) + m.group(2)[:-1] + m.group(1) + ")", "objectid_short"))
        for m in re.finditer(r"(?:ISODate|new Date)\(([\"'])(\d{4}-\d{2}-\d{2})T00:00:00Z\1\)", text):
            opts.append((m.start(), m.end(), m.group(1) + m.group(2) + m.group(1), "date_as_string"))
        for m in re.finditer(r"NumberLong\(([\"'])(\d+)\1\)", text):
            opts.append((m.start(), m.end(), "NumberLong(" + m.group(2) + ")", "numberlong_unquoted"))
        for i, t in enumerate(toks):
            if t[0] == "num" and i >= 2 and toks[i - 1][3] == ":" and toks[i - 2][0] in ("id", "str"):
                key = toks[i - 2][3].strip("\"'")
                if key not in ("$limit", "$skip", "$size", "$round", "$sum") and not key.startswith("$") or key in (
                        "$gt", "$gte", "$lt", "$lte", "$eq", "$ne"):
                    opts.append((t[1], t[2], '"' + t[3] + '"', "number_as_string"))
        if not opts:
            return None
        a, b, new, sub = rng.choice(opts)
        return text[:a] + new + text[b:], kind, {"kind": sub}
    if kind == "invalid_api":
        opts = []
        body = text.rstrip()
        semi = body.endswith(";")
        core = body[:-1] if semi else body
        opts.append((core + ".pretty()" + (";" if semi else ""), {"kind": "pretty"}))
        if ".countDocuments(" in text:
            opts.append((text.replace(".countDocuments(", ".count(", 1), {"kind": "count"}))
        if ".toArray()" in text:
            opts.append((text.replace(".toArray()", ".toarray()", 1), {"kind": "toarray_case"}))
        if "ConnectionPool" in text:
            opts.append((text.replace("ConnectionPool", "ConnectionPull", 1), {"kind": "ConnectionPull"}))
        if ".getCollection(" in text:
            opts.append((text.replace(".getCollection(", ".GetCollection(", 1), {"kind": "GetCollection"}))
        for old, new, sub in ((".find(", ".Find(", "Find"), (".aggregate(", ".Aggregate(", "Aggregate"),
                              (".limit(", ".take(", "take"), (".sort(", ".orderBy(", "orderBy")):
            if old in text:
                opts.append((text.replace(old, new, 1), {"kind": sub}))
        t, d = rng.choice(opts)
        return t, kind, d
    return None


def error_message(defect_type: str, stage: str, err: str | None, text: str) -> str | None:
    """Mensagem tipo Console para a instrucao (so quando ha erro real do motor/JS)."""
    if not err:
        return None
    if stage == "syntax":
        return err
    if stage == "fields":
        m = re.match(r"campo/estrutura invalido: stage:(\w+)", err)
        if m:
            return f"MongoServerError: Unrecognized pipeline stage name: '{m.group(1)}'"
        return None
    if stage == "exec":
        return "MongoServerError: " + re.sub(r", full error:.*$", "", err)[:200]
    if stage == "api":
        m = re.match(r"metodo (?:de cursor )?inexistente no Console: (\w+)", err)
        if m:
            return f"TypeError: {m.group(1)} is not a function"
        m = re.match(r"identificador desconhecido: (\w+)", err)
        if m:
            return f"ReferenceError: {m.group(1)} is not defined"
        if err.startswith("operador/estagio inexistente"):
            return "MongoServerError: unknown operator: " + err.split(": ", 1)[1]
        return None
    return None


# --------------------------------------------------------------------------- textos fixos (negativos)

REFUSAL_WRITE = {
    "pt": ["Não gero comandos que alteram ou apagam dados.", "Não executo escrita ou exclusão por aqui; este chat só monta consultas de leitura.",
           "Pedidos que modificam ou removem dados ficam fora do chat: ele só gera consultas de leitura."],
    "en": ["I do not generate commands that modify or delete data.", "Writes and deletions are out of scope here; this chat only builds read queries.",
           "Requests that change or remove data are not handled in chat: it only generates read queries."],
}
PREVIEW = {"pt": ["Para conferir os documentos que seriam afetados:", "Uma consulta de leitura equivalente, para conferir antes:"],
           "en": ["To review the documents that would be affected:", "An equivalent read query to check first:"]}
OUT_OF_SCOPE_REPLY = {
    "pt": ["Fora do escopo: aqui eu só ajudo com consultas MongoDB.", "Isso não é uma consulta MongoDB; só consigo ajudar com o Console do MongoDB."],
    "en": ["Out of scope: I can only help with MongoDB queries here.", "That is not a MongoDB query; I can only help with the MongoDB Console."],
}
OUT_OF_SCOPE = {
    "pt": ["escreve um poema sobre o mar", "qual a capital da Austrália?", "como eu faço deploy no kubernetes", "me ajuda a montar um currículo",
           "converte esse texto para maiúsculas: bom dia", "quanto é 15% de 2300?", "escreva uma função em Python que ordena uma lista",
           "me conta uma piada", "traduz 'bom dia' para o francês", "qual a previsão do tempo para amanhã?", "explica o que é uma blockchain",
           "monta uma query SQL com JOIN entre clientes e pedidos", "resolve essa equação: 2x + 3 = 11", "sugere um nome para o meu cachorro"],
    "en": ["write a poem about the sea", "what is the capital of Australia?", "how do I deploy to kubernetes", "help me write a resume",
           "convert this text to uppercase: good morning", "what is 15% of 2300?", "write a python function that sorts a list",
           "tell me a joke", "translate 'good morning' to French", "what is the weather forecast for tomorrow?", "explain what a blockchain is",
           "build a SQL query with a JOIN between customers and orders", "solve this equation: 2x + 3 = 11", "suggest a name for my dog"],
}
GENERIC_WRITE = {
    "pt": ["Apague a coleção {c}", "dropa a coleção {c}", "Exclua o banco de dados inteiro", "limpa tudo {cin}", "Insira um documento novo {cin}",
           "Renomeie a coleção {c} para {c}_old", "Remova todos os documentos {cof}", "Crie uma coleção chamada {c}_backup e copie tudo para ela"],
    "en": ["Drop the {c} collection", "drop collection {c}", "Delete the whole database", "wipe everything {cin}", "Insert a new document {cin}",
           "Rename the {c} collection to {c}_old", "Remove all the documents {cof}", "Create a collection called {c}_backup and copy everything into it"],
}
FAKE_COLLECTIONS = ["foguetes", "planetas", "receitas", "rockets", "planets", "recipes", "historico_antigo", "old_archive", "faturas_2019",
                    "legacy_users", "satelites", "stadiums", "temp_import", "relatorios_v0"]
CORRECT_PHRASES = {
    "pt": ["Corrija essa consulta", "corrige essa query aí", "Essa consulta está com erro, arruma pra mim", "Conserte a consulta do editor",
           "tem algo errado nisso, corrige sem mudar o que ela faz", "Revise e corrija a consulta"],
    "en": ["Fix this query", "fix this query for me", "This query has an error, please correct it", "Repair the query in the editor",
           "something is wrong here, fix it without changing what it does", "Review and fix the query"],
}
CORRECT_WITH_ERR = {
    "pt": ["Deu este erro: {e}. Corrija a consulta.", "{e}\ncorrige aí", "Erro no Console: {e}. Pode arrumar?"],
    "en": ["I got this error: {e}. Fix the query.", "{e}\nplease fix", "Console error: {e}. Can you repair it?"],
}
CORRECT_SILENT = {
    "pt": ["A consulta retorna 0 documentos mas deveria retornar resultados. Corrija.", "Não volta nada e eu esperava resultados, corrige a consulta"],
    "en": ["The query returns 0 documents but should return results. Fix it.", "It returns nothing and I expected results, fix the query"],
}
EXPLAIN_ASK = {
    "pt": ["Explique essa consulta", "o que essa query faz?", "Explica rapidinho o que isso faz", "Descreva o que o código do editor faz"],
    "en": ["Explain this query", "what does this query do?", "Briefly explain what this does", "Describe what the editor code does"],
}
COMMENTS = ["// consultas", "// TODO: filtrar", "// rascunho", "// scratch"]


def default_command(plan_coll_name: str, sch, coll, cfg_style, rng):
    plan = rng.choice([{"op": "find", "filter": {}}, {"op": "find", "filter": {}, "limit": 20},
                       {"op": "countDocuments", "filter": {}}])
    st_cfg = copy.deepcopy(cfg_style)
    st_cfg["snippets"].update({"layouts": {"compact": 1.0}, "crlf_ratio": 0.0, "const_extract": 0.0, "root_forms": {"auto": 1.0},
                               "to_array": 0.0, "semicolon": 0.0})
    text, _ = render_text(plan, sch, coll, st_cfg, rng)
    return text


# --------------------------------------------------------------------------- gerador

class ChatGen:
    def __init__(self, cfg: dict, snippets: list[dict], validation: dict, schemas, executor, node):
        self.cfg = cfg
        self.cc = cfg["chat"]
        self.cfg_style = style_cfg(cfg)
        self.sch = {s.full_id: s for s in schemas}
        self.ex, self.node = executor, node
        self.seed = cfg.get("seed", 20261008)
        reads = [r for r in snippets if r["kind"] == "read" and validation.get(r["id"], {}).get("exec") == "ok"
                 and not validation[r["id"]].get("empty")]
        n_all = len(reads)
        reads = [r for r in reads if not order_dependent(r["plan"])]
        self.order_dependent_dropped = n_all - len(reads)
        writes = [r for r in snippets if r["kind"] == "write"]
        order = random.Random(f"{self.seed}:chat:order")
        order.shuffle(reads)
        order.shuffle(writes)
        self.reads, self.writes = reads, writes
        self.ri = self.wi = 0
        self.stats = Counter()
        self.discards = Counter()
        self.seen: set = set()
        self.exec_attempts = self.exec_ok = 0
        self.defect_attempts = self.defect_failed = 0

    # ---- amostragem de snippet ----
    def next_read(self):
        r = self.reads[self.ri % len(self.reads)]
        self.ri += 1
        return r

    def next_write(self):
        r = self.writes[self.wi % len(self.writes)]
        self.wi += 1
        return r

    # ---- util ----
    def sch_coll(self, snip):
        sch = self.sch[snip["schema_id"]]
        coll = next(c for c in sch.colls if c.name == snip["collection"])
        return sch, coll

    def validate(self, text, sch, coll, snip):
        syn = self.node.check(text)
        self.exec_attempts += 1
        v = cv.validate_text(text, sch, coll, self.ex, syn, db=snip["db"])
        if v["stage"] == "ok":
            self.exec_ok += 1
        return v

    def base_record(self, i, kind, lang, snip, sch, coll, template_id, instruction, context, output):
        return {
            "id": f"chat-{snip['domain']}.{sch.schema_id}.{coll.key.replace(' ', '_')}-{kind}-{i:06d}-{'pt' if lang == 'pt' else 'en'}",
            "schema_version": SCHEMA_VERSION,
            "type": "chat",
            "task": "edit_short" if kind != "explain" else "explain_short",
            "kind": kind,
            "language": LANG_CODE[lang],
            "domain": snip["domain"],
            "schema_id": snip["schema_id"],
            "collection": snip["collection"],
            "group": snip["group_id"],
            "group_id": snip["group_id"],
            "split_key": snip["split_key"],
            "split": snip["split"],
            "holdout": snip["holdout"],
            "template_id": template_id,
            "snippet_id": snip["id"],
            "instruction": instruction,
            "context": context,
            "output": output,
            "origin": "gen:chat",
            "source_license": "generated",
            "label_source": "reference",
            "weight": 1.0,
        }

    def make_context(self, rng, snip, sch, coll, editor, collection_set=True, info=None):
        conn = "local" if (info and info.get("root_form") == "pool") else rng.choice(CONNECTIONS)
        known = sorted({c.name for c in sch.colls})[:40]
        return {"connection": conn, "database": sch.schema_id, "collection": coll.name if collection_set else None,
                "selection": None, "editor": editor, "known_collections": known}

    def instruction(self, rng, plan, coll_name, lang, reg, amb):
        d = cd.Describer(lang, rng, reg, amb)
        raw = d.describe(plan, coll_name)
        return raw

    def post(self, rng, raw, lang, reg, info=None):
        return cd.finalize(raw, lang, rng, reg, self.cc.get("pt_deaccent", 0.18), self.cc.get("typo", 0.04), info)

    # ---- chat ----
    def gen_chat(self, i, rng, lang, reg):
        snip = self.next_read()
        sch, coll = self.sch_coll(snip)
        plan = snip["plan"]
        amb = rng.random() < self.cc.get("ambiguous_collection", 0.10)
        text, info = render_text(plan, sch, coll, self.cfg_style, rng)
        modify = False
        removed_stages: list = []
        if rng.random() < self.cc.get("modify_fraction", 0.5):
            if snip["op"] == "find" and any(plan.get(k) is not None for k in ("sort", "limit", "projection", "skip")):
                modify = True
            elif snip["op"] == "aggregate":
                st = plan["pipeline"]
                k = len(st)
                while k > 1 and list(st[k - 1])[0] in ("$sort", "$limit", "$skip"):
                    k -= 1
                if k < len(st):
                    modify = True
                    removed_stages = st[k:]
        if modify and snip["op"] == "find":
            present = [k for k in ("sort", "skip", "limit", "projection") if plan.get(k) is not None]
            removed = rng.sample(present, rng.randint(1, len(present)))
            base_plan = {k: v for k, v in plan.items() if k not in removed}
            base_text = self.render_same_style(base_plan, sch, coll, text, rng)
            if base_text is None:
                self.discards["modify_base_render"] += 1
                return None
            d = cd.Describer(lang, rng, reg, False)
            raw = self.modify_text(d, plan, removed, lang)
            editor = base_text
            tid = "chat.modify." + snip["template_id"]
            intent = {"kind": "modify", "operations": ["find"] + present, "added": removed}
            lit_ok = True
        elif modify:
            base_plan = {"op": "aggregate", "pipeline": plan["pipeline"][:len(plan["pipeline"]) - len(removed_stages)]}
            base_text = self.render_same_style(base_plan, sch, coll, text, rng)
            if base_text is None:
                self.discards["modify_base_render"] += 1
                return None
            d = cd.Describer(lang, rng, reg, False)
            steps = [d.stage_text(x) for x in removed_stages]
            lead = d.ch(["Agora ", "Na agregação do editor, ", "Altere o pipeline: ", "Acrescente ao final: "],
                        ["Now ", "In the editor's aggregation, ", "Update the pipeline: ", "Add at the end: "])
            raw = lead + d.join(steps)
            editor = base_text
            tid = "chat.modify." + snip["template_id"]
            intent = {"kind": "modify", "operations": self.operations(plan), "added": [list(x)[0] for x in removed_stages]}
            lit_ok = True
        else:
            raw = self.instruction(rng, plan, coll.name, lang, reg, amb)
            r = rng.random()
            if r < 0.55:
                editor = ""
            elif r < 0.85:
                editor = default_command(coll.name, sch, coll, self.cfg_style, rng)
            else:
                editor = rng.choice(COMMENTS)
            tid = "chat.create." + snip["template_id"]
            intent = {"kind": "create", "operations": self.operations(plan)}
            if amb:
                intent["ambiguous_collection"] = True
            lit_ok = literals_present(plan, raw)
        if not lit_ok:
            self.discards["literal_missing"] += 1
            return None
        pinfo: dict = {}
        instr = self.post(rng, raw, lang, reg, pinfo)
        v = self.validate(text, sch, coll, snip)
        if v["stage"] != "ok":
            self.discards["output_" + v["stage"]] += 1
            return None
        if not cv.plans_equal(v["plan"], plan):
            self.discards["output_plan_mismatch"] += 1
            return None
        if v["n_docs"] == 0:
            self.discards["output_empty"] += 1
            return None
        ctx = self.make_context(rng, snip, sch, coll, editor, True, info)
        rec = self.base_record(i, "chat", lang, snip, sch, coll, tid, instr, ctx, text)
        rec["intent"] = intent
        rec["register"] = reg
        rec["pt_accents_stripped"] = bool(pinfo.get("deaccent"))
        rec["typo_injected"] = bool(pinfo.get("typo"))
        rec["validation"] = {"syntax": "ok", "api": "ok", "fields": "ok", "types": "ok", "exec": "ok", "nonempty": True,
                             "plan_equal": True, "fixture": snip["db"]}
        return rec

    def render_same_style(self, base_plan, sch, coll, text, rng):
        """Renderiza o editor base com estilo parecido com o do output (CRLF, aspas)."""
        try:
            cfg = copy.deepcopy(self.cfg_style)
            sn = cfg["snippets"]
            sn["crlf_ratio"] = 1.0 if "\r\n" in text else 0.0
            sn["quotes"] = {"single": 1.0} if "'" in text and '"' not in text else {"double": 1.0}
            sn["root_forms"] = {"auto": 1.0}
            sn["const_extract"] = 0.0
            t, _ = render_text(base_plan, sch, coll, cfg, rng)
            return t
        except Exception:
            return None

    def operations(self, plan):
        if plan["op"] == "aggregate":
            return ["aggregate"] + [list(s)[0].lstrip("$") for s in plan["pipeline"]]
        ops = [plan["op"]]
        for k in ("sort", "skip", "limit", "projection"):
            if plan.get(k) is not None:
                ops.append(k)
        return ops

    def modify_text(self, d: cd.Describer, plan, removed, lang):
        bits = []
        for k in removed:
            if k == "sort":
                bits.append(d.ch([f"ordene por {d.sort_text(plan['sort'])}", f"ordenando por {d.sort_text(plan['sort'])}"],
                                 [f"sort by {d.sort_text(plan['sort'])}", f"order by {d.sort_text(plan['sort'])}"]))
            elif k == "limit":
                n = plan["limit"]
                bits.append(d.ch([f"limite a {n}", f"só os {n} primeiros", f"no máximo {n}"], [f"limit to {n}", f"only the first {n}", f"at most {n}"]))
            elif k == "skip":
                n = plan["skip"]
                bits.append(d.ch([f"pule {n}", f"ignore os {n} primeiros"], [f"skip {n}", f"skip the first {n}"]))
            elif k == "projection":
                bits.append(d.proj_text(plan["projection"]))
        body = d.join(bits) if len(bits) < 3 else ", ".join(bits)
        lead = d.ch(["Agora ", "Na consulta do editor, ", "Altere a consulta: ", "Mantenha o filtro e ", ""],
                    ["Now ", "In the editor query, ", "Update the query: ", "Keep the filter and ", ""])
        return lead + body

    # ---- correction ----
    def gen_correction(self, i, rng, lang, reg):
        want = pick(rng, self.cc["defects"])      # familia de defeito sorteada primeiro (mantem a distribuicao alvo)
        for _ in range(10):                       # ate 10 snippets ate achar um que admita a familia
            snip = self.next_read()
            sch, coll = self.sch_coll(snip)
            plan = snip["plan"]
            text, info = render_text(plan, sch, coll, self.cfg_style, rng)
            v = self.validate(text, sch, coll, snip)
            if v["stage"] != "ok" or not cv.plans_equal(v["plan"], plan) or v["n_docs"] == 0:
                self.discards["correction_reference_invalid"] += 1
                continue
            for _t in range(4):
                inj = inject_defect(text, plan, v.get("fields_used", snip["fields_used"]), coll, rng, self.cc["defects"], kind=want)
                if inj is None or inj[0] == text:
                    break
                bad, dtype, detail = inj
                self.defect_attempts += 1
                syn = self.node.check(bad)
                vb = cv.validate_text(bad, sch, coll, self.ex, syn, db=snip["db"])
                if vb["stage"] == "ok":
                    # o defeituoso passa em tudo: nao e um defeito detectavel -> descarta (spec: tem de FALHAR)
                    self.discards["defect_not_failing"] += 1
                    continue
                self.defect_failed += 1
                stage, err = vb["stage"], vb["error"]
                msg = error_message(dtype, stage, err, bad)
                if rng.random() < self.cc.get("correction_error_message", 0.5):
                    if msg:
                        raw = rng.choice(CORRECT_WITH_ERR[lang]).format(e=msg)
                    elif stage in ("fields", "type"):
                        raw = rng.choice(CORRECT_SILENT[lang])
                    else:
                        raw = rng.choice(CORRECT_PHRASES[lang])
                else:
                    raw = rng.choice(CORRECT_PHRASES[lang])
                instr = raw if chr(10) in raw else self.post_plain(rng, raw, lang, reg)
                if msg and msg not in instr:
                    instr = raw   # a mensagem de erro e literal: nao passa por acento/caixa
                ctx = self.make_context(rng, snip, sch, coll, bad, True, info)
                rec = self.base_record(i, "correction", lang, snip, sch, coll, f"correction.{dtype}." + snip["template_id"], instr, ctx, text)
                rec["intent"] = {"kind": "correction", "operations": self.operations(plan)}
                rec["defect"] = {"type": dtype, "detail": detail, "failed_stage": stage, "error": err,
                                 "error_in_instruction": bool(msg and msg in instr)}
                rec["register"] = reg
                rec["validation"] = {"syntax": "ok", "api": "ok", "fields": "ok", "types": "ok", "exec": "ok", "nonempty": True,
                                     "plan_equal": True, "fixture": snip["db"],
                                     "defective": {"stage": stage, "error": err}}
                return rec
        self.discards["correction_no_defect"] += 1
        return None

    def post_plain(self, rng, raw, lang, reg):
        return cd.finalize(raw, lang, rng, reg, self.cc.get("pt_deaccent", 0.18), 0.0)

    # ---- explain ----
    def gen_explain(self, i, rng, lang, reg):
        snip = self.next_read()
        sch, coll = self.sch_coll(snip)
        plan = snip["plan"]
        text, info = render_text(plan, sch, coll, self.cfg_style, rng)
        d = cd.Describer(lang, rng, "n", False)
        try:
            expl = cd.strip_marks(d.explain(plan, coll.name))
        except cd.Unsupported:
            self.discards["explain_unsupported"] += 1
            return None
        if len(expl) > 320 or "\n" in expl:
            self.discards["explain_long"] += 1
            return None
        v = self.validate(text, sch, coll, snip)
        if v["stage"] != "ok" or not cv.plans_equal(v["plan"], plan) or v["n_docs"] == 0:
            self.discards["output_" + v["stage"]] += 1
            return None
        out = "// " + expl + "\n" + text
        instr = self.post_plain(rng, rng.choice(EXPLAIN_ASK[lang]), lang, reg)
        ctx = self.make_context(rng, snip, sch, coll, text, True, info)
        rec = self.base_record(i, "explain", lang, snip, sch, coll, "explain." + snip["template_id"], instr, ctx, out)
        rec["intent"] = {"kind": "explain", "operations": self.operations(plan)}
        rec["register"] = reg
        rec["validation"] = {"syntax": "ok", "api": "ok", "fields": "ok", "types": "ok", "exec": "ok", "nonempty": True,
                             "plan_equal": True, "fixture": snip["db"]}
        return rec

    # ---- negativos ----
    def gen_negative(self, i, rng, lang, reg):
        ntype = pick(rng, self.cc["negative_types"])
        if ntype == "write":
            snip = self.next_write()
            sch, coll = self.sch_coll(snip)
            d = cd.Describer(lang, rng, reg, False)
            generic = rng.random() < 0.3
            try:
                if generic:
                    t = rng.choice(GENERIC_WRITE[lang])
                    raw = t.format(c=cd.F(coll.name), cin=d.coll_in(coll.name), cof=d.coll_of(coll.name))
                else:
                    raw = d.describe_write(snip["plan"], coll.name)
            except cd.Unsupported:
                self.discards["negative_unsupported"] += 1
                return None
            out_lines = [rng.choice(REFUSAL_WRITE[lang])]
            preview = None
            flt = snip["plan"].get("filter") or {}
            if not generic and flt and rng.random() < self.cc.get("preview_in_write_refusal", 0.35):
                pplan = {"op": "find", "filter": flt, "limit": 20}
                ptxt, _ = render_text(pplan, sch, coll, self.cfg_style, rng)
                pv = self.validate(ptxt, sch, coll, snip)
                if pv["stage"] == "ok" and cv.plans_equal(pv["plan"], pplan):
                    preview = ptxt
                else:
                    self.discards["negative_preview_invalid"] += 1
            out = refusal(out_lines) + (("\n" + refusal([rng.choice(PREVIEW[lang])]) + "\n" + preview) if preview else "")
            editor = "" if rng.random() < 0.6 else default_command(coll.name, sch, coll, self.cfg_style, rng)
            ctx = self.make_context(rng, snip, sch, coll, editor, True)
            tid = "negative.write." + ("generic" if generic else snip["template_id"])
            ndet = {"type": "write", "generic": generic, "preview": preview is not None, "requested_op": snip["op"] if not generic else None}
        elif ntype == "out_of_scope":
            snip = self.next_read()
            sch, coll = self.sch_coll(snip)
            raw = rng.choice(OUT_OF_SCOPE[lang])
            out = refusal([rng.choice(OUT_OF_SCOPE_REPLY[lang])])
            ctx = self.make_context(rng, snip, sch, coll, "" if rng.random() < 0.7 else default_command(coll.name, sch, coll, self.cfg_style, rng), True)
            tid = "negative.out_of_scope"
            ndet = {"type": "out_of_scope"}
        elif ntype == "no_collection":
            snip = self.next_read()
            sch, coll = self.sch_coll(snip)
            known = {c.name.lower() for c in sch.colls}
            fakes = [f for f in FAKE_COLLECTIONS if f.lower() not in known]
            fake = rng.choice(fakes)
            d = cd.Describer(lang, rng, reg, False)
            try:
                raw = d.describe(snip["plan"], fake)
            except cd.Unsupported:
                self.discards["negative_unsupported"] += 1
                return None
            names = sorted({c.name for c in sch.colls})
            shown = ", ".join(names[:6]) + (", ..." if len(names) > 6 else "")
            if lang == "pt":
                out = refusal([f'Não encontrei a coleção "{fake}" neste banco. Coleções disponíveis: {shown}.', "Qual delas devo consultar?"])
            else:
                out = refusal([f'I could not find a collection named "{fake}" in this database. Available collections: {shown}.',
                               "Which one should I query?"])
            ctx = self.make_context(rng, snip, sch, coll, "", rng.random() < 0.5)
            tid = "negative.no_collection"
            ndet = {"type": "no_collection", "requested": fake}
        else:  # ambiguous
            snip = self.next_read()
            sch, coll = self.sch_coll(snip)
            d = cd.Describer(lang, rng, reg, True)
            try:
                raw = d.describe(snip["plan"], coll.name)
            except cd.Unsupported:
                self.discards["negative_unsupported"] += 1
                return None
            out = refusal(["Qual coleção devo consultar? Selecione uma coleção ou informe o nome."] if lang == "pt"
                          else ["Which collection should I query? Select a collection or give me its name."])
            ctx = self.make_context(rng, snip, sch, coll, "", False)
            tid = "negative.ambiguous"
            ndet = {"type": "ambiguous"}
        pinfo: dict = {}
        instr = self.post(rng, raw, lang, reg, pinfo)
        if cv.has_executable_write(out):
            self.discards["negative_has_write"] += 1
            return None
        rec = self.base_record(i, "negative", lang, snip, sch, coll, tid, instr, ctx, out)
        rec["intent"] = {"kind": "negative", "operations": []}
        rec["negative"] = ndet
        rec["register"] = reg
        rec["pt_accents_stripped"] = bool(pinfo.get("deaccent"))
        rec["typo_injected"] = bool(pinfo.get("typo"))
        rec["validation"] = {"syntax": "ok", "api": "ok", "fields": "ok", "exec": "ok" if ndet.get("preview") else "skipped",
                             "writes_in_output": 0, "fixture": snip["db"]}
        return rec

    # ---- loop ----
    def guard_ok(self, rec) -> bool:
        ctx = rec["context"]
        lines = [f"Conexão (ID lógico): {ctx['connection']}", f"Banco: {ctx['database']}"]
        if ctx["collection"]:
            lines.append(f"Coleção: {ctx['collection']}")
        if ctx["editor"]:
            lines += ["", "Conteúdo do editor:", ctx["editor"]]
        variants = [ctx["editor"], "\n".join(lines), json.dumps(ctx, ensure_ascii=False, sort_keys=True)]
        try:
            blind_guard.check_record(rec)
            for v in variants:
                blind_guard.check_text(blind_guard.prompt_text({"kind": "chat", "instruction": rec["instruction"], "context": v}))
        except blind_guard.BlindAccessError:
            return False
        return True

    def generate(self, count: int) -> list[dict]:
        out: list[dict] = []
        kw = self.cc["kinds"]
        attempts = 0
        # alocacao deterministica por quota (evita deriva de proporcoes por descartes)
        quota = {k: int(round(count * kw[k] / sum(kw.values()))) for k in kw}
        quota["chat"] += count - sum(quota.values())
        got = Counter()
        i = 0
        while sum(got.values()) < count and attempts < count * 12:
            attempts += 1
            rng = random.Random(f"{self.seed}:chat:{i}")
            i += 1
            open_kinds = [k for k in kw if got[k] < quota[k]]
            if not open_kinds:
                break
            kind = rng.choices(open_kinds, weights=[quota[k] - got[k] for k in open_kinds])[0]
            lang = "pt" if rng.random() < self.cc["languages"].get("pt-BR", 0.5) else "en"
            reg = pick(rng, self.cc["register"])
            fn = {"chat": self.gen_chat, "correction": self.gen_correction, "explain": self.gen_explain, "negative": self.gen_negative}[kind]
            try:
                rec = fn(i, rng, lang, reg)
            except cd.Unsupported:
                self.discards["describer_unsupported"] += 1
                continue
            if rec is None:
                continue
            if "<|" in rec["output"] + rec["instruction"] or "```" in rec["output"] + rec["instruction"]:
                self.discards["special_tokens"] += 1
                continue
            if "\x01" in rec["instruction"] or "\x02" in rec["instruction"]:
                self.discards["marker_leak"] += 1
                continue
            if len(rec["instruction"]) > self.cc.get("max_instruction_chars", 520):
                self.discards["instruction_too_long"] += 1
                continue
            key = (rec["schema_id"], rec["collection"], norm_text(rec["instruction"]), rec["output"] if rec["kind"] != "correction" else rec["context"]["editor"])
            if key in self.seen:
                self.discards["duplicate"] += 1
                continue
            if not self.guard_ok(rec):
                self.discards["blind_guard"] += 1
                continue
            self.seen.add(key)
            rec["sha256"] = hashlib.sha256(json.dumps(rec, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            out.append(rec)
            got[kind] += 1
        self.shortfall = {k: quota[k] - got[k] for k in quota if quota[k] > got[k]}
        return out


# --------------------------------------------------------------------------- relatorio / main

def table(rows: list[tuple], header: tuple) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def build_report(recs: list[dict], gen: ChatGen, cfg: dict, ver: str, elapsed: float) -> str:
    n = len(recs)
    by_kind = Counter(r["kind"] for r in recs)
    by_lang = Counter(r["language"] for r in recs)
    splits = ("train", "val", "test", "bench")
    kl = [(k, by_kind[k], f"{by_kind[k] / n:.1%}", *(sum(1 for r in recs if r["kind"] == k and r["language"] == l) for l in ("pt-BR", "en")),
           *(sum(1 for r in recs if r["kind"] == k and r["split"] == s) for s in splits)) for k in KINDS]
    doms = sorted({r["domain"] for r in recs})
    dl = [(d, sum(1 for r in recs if r["domain"] == d), *(sum(1 for r in recs if r["domain"] == d and r["split"] == s) for s in splits)) for d in doms]
    corr = [r for r in recs if r["kind"] == "correction"]
    dt = Counter(r["defect"]["type"] for r in corr)
    dstage = Counter(r["defect"]["failed_stage"] for r in corr)
    neg = Counter(r["negative"]["type"] for r in recs if r["kind"] == "negative")
    chat = [r for r in recs if r["kind"] == "chat"]
    intents = Counter(r["intent"]["kind"] for r in chat)
    amb = sum(1 for r in chat if r["intent"].get("ambiguous_collection"))
    reg = Counter(r["register"] for r in recs)
    pt = [r for r in recs if r["language"] == "pt-BR"]
    noacc = sum(1 for r in pt if r.get("pt_accents_stripped"))
    typos = sum(1 for r in recs if r.get("typo_injected"))
    code_recs = [r for r in recs if r["kind"] in ("chat", "correction", "explain")]
    ex = gen.exec_ok / gen.exec_attempts if gen.exec_attempts else 0
    nonempty = sum(1 for r in code_recs if r["validation"].get("nonempty"))
    mix = cfg["chat"]["mixture"]
    holdout_bad = [r for r in recs if r["holdout"] and r["split"] not in ("test", "bench")]
    viol = gen.ex.violations
    L = []
    L.append(f"# Dataset {ver} - chat curto / correcao / negativos / explain (relatorio)\n")
    L.append(f"Gerado com `gen_chat.py --ver {ver}` (seed {cfg.get('seed')}, configs/dataset.yaml `chat:`), {n} registros em {elapsed:.0f} s. "
             "Formato estruturado independente de contrato (`instruction`, `context`, `output`); nenhum renderizador `local-agent-fim-v0` aplicado.\n")
    L.append("## Totais por kind / idioma / split\n")
    L.append(table(kl, ("kind", "n", "%", "pt-BR", "en") + splits))
    L.append(f"\nIdiomas: pt-BR {by_lang['pt-BR']} ({by_lang['pt-BR'] / n:.1%}), en {by_lang['en']} ({by_lang['en'] / n:.1%}). "
             f"Registro: {dict(reg)}. pt-BR com acentos removidos: {noacc} de {len(pt)} "
             f"(config `pt_deaccent` = {cfg['chat'].get('pt_deaccent')}); erros de digitacao injetados: {typos}.\n")
    L.append(f"Chat: intents {dict(intents)}; referencia ambigua a colecao em {amb} ({amb / max(1, len(chat)):.1%}) dos chat 'create'.\n")
    L.append("## Por dominio e split\n")
    L.append(table(dl, ("dominio", "n") + splits))
    L.append("\nDominios holdout (education, legal) ficam so em test/bench; "
             f"violacoes de holdout: {len(holdout_bad)}. Split e `group` herdados do snippet (nenhum grupo cruza splits).\n")
    L.append("## Validacao (exec-valid)\n")
    L.append(f"- Execucoes de `output`/referencia tentadas: {gen.exec_attempts}; ok: {gen.exec_ok} (**exec-valid {ex:.2%}**; "
             "sintaxe node + API + campos/tipos + execucao em kapi_fx_* + plano reconstruido == plano de origem).")
    L.append(f"- Registros de codigo com resultado nao vazio: {nonempty}/{len(code_recs)} ({nonempty / max(1, len(code_recs)):.1%}); vazios sao descartados.")
    L.append(f"- Snippets de leitura com resultado dependente de ordem instavel ($group/$sortByCount + $limit/$skip sem $sort) "
             f"excluidos do pool de chat: {gen.order_dependent_dropped}.")
    L.append(f"- Comandos de escrita vistos pelo servidor durante a validacao: {len(viol)} {sorted(set(viol))}.")
    L.append(f"- Outputs com escrita executavel (escrita/DDL/$out/$merge fora de comentario): "
             f"{sum(1 for r in recs if cv.has_executable_write(r['output']))}.\n")
    L.append("## Correcao: defeitos e falha do input defeituoso\n")
    L.append(table([(k, v, f"{v / max(1, len(corr)):.1%}") for k, v in dt.most_common()], ("defeito", "n", "%")))
    L.append("\nEtapa em que o input defeituoso falhou: " + ", ".join(f"{k}={v}" for k, v in dstage.most_common()) +
             f". Tentativas de defeito: {gen.defect_attempts}; todo par mantido tem input defeituoso que falha e corrigido que passa "
             f"(`semantic` = valido mas com plano diferente do pretendido: {dstage.get('semantic', 0)}).\n")
    L.append("## Negativos\n")
    L.append(table([(k, v) for k, v in neg.most_common()], ("tipo", "n")))
    L.append("\nRecusas de escrita/exclusao sao comentarios `//`; 35% trazem uma consulta `find` de leitura equivalente (executada e validada).\n")
    L.append("## Descartes (geracao)\n")
    L.append(table(gen.discards.most_common(), ("motivo", "n")))
    L.append(f"\nShortfall de quota: {getattr(gen, 'shortfall', {})}.\n")
    L.append("## Mistura alvo (config `chat.mixture`, fracao do treino total)\n")
    L.append(table([(k, v) for k, v in mix.items()], ("componente", "fracao")))
    L.append(f"\nchat+explain+correcao = {mix['chat'] + mix['correction']:.0%} (limite 25%); negativos = {mix['negative']:.0%}.\n")
    L.append("## Limites conhecidos\n")
    for s in [
        "Instrucoes vem de templates parametrizados pelo plano (nenhum LLM): paraphrases variam, mas a cobertura e a de `gen_snippets` "
        "(estagios e operadores do plano); construcoes nao descritas (`$lookup` com `let/pipeline`, `$expr` com variaveis, etc.) sao descartadas. "
        "Risco v1: saturar benchmark programatico e falhar em pedido livre -> o blind escrito a mao continua sendo o gate.",
        "Valores citados aparecem literalmente; nomes de campo aparecem como no schema (sem sinonimos de dominio).",
        "Validacao sem ContractProbe/Acornima: sintaxe por `node vm.Script`, API por parser do subconjunto Console proprio; $search/$vectorSearch "
        "e estagios fora da allowlist nao entram.",
        "Correcao: defeitos sao edicoes textuais do output (8 familias); `semantic` e raro por construcao. Mensagens de erro de sintaxe vem do node (V8), nao do Acornima da IDE.",
        "Negativos: `known_collections` no contexto e necessario para `no_collection`; o contrato `local-agent-fim-v0` atual nao carrega essa lista (decisao pendente).",
        "Referencia ambigua so de colecao; nao ha chat multi-turno nem uso de `selection` (sempre null) nesta fatia.",
        "Dominios holdout so alimentam test/bench; train de chat depende dos mesmos grupos do FIM (sem vazamento entre eles).",
    ]:
        L.append("- " + s)
    return "\n".join(L) + "\n"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--count", type=int)
    ap.add_argument("--ver")
    ap.add_argument("--snippets")
    ap.add_argument("--validation")
    ap.add_argument("--out")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg["seed"] = a.seed
    ver = a.ver or cfg.get("version", "v0")
    gdir = ROOT / cfg.get("out_dir", "data/generated") / ver
    snips = load_jsonl(Path(a.snippets) if a.snippets else gdir / "snippets.jsonl")
    val = {r["id"]: r for r in load_jsonl(Path(a.validation) if a.validation else gdir / "validation.jsonl")}
    schemas = load_schemas(ROOT / cfg.get("schemas_dir", "data/schemas"))
    count = a.count or cfg["chat"]["count"]
    t0 = time.perf_counter()
    ex = cv.Executor(cfg)
    node = cv.NodeChecker()
    try:
        gen = ChatGen(cfg, snips, val, schemas, ex, node)
        recs = gen.generate(count)
    finally:
        node.close()
    out = Path(a.out) if a.out else gdir / "chat.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            line = json.dumps(r, ensure_ascii=False, separators=(",", ":"))
            f.write(line + "\n")
            h.update(line.encode("utf-8"))
    n = max(1, len(recs))
    man = {"seed": cfg["seed"], "count": len(recs), "requested": count, "sha256": h.hexdigest(),
           "kinds": dict(Counter(r["kind"] for r in recs)), "languages": dict(Counter(r["language"] for r in recs)),
           "splits": {s: sum(1 for r in recs if r["split"] == s) for s in ("train", "val", "test", "bench")},
           "exec_valid_rate": round(gen.exec_ok / gen.exec_attempts, 4) if gen.exec_attempts else None,
           "discards": dict(gen.discards), "shortfall": gen.shortfall, "write_commands_seen": len(ex.violations),
           "node_syntax_check": node_ok(), "mixture": cfg["chat"]["mixture"]}
    (out.parent / "chat.manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps(man))
    if a.report:
        rp = ROOT / "reports" / "datasets" / f"{ver}-chat-report.md"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(build_report(recs, gen, cfg, ver, time.perf_counter() - t0), encoding="utf-8")
    if ex.violations:
        sys.exit(3)


def node_ok() -> bool:
    return cv.node_available()


if __name__ == "__main__":
    main()
