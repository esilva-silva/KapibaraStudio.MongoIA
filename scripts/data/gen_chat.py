#!/usr/bin/env python3
"""Gerador de chat curto / correcao / negativos / explain (spec fase-03 §4.3 `gen_chat.py`, §4.4, §4.5).

  python scripts/data/gen_chat.py --ver v1 [--count 6000] [--seed N] [--report]
  python scripts/data/gen_chat.py --ver v1 --capacity-report [--capacity-sample 6] [--capacity-rounds 3] [--capacity-target 15000]
      (mede registros unicos por familia/schema e aponta o gargalo; nao grava chat.jsonl)

Ciclo 2: kinds chat | chat_multiturn (2 turnos, campo `history`) | correction (19 familias de defeito, ate 4 combinados) |
negative (write, ddl, out_of_scope, no_collection, ambiguous, ambiguous_field) | explain. Registros de instrucao n/i/t/g/q.
`split` herda do snippet; schema com `split_role` (train|eval) incompativel com o split do snippet tira o snippet do pool.

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
import hom_guard  # noqa: E402
from gen_snippets import CONFIG_PATH, ROOT, Ctor, Style, load_config, load_schemas, render_snippet  # noqa: E402

SCHEMA_VERSION = "kapi-ds-record-v2"
LANG_CODE = {"pt": "pt-BR", "en": "en"}
CONNECTIONS = ["dev-cluster", "staging", "prod-readonly", "atlas-dev", "homolog", "analytics", "local"]
KINDS = ("chat", "chat_multiturn", "correction", "negative", "explain")


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


def inject_defect(text: str, plan: dict, used: list[str], coll, rng: random.Random, weights: dict, kind: str | None = None,
                  known_names=()):
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
    if kind in NEW_DEFECTS:
        toks = sig_tokens(text)
        r = {
            "misplaced_operator": lambda: d_misplaced_operator(text, toks, rng),
            "group_no_id": lambda: d_group_no_id(text, toks, rng),
            "type_mismatch": lambda: d_type_mismatch(text, toks, coll, rng),
            "lookup_field": lambda: d_lookup_field(text, toks, coll, set(known_names), rng),
            "stage_order": lambda: d_stage_order(text, toks, rng),
            "unknown_method": lambda: d_unknown_method(text, rng),
            "hint_index": lambda: d_hint_index(text, toks, coll, rng),
            "projection_mix": lambda: d_projection_mix(text, toks, rng),
            "sort_direction": lambda: d_sort_direction(text, toks, rng),
            "limit_value": lambda: d_limit_value(text, toks, rng),
            "wrong_collection": lambda: d_wrong_collection(text, coll, set(known_names), rng),
        }
        try:
            r = r[kind]()
        except (IndexError, KeyError):      # texto truncado/incomum: sem defeito desta familia
            return None
        if r is None:
            return None
        return r[0], kind, r[1]
    return None


# --------------------------------------------------------------------------- defeitos novos (ciclo 2)

OLD_DEFECTS = ("wrong_field", "wrong_operator", "missing_dollar", "syntax_comma", "syntax_quote", "syntax_bracket", "wrong_type",
               "invalid_api")
NEW_DEFECTS = ("misplaced_operator", "group_no_id", "type_mismatch", "lookup_field", "stage_order", "unknown_method", "hint_index",
               "projection_mix", "sort_direction", "limit_value", "wrong_collection")
ALL_DEFECTS = OLD_DEFECTS + NEW_DEFECTS
QUERY_CMP = ["$gte", "$gt", "$lte", "$lt", "$ne", "$eq", "$in", "$nin", "$exists", "$regex"]
AGG_ACC = ["$sum", "$avg", "$max", "$min", "$first", "$push", "$addToSet"]
ACC_AS_QUERY = ["$gt", "$in", "$ne", "$exists", "$regex"]
METHOD_SWAP = {
    ".find(": [".findAll(", ".select(", ".query(", ".getDocuments("],
    ".findOne(": [".first(", ".findFirst(", ".getOne("],
    ".aggregate(": [".agg(", ".pipeline(", ".aggregation("],
    ".countDocuments(": [".length(", ".size(", ".total("],
    ".distinct(": [".unique(", ".distinctValues("],
}


def sig_tokens(text: str):
    return [t for t in tokens_with_pos(text) if t[0] not in ("ws", "lc", "bc")]


def key_name(tok) -> str | None:
    if tok[0] == "id":
        return tok[3]
    if tok[0] == "str":
        return tok[3][1:-1]
    return None


def match_close(toks, i: int) -> int | None:
    depth = 0
    for j in range(i, len(toks)):
        k, _s, _e, t = toks[j]
        if k == "p":
            if t in "{[(":
                depth += 1
            elif t in "}])":
                depth -= 1
                if depth == 0:
                    return j
    return None


def value_end(toks, i: int) -> int:
    """Indice do ultimo token do valor que comeca em toks[i] (ate virgula/fecho de profundidade 0)."""
    depth = 0
    j = i
    while j < len(toks):
        k, _s, _e, t = toks[j]
        if k == "p":
            if t in "{[(":
                depth += 1
            elif t in "}])":
                if depth == 0:
                    return j - 1
                depth -= 1
            elif t == "," and depth == 0:
                return j - 1
        j += 1
    return len(toks) - 1


def owner_field(toks, key_idx: int) -> str | None:
    """Campo dono do dict de operadores que contem a chave `$op` em toks[key_idx]."""
    depth = 0
    j = key_idx - 1
    while j >= 0:
        k, _s, _e, t = toks[j]
        if k == "p" and t == "}":
            depth += 1
        elif k == "p" and t == "{":
            if depth == 0:
                break
            depth -= 1
        j -= 1
    if j >= 2 and toks[j - 1][3] == ":":
        return key_name(toks[j - 2])
    return None


def quote_of(text: str) -> str:
    return '"' if text.count('"') >= text.count("'") else "'"


def requote(tok_text: str, new_inner: str) -> str:
    return (tok_text[0] + new_inner + tok_text[0]) if tok_text[:1] in "\"'" else new_inner


def rename_collection(name: str, known: set, rng: random.Random) -> str | None:
    opts = []
    if len(name) >= 4:
        i = rng.randrange(1, len(name) - 1)
        opts.append(name[:i] + name[i + 1:])
        opts.append(name[:i] + name[i + 1] + name[i] + name[i + 2:] if i + 2 <= len(name) else name)
    opts.append(name + "s" if not name.endswith("s") else name[:-1])
    opts.append(name[0].swapcase() + name[1:])
    opts.append(name + "_old")
    rng.shuffle(opts)
    low = {k.lower() for k in known}
    for o in opts:
        if o and o != name and o.lower() not in low and re.fullmatch(r"[A-Za-z_][\w]*", o):
            return o
    return None


def pipeline_elems(text: str, toks):
    """Spans (ini, fim) dos estagios do pipeline (primeiro array de objetos cujo primeiro membro e um estagio)."""
    stages = cv.vs.ALLOWED_STAGES
    for i, (k, _s, _e, t) in enumerate(toks):
        if k == "p" and t == "[" and i + 3 < len(toks) and toks[i + 1][3] == "{":
            kn = key_name(toks[i + 2])
            if kn in stages and toks[i + 3][3] == ":":
                close = match_close(toks, i)
                if close is None:
                    return None
                spans, j = [], i + 1
                while j < close:
                    ve = value_end(toks, j)
                    spans.append((toks[j][1], toks[ve][2]))
                    j = ve + 2 if ve + 1 < len(toks) and toks[ve + 1][3] == "," else ve + 1
                return spans
    return None


def find_obj_after_key(toks, names: set):
    """(open_idx, close_idx) do objeto `{...}` cujo rotulo e um dos `names` (chave `$project: {` ou chamada `project({`)."""
    for i, tk in enumerate(toks):
        kn = key_name(tk)
        if kn in names and i + 2 < len(toks):
            if toks[i + 1][3] == ":" and toks[i + 2][3] == "{":
                return i + 2, match_close(toks, i + 2)
            if toks[i + 1][3] == "(" and toks[i + 2][3] == "{" and tk[0] == "id":
                return i + 2, match_close(toks, i + 2)
    return None


def obj_entries(toks, o: int, c: int):
    """[(key_idx, val_start_idx, val_end_idx)] das entradas de profundidade 1 do objeto toks[o]..toks[c]."""
    out, j = [], o + 1
    while j < c:
        if key_name(toks[j]) is not None and j + 1 < c + 1 and toks[j + 1][3] == ":":
            ve = value_end(toks, j + 2)
            out.append((j, j + 2, min(ve, c - 1)))
            j = ve + 2
        else:
            j += 1
    return out


def d_group_no_id(text, toks, rng):
    cands = []
    for i, tk in enumerate(toks):
        if key_name(tk) == "$group" and i + 2 < len(toks) and toks[i + 1][3] == ":" and toks[i + 2][3] == "{":
            c = match_close(toks, i + 2)
            if c is None:
                continue
            for (kj, vs_, ve_) in obj_entries(toks, i + 2, c):
                if key_name(toks[kj]) == "_id":
                    cands.append((kj, ve_))
    if not cands:
        return None
    kj, ve_ = rng.choice(cands)
    s = toks[kj][1]
    if ve_ + 1 < len(toks) and toks[ve_ + 1][3] == ",":
        nxt = toks[ve_ + 2][1] if ve_ + 2 < len(toks) else toks[ve_ + 1][2]
        return text[:s] + text[nxt:], {"removed": "_id"}
    if toks[kj - 1][3] == ",":
        return text[:toks[kj - 1][1]] + text[toks[ve_][2]:], {"removed": "_id"}
    return text[:s] + text[toks[ve_][2]:], {"removed": "_id"}


def d_misplaced_operator(text, toks, rng):
    q = [(i, "query_to_acc") for i, tk in enumerate(toks) if key_name(tk) in QUERY_CMP and i + 1 < len(toks) and toks[i + 1][3] == ":"]
    a = [(i, "acc_to_query") for i, tk in enumerate(toks) if key_name(tk) in AGG_ACC and i + 1 < len(toks) and toks[i + 1][3] == ":"]
    pool = q + a
    if not pool:
        return None
    # equilibra as duas direcoes quando ambas existem
    if q and a and rng.random() < 0.5:
        pool = a
    elif q and a:
        pool = q
    i, sub = rng.choice(pool)
    old = key_name(toks[i])
    new = rng.choice(AGG_ACC if sub == "query_to_acc" else ACC_AS_QUERY)
    if new == old:
        return None
    s, e = toks[i][1], toks[i][2]
    return text[:s] + requote(toks[i][3], new) + text[e:], {"from": old, "to": new, "kind": sub}


def d_type_mismatch(text, toks, coll, rng):
    opts = []
    q = quote_of(text)
    for i in range(2, len(toks)):
        if toks[i - 1][3] != ":":
            continue
        kn = key_name(toks[i - 2])
        if kn is None:
            continue
        field = owner_field(toks, i - 2) if kn.startswith("$") else kn
        f = coll.fields.get(field) if field else None
        if f is None:
            continue
        k, s, e, t = toks[i]
        if k == "str" and f.leaf_t == "string" and not t[1:-1].startswith("$"):
            opts.append((s, e, f"ObjectId({q}{''.join(rng.choice('0123456789abcdef') for _ in range(24))}{q})", "string_as_objectid"))
        elif k == "id" and t in ("true", "false") and f.leaf_t == "bool":
            opts.append((s, e, f"{q}{t}{q}", "bool_as_string"))
        elif k == "id" and t in ("ISODate", "Date", "new") and f.leaf_t == "date":
            j = i + 1 if t != "new" else i + 2
            if j + 2 < len(toks) + 0 and toks[j][3] == "(" and toks[j + 1][0] == "str" and toks[j + 2][3] == ")":
                import datetime as _dt
                try:
                    d = _dt.datetime.fromisoformat(toks[j + 1][3][1:-1].replace("Z", "+00:00"))
                    ms = int(d.replace(tzinfo=d.tzinfo or _dt.timezone.utc).timestamp() * 1000)
                except ValueError:
                    continue
                opts.append((s, toks[j + 2][2], str(ms), "date_as_number"))
        elif k == "num" and f.leaf_t == "objectid":
            opts.append((s, e, f"{q}{t}{q}", "number_as_string_on_objectid"))
    if not opts:
        return None
    a, b, new, sub = rng.choice(opts)
    return text[:a] + new + text[b:], {"kind": sub}


def d_lookup_field(text, toks, coll, known, rng):
    opts = []
    for i, tk in enumerate(toks):
        kn = key_name(tk)
        if kn in ("localField", "foreignField", "from") and i + 2 < len(toks) and toks[i + 1][3] == ":" and toks[i + 2][0] == "str":
            opts.append((i + 2, kn))
    if not opts:
        return None
    j, kn = rng.choice(opts)
    old = toks[j][3][1:-1]
    if kn == "from":
        new = rename_collection(old, known, rng)
    else:
        class _C:  # wrong_name so precisa de `.fields`
            fields = coll.fields
        new = wrong_name(old, _C, rng)
        if new is None:
            new = old + "_id" if not old.endswith("_id") else old[:-3]
    if not new or new == old:
        return None
    return text[:toks[j][1]] + toks[j][3][0] + new + toks[j][3][0] + text[toks[j][2]:], {"field": kn, "from": new, "to": old}


def d_stage_order(text, toks, rng):
    spans = pipeline_elems(text, toks)
    if not spans or len(spans) < 2:
        return None
    idx = list(range(len(spans) - 1))
    rng.shuffle(idx)
    for i in idx:
        a, b = spans[i], spans[i + 1]
        ta, tb = text[a[0]:a[1]], text[b[0]:b[1]]
        ka = re.search(r"\$[A-Za-z]+", ta)
        kb = re.search(r"\$[A-Za-z]+", tb)
        if ta == tb or (ka and kb and ka.group() == kb.group()):
            continue
        new = text[:a[0]] + tb + text[a[1]:b[0]] + ta + text[b[1]:]
        return new, {"swapped": [ka.group() if ka else "?", kb.group() if kb else "?"], "index": i}
    return None


def d_unknown_method(text, rng):
    opts = [(old, new) for old, news in METHOD_SWAP.items() if old in text for new in news]
    if not opts:
        return None
    old, new = rng.choice(opts)
    return text.replace(old, new, 1), {"from": new.strip(".("), "to": old.strip(".(")}


def d_hint_index(text, toks, coll, rng):
    for i, tk in enumerate(toks):
        if tk[0] == "id" and tk[3] == "find" and i > 0 and toks[i - 1][3] == "." and i + 1 < len(toks) and toks[i + 1][3] == "(":
            c = match_close(toks, i + 1)
            if c is None:
                return None
            names = [p for p in coll.fields if p != "_id" and re.fullmatch(r"[A-Za-z_]\w*", p)]
            if not names:
                return None
            f = rng.choice(names)
            q = quote_of(text)
            ins = rng.choice([f".hint({{ {f}: 1 }})", f".hint({q}{f}_1{q})", f".hint({{ {f}: -1 }})"])
            return text[:toks[c][2]] + ins + text[toks[c][2]:], {"hint": f}
    return None


def d_projection_mix(text, toks, rng):
    cands = []
    for names in ({"$project"}, {"project"}):
        r = find_obj_after_key(toks, names)
        if r and r[1]:
            cands.append(r)
    # segundo argumento de find/findOne
    for i, tk in enumerate(toks):
        if tk[0] == "id" and tk[3] in ("find", "findOne") and i + 1 < len(toks) and toks[i + 1][3] == "(":
            c = match_close(toks, i + 1)
            if c is None:
                continue
            ve = value_end(toks, i + 2)
            if ve + 2 < c and toks[ve + 1][3] == "," and toks[ve + 2][3] == "{":
                cc = match_close(toks, ve + 2)
                if cc:
                    cands.append((ve + 2, cc))
            break
    flips = []
    for o, c in cands:
        ents = obj_entries(toks, o, c)
        inc = [(kj, vs_) for kj, vs_, ve_ in ents if key_name(toks[kj]) != "_id" and vs_ == ve_ and toks[vs_][3] in ("1", "true")]
        if len(inc) >= 2:
            flips += inc
    if not flips:
        return None
    kj, vs_ = rng.choice(flips)
    new = "0" if toks[vs_][3] == "1" else "false"
    return text[:toks[vs_][1]] + new + text[toks[vs_][2]:], {"field": key_name(toks[kj])}


def d_sort_direction(text, toks, rng):
    cands = []
    for names in ({"$sort"}, {"sort"}):
        r = find_obj_after_key(toks, names)
        if r and r[1]:
            for kj, vs_, ve_ in obj_entries(toks, *r):
                cands.append((kj, vs_, ve_))
    if not cands:
        return None
    kj, vs_, ve_ = rng.choice(cands)
    q = quote_of(text)
    if toks[vs_][3] == "-" and ve_ == vs_ + 1 and toks[ve_][3] == "1":
        a, b, d = toks[vs_][1], toks[ve_][2], -1
    elif vs_ == ve_ and toks[vs_][3] == "1":
        a, b, d = toks[vs_][1], toks[vs_][2], 1
    else:
        return None
    new = rng.choice([f"{q}desc{q}" if d == -1 else f"{q}asc{q}", "2", f"{q}descending{q}" if d == -1 else f"{q}ascending{q}"])
    return text[:a] + new + text[b:], {"field": key_name(toks[kj]), "value": new.strip("\"'")}


def d_limit_value(text, toks, rng):
    cands = []
    for i, tk in enumerate(toks):
        kn = key_name(tk)
        if kn in ("$limit", "$skip") and i + 2 < len(toks) and toks[i + 1][3] == ":" and toks[i + 2][0] == "num":
            cands.append((i + 2, kn))
        elif tk[0] == "id" and tk[3] in ("limit", "skip") and i + 2 < len(toks) and toks[i + 1][3] == "(" and toks[i + 2][0] == "num" \
                and toks[i + 3][3] == ")" and toks[i - 1][3] == ".":
            cands.append((i + 2, tk[3]))
    if not cands:
        return None
    j, kn = rng.choice(cands)
    n = toks[j][3]
    q = quote_of(text)
    if kn in ("$limit", "limit"):
        new = rng.choice([f"{q}{n}{q}", "0"] if kn == "$limit" else [f"{q}{n}{q}", f"{n}.5"])
    else:
        new = rng.choice([f"{q}{n}{q}", f"-{n}"])
    return text[:toks[j][1]] + new + text[toks[j][2]:], {"where": kn, "value": new.strip("\"'")}


def d_wrong_collection(text, coll, known, rng):
    name = coll.name
    m = re.search(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])", text)
    if not m:
        return None
    new = rename_collection(name, known, rng)
    if not new:
        return None
    return text[:m.start()] + new + text[m.end():], {"from": new, "to": name}


def plan_features(plan: dict) -> dict:
    """Propriedades baratas do plano para decidir quais familias de defeito sao aplicaveis."""
    js = json.dumps(plan.get("filter") or plan.get("pipeline") or {}, default=str)
    ft = {"group": False, "lookup": False, "stages": 0, "sort": False, "limit": False, "proj_inc": 0, "find": plan["op"] == "find",
          "filter": bool(plan.get("filter")),
          "typed": any(t in js for t in ('"$oid"', '"$date"', '"$numberLong"')) or bool(re.search(r": -?\d", js))}
    if plan["op"] == "aggregate":
        st = plan["pipeline"]
        ft["stages"] = len(st)
        for s in st:
            (n, b), = s.items()
            ft["group"] |= n == "$group"
            ft["lookup"] |= n == "$lookup"
            ft["sort"] |= n == "$sort"
            ft["limit"] |= n in ("$limit", "$skip")
            ft["filter"] |= n == "$match"
            if n == "$project" and isinstance(b, dict):
                ft["proj_inc"] = max(ft["proj_inc"], sum(1 for k, v in b.items() if k != "_id" and v in (1, True)))
    else:
        ft["sort"] = bool(plan.get("sort"))
        ft["limit"] = plan.get("limit") is not None or plan.get("skip") is not None
        p = plan.get("projection") or {}
        ft["proj_inc"] = sum(1 for k, v in p.items() if k != "_id" and v in (1, True))
    return ft


def applicable(kind: str, plan: dict) -> bool:
    ft = plan_features(plan)
    return {
        "group_no_id": ft["group"], "lookup_field": ft["lookup"], "stage_order": ft["stages"] >= 2, "hint_index": ft["find"],
        "projection_mix": ft["proj_inc"] >= 2, "sort_direction": ft["sort"], "limit_value": ft["limit"],
        "type_mismatch": ft["filter"], "misplaced_operator": ft["filter"] or ft["group"],
        "wrong_operator": ft["filter"] or ft["group"], "missing_dollar": ft["filter"] or ft["group"],
        "wrong_type": ft["typed"],
    }.get(kind, True)


def error_message(defect_type: str, stage: str, err: str | None, text: str) -> str | None:
    """Mensagem tipo Console para a instrucao (so quando ha erro real do motor/JS)."""
    if not err:
        return None
    if stage == "syntax":
        return err
    if stage in ("api", "fields"):
        if err.startswith("a group specification"):
            return "MongoServerError: a group specification must include an _id"
        m = re.match(r"operador de agregacao fora de lugar \(em filtro\): (\$\w+)", err)
        if m:
            return f"MongoServerError: unknown operator: {m.group(1)}"
        m = re.match(r"operador fora de lugar em \$group.*: (\$\w+)", err)
        if m:
            return f"MongoServerError: unknown group operator '{m.group(1)}'"
        m = re.match(r"projecao mista: exclusao de '(.+)' em", err)
        if m:
            return f"MongoServerError: Cannot do exclusion on field {m.group(1)} in inclusion projection"
        if err.startswith("ordenacao invalida"):
            return "MongoServerError: $sort key ordering must be 1 or -1"
        if err.startswith("$limit deve"):
            return "MongoServerError: the limit must be positive"
        if err.startswith("$skip deve"):
            return "MongoServerError: Expected a non-negative number in: $skip"
        if err.startswith("limit exige") or err.startswith("skip exige"):
            return "MongoServerError: " + err.split(" ")[0] + " requires an integer"
        if err.startswith("hint:"):
            return "MongoServerError: hint provided does not correspond to an existing index"
        if err.startswith("$lookup exige"):
            return "MongoServerError: $lookup requires 'from', 'localField', 'foreignField' and 'as'"
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
        m = re.match(r"operador de consulta inexistente: (\$\w+)", err)
        if m:
            return f"MongoServerError: unknown operator: {m.group(1)}"
        return None
    return None


# --------------------------------------------------------------------------- textos fixos (negativos)

REFUSAL_WRITE = {
    "pt": ["Não gero comandos que alteram ou apagam dados.", "Não executo escrita ou exclusão por aqui; este chat só monta consultas de leitura.",
           "Pedidos que modificam ou removem dados ficam fora do chat: ele só gera consultas de leitura.",
           "Não posso alterar nem excluir documentos; posso montar uma consulta de leitura."],
    "en": ["I do not generate commands that modify or delete data.", "Writes and deletions are out of scope here; this chat only builds read queries.",
           "Requests that change or remove data are not handled in chat: it only generates read queries.",
           "I cannot change or delete documents; I can build a read query instead."],
}
REFUSAL_DDL = {
    "pt": ["Não executo operações de esquema ou índice (drop, rename, createIndex); este chat só gera consultas de leitura.",
           "Alterar a estrutura do banco (coleções e índices) está fora do escopo: aqui só monto consultas de leitura.",
           "Não removo, renomeio nem crio coleções ou índices por aqui."],
    "en": ["I do not run schema or index operations (drop, rename, createIndex); this chat only generates read queries.",
           "Changing the database structure (collections and indexes) is out of scope: I only build read queries here.",
           "I cannot drop, rename or create collections or indexes from here."],
}
PREVIEW = {"pt": ["Para conferir os documentos que seriam afetados:", "Uma consulta de leitura equivalente, para conferir antes:",
                  "Se quiser só olhar os documentos, use esta leitura:"],
           "en": ["To review the documents that would be affected:", "An equivalent read query to check first:",
                  "If you only want to look at the documents, use this read query:"]}
PREVIEW_COUNT = {"pt": ["Para ver quantos documentos seriam afetados:", "Uma contagem de leitura para dimensionar antes:"],
                 "en": ["To see how many documents would be affected:", "A read-only count to size it up first:"]}
OUT_OF_SCOPE_REPLY = {
    "pt": ["Fora do escopo: aqui eu só ajudo com consultas MongoDB.", "Isso não é uma consulta MongoDB; só consigo ajudar com o Console do MongoDB.",
           "Não consigo ajudar com isso aqui; posso montar consultas para o Console do MongoDB."],
    "en": ["Out of scope: I can only help with MongoDB queries here.", "That is not a MongoDB query; I can only help with the MongoDB Console.",
           "I cannot help with that here; I can build queries for the MongoDB Console."],
}
OUT_OF_SCOPE = {
    "pt": ["escreve um poema sobre o mar", "qual a capital da Austrália?", "como eu faço deploy no kubernetes", "me ajuda a montar um currículo",
           "converte esse texto para maiúsculas: bom dia", "quanto é 15% de 2300?", "escreva uma função em Python que ordena uma lista",
           "me conta uma piada", "traduz 'bom dia' para o francês", "qual a previsão do tempo para amanhã?", "explica o que é uma blockchain",
           "monta uma query SQL com JOIN entre clientes e pedidos", "resolve essa equação: 2x + 3 = 11", "sugere um nome para o meu cachorro",
           "qual a melhor linguagem de programação?", "escreve um e-mail pedindo férias", "como configuro o nginx como proxy reverso?",
           "faz um resumo do filme Matrix", "me ensina a fazer pão caseiro", "qual a diferença entre TCP e UDP?",
           "gera um regex para validar CPF", "escreva uma query em Cypher para o Neo4j", "monta um SELECT * FROM pedidos WHERE total > 100",
           "como reinstalo o windows?"],
    "en": ["write a poem about the sea", "what is the capital of Australia?", "how do I deploy to kubernetes", "help me write a resume",
           "convert this text to uppercase: good morning", "what is 15% of 2300?", "write a python function that sorts a list",
           "tell me a joke", "translate 'good morning' to French", "what is the weather forecast for tomorrow?", "explain what a blockchain is",
           "build a SQL query with a JOIN between customers and orders", "solve this equation: 2x + 3 = 11", "suggest a name for my dog",
           "what is the best programming language?", "write an email asking for time off", "how do I configure nginx as a reverse proxy?",
           "summarize the movie Matrix", "teach me how to bake bread", "what is the difference between TCP and UDP?",
           "generate a regex to validate a phone number", "write a Cypher query for Neo4j", "write SELECT * FROM orders WHERE total > 100",
           "how do I reinstall windows?"],
}
GENERIC_WRITE = {
    "pt": ["Apague a coleção {c}", "dropa a coleção {c}", "Exclua o banco de dados inteiro", "limpa tudo {cin}", "Insira um documento novo {cin}",
           "Renomeie a coleção {c} para {c}_old", "Remova todos os documentos {cof}", "Crie uma coleção chamada {c}_backup e copie tudo para ela",
           "deleta tudo {cin}", "Zere os dados {cof}", "Atualize todos os documentos {cin} marcando como arquivado", "Faça um bulkWrite {cin}"],
    "en": ["Drop the {c} collection", "drop collection {c}", "Delete the whole database", "wipe everything {cin}", "Insert a new document {cin}",
           "Rename the {c} collection to {c}_old", "Remove all the documents {cof}", "Create a collection called {c}_backup and copy everything into it",
           "nuke everything {cin}", "Truncate the data {cof}", "Update all the documents {cin} to mark them archived", "Run a bulkWrite {cin}"],
}
DDL_ASK = {
    "pt": ["Crie um índice em {f} na coleção {c}", "cria um índice único por {f} em {c}", "Apague o índice de {f} da coleção {c}",
           "Recrie a coleção {c} do zero", "Crie a coleção {c}_arquivo com validação de esquema", "Faça dropDatabase", "dropa o banco todo",
           "Renomeie {c} para {c}_v2", "Compacte a coleção {c}", "Ative o sharding da coleção {c}"],
    "en": ["Create an index on {f} in the {c} collection", "create a unique index on {f} for {c}", "Drop the index on {f} from {c}",
           "Recreate the {c} collection from scratch", "Create the {c}_archive collection with schema validation", "Run dropDatabase",
           "drop the whole database", "Rename {c} to {c}_v2", "Compact the {c} collection", "Enable sharding on the {c} collection"],
}
FAKE_COLLECTIONS = ["foguetes", "planetas", "receitas", "rockets", "planets", "recipes", "historico_antigo", "old_archive", "faturas_2019",
                    "legacy_users", "satelites", "stadiums", "temp_import", "relatorios_v0", "mensagens_antigas", "audit_trail_2018",
                    "cache_sessions", "tmp_export", "feriados", "lunar_phases"]
CORRECT_PHRASES = {
    "pt": ["Corrija essa consulta", "corrige essa query aí", "Essa consulta está com erro, arruma pra mim", "Conserte a consulta do editor",
           "tem algo errado nisso, corrige sem mudar o que ela faz", "Revise e corrija a consulta", "Não está funcionando, ajusta a consulta",
           "Arrume o comando mantendo a mesma intenção", "Acho que errei algo aqui, pode consertar?", "Debuga essa consulta pra mim"],
    "en": ["Fix this query", "fix this query for me", "This query has an error, please correct it", "Repair the query in the editor",
           "something is wrong here, fix it without changing what it does", "Review and fix the query", "It is not working, adjust the query",
           "Fix the command keeping the same intent", "I think I made a mistake here, can you fix it?", "Debug this query for me"],
}
CORRECT_Q = {
    "pt": ["Você consegue corrigir essa consulta?", "Dá pra arrumar o erro dessa query?", "Tem como consertar isso aqui?"],
    "en": ["Can you fix this query?", "Could you repair the error in this command?", "Is there a way to correct this?"],
}
CORRECT_G = {
    "pt": ["corrige query", "fix: query com erro", "consertar comando do editor", "erro na consulta; corrigir"],
    "en": ["fix query", "fix: broken query", "repair editor command", "query error; fix"],
}
CORRECT_WITH_ERR = {
    "pt": ["Deu este erro: {e}. Corrija a consulta.", "{e}\ncorrige aí", "Erro no Console: {e}. Pode arrumar?",
           "Recebi: {e}\nO que está errado? Corrija.", "{e} - conserta isso"],
    "en": ["I got this error: {e}. Fix the query.", "{e}\nplease fix", "Console error: {e}. Can you repair it?",
           "Got: {e}\nWhat is wrong? Fix it.", "{e} - fix this"],
}
CORRECT_SILENT = {
    "pt": ["A consulta retorna 0 documentos mas deveria retornar resultados. Corrija.", "Não volta nada e eu esperava resultados, corrige a consulta",
           "Roda sem erro mas vem vazio; ajuste a consulta", "O resultado não faz sentido, acho que tem um campo errado. Corrija."],
    "en": ["The query returns 0 documents but should return results. Fix it.", "It returns nothing and I expected results, fix the query",
           "It runs but comes back empty; adjust the query", "The result makes no sense, a field is wrong. Fix it."],
}
EXPLAIN_ASK = {
    "pt": ["Explique essa consulta", "o que essa query faz?", "Explica rapidinho o que isso faz", "Descreva o que o código do editor faz",
           "Resuma em uma frase essa consulta", "Pra que serve esse comando?"],
    "en": ["Explain this query", "what does this query do?", "Briefly explain what this does", "Describe what the editor code does",
           "Summarize this query in one sentence", "What is this command for?"],
}
EXPLAIN_Q = {
    "pt": ["Você consegue me explicar o que essa consulta faz?", "Dá pra dizer o que isso aqui retorna?"],
    "en": ["Can you explain what this query does?", "Could you tell me what this returns?"],
}
EXPLAIN_G = {"pt": ["explica query", "o que faz; resumo curto"], "en": ["explain query", "what it does; short"]}
AMBIG_FIELD_ASK = {
    "recent": {
        "pt": ["Me mostra os mais recentes {cof}", "traz os registros mais novos {cof}", "Liste os últimos documentos {cof}",
               "quais os mais recentes {cin}?", "Mostre os {n} documentos mais recentes {cof}"],
        "en": ["Show me the most recent ones {cof}", "get the newest records {cof}", "List the latest documents {cof}",
               "which are the most recent {cin}?", "Show the {n} most recent documents {cof}"],
    },
    "top": {
        "pt": ["Traga os maiores {cof}", "me mostra os {n} maiores {cof}", "Liste os top {n} {cof}", "quais os maiores valores {cin}?"],
        "en": ["Get the biggest ones {cof}", "show me the top {n} {cof}", "List the top {n} {cof}", "which have the highest values {cin}?"],
    },
}
AMBIG_FIELD_REPLY = {
    "recent": {"pt": ["Qual campo de data devo usar para ordenar por recência: {fl}?"],
               "en": ["Which date field should I use to sort by recency: {fl}?"]},
    "top": {"pt": ["Maiores segundo qual campo numérico: {fl}?"], "en": ["Largest according to which numeric field: {fl}?"]},
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

def norm_code(s: str) -> str:
    """Normaliza codigo para a chave de unicidade: sem espacos/CRLF, aspas unificadas, sem ';' final."""
    return re.sub(r"\s+", "", s).replace('"', "'").rstrip(";")


def plan_key(plan: dict) -> str:
    return json.dumps(plan, sort_keys=False, ensure_ascii=False, separators=(",", ":"))


def schema_role(sch) -> str | None:
    """`split_role` do schema (train|eval); leitura defensiva: ausente/invalido => None."""
    r = getattr(sch, "split_role", None)
    return r if r in ("train", "eval") else None


def role_conflict(role: str | None, snip: dict) -> bool:
    """Snippet cujo split contradiz o papel do schema (schema eval nunca em train; schema train nunca em test/bench)."""
    if role == "eval":
        return snip.get("split") == "train"
    if role == "train":
        return snip.get("split") in ("test", "bench") and not snip.get("holdout")
    return False


def has_atlas_stage(plan: dict) -> bool:
    return plan.get("op") == "aggregate" and any(isinstance(st, dict) and (set(st) & cv.ATLAS_STAGES)
                                                 for st in plan.get("pipeline") or [])


def cap_eval_pool(rows: list[dict], caps: dict | None, seed, tag: str) -> list[dict]:
    """Limita, por split de avaliacao, o numero de snippets do pool (ordem por hash, deterministica). `caps` None = sem teto;
    valor < 1 = fracao do pool de treino (assim o peso da avaliacao no chat escala com o volume)."""
    if not caps:
        return rows
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r.get("split"), []).append(r)
    out: list[dict] = list(by.get("train", []))
    for sp, lst in by.items():
        if sp == "train":
            continue
        cap = caps.get(sp)
        if cap is not None:
            cap = int(cap * len(by.get("train", []))) if float(cap) < 1 else int(cap)    # < 1: fracao do pool de treino; >= 1: absoluto
        if cap is not None and len(lst) > cap:
            lst = sorted(lst, key=lambda r: hashlib.sha256(f"{seed}:capchat{tag}:{r['id']}".encode()).hexdigest())[:cap]
        out.extend(lst)
    return out


class ChatGen:
    def __init__(self, cfg: dict, snippets: list[dict], validation: dict, schemas, executor, node):
        self.cfg = cfg
        self.cc = cfg["chat"]
        self.cfg_style = style_cfg(cfg)
        self.sch = {s.full_id: s for s in schemas}
        self.ex, self.node = executor, node
        self.seed = cfg.get("seed", 20261008)
        self.stale_snippets = self.role_conflicts = 0

        def usable(r):
            s = self.sch.get(r["schema_id"])
            if s is None or not any(c.name == r["collection"] for c in s.colls):
                self.stale_snippets += 1      # snippet de uma versao anterior dos schemas
                return False
            if role_conflict(schema_role(s), r):
                self.role_conflicts += 1      # split do snippet contradiz o split_role do schema: nao entra
                return False
            return True

        reads = [r for r in snippets if r["kind"] == "read" and validation.get(r["id"], {}).get("exec") == "ok"
                 and not validation[r["id"]].get("empty") and usable(r)]
        # Atlas Search ($search/$searchMeta/$vectorSearch) so tem validacao estatica: nunca vira chat/correcao (sem execucao/descricao)
        reads = [r for r in reads if not has_atlas_stage(r["plan"])]
        n_all = len(reads)
        reads = [r for r in reads if not order_dependent(r["plan"])]
        self.order_dependent_dropped = n_all - len(reads)
        writes = [r for r in snippets if r["kind"] == "write" and r.get("op") not in ("createIndex", "dropIndex") and usable(r)]
        # tetos para val/test/bench (spec §4.6: avaliacao com tamanho fixo); sem isso ~1/3 dos registros iria para avaliacao
        reads = cap_eval_pool(reads, self.cc.get("eval_read_caps"), self.seed, "r")
        writes = cap_eval_pool(writes, self.cc.get("eval_write_caps"), self.seed, "w")
        order = random.Random(f"{self.seed}:chat:order")
        order.shuffle(reads)
        order.shuffle(writes)
        self.reads, self.writes = reads, writes
        self.distinct_plans = len({(r["schema_id"], r["collection"], plan_key(r["plan"])) for r in reads})
        self.ri = self.wi = 0
        self.stats = Counter()
        self.discards = Counter()
        self.seen: set = set()
        self.exec_attempts = self.exec_ok = 0
        self.defect_attempts = self.defect_failed = 0
        self.attempts_total = 0
        # controles de forca (usados pelo capacity-report)
        self._force_snip = None
        self._force_defect = None
        self._force_modify = None
        self._force_refine = None
        self._force_neg = None

    # ---- amostragem de snippet ----
    def next_read(self):
        if self._force_snip is not None:
            return self._force_snip
        r = self.reads[self.ri % len(self.reads)]
        self.ri += 1
        return r

    def next_read_where(self, pred, tries: int = 60):
        """Primeiro snippet (na ordem de ciclo) que satisfaz `pred(plan)`; None se nenhum em `tries` leituras."""
        if self._force_snip is not None:
            r = self._force_snip
            return r if pred(r["plan"]) else None
        for _ in range(tries):
            r = self.next_read()
            if pred(r["plan"]):
                return r
        return None

    def next_write(self):
        if self._force_snip is not None and self._force_snip["kind"] == "write":
            return self._force_snip
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
            "split": snip["split"],            # herdado do snippet (que ja respeita o split_role do schema)
            "holdout": snip["holdout"],
            "schema_split_role": schema_role(sch),
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

    def render_seeded(self, plan, sch, coll, seed):
        """Renderiza com estilo fixado por `seed` (pares base/final compartilham layout, aspas, CRLF, raiz)."""
        return render_text(plan, sch, coll, self.cfg_style, random.Random(seed))

    def ok_record_validation(self, snip, extra=None):
        v = {"syntax": "ok", "api": "ok", "fields": "ok", "types": "ok", "exec": "ok", "nonempty": True,
             "plan_equal": True, "fixture": snip["db"]}
        if extra:
            v.update(extra)
        return v

    def check_code(self, text, sch, coll, snip, plan, need_nonempty=True):
        """Valida `text` contra `plan`; retorna (ok, stage)."""
        v = self.validate(text, sch, coll, snip)
        if v["stage"] != "ok":
            return False, "output_" + v["stage"]
        if not cv.plans_equal(v["plan"], plan):
            return False, "output_plan_mismatch"
        if need_nonempty and v["n_docs"] == 0:
            return False, "output_empty"
        return True, "ok"

    # ---- decomposicao de um plano em (base, refinamento) ----
    @staticmethod
    def tail_options(plan):
        """[(base_plan, removed)] retirando caudas (sort/limit/skip/projection ou $sort/$limit/$skip finais)."""
        out = []
        if plan["op"] == "find":
            present = [k for k in ("sort", "skip", "limit", "projection") if plan.get(k) is not None]
            if present:
                out.append(("find", present))
        elif plan["op"] == "aggregate":
            st = plan["pipeline"]
            k = len(st)
            while k > 1 and list(st[k - 1])[0] in ("$sort", "$limit", "$skip"):
                k -= 1
            if k < len(st):
                out.append(("aggregate", k))
        return out

    @staticmethod
    def filter_option(plan):
        """(base_plan, cond) removendo uma condicao de topo do filtro (ou do $match inicial)."""
        if plan["op"] in ("find", "findOne", "countDocuments", "distinct"):
            flt = plan.get("filter") or {}
            keys = [k for k in flt if not k.startswith("$")]
            if len(flt) >= 2 and keys:
                return flt, "filter", keys
        elif plan["op"] == "aggregate" and plan["pipeline"] and list(plan["pipeline"][0])[0] == "$match":
            flt = plan["pipeline"][0]["$match"]
            keys = [k for k in flt if not k.startswith("$")]
            if isinstance(flt, dict) and len(flt) >= 2 and keys:
                return flt, "match", keys
        return None

    # ---- chat ----
    def gen_chat(self, i, rng, lang, reg):
        snip = self.next_read()
        sch, coll = self.sch_coll(snip)
        plan = snip["plan"]
        amb = rng.random() < self.cc.get("ambiguous_collection", 0.10)
        seed = rng.getrandbits(32)
        text, info = self.render_seeded(plan, sch, coll, seed)
        want_modify = self._force_modify if self._force_modify is not None else rng.random() < self.cc.get("modify_fraction", 0.5)
        modify = False
        removed_stages: list = []
        if want_modify:
            for kind_, arg in self.tail_options(plan):
                modify = True
                if kind_ == "aggregate":
                    removed_stages = plan["pipeline"][arg:]
        if self._force_modify and not modify:
            return None
        rege = "n" if reg == "q" else reg
        if modify and snip["op"] == "find":
            present = [k for k in ("sort", "skip", "limit", "projection") if plan.get(k) is not None]
            removed = rng.sample(present, rng.randint(1, len(present)))
            base_plan = {k: v for k, v in plan.items() if k not in removed}
            try:
                base_text, _ = self.render_seeded(base_plan, sch, coll, seed)
            except Exception:
                self.discards["modify_base_render"] += 1
                return None
            d = cd.Describer(lang, rng, rege, False)
            raw = self.modify_text(d, plan, removed, lang)
            editor = base_text
            tid = "chat.modify." + snip["template_id"]
            intent = {"kind": "modify", "operations": ["find"] + present, "added": removed}
            lit_ok = True
            reg_used = rege
        elif modify:
            base_plan = {"op": "aggregate", "pipeline": plan["pipeline"][:len(plan["pipeline"]) - len(removed_stages)]}
            try:
                base_text, _ = self.render_seeded(base_plan, sch, coll, seed)
            except Exception:
                self.discards["modify_base_render"] += 1
                return None
            d = cd.Describer(lang, rng, rege, False)
            steps = [d.stage_text(x) for x in removed_stages]
            if rege == "g":
                raw = "+ " + "; ".join(steps)
            else:
                lead = d.ch(["Agora ", "Na agregação do editor, ", "Altere o pipeline: ", "Acrescente ao final: ", "Estenda o pipeline: "],
                            ["Now ", "In the editor's aggregation, ", "Update the pipeline: ", "Add at the end: ", "Extend the pipeline: "])
                raw = lead + d.join(steps)
            editor = base_text
            tid = "chat.modify." + snip["template_id"]
            intent = {"kind": "modify", "operations": self.operations(plan), "added": [list(x)[0] for x in removed_stages]}
            lit_ok = True
            reg_used = rege
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
            reg_used = reg
        if not lit_ok:
            self.discards["literal_missing"] += 1
            return None
        pinfo: dict = {}
        instr = self.post(rng, raw, lang, reg_used, pinfo)
        ok, why = self.check_code(text, sch, coll, snip, plan)
        if not ok:
            self.discards[why] += 1
            return None
        if modify:
            vb = self.validate(editor, sch, coll, snip)
            if vb["stage"] != "ok" or not cv.plans_equal(vb["plan"], base_plan):
                self.discards["modify_base_invalid"] += 1
                return None
        ctx = self.make_context(rng, snip, sch, coll, editor, True, info)
        rec = self.base_record(i, "chat", lang, snip, sch, coll, tid, instr, ctx, text)
        rec["intent"] = intent
        rec["register"] = reg_used
        rec["pt_accents_stripped"] = bool(pinfo.get("deaccent"))
        rec["typo_injected"] = bool(pinfo.get("typo"))
        rec["validation"] = self.ok_record_validation(snip)
        return rec

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
        tele = d.reg == "g"
        for k in removed:
            if k == "sort":
                st = d.sort_text(plan["sort"])
                bits.append((f"ordem {st}" if d.pt else f"sort {st}") if tele else d.ch(
                    [f"ordene por {st}", f"ordenando por {st}", f"classifique por {st}"],
                    [f"sort by {st}", f"order by {st}", f"rank by {st}"]))
            elif k == "limit":
                n = plan["limit"]
                bits.append((f"limite {n}" if d.pt else f"limit {n}") if tele else d.ch(
                    [f"limite a {n}", f"só os {n} primeiros", f"no máximo {n}", f"traga apenas {n}"],
                    [f"limit to {n}", f"only the first {n}", f"at most {n}", f"just {n}"]))
            elif k == "skip":
                n = plan["skip"]
                bits.append((f"pular {n}" if d.pt else f"skip {n}") if tele else d.ch(
                    [f"pule {n}", f"ignore os {n} primeiros"], [f"skip {n}", f"skip the first {n}"]))
            elif k == "projection":
                bits.append(d.proj_short(plan["projection"]) if tele else d.proj_text(plan["projection"]))
        if tele:
            return "+ " + "; ".join(bits)
        body = d.join(bits) if len(bits) < 3 else ", ".join(bits)
        lead = d.ch(["Agora ", "Na consulta do editor, ", "Altere a consulta: ", "Mantenha o filtro e ", "Ajuste: ", ""],
                    ["Now ", "In the editor query, ", "Update the query: ", "Keep the filter and ", "Tweak it: ", ""])
        return lead + body

    # ---- multi-turno (2 turnos: pedido + refinamento) ----
    def gen_multiturn(self, i, rng, lang, reg):
        def has_ref(p):
            return bool(self.tail_options(p)) or self.filter_option(p) is not None
        snip = self.next_read_where(has_ref)
        if snip is None:
            self.discards["multiturn_no_refinement"] += 1
            return None
        sch, coll = self.sch_coll(snip)
        plan = snip["plan"]
        opts = []
        if self._force_refine in (None, "tail"):
            opts += [("tail",) + o for o in self.tail_options(plan)]
        fo = self.filter_option(plan) if self._force_refine in (None, "filter") else None
        if fo:
            opts.append(("filter",) + fo)
        if not opts:
            self.discards["multiturn_no_refinement"] += 1
            return None
        pick_o = rng.choice(opts)
        seed = rng.getrandbits(32)
        rege = "n" if reg == "q" else reg
        try:
            text, info = self.render_seeded(plan, sch, coll, seed)
        except Exception:
            self.discards["multiturn_render"] += 1
            return None
        refine_type = pick_o[0]
        d2 = cd.Describer(lang, rng, rege, False)
        if refine_type == "tail":
            if pick_o[1] == "find":
                removed = rng.sample(pick_o[2], rng.randint(1, len(pick_o[2])))
                base_plan = {k: v for k, v in plan.items() if k not in removed}
                raw2 = self.modify_text(d2, plan, removed, lang)
                added = removed
            else:
                k = pick_o[2]
                rem = plan["pipeline"][k:]
                base_plan = {"op": "aggregate", "pipeline": plan["pipeline"][:k]}
                steps = [d2.stage_text(x) for x in rem]
                raw2 = ("+ " + "; ".join(steps)) if rege == "g" else (
                    d2.ch(["Agora ", "Depois disso, ", "Em seguida ", ""], ["Now ", "After that, ", "Then ", ""]) + d2.join(steps))
                added = [list(x)[0] for x in rem]
            lit2_ok = True
        else:
            flt, where, keys = pick_o[1], pick_o[0], pick_o[3]
            k = rng.choice(keys)
            cond = {k: flt[k]}
            nflt = {kk: vv for kk, vv in flt.items() if kk != k}
            if plan["op"] == "aggregate":
                base_plan = {"op": "aggregate", "pipeline": [{"$match": nflt}] + plan["pipeline"][1:]}
            else:
                base_plan = {**plan, "filter": nflt}
            added = [k]
            ymatch = None
            v = flt[k]
            if isinstance(v, dict) and set(v) == {"$gte", "$lt"} and all(isinstance(x, dict) and "$date" in x for x in v.values()):
                lo, hi = v["$gte"]["$date"], v["$lt"]["$date"]
                if lo[5:10] == "01-01" and hi[5:10] == "01-01" and lo[11:] in ("", "00:00:00Z") and int(hi[:4]) == int(lo[:4]) + 1:
                    ymatch = int(lo[:4])
            try:
                if ymatch is not None:
                    n_dates = sum(1 for f in coll.fields.values() if f.leaf_t == "date")
                    raw2 = d2.refine_year(k, ymatch, with_field=not (n_dates == 1 and rng.random() < 0.5))
                    lit2_ok = True
                else:
                    raw2 = d2.refine_filter(cond) if rege != "g" else "+ " + d2.filter_text(cond)
                    lit2_ok = literals_present({"filter": cond}, raw2)
            except cd.Unsupported:
                self.discards["multiturn_unsupported"] += 1
                return None
        if not lit2_ok:
            self.discards["literal_missing"] += 1
            return None
        try:
            base_text, _ = self.render_seeded(base_plan, sch, coll, seed)
        except Exception:
            self.discards["multiturn_render"] += 1
            return None
        amb = rng.random() < self.cc.get("ambiguous_collection", 0.10)
        d1 = cd.Describer(lang, rng, reg, amb)
        try:
            raw1 = d1.describe(base_plan, coll.name)
        except cd.Unsupported:
            self.discards["describer_unsupported"] += 1
            return None
        if not literals_present(base_plan, raw1):
            self.discards["literal_missing"] += 1
            return None
        pinfo1: dict = {}
        pinfo2: dict = {}
        instr1 = self.post(rng, raw1, lang, reg, pinfo1)
        instr2 = self.post(rng, raw2, lang, rege, pinfo2)
        ok, why = self.check_code(text, sch, coll, snip, plan)
        if not ok:
            self.discards[why] += 1
            return None
        okb, whyb = self.check_code(base_text, sch, coll, snip, base_plan, need_nonempty=False)
        if not okb:
            self.discards["history_" + whyb] += 1
            return None
        if norm_code(base_text) == norm_code(text):
            self.discards["multiturn_noop"] += 1
            return None
        ctx = self.make_context(rng, snip, sch, coll, base_text, True, info)
        rec = self.base_record(i, "chat_multiturn", lang, snip, sch, coll,
                               f"chat.multiturn.{refine_type}." + snip["template_id"], instr2, ctx, text)
        rec["history"] = [{"role": "user", "content": instr1, "editor": ""}, {"role": "assistant", "content": base_text}]
        rec["intent"] = {"kind": "refine", "refine": refine_type, "operations": self.operations(plan), "added": added, "turns": 2}
        if amb:
            rec["intent"]["ambiguous_collection"] = True
        rec["register"] = rege
        rec["pt_accents_stripped"] = bool(pinfo2.get("deaccent"))
        rec["typo_injected"] = bool(pinfo1.get("typo") or pinfo2.get("typo"))
        rec["validation"] = self.ok_record_validation(snip, {"history_code_valid": True})
        return rec

    # ---- correction ----
    def pick_defect_kinds(self, rng, want: str, n: int, plan: dict) -> list[str]:
        """`n` familias distintas (a primeira e `want`); as de sintaxe vao por ultimo (editam texto ja alterado)."""
        kinds = [want]
        pool = {k: v for k, v in self.cc["defects"].items() if k != want and applicable(k, plan)}
        while len(kinds) < n and pool:
            k = pick(rng, pool)
            kinds.append(k)
            pool.pop(k)
        syn = [k for k in kinds[1:] if k.startswith("syntax_")]
        return [kinds[0]] + [k for k in kinds[1:] if k not in syn] + syn

    def gen_correction(self, i, rng, lang, reg):
        want = self._force_defect or pick(rng, self.cc["defects"])
        ndef = pick(rng, {int(k): v for k, v in self.cc.get("defect_count", {1: 1.0}).items()})
        self.stats["corr_try:" + want] += 1
        for _a in range(1 if self._force_snip is not None else 4):     # outro snippet se a familia nao se aplica ao texto
            snip = self.next_read_where(lambda p: applicable(want, p))
            if snip is None:
                break
            rec = self.correction_once(i, rng, lang, reg, want, ndef, snip)
            if rec is not None:
                self.stats["corr_ok:" + want] += 1
                return rec
        else:
            return None
        self.discards["correction_no_applicable"] += 1
        return None

    def correction_once(self, i, rng, lang, reg, want, ndef, snip):
        sch, coll = self.sch_coll(snip)
        plan = snip["plan"]
        known = [c.name for c in sch.colls]
        text, info = self.render_seeded(plan, sch, coll, rng.getrandbits(32))
        v = self.validate(text, sch, coll, snip)
        if v["stage"] != "ok" or not cv.plans_equal(v["plan"], plan) or v["n_docs"] == 0:
            self.discards["correction_reference_invalid"] += 1
            return None
        used = v.get("fields_used", snip["fields_used"])
        for _t in range(6):
            kinds = self.pick_defect_kinds(rng, want, ndef, plan)
            cur, applied = text, []
            for j, kd in enumerate(kinds):
                inj = None
                for _u in range(3 if j else 1):
                    inj = inject_defect(cur, plan, used, coll, rng, self.cc["defects"], kind=kd, known_names=known)
                    if inj is not None and inj[0] != cur:
                        break
                    inj = None
                if inj is None:
                    if j == 0:
                        break
                    continue
                cur = inj[0]
                applied.append((inj[1], inj[2]))
            if not applied or cur == text or applied[0][0] != want:
                continue
            bad = cur
            self.defect_attempts += 1
            syn = self.node.check(bad)
            vb = cv.validate_text(bad, sch, coll, self.ex, syn, db=snip["db"])
            if vb["stage"] == "ok":
                self.discards["defect_not_failing"] += 1      # passa em tudo: nao e defeito detectavel -> descarta
                continue
            self.defect_failed += 1
            stage, err = vb["stage"], vb["error"]
            msg = error_message(want, stage, err, bad)
            rr = rng.random()
            if rr < self.cc.get("correction_error_message", 0.5) and msg:
                raw = rng.choice(CORRECT_WITH_ERR[lang]).format(e=msg)
                if len(re.findall(r"\w+", re.sub(r"[\"'`][^\"'`]*[\"'`]", " x ", raw))) > 11:
                    # <= 12 tokens: a mensagem de erro e boilerplate compartilhado; nao deve virar 13-grama de decontaminacao
                    raw = msg + rng.choice(["", " - fix", " fix it"] if lang == "en" else ["", " - corrige", " arruma"])
            elif rr < self.cc.get("correction_error_message", 0.5) + 0.25 and stage in ("fields", "type") and not msg:
                raw = rng.choice(CORRECT_SILENT[lang])
            elif reg == "q":
                raw = rng.choice(CORRECT_Q[lang])
            elif reg == "g":
                raw = rng.choice(CORRECT_G[lang])
            else:
                raw = rng.choice(CORRECT_PHRASES[lang])
            if chr(10) in raw or (msg and msg in raw):
                instr = raw   # a mensagem de erro e literal: nao passa por acento/caixa
            else:
                instr = self.post_plain(rng, raw, lang, "n" if reg in ("q", "g") else reg)
            ctx = self.make_context(rng, snip, sch, coll, bad, True, info)
            dtype, detail = applied[0]
            rec = self.base_record(i, "correction", lang, snip, sch, coll, f"correction.{dtype}." + snip["template_id"], instr, ctx, text)
            rec["intent"] = {"kind": "correction", "operations": self.operations(plan)}
            rec["defect"] = {"type": dtype, "detail": detail, "failed_stage": stage, "error": err,
                             "error_in_instruction": bool(msg and msg in instr), "n_defects": len(applied),
                             "types": [a[0] for a in applied]}
            if len(applied) > 1:
                rec["defect"]["combined"] = [{"type": a[0], "detail": a[1]} for a in applied[1:]]
            rec["register"] = reg
            rec["validation"] = self.ok_record_validation(snip, {"defective": {"stage": stage, "error": err}})
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
        text, info = self.render_seeded(plan, sch, coll, rng.getrandbits(32))
        d = cd.Describer(lang, rng, "n", False)
        try:
            expl = cd.strip_marks(d.explain(plan, coll.name))
        except cd.Unsupported:
            self.discards["explain_unsupported"] += 1
            return None
        if len(expl) > 320 or "\n" in expl:
            self.discards["explain_long"] += 1
            return None
        ok, why = self.check_code(text, sch, coll, snip, plan)
        if not ok:
            self.discards[why] += 1
            return None
        out = "// " + expl + "\n" + text
        pool = EXPLAIN_Q if reg == "q" else (EXPLAIN_G if reg == "g" else EXPLAIN_ASK)
        instr = self.post_plain(rng, rng.choice(pool[lang]), lang, "n" if reg in ("q", "g") else reg)
        ctx = self.make_context(rng, snip, sch, coll, text, True, info)
        rec = self.base_record(i, "explain", lang, snip, sch, coll, "explain." + snip["template_id"], instr, ctx, out)
        rec["intent"] = {"kind": "explain", "operations": self.operations(plan)}
        rec["register"] = reg
        rec["validation"] = self.ok_record_validation(snip)
        return rec

    # ---- negativos ----
    def preview_block(self, plan_flt, sch, coll, snip, rng, lang, allow_count=True):
        """Consulta de leitura equivalente (find limit 20 ou countDocuments), validada. Retorna (texto, rotulo) ou (None, None)."""
        if allow_count and (not plan_flt or rng.random() < 0.4):
            pplan = {"op": "countDocuments", "filter": plan_flt or {}}
            label = rng.choice(PREVIEW_COUNT[lang])
        else:
            pplan = {"op": "find", "filter": plan_flt, "limit": 20}
            label = rng.choice(PREVIEW[lang])
        ptxt, _ = self.render_seeded(pplan, sch, coll, rng.getrandbits(32))
        pv = self.validate(ptxt, sch, coll, snip)
        if pv["stage"] == "ok" and cv.plans_equal(pv["plan"], pplan):
            return ptxt, label
        self.discards["negative_preview_invalid"] += 1
        return None, None

    def gen_negative(self, i, rng, lang, reg):
        ntype = self._force_neg or pick(rng, self.cc["negative_types"])
        if reg == "q" and ntype in ("write", "ddl", "out_of_scope"):
            reg = "n"     # texto fixo: pergunta indireta artificial
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
            plabel = None
            flt = snip["plan"].get("filter") or {}
            if rng.random() < self.cc.get("preview_in_write_refusal", 0.35):
                if generic:
                    preview, plabel = self.preview_block({}, sch, coll, snip, rng, lang)
                elif flt:
                    preview, plabel = self.preview_block(flt, sch, coll, snip, rng, lang)
            out = refusal(out_lines) + (("\n" + refusal([plabel]) + "\n" + preview) if preview else "")
            editor = "" if rng.random() < 0.6 else default_command(coll.name, sch, coll, self.cfg_style, rng)
            ctx = self.make_context(rng, snip, sch, coll, editor, True)
            tid = "negative.write." + ("generic" if generic else snip["template_id"])
            ndet = {"type": "write", "generic": generic, "preview": preview is not None, "requested_op": snip["op"] if not generic else None}
        elif ntype == "ddl":
            snip = self.next_read()
            sch, coll = self.sch_coll(snip)
            names = [p for p in coll.fields if p != "_id" and re.fullmatch(r"[A-Za-z_]\w*", p)] or ["_id"]
            raw = rng.choice(DDL_ASK[lang]).format(c=cd.F(coll.name), f=cd.F(rng.choice(names)))
            out = refusal([rng.choice(REFUSAL_DDL[lang])])
            preview = None
            if rng.random() < 0.3:
                preview, plabel = self.preview_block({}, sch, coll, snip, rng, lang)
                if preview:
                    out += "\n" + refusal([plabel]) + "\n" + preview
            ctx = self.make_context(rng, snip, sch, coll, "" if rng.random() < 0.7 else default_command(coll.name, sch, coll, self.cfg_style, rng), True)
            tid = "negative.ddl"
            ndet = {"type": "ddl", "preview": preview is not None}
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
        elif ntype == "ambiguous_field":
            snip = None
            for _try in range(1 if self._force_snip is not None else 40):
                c = self.next_read()
                s_, c_ = self.sch_coll(c)
                dates = sorted(p for p, f in c_.fields.items() if f.leaf_t == "date" and "." not in p)
                nums = sorted(p for p, f in c_.fields.items() if f.leaf_t in ("int32", "int64", "double", "decimal128") and "." not in p
                              and p != "_id")
                opts = [("recent", dates)] if len(dates) >= 2 else []
                if len(nums) >= 2:
                    opts.append(("top", nums))
                if opts:
                    snip = c
                    break
            if snip is None:
                self.discards["negative_no_ambiguous_field"] += 1
                return None
            sch, coll = self.sch_coll(snip)
            sub, flds = rng.choice(opts)
            d = cd.Describer(lang, rng, reg, False)
            n = rng.choice([5, 10, 20])
            raw = rng.choice(AMBIG_FIELD_ASK[sub][lang]).format(cof=d.coll_of(coll.name), cin=d.coll_in(coll.name), n=n)
            shown = flds[:4]
            out = refusal([rng.choice(AMBIG_FIELD_REPLY[sub][lang]).format(fl=" ou ".join(shown) if lang == "pt" else " or ".join(shown))])
            ctx = self.make_context(rng, snip, sch, coll, "", True)
            tid = "negative.ambiguous_field." + sub
            ndet = {"type": "ambiguous_field", "subtype": sub, "candidates": shown}
        else:  # ambiguous (colecao)
            snip = self.next_read()
            sch, coll = self.sch_coll(snip)
            d = cd.Describer(lang, rng, reg, True)
            try:
                raw = d.describe(snip["plan"], coll.name)
            except cd.Unsupported:
                self.discards["negative_unsupported"] += 1
                return None
            out = refusal([rng.choice(["Qual coleção devo consultar? Selecione uma coleção ou informe o nome.",
                                       "Em qual coleção? Informe o nome ou selecione uma no painel."])] if lang == "pt"
                          else [rng.choice(["Which collection should I query? Select a collection or give me its name.",
                                            "Which collection do you mean? Give me its name or select one."])])
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

    FN = {"chat": "gen_chat", "chat_multiturn": "gen_multiturn", "correction": "gen_correction", "explain": "gen_explain",
          "negative": "gen_negative"}

    @staticmethod
    def record_key(rec: dict) -> tuple:
        """Chave normalizada de unicidade (instrucao sem acento/caixa/espacos; codigo sem layout/aspas)."""
        core = rec["context"]["editor"] if rec["kind"] == "correction" else rec["output"]
        key = (rec["schema_id"], rec["collection"], rec["kind"], norm_text(rec["instruction"]), norm_code(core))
        if rec["kind"] == "chat_multiturn":
            key += (norm_text(rec["history"][0]["content"]),)
        return key

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
            fn = getattr(self, self.FN[kind])
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
            key = self.record_key(rec)
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
        self.attempts_total = attempts
        self.shortfall = {k: quota[k] - got[k] for k in quota if quota[k] > got[k]}
        return out

    # ---- capacidade (exemplos unicos por familia e por schema) ----
    def families(self) -> list[tuple[str, dict]]:
        fam: list[tuple[str, dict]] = [("chat.create", {"kind": "chat", "force_modify": False}),
                                       ("chat.modify", {"kind": "chat", "force_modify": True}),
                                       ("multiturn.tail", {"kind": "chat_multiturn", "force_refine": "tail"}),
                                       ("multiturn.filter", {"kind": "chat_multiturn", "force_refine": "filter"})]
        fam += [(f"correction.{d}", {"kind": "correction", "force_defect": d}) for d in ALL_DEFECTS]
        fam.append(("explain", {"kind": "explain"}))
        fam += [(f"negative.{t}", {"kind": "negative", "force_neg": t}) for t in
                ("write", "ddl", "out_of_scope", "no_collection", "ambiguous", "ambiguous_field")]
        return fam

    def capacity(self, per_schema: int = 6, rounds: int = 3, progress=None) -> dict:
        """Mede, por familia e por schema, quantos registros unicos (chave normalizada) o gerador consegue.
        Amostra `per_schema` planos distintos por schema, sorteia `rounds` instrucoes/estilos por plano e extrapola
        pelo numero de planos distintos do schema (limite inferior: mais sorteios acham mais variantes)."""
        def group(rows):
            g: dict[str, dict] = {}
            for r in rows:
                g.setdefault(r["schema_id"], {}).setdefault((r["collection"], plan_key(r["plan"])), r)
            return g
        by_schema, by_schema_w = group(self.reads), group(self.writes)
        fams = self.families()
        res: dict = {f: {} for f, _ in fams}
        kw = self.cc["register"]
        saved = (self.exec_attempts, self.exec_ok, Counter(self.discards))
        for si, sid in enumerate(sorted(by_schema)):
            for fname, spec in fams:
                plans = (by_schema_w if fname == "negative.write" else by_schema).get(sid, {})
                items = sorted(plans.items(), key=lambda kv: hash_str(f"{self.seed}:cap:{kv[0]}"))
                sample = [v for _k, v in items[:per_schema]]
                n_plans = len(plans)
                keys: set = set()
                ok_snips = 0
                for snip in sample:
                    got_any = False
                    for rd in range(rounds):
                        rng = random.Random(f"{self.seed}:cap:{rd}:{snip['id']}:{fname}")
                        lang = "pt" if rng.random() < 0.5 else "en"
                        reg = pick(rng, kw)
                        self._force_snip = snip
                        self._force_defect = spec.get("force_defect")
                        self._force_modify = spec.get("force_modify")
                        self._force_refine = spec.get("force_refine")
                        self._force_neg = spec.get("force_neg")
                        try:
                            rec = getattr(self, self.FN[spec["kind"]])(0, rng, lang, reg)
                        except cd.Unsupported:
                            rec = None
                        finally:
                            self._force_snip = self._force_defect = self._force_modify = self._force_refine = self._force_neg = None
                        if rec is None:
                            continue
                        got_any = True
                        keys.add(self.record_key(rec))
                    ok_snips += got_any
                n = len(sample)
                elig = ok_snips / n if n else 0.0
                uniq_per_ok = (len(keys) / ok_snips) if ok_snips else 0.0
                res[fname][sid] = {"plans": n_plans, "sample": n, "eligible_rate": round(elig, 3),
                                   "unique_per_eligible_plan": round(uniq_per_ok, 2),
                                   "est_unique": int(round(n_plans * elig * uniq_per_ok)), "sample_unique": len(keys)}
            if progress:
                progress(si + 1, len(by_schema))
        self.exec_attempts, self.exec_ok = saved[0], saved[1]
        self.discards = saved[2]
        return {"rounds": rounds, "per_schema_sample": per_schema, "distinct_plans": self.distinct_plans, "schemas": len(by_schema),
                "families": res}


def hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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
    code_recs = [r for r in recs if r["kind"] in ("chat", "chat_multiturn", "correction", "explain")]
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
    mt = [r for r in recs if r["kind"] == "chat_multiturn"]
    if mt:
        L.append(f"Multi-turno (`chat_multiturn`, 2 turnos, `history` estruturado): {len(mt)}; refinamentos "
                 f"{dict(Counter(r['intent']['refine'] for r in mt))}.\n")
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
    nd = Counter(r["defect"].get("n_defects", 1) for r in corr)
    L.append("\nDefeitos combinados por correcao: " + ", ".join(f"{k}={v} ({v / max(1, len(corr)):.1%})" for k, v in sorted(nd.items())) + ".")
    L.append("\nEtapa em que o input defeituoso falhou: " + ", ".join(f"{k}={v}" for k, v in dstage.most_common()) +
             f". Tentativas de defeito: {gen.defect_attempts}; todo par mantido tem input defeituoso que falha e corrigido que passa "
             f"(`semantic` = valido mas com plano diferente do pretendido: {dstage.get('semantic', 0)}).\n")
    L.append("## Negativos\n")
    L.append(table([(k, v) for k, v in neg.most_common()], ("tipo", "n")))
    L.append("\nRecusas de escrita/DDL sao comentarios `//`; ~35% trazem uma leitura equivalente (`find` limit 20 ou `countDocuments`, executada e validada).\n")
    L.append("## Descartes (geracao)\n")
    L.append(table(gen.discards.most_common(), ("motivo", "n")))
    L.append(f"\nShortfall de quota: {getattr(gen, 'shortfall', {})}.\n")
    L.append("## Mistura alvo (config `chat.mixture`, fracao do treino total)\n")
    L.append(table([(k, v) for k, v in mix.items()], ("componente", "fracao")))
    L.append(f"\nchat+explain+multi-turno+correcao = {mix['chat'] + mix['correction']:.0%} (limite 25%); negativos = {mix['negative']:.0%}.\n")
    L.append("## Limites conhecidos\n")
    for s in [
        "Instrucoes vem de templates parametrizados pelo plano (nenhum LLM): paraphrases variam, mas a cobertura e a de `gen_snippets` "
        "(estagios e operadores do plano); construcoes nao descritas (`$lookup` com `let/pipeline`, `$expr` com variaveis, etc.) sao descartadas. "
        "Risco v1: saturar benchmark programatico e falhar em pedido livre -> o blind escrito a mao continua sendo o gate.",
        "Valores citados aparecem literalmente; nomes de campo aparecem como no schema (sem sinonimos de dominio).",
        "Validacao sem ContractProbe/Acornima: sintaxe por `node vm.Script`, API por parser do subconjunto Console proprio; $search/$vectorSearch "
        "e estagios fora da allowlist nao entram.",
        "Correcao: defeitos sao edicoes textuais do output (19 familias, ate 4 combinados); `semantic` e raro por construcao. Mensagens de erro de sintaxe vem do node (V8), nao do Acornima da IDE; `.hint()` e tratado como fora do subconjunto do Console.",
        "Negativos: `known_collections` no contexto e necessario para `no_collection`; o contrato `local-agent-fim-v0` atual nao carrega essa lista (decisao pendente).",
        "Referencia ambigua so de colecao; multi-turno tem 2 turnos fixos; `selection` e sempre null nesta fatia.",
        "Dominios holdout so alimentam test/bench; train de chat depende dos mesmos grupos do FIM (sem vazamento entre eles).",
    ]:
        L.append("- " + s)
    return "\n".join(L) + "\n"


def summarize_capacity(cap: dict, kind_weights: dict, target: int) -> dict:
    fam_tot = {f: sum(v["est_unique"] for v in per.values()) for f, per in cap["families"].items()}
    kind_of = lambda f: ("chat_multiturn" if f.startswith("multiturn") else f.split(".")[0])  # noqa: E731
    kind_tot: dict[str, int] = {}
    for f, t in fam_tot.items():
        kind_tot[kind_of(f)] = kind_tot.get(kind_of(f), 0) + t
    tw = sum(kind_weights.values())
    need = {k: int(round(target * kind_weights.get(k, 0) / tw)) for k in kind_weights}
    ratio = {k: (kind_tot.get(k, 0) / need[k] if need[k] else None) for k in need}
    worst = min((k for k in ratio if ratio[k] is not None), key=lambda k: ratio[k], default=None)
    schemas = sorted({sid for per in cap["families"].values() for sid in per})
    by_schema = {sid: {f: per[sid]["est_unique"] for f, per in cap["families"].items() if sid in per} for sid in schemas}
    sch_total = {sid: sum(v.values()) for sid, v in by_schema.items()}
    weak = sorted(sch_total, key=sch_total.get)[:5]
    low_fam = sorted(fam_tot, key=fam_tot.get)[:6]
    return {"target": target, "families_total": fam_tot, "kinds_total": kind_tot, "needed": need,
            "coverage": {k: (round(v, 2) if v is not None else None) for k, v in ratio.items()},
            "bottleneck_kind": worst, "weakest_families": low_fam, "weakest_schemas": {w: sch_total[w] for w in weak},
            "total_est_unique": sum(fam_tot.values()), "distinct_plans": cap["distinct_plans"]}


def format_capacity(cap: dict, summ: dict) -> str:
    L = [f"capacidade medida: {cap['schemas']} schemas, {cap['distinct_plans']} planos distintos, amostra {cap['per_schema_sample']}/schema, "
         f"{cap['rounds']} sorteios/plano (limite inferior)"]
    L.append(table(sorted(((f, t) for f, t in summ["families_total"].items()), key=lambda x: x[1]), ("familia", "unicos estimados")))
    L.append("")
    L.append(table([(k, summ["kinds_total"].get(k, 0), summ["needed"][k], summ["coverage"][k]) for k in summ["needed"]],
                   ("kind", "capacidade", f"necessario p/ {summ['target']}", "cobertura")))
    L.append(f"\nTotal estimado: {summ['total_est_unique']} | gargalo: {summ['bottleneck_kind']} | familias mais fracas: "
             f"{summ['weakest_families']} | schemas mais fracos: {summ['weakest_schemas']}")
    return "\n".join(L)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


CHAT_SNIP_KEYS = ("id", "kind", "op", "plan", "schema_id", "collection", "domain", "db", "template_id", "group_id", "split_key",
                  "split", "holdout", "split_role", "fields_used", "category", "dialect", "exec_mode")


def load_chat_pool(snip_path: Path, val_path: Path, max_train: int = 0, max_eval: int = 0, salt: str = "",
                   survivors: set | None = None, include_hom: bool = False) -> tuple[list[dict], dict]:
    """Streaming: so os snippets uteis ao chat (leituras executadas e nao vazias + escritas), com os campos usados.
    `max_train`/`max_eval` > 0: teto por (tipo leitura/escrita x split) por amostra uniforme deterministica (hash) -- com ~3 M de
    snippets so uma fracao e consumida pelo chat (<= ~40 mil registros). Devolve (snippets enxutos, validacao {id: {exec, empty}})."""
    from jsonl_shards import HashReservoir, iter_jsonl
    val: dict = {}
    shared: dict = {}
    for r in iter_jsonl(val_path):
        key = (r.get("exec"), r.get("empty"))
        val[r["id"]] = shared.setdefault(key, {"exec": key[0], "empty": key[1]})
    res: dict = {}
    keep: list[dict] = []
    for r in hom_guard.drop_hom(iter_jsonl(snip_path), include_hom):
        if survivors is not None and r["split"] == "train" and r["id"] not in survivors:
            continue            # pre-filtro de decontaminacao (mesmo indice do gen_fim): o chat so parte de documentos que sobrevivem
        if r["kind"] == "read":
            v = val.get(r["id"]) or {}
            if v.get("exec") != "ok" or v.get("empty"):
                continue
        slim = {k: r[k] for k in CHAT_SNIP_KEYS if k in r}
        cap = max_train if r["split"] == "train" else max_eval
        if cap > 0:
            k = (r["kind"], r["split"])
            if k not in res:
                res[k] = HashReservoir(cap, f"{salt}:chatpool:{k[0]}:{k[1]}")
            res[k].add(r["id"], slim)
        else:
            keep.append(slim)
    for k in sorted(res):
        keep.extend(res[k].items())
    return keep, val


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--count", type=int)
    ap.add_argument("--ver")
    ap.add_argument("--snippets")
    ap.add_argument("--validation")
    ap.add_argument("--include-hom", action="store_true",
                    help="SO para scripts de homologacao (outro diretorio, split `homolog`); nunca em jobs de treino")
    ap.add_argument("--survivors", help="fim.survivors.txt do gen_fim (ids de snippets de treino que sobrevivem a decontaminacao)")
    ap.add_argument("--out")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--capacity-report", action="store_true",
                    help="mede exemplos unicos por familia/schema (nao grava chat.jsonl); saida em chat.capacity.json")
    ap.add_argument("--capacity-sample", type=int, default=6, help="planos distintos amostrados por schema")
    ap.add_argument("--capacity-rounds", type=int, default=3, help="sorteios de instrucao/estilo por plano")
    ap.add_argument("--capacity-target", type=int, default=15000, help="total de registros distintos desejado")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg["seed"] = a.seed
    ver = a.ver or cfg.get("version", "v0")
    gdir = ROOT / cfg.get("out_dir", "data/generated") / ver
    surv = None
    sp = Path(a.survivors) if a.survivors else gdir / "fim.survivors.txt"
    if cfg["chat"].get("use_fim_survivors", True) and sp.exists():
        surv = {l.strip() for l in open(sp, encoding="utf-8") if l.strip()}
    snips, val = load_chat_pool(Path(a.snippets) if a.snippets else gdir / "snippets.jsonl",
                                Path(a.validation) if a.validation else gdir / "validation.jsonl",
                                int(cfg["chat"].get("pool_max_train", 0) or 0), int(cfg["chat"].get("pool_max_eval", 0) or 0),
                                str(cfg.get("seed")), surv, a.include_hom)
    schemas = load_schemas(ROOT / cfg.get("schemas_dir", "data/schemas"), include_hom=a.include_hom)
    count = a.count or cfg["chat"]["count"]
    t0 = time.perf_counter()
    ex = cv.Executor(cfg)
    node = cv.NodeChecker()
    try:
        gen = ChatGen(cfg, snips, val, schemas, ex, node)
        if a.capacity_report:
            cap = gen.capacity(a.capacity_sample, a.capacity_rounds,
                               progress=lambda i, n: print(f"capacidade: schema {i}/{n}", file=sys.stderr))
            print(file=sys.stderr)
            summ = summarize_capacity(cap, cfg["chat"]["kinds"], a.capacity_target)
            cap["summary"] = summ
            cp = (Path(a.out).parent if a.out else gdir) / "chat.capacity.json"
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(cap, indent=1, ensure_ascii=False), encoding="utf-8")
            print(format_capacity(cap, summ))
            print(f"[capacity] {cp}", file=sys.stderr)
            return
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
           "node_syntax_check": node_ok(), "mixture": cfg["chat"]["mixture"],
           "pool": {"reads": len(gen.reads), "writes": len(gen.writes), "distinct_plans": gen.distinct_plans,
                    "stale_snippets_skipped": gen.stale_snippets, "split_role_conflicts_skipped": gen.role_conflicts},
           "attempts": gen.attempts_total, "unique_keys": len(gen.seen),
           "defect_success": {k[len("corr_try:"):]: [gen.stats["corr_ok:" + k[len("corr_try:"):]], v]
                              for k, v in sorted(gen.stats.items()) if k.startswith("corr_try:")}}
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
