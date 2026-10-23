#!/usr/bin/env python3
"""Prompts da geracao por LLM: personas, gabaritos de tarefa (30+), variacoes de estilo e o bloco de schema.

Tudo que e SORTEADO (persona, tarefa, estilo) sai de um RNG semeado por (seed, schema, colecao, indice da chamada): o LLM so escreve a
consulta; a distribuicao de tarefas/estilos e deterministica e controlada aqui."""
from __future__ import annotations

import json
import random
import re

PERSONAS = [
    ("bi_analyst", "a BI analyst building a dashboard query for stakeholders"),
    ("backend_dev", "a backend developer writing a query used by an API endpoint"),
    ("dba", "a DBA investigating data quality and index usage"),
    ("support", "a customer-support engineer looking up one customer's records to answer a ticket"),
    ("monthly_report", "a finance/operations person preparing the monthly report"),
    ("prod_debug", "an engineer debugging a production incident, checking suspicious records"),
    ("data_engineer", "a data engineer cleaning and auditing a collection before a migration"),
    ("qa_tester", "a QA engineer checking test data and edge cases"),
    ("product_manager", "a product manager asking for quick numbers about usage"),
    ("junior_dev", "a junior developer who writes plain, slightly verbose, well-commented code"),
    ("security_auditor", "an auditor checking for anomalies and policy violations"),
    ("sre", "an SRE writing a quick one-off check during an on-call shift"),
    ("terse_power_user", "an expert who types terse one-liners in the shell without comments"),
    ("tech_writer", "a documentation author writing a clear, didactic example"),
]

# id, peso, descricao (em ingles; o modelo adapta aos campos reais)
TASKS = [
    ("cursor_pagination", 1.0, "keyset (cursor) pagination: filter with _id (or a sort key) greater than a last-seen value, sort, limit a page"),
    ("top_n", 1.4, "top-N ranking: $group by a field, aggregate a metric, $sort descending, $limit N"),
    ("monthly_report", 1.2, "a monthly report: group by month/year using $dateToString, $year/$month or $dateTrunc with totals and counts"),
    ("facet_dashboard", 0.9, "a $facet pipeline computing several summaries at once (counts per category, price/amount stats, top items)"),
    ("bucket_distribution", 0.8, "a distribution of a numeric field using $bucket or $bucketAuto with accumulator outputs"),
    ("window_running", 0.8, "$setWindowFields: running total, moving average, rank or dense rank partitioned by a field"),
    ("dedupe", 0.9, "find duplicated values: $group by a key, count > 1 with $match, optionally list the _ids with $push"),
    ("date_window", 1.3, "records inside a date window (last N days via new Date(Date.now() - ...), or between two ISODate values)"),
    ("text_regex", 0.9, "case-insensitive search on string fields with $regex / regex literal, combined with other filters"),
    ("lookup_join", 1.4, "a multi-stage join: $lookup into a related collection, $unwind, $match on joined fields, $project the result"),
    ("data_cleaning", 1.0, "data-cleaning queries: documents with missing/null fields ($exists, $type), wrong types, empty arrays or strings"),
    ("audit_invariants", 0.8, "audit invariants with $expr (e.g. one date before another, a total different from the sum of parts)"),
    ("script_vars", 1.3, "a short multi-statement script: const/let variables (dates, ids, thresholds), then 1-3 queries using them, with comments"),
    ("util_function", 0.7, "define a small helper function (arrow or function) that builds a filter or runs a read-only query, then call it"),
    ("sibling_db", 0.7, "use db.getSiblingDB('<database>') and getCollection('<collection>') explicitly instead of db.<collection>"),
    ("bracket_access", 0.6, "access the collection with db.getCollection('<name>') or db['<name>'] (also valid for names with hyphens)"),
    ("bson_literals", 1.0, "queries with ObjectId('...'), ISODate('...'), NumberDecimal('...'), NumberLong(...), UUID('...') literals as appropriate for the field types"),
    ("find_projection_sort", 1.4, "find with a filter, a projection (inclusion or exclusion), a multi-key sort, skip and limit"),
    ("count_distinct", 0.8, "countDocuments with filters and distinct on a field (optionally with a filter)"),
    ("array_ops", 1.0, "array operators: $elemMatch, $all, $size, $in on arrays, or $unwind / $filter / $map on an array field"),
    ("conditional_expr", 0.9, "computed fields with $cond, $switch, $ifNull in $addFields/$project"),
    ("string_ops", 0.6, "string expressions: $concat, $toUpper, $substrCP, $split, $regexMatch, $trim in a pipeline"),
    ("union_with", 0.4, "combine two collections of the same database with $unionWith and aggregate the combined stream"),
    ("graph_hierarchy", 0.3, "hierarchy or self-reference traversal with $graphLookup if the schema has a parent/self reference, otherwise a tree-like grouping"),
    ("nested_paths", 1.2, "queries that use nested object paths with dot notation (filters, projections, $group keys) from the schema"),
    ("stats", 0.7, "statistics: $avg, $stdDevPop, $min/$max, $median or $percentile over a numeric field per group"),
    ("single_entity_debug", 1.0, "look up one entity by a specific id/key and then fetch its related documents from another collection (two statements)"),
    ("conditional_pivot", 0.6, "pivot-like counts with $sum + $cond per category in one $group"),
    ("object_reshape", 0.6, "reshape documents with $replaceRoot, $mergeObjects, $objectToArray or $arrayToObject"),
    ("date_arithmetic", 0.8, "date arithmetic: $dateDiff, $dateAdd, $dateSubtract, $dateTrunc, day-of-week or age buckets"),
    ("sample_qa", 0.3, "random sampling with $sample for QA, optionally followed by a $project"),
    ("range_filters", 1.2, "combined range filters with $gte/$lt/$in/$nin/$ne/$or/$and on several fields of different types"),
    ("sort_group_limit_chain", 1.0, "a classic $match -> $group -> $sort -> $limit pipeline with a computed field in $project at the end"),
    ("compare_fields_expr", 0.6, "compare two fields of the same document with $expr ($gt/$eq/$cmp) in find or $match"),
]

LAYOUTS = [
    ("pretty2", 0.28, "multi-line, 2-space indentation, one pipeline stage per block"),
    ("pretty4", 0.18, "multi-line, 4-space indentation"),
    ("compact", 0.22, "compact: everything on one or two lines"),
    ("lines", 0.18, "one stage / one argument per line, no deep indentation"),
    ("tabs", 0.04, "multi-line indented with tab characters"),
    ("extracted", 0.10, "the pipeline or filter extracted to a const variable first, then used"),
]
QUOTES = [("double", 0.5, 'double quotes for string values; keys may be quoted or unquoted'),
          ("single", 0.5, "single quotes for strings; unquoted keys when possible")]
COMMENTS = [("none", 0.40, "no comments"), ("en", 0.30, "a few short comments in English"), ("pt", 0.30, "a few short comments in Brazilian Portuguese")]
VARNAMES = [("camel", 0.5, "camelCase variable names"), ("snake", 0.25, "snake_case variable names"),
            ("pt", 0.25, "variable names in Portuguese (e.g. inicio, limite, resultado)")]
ENDINGS = [("toArray", 0.40, "end with .toArray()"), ("plain", 0.25, "no explicit .toArray() (a cursor is fine)"),
           ("print", 0.20, "wrap the result with printjson(...) or console.log(...)"), ("semicolon", 0.15, "end statements with semicolons")]
ROOT_FORMS = [("auto", 0.60, "db.<collection>"), ("getCollection", 0.12, "db.getCollection('<collection>')"),
              ("sibling", 0.12, "db.getSiblingDB('<database>').<collection>"), ("bracket", 0.06, "db['<collection>']"),
              ("var", 0.10, "const coll = db.<collection>; then use coll")]


def pick(rng: random.Random, table):
    w = [t[1] for t in table]
    return rng.choices(table, weights=w)[0]


def make_specs(rng: random.Random, n: int, short_fraction: float, task_ids: list[str] | None = None) -> list[dict]:
    """n especificacoes (tarefa, persona, estilo) para uma chamada. Sem repetir tarefa dentro da chamada."""
    tasks = [t for t in TASKS if task_ids is None or t[0] in task_ids]
    chosen, used = [], set()
    for _ in range(n):
        for _try in range(10):
            t = pick(rng, tasks)
            if t[0] not in used:
                break
        used.add(t[0])
        chosen.append(t)
    specs = []
    for t in chosen:
        persona = rng.choice(PERSONAS)
        short = rng.random() < short_fraction
        spec = {"task": t[0], "task_text": t[2], "persona": persona[0], "persona_text": persona[1], "short": short,
                "layout": pick(rng, LAYOUTS)[0], "quote": pick(rng, QUOTES)[0], "comments": pick(rng, COMMENTS)[0],
                "varnames": pick(rng, VARNAMES)[0], "ending": pick(rng, ENDINGS)[0], "root": pick(rng, ROOT_FORMS)[0]}
        specs.append(spec)
    return specs


def style_text(spec: dict) -> str:
    d = {k: {x[0]: x[2] for x in tab} for k, tab in
         {"layout": LAYOUTS, "quote": QUOTES, "comments": COMMENTS, "varnames": VARNAMES, "ending": ENDINGS, "root": ROOT_FORMS}.items()}
    if spec["short"]:
        return ("SHORT and TERSE, as typed quickly by a person: ONE statement, no comments, minimal formatting, no variables; "
                f"{d['quote'][spec['quote']]}; collection access: {d['root'][spec['root']]}")
    parts = [f"layout: {d['layout'][spec['layout']]}", f"{d['quote'][spec['quote']]}", f"comments: {d['comments'][spec['comments']]}",
             f"{d['varnames'][spec['varnames']]} (only if variables are used)", f"{d['ending'][spec['ending']]}",
             f"collection access: {d['root'][spec['root']]}"]
    return "; ".join(parts)


# ------------------------------------------------------------------------------------------------------------------------------
# bloco de schema
# ------------------------------------------------------------------------------------------------------------------------------
def field_line(f) -> str:
    sp = f.spec if isinstance(f.spec, dict) else {}
    t = f.t
    extra = []
    if t == "array":
        t = f"array<{f.item_t or 'any'}>"
        isp = f.item_spec or {}
        if isp.get("values"):
            extra.append("items in " + json.dumps(isp["values"][:8], ensure_ascii=False))
    if sp.get("values"):
        extra.append("one of " + json.dumps(sp["values"][:10], ensure_ascii=False))
    if "lo" in sp and "hi" in sp:
        extra.append(f"range {sp['lo']}..{sp['hi']}")
    if f.ref:
        extra.append(f"references {f.ref}._id")
    if (f.presence or 1.0) < 0.95:
        extra.append("optional")
    if f.nullable:
        extra.append("may be null")
    return f"  {f.path}: {t}" + (f"  ({'; '.join(extra)})" if extra else "")


def schema_block(info, coll, docs: list[str], related: list[str], index_lines: list[str], max_fields: int = 70) -> str:
    lines = [f"DATABASE (current db): {info.schema_id}", f"COLLECTION: {coll.name}" + (f"   (~{info.coll_counts.get(coll.name)} documents)" if info.coll_counts.get(coll.name) else ""),
             "FIELDS (dotted paths are nested; these are the ONLY fields that exist besides _id):"]
    fl = [field_line(f) for f in coll.fields.values()]
    lines += fl[:max_fields]
    if len(fl) > max_fields:
        lines.append(f"  ... ({len(fl) - max_fields} more fields omitted)")
    if index_lines:
        lines.append("INDEXES: " + "; ".join(index_lines[:6]))
    if docs:
        lines.append("SAMPLE DOCUMENTS (real, possibly truncated):")
        lines += ["  " + d for d in docs]
    if related:
        lines.append("OTHER COLLECTIONS IN THE SAME DATABASE (you may $lookup / $unionWith them):")
        lines += ["  " + r for r in related]
    return "\n".join(lines)


def related_lines(info, coll, limit: int, rng: random.Random) -> list[str]:
    others = [c for c in info.schema.colls if c.name != coll.name]
    linked = [c for c in others if any(f.ref and (f.ref == coll.name or f.ref == coll.key) for f in c.fields.values())
              or any(f.ref and f.ref in (c.name, c.key) for f in coll.fields.values())]
    rest = [c for c in others if c not in linked]
    rng.shuffle(rest)
    out = []
    for c in (linked + rest)[:limit]:
        flds = []
        for f in c.fields.values():
            if f.path == "_id":
                continue
            flds.append(f"{f.path}:{f.t}" + (f"->{f.ref}" if f.ref else ""))
            if len(flds) >= 9:
                break
        out.append(f"{c.name} ({', '.join(flds)})")
    return out


def index_lines(coll) -> list[str]:
    out = []
    for ix in coll.indexes or []:
        keys = ix.get("keys") if isinstance(ix, dict) else None
        if keys:
            out.append("{" + ", ".join(f"{k}: {d}" for k, d in keys) + "}" + (" unique" if ix.get("unique") else ""))
    return out


def bson_repr(v, depth=0):
    """Documento do Mongo -> texto no estilo do shell (ObjectId("..."), ISODate("...")), para o prompt."""
    import datetime
    from bson import Binary, Decimal128, Int64, ObjectId
    if isinstance(v, ObjectId):
        return f'ObjectId("{v}")'
    if isinstance(v, datetime.datetime):
        return f'ISODate("{v.strftime("%Y-%m-%dT%H:%M:%S.") + format(v.microsecond // 1000, "03d")}Z")'
    if isinstance(v, Decimal128):
        return f'NumberDecimal("{v}")'
    if isinstance(v, Int64):
        return f"NumberLong({int(v)})"
    if isinstance(v, Binary):
        if v.subtype in (3, 4) and len(v) == 16:
            import uuid
            return f'UUID("{uuid.UUID(bytes=bytes(v))}")'
        return "BinData(...)"
    if isinstance(v, dict):
        if depth > 3:
            return "{...}"
        return "{ " + ", ".join(f"{k}: {bson_repr(x, depth + 1)}" for k, x in v.items()) + " }"
    if isinstance(v, (list, tuple)):
        items = [bson_repr(x, depth + 1) for x in v[:4]]
        return "[" + ", ".join(items) + (", ..." if len(v) > 4 else "") + "]"
    if isinstance(v, str):
        return json.dumps(v[:60], ensure_ascii=False)
    return json.dumps(v) if isinstance(v, (int, float, bool)) or v is None else repr(v)


def sample_docs(client, db: str, coll: str, n: int, max_chars: int, rng: random.Random) -> list[str]:
    try:
        total = client[db][coll].estimated_document_count()
        skip = rng.randrange(max(1, total - n)) if total > n else 0
        docs = list(client[db][coll].find({}, max_time_ms=3000).skip(skip).limit(n))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for d in docs:
        s = bson_repr(d)
        out.append(s if len(s) <= max_chars else s[:max_chars] + " ...}")
    return out


QUERY_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {
        "type": "object",
        "properties": {"intent": {"type": "string"}, "query": {"type": "string"}},
        "required": ["intent", "query"], "additionalProperties": False}}},
    "required": ["queries"], "additionalProperties": False,
}
QUERY_SCHEMA_NOINTENT = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {
        "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}},
    "required": ["queries"], "additionalProperties": False,
}

SYSTEM = (
    "You write realistic, READ-ONLY MongoDB shell (mongosh, JavaScript) code that people actually type in a MongoDB console. "
    "You always follow the field names and types of the schema you are given and never invent fields, collections or databases. "
    "You output only JSON. /no_think"
)


def build_messages(info, coll, specs: list[dict], docs: list[str], rng: random.Random, max_related: int = 3, extra_note: str = "",
                   want_intent: bool = False) -> list[dict]:
    """O PREFIXO do prompt (schema + amostras + regras) e identico para todas as chamadas de um mesmo grupo (schema, colecao, rng do grupo);
    so as fichas das consultas (persona/tarefa/estilo) vem depois -> o llama-server reaproveita o KV do prefixo (cache_prompt) no mesmo slot."""
    block = schema_block(info, coll, docs, related_lines(info, coll, max_related, rng), index_lines(coll))
    fmt = ('{"queries":[{"intent":"one sentence describing the query in Portuguese or English","query":"<the code>"} ...]}' if want_intent
           else '{"queries":[{"query":"<the code>"} ...]}')
    lines = [block, "",
             "RULES:",
             "- READ-ONLY: only find, findOne, aggregate, countDocuments, estimatedDocumentCount, distinct (and cursor methods like sort/limit/skip/project/"
             "toArray/map/forEach on results). NEVER insert, update, delete, replace, drop, create indexes, $out, $merge, bulkWrite, eval, $function, $where, load or require.",
             "- Use ONLY the fields listed above (dotted paths for nested ones, `$field` references in aggregation). Do not invent fields or enum values; "
             "use realistic values taken from the samples/ranges so that the query returns data.",
             f"- The current database is `{info.schema_id}` (so `db.{coll.name}` works); for getSiblingDB use exactly `{info.schema_id}`.",
             "- Each query must be complete, valid JavaScript for mongosh and must run as-is (declare every variable it uses).",
             "- Keep each query under ~35 lines. Do not return explanations inside the code.",
             "- Vary field choices, operators and values across the queries; do not copy the sample documents.",
             f"- Return JSON: {fmt}", ""]
    lines.append(f"Write {len(specs)} DIFFERENT queries for this collection, in order, exactly {len(specs)} items. Each query has its own brief:")
    for i, sp in enumerate(specs, 1):
        lines.append(f"QUERY {i}: written by {sp['persona_text']}.")
        lines.append(f"  Task: {sp['task_text']}.")
        lines.append(f"  Style: {style_text(sp)}.")
    if extra_note:
        lines.append(extra_note)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines) + "\n/no_think"}]


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def clean_query(q: str) -> str:
    q = q.strip("﻿")
    q = _FENCE.sub("", q.strip()).strip("\n")
    return q


def _salvage(text: str) -> list:
    """JSON truncado (finish=length): recupera os objetos {"query": ...} completos."""
    dec = json.JSONDecoder()
    out, i = [], 0
    while True:
        j = text.find('{"', i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text, j)
        except ValueError:
            i = j + 2
            continue
        if isinstance(obj, dict) and isinstance(obj.get("query"), str):
            out.append(obj)
        i = end
    return out


def parse_response(text: str) -> list[dict]:
    """JSON da resposta -> [{intent, query}] (tolerante a cercas de codigo, texto em volta e truncamento)."""
    text = text.strip()
    qs = None
    try:
        obj = json.loads(text)
        qs = obj.get("queries") if isinstance(obj, dict) else obj
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                qs = obj.get("queries") if isinstance(obj, dict) else obj
            except ValueError:
                qs = None
        if qs is None:
            qs = _salvage(text)
    out = []
    for q in qs or []:
        if isinstance(q, dict) and isinstance(q.get("query"), str) and q["query"].strip():
            out.append({"intent": str(q.get("intent") or "")[:300], "query": clean_query(q["query"])})
    return out
