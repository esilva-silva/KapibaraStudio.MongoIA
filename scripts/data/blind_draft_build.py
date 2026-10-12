#!/usr/bin/env python3
"""Gera o RASCUNHO do conjunto cego (40 itens `author: draft-agent`, `status: draft`) e valida as
referencias executando-as, SOMENTE LEITURA (find/aggregate/count/distinct com maxTimeMS), nos bancos
`kapi_fx_*` do MongoDB local.

  python scripts/data/blind_draft_build.py [--uri mongodb://127.0.0.1:27017] [--out FILE] [--no-exec]

Cada referencia e renderizada a partir de uma especificacao estruturada (extended JSON); a mesma
especificacao e executada, entao texto da referencia == o que foi validado. Alternativas (`alts`) sao
executadas e devem produzir o mesmo conjunto de resultados.

ATENCAO: sao rascunhos para revisao/reescrita humana; NAO sao o blind final e NAO devem ser selados.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind_guard as G  # noqa: E402

ROOT = G.ROOT
MAXMS = 8000
IDENT = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")


# ---- renderizacao mongosh -----------------------------------------------------------------

def jr(v) -> str:
    if isinstance(v, dict):
        if set(v) == {"$date"}:
            return f'ISODate("{v["$date"]}")'
        if set(v) == {"$numberDecimal"}:
            return f'NumberDecimal("{v["$numberDecimal"]}")'
        if not v:
            return "{}"
        return "{ " + ", ".join(f"{k if IDENT.match(k) else json.dumps(k)}: {jr(x)}" for k, x in v.items()) + " }"
    if isinstance(v, list):
        return "[" + ", ".join(jr(x) for x in v) + "]"
    return json.dumps(v, ensure_ascii=False)


def coll_ref(c: str) -> str:
    return f"db.{c}" if IDENT.match(c) else f'db.getCollection({json.dumps(c)})'


def render(coll: str, spec: dict) -> str:
    op = spec["op"]
    c = coll_ref(coll)
    if op == "find":
        s = f"{c}.find({jr(spec.get('filter', {}))}"
        if spec.get("projection"):
            s += f", {jr(spec['projection'])}"
        s += ")"
        if spec.get("sort"):
            s += f".sort({jr(spec['sort'])})"
        if spec.get("limit"):
            s += f".limit({spec['limit']})"
        return s
    if op == "aggregate":
        return f"{c}.aggregate({jr(spec['pipeline'])})"
    if op == "count":
        return f"{c}.countDocuments({jr(spec.get('filter', {}))})"
    if op == "distinct":
        return f'{c}.distinct({json.dumps(spec["field"])}, {jr(spec.get("filter", {}))})'
    raise ValueError(op)


# ---- execucao somente leitura -------------------------------------------------------------

def execute(client, dbname: str, coll: str, spec: dict):
    """Retorna lista de documentos/valores (JSON-safe, via bson.json_util)."""
    from bson import json_util
    col = client[dbname][coll]
    sp = json_util.loads(json.dumps(spec))
    op = sp["op"]
    if op == "find":
        cur = col.find(sp.get("filter", {}), sp.get("projection"), max_time_ms=MAXMS)
        if sp.get("sort"):
            cur = cur.sort(list(sp["sort"].items()))
        cur = cur.limit(sp.get("limit") or 5000)
        out = list(cur)
    elif op == "aggregate":
        if any(("$out" in s or "$merge" in s) for s in sp["pipeline"]):
            raise ValueError("pipeline com escrita proibido")
        out = list(col.aggregate(sp["pipeline"], maxTimeMS=MAXMS))
    elif op == "count":
        out = [col.count_documents(sp.get("filter", {}), maxTimeMS=MAXMS)]
    elif op == "distinct":
        out = list(col.distinct(sp["field"], sp.get("filter", {}), maxTimeMS=MAXMS))
    else:
        raise ValueError(op)
    return [json.loads(json_util.dumps(d, sort_keys=True)) for d in out]


def digest(rows) -> str:
    return hashlib.sha256("\n".join(sorted(json.dumps(r, sort_keys=True) for r in rows)).encode()).hexdigest()


# ---- contexto (resumo do schema) ----------------------------------------------------------

def schema_summary(schema_id: str, coll: str, extra=()):
    sch = None
    for f in (ROOT / "data" / "schemas").glob("*/*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d["schema_id"] == schema_id:
            sch = d
            break
    def fl(fields, p=""):
        o = []
        for k, v in fields.items():
            t = v["t"]
            if t == "object":
                o += fl(v["f"], p + k + ".")
            elif t == "array":
                it = v["item"]
                o += fl(it["f"], p + k + "[].") if it["t"] == "object" else [f"{p}{k}[]: {it['t']}"]
            else:
                s = f"{p}{k}: {t}"
                if v.get("values"):
                    s += " " + "|".join(map(str, v["values"]))
                o.append(s)
        return o
    lines = []
    for c in sch["collections"]:
        if c["name"] == coll or c["name"] in extra:
            lines.append(f"{c['name']}: " + ", ".join(fl(c["fields"])))
    return sch["domain"], "\n".join(lines)


# ---- itens --------------------------------------------------------------------------------

D = lambda s: {"$date": s}          # noqa: E731
DEC = lambda s: {"$numberDecimal": s}  # noqa: E731

ITEMS = []


def A(sid, coll, lang, before, spec, hole, after="\n", alts=(), note=None):
    ITEMS.append(dict(kind="autocomplete", schema=sid, coll=coll, lang=lang, before=before, spec=spec,
                      hole=hole, after=after, alts=list(alts), note=note))


def C(sid, coll, lang, instr, spec, alts=(), extra=(), note=None):
    ITEMS.append(dict(kind="chat", schema=sid, coll=coll, lang=lang, instr=instr, spec=spec,
                      alts=list(alts), extra=extra, note=note))


def K(sid, coll, lang, instr, broken, spec, alts=(), note=None):
    ITEMS.append(dict(kind="correction", schema=sid, coll=coll, lang=lang, instr=instr, broken=broken,
                      spec=spec, alts=list(alts), note=note))


F = lambda **kw: dict(op="find", **kw)  # noqa: E731

# ---- education / campus (holdout) ----
A("campus", "students", "pt-BR", "// alunos de ciência da computação com bolsa e média alta\nconst r = ",
  F(filter={"program": "cs", "gpa": {"$gte": 3.5}, "scholarship": True}), 'gpa: { $gte: 3.5 }')
A("campus", "enrollments", "en", "// average grade per term, passed enrollments only\nconst pipeline = ",
  dict(op="aggregate", pipeline=[{"$match": {"status": "passed"}},
                                 {"$group": {"_id": "$term", "avgGrade": {"$avg": "$grade"}}},
                                 {"$sort": {"_id": 1}}]),
  '{ $group: { _id: "$term", avgGrade: { $avg: "$grade" } } }')
C("campus", "courses", "pt-BR",
  "Liste as disciplinas com 4 ou mais créditos que têm aula na sexta-feira. Mostre só o código e o título.",
  F(filter={"credits": {"$gte": 4}, "schedule.day": "fri"}, projection={"code": 1, "title": 1, "_id": 0}),
  alts=[F(filter={"credits": {"$gte": 4}, "schedule": {"$elemMatch": {"day": "fri"}}},
          projection={"code": 1, "title": 1, "_id": 0})])
C("campus", "teachers", "en", "How many teachers in the math department have tenure?",
  dict(op="count", filter={"dept": "math", "tenured": True}))
A("campus", "teachers", "pt-BR", "// professores efetivos, os mais antigos primeiro\ndb.teachers.find({ tenured: true })",
  F(filter={"tenured": True}, sort={"hiredAt": 1}, limit=5), '.sort({ hiredAt: 1 }).limit(5)', after="")
C("campus", "students", "en",
  "For each program, give the number of students and the average GPA, highest average first.",
  dict(op="aggregate", pipeline=[{"$group": {"_id": "$program", "students": {"$sum": 1}, "avgGpa": {"$avg": "$gpa"}}},
                                 {"$sort": {"avgGpa": -1}}]))
K("campus", "enrollments", "pt-BR",
  "Esta consulta não retorna nenhum documento, mas existem matrículas aprovadas com nota alta. Corrija.",
  'db.enrollments.find({ status: "passed", grade: { $gte: "7" } })',
  F(filter={"status": "passed", "grade": {"$gte": 7}}))
K("campus", "courses", "en",
  "This returns no documents. The schedule days are stored as three-letter abbreviations. Fix the query.",
  'db.courses.find({ "schedule.day": "friday" })', F(filter={"schedule.day": "fri"}))
A("campus", "enrollments", "en", "// failed enrollments joined with the student document\ndb.enrollments.aggregate([\n  { $match: { status: \"failed\" } },\n  ",
  dict(op="aggregate", pipeline=[{"$match": {"status": "failed"}},
                                 {"$lookup": {"from": "students", "localField": "studentId", "foreignField": "_id", "as": "student"}},
                                 {"$unwind": "$student"}, {"$limit": 20}]),
  '{ $lookup: { from: "students", localField: "studentId", foreignField: "_id", as: "student" } }',
  after=",\n  { $unwind: \"$student\" },\n  { $limit: 20 }\n])", note="before/after montados manualmente")

# ---- legal / juris (holdout) ----
A("juris", "Cases", "pt-BR", "// processos abertos no TJSP distribuídos a partir de 2024\nconst abertos = ",
  F(filter={"Court": "TJSP", "Status": "open", "FiledAt": {"$gte": D("2024-01-01T00:00:00Z")}}),
  'FiledAt: { $gte: ISODate("2024-01-01T00:00:00Z") }')
A("juris", "Documents", "en", "// count confidential rulings\nconst n = ",
  dict(op="count", filter={"Kind": "ruling", "Confidential": True}), 'Confidential: true')
C("juris", "Hearings", "pt-BR",
  "Quais audiências estão marcadas a partir de 1º de julho de 2026? Mostre data, sala e juiz, da mais próxima para a mais distante.",
  F(filter={"At": {"$gte": D("2026-07-01T00:00:00Z")}}, projection={"At": 1, "Room": 1, "Judge": 1, "_id": 0}, sort={"At": 1}))
C("juris", "Cases", "en", "List the distinct courts that have at least one case with status stayed.",
  dict(op="distinct", field="Court", filter={"Status": "stayed"}))
C("juris", "Cases", "pt-BR",
  "Para os processos do TJSP, conte quantas partes existem de cada papel (autor, réu, testemunha).",
  dict(op="aggregate", pipeline=[{"$match": {"Court": "TJSP"}}, {"$unwind": "$Parties"},
                                 {"$group": {"_id": "$Parties.Role", "total": {"$sum": 1}}}, {"$sort": {"_id": 1}}]))
A("juris", "Cases", "en", "// total claimed value per court for closed cases\ndb.Cases.aggregate([\n  { $match: { Status: \"closed\" } },\n  { $group: { _id: \"$Court\", ",
  dict(op="aggregate", pipeline=[{"$match": {"Status": "closed"}},
                                 {"$group": {"_id": "$Court", "total": {"$sum": "$Value"}}}]),
  'total: { $sum: "$Value" }', after=" } }\n])", note="before/after montados manualmente")
K("juris", "Lawyers", "pt-BR",
  "Não retorna nada. Quero os advogados ativos; os campos desta coleção são em PascalCase.",
  'db.Lawyers.find({ active: true })', F(filter={"Active": True}))
K("juris", "Cases", "en",
  "This returns nothing. I want cases that have at least one party with the plaintiff role.",
  'db.Cases.find({ Parties: { Role: "plaintiff" } })',
  F(filter={"Parties": {"$elemMatch": {"Role": "plaintiff"}}}), alts=[F(filter={"Parties.Role": "plaintiff"})])
A("juris", "Documents", "pt-BR", "// as 10 petições mais recentes com mais de 100 páginas\ndb.Documents.find({ Kind: \"petition\", Pages: { $gt: 100 } })",
  F(filter={"Kind": "petition", "Pages": {"$gt": 100}}, sort={"Filed": -1}, limit=10), '.sort({ Filed: -1 }).limit(10)', after="")

# ---- finance / ledgerly ----
A("ledgerly", "transactions", "pt-BR", "// débitos de viagem acima de mil\nconst viagens = ",
  F(filter={"direction": "debit", "category": "travel", "amount": {"$gt": DEC("1000")}}),
  'amount: { $gt: NumberDecimal("1000") }')
C("ledgerly", "transactions", "en",
  "Show the total credited amount per category for transactions that were not reversed, largest first.",
  dict(op="aggregate", pipeline=[{"$match": {"direction": "credit", "reversed": False}},
                                 {"$group": {"_id": "$category", "total": {"$sum": "$amount"}}},
                                 {"$sort": {"total": -1}}]))
C("ledgerly", "accounts", "pt-BR",
  "Quais contas congeladas em BRL têm saldo acima de 10 mil? Traga o IBAN e o titular.",
  F(filter={"frozen": True, "currency": "BRL", "balance": {"$gt": DEC("10000")}}, projection={"iban": 1, "holder": 1, "_id": 0}))

# ---- saas / tenantly ----
A("tenantly", "users", "en", "// owners and admins without MFA\nconst risky = ",
  F(filter={"role": {"$in": ["owner", "admin"]}, "mfa_enabled": False}), 'role: { $in: ["owner", "admin"] }')
C("tenantly", "invoices", "pt-BR", "Quantas faturas em USD estão em aberto (status open)?",
  dict(op="count", filter={"currency": "USD", "status": "open"}))
A("tenantly", "invoices", "pt-BR", "// os 5 tenants que mais pagaram\ndb.invoices.aggregate([\n  { $match: { status: \"paid\" } },\n  { $group: { _id: \"$tenant_id\", totalPago: { $sum: \"$amount\" } } },\n  ",
  dict(op="aggregate", pipeline=[{"$match": {"status": "paid"}},
                                 {"$group": {"_id": "$tenant_id", "totalPago": {"$sum": "$amount"}}},
                                 {"$sort": {"totalPago": -1}}, {"$limit": 5}]),
  '{ $sort: { totalPago: -1 } }', after=",\n  { $limit: 5 }\n])", note="before/after montados manualmente")

# ---- hr / pessoas ----
C("pessoas", "vagas", "en",
  "Which job openings are still open and require 'cedar'? Return only the title and the salary range.",
  F(filter={"aberta": True, "requisitos": "cedar"}, projection={"titulo": 1, "faixa": 1, "_id": 0}))
A("pessoas", "funcionarios", "pt-BR", "// gerentes ainda na empresa (sem data de desligamento)\nconst ativos = ",
  F(filter={"cargo": "gerente", "desligamento": {"$exists": False}}), 'desligamento: { $exists: false }')

# ---- healthcare-synthetic / clinica_sint ----
C("clinica_sint", "consultas", "pt-BR",
  "Liste as consultas de urgência já realizadas com pressão sistólica acima de 140. Mostre o início e os sinais.",
  F(filter={"tipo": "urgencia", "situacao": "realizada", "sinais.pa_sis": {"$gt": 140}},
    projection={"inicio": 1, "sinais": 1, "_id": 0}))
A("clinica_sint", "exames", "en", "// average result value per exam type\ndb.exames.aggregate([\n  { $group: { _id: \"$tipo\", ",
  dict(op="aggregate", pipeline=[{"$group": {"_id": "$tipo", "avgValue": {"$avg": "$resultado.valor"}}}]),
  'avgValue: { $avg: "$resultado.valor" }', after=" } }\n])", note="before/after montados manualmente")

# ---- logistics / freightline ----
A("freightline", "shipments", "pt-BR", "// remessas em trânsito com destino ao Brasil\nconst r = ",
  F(filter={"status": "in_transit", "destination.country": "BR"}), '"destination.country": "BR"')
C("freightline", "routes", "en",
  "Find road routes longer than 500 km that are not restricted, shortest transit time first, top 10.",
  F(filter={"mode": "road", "distanceKm": {"$gt": 500}, "restricted": False}, sort={"transitHours": 1}, limit=10))
C("freightline", "vehicles", "en", "Which trucks have less than 20% fuel left? Return the plate and the fuel level.",
  F(filter={"type": "truck", "telemetry.fuelPct": {"$lt": 20}}, projection={"plate": 1, "telemetry.fuelPct": 1, "_id": 0}))

# ---- commerce / shopfront ----
A("shopfront", "products", "en", "// active audio products running low on stock\nconst low = ",
  F(filter={"category": "audio", "active": True, "stock": {"$lt": 10}}), 'stock: { $lt: 10 }')
C("shopfront", "orders", "pt-BR", "Quais pedidos pagos tiveram cupom aplicado? Mostre o total e a moeda.",
  F(filter={"status": "paid", "coupon": {"$ne": None}}, projection={"total": 1, "currency": 1, "_id": 0}))
A("shopfront", "orders", "pt-BR", "// unidades vendidas por produto\ndb.orders.aggregate([\n  ",
  dict(op="aggregate", pipeline=[{"$unwind": "$items"},
                                 {"$group": {"_id": "$items.productId", "unidades": {"$sum": "$items.qty"}}},
                                 {"$sort": {"unidades": -1}}, {"$limit": 10}]),
  '{ $unwind: "$items" }', after=",\n  { $group: { _id: \"$items.productId\", unidades: { $sum: \"$items.qty\" } } },\n  { $sort: { unidades: -1 } },\n  { $limit: 10 }\n])",
  note="before/after montados manualmente")

# ---- travel / roteiros ----
C("roteiros", "voos", "en", "Scheduled flights out of Salvador under 2000, cheapest first.",
  F(filter={"origem": "Salvador", "status": "agendado", "preco": {"$lt": DEC("2000")}}, sort={"preco": 1}))

# ---- gaming / arcadia ----
A("arcadia", "players", "en", "// active high-level players in ap-south\nconst vets = ",
  F(filter={"banned": False, "level": {"$gte": 50}, "region": "ap-south"}), 'region: "ap-south"')
C("arcadia", "players", "pt-BR", "Qual a média de vitórias dos jogadores por região, considerando só os não banidos?",
  dict(op="aggregate", pipeline=[{"$match": {"banned": False}},
                                 {"$group": {"_id": "$region", "mediaVitorias": {"$avg": "$stats.wins"}}}]))

# ---- iot / energy / content ----
A("sensorgrid", "devices", "en", "// online devices with a low battery\nconst weak = ",
  F(filter={"online": True, "battery_pct": {"$lt": 20}}), 'battery_pct: { $lt: 20 }')
A("gridwatt", "outages", "en", "// affected customers per outage cause\ndb.outages.aggregate([\n  { $group: { _id: \"$cause\", ",
  dict(op="aggregate", pipeline=[{"$group": {"_id": "$cause", "affected": {"$sum": "$affected"}}}]),
  'affected: { $sum: "$affected" }', after=" } }\n])", note="before/after montados manualmente")
A("newsroom", "Articles", "pt-BR", "// as 5 matérias publicadas mais recentes com a tag nova\nconst r = ",
  F(filter={"Status": "published", "Tags": "nova"}, sort={"PublishedAt": -1}, limit=5), 'Tags: "nova"')


# ---- montagem -----------------------------------------------------------------------------

def hole_cut(stmt: str, hole: str):
    n = stmt.count(hole)
    if n != 1:
        raise SystemExit(f"hole ocorre {n}x em: {stmt!r} / {hole!r}")
    i = stmt.index(hole)
    return i, i + len(hole)


def build(client):
    from pymongo.errors import PyMongoError  # noqa: F401
    server = client.server_info()["version"]
    out, counters = [], {"autocomplete": 0, "chat": 0, "correction": 0}
    meta = json.loads((ROOT / "data" / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
    dbs = {}
    for e in meta["entries"]:
        dbs[e["db"]] = True
    for it in ITEMS:
        domain, summary = schema_summary(it["schema"], it["coll"], it.get("extra", ()))
        db = f"kapi_fx_{domain.replace('-', '_')}_{it['schema']}"
        if db not in dbs:
            raise SystemExit(f"banco inesperado {db}")
        stmt = render(it["coll"], it["spec"])
        rows = execute(client, db, it["coll"], it["spec"]) if client else None
        if client:
            # count/distinct podem retornar [0]; exige resultado nao vazio
            if not rows or rows == [0]:
                raise SystemExit(f"referencia sem resultados: {db} {stmt}")
        alt_texts, alt_ok = [], []
        counters[it["kind"]] += 1
        iid = f"blind-{it['kind']}-{counters[it['kind']]:03d}"
        item = {"id": iid, "kind": it["kind"], "language": it["lang"], "domain": domain,
                "schema_id": it["schema"], "collection": it["coll"],
                "group": f"{it['schema']}::{it['coll']}",
                "author": "draft-agent", "reviewed_by": None, "status": "draft"}
        if it["kind"] == "autocomplete":
            i, j = hole_cut(stmt, it["hole"])
            before = it["before"]
            # quando `before` ja contem parte do statement (montagem manual), o prefixo e before + stmt[:i] apenas se
            # before nao repete o inicio; aqui `before` e sempre so o contexto anterior ao statement renderizado
            manual = "db." in before
            if manual:
                prefix, suffix = before, it["after"]
                full = before + it["hole"] + it["after"]
            else:
                prefix, suffix = before + stmt[:i], stmt[j:] + it["after"]
                full = prefix + it["hole"] + suffix
            chk = re.sub(r"\s+", "", re.sub(r"//[^\n]*\n", "", full))
            chk = re.sub(r"^const\w+=", "", chk).rstrip(";")
            if chk != re.sub(r"\s+", "", stmt):
                raise SystemExit(f"script do rascunho nao equivale a referencia executada: {full!r} vs {stmt!r}")
            item.update(prefix=prefix, suffix=suffix, references=[it["hole"]])
            item["notes"] = f"script completo (para conferencia): {full}"
            for alt in it["alts"]:
                at = render(it["coll"], alt)
                if not (at.startswith(stmt[:i]) and at.endswith(stmt[j:])):
                    raise SystemExit(f"alt incompativel com o corte: {at}")
                alt_texts.append(at[i:len(at) - (len(stmt) - j)])
                alt_ok.append(alt)
        else:
            item["instruction"] = it["instr"]
            if it["kind"] == "chat":
                item["context"] = summary
                item["references"] = [stmt]
            else:
                item["context"] = it["broken"] + "\n\n" + summary
                item["references"] = [stmt]
            for alt in it["alts"]:
                alt_texts.append(render(it["coll"], alt))
                alt_ok.append(alt)
        for text, alt in zip(alt_texts, alt_ok):
            if client:
                r2 = execute(client, db, it["coll"], alt)
                if digest(r2) != digest(rows):
                    raise SystemExit(f"alternativa diverge em {iid}: {text}")
            item["references"].append(text)
        item["validation"] = {
            "executed": bool(client), "server_version": server, "db": db, "op": it["spec"]["op"],
            "n_results": len(rows) if rows is not None else None,
            "result_sha256": digest(rows) if rows is not None else None,
            "alternatives_checked": len(alt_ok) if client else 0,
            "validated_at": None,
            "executable_spec": it["spec"],
        }
        if it["kind"] != "autocomplete" and it.get("note"):
            item["notes"] = it["note"]
        item["created_at"] = "2026-10-11"
        out.append(item)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="mongodb://127.0.0.1:27017")
    ap.add_argument("--out", default=str(G.blind_dir() / "draft" / "draft_items.jsonl"),
                    help="padrao: dentro do blind (gitignored e protegido pelo guard)")
    ap.add_argument("--no-exec", action="store_true")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    client = None
    if not a.no_exec:
        from pymongo import MongoClient
        client = MongoClient(a.uri, serverSelectionTimeoutMS=5000)
    items = build(client)
    import datetime as dt
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for it in items:
        it["validation"]["validated_at"] = ts if client else None
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"ok: {len(items)} itens -> {a.out}")


if __name__ == "__main__":
    main()
