#!/usr/bin/env python3
"""Gera data/schemas/<domain>/<schema_id>.json (spec fase-03 §4.2).

Determinístico: seed fixa 20261008. Saída só depende deste arquivo + seed.
Uso: python scripts/data/build_schemas.py [--seed 20261008] [--out data/schemas]
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_roles import assign_roles  # noqa: E402

SEED = 20261008
ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------- DSL ----
def _f(t, p=1.0, n=0.0, **kw):
    d = {"t": t, "p": p, "n": n}
    d.update(kw)
    return d


def oid(ref=None, p=1.0, n=0.0): return _f("objectid", p, n, ref=ref)
def uid(v=4, p=1.0, n=0.0): return _f("uuid", p, n, v=v)
def text(g="word", p=1.0, n=0.0, **kw): return _f("string", p, n, g=g, **kw)
def enum(vals, w=None, p=1.0, n=0.0): return _f("string", p, n, g="enum", values=vals, w=w)
def i32(lo, hi, p=1.0, n=0.0): return _f("int32", p, n, lo=lo, hi=hi)
def i64(lo, hi, p=1.0, n=0.0): return _f("int64", p, n, lo=lo, hi=hi)
def dbl(lo, hi, p=1.0, n=0.0): return _f("double", p, n, lo=lo, hi=hi)
def dec(lo, hi, p=1.0, n=0.0): return _f("decimal128", p, n, lo=lo, hi=hi)
def flag(pt=0.5, p=1.0, n=0.0): return _f("bool", p, n, pt=pt)
def date(lo="2025-01-01", hi="2026-09-30", p=1.0, n=0.0): return _f("date", p, n, lo=lo, hi=hi)
def obj(f, p=1.0, n=0.0): return _f("object", p, n, f=f)
def arr(item, lo=0, hi=4, p=1.0, n=0.0): return _f("array", p, n, item=item, min=lo, max=hi)
def tags(p="o"): return arr(text("word"), 0, 4, p=p)


def IX(keys, **opt):
    return {"keys": keys, **opt}


def C(key, name, fields, idx=(), _id=None, style=None):
    return {"key": key, "name": name, "fields": fields, "idx": list(idx), "_id": _id, "style": style}


# ------------------------------------------------------------- schemas ----
def schema_defs():
    S = []

    def add(domain, sid, style, desc, cols, holdout=False):
        S.append(dict(domain=domain, schema_id=sid, style=style, desc=desc, cols=cols, holdout=holdout))

    # ---- commerce / shopfront (camelCase, EN, hífen/espaço em coleções)
    add("commerce", "shopfront", "camel", "Loja online EN, camelCase", [
        C("customers", "customers", {
            "full_name": text("name"), "email": text("email"), "phone": text("phone", p="o"),
            "loyalty_tier": enum(["bronze", "silver", "gold", "platinum"], [50, 30, 15, 5], p=0.8),
            "marketing_opt_in": flag(0.4),
            "address": obj({"street": text("title"), "city": text("city"), "country": text("country"),
                            "geo": obj({"lat": dbl(-34, 5), "lng": dbl(-74, -35)}, p="o")}, p=0.9),
            "created_at": date(), "last_login_at": date(p="o", n=0.1),
            "tags": tags()},
            [IX("email", unique=True), IX("address.country,loyalty_tier"), IX("created_at,-last_login_at")]),
        C("products", "products", {
            "sku": text("sku"), "title": text("title"), "category": enum(["audio", "video", "home", "toys", "books", "garden"]),
            "price": dec(5, 900), "stock": i32(0, 500), "active": flag(0.8),
            "dimensions": obj({"w": dbl(1, 90), "h": dbl(1, 90), "d": dbl(1, 90), "unit": enum(["cm", "in"], [80, 20])}, p="o"),
            "attributes": arr(obj({"k": text("word"), "v": text("word")}), 0, 5, p="o"),
            "supplier_id": oid("suppliers", p="o"), "tags": tags()},
            [IX("sku", unique=True), IX("category,-price"), IX("tags"),
             IX("stock", partial={"active": True}), IX("title", unique=False, text=True)]),
        C("orders", "orders", {
            "customer_id": oid("customers"), "status": enum(["pending", "paid", "shipped", "delivered", "cancelled"], [10, 25, 25, 30, 10]),
            "total": dec(10, 3000), "currency": enum(["BRL", "USD", "EUR"], [70, 20, 10]),
            "placed_at": date(), "paid_at": date(p="o", n=0.05),
            "items": arr(obj({"product_id": oid("products"), "qty": i32(1, 6), "unit_price": dec(5, 900),
                              "discount": dbl(0, 0.4, p="o")}), 1, 5),
            "shipping": obj({"method": enum(["standard", "express", "pickup"]), "tracking": text("hex", p="o"),
                             "address": obj({"city": text("city"), "zip": text("code", prefix="Z"), "geo": obj({"lat": dbl(-34, 5), "lng": dbl(-74, -35)}, p="o")}, p=0.9)}, p="o"),
            "coupon": text("code", prefix="CP", p="o", n=0.1), "flags": arr(text("word"), 0, 3, p="o")},
            [IX("customer_id,-placed_at"), IX("status,placed_at"), IX("items.product_id"),
             IX("coupon", partial={"coupon": {"$exists": True}}, sparse_note=None)]),
        C("order-items-archive", "order-items-archive", {
            "order_id": oid("orders"), "line": i32(1, 20), "sku": text("sku"), "qty": i32(1, 10),
            "price": dec(1, 500), "archived_at": date("2025-06-01", "2026-09-30")},
            [IX("order_id,line", unique=False), IX("archived_at", ttl=315360000)]),
        C("suppliers", "suppliers", {
            "name": text("title"), "country": text("country"), "rating": dbl(1, 5, p="o"),
            "contact": obj({"email": text("email"), "phone": text("phone", p="o")}),
            "active": flag(0.85)},
            [IX("name", unique=True, partial={"active": True}, uniq_note=None)]),
    ])

    # ---- commerce / loja_virtual (pt-BR sem acento, snake)
    add("commerce", "loja_virtual", "snake", "Loja pt-BR sem acento, snake_case", [
        C("clientes", "clientes", {
            "nome_completo": text("name"), "email": text("email"), "telefone": text("phone", p="o"),
            "endereco": obj({"rua": text("title"), "cidade": text("city"), "estado": enum(["SP", "RJ", "MG", "BA", "RS", "PR"]),
                             "cep": text("code", prefix="CEP")}, p=0.9),
            "data_cadastro": date(), "ativo": flag(0.9), "preferencias": arr(text("word"), 0, 4, p="o")},
            [IX("email", unique=True), IX("endereco.estado,ativo"), IX("data_cadastro")]),
        C("produtos", "produtos", {
            "codigo": text("sku"), "descricao": text("title"), "preco": dec(3, 1500), "estoque": i32(0, 800),
            "categoria": enum(["eletronicos", "casa", "moda", "esporte", "livros"]),
            "peso_kg": dbl(0.05, 30, p="o"), "variacoes": arr(obj({"cor": text("word"), "tamanho": enum(["P", "M", "G"])}), 0, 4, p="o")},
            [IX("codigo", unique=True), IX("categoria,-preco"), IX("estoque", partial={"estoque": {"$gt": 0}})]),
        C("pedidos", "pedidos", {
            "cliente_id": oid("clientes"), "situacao": enum(["aberto", "pago", "enviado", "entregue", "cancelado"], [10, 25, 25, 30, 10]),
            "valor_total": dec(10, 5000), "criado_em": date(),
            "itens": arr(obj({"produto_id": oid("produtos"), "quantidade": i32(1, 5), "preco_unitario": dec(3, 1500)}), 1, 4),
            "pagamento": obj({"forma": enum(["pix", "cartao", "boleto"], [50, 40, 10]), "parcelas": i32(1, 12, p="o"),
                              "aprovado": flag(0.9)}, p=0.95)},
            [IX("cliente_id,-criado_em"), IX("situacao"), IX("itens.produto_id"), IX("pagamento.forma,pagamento.aprovado")]),
        C("Avaliacoes", "Avaliacoes", {
            "produto_id": oid("produtos"), "cliente_id": oid("clientes"), "nota": i32(1, 5),
            "comentario": text("sentence", p="o", n=0.1), "data": date()},
            [IX("produto_id,nota")]),
    ])

    S_ = add
    S_("saas", "tenantly", "snake", "SaaS multi-tenant, snake_case", [
        C("tenants", "tenants", {
            "name": text("title"), "plan": enum(["free", "team", "business", "enterprise"], [40, 30, 20, 10]),
            "seats": i32(1, 500), "mrr": dec(0, 20000), "created_at": date(), "trial_ends_at": date("2026-01-01", "2026-12-31", p="o", n=0.3),
            "settings": obj({"sso": flag(0.2), "locale": enum(["en", "pt-BR", "es"]), "limits": obj({"api_rpm": i32(60, 6000), "storage_gb": i64(1, 5000)})}, p=0.9)},
            [IX("name", unique=True), IX("plan,-mrr"), IX("trial_ends_at", partial={"trial_ends_at": {"$exists": True}})]),
        C("users", "users", {
            "tenant_id": oid("tenants"), "email": text("email"), "role": enum(["owner", "admin", "member", "viewer"], [5, 15, 60, 20]),
            "display_name": text("name"), "mfa_enabled": flag(0.35), "last_seen_at": date(p="o", n=0.1),
            "prefs": obj({"theme": enum(["light", "dark"]), "notifications": obj({"email": flag(0.7), "push": flag(0.4)})}, p="o")},
            [IX("email", unique=True), IX("tenant_id,role")]),
        C("api_keys", "api_keys", {
            "tenant_id": oid("tenants"), "key_id": uid(4), "prefix": text("hex"), "scopes": arr(enum(["read", "write", "admin", "billing"]), 1, 3),
            "created_at": date(), "revoked": flag(0.15), "expires_at": date("2026-06-01", "2027-12-31", p="o")},
            [IX("key_id", unique=True), IX("tenant_id,revoked")]),
        C("audit_log", "audit_log", {
            "tenant_id": oid("tenants"), "actor_id": oid("users"), "action": enum(["login", "create", "update", "delete", "export", "invite"]),
            "at": date(), "ip": text("ip"), "meta": obj({"user_agent": text("word"), "request_id": uid(4), "changes": arr(obj({"field": text("word"), "from": text("word", n=0.3), "to": text("word", n=0.3)}), 0, 3, p="o")}, p=0.9)},
            [IX("tenant_id,-at"), IX("action,at"), IX("at", ttl=315360000)]),
        C("invoices", "invoices", {
            "tenant_id": oid("tenants"), "number": text("code", prefix="INV-"), "amount": dec(0, 20000), "currency": enum(["USD", "BRL", "EUR"]),
            "status": enum(["draft", "open", "paid", "void"], [5, 20, 70, 5]), "issued_at": date(),
            "lines": arr(obj({"desc": text("title"), "qty": i32(1, 50), "unit": dec(1, 400)}), 1, 4)},
            [IX("number", unique=True), IX("tenant_id,status")]),
    ])

    S_("logistics", "freightline", "camel", "Logística / frete, camelCase", [
        C("warehouses", "warehouses", {
            "code": text("code", prefix="WH"), "name": text("title"), "city": text("city"),
            "capacityPallets": i32(100, 20000), "location": obj({"type": enum(["Point"]), "coordinates": arr(dbl(-74, -35), 2, 2)}),
            "docks": arr(obj({"id": i32(1, 30), "kind": enum(["inbound", "outbound", "cross"]), "busy": flag(0.3)}), 1, 6)},
            [IX("code", unique=True), IX("city")]),
        C("shipments", "shipments", {
            "trackingCode": text("code", prefix="TRK"), "origin": oid("warehouses"), "destination": obj({"city": text("city"), "country": text("country"), "zip": text("code", prefix="Z", p="o")}),
            "weightKg": dbl(0.1, 2000), "volumeM3": dec(0, 30, p="o"), "status": enum(["created", "in_transit", "customs", "delivered", "returned"], [10, 40, 10, 35, 5]),
            "createdAt": date(), "deliveredAt": date(p="o", n=0.2), "carrierId": oid("carriers"),
            "events": arr(obj({"at": date(), "type": enum(["scan", "depart", "arrive", "exception"]), "note": text("sentence", p="o")}), 1, 6)},
            [IX("trackingCode", unique=True), IX("status,-createdAt"), IX("carrierId,status"), IX("destination.country,destination.city")]),
        C("carriers", "carriers", {
            "name": text("title"), "scac": text("code", prefix="C"), "modes": arr(enum(["road", "air", "sea", "rail"]), 1, 3),
            "onTimeRate": dbl(0.5, 1), "active": flag(0.9), "contracts": arr(obj({"lane": text("word"), "ratePerKg": dec(0.1, 12)}), 0, 4, p="o")},
            [IX("scac", unique=True), IX("modes,onTimeRate")]),
        C("routes", "routes", {
            "fromWh": oid("warehouses"), "toWh": oid("warehouses"), "distanceKm": i32(10, 4000), "mode": enum(["road", "air", "sea", "rail"], [60, 15, 15, 10]),
            "transitHours": dbl(2, 400), "restricted": flag(0.1, p="o")},
            [IX("fromWh,toWh"), IX("mode,distanceKm")]),
        C("vehicles", "vehicles", {
            "plate": text("code", prefix="VX"), "carrierId": oid("carriers"), "type": enum(["truck", "van", "bike", "cargo-plane"]),
            "odometerKm": i64(0, 900000), "lastService": date(p="o"), "telemetry": obj({"fuelPct": dbl(0, 100), "engineTempC": dbl(60, 120), "gps": obj({"lat": dbl(-34, 5), "lng": dbl(-74, -35)})}, p=0.7)},
            [IX("plate", unique=True), IX("carrierId,type")]),
    ])

    S_("iot", "sensorgrid", "snake", "IoT sensores e leituras, snake_case", [
        C("devices", "devices", {
            "device_uuid": uid(4), "model": enum(["TX-100", "TX-200", "HX-9", "ZX-Mini"]), "firmware": text("code", prefix="v"),
            "site": text("city"), "installed_at": date(), "online": flag(0.8), "battery_pct": dbl(0, 100, p="o", n=0.1),
            "labels": obj({"floor": i32(0, 20, p="o"), "zone": text("word", p="o")}, p="o")},
            [IX("device_uuid", unique=True), IX("model,online"), IX("site")]),
        C("readings", "readings", {
            "device_id": oid("devices"), "ts": date("2026-08-01", "2026-09-30"), "metric": enum(["temp", "humidity", "pressure", "co2", "vibration"]),
            "value": dbl(-10, 400), "unit": enum(["C", "pct", "hPa", "ppm", "mm_s"]), "quality": i32(0, 100, p="o"),
            "tags": obj({"batch": i32(1, 50, p="o"), "calibrated": flag(0.9)}, p="o")},
            [IX("device_id,-ts"), IX("metric,ts"), IX("ts", ttl=315360000), IX("value", partial={"quality": {"$gte": 90}})]),
        C("alerts", "alerts", {
            "device_id": oid("devices"), "severity": enum(["info", "warn", "critical"], [60, 30, 10]), "raised_at": date(),
            "cleared_at": date(p="o", n=0.4), "rule": text("slug"), "ack_by": text("name", p="o", n=0.3)},
            [IX("device_id,raised_at"), IX("severity,cleared_at")]),
        C("firmware releases", "firmware releases", {
            "version": text("code", prefix="fw"), "released_at": date(), "notes": text("sentence", p="o"),
            "checksums": arr(obj({"algo": enum(["sha256", "md5"]), "hex": text("hex")}), 1, 2), "mandatory": flag(0.2)},
            [IX("version", unique=True)]),
    ])

    S_("hr", "pessoas", "snake", "RH pt-BR sem acento, snake_case", [
        C("funcionarios", "funcionarios", {
            "matricula": text("code", prefix="M"), "nome": text("name"), "email_corporativo": text("email"),
            "cargo": enum(["analista", "gerente", "diretor", "estagiario", "coordenador"], [40, 15, 3, 12, 30]),
            "departamento_id": oid("departamentos"), "salario": dec(1800, 40000), "admissao": date("2015-01-01", "2026-09-01"),
            "desligamento": date("2024-01-01", "2026-09-30", p="o", n=0.5), "ferias": arr(obj({"inicio": date(), "dias": i32(5, 30)}), 0, 3, p="o"),
            "dependentes": arr(obj({"nome": text("name"), "parentesco": enum(["filho", "conjuge", "pai", "mae"])}), 0, 3, p="o")},
            [IX("matricula", unique=True), IX("departamento_id,cargo"), IX("admissao")]),
        C("departamentos", "departamentos", {
            "nome": text("title"), "centro_custo": text("code", prefix="CC"), "orcamento": dec(10000, 5000000), "gestor_id": oid("funcionarios", p="o")},
            [IX("centro_custo", unique=True)]),
        C("vagas", "vagas", {
            "titulo": text("title"), "departamento_id": oid("departamentos"), "aberta": flag(0.6),
            "requisitos": arr(text("word"), 1, 5), "faixa": obj({"min": dec(1500, 10000), "max": dec(10000, 30000)}, p="o"), "publicada_em": date()},
            [IX("aberta,-publicada_em"), IX("requisitos")]),
        C("ponto", "ponto", {
            "funcionario_id": oid("funcionarios"), "dia": date("2026-07-01", "2026-09-30"), "entrada": text("code", prefix="0"),
            "horas": dbl(0, 12), "justificativa": text("sentence", p=0.3)},
            [IX("funcionario_id,-dia"), IX("dia", ttl=315360000)]),
    ])

    S_("content", "newsroom", "pascal", "CMS / portal de notícias, PascalCase", [
        C("Articles", "Articles", {
            "Slug": text("slug"), "Title": text("title"), "Body": text("sentence"), "AuthorId": oid("Authors"),
            "Status": enum(["draft", "review", "published", "archived"], [15, 10, 65, 10]), "PublishedAt": date(p="o", n=0.1),
            "Tags": arr(text("word"), 0, 5), "Stats": obj({"Views": i64(0, 5000000), "Likes": i32(0, 20000), "ShareCount": i32(0, 5000, p="o")}, p=0.9),
            "Seo": obj({"Description": text("sentence"), "Keywords": arr(text("word"), 0, 5)}, p="o")},
            [IX("Slug", unique=True), IX("Status,-PublishedAt"), IX("Tags"), IX("AuthorId,Status")]),
        C("Authors", "Authors", {
            "Name": text("name"), "Email": text("email"), "Bio": text("sentence", p="o"), "Verified": flag(0.4),
            "Social": obj({"Twitter": text("slug", p="o"), "Site": text("url", p="o")}, p="o")},
            [IX("Email", unique=True)]),
        C("Comments", "Comments", {
            "ArticleId": oid("Articles"), "UserName": text("name"), "Text": text("sentence"), "CreatedAt": date(),
            "Score": i32(-5, 50), "Replies": arr(obj({"UserName": text("name"), "Text": text("sentence"), "CreatedAt": date()}), 0, 3, p="o"),
            "Flagged": flag(0.05)},
            [IX("ArticleId,-CreatedAt"), IX("Flagged", partial={"Flagged": True})]),
        C("Media Library", "Media Library", {
            "FileName": text("slug"), "MimeType": enum(["image/jpeg", "image/png", "video/mp4", "application/pdf"]),
            "SizeBytes": i64(1000, 900000000), "Uploaded": date(), "Meta": obj({"Width": i32(100, 8000, p="o"), "Height": i32(100, 8000, p="o"), "DurationSec": dbl(1, 3600, p="o")}, p="o")},
            [IX("MimeType,Uploaded")]),
    ])

    S_("finance", "ledgerly", "camel", "Contabilidade / pagamentos, camelCase", [
        C("accounts", "accounts", {
            "iban": text("code", prefix="BR"), "holder": text("name"), "type": enum(["checking", "savings", "credit", "investment"], [50, 30, 15, 5]),
            "balance": dec(-5000, 250000), "currency": enum(["BRL", "USD", "EUR"], [80, 15, 5]), "openedAt": date("2015-01-01", "2026-06-01"),
            "limits": obj({"daily": dec(100, 20000), "monthly": dec(1000, 200000)}, p="o"), "frozen": flag(0.03)},
            [IX("iban", unique=True), IX("type,currency"), IX("balance")]),
        C("transactions", "transactions", {
            "accountId": oid("accounts"), "txId": uid(4), "amount": dec(0.5, 15000), "direction": enum(["debit", "credit"]),
            "category": enum(["food", "rent", "salary", "transfer", "fees", "travel", "health"]), "bookedAt": date(),
            "counterparty": obj({"name": text("name"), "bank": text("title", p="o"), "iban": text("code", prefix="BR", p="o")}, p=0.9),
            "fraudScore": dbl(0, 1, p="o"), "reversed": flag(0.02)},
            [IX("txId", unique=True), IX("accountId,-bookedAt"), IX("category,amount"), IX("fraudScore", partial={"fraudScore": {"$gt": 0.8}})]),
        C("cards", "cards", {
            "accountId": oid("accounts"), "last4": text("code", prefix=""), "network": enum(["visa", "master", "elo"]),
            "expires": date("2026-10-01", "2031-12-31"), "blocked": flag(0.05)},
            [IX("accountId")]),
        C("ledger-entries", "ledger-entries", {
            "txId": oid("transactions"), "account": text("code", prefix="L"), "debit": dec(0, 15000), "credit": dec(0, 15000), "postedAt": date(),
            "dims": obj({"costCenter": text("code", prefix="CC", p="o"), "project": text("slug", p="o")}, p="o")},
            [IX("txId"), IX("account,postedAt")]),
    ])

    S_("travel", "roteiros", "camel", "Viagens pt-BR, camelCase", [
        C("viajantes", "viajantes", {
            "nome": text("name"), "email": text("email"), "nacionalidade": text("country"), "nascimento": date("1950-01-01", "2006-01-01"),
            "milhas": i64(0, 2000000, p="o"), "documentos": arr(obj({"tipo": enum(["passaporte", "rg", "cnh"]), "numero": text("code", prefix="D"), "validade": date("2026-10-01", "2034-12-31")}), 1, 2)},
            [IX("email", unique=True), IX("nacionalidade")]),
        C("voos", "voos", {
            "numeroVoo": text("code", prefix="KP"), "origem": text("city"), "destino": text("city"), "partida": date("2026-10-01", "2027-03-31"),
            "duracaoMin": i32(40, 900), "assentos": obj({"economica": i32(0, 300), "executiva": i32(0, 40)}), "preco": dec(120, 9000), "status": enum(["agendado", "embarque", "atrasado", "cancelado"], [70, 10, 15, 5])},
            [IX("numeroVoo,partida", unique=False), IX("origem,destino,partida"), IX("status")]),
        C("reservas", "reservas", {
            "viajanteId": oid("viajantes"), "vooId": oid("voos"), "criadaEm": date(), "situacao": enum(["confirmada", "pendente", "cancelada"], [70, 20, 10]),
            "passageiros": arr(obj({"nome": text("name"), "assento": text("code", prefix="S"), "bagagemKg": i32(0, 40, p="o")}), 1, 4),
            "total": dec(120, 30000), "extras": arr(text("word"), 0, 3, p="o")},
            [IX("viajanteId,-criadaEm"), IX("vooId,situacao")]),
        C("hoteis", "hoteis", {
            "nome": text("title"), "cidade": text("city"), "estrelas": i32(1, 5), "diaria": dec(80, 4000),
            "comodidades": arr(enum(["wifi", "piscina", "cafe", "academia", "spa", "pet"]), 0, 4),
            "avaliacoes": arr(obj({"nota": dbl(1, 5), "texto": text("sentence", p="o")}), 0, 4, p="o")},
            [IX("cidade,estrelas"), IX("comodidades")]),
    ])

    S_("workspace", "teamboard", "snake", "Gestão de tarefas / docs, snake_case", [
        C("workspaces", "workspaces", {"name": text("title"), "created_at": date(), "members": arr(obj({"user_id": oid("people"), "role": enum(["owner", "editor", "viewer"])}), 1, 6),
                                       "archived": flag(0.1)}, [IX("name"), IX("members.user_id")]),
        C("people", "people", {"name": text("name"), "email": text("email"), "avatar_url": text("url", p="o", n=0.2), "timezone": enum(["UTC", "America/Sao_Paulo", "Europe/Lisbon", "Asia/Tokyo"]),
                               "joined_at": date()}, [IX("email", unique=True)]),
        C("tasks", "tasks", {
            "workspace_id": oid("workspaces"), "title": text("title"), "assignee_id": oid("people", p="o", n=0.15),
            "state": enum(["todo", "doing", "review", "done"], [30, 25, 15, 30]), "priority": i32(1, 5), "estimate_h": dbl(0.5, 80, p="o"),
            "due_at": date("2026-09-01", "2026-12-31", p="o"), "labels": tags(), "checklist": arr(obj({"text": text("sentence"), "done": flag(0.5)}), 0, 5, p="o"),
            "parent": oid("tasks", p="o")},
            [IX("workspace_id,state,-priority"), IX("assignee_id,due_at"), IX("labels")]),
        C("documents", "documents", {
            "workspace_id": oid("workspaces"), "doc_uuid": uid(3), "title": text("title"), "revision": i32(1, 200),
            "blocks": arr(obj({"type": enum(["h1", "p", "code", "list"]), "text": text("sentence"), "style": obj({"bold": flag(0.1), "color": text("word", p="o")}, p="o")}), 1, 6),
            "updated_at": date()},
            [IX("doc_uuid", unique=True), IX("workspace_id,-updated_at")]),
        C("activity", "activity", {"workspace_id": oid("workspaces"), "who": oid("people"), "verb": enum(["created", "commented", "moved", "closed"]), "at": date(),
                                   "payload": obj({"task_id": oid("tasks", p="o"), "from_state": text("word", p="o"), "to_state": text("word", p="o")}, p="o")},
          [IX("workspace_id,-at"), IX("at", ttl=315360000)]),
    ])

    S_("healthcare-synthetic", "clinica_sint", "snake", "Clínica SINTÉTICA (dados fictícios), snake_case", [
        C("pacientes", "pacientes", {
            "codigo": text("code", prefix="PX"), "nome": text("name"), "nascimento": date("1935-01-01", "2024-12-31"),
            "sexo": enum(["F", "M", "O"], [48, 48, 4]), "convenio": enum(["particular", "plano_a", "plano_b", "sus"], [10, 30, 20, 40], p=0.9),
            "alergias": arr(text("word"), 0, 3, p="o"), "contato": obj({"telefone": text("phone"), "email": text("email", p="o")}, p=0.9)},
            [IX("codigo", unique=True), IX("convenio,sexo")]),
        C("medicos", "medicos", {"crm_sint": text("code", prefix="CRM"), "nome": text("name"),
                                 "especialidade": enum(["clinica", "cardio", "pediatria", "orto", "derma"]), "agenda_aberta": flag(0.7)},
          [IX("crm_sint", unique=True), IX("especialidade")]),
        C("consultas", "consultas", {
            "paciente_id": oid("pacientes"), "medico_id": oid("medicos"), "inicio": date("2026-01-01", "2026-12-31"), "duracao_min": i32(10, 90),
            "tipo": enum(["rotina", "retorno", "urgencia"], [60, 30, 10]), "situacao": enum(["marcada", "realizada", "faltou", "cancelada"], [20, 60, 10, 10]),
            "notas": text("sentence", p="o"), "sinais": obj({"pa_sis": i32(90, 180), "pa_dia": i32(50, 110), "temp_c": dbl(35, 40), "peso_kg": dbl(3, 140)}, p=0.6)},
            [IX("paciente_id,-inicio"), IX("medico_id,inicio"), IX("situacao")]),
        C("exames", "exames", {
            "consulta_id": oid("consultas"), "tipo": enum(["hemograma", "glicemia", "raio_x", "ecg", "ultrassom"]), "solicitado_em": date(),
            "resultado": obj({"valor": dbl(0, 500, p="o"), "unidade": text("word", p="o"), "laudo": text("sentence", p="o")}, p=0.8)},
            [IX("consulta_id"), IX("tipo,solicitado_em")]),
    ])

    S_("gaming", "arcadia", "camel", "Jogo online (jogadores, partidas, loja), camelCase", [
        C("players", "players", {
            "handle": text("slug"), "email": text("email"), "level": i32(1, 120), "xp": i64(0, 9000000), "region": enum(["sa-east", "us-east", "eu-west", "ap-south"]),
            "createdAt": date(), "banned": flag(0.03), "inventory": arr(obj({"itemId": oid("items"), "qty": i32(1, 99), "equipped": flag(0.2, p="o")}), 0, 6, p="o"),
            "stats": obj({"wins": i32(0, 3000), "losses": i32(0, 3000), "kda": dbl(0, 12), "bestStreak": i32(0, 40, p="o")}, p=0.95)},
            [IX("handle", unique=True), IX("region,level"), IX("inventory.itemId")]),
        C("matches", "matches", {
            "mode": enum(["ranked", "casual", "custom", "tournament"], [40, 40, 10, 10]), "map": text("word"), "startedAt": date("2026-06-01", "2026-09-30"),
            "durationSec": i32(120, 3600), "winnerTeam": enum(["red", "blue", "draw"], [45, 45, 10]),
            "participants": arr(obj({"playerId": oid("players"), "team": enum(["red", "blue"]), "score": i32(0, 5000), "ping": i32(5, 300, p="o")}), 2, 8)},
            [IX("startedAt"), IX("mode,startedAt"), IX("participants.playerId")]),
        C("items", "items", {
            "name": text("title"), "rarity": enum(["common", "rare", "epic", "legendary"], [60, 25, 10, 5]), "priceCoins": i32(10, 50000),
            "slot": enum(["head", "body", "weapon", "pet"]), "tradable": flag(0.7), "attrs": obj({"atk": i32(0, 500, p="o"), "def": i32(0, 500, p="o")}, p="o")},
            [IX("rarity,priceCoins"), IX("name", unique=True)]),
        C("leaderboard-season", "leaderboard-season", {
            "playerId": oid("players"), "season": i32(1, 12), "points": i64(0, 100000), "rank": i32(1, 100000)},
            [IX("season,rank"), IX("playerId,season", unique=False)]),
        C("purchases", "purchases", {
            "playerId": oid("players"), "itemId": oid("items"), "paidReal": dec(1, 200, p="o"), "paidCoins": i32(0, 50000, p="o"), "at": date()},
            [IX("playerId,-at")]),
    ])

    S_("energy", "gridwatt", "snake", "Energia / medidores, snake_case", [
        C("meters", "meters", {
            "meter_id": uid(4), "customer": text("name"), "tariff": enum(["residential", "commercial", "industrial", "rural"], [60, 20, 10, 10]),
            "installed_at": date("2012-01-01", "2026-06-01"), "location": obj({"city": text("city"), "substation": text("code", prefix="SE"), "lat": dbl(-34, 5), "lng": dbl(-74, -35)}),
            "smart": flag(0.55), "capacity_kva": dbl(1, 500, p="o")},
            [IX("meter_id", unique=True), IX("tariff,smart"), IX("location.substation")]),
        C("consumption", "consumption", {
            "meter_id": oid("meters"), "period": date("2026-01-01", "2026-09-30"), "kwh": dbl(0, 2500), "peak_kw": dbl(0, 90, p="o"),
            "cost": dec(0, 2500), "estimated": flag(0.1)},
            [IX("meter_id,-period"), IX("period,kwh")]),
        C("outages", "outages", {
            "substation": text("code", prefix="SE"), "started_at": date(), "restored_at": date(p="o", n=0.1), "cause": enum(["storm", "equipment", "planned", "unknown"], [20, 40, 30, 10]),
            "affected": i32(1, 90000), "crews": arr(obj({"crew_id": text("code", prefix="CR"), "eta_min": i32(5, 240, p="o")}), 0, 3, p="o")},
            [IX("substation,-started_at"), IX("cause")]),
        C("generation-units", "generation-units", {
            "name": text("title"), "source": enum(["hydro", "solar", "wind", "thermal"], [40, 25, 25, 10]), "mw": dbl(0.5, 900),
            "online": flag(0.85), "readings": arr(obj({"at": date("2026-09-01", "2026-09-30"), "mw": dbl(0, 900)}), 0, 5, p="o")},
            [IX("source,online"), IX("name", unique=True)]),
    ])

    # ---- holdout
    add("education", "campus", "camel", "Educação (holdout), camelCase", [
        C("students", "students", {"name": text("name"), "email": text("email"), "enrolledAt": date("2019-01-01", "2026-08-01"),
                                   "program": enum(["cs", "law", "med", "arts", "eng"]), "gpa": dbl(0, 10, p="o"), "scholarship": flag(0.15),
                                   "guardian": obj({"name": text("name"), "phone": text("phone")}, p=0.3)},
          [IX("email", unique=True), IX("program,gpa")]),
        C("courses", "courses", {"code": text("code", prefix="CRS"), "title": text("title"), "credits": i32(1, 8), "teacherId": oid("teachers"),
                                 "schedule": arr(obj({"day": enum(["mon", "tue", "wed", "thu", "fri"]), "start": text("code", prefix="T"), "room": text("code", prefix="R")}), 1, 3)},
          [IX("code", unique=True), IX("teacherId")]),
        C("teachers", "teachers", {"name": text("name"), "dept": enum(["math", "physics", "letters", "cs"]), "tenured": flag(0.5), "hiredAt": date("2000-01-01", "2026-01-01")},
          [IX("dept,tenured")]),
        C("enrollments", "enrollments", {"studentId": oid("students"), "courseId": oid("courses"), "term": enum(["2025-1", "2025-2", "2026-1", "2026-2"]),
                                         "grade": dbl(0, 10, p="o", n=0.2), "status": enum(["active", "passed", "failed", "dropped"], [30, 50, 12, 8])},
          [IX("studentId,term"), IX("courseId,status")]),
    ], holdout=True)

    add("legal", "juris", "pascal", "Jurídico (holdout), PascalCase", [
        C("Cases", "Cases", {"Number": text("code", prefix="CNJ"), "Court": enum(["TJSP", "TRF3", "STJ", "TRT2"]), "Subject": text("title"),
                             "FiledAt": date("2018-01-01", "2026-09-01"), "Status": enum(["open", "stayed", "closed"], [50, 10, 40]),
                             "Value": dec(100, 5000000, p="o"), "Parties": arr(obj({"Name": text("name"), "Role": enum(["plaintiff", "defendant", "witness"])}), 2, 5)},
          [IX("Number", unique=True), IX("Court,Status"), IX("Parties.Name")]),
        C("Lawyers", "Lawyers", {"Name": text("name"), "Bar": text("code", prefix="OAB"), "Firm": text("title", p="o"), "Active": flag(0.9)}, [IX("Bar", unique=True)]),
        C("Hearings", "Hearings", {"CaseId": oid("Cases"), "At": date("2026-01-01", "2027-03-31"), "Room": text("code", prefix="R"), "Judge": text("name"),
                                   "Notes": text("sentence", p="o")}, [IX("CaseId,At")]),
        C("Documents", "Documents", {"CaseId": oid("Cases"), "Kind": enum(["petition", "motion", "ruling", "evidence"]), "Pages": i32(1, 400),
                                     "Hash": text("hex"), "Filed": date("2018-01-01", "2026-09-01"), "Confidential": flag(0.1)},
          [IX("CaseId,Kind"), IX("Hash", unique=True)]),
    ], holdout=True)
    return S


# ======================================================== procedural ======
# Biblioteca de entidades por domínio -> schemas adicionais (6–10 por domínio no total).
# Mini-DSL de campos: nome=tipo; separados por ';' (',' dentro de o(...)/a(...)).
#  str: name email city country title sentence word phone url ip slug hex sku | code:PFX
#  e:a|b|c enum | i:lo..hi int32 | l:lo..hi int64 | d:lo..hi double | m:lo..hi decimal128
#  b[:pt] bool | t[:lo..hi] date | u3/u4 uuid | ref:chave | o(...) objeto | a(...) array de subdocs | as:tipo array escalar
#  sufixo '?' = opcional (30-95%), '??' = opcional + null 10%
# Índices: "a,-b" composto | "!a" único | "ttl:campo" (10 anos) | "~a,b@flag" parcial | "text:campo"
TR = dict(customer="cliente", customers="clientes", name="nome", phone="telefone", address="endereco", street="rua",
          city="cidade", country="pais", state="estado", zip="cep", created="criado", updated="atualizado", at="em",
          date="data", status="situacao", price="preco", stock="estoque", quantity="quantidade", qty="qtd",
          amount="valor", currency="moeda", category="categoria", description="descricao", title="titulo",
          order="pedido", product="produto", items="itens", line="linha", lines="linhas", tags="etiquetas",
          active="ativo", owner="dono", user="usuario", plan="plano", role="papel", code="codigo", type="tipo",
          kind="tipo", number="numero", notes="observacoes", note="observacao", start="inicio", end="fim",
          started="iniciado", ended="encerrado", due="vencimento", paid="pago", balance="saldo",
          account="conta", card="cartao", rating="nota", score="pontuacao", level="nivel", size="tamanho",
          weight="peso", color="cor", brand="marca", supplier="fornecedor", employee="funcionario", salary="salario",
          department="departamento", manager="gerente", hired="admitido", birth="nascimento", gender="genero",
          patient="paciente", doctor="medico", visit="consulta", diagnosis="diagnostico", device="dispositivo",
          reading="leitura", value="valor", unit="unidade", site="local", location="localizacao", route="rota",
          driver="motorista", vehicle="veiculo", plate="placa", origin="origem", destination="destino",
          delivered="entregue", sent="enviado", content="conteudo", author="autor", comment="comentario",
          published="publicado", views="visualizacoes", language="idioma", player="jogador", match="partida",
          team="equipe", member="membro", project="projeto", task="tarefa", priority="prioridade", assignee="responsavel",
          student="aluno", course="curso", teacher="professor", grade="nota", term="periodo", case="processo",
          court="vara", lawyer="advogado", party="parte", hearing="audiencia", client="cliente", fee="honorario",
          meter="medidor", consumption="consumo", tariff="tarifa", outage="queda", cause="causa", flight="voo",
          booking="reserva", guest="hospede", nights="diarias",
          settings="configuracoes", config="configuracao", metadata="metadados",
          count="contagem", limit="limite", enabled="habilitado", verified="verificado", deleted="excluido",
          first="primeiro", last="ultimo", login="acesso", seen="visto", expires="expira", discount="desconto",
          tax="imposto", shipping="frete", payment="pagamento", method="metodo", invoice="fatura", receipt="recibo",
          bank="banco", branch="agencia", rate="taxa", risk="risco", position="posicao")
DS = {"commerce": "com", "saas": "saas", "logistics": "log", "iot": "iot", "hr": "hr", "content": "cnt", "finance": "fin",
      "travel": "trv", "workspace": "wks", "healthcare-synthetic": "hlt", "gaming": "gam", "energy": "nrg",
      "education": "edu", "legal": "leg"}
BRANDS = ("atlas borealis cobalt delta ember fjord granite harbor indigo juniper kestrel lumen meridian nimbus onyx prism "
          "quasar rivet saffron tundra umbra vertex willow xylem yonder zenith aurora beacon cinder drift").split()
ST = ["camel", "pascal", "snake"]


def EE(*a):
    return a  # (key, pt_key, fields, indexes)


ENT = {
 "commerce": [
  EE("customers", "clientes", "full_name=name;email=email;phone=phone?;birth_date=t:1950-01-01..2005-12-31?;loyalty=e:bronze|silver|gold|platinum?;address=o(street=title,city=city,country=country,geo=o(lat=d:-34..5,lng=d:-74..-35)?)?;created_at=t;marketing_opt_in=b:0.4", "!email;address.country,loyalty;-created_at"),
  EE("products", "produtos", "sku=sku;title=title;category=e:audio|video|home|toys|books|garden|sport;price=m:5..900;stock=i:0..500;active=b:0.8;attributes=a(k=word,v=word)?;dimensions=o(w=d:1..90,h=d:1..90,d=d:1..90,unit=e:cm|in)?;supplier_id=ref:suppliers?;tags=as:word?", "!sku;category,-price;tags;~stock@active;text:title"),
  EE("orders", "pedidos", "customer_id=ref:customers;status=e:pending|paid|shipped|delivered|cancelled;total=m:10..3000;currency=e:BRL|USD|EUR;placed_at=t;lines=a(product_id=ref:products,qty=i:1..6,unit_price=m:3..900);payment=o(method=e:pix|card|boleto,installments=i:1..12?,approved=b:0.9,gateway=o(provider=word,tx_ref=u4,raw=o(code=i:0..99,msg=sentence?)?)?)?", "customer_id,-placed_at;status;lines.product_id;payment.method,payment.approved"),
  EE("reviews", "avaliacoes", "product_id=ref:products;customer_id=ref:customers;stars=i:1..5;body=sentence??;created_at=t;helpful_votes=l:0..900?", "product_id,stars;customer_id"),
  EE("carts", "carrinhos", "customer_id=ref:customers?;session=u4;items=a(product_id=ref:products,qty=i:1..4);updated_at=t;coupon=code:CP?", "session;ttl:updated_at;customer_id"),
  EE("coupons", "cupons", "code=code:CP;discount_pct=i:5..70;valid_from=t;valid_until=t;max_uses=i:1..5000;used=i:0..500;channels=as:e:web|app|store?", "!code;valid_until"),
  EE("shipments", "entregas", "order_id=ref:orders;carrier=e:correios|jadlog|fedex|dhl;tracking=code:TRK;status=e:created|in_transit|delivered|returned;events=a(at=t,place=city,note=sentence?);eta=t?", "!tracking;order_id;status,carrier"),
  EE("suppliers", "fornecedores", "legal_name=title;tax_id=code:TX;country=country;rating=d:1..5?;contact=o(name=name,email=email,phone=phone?);active=b:0.9", "!tax_id;country,active"),
  EE("invoices", "notas_fiscais", "order_id=ref:orders;number=code:NF;issued_at=t;total=m:10..3000;tax=m:0..600;xml_hash=hex;canceled=b:0.04", "!number;order_id;issued_at"),
  EE("returns", "devolucoes", "order_id=ref:orders;reason=e:defect|wrong_item|regret|late;requested_at=t;refund=m:5..2000?;approved=b:0.7", "order_id;reason,approved"),
  EE("wishlists", "listas_desejo", "customer_id=ref:customers;name=title;public=b:0.2;products=as:ref:products?;created_at=t", "customer_id;public"),
  EE("categories", "categorias", "name=title;slug=slug;parent=ref:categories?;position=i:0..40;visible=b:0.9", "!slug;parent,position"),
 ],
 "saas": [
  EE("tenants", "inquilinos", "name=title;plan=e:free|team|business|enterprise;seats=i:1..500;mrr=m:0..20000;created_at=t;settings=o(sso=b:0.2,locale=e:en|pt-BR|es,limits=o(api_rpm=i:60..6000,storage_gb=l:1..5000,burst=o(window_s=i:1..60,cap=i:10..900)))?", "!name;plan,-mrr"),
  EE("users", "usuarios", "tenant_id=ref:tenants;email=email;role=e:owner|admin|member|viewer;display_name=name;mfa=b:0.35;last_seen_at=t??;prefs=o(theme=e:dark|light,tz=word,notify=as:word)?", "tenant_id,email;role;-last_seen_at"),
  EE("subscriptions", "assinaturas", "tenant_id=ref:tenants;plan=e:free|team|business|enterprise;status=e:active|past_due|canceled|trialing;started_at=t;renews_at=t?;price=m:0..900;addons=a(code=code:AD,qty=i:1..20)?", "tenant_id,status;renews_at"),
  EE("invoices", "faturas", "tenant_id=ref:tenants;number=code:INV;period=t;amount=m:0..20000;paid=b:0.85;line_items=a(desc=sentence,amount=m:1..5000,tax=o(rate=d:0..0.3,jurisdiction=country)?)", "!number;tenant_id,-period;~period@paid"),
  EE("api_keys", "chaves_api", "tenant_id=ref:tenants;key_id=u4;prefix=hex;scopes=as:e:read|write|admin;created_at=t;revoked=b:0.1;last_used=t??", "!key_id;tenant_id;scopes"),
  EE("audit_events", "eventos_auditoria", "tenant_id=ref:tenants;actor=ref:users?;action=e:login|create|update|delete|export|invite;target=o(kind=word,id=hex);at=t;ip=ip;details=o(before=o(v=word?)?,after=o(v=word?)?)?", "tenant_id,-at;action;ttl:at"),
  EE("feature_flags", "flags_recurso", "key=slug;description=sentence;enabled=b:0.5;rollout_pct=i:0..100;rules=a(attr=word,op=e:eq|in|gt,value=word)?;updated_at=t", "!key;enabled"),
  EE("webhooks", "webhooks", "tenant_id=ref:tenants;url=url;events=as:e:invoice.paid|user.created|plan.changed;signature_salt=code:SALT-;active=b:0.9;failures=i:0..30", "tenant_id,active;events"),
  EE("usage_records", "registros_uso", "tenant_id=ref:tenants;metric=e:api_calls|storage|seats|jobs;day=t;value=l:0..900000;cost=m:0..400?", "tenant_id,metric,-day"),
  EE("support_tickets", "chamados", "tenant_id=ref:tenants;opened_by=ref:users;subject=title;priority=e:low|normal|high|urgent;status=e:open|pending|solved;messages=a(at=t,from=e:agent|customer,body=sentence);csat=i:1..5??", "tenant_id,status;priority,-opened_by"),
 ],
 "logistics": [
  EE("warehouses", "armazens", "code=code:WH;name=title;city=city;capacity_m3=i:500..90000;docks=i:1..60;geo=o(lat=d:-34..5,lng=d:-74..-35);open=b:0.9", "!code;city,open"),
  EE("shipments", "remessas", "tracking=code:SHP;origin=ref:warehouses;destination=o(city=city,country=country,zip=code:Z);status=e:booked|picked|in_transit|delivered|exception;weight_kg=d:0.1..900;created_at=t;carrier_id=ref:carriers?;legs=a(from=city,to=city,mode=e:road|air|sea|rail,eta=t)", "!tracking;origin,status;legs.mode"),
  EE("vehicles", "veiculos", "plate=code:PL;type=e:van|truck|bike|ship;capacity_kg=i:50..30000;odometer=l:0..900000;gps=o(lat=d:-34..5,lng=d:-74..-35,at=t)?;in_service=b:0.85;vin=u4", "!plate;type,in_service"),
  EE("drivers", "motoristas", "name=name;license=code:CNH;phone=phone;vehicle_id=ref:vehicles?;rating=d:1..5?;active=b:0.9;hired_at=t", "!license;active"),
  EE("routes", "rotas", "code=code:RT;driver_id=ref:drivers;vehicle_id=ref:vehicles;date=t;stops=a(seq=i:1..40,address=title,window=o(from=t,to=t),done=b:0.6);km=d:2..900", "!code;driver_id,-date;stops.done"),
  EE("packages", "pacotes", "shipment_id=ref:shipments;sku=sku;dims=o(l=d:1..90,w=d:1..90,h=d:1..90);fragile=b:0.12;declared_value=m:5..9000;barcode=hex", "shipment_id;!barcode;fragile"),
  EE("carriers", "transportadoras", "name=title;scac=code:SC;services=as:e:express|economy|freight;sla_hours=i:12..240;contract=o(start=t,end=t,rate=m:1..80)?", "!scac;services"),
  EE("tracking_events", "eventos_rastreio", "shipment_id=ref:shipments;at=t;kind=e:scan|load|unload|delay|deliver;place=city;details=o(operator=name?,device=u3?)?", "shipment_id,-at;ttl:at;kind"),
  EE("manifests", "manifestos", "number=code:MF;route_id=ref:routes;issued_at=t;shipments=as:ref:shipments;signed=b:0.8;total_weight=d:10..20000", "!number;route_id"),
  EE("stock_moves", "movimentos_estoque", "warehouse_id=ref:warehouses;sku=sku;delta=i:-200..200;reason=e:inbound|outbound|adjust|transfer;at=t;batch=u3?", "warehouse_id,sku,-at;reason"),
 ],
 "iot": [
  EE("sites", "locais", "name=title;city=city;geo=o(lat=d:-34..5,lng=d:-74..-35);timezone=word;devices_expected=i:1..300", "!name;city"),
  EE("devices", "dispositivos", "serial=hex;model=e:th-100|pm-200|gw-300|cam-9;site_id=ref:sites;firmware=code:FW;online=b:0.85;installed_at=t;battery_pct=i:1..100?;uuid=u4;labels=o(env=e:prod|test,zone=word,owner=word?)?", "!serial;site_id,online;model;firmware"),
  EE("sensors", "sensores", "device_id=ref:devices;kind=e:temperature|humidity|pressure|vibration|co2;unit=e:C|%|hPa|mm_s|ppm;calibrated_at=t?;range=o(min=d:-50..0,max=d:10..200)", "device_id,kind"),
  EE("readings", "leituras", "sensor_id=ref:sensors;at=t;value=d:-20..120;quality=i:0..100;raw=l:0..65535?;flags=as:e:spike|stale|interpolated?", "sensor_id,-at;ttl:at"),
  EE("alerts", "alertas", "device_id=ref:devices;rule=slug;severity=e:info|warn|crit;opened_at=t;closed_at=t??;ack=o(by=name,at=t)?;context=o(value=d:0..200,threshold=d:0..200,window=o(from=t,to=t))?", "device_id,severity;-opened_at"),
  EE("firmware", "firmwares", "version=code:FW;model=e:th-100|pm-200|gw-300|cam-9;released_at=t;size_kb=i:64..9000;checksum=hex;notes=sentence?;mandatory=b:0.15", "!version;model,-released_at"),
  EE("gateways", "gateways", "mac=hex;site_id=ref:sites;ip=ip;uplink=e:lte|wifi|eth;connected=as:ref:devices?;uptime_s=l:0..9000000", "!mac;site_id;uplink"),
  EE("commands", "comandos", "device_id=ref:devices;name=e:reboot|calibrate|update|sample_rate;params=o(rate_hz=i:1..100?,force=b:0.2?);issued_at=t;status=e:queued|sent|ack|failed", "device_id,-issued_at;status"),
  EE("maintenance", "manutencoes", "device_id=ref:devices;technician=name;performed_at=t;cost=m:10..3000;parts=a(code=code:PT,qty=i:1..5);notes=sentence?", "device_id,-performed_at"),
  EE("telemetry_batches", "lotes_telemetria", "gateway_id=ref:gateways;received_at=t;count=i:1..500;points=a(device=hex,t=t,v=d:0..100);compressed=b:0.5", "gateway_id,-received_at;ttl:received_at"),
 ],
 "hr": [
  EE("departments", "departamentos", "name=title;code=code:DP;budget=m:10000..2000000;parent=ref:departments?;cost_center=code:CC", "!code;parent"),
  EE("employees", "funcionarios", "full_name=name;email=email;department_id=ref:departments;position=title;salary=m:1500..45000;hired_at=t;manager_id=ref:employees?;status=e:active|leave|terminated;address=o(city=city,country=country,geo=o(lat=d:-34..5,lng=d:-74..-35)?)?;skills=as:word?;uuid=u3", "!email;department_id,status;-hired_at;skills"),
  EE("positions", "cargos", "title=title;level=e:junior|mid|senior|lead|director;band=o(min=m:1500..8000,max=m:8000..60000);open_headcount=i:0..12", "!title;level"),
  EE("payroll", "folha", "employee_id=ref:employees;month=t;gross=m:1500..60000;net=m:1200..48000;deductions=a(kind=e:inss|irrf|health|union,amount=m:10..5000);paid_at=t??", "employee_id,-month;paid_at"),
  EE("leave_requests", "pedidos_ferias", "employee_id=ref:employees;kind=e:vacation|sick|parental|unpaid;from=t;to=t;status=e:pending|approved|rejected;approved_by=ref:employees?", "employee_id,-from;status"),
  EE("candidates", "candidatos", "name=name;email=email;position=title;stage=e:applied|screen|interview|offer|hired|rejected;source=e:referral|site|agency|social;resume_url=url?;score=d:0..10??;applied_at=t", "!email;stage,source;position"),
  EE("interviews", "entrevistas", "candidate_id=ref:candidates;interviewer=ref:employees;at=t;kind=e:phone|technical|culture;feedback=o(score=i:1..5,notes=sentence?,recommend=b:0.5)?", "candidate_id,-at;interviewer"),
  EE("timesheets", "folhas_ponto", "employee_id=ref:employees;date=t;entries=a(project=code:PR,hours=d:0.5..9);approved=b:0.8", "employee_id,-date;~date@approved"),
  EE("benefits", "beneficios", "employee_id=ref:employees;kind=e:health|dental|meal|transport|gym;monthly_cost=m:20..900;since=t;dependents=a(name=name,relation=e:spouse|child|parent)?", "employee_id,kind"),
  EE("performance_reviews", "avaliacoes_desempenho", "employee_id=ref:employees;cycle=e:2025-1|2025-2|2026-1;rating=i:1..5;reviewer=ref:employees;goals=a(title=title,progress=i:0..100,weight=d:0..1);comments=sentence?", "employee_id,cycle;rating"),
 ],
 "content": [
  EE("authors", "autores", "name=name;email=email;bio=sentence?;social=o(twitter=slug?,site=url?)?;joined_at=t;verified=b:0.4", "!email;verified"),
  EE("articles", "artigos", "title=title;slug=slug;author_id=ref:authors;body=sentence;status=e:draft|review|published|archived;published_at=t??;tags=as:word;views=l:0..900000;seo=o(title=title?,keywords=as:word?,og=o(image=url?,type=word?)?)?;lang=e:pt|en|es", "!slug;author_id,-published_at;tags;status,lang;text:title"),
  EE("comments", "comentarios", "article_id=ref:articles;user=name;body=sentence;at=t;likes=i:0..300;flagged=b:0.03;replies=a(user=name,body=sentence,at=t)?", "article_id,-at;~article_id@flagged"),
  EE("media", "midias", "kind=e:image|video|audio|pdf;url=url;bytes=l:1000..90000000;width=i:100..4000?;height=i:100..4000?;uploaded_by=ref:authors;checksum=hex;uuid=u4", "!checksum;kind;uploaded_by"),
  EE("categories", "categorias", "name=title;slug=slug;parent=ref:categories?;sort=i:0..50", "!slug;parent"),
  EE("subscribers", "assinantes", "email=email;plan=e:free|plus|pro;since=t;topics=as:word;confirmed=b:0.8;last_open=t??", "!email;plan,confirmed;topics"),
  EE("newsletters", "newsletters", "subject=title;sent_at=t;audience=i:10..90000;open_rate=d:0..1;click_rate=d:0..0.4;articles=as:ref:articles", "-sent_at;articles"),
  EE("revisions", "revisoes", "article_id=ref:articles;editor=ref:authors;at=t;diff=o(added=i:0..900,removed=i:0..900);note=sentence?", "article_id,-at;ttl:at"),
  EE("pages", "paginas", "path=slug;title=title;layout=e:home|landing|doc;blocks=a(type=e:hero|text|gallery|cta,data=o(text=sentence?,src=url?));published=b:0.8;updated_at=t", "!path;published"),
 ],
 "finance": [
  EE("customers", "clientes", "name=name;tax_id=code:TX;segment=e:retail|premium|corporate;onboarded_at=t;kyc=o(status=e:ok|pending|review,checked_at=t,docs=as:word)?;risk_score=i:0..100;uuid=u4", "!tax_id;segment,risk_score"),
  EE("accounts", "contas", "customer_id=ref:customers;number=code:AC;type=e:checking|savings|investment;balance=m:-500..90000;currency=e:BRL|USD|EUR;opened_at=t;closed=b:0.05;limits=o(daily=m:100..9000,overdraft=m:0..5000)?", "!number;customer_id,type;~balance@closed"),
  EE("transactions", "transacoes", "account_id=ref:accounts;kind=e:debit|credit|pix|transfer|fee;amount=m:0.5..9000;at=t;counterparty=o(name=name,bank=code:BK,account=code:AC)?;status=e:posted|pending|reversed;memo=sentence?;ref=u4", "account_id,-at;kind,status;ttl:at"),
  EE("cards", "cartoes", "account_id=ref:accounts;last4=code:;brand=e:visa|master|elo|amex;limit=m:300..30000;status=e:active|blocked|expired;expires=t;contactless=b:0.8", "account_id;brand,status"),
  EE("loans", "emprestimos", "customer_id=ref:customers;principal=m:500..200000;rate_pct=d:0.5..9;term_months=i:6..120;status=e:active|paid|default;schedule=a(n=i:1..120,due=t,amount=m:50..9000,paid=b:0.5)", "customer_id,status;schedule.paid"),
  EE("payments", "pagamentos", "account_id=ref:accounts;method=e:boleto|pix|card;amount=m:1..5000;due=t;paid_at=t??;barcode=hex", "!barcode;account_id,-due"),
  EE("ledger_entries", "lancamentos", "journal=code:J;account=code:AC;debit=m:0..9000;credit=m:0..9000;at=t;batch=l:1..99999", "journal,-at;account"),
  EE("budgets", "orcamentos", "owner=ref:customers;category=e:food|rent|travel|health|fun;month=t;limit=m:50..9000;spent=m:0..9000", "owner,month;category"),
  EE("fx_rates", "cotacoes", "pair=e:USD/BRL|EUR/BRL|GBP/USD|JPY/USD;at=t;bid=d:0.5..6;ask=d:0.5..6;source=word", "pair,-at;ttl:at"),
  EE("positions", "posicoes", "account_id=ref:accounts;ticker=code:TK;qty=l:1..100000;avg_price=m:1..900;opened_at=t;tags=as:word?", "account_id,ticker;ticker"),
 ],
 "travel": [
  EE("travelers", "viajantes", "name=name;email=email;passport=code:PP;nationality=country;birth=t:1950-01-01..2008-01-01;loyalty=o(tier=e:blue|silver|gold,points=l:0..900000)?;uuid=u4", "!email;nationality;loyalty.tier"),
  EE("destinations", "destinos", "city=city;country=country;climate=e:tropical|dry|temperate|cold;best_months=as:i:1..12;popularity=i:1..100;geo=o(lat=d:-60..60,lng=d:-180..180)", "city,country;popularity"),
  EE("hotels", "hoteis", "name=title;city=city;stars=i:1..5;amenities=as:e:pool|wifi|gym|spa|parking;rooms=a(type=e:single|double|suite,price=m:50..2500,available=i:0..40);rating=d:1..10?;chain=word?", "city,stars;amenities;!name"),
  EE("flights", "voos", "number=code:FL;origin=code:AP;dest=code:AP;departs=t;arrives=t;aircraft=e:A320|B737|E195|A350;seats=o(economy=i:0..250,business=i:0..40);status=e:scheduled|delayed|cancelled|landed", "!number;origin,dest,departs;status"),
  EE("bookings", "reservas", "traveler_id=ref:travelers;kind=e:hotel|flight|package|car;ref_id=u3;total=m:50..30000;status=e:held|confirmed|cancelled|completed;booked_at=t;passengers=a(name=name,doc=code:DC,seat=code:S?)?", "traveler_id,-booked_at;status,kind"),
  EE("itineraries", "roteiros", "traveler_id=ref:travelers;title=title;days=a(day=i:1..21,city=city,activities=as:word);shared=b:0.2", "traveler_id;days.city"),
  EE("reviews", "avaliacoes", "traveler_id=ref:travelers;hotel_id=ref:hotels;score=i:1..10;text=sentence??;stayed=t;photos=as:url?", "hotel_id,score;traveler_id"),
  EE("loyalty_events", "eventos_fidelidade", "traveler_id=ref:travelers;kind=e:earn|burn|expire|bonus;points=l:1..90000;at=t", "traveler_id,-at;ttl:at"),
  EE("tours", "passeios", "title=title;destination_id=ref:destinations;price=m:20..900;duration_h=d:0.5..12;guide=name?;slots=a(date=t,left=i:0..30)", "destination_id;slots.date"),
  EE("cars", "carros", "plate=code:PL;category=e:eco|suv|luxury|van;daily_rate=m:40..600;city=city;available=b:0.7;mileage=l:0..200000", "!plate;city,available;category"),
 ],
 "workspace": [
  EE("members", "membros", "name=name;email=email;role=e:admin|editor|viewer|guest;joined_at=t;avatar=url?;timezone=word?;uuid=u4", "!email;role"),
  EE("projects", "projetos", "key=code:PRJ;name=title;owner=ref:members;status=e:planned|active|paused|done;start=t;end=t??;budget=m:0..900000?;labels=as:word?;settings=o(visibility=e:private|team|public,integrations=a(kind=word,enabled=b:0.6,config=o(url=url?,token=hex?)))?", "!key;owner,status;labels"),
  EE("tasks", "tarefas", "project_id=ref:projects;title=title;assignee=ref:members??;priority=e:low|medium|high|urgent;state=e:todo|doing|review|done;estimate_h=d:0.5..40?;due=t??;subtasks=a(title=title,done=b:0.5);created_at=t;checklist=a(item=sentence,done=b:0.4)?", "project_id,state;assignee,due;priority"),
  EE("boards", "quadros", "project_id=ref:projects;name=title;columns=a(name=word,limit=i:0..20,cards=as:ref:tasks)", "project_id"),
  EE("documents", "documentos", "title=title;author=ref:members;content=sentence;version=i:1..60;shared_with=as:ref:members?;updated_at=t;history=a(v=i:1..60,at=t,by=ref:members)?", "author,-updated_at;text:title;shared_with"),
  EE("comments", "comentarios", "task_id=ref:tasks;author=ref:members;text=sentence;at=t;reactions=a(emoji=e:+1|heart|eyes,by=ref:members)?;edited=b:0.1", "task_id,at"),
  EE("sprints", "sprints", "project_id=ref:projects;name=code:SP;start=t;end=t;goal=sentence?;velocity=i:5..80?;tasks=as:ref:tasks", "project_id,-start"),
  EE("files", "arquivos", "project_id=ref:projects;name=slug;mime=e:image/png|application/pdf|text/plain|video/mp4;bytes=l:100..900000000;uploaded_by=ref:members;sha=hex;at=t", "project_id;!sha;uploaded_by,-at"),
  EE("notifications", "notificacoes", "member_id=ref:members;kind=e:mention|assign|due|comment;read=b:0.5;at=t;payload=o(task=ref:tasks?,by=ref:members?,preview=sentence?)", "member_id,read,-at;ttl:at"),
  EE("calendar_events", "eventos_agenda", "title=title;organizer=ref:members;start=t;end=t;attendees=a(member=ref:members,status=e:yes|no|maybe);recurrence=o(freq=e:daily|weekly|monthly,until=t?)?;room=code:R?", "organizer,start;attendees.member"),
 ],
 "healthcare-synthetic": [
  EE("patients", "pacientes", "mrn=code:MRN;name=name;birth=t:1930-01-01..2024-01-01;sex=e:F|M|X;blood=e:A+|A-|B+|O+|O-|AB+;address=o(city=city,state=e:SP|RJ|MG|BA|RS,zip=code:Z)?;insurance=o(plan=word,member_id=u4)?;consent=b:0.95;allergies=as:word?", "!mrn;sex,blood;address.city"),
  EE("practitioners", "profissionais", "name=name;license=code:CRM;specialty=e:cardio|derm|ortho|peds|neuro|gp;active=b:0.9;facility_id=ref:facilities", "!license;specialty,active;facility_id"),
  EE("facilities", "unidades", "name=title;kind=e:hospital|clinic|lab|pharmacy;city=city;beds=i:0..900?;accredited=b:0.6", "!name;kind,city"),
  EE("visits", "atendimentos", "patient_id=ref:patients;practitioner_id=ref:practitioners;at=t;kind=e:outpatient|emergency|inpatient|telehealth;reason=sentence;vitals=o(bp=o(sys=i:80..190,dia=i:50..120),hr=i:40..160,temp=d:35..41,spo2=i:80..100)?;diagnoses=a(code=code:ICD,desc=title);discharged=t??", "patient_id,-at;practitioner_id;kind"),
  EE("prescriptions", "prescricoes", "visit_id=ref:visits;patient_id=ref:patients;drug=word;dose_mg=d:1..1000;frequency=e:qd|bid|tid|prn;days=i:1..90;refills=i:0..5;issued_at=t", "patient_id,-issued_at;drug"),
  EE("lab_results", "exames", "patient_id=ref:patients;test=e:cbc|glucose|lipid|tsh|creatinine;value=d:0..300;unit=word;ref_range=o(low=d:0..100,high=d:100..300);flag=e:normal|high|low;at=t", "patient_id,test,-at;flag"),
  EE("appointments", "consultas", "patient_id=ref:patients;practitioner_id=ref:practitioners;slot=t;status=e:booked|arrived|done|noshow|cancelled;channel=e:phone|app|walkin", "practitioner_id,slot;status;patient_id"),
  EE("claims", "sinistros", "patient_id=ref:patients;visit_id=ref:visits?;amount=m:20..90000;status=e:submitted|paid|denied|appeal;submitted_at=t;lines=a(code=code:CPT,units=i:1..5,amount=m:5..9000)", "patient_id,status;-submitted_at"),
  EE("immunizations", "vacinas", "patient_id=ref:patients;vaccine=e:flu|covid|hepb|mmr|tdap;dose=i:1..4;at=t;lot=hex;site=e:arm|thigh", "patient_id,vaccine;lot"),
  EE("care_plans", "planos_cuidado", "patient_id=ref:patients;goal=sentence;status=e:active|completed|stopped;activities=a(kind=word,due=t?,done=b:0.4);period=o(start=t,end=t??)", "patient_id,status"),
 ],
 "gaming": [
  EE("players", "jogadores", "handle=slug;email=email;level=i:1..120;xp=l:0..9000000;region=e:sa|na|eu|ap;premium=b:0.15;created_at=t;banned=o(reason=sentence,until=t)?;uuid=u4;friends=as:ref:players?", "!handle;region,level;premium"),
  EE("matches", "partidas", "mode=e:ranked|casual|custom;map=word;started_at=t;duration_s=i:60..3600;players=a(player_id=ref:players,team=e:red|blue,kills=i:0..40,deaths=i:0..40,score=i:0..9999);winner=e:red|blue|draw", "mode,-started_at;players.player_id;map"),
  EE("items", "itens", "sku=sku;name=title;rarity=e:common|rare|epic|legendary;slot=e:head|body|weapon|pet;price_coins=i:10..90000;stats=o(atk=i:0..300?,def=i:0..300?,spd=i:0..100?);tradable=b:0.7", "!sku;rarity,slot;price_coins"),
  EE("inventories", "inventarios", "player_id=ref:players;slots=a(item_id=ref:items,qty=i:1..99,equipped=b:0.2,meta=o(enchant=i:0..10?,bound=b:0.3?)?);gold=l:0..90000000;capacity=i:20..300", "player_id;slots.item_id"),
  EE("achievements", "conquistas", "key=slug;title=title;points=i:5..100;hidden=b:0.2;unlocked_by=as:ref:players?", "!key;hidden"),
  EE("guilds", "guildas", "name=title;tag=code:G;leader=ref:players;members=as:ref:players;level=i:1..30;treasury=l:0..9000000;founded=t", "!name;level;members"),
  EE("leaderboards", "rankings", "season=code:S;mode=e:ranked|casual;entries=a(player_id=ref:players,rank=i:1..1000,rating=i:800..3500);closed=b:0.5", "season,mode;entries.player_id"),
  EE("purchases", "compras", "player_id=ref:players;item_id=ref:items;paid_real=m:1..200??;paid_coins=i:0..50000?;at=t;platform=e:ios|android|pc|console;receipt=u4", "player_id,-at;platform"),
  EE("sessions", "sessoes", "player_id=ref:players;started_at=t;ended_at=t??;device=o(os=e:ios|android|win|mac,model=word,ver=code:V);ip=ip;events=a(t=t,kind=e:login|spawn|death|buy|quit)?", "player_id,-started_at;ttl:started_at"),
  EE("quests", "missoes", "title=title;min_level=i:1..100;rewards=o(xp=i:10..9000,items=as:ref:items?,gold=i:0..9000?);steps=a(order=i:1..8,text=sentence,count=i:1..30);repeatable=b:0.3", "min_level;repeatable"),
 ],
 "energy": [
  EE("substations", "subestacoes", "code=code:SE;name=title;voltage_kv=e:13.8|69|138|230|500;city=city;geo=o(lat=d:-34..5,lng=d:-74..-35);online=b:0.95", "!code;voltage_kv,online"),
  EE("meters", "medidores", "meter_id=u4;customer=name;tariff=e:residential|commercial|industrial|rural;installed_at=t:2012-01-01..2026-06-01;substation_id=ref:substations;smart=b:0.55;capacity_kva=d:1..500?;legacy_id=u3?", "!meter_id;tariff,smart;substation_id"),
  EE("consumption", "consumo", "meter_id=ref:meters;period=t;kwh=d:0..2500;peak_kw=d:0..90?;cost=m:0..2500;estimated=b:0.1", "meter_id,-period;period,kwh"),
  EE("outages", "quedas", "substation_id=ref:substations;started_at=t;restored_at=t??;cause=e:storm|equipment|planned|unknown;affected=i:1..90000;crews=a(crew=code:CR,eta_min=i:5..240?)?", "substation_id,-started_at;cause"),
  EE("generation_units", "unidades_geracao", "name=title;source=e:hydro|solar|wind|thermal;mw=d:0.5..900;online=b:0.85;readings=a(at=t,mw=d:0..900)?", "source,online;!name"),
  EE("tariffs", "tarifas", "code=code:TF;class=e:residential|commercial|industrial;bands=a(name=e:off|mid|peak,from_h=i:0..23,to_h=i:0..23,price=m:0.1..3);valid_from=t;valid_until=t??", "!code;class,valid_from"),
  EE("contracts", "contratos", "customer=name;meter_id=ref:meters;tariff_id=ref:tariffs;signed_at=t;term_months=i:6..60;green=b:0.2;terms=o(penalty=m:0..9000,notice_days=i:15..90)", "meter_id;tariff_id;green"),
  EE("maintenance_orders", "ordens_manutencao", "asset=ref:generation_units;kind=e:preventive|corrective|inspection;opened_at=t;closed_at=t??;cost=m:100..90000?;crew=as:name;tasks=a(desc=sentence,done=b:0.6)", "asset,-opened_at;kind"),
  EE("grid_events", "eventos_rede", "substation_id=ref:substations;at=t;kind=e:overload|fault|switch|reclose;severity=i:1..5;details=o(phase=e:A|B|C,current_a=d:0..9000?)?", "substation_id,-at;ttl:at;kind"),
  EE("billing_cycles", "ciclos_faturamento", "label=code:CY;from=t;to=t;total_kwh=l:0..900000000;revenue=m:0..90000000;closed=b:0.7", "!label;closed"),
 ],
 "education": [
  EE("students", "alunos", "name=name;email=email;enrolled_at=t:2019-01-01..2026-08-01;program=e:cs|law|med|arts|eng;gpa=d:0..10?;scholarship=b:0.15;guardian=o(name=name,phone=phone,contact=o(email=email?,addr=o(city=city,state=word)?)?)?;uuid=u4", "!email;program,gpa;scholarship"),
  EE("teachers", "professores", "name=name;dept=e:math|physics|letters|cs|bio;tenured=b:0.5;hired_at=t:2000-01-01..2026-01-01;office=code:R?;research=as:word?", "dept,tenured"),
  EE("courses", "cursos", "code=code:CRS;title=title;credits=i:1..8;teacher_id=ref:teachers;schedule=a(day=e:mon|tue|wed|thu|fri,start=code:T,room=code:R);syllabus=o(topics=as:word,books=as:title?)?", "!code;teacher_id;schedule.day"),
  EE("enrollments", "matriculas", "student_id=ref:students;course_id=ref:courses;term=e:2025-1|2025-2|2026-1|2026-2;grade=d:0..10??;status=e:active|passed|failed|dropped;attendance_pct=i:0..100?", "student_id,term;course_id,status"),
  EE("assignments", "trabalhos", "course_id=ref:courses;title=title;due=t;max_points=i:5..100;kind=e:essay|quiz|project|lab;rubric=a(criterion=word,weight=d:0..1)?", "course_id,-due;kind"),
  EE("submissions", "entregas", "assignment_id=ref:assignments;student_id=ref:students;at=t;points=d:0..100??;late=b:0.15;files=as:url?;feedback=sentence?", "assignment_id,student_id;late"),
  EE("classrooms", "salas", "code=code:R;building=word;capacity=i:10..300;equipment=as:e:projector|lab|ac|mic;floor=i:0..9", "!code;building,floor"),
  EE("exams", "provas", "course_id=ref:courses;date=t;room_id=ref:classrooms;duration_min=i:30..240;questions=a(n=i:1..60,kind=e:mcq|open,points=d:0.5..5)", "course_id,date"),
  EE("library_loans", "emprestimos_biblioteca", "student_id=ref:students;isbn=code:ISBN;borrowed_at=t;returned_at=t??;fine=m:0..80?", "student_id,-borrowed_at"),
  EE("scholarships", "bolsas", "name=title;sponsor=word;amount=m:200..9000;eligible_programs=as:e:cs|law|med|arts|eng;recipients=as:ref:students?", "!name;eligible_programs"),
 ],
 "legal": [
  EE("clients", "clientes", "name=name;tax_id=code:TX;kind=e:person|company;email=email;since=t;address=o(city=city,state=word,country=country)?", "!tax_id;kind"),
  EE("lawyers", "advogados", "name=name;bar=code:OAB;firm=title?;active=b:0.9;areas=as:e:civil|criminal|labor|tax|ip;rate_hour=m:100..2000?", "!bar;active;areas"),
  EE("cases", "processos", "number=code:CNJ;court=e:TJSP|TRF3|STJ|TRT2;subject=title;client_id=ref:clients;filed_at=t:2018-01-01..2026-09-01;status=e:open|stayed|closed;value=m:100..5000000?;parties=a(name=name,role=e:plaintiff|defendant|witness,counsel=o(name=name?,bar=code:OAB?)?);lead_id=ref:lawyers", "!number;court,status;parties.name;client_id,-filed_at"),
  EE("hearings", "audiencias", "case_id=ref:cases;at=t:2026-01-01..2027-03-31;room=code:R;judge=name;notes=sentence?;outcome=e:held|postponed|settled??", "case_id,at;judge"),
  EE("documents", "documentos", "case_id=ref:cases;kind=e:petition|motion|ruling|evidence;pages=i:1..400;hash=hex;filed=t;confidential=b:0.1;uuid=u3", "case_id,kind;!hash;~kind@confidential"),
  EE("contracts", "contratos", "client_id=ref:clients;title=title;signed_at=t;expires_at=t??;value=m:500..9000000;clauses=a(num=i:1..60,title=title,text=sentence,risk=e:low|med|high);parties=as:name", "client_id;expires_at;clauses.risk"),
  EE("filings", "peticoes", "case_id=ref:cases;lawyer_id=ref:lawyers;protocol=code:PT;at=t;kind=word;deadline=t?", "!protocol;case_id,-at"),
  EE("courts", "tribunais", "acronym=code:TR;name=title;jurisdiction=e:state|federal|labor;city=city;judges=i:5..500", "!acronym;jurisdiction"),
  EE("billing", "cobrancas", "client_id=ref:clients;case_id=ref:cases?;hours=d:0.5..90;rate=m:100..2000;total=m:50..180000;issued_at=t;paid=b:0.7", "client_id,-issued_at;~issued_at@paid"),
  EE("deadlines", "prazos", "case_id=ref:cases;title=title;due=t;done=b:0.5;reminders=a(at=t,channel=e:email|sms|push)?", "case_id,due;ttl:due"),
 ],
}


def split_top(s, sep):
    out, d, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            d += 1
        elif ch == ")":
            d -= 1
        if ch == sep and d == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [x.strip() for x in out if x.strip()]


def _rng_num(r, cast):
    lo, hi = r.split("..")
    return cast(lo), cast(hi)


def parse_type(s):
    p, n = 1.0, 0.0
    if s.endswith("??"):
        s, p, n = s[:-2], "o", 0.1
    elif s.endswith("?"):
        s, p = s[:-1], "o"
    if s.startswith("o("):
        return obj(parse_fields(s[2:-1], ","), p, n)
    if s.startswith("a("):
        return arr(obj(parse_fields(s[2:-1], ",")), 1, 4, p, n)
    if s.startswith("as:"):
        return arr(parse_type(s[3:]), 0, 4, p, n)
    k, _, a = s.partition(":")
    if k == "ref":
        return oid(a, p, n)
    if k in ("u3", "u4"):
        return uid(int(k[1]), p, n)
    if k == "e":
        return enum(a.split("|"), None, p, n)
    if k == "i":
        lo, hi = _rng_num(a, int)
        return i32(lo, hi, p, n)
    if k == "l":
        lo, hi = _rng_num(a, int)
        return i64(lo, hi, p, n)
    if k == "d":
        lo, hi = _rng_num(a, float)
        return dbl(lo, hi, p, n)
    if k == "m":
        lo, hi = _rng_num(a, float)
        return dec(lo, hi, p, n)
    if k == "b":
        return flag(float(a) if a else 0.5, p, n)
    if k == "t":
        if a:
            lo, hi = a.split("..")
            return date(lo, hi, p, n)
        return date(p=p, n=n)
    if k == "code":
        return text("code", p, n, prefix=a)
    return text(k, p, n)


def parse_fields(s, sep):
    out = {}
    for it in split_top(s, sep):
        name, _, ty = it.partition("=")
        out[name.strip()] = parse_type(ty.strip())
    return out


def tr_name(name, pt):
    if not pt:
        return name
    return "_".join(TR.get(t, t) for t in name.split("_"))


def tr_fields(fields, pt):
    if not pt:
        return fields
    out = {}
    for k, s in fields.items():
        s = dict(s)
        if s["t"] == "object":
            s["f"] = tr_fields(s["f"], pt)
        elif s["t"] == "array" and s["item"]["t"] == "object":
            it = dict(s["item"])
            it["f"] = tr_fields(it["f"], pt)
            s["item"] = it
        out[tr_name(k, pt)] = s
    return out


def _walk(fields, path):
    """-> (spec_final, n_arrays) ou None."""
    f, arrs, spec = fields, 0, None
    parts = path.split(".")
    for i, part in enumerate(parts):
        spec = f.get(part)
        if spec is None:
            return None
        if spec["t"] == "array":
            arrs += 1
            if spec["item"]["t"] == "object":
                f = spec["item"]["f"]
            elif i != len(parts) - 1:
                return None
        elif spec["t"] == "object":
            f = spec["f"]
        elif i != len(parts) - 1:
            return None
    return spec, arrs


def idx_ok(fields, ix):
    keys = [k.lstrip("-") for k in ix["keys"].split(",")]
    arrays, specs = 0, []
    for k in keys:
        w = _walk(fields, k)
        if w is None:
            return False
        specs.append(w[0])
        arrays += w[1]
    if arrays > 1 or len(set(keys)) != len(keys):
        return False
    # TTL de 10 anos: so em data recente (lo >= 2025), senao o TTL monitor apaga docs antigos e quebra o hash
    if ix.get("ttl") and not (len(keys) == 1 and specs[0]["t"] == "date" and specs[0]["lo"] >= "2025-01-01"
                              and not arrays):
        return False
    if ix.get("text") and specs[0]["t"] != "string":
        return False
    if ix.get("unique") and (arrays or any(s["t"] in ("object", "array", "objectid", "bool") for s in specs)):
        return False
    return True


TTL_10Y = 315360000


def parse_idx(spec):
    out = []
    for it in [x.strip() for x in spec.split(";") if x.strip()]:
        if it.startswith("!"):
            out.append(dict(keys=it[1:], unique=True))
        elif it.startswith("ttl:"):
            out.append(dict(keys=it[4:], ttl=TTL_10Y))
        elif it.startswith("text:"):
            out.append(dict(keys=it[5:], text=True))
        elif it.startswith("~"):
            ks, _, fl = it[1:].partition("@")
            out.append(dict(keys=ks, partial_flag=fl))
        else:
            out.append(dict(keys=it))
    return out


NEST_TOK = "settings audit limits stats attrs extra window quota".split()
LEAF = [("count", lambda: i32(0, 900)), ("score", lambda: dbl(0, 100)), ("enabled", lambda: flag(0.5)),
        ("label", lambda: text("word")), ("at", lambda: date()), ("total", lambda: dec(0, 5000)),
        ("size", lambda: i64(0, 9_000_000_000)), ("mode", lambda: enum(["a", "b", "c"]))]


def nested(rng, depth):
    leaves = {}
    for nm, mk in rng.sample(LEAF, rng.randint(2, 3)):
        v = mk()
        v["p"] = rng.choice([1.0, "o"])
        leaves[nm] = v
    if depth > 1:
        leaves[rng.choice(NEST_TOK)] = nested(rng, depth - 1)
    return obj(leaves, rng.choice([1.0, "o", "o"]))


def make_extras(rng, fields, idx):
    """Variedade extra (tipos BSON, aninhamento até 4 níveis, arrays de subdocs)."""
    if rng.random() < 0.40:
        fields[rng.choice(["meta", "extra", "attrs", "info"])] = nested(rng, rng.randint(2, 3))
    if rng.random() < 0.35 and not any(s["t"] == "uuid" for s in fields.values()):
        fields[rng.choice(["external_id", "legacy_id", "uid"])] = uid(rng.choice([3, 4]), rng.choice([1.0, "o"]))
    if rng.random() < 0.25:
        fields["version_no"] = i64(1, 9_000_000_000, "o")
    if rng.random() < 0.25:
        fields["amount_total"] = dec(0, 90000, "o")
    if rng.random() < 0.3 and "tags" not in fields:
        fields["tags"] = arr(text("word"), 0, 4, p="o")
        if rng.random() < 0.6:
            idx.append(dict(keys="tags"))
    if rng.random() < 0.20:
        fields["history"] = arr(obj({"at": date(), "by": text("name"),
                                     "change": obj({"field": text("word"), "from": text("word", p="o"), "to": text("word", p="o")})}), 0, 3, p="o")
    if not any(s["t"] == "date" for s in fields.values()):
        fields["created_at"] = date()
        if rng.random() < 0.5:
            idx.append(dict(keys="-created_at"))


def search_def(rng, name, fields):
    """Definição desejada de Atlas Search/Vector (static_only; NÃO criada no servidor)."""
    strs = [k for k, s in fields.items() if s["t"] == "string" and s.get("g") != "enum"]
    ens = [k for k, s in fields.items() if s["t"] == "string" and s.get("g") == "enum"]
    nums = [k for k, s in fields.items() if s["t"] in ("int32", "int64", "double", "decimal128", "date")]
    emb = next((k for k in fields if k.lower() == "embedding"), None)
    safe = name.replace(" ", "_").replace("-", "_").lower()
    out = []
    mp = {}
    for k in strs[:3]:
        mp[k] = {"type": "string", "analyzer": rng.choice(["lucene.standard", "lucene.portuguese", "lucene.english"])}
    for k in ens[:2]:
        mp[k] = {"type": "token"}
    for k in nums[:2]:
        mp[k] = {"type": "date" if fields[k]["t"] == "date" else "number"}
    if mp and (emb is None or rng.random() < 0.6):
        out.append({"name": f"{safe}_text", "type": "search",
                    "definition": {"mappings": {"dynamic": False, "fields": mp}}, "static_only": True})
    if emb:
        flt = [{"type": "filter", "path": k} for k in (ens[:1] + nums[:1])]
        out.append({"name": f"{safe}_vec", "type": "vectorSearch",
                    "definition": {"fields": [{"type": "vector", "path": emb, "numDimensions": 8,
                                               "similarity": rng.choice(["cosine", "euclidean", "dotProduct"])}] + flt},
                    "static_only": True})
    return out


def procedural_defs(seed, hand):
    """Completa cada domínio até 6–10 schemas com coleções montadas da biblioteca ENT."""
    existing = {}
    for d in hand:
        existing[d["domain"]] = existing.get(d["domain"], 0) + 1
    out = []
    for dom, ents in ENT.items():
        r = random.Random(f"{seed}:proc:{dom}")
        target = r.randint(7, 10) if dom not in ("education", "legal") else r.randint(6, 8)
        n_new = max(0, target - existing.get(dom, 0))
        brands = r.sample(BRANDS, n_new)
        styles = ["camel", "pascal", "snake", "snake", "camel", "pascal"]
        for k, brand in enumerate(brands):
            rng = random.Random(f"{seed}:proc:{dom}:{brand}")
            pt = k % 3 == 2  # ~1/3 dos schemas em pt-BR sem acento
            style = styles[(k + (1 if pt else 0)) % len(styles)]
            pick = rng.sample(ents, min(rng.randint(4, 12), len(ents)))
            keys = {e[0] for e in pick}
            changed = True
            while changed:  # fecha dependências (refs) até 12 coleções
                changed = False
                for e in list(pick):
                    for m in re.findall(r"ref:(\w+)", e[2]):
                        if m not in keys and len(pick) < 12 and any(x[0] == m for x in ents):
                            pick.append(next(x for x in ents if x[0] == m))
                            keys.add(m)
                            changed = True
            cols = []
            for key, ptkey, fs, ixs in pick:
                fields = parse_fields(fs, ";")
                names = list(fields)
                for nm in names:
                    s = fields[nm]
                    if s["t"] == "objectid" and s.get("ref") and s["ref"] not in keys:
                        del fields[nm]
                    elif nm not in names[:3] and rng.random() < 0.15 and len(fields) > 4:
                        del fields[nm]
                for s in fields.values():
                    if s["t"] == "object":
                        s["f"] = {a: b for a, b in s["f"].items()
                                  if not (b["t"] == "objectid" and b.get("ref") and b["ref"] not in keys)}
                idx = parse_idx(ixs)
                make_extras(rng, fields, idx)
                if rng.random() < 0.25 and key in ("articles", "products", "documents", "cases", "courses", "support_tickets", "hotels", "tours", "posts"):
                    fields["embedding"] = arr(dbl(-1, 1), 8, 8, p="o")
                fin = []
                for ix in idx:
                    if "partial_flag" in ix:
                        fl = ix.pop("partial_flag")
                        if fl in fields and fields[fl]["t"] == "bool":
                            ix["partial"] = {fl: True}
                        else:
                            root = ix["keys"].split(",")[0].lstrip("-")
                            ix["partial"] = {root: {"$exists": True}}
                    if idx_ok(fields, ix):
                        fin.append(ix)
                seen = {}
                for ix in fin:  # dedupe por chaves (TTL tem precedência)
                    kk = ix["keys"]
                    if kk not in seen or ix.get("ttl"):
                        seen[kk] = ix
                fin, has_text = [], False
                for ix in seen.values():
                    if ix.get("text"):
                        if has_text:
                            continue
                        has_text = True
                    fin.append(ix)
                if not any(x.get("ttl") for x in fin) and rng.random() < 0.18:
                    used = {x["keys"].lstrip("-") for x in fin}
                    dts = [nm for nm, s in fields.items() if s["t"] == "date" and s["lo"] >= "2025-01-01" and nm not in used]
                    if dts:
                        fin.append(dict(keys=rng.choice(dts), ttl=TTL_10Y))
                toks = (ptkey if pt else key).split("_")
                cname = "_".join(toks)
                if len(toks) > 1 and rng.random() < 0.18:
                    if rng.random() < 0.5:
                        cname = "-".join(toks)
                    else:
                        cname = " ".join(w.capitalize() for w in toks) if style == "pascal" else " ".join(toks)
                cstyle = rng.choice(ST) if rng.random() < 0.12 else None
                fields = tr_fields(fields, pt)

                def trk(path):
                    return ".".join(tr_name(p, pt) for p in path.split("."))
                idx2 = []
                for ix in fin:
                    ix = dict(ix)
                    ix["keys"] = ",".join(("-" if x.startswith("-") else "") + trk(x.lstrip("-")) for x in ix["keys"].split(","))
                    if ix.get("partial"):
                        ix["partial"] = {trk(a): b for a, b in ix["partial"].items()}
                    idx2.append(ix)
                cols.append(C(key, cname, fields, idx2, style=cstyle))
            lang = "pt-BR sem acento" if pt else "EN"
            out.append(dict(domain=dom, schema_id=f"{brand}_{DS[dom]}", style=style,
                            desc=f"{dom} procedural {lang}, {style}", cols=cols,
                            holdout=dom in ("education", "legal"), lang="pt-BR" if pt else "en"))
    return out


# ------------------------------------------------------------- helpers ----
def style_name(name, style):
    if "-" in name or " " in name or name == "_id":
        return name
    words = name.split("_")
    if style == "snake":
        return "_".join(words)
    if style == "camel":
        return words[0] + "".join(w[:1].upper() + w[1:] for w in words[1:])
    if style == "pascal":
        return "".join(w[:1].upper() + w[1:] for w in words)
    raise ValueError(style)


def style_path(path, style):
    return ".".join(style_name(p, style) for p in path.split("."))


def style_fields(fields, style, rng):
    out = {}
    for k, spec in fields.items():
        s = dict(spec)
        if s["p"] == "o":
            s["p"] = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        if s["t"] == "object":
            s["f"] = style_fields(s["f"], style, rng)
        elif s["t"] == "array":
            item = dict(s["item"])
            item["p"] = 1.0
            if item["t"] == "object":
                item["f"] = style_fields(item["f"], style, rng)
            s["item"] = item
        out[style_name(k, style)] = s
    return out


def style_partial(d, style):
    out = {}
    for k, v in d.items():
        nk = k if k.startswith("$") else style_path(k, style)
        out[nk] = style_partial(v, style) if isinstance(v, dict) else v
    return out


def depth(fields):
    best = 1
    for s in fields.values():
        if s["t"] == "object":
            best = max(best, 1 + depth(s["f"]))
        elif s["t"] == "array" and s["item"]["t"] == "object":
            best = max(best, 2 + depth(s["item"]["f"]) - 1)
    return best


BSON_TYPE = {"objectid": "objectId", "uuid": "binData", "string": "string", "int32": "int", "int64": "long",
             "double": "double", "decimal128": "decimal", "bool": "bool", "date": "date", "object": "object", "array": "array"}


def json_schema(fields, id_type):
    props, req = {}, []
    for k, s in fields.items():
        bt = BSON_TYPE[s["t"]]
        p = {"bsonType": [bt, "null"] if s["n"] else bt}
        if s["t"] == "string" and s.get("g") == "enum":
            p["enum"] = list(s["values"]) + ([None] if s["n"] else [])
        if s["t"] == "object":
            sub = json_schema(s["f"], None)
            p.update({k2: v for k2, v in sub.items() if k2 != "bsonType"})
        if s["t"] == "array":
            it = s["item"]
            ip = {"bsonType": BSON_TYPE[it["t"]]}
            if it["t"] == "object":
                ip.update({k2: v for k2, v in json_schema(it["f"], None).items() if k2 != "bsonType"})
            p["items"] = ip
        props[k] = p
        if s["p"] == 1.0:
            req.append(k)
    out = {"bsonType": "object", "properties": props}
    if req:
        out["required"] = req
    return out


def build(defn, seed):
    sid = defn["schema_id"]
    rng = random.Random(f"{seed}:{sid}")
    style = defn["style"]
    cols = []
    for c in defn["cols"]:
        cstyle = c["style"] or style
        fields = style_fields(c["fields"], cstyle, rng)
        name = c["name"] if ("-" in c["name"] or " " in c["name"]) else style_name(c["name"], cstyle)
        if c["_id"] is None:
            idspec = {"t": "objectid"}
        else:
            idspec = c["_id"]
        col = {
            "key": c["key"], "name": name,
            "doc_count": rng.randrange(200, 2001, 50),
            "fields": fields,
            "indexes": [],
        }
        if idspec["t"] != "objectid":
            col["_id"] = idspec
        for ix in c["idx"]:
            keys = []
            for part in ix["keys"].split(","):
                d = -1 if part.startswith("-") else 1
                keys.append([style_path(part.lstrip("-"), cstyle), d])
            e = {"keys": keys}
            if ix.get("unique"):
                e["unique"] = True
            if ix.get("ttl"):
                e["expireAfterSeconds"] = ix["ttl"]
            if ix.get("partial"):
                e["partialFilterExpression"] = style_partial(ix["partial"], cstyle)
            if ix.get("text"):
                e["keys"] = [[keys[0][0], "text"]]
            col["indexes"].append(e)
        # unique em campo opcional/nulo => partial implícito $exists (evita colisão null)
        for e in col["indexes"]:
            if e.get("unique") and "partialFilterExpression" not in e and len(e["keys"]) == 1:
                f = fields.get(e["keys"][0][0])
                if f and (f["p"] < 1.0 or f["n"]):
                    e["partialFilterExpression"] = {e["keys"][0][0]: {"$exists": True}}
        has_enum = any(s.get("g") == "enum" for s in fields.values())
        if rng.random() < (0.70 if has_enum else 0.15):  # enum em >=30% das colecoes
            col["validator"] = {"$jsonSchema": json_schema(fields, idspec)}
            col["validationAction"] = "error"
        srng = random.Random(f"{seed}:{sid}:{c['key']}:search")  # fluxo separado: nao altera o resto
        emb = any(k.lower() == "embedding" for k in fields)
        if emb or srng.random() < 0.22:
            sd = search_def(srng, name, fields)
            if sd:
                col["search_indexes"] = sd
        col["max_depth"] = depth(fields)
        cols.append(col)
    return {
        "schema_id": sid, "domain": defn["domain"], "holdout": defn["holdout"], "description": defn["desc"],
        "name_style": style, "seed": seed, "format_version": 1, "collections": cols,
        "language": defn.get("lang", "pt-BR" if "pt-BR" in defn["desc"] else "en"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(ROOT / "data" / "schemas"))
    a = ap.parse_args()
    out = Path(a.out)
    n_s = n_c = 0
    hand = schema_defs()
    defs = hand + procedural_defs(a.seed, hand)
    built = [(d, build(d, a.seed)) for d in defs]
    roles = assign_roles([{k: s[k] for k in ("schema_id", "domain", "holdout", "name_style", "language")} for _, s in built], a.seed)
    for d, s in built:
        s = {**{k: v for k, v in s.items() if k != "collections"}, "split_role": roles[s["schema_id"]], "collections": s["collections"]}
        p = out / d["domain"] / f"{d['schema_id']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(s, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        n_s += 1
        n_c += len(s["collections"])
        print(f"{d['domain']:22s} {d['schema_id']:14s} colls={len(s['collections'])} "
              f"validators={sum('validator' in c for c in s['collections'])} depth<={max(c['max_depth'] for c in s['collections'])}"
              f"{' HOLDOUT' if d['holdout'] else ''}")
    print(f"schemas={n_s} collections={n_c} -> {out}")


if __name__ == "__main__":
    sys.exit(main())
