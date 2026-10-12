"""Gerador de exemplos de chat bilíngues (PT-BR/EN) no contrato real do chat da IDE (LocalModelAiChatService).

Cada exemplo: instrução + snapshot AiEditorContext (editor, banco, coleção, tipo de operação inferido como a IDE, contexto adicional
com conexão, destino visual e campos conhecidos) → código completo de substituição do editor, sem Markdown.
- create: nova consulta (editor vazio ou comando padrão do Explorer);
- modify: altera o código do editor (filtro, ordenação/limite, projeção, find→aggregate, contagem, conexão), com statements extras
  preservados;
- correction: corrige erros comuns (ConnectionPull, GetDatabase, operador sem $, estágio/método digitado errado, parêntese faltando,
  count() no cursor do Console, insert legado, data como texto);
- explain: explicação mínima em comentário // seguida do código (única forma de texto aceita pelo contrato atual da IDE).

Valores citados na instrução aparecem literalmente no código (datas com precisão de dia, ObjectId explícito, flags de regex
descritas), para que a saída seja aprendível. Cada exemplo carrega uma intenção verificável (operações aceitas + elementos
obrigatórios) usada pelas métricas Chat Intent Accuracy e PT-BR/English Understanding do benchmark.
"""
from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from . import schemas as S
from .codegen_ext import CompactSnippetGenerator
from .common import pick
from .contracts import chat_context
from .jsrender import IDENT, DateV, Money, Oid, Raw, Regex, Style, call, number, render, render_inline

JS_STYLE = Style(indent=2, width=80, key_quotes="minimal", string_quote='"', brace_space=True, json=False, semicolons=False)
JSON_STYLE = Style(indent=2, width=80, key_quotes="all", string_quote='"', brace_space=True, json=True)
MODE_DIALECT = {"console": "console", "script": "mongosh", "aggregation": "json"}

NOUNS = {
    "orders": ("pedidos", "orders"), "customers": ("clientes", "customers"), "products": ("produtos", "products"),
    "reviews": ("avaliações", "reviews"), "projects": ("projetos", "projects"), "users": ("usuários", "users"),
    "events": ("eventos", "events"), "subscriptions": ("assinaturas", "subscriptions"), "tickets": ("chamados", "tickets"),
    "shipments": ("envios", "shipments"), "warehouses": ("armazéns", "warehouses"), "vehicles": ("veículos", "vehicles"),
    "inventory": ("itens de estoque", "inventory items"), "devices": ("dispositivos", "devices"), "readings": ("leituras", "readings"),
    "alerts": ("alertas", "alerts"), "employees": ("funcionários", "employees"), "departments": ("departamentos", "departments"),
    "timesheets": ("apontamentos de horas", "timesheets"), "articles": ("artigos", "articles"), "comments": ("comentários", "comments"),
    "authors": ("autores", "authors"), "invoices": ("faturas", "invoices"), "payments": ("pagamentos", "payments"),
    "accounts": ("contas", "accounts"), "bookings": ("reservas", "bookings"), "hotels": ("hotéis", "hotels"), "flights": ("voos", "flights"),
    "courses": ("cursos", "courses"), "enrollments": ("matrículas", "enrollments"), "students": ("alunos", "students"),
    "tasks": ("tarefas", "tasks"),
}
FIELD_WORDS = {
    "status": ("status", "status"), "state": ("estado", "state"), "category": ("categoria", "category"), "total": ("total", "total"),
    "amount": ("valor", "amount"), "price": ("preço", "price"), "budget": ("orçamento", "budget"), "city": ("cidade", "city"),
    "country": ("país", "country"), "plan": ("plano", "plan"), "role": ("perfil", "role"), "priority": ("prioridade", "priority"),
    "severity": ("severidade", "severity"), "department": ("departamento", "department"), "salary": ("salário", "salary"),
    "rating": ("nota", "rating"), "stock": ("estoque", "stock"), "quantity": ("quantidade", "quantity"), "views": ("visualizações", "views"),
    "tier": ("nível", "tier"), "carrier": ("transportadora", "carrier"), "paymentmethod": ("forma de pagamento", "payment method"),
    "type": ("tipo", "type"), "brand": ("marca", "brand"), "seats": ("assentos", "seats"), "channel": ("canal", "channel"),
    "metric": ("métrica", "metric"), "level": ("nível", "level"), "credits": ("créditos", "credits"), "createdat": ("data de criação", "creation date"),
    "createdon": ("data de criação", "creation date"), "updatedat": ("data de atualização", "update date"),
    "updatedon": ("data de atualização", "update date"), "email": ("e-mail", "email"), "active": ("ativo", "active"),
    "stars": ("estrelas", "stars"), "nights": ("diárias", "nights"), "progress": ("progresso", "progress"), "grade": ("nota", "grade"),
    "likes": ("curtidas", "likes"), "followers": ("seguidores", "followers"), "hours": ("horas", "hours"), "capacity": ("capacidade", "capacity"),
    "balance": ("saldo", "balance"), "method": ("método", "method"), "gateway": ("gateway", "gateway"), "language": ("idioma", "language"),
    "lastloginat": ("último login", "last login"), "lastloginon": ("último login", "last login"), "completedon": ("data de conclusão", "completion date"),
    "resolvedat": ("data de resolução", "resolution date"), "deliveredat": ("data de entrega", "delivery date"), "paidat": ("data de pagamento", "payment date"),
    "publishedat": ("data de publicação", "publish date"), "duedate": ("prazo", "due date"), "dueon": ("prazo", "due date"),
}
PARENT_WORDS = {"account": ("conta", "account"), "customer": ("cliente", "customer"), "owner": ("responsável", "owner"),
                "project": ("projeto", "project"), "assignee": ("responsável", "assignee"), "system": ("sistema", "system"),
                "billing": ("cobrança", "billing"), "shipping": ("entrega", "shipping"), "address": ("endereço", "address"),
                "destination": ("destino", "destination"), "origin": ("origem", "origin"), "device": ("dispositivo", "device"),
                "profile": ("perfil", "profile"), "contract": ("contrato", "contract")}
CREATED = {"createdat", "createdon"}
# Datas com verbo natural (PT, EN particípio para "criados/created", EN verbo para "users that logged in").
DATE_EVENTS = {
    "createdat": ("criados", "created", "were created"), "createdon": ("criados", "created", "were created"),
    "updatedat": ("atualizados", "updated", "were updated"), "updatedon": ("atualizados", "updated", "were updated"),
    "lastloginat": ("que fizeram login", "that logged in", "logged in"), "lastloginon": ("que fizeram login", "that logged in", "logged in"),
    "lastseenat": ("vistos", "seen", "were seen"), "completedon": ("concluídos", "completed", "were completed"),
    "completedat": ("concluídos", "completed", "were completed"), "resolvedat": ("resolvidos", "resolved", "were resolved"),
    "deliveredat": ("entregues", "delivered", "were delivered"), "paidat": ("pagos", "paid", "were paid"),
    "publishedat": ("publicados", "published", "were published"), "hiredat": ("contratados", "hired", "were hired"),
    "processedat": ("processados", "processed", "were processed"), "enrolledat": ("matriculados", "enrolled", "enrolled"),
}
# v1.2 — adjetivos booleanos: ((PT verdadeiro, PT falso), (posição EN "pre"/"post", EN verdadeiro, EN falso)).
BOOL_WORDS = {
    "active": (("ativos", "inativos"), ("pre", "active", "inactive")),
    "verified": (("verificados", "não verificados"), ("pre", "verified", "unverified")),
    "approved": (("aprovados", "não aprovados"), ("pre", "approved", "unapproved")),
    "acknowledged": (("já reconhecidos", "ainda não reconhecidos"), ("post", "that were acknowledged", "that weren't acknowledged")),
    "mfaenabled": (("com MFA ativado", "sem MFA"), ("post", "with MFA enabled", "without MFA")),
    "newsletter": (("que assinam a newsletter", "que não assinam a newsletter"), ("post", "subscribed to the newsletter", "not subscribed to the newsletter")),
}
ADJECTIVE_ENUMS = {"priority", "severity", "tier", "plan"}              # "high-priority tickets", "gold-tier customers"
PRICE_LEAVES = {"price", "total", "totalprice", "amount", "pricepernight"}  # "mais caro", "cheapest first"
PEOPLE_COLLECTIONS = {"users", "customers", "students", "employees", "authors", "accounts"}


def agree_pt(text: str, noun: str) -> str:
    """Concordância de gênero mínima: adjetivos/particípios em -os viram -as quando o substantivo PT é feminino (faturas pagas)."""
    if not noun.endswith(("as", "ões")):
        return text
    return re.sub(r"\b(\w{2,})os\b", lambda m: m.group(1) + "as", text)


DOUBLED = re.compile(r"\b(with|com|total|by|por|the|os|as) \1\b", re.IGNORECASE)
INSERT = re.compile(r"\.(?:insert|insertOne|insertMany)\s*\(", re.IGNORECASE)
DELETE = re.compile(r"\.(?:delete|deleteOne|deleteMany|drop|dropDatabase)\s*\(", re.IGNORECASE)
UPDATE = re.compile(r"\.(?:update|updateOne|updateMany|replace|replaceOne|bulkWrite)\s*\(", re.IGNORECASE)


def infer_operation(text: str, mode: str) -> str:
    """WorkspaceTabViewModel.InferOperationType."""
    if mode == "script":
        return "script"
    if INSERT.search(text):
        return "inserção"
    if DELETE.search(text):
        return "exclusão"
    if UPDATE.search(text):
        return "atualização"
    return "consulta"


def deaccent(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


@dataclass
class Scenario:
    kind: str
    intent: str
    category: str
    pt: str
    en: str
    editor: str
    output: str
    operations: list[str]
    must: list[list[str]]
    collection: str | None
    desc_pt: str = ""
    desc_en: str = ""
    semantic: bool = False
    stages: list | None = None
    extra: dict = field(default_factory=dict)


class ChatGenerator:
    def __init__(self, rng: random.Random, domain: str, mode: str, cfg: dict) -> None:
        self.rng = r = rng
        self.cfg = cfg
        self.domain = domain
        self.mode = mode
        self.json = mode == "aggregation"
        self.g = CompactSnippetGenerator(rng, domain, MODE_DIALECT[mode], "crud", None)
        self.style = JSON_STYLE if self.json else JS_STYLE
        self.g.style = self.style
        self.g.date_form = "isodate"
        self.g.json = self.json
        self.name = self.g.cname(self.g.coll_key)
        self.natural = r.random() < cfg.get("natural_field_label_probability", .5)
        noun = NOUNS.get(self.g.coll_key.lower())
        self.coll_natural = bool(noun) and r.random() < .5
        self.noun = noun
        self.target_kind = "dot" if IDENT.match(self.name) and r.random() < .35 else "get"

    # -- nomes e rótulos ---------------------------------------------------------------------------------------------------------
    def coll(self, lang: str) -> str:
        if self.coll_natural and self.noun:
            return self.noun[0 if lang == "pt" else 1]
        return self.name

    def path(self, f: S.Field) -> str:
        return self.g.fpath(f)

    def label(self, f: S.Field, lang: str) -> str:
        path = self.path(f)
        if not self.natural:
            return path
        index = 0 if lang == "pt" else 1
        parts = f.path.split(".")
        leaf = parts[-1].lower().replace("_", "")
        parent = parts[-2].lower() if len(parts) > 1 else ""
        if leaf == "id" and parent in PARENT_WORDS:
            text = PARENT_WORDS[parent][index]
        elif leaf in FIELD_WORDS:
            word = FIELD_WORDS[leaf][index]
            text = (f"{word} de {PARENT_WORDS[parent][index]}" if lang == "pt" else f"{PARENT_WORDS[parent][index]} {word}") if parent in PARENT_WORDS else word
        else:
            return path
        others = [x for x in self.g.coll.fields if x is not f and not x.array_parent]
        if any(self.label_raw(x, index) == text for x in others):
            return path
        return text

    def label_raw(self, f: S.Field, index: int) -> str | None:
        parts = f.path.split(".")
        leaf = parts[-1].lower().replace("_", "")
        parent = parts[-2].lower() if len(parts) > 1 else ""
        if leaf == "id" and parent in PARENT_WORDS:
            return PARENT_WORDS[parent][index]
        return FIELD_WORDS.get(leaf, (None, None))[index]

    def cond_fields(self) -> list[S.Field]:
        return [f for f in self.g.coll.fields if not f.array_parent and f.type not in ("geo", "uuid", "version")]

    # -- renderização --------------------------------------------------------------------------------------------------------------
    def q(self, text: str) -> str:
        return render_inline(text, self.style)

    def target(self, name: str | None = None) -> list[str]:
        name = name or self.name
        if self.target_kind == "dot" and IDENT.match(name):
            return ["db", "." + name]
        return ["db", f".getCollection({self.q(name)})"]

    def chain(self, target: list[str], calls: list[tuple[str, list]]) -> str:
        text = "".join(target)
        for method, args in calls:
            used = len(text) - (text.rfind("\n") + 1)
            text += call("." + method, args, self.style, level=0, used=used)
        return text

    def pipeline(self, stages: list[dict], target: list[str] | None = None) -> str:
        if self.json:
            return render(stages, self.style)
        return self.chain(target or self.target(), [("aggregate", [stages])])

    def default_editor(self) -> str:
        if self.json:
            return self.rng.choice(["", "[]", '[\n  { "$match": {} }\n]'])
        return self.rng.choice(["", "", f"db.getCollection({self.q(self.name)}).find({{}}).limit(100)"])

    # -- condições com valores explícitos ------------------------------------------------------------------------------------------------
    def cond(self, f: S.Field) -> tuple[Any, str, str, list[list[str]]]:
        r, g = self.rng, self.g
        path = self.path(f)
        lp, le = self.label(f, "pt"), self.label(f, "en")
        must = [[path]]
        t = f.type
        if t == "bool":
            value = r.random() < .6
            text = "true" if value else "false"
            return value, f"{lp} = {text}", f"{le} = {text}", must + [[text]]
        if t in ("enum", "code", "city", "state", "country", "name", "word", "text", "email"):
            regexable = t in ("text", "name", "word", "email", "city")
            k = pick(r, {"eq": .5, "in": .2, "ne": .1, "regex": .12 if regexable else 0, "exists": .08})
            value = str(g.value(f))
            if k == "eq":
                return value, f"{lp} igual a {value}", f"{le} equal to {value}", must + [[value]]
            if k == "in":
                values = [str(v) for v in g.distinct_values(f, r.randint(2, 3))]
                if len(values) < 2:
                    return value, f"{lp} igual a {value}", f"{le} equal to {value}", must + [[value]]
                return {"$in": values}, f"{lp} em {', '.join(values)}", f"{le} in {', '.join(values)}", must + [["$in"]] + [[v] for v in values]
            if k == "ne":
                return {"$ne": value}, f"{lp} diferente de {value}", f"{le} not equal to {value}", must + [["$ne"], [value]]
            if k == "regex":
                prefix = re.sub(r"[^A-Za-z0-9]", "", value)[: r.randint(2, 4)] or "a"
                insensitive = r.random() < .6
                regex: Any = ({"$regex": "^" + prefix, **({"$options": "i"} if insensitive else {})} if self.json
                              else Regex("^" + prefix, "i" if insensitive else ""))
                return (regex, f"{lp} começando com {prefix}" + (" (sem diferenciar maiúsculas)" if insensitive else ""),
                        f"{le} starting with {prefix}" + (" (case-insensitive)" if insensitive else ""), must + [["^" + prefix]])
            exists = r.random() < .7
            return {"$exists": exists}, f"{'com' if exists else 'sem'} {lp}", f"{'with' if exists else 'without'} {le}", must + [["$exists"]]
        if f.numeric:
            lo, hi = f.range()
            a = r.randint(int(lo), int(hi)) if t == "int" else Money(round(r.uniform(lo, hi), r.choice([0, 2])))
            b = r.randint(int(a), int(hi)) if t == "int" else Money(round(r.uniform(float(a), hi), 2))
            na, nb = number(a), number(b)
            k = pick(r, {"$gte": .3, "$gt": .2, "$lt": .15, "$lte": .1, "range": .25 if nb != na else 0})
            if k == "range":
                return ({"$gte": a, "$lte": b}, f"{lp} entre {na} e {nb}", f"{le} between {na} and {nb}",
                        must + [["$gte"], ["$lte"], [na], [nb]])
            words = {"$gte": ("maior ou igual a", "greater than or equal to"), "$gt": ("maior que", "greater than"),
                     "$lt": ("menor que", "less than"), "$lte": ("menor ou igual a", "less than or equal to")}[k]
            return {k: a}, f"{lp} {words[0]} {na}", f"{le} {words[1]} {na}", must + [[k], [na]]
        if t == "date":
            d1 = (S.EPOCH + timedelta(days=r.randint(0, 900))).strftime("%Y-%m-%d")
            d2 = (S.EPOCH + timedelta(days=r.randint(901, 1000))).strftime("%Y-%m-%d")
            k = pick(r, {"$gte": .45, "$lt": .2, "range": .35})
            v1, v2 = DateV(d1 + "T00:00:00Z"), DateV(d2 + "T00:00:00Z")
            if k == "range":
                return {"$gte": v1, "$lt": v2}, f"{lp} entre {d1} e {d2}", f"{le} between {d1} and {d2}", must + [["$gte"], ["$lt"], [d1], [d2]]
            words = {"$gte": ("a partir de", "on or after"), "$lt": ("antes de", "before")}[k]
            return {k: v1}, f"{lp} {words[0]} {d1}", f"{le} {words[1]} {d1}", must + [[k], [d1]]
        if t == "tags":
            k = pick(r, {"eq": .4, "$all": .25, "$in": .2, "$size": .15})
            if k == "eq":
                value = str(g.value(f))
                return value, f"{lp} contendo {value}", f"{le} containing {value}", must + [[value]]
            if k == "$size":
                n = r.randint(1, 3)
                return {"$size": n}, f"{lp} com {n} itens", f"{le} with {n} items", must + [["$size"]]
            values = [str(v) for v in g.distinct_values(f, 2)]
            word = ("com todas as tags", "with all of") if k == "$all" else ("com alguma das tags", "with any of")
            return {k: values}, f"{lp} {word[0]} {', '.join(values)}", f"{le} {word[1]} {', '.join(values)}", must + [[k]] + [[v] for v in values]
        if t in ("ref", "oid"):
            oid = S.oid(r)
            return Oid(oid.hex), f"{lp} = ObjectId {oid.hex}", f"{le} = ObjectId {oid.hex}", must + [[oid.hex]]
        raise ValueError(f"tipo sem condição de chat: {t}")

    def filter(self, n: int) -> tuple[dict, str, str, list[list[str]]]:
        fields = self.rng.sample(self.cond_fields(), min(n, len(self.cond_fields())))
        flt, pts, ens, must = {}, [], [], []
        for f in fields:
            value, pt, en, m = self.cond(f)
            flt[self.path(f)] = value
            pts.append(pt)
            ens.append(en)
            must += m
        return flt, " e ".join(pts), " and ".join(ens), must

    # -- intenções: criação ----------------------------------------------------------------------------------------------------------------
    def limit_phrase(self, limit: int) -> tuple[str, str, list[str]]:
        r = self.rng
        return (r.choice([f", limite {limit}", f" (no máximo {limit})", f" e limite a {limit}"]),
                r.choice([f", limit {limit}", f" (at most {limit})", f" and limit to {limit}"]),
                [f"limit({limit})", f"$limit:{limit}", f'"$limit":{limit}'])

    def s_find_filter(self) -> Scenario:
        r = self.rng
        flt, cpt, cen, must = self.filter(pick(r, {1: .55, 2: .35, 3: .1}))
        limit = r.choice([5, 10, 20, 50]) if r.random() < .35 else 0
        if self.json:
            output = self.pipeline([{"$match": flt}] + ([{"$limit": limit}] if limit else []))
        else:
            output = self.chain(self.target(), [("find", [flt])] + ([("limit", [limit])] if limit else []))
        cp, ce = self.coll("pt"), self.coll("en")
        pt = r.choice(["liste {c} onde {x}", "busque {c} com {x}", "mostre {c} em que {x}", "quero ver {c} com {x}", "me traga {c} onde {x}",
                       "encontre {c} com {x}", "filtre {c} por {x}"]).format(c=cp, x=cpt)
        en = r.choice(["find {c} where {x}", "list {c} with {x}", "show me {c} where {x}", "get {c} where {x}", "query {c} with {x}",
                       "filter {c} by {x}"]).format(c=ce, x=cen)
        if limit:
            lp, le, alts = self.limit_phrase(limit)
            pt, en, must = pt + lp, en + le, must + [alts]
        return Scenario("create", "find_filter", "crud", pt, en, self.default_editor(), output, ["find", "aggregate"], must, self.name,
                        f"Busca {self.name} onde {cpt}.", f"Finds {self.name} where {cen}.")

    def s_recent_days(self) -> Scenario:
        r = self.rng
        dates = [f for f in self.cond_fields() if f.type == "date"]
        f = next((x for x in dates if x.path.split(".")[-1].lower() in CREATED), None) or r.choice(dates)
        path = self.path(f)
        days = r.choice([7, 15, 30, 60, 90])
        created = f.path.split(".")[-1].lower() in CREATED
        if self.json:
            expr = {"$expr": {"$gte": ["$" + path, {"$dateSubtract": {"startDate": "$$NOW", "unit": "day", "amount": days}}]}}
            output = self.pipeline([{"$match": expr}])
            must = [[path], ["$dateSubtract", "$$NOW"], [str(days)]]
        else:
            value = {"$gte": Raw(f"new Date(Date.now() - {days} * 24 * 60 * 60 * 1000)")}
            output = self.chain(self.target(), [("find", [{path: value}])])
            must = [[path], ["$gte"], ["Date.now()"], [str(days)]]
        cp, ce = self.coll("pt"), self.coll("en")
        lp, le = self.label(f, "pt"), self.label(f, "en")
        if created:
            pt = r.choice(["liste {c} criados nos últimos {d} dias", "{c} criados nos últimos {d} dias", "quais {c} foram criados nos últimos {d} dias?",
                           "mostre {c} dos últimos {d} dias"]).format(c=cp, d=days)
            en = r.choice(["list {c} created in the last {d} days", "{c} created in the last {d} days", "which {c} were created in the last {d} days?",
                           "show {c} from the last {d} days"]).format(c=ce, d=days)
        else:
            pt = r.choice(["liste {c} com {f} nos últimos {d} dias", "{c} com {f} dentro dos últimos {d} dias"]).format(c=cp, f=lp, d=days)
            en = r.choice(["list {c} with {f} in the last {d} days", "{c} whose {f} is within the last {d} days"]).format(c=ce, f=le, d=days)
        return Scenario("create", "recent_days", "crud", pt, en, self.default_editor(), output, ["find", "aggregate"], must, self.name,
                        f"Busca {self.name} com {path} nos últimos {days} dias.", f"Finds {self.name} with {path} in the last {days} days.")

    def s_count(self) -> Scenario:
        r = self.rng
        flt, cpt, cen, must = self.filter(pick(r, {1: .7, 2: .3}))
        if self.json:
            output = self.pipeline([{"$match": flt}, {"$count": "total"}])
        else:
            output = self.chain(self.target(), [("countDocuments", [flt])])
        cp, ce = self.coll("pt"), self.coll("en")
        pt = r.choice(["quantos {c} têm {x}?", "conte {c} com {x}", "qual o total de {c} onde {x}?", "contar {c} onde {x}"]).format(c=cp, x=cpt)
        en = r.choice(["how many {c} have {x}?", "count {c} where {x}", "count the {c} with {x}", "total number of {c} where {x}"]).format(c=ce, x=cen)
        return Scenario("create", "count", "crud", pt, en, self.default_editor(), output, ["countDocuments", "aggregate"],
                        must + [["countDocuments", "$count"]], self.name, f"Conta {self.name} onde {cpt}.", f"Counts {self.name} where {cen}.")

    def group_field(self) -> S.Field:
        pool = [f for f in self.cond_fields() if f.type in ("enum", "code", "city", "state", "country", "ref")]
        preferred = [f for f in pool if self.label_raw(f, 0)]
        return self.rng.choice(preferred if preferred and self.rng.random() < .7 else pool)

    def s_group_count(self) -> Scenario:
        r = self.rng
        f = self.group_field()
        path = self.path(f)
        sort = r.random() < .4
        match = r.random() < .25
        stages: list[dict] = []
        must: list[list[str]] = []
        cpt = cen = ""
        if match:
            others = [x for x in self.cond_fields() if x is not f]
            value, cpt, cen, m = self.cond(r.choice(others))
            key = m[0][0]
            stages.append({"$match": {key: value}})
            must += m + [["$match"]]
        stages.append({"$group": {"_id": "$" + path, "count": {"$sum": 1}}})
        if sort:
            stages.append({"$sort": {"count": -1}})
        output = self.pipeline(stages)
        cp, ce = self.coll("pt"), self.coll("en")
        lp, le = self.label(f, "pt"), self.label(f, "en")
        pt = r.choice(["conte {c} por {f}", "agrupe {c} por {f} e conte", "quantos {c} por {f}?", "total de {c} por {f}"]).format(c=cp, f=lp)
        en = r.choice(["group {c} by {f} and count them", "count {c} per {f}", "how many {c} per {f}?", "count {c} grouped by {f}"]).format(c=ce, f=le)
        if match:
            pt, en = pt + f", só com {cpt}", en + f", only with {cen}"
        if sort:
            pt += r.choice([" ordenando do maior para o menor", ", do maior para o menor"])
            en += r.choice([" sorted by count descending", ", highest first"])
            must.append(["$sort"])
        return Scenario("create", "group_count", "aggregation", pt, en, self.default_editor(), output, ["aggregate"],
                        must + [["$group"], ["$" + path], ["$sum"]], self.name, f"Conta {self.name} agrupados por {path}.",
                        f"Counts {self.name} grouped by {path}.", stages=stages)

    def s_group_metric(self) -> Scenario:
        r = self.rng
        numeric = [f for f in self.cond_fields() if f.numeric]
        f = self.group_field()
        n = r.choice(numeric)
        op, out, wpt, wen = r.choice([("$sum", "total", "some {n} por {f}", "sum {n} by {f}"), ("$avg", "average", "média de {n} por {f}", "average {n} per {f}"),
                                      ("$max", "max", "maior {n} por {f}", "max {n} per {f}"), ("$min", "min", "menor {n} por {f}", "min {n} per {f}")])
        stages = [{"$group": {"_id": "$" + self.path(f), out: {op: "$" + self.path(n)}}}]
        output = self.pipeline(stages)
        pt = wpt.format(n=self.label(n, "pt"), f=self.label(f, "pt")) + f" em {self.coll('pt')}"
        en = wen.format(n=self.label(n, "en"), f=self.label(f, "en")) + f" in {self.coll('en')}"
        return Scenario("create", "group_metric", "aggregation", pt, en, self.default_editor(), output, ["aggregate"],
                        [["$group"], ["$" + self.path(f)], [op], ["$" + self.path(n)]], self.name,
                        f"Calcula {op} de {self.path(n)} por {self.path(f)} em {self.name}.", f"Computes {op} of {self.path(n)} per {self.path(f)} in {self.name}.",
                        stages=stages)

    def s_top_n(self) -> Scenario:
        r = self.rng
        n = r.choice([f for f in self.cond_fields() if f.numeric or f.type == "date"])
        limit = r.choice([3, 5, 10, 20])
        desc = r.random() < .75
        direction = -1 if desc else 1
        path = self.path(n)
        output = (self.pipeline([{"$sort": {path: direction}}, {"$limit": limit}]) if self.json
                  else self.chain(self.target(), [("find", [{}]), ("sort", [{path: direction}]), ("limit", [limit])]))
        cp, ce = self.coll("pt"), self.coll("en")
        lp, le = self.label(n, "pt"), self.label(n, "en")
        if desc:
            pt = r.choice(["os {k} {c} com maior {f}", "top {k} {c} por {f}", "liste os {k} {c} com {f} mais alto"]).format(k=limit, c=cp, f=lp)
            en = r.choice(["top {k} {c} by {f}", "the {k} {c} with the highest {f}", "list the {k} {c} with the largest {f}"]).format(k=limit, c=ce, f=le)
        else:
            pt = r.choice(["os {k} {c} com menor {f}", "liste os {k} {c} com {f} mais baixo"]).format(k=limit, c=cp, f=lp)
            en = r.choice(["the {k} {c} with the lowest {f}", "bottom {k} {c} by {f}"]).format(k=limit, c=ce, f=le)
        must = [[path], [f"{path}:{direction}", f'"{path}":{direction}'], [f"limit({limit})", f"$limit:{limit}", f'"$limit":{limit}']]
        return Scenario("create", "top_n", "crud", pt, en, self.default_editor(), output, ["find", "aggregate"], must, self.name,
                        f"Lista os {limit} {self.name} ordenados por {path}.", f"Lists {limit} {self.name} sorted by {path}.")

    def s_distinct(self) -> Scenario:
        r = self.rng
        f = self.group_field()
        path = self.path(f)
        output = self.pipeline([{"$group": {"_id": "$" + path}}]) if self.json else self.chain(self.target(), [("distinct", [path])])
        pt = r.choice(["quais os valores distintos de {f} em {c}?", "liste os {f} distintos de {c}", "valores únicos de {f} em {c}"]).format(f=self.label(f, "pt"), c=self.coll("pt"))
        en = r.choice(["distinct {f} values in {c}", "list the unique {f} values of {c}", "which {f} values exist in {c}?"]).format(f=self.label(f, "en"), c=self.coll("en"))
        return Scenario("create", "distinct", "crud", pt, en, self.default_editor(), output, ["distinct", "aggregate"],
                        [[path], ["distinct", "$group"]], self.name, f"Lista valores distintos de {path} em {self.name}.",
                        f"Lists distinct {path} values in {self.name}.")

    def s_lookup(self) -> Scenario:
        r = self.rng
        refs = [f for f in self.cond_fields() if f.type == "ref" and f.arg in self.g.colls]
        f = r.choice(refs)
        other = self.g.cname(f.arg)
        alias = self.g.out(re.sub(r"[^A-Za-z]", "", f.arg[:1].lower() + f.arg[1:]))
        stages: list[dict] = [{"$lookup": {"from": other, "localField": self.path(f), "foreignField": "_id", "as": alias}}]
        unwind = r.random() < .45
        if unwind:
            stages.append({"$unwind": "$" + alias})
        output = self.pipeline(stages)
        onoun = NOUNS.get(f.arg.lower(), (other, other))
        pt = r.choice(["junte {c} com {o} por {f}", "traga {c} com os dados de {o}", "faça um lookup de {o} em {c}"]).format(c=self.coll("pt"), o=other if not self.natural else onoun[0], f=self.path(f))
        en = r.choice(["join {c} with {o} on {f}", "lookup {o} into {c}", "bring {o} details into {c}"]).format(c=self.coll("en"), o=other if not self.natural else onoun[1], f=self.path(f))
        must = [["$lookup"], [other], [self.path(f)]]
        if unwind:
            pt += r.choice([" e desfaça o array", " com $unwind"])
            en += r.choice([" and unwind the result", " with $unwind"])
            must.append(["$unwind"])
        return Scenario("create", "lookup", "aggregation", pt, en, self.default_editor(), output, ["aggregate"], must, self.name,
                        f"Junta {self.name} com {other} por {self.path(f)}.", f"Joins {self.name} with {other} on {self.path(f)}.", stages=stages)

    def s_search(self) -> Scenario:
        r = self.rng
        texts = [f for f in self.cond_fields() if f.type in ("text", "name")] or [f for f in self.cond_fields() if f.type in ("enum", "word", "code", "city")]
        f = r.choice(texts)
        path = self.path(f)
        term = self.g.text_term(f)
        index = r.choice(S.DOMAINS[self.domain]["search_index"])
        auto = r.random() < .3
        say_index = r.random() < .5 or index != "default"
        operator = "autocomplete" if auto else "text"
        query = term[: r.randint(2, 4)] if auto else term
        body: dict[str, Any] = {"index": index} if say_index else {}
        body[operator] = {"query": query, "path": path}
        fuzzy = not auto and r.random() < .25
        if fuzzy:
            body[operator]["fuzzy"] = {"maxEdits": 1}
        limit = r.choice([5, 10, 20]) if r.random() < .6 else 0
        stages: list[dict] = [{"$search": body}] + ([{"$limit": limit}] if limit else [])
        output = self.pipeline(stages)
        idx_pt, idx_en = (f" no índice {index}", f" on the {index} index") if say_index else ("", "")
        if auto:
            pt = f"autocomplete de {self.label(f, 'pt')} começando com \"{query}\" em {self.coll('pt')} usando Atlas Search{idx_pt}"
            en = f"Atlas Search autocomplete on {self.label(f, 'en')} for \"{query}\" in {self.coll('en')}{idx_en}"
        else:
            pt = r.choice(["busque {c} com \"{t}\" em {f} usando Atlas Search", "pesquisa textual por \"{t}\" no campo {f} de {c}",
                           "Atlas Search: {c} contendo {t} em {f}"]).format(c=self.coll("pt"), t=query, f=self.label(f, "pt")) + idx_pt
            en = r.choice(["search {c} for \"{t}\" in {f} with Atlas Search", "full-text search for \"{t}\" on {f} in {c}",
                           "Atlas Search: {c} matching {t} in {f}"]).format(c=self.coll("en"), t=query, f=self.label(f, "en")) + idx_en
            if fuzzy:
                pt, en = pt + ", tolerando erro de digitação", en + ", allowing typos"
        must = [["$search"], [query], [path], [operator]] + ([[index]] if say_index else []) + ([["fuzzy"]] if fuzzy else [])
        if limit:
            lp, le, alts = self.limit_phrase(limit)
            pt, en, must = pt + lp, en + le, must + [alts]
        return Scenario("create", "search", "atlas_search", pt, en, self.default_editor(), output, ["aggregate"], must, self.name,
                        f"Busca textual Atlas Search em {path} de {self.name}.", f"Atlas Search query on {path} of {self.name}.", stages=stages)

    def s_index(self) -> Scenario:
        r = self.rng
        kind = pick(r, {"single": .45, "unique": .2, "ttl": .2, "compound": .15})
        fields = self.cond_fields()
        if kind == "unique":
            f = r.choice([x for x in fields if x.type in ("email", "code")])
            args: list = [{self.path(f): 1}, {"unique": True}]
            pt, en = f"crie um índice único em {self.label(f, 'pt')}", f"create a unique index on {self.label(f, 'en')}"
            must = [["createIndex"], [self.path(f)], ["unique"]]
        elif kind == "ttl":
            f = r.choice([x for x in fields if x.type == "date"])
            days = r.choice([1, 7, 30, 90])
            args = [{self.path(f): 1}, {"expireAfterSeconds": days * 86400}]
            pt, en = f"índice TTL em {self.label(f, 'pt')} expirando em {days} dias", f"TTL index on {self.label(f, 'en')} expiring after {days} days"
            must = [["createIndex"], [self.path(f)], ["expireAfterSeconds"], [str(days * 86400)]]
        elif kind == "compound":
            a, b = r.sample([x for x in fields if x.type in ("enum", "date", "int", "money", "code", "ref", "bool")], 2)
            db_ = r.choice([1, -1])
            args = [{self.path(a): 1, self.path(b): db_}]
            order_pt, order_en = ("decrescente", "descending") if db_ == -1 else ("crescente", "ascending")
            pt = f"crie um índice composto em {self.label(a, 'pt')} e {self.label(b, 'pt')} ({order_pt})"
            en = f"create a compound index on {self.label(a, 'en')} and {self.label(b, 'en')} ({order_en})"
            must = [["createIndex"], [self.path(a)], [self.path(b)]]
        else:
            f = r.choice([x for x in fields if x.type in ("enum", "date", "int", "money", "code", "ref", "email", "name")])
            direction = r.choice([1, -1])
            args = [{self.path(f): direction}]
            word = ("decrescente", "descending") if direction == -1 else ("crescente", "ascending")
            pt = r.choice([f"crie um índice em {self.label(f, 'pt')} {word[0]}", f"criar índice {word[0]} para {self.label(f, 'pt')}"])
            en = r.choice([f"create a {word[1]} index on {self.label(f, 'en')}", f"add an index on {self.label(f, 'en')} ({word[1]})"])
            must = [["createIndex"], [self.path(f)], [f"{self.path(f)}:{direction}", f'"{self.path(f)}":{direction}']]
        pt, en = pt + f" em {self.coll('pt')}", en + f" on {self.coll('en')}"
        output = self.chain(self.target(), [("createIndex", args)])
        return Scenario("create", "index", "indexes", pt, en, self.default_editor(), output, ["createIndex"], must, self.name,
                        f"Cria índice em {self.name}.", f"Creates an index on {self.name}.")

    def s_update(self) -> Scenario:
        r = self.rng
        flt, cpt, cen, must = self.filter(1)
        targets = [f for f in self.cond_fields() if f.type in ("enum", "bool", "int", "money") and self.path(f) not in flt]
        f = r.choice(targets)
        value, _, _, _ = self.cond(f)
        if isinstance(value, dict):
            value = next(iter(value.values()))
            value = value[0] if isinstance(value, list) else value
        text = render_inline(value, JS_STYLE).strip('"')
        many = r.random() < .75
        output = self.chain(self.target(), [("updateMany" if many else "updateOne", [flt, {"$set": {self.path(f): value}}])])
        cp, ce = self.coll("pt"), self.coll("en")
        if many:
            pt = r.choice(["atualize {c} onde {x}: defina {f} como {v}", "em {c} com {x}, altere {f} para {v}"]).format(c=cp, x=cpt, f=self.label(f, "pt"), v=text)
            en = r.choice(["set {f} to {v} for {c} where {x}", "update {c} with {x}: set {f} = {v}"]).format(c=ce, x=cen, f=self.label(f, "en"), v=text)
        else:
            pt = f"atualize um documento de {cp} onde {cpt}: {self.label(f, 'pt')} = {text}"
            en = f"update one {ce} document where {cen}: {self.label(f, 'en')} = {text}"
        return Scenario("create", "update", "crud", pt, en, self.default_editor(), output, ["updateMany", "updateOne"],
                        must + [["$set"], [self.path(f)], ["updateMany" if many else "updateOne"]], self.name,
                        f"Atualiza {self.name} onde {cpt}.", f"Updates {self.name} where {cen}.")

    def s_delete(self) -> Scenario:
        r = self.rng
        dates = [f for f in self.cond_fields() if f.type == "date"]
        if dates and r.random() < .6:
            f = r.choice(dates)
            days = r.choice([30, 90, 180, 365])
            flt = {self.path(f): {"$lt": Raw(f"new Date(Date.now() - {days} * 24 * 60 * 60 * 1000)")}}
            pt = r.choice(["remova {c} com {f} anterior a {d} dias", "apague {c} mais antigos que {d} dias (por {f})"]).format(c=self.coll("pt"), f=self.label(f, "pt"), d=days)
            en = r.choice(["delete {c} with {f} older than {d} days", "remove {c} older than {d} days based on {f}"]).format(c=self.coll("en"), f=self.label(f, "en"), d=days)
            must = [[self.path(f)], ["$lt", "$lte"], [str(days)]]
            cpt, cen = f"{self.path(f)} anterior a {days} dias", f"{self.path(f)} older than {days} days"
        else:
            flt, cpt, cen, must = self.filter(1)
            pt = r.choice(["remova {c} onde {x}", "apague os {c} com {x}"]).format(c=self.coll("pt"), x=cpt)
            en = r.choice(["delete {c} where {x}", "remove the {c} with {x}"]).format(c=self.coll("en"), x=cen)
        output = self.chain(self.target(), [("deleteMany", [flt])])
        return Scenario("create", "delete", "crud", pt, en, self.default_editor(), output, ["deleteMany", "deleteOne"], must + [["deleteMany", "deleteOne"]],
                        self.name, f"Remove {self.name} onde {cpt}.", f"Deletes {self.name} where {cen}.")

    def s_insert(self) -> Scenario:
        r = self.rng
        fields = r.sample([f for f in self.cond_fields() if f.type in ("enum", "code", "name", "text", "int", "money", "bool", "city", "email")],
                          k=r.randint(2, 3))
        doc: dict[str, Any] = {}
        pts, ens, must = [], [], [["insertOne"]]
        for f in fields:
            value = self.g.value(f)
            if isinstance(value, float) and not isinstance(value, Money):
                value = Money(round(value, 2))
            text = render_inline(value, JS_STYLE).strip('"')
            node = doc
            parts = self.path(f).split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
            pts.append(f"{self.path(f)} = {text}")
            ens.append(f"{self.path(f)} = {text}")
            must += [[parts[-1]], [text]]
        output = self.chain(self.target(), [("insertOne", [doc])])
        pt = r.choice(["insira um documento em {c} com {x}", "adicione em {c} um registro com {x}"]).format(c=self.coll("pt"), x=", ".join(pts))
        en = r.choice(["insert a document into {c} with {x}", "add a record to {c} with {x}"]).format(c=self.coll("en"), x=", ".join(ens))
        return Scenario("create", "insert", "crud", pt, en, self.default_editor(), output, ["insertOne", "insertMany"], must, self.name,
                        f"Insere um documento em {self.name}.", f"Inserts a document into {self.name}.")

    def s_dsl_connection(self) -> Scenario:
        r = self.rng
        conn, database = self.g.connection, self.g.database
        pool = r.random() < .4
        ident = all(IDENT.match(s) for s in (conn, database, self.name))
        if pool:
            target = ["ConnectionPool." + ".".join((conn, database, self.name))] if ident else \
                ["ConnectionPool" + "".join(f"[{self.q(s)}]" for s in (conn, database, self.name))]
        else:
            target = [f"getConnection({self.q(conn)})", f".getDatabase({self.q(database)})", f".getCollection({self.q(self.name)})"]
        count = r.random() < .3
        flt, cpt, cen, must = self.filter(1) if r.random() < .7 else ({}, "", "", [])
        calls = [("countDocuments", [flt])] if count else [("find", [flt])] + ([("limit", [r.choice([10, 20, 50])])] if r.random() < .5 else [])
        output = self.chain(target, calls)
        via_pt, via_en = (" via ConnectionPool", " using ConnectionPool") if pool else ("", "")
        verb_pt, verb_en = ("conte", "count") if count else ("liste", "list")
        pt = r.choice(["{v} {c} na conexão {k}, banco {d}{p}", "use a conexão {k} e o banco {d} para {v} {c}{p}"]).format(v=verb_pt, c=self.coll("pt"), k=conn, d=database, p=via_pt)
        en = r.choice(["{v} {c} on connection {k}, database {d}{p}", "use connection {k} and database {d} to {v} {c}{p}"]).format(v=verb_en, c=self.coll("en"), k=conn, d=database, p=via_en)
        if cpt:
            pt, en = pt + f" com {cpt}", en + f" with {cen}"
        conn_alts = [f'getConnection("{conn}")', f"ConnectionPool.{conn}", f'ConnectionPool["{conn}"]']
        must = must + [conn_alts, [database], [self.name], ["ConnectionPool"] if pool else ["getConnection"]]
        if pool:
            must[-1] = ["ConnectionPool"]
        if limit := next((a[0] for m, a in calls if m == "limit"), None):
            lp, le, alts = self.limit_phrase(limit)
            pt, en, must = pt + lp, en + le, must + [alts]
        return Scenario("create", "dsl_connection", "slop_dsl", pt, en, self.default_editor(), output, ["find", "countDocuments"], must, self.name,
                        f"Consulta {self.name} em {conn}/{database}.", f"Queries {self.name} on {conn}/{database}.")

    def s_ejson(self) -> Scenario:
        r = self.rng
        oid = S.oid(r)
        by_id = r.random() < .55
        dates = [f for f in self.cond_fields() if f.type == "date"]
        if by_id or not dates:
            body: dict[str, Any] = {"_id": {"$oid": oid.hex}}
            pt = f"busque o documento de {self.coll('pt')} com _id {oid.hex} usando Extended JSON"
            en = f"find the {self.coll('en')} document with _id {oid.hex} using Extended JSON"
            must = [["$oid"], [oid.hex], ["_id"]]
        else:
            f = r.choice(dates)
            day = (S.EPOCH + timedelta(days=r.randint(0, 900))).strftime("%Y-%m-%d")
            body = {self.path(f): {"$gte": {"$date": day + "T00:00:00Z"}}}
            pt = f"filtre {self.coll('pt')} com {self.label(f, 'pt')} a partir de {day} em Extended JSON"
            en = f"filter {self.coll('en')} with {self.label(f, 'en')} on or after {day} in Extended JSON"
            must = [["$date"], [day], [self.path(f)], ["$gte"]]
        if self.json:
            output = self.pipeline([{"$match": body}])
        else:
            import json as _json

            text = _json.dumps(body, ensure_ascii=False)
            output = self.chain(self.target(), [("find", [Raw(f"EJSON.parse('{text}')")])])
            must.append(["EJSON.parse"])
        return Scenario("create", "ejson", "json_ejson", pt, en, self.default_editor(), output, ["find", "aggregate"], must, self.name,
                        f"Filtra {self.name} com Extended JSON.", f"Filters {self.name} with Extended JSON.")

    # -- famílias adicionais (v1.1): direção temporal, eventos com verbo natural, condição combinada e top N por contagem --------------
    def date_event(self, f: S.Field) -> tuple[str, str, str] | None:
        return DATE_EVENTS.get(f.path.split(".")[-1].lower().replace("_", ""))

    def s_temporal(self) -> Scenario:
        """"mais de N dias atrás", "older than", "sem atualização desde <data>", "since <data>" → $lt/$gte com a direção certa."""
        r = self.rng
        dates = [f for f in self.cond_fields() if f.type == "date"]
        f = r.choice([x for x in dates if self.date_event(x)] or dates)
        path = self.path(f)
        real_event = self.date_event(f)
        event = real_event or (f"com {self.label(f, 'pt')}", f"with {self.label(f, 'en')}", f"have {self.label(f, 'en')}")
        # Formas negativas só existem para eventos com verbo ("que fizeram login" → "que não fizeram login"; "that logged in" → "that have not logged in").
        if real_event:
            neg_pt = "que não " + event[0][4:] if event[0].startswith("que ") else "não " + event[0]
            neg_en = f"that have not {event[2]}" if event[1].startswith("that ") else f"not {event[1]}"
        cp, ce = self.coll("pt"), self.coll("en")
        kind = pick(r, {"older": .35, "recent": .25, "not_since": .2, "since": .2})
        if kind in ("older", "recent"):
            days = r.choice([7, 14, 30, 60, 90, 180, 365])
            op = "$lt" if kind == "older" else "$gte"
            span_pt = "um ano" if days == 365 else f"{days} dias"
            span_en = "one year" if days == 365 else f"{days} days"
            if self.json:
                value: Any = {"$expr": {"$lt" if kind == "older" else "$gte": ["$" + path, {"$dateSubtract": {"startDate": "$$NOW", "unit": "day", "amount": days}}]}}
                condition, must_value = value, [["$dateSubtract", "$$NOW"], [str(days)]]
            else:
                condition = {path: {op: Raw(f"new Date(Date.now() - {days} * 24 * 60 * 60 * 1000)")}}
                must_value = [{"$lt": ["$lt", "$lte"], "$gte": ["$gte", "$gt"]}[op], [str(days)]]
            if kind == "older":
                pt = r.choice([f"{cp} {event[0]} há mais de {span_pt}", f"{cp} mais antigos que {span_pt} (por {self.label(f, 'pt')})"])
                en = r.choice([f"{ce} {event[1]} more than {span_en} ago", f"{ce} older than {span_en} by {self.label(f, 'en')}", f"{ce} {event[1]} over {span_en} ago"])
            else:
                pt = r.choice([f"{cp} {event[0]} nos últimos {span_pt}", f"{cp} {event[0]} há menos de {span_pt}"])
                en = r.choice([f"{ce} {event[1]} in the last {span_en}", f"{ce} {event[1]} within the past {span_en}", f"{ce} that {event[2]} during the last {span_en}"])
        else:
            day = (S.EPOCH + timedelta(days=r.randint(0, 900))).strftime("%Y-%m-%d")
            op = "$lt" if kind == "not_since" else "$gte"
            condition = {path: {op: DateV(day + "T00:00:00Z")}}
            must_value = [[op], [day]]
            if kind == "not_since":
                pt = r.choice([f"{cp} com {self.label(f, 'pt')} anterior a {day}"] + ([f"{cp} {neg_pt} desde {day}"] * 2 if real_event else []))
                en = r.choice([f"{ce} whose {self.label(f, 'en')} is before {day}"] + ([f"{ce} {neg_en} since {day}"] * 2 if real_event else []))
            else:
                pt = r.choice([f"{cp} {event[0]} desde {day}", f"{cp} com {self.label(f, 'pt')} a partir de {day}"])
                en = r.choice([f"{ce} {event[1]} since {day}", f"{ce} with {self.label(f, 'en')} on or after {day}"])
        verb = pick(r, {"find": .5, "count": .3, "delete": 0 if self.json else .2})
        must = [[path]] + must_value
        extra: dict[str, Any] = {}
        enums = [x for x in self.cond_fields() if x.type == "enum"]
        if enums and r.random() < .45:  # condição combinada: status + data
            e = r.choice(enums)
            value = str(self.g.value(e))
            extra = {self.path(e): value}
            pt, en = pt + f" com {self.label(e, 'pt')} {value}", en + f" with {self.label(e, 'en')} {value}"
            must += [[self.path(e)], [value]]
        flt = {**extra, **condition} if not self.json else ({"$match": {**extra, **condition}})
        if self.json:
            stages: list[dict] = [flt] + ([{"$count": "total"}] if verb == "count" else [])
            output = self.pipeline(stages)
            operations = ["aggregate"]
            if verb == "count":
                pt, en = "quantos " + pt, "count " + en
                must.append(["$count"])
        elif verb == "count":
            output = self.chain(self.target(), [("countDocuments", [flt])])
            pt, en = r.choice(["quantos ", "conte os "]) + pt + ("?" if r.random() < .5 else ""), r.choice(["how many ", "count "]) + en
            operations, must = ["countDocuments", "aggregate"], must + [["countDocuments", "$count"]]
        elif verb == "delete":
            output = self.chain(self.target(), [("deleteMany", [flt])])
            pt, en = r.choice(["remova ", "apague "]) + pt, r.choice(["delete ", "remove "]) + en
            operations, must = ["deleteMany", "deleteOne"], must + [["deleteMany", "deleteOne"]]
        else:
            output = self.chain(self.target(), [("find", [flt])])
            pt, en = r.choice(["liste ", "mostre ", ""]) + pt, r.choice(["list ", "show ", "find "]) + en
            operations = ["find", "aggregate"]
        return Scenario("create", "temporal_" + kind, "crud", pt, en, self.default_editor(), output, operations, must, self.name,
                        f"Consulta temporal em {self.name} por {path}.", f"Temporal query on {self.name} by {path}.")

    def s_top_groups(self) -> Scenario:
        """"top N <entidade do grupo> por número/total de <coleção>" → $group + $sort + $limit na coleção atual."""
        r = self.rng
        f = self.group_field()
        path = self.path(f)
        limit = r.choice([3, 5, 10])
        metric = r.random() < .3 and [x for x in self.cond_fields() if x.numeric]
        if metric:
            n = r.choice(metric)
            group = {"_id": "$" + path, "total": {"$sum": "$" + self.path(n)}}
            key = "total"
            pt = f"top {limit} {self.label(f, 'pt')} por total de {self.label(n, 'pt')} em {self.coll('pt')}"
            en = f"top {limit} {self.label(f, 'en')} by total {self.label(n, 'en')} across {self.coll('en')}"
            must = [["$group"], ["$" + path], ["$sum"], ["$" + self.path(n)], ["$sort"], [f"$limit:{limit}", f'"$limit":{limit}']]
        else:
            group = {"_id": "$" + path, "count": {"$sum": 1}}
            key = "count"
            pt = r.choice([f"top {limit} {self.label(f, 'pt')} por número de {self.coll('pt')}", f"os {limit} {self.label(f, 'pt')} com mais {self.coll('pt')}"])
            en = r.choice([f"top {limit} {self.label(f, 'en')} by number of {self.coll('en')}", f"the {limit} {self.label(f, 'en')} with the most {self.coll('en')}"])
            must = [["$group"], ["$" + path], ["$sum", "$count"], ["$sort"], [f"$limit:{limit}", f'"$limit":{limit}']]
        output = self.pipeline([{"$group": group}, {"$sort": {key: -1}}, {"$limit": limit}])
        return Scenario("create", "top_groups", "aggregation", pt, en, self.default_editor(), output, ["aggregate"], must, self.name,
                        f"Top {limit} {path} em {self.name}.", f"Top {limit} {path} in {self.name}.")

    # -- famílias v1.2: linguagem livre (condições naturais, adjetivos, nulos, ordenação natural, verbos de update, índice por lentidão) ----
    def natural_cond(self, f: S.Field) -> tuple[Any, str, str, str | None, list[list[str]]] | None:
        """(valor, modificador PT, modificador EN pós-substantivo, adjetivo EN pré-substantivo ou None, obrigatórios)."""
        r, path = self.rng, self.path(f)
        lp, le = self.label(f, "pt"), self.label(f, "en")
        leaf = f.path.split(".")[-1].lower().replace("_", "")
        if f.type == "bool" and leaf in BOOL_WORDS:
            value = r.random() < .6
            (pt_true, pt_false), (en_pos, en_true, en_false) = BOOL_WORDS[leaf]
            pt, en = (pt_true, en_true) if value else (pt_false, en_false)
            return value, pt, en if en_pos == "post" else "", en if en_pos == "pre" else None, [[path], ["true" if value else "false"]]
        if f.numeric and f.type in ("int", "money"):
            lo, hi = f.range()
            raw = r.uniform(lo, hi)
            step = 100 if hi - lo >= 5000 else 10 if hi - lo >= 100 else 1
            base = max(int(lo), int(round(raw / step) * step))
            value_num: Any = base if f.type == "int" else Money(float(base))
            text = number(value_num)
            op = pick(r, {"$gt": .35, "$gte": .15, "$lt": .35, "$lte": .15})
            words = {"$gt": (["acima de", "mais de", "maior que"], ["over", "more than", "above"]),
                     "$gte": (["pelo menos", "no mínimo"], ["at least", "no less than"]),
                     "$lt": (["abaixo de", "menos de", "menor que"], ["under", "below", "less than"]),
                     "$lte": (["no máximo", "até"], ["at most", "up to"])}[op]
            alts = {"$gt": ["$gt", "$gte"], "$gte": ["$gte", "$gt"], "$lt": ["$lt", "$lte"], "$lte": ["$lte", "$lt"]}[op]
            return ({op: value_num}, f"com {lp} {r.choice(words[0])} {text}", f"with {le} {r.choice(words[1])} {text}", None,
                    [[path], alts, [text]])
        if f.type == "date":
            event = self.date_event(f)
            # Criação/atualização existem em todo documento: "ainda não foram criados" não faz sentido.
            if not event or leaf in ("createdat", "createdon", "updatedat", "updatedon"):
                return None
            pending = r.random() < .6
            if pending:
                pt = "que ainda não " + event[0][4:] if event[0].startswith("que ") else "que ainda não foram " + event[0]
                en = f"that have never {event[2]}" if event[1].startswith("that ") else f"that haven't been {event[1]} yet"
                return None, pt, en, None, [[path], ["null", "$exists"]]
            pt = "que já " + event[0][4:] if event[0].startswith("que ") else "já " + event[0]
            en = f"that have already {event[2]}" if event[1].startswith("that ") else f"already {event[1]}"
            return {"$ne": None}, pt, en, None, [[path], ["$ne", "$exists"]]
        if f.type == "enum":
            value = str(self.g.value(f))
            if leaf in ADJECTIVE_ENUMS and le != path:
                return value, f"com {lp} {value}", "", f"{value}-{le}", [[path], [value]]
            return value, f"com {lp} {value}", f"with {le} {value}", None, [[path], [value]]
        return None

    def natural_sort(self) -> tuple[dict, str, str, list[list[str]]]:
        r = self.rng
        candidates = [f for f in self.cond_fields() if f.type in ("money", "int", "date")]
        f = r.choice(candidates)
        path, lp, le = self.path(f), self.label(f, "pt"), self.label(f, "en")
        leaf = f.path.split(".")[-1].lower().replace("_", "")
        direction = r.choice([1, -1])
        if f.type == "money" and leaf in PRICE_LEAVES:
            pt = r.choice(["do mais caro pro mais barato", "mais caros primeiro"]) if direction == -1 else r.choice(["do mais barato pro mais caro", "mais baratos primeiro"])
            en = r.choice(["most expensive first", "priciest first"]) if direction == -1 else r.choice(["cheapest first", "least expensive first"])
        elif f.type == "date":
            recent = f.path.split(".")[-1].lower() in CREATED
            pt = (r.choice(["mais recentes primeiro", "dos mais novos pros mais antigos"]) if recent else f"por {lp}, mais recentes primeiro") if direction == -1 \
                else (r.choice(["mais antigos primeiro", "dos mais antigos pros mais novos"]) if recent else f"por {lp}, mais antigos primeiro")
            en = (r.choice(["newest first", "most recent first"]) if recent else f"by {le}, most recent first") if direction == -1 \
                else (r.choice(["oldest first", "earliest first"]) if recent else f"by {le}, oldest first")
        else:
            pt = r.choice([f"do maior pro menor {lp}", f"com maior {lp} primeiro"]) if direction == -1 else r.choice([f"do menor pro maior {lp}", f"com menor {lp} primeiro"])
            en = r.choice([f"highest {le} first", f"sorted by {le}, highest first"]) if direction == -1 else r.choice([f"lowest {le} first", f"sorted by {le}, lowest first"])
        return {path: direction}, pt, en, [[f"{path}:{direction}", f'"{path}":{direction}']]

    def natural_filter(self, n: int) -> tuple[dict, str, str, list[list[str]]]:
        r = self.rng
        fields = [f for f in self.cond_fields()]
        r.shuffle(fields)
        flt: dict[str, Any] = {}
        pt_mods, en_post, en_pre, must = [], [], [], []
        for f in fields:
            if len(flt) >= n:
                break
            got = self.natural_cond(f)
            if not got:
                continue
            value, pt, en, pre, m = got
            flt[self.path(f)] = value
            pt_mods.append(pt)
            if pre:
                en_pre.append(pre)
            if en:
                en_post.append(en)
            must += m
        if not flt:
            raise ValueError("coleção sem condições naturais")
        cp, ce = self.coll("pt"), self.coll("en")
        pt = f"{cp} " + " e ".join(agree_pt(m, cp) for m in pt_mods)
        en = " ".join(en_pre + [ce]) + ((" " + " and ".join(en_post)) if en_post else "")
        return flt, pt, en, must

    def s_natural_filter(self) -> Scenario:
        r = self.rng
        flt, pt, en, must = self.natural_filter(pick(r, {1: .45, 2: .45, 3: .1}))
        sort, spt, sen, smust = self.natural_sort() if r.random() < .4 else ({}, "", "", [])
        limit = r.choice([5, 10, 20]) if r.random() < .3 else 0
        if self.json:
            output = self.pipeline([{"$match": flt}] + ([{"$sort": sort}] if sort else []) + ([{"$limit": limit}] if limit else []))
        else:
            output = self.chain(self.target(), [("find", [flt])] + ([("sort", [sort])] if sort else []) + ([("limit", [limit])] if limit else []))
        if sort:
            pt, en, must = f"{pt}, {agree_pt(spt, self.coll('pt'))}", f"{en}, {sen}", must + smust
        if limit:
            lp, le, alts = self.limit_phrase(limit)
            pt, en, must = pt + lp, en + le, must + [alts]
        pt = r.choice(["", "me passa ", "preciso dos ", "lista ", "quero ver ", "traz "]) + pt
        en = r.choice(["", "gimme ", "I need the ", "list ", "show ", "get me "]) + en
        return Scenario("create", "natural_filter", "crud", pt, en, self.default_editor(), output, ["find", "aggregate"], must, self.name,
                        f"Busca {self.name} com condições em linguagem natural.", f"Finds {self.name} with natural-language conditions.")

    def s_set_flag(self) -> Scenario:
        r = self.rng
        flags = [f for f in self.cond_fields() if f.type == "bool" and f.path.split(".")[-1].lower() == "active"]
        f = r.choice(flags)
        path = self.path(f)
        value = r.random() < .35
        others = [x for x in self.cond_fields() if x is not f]
        got = None
        for candidate in r.sample(others, len(others)):
            got = self.natural_cond(candidate)
            if got:
                break
        flt = {self.path(candidate): got[0]} if got else {}
        cp, ce = self.coll("pt"), self.coll("en")
        cond_pt, cond_en = (agree_pt(got[1], cp), (got[3] + " " if got[3] else "") + "{c}" + (" " + got[2] if got[2] else "")) if got else ("", "{c}")
        verb_pt = r.choice(["ative", "reative"]) if value else r.choice(["desative", "inative"])
        verb_en = r.choice(["activate", "re-enable"]) if value else r.choice(["deactivate", "disable"])
        pt = f"{verb_pt} {'os ' if r.random() < .5 else ''}{cp}" + (f" {cond_pt}" if cond_pt else "")
        en = f"{verb_en} " + cond_en.format(c=ce)
        output = self.chain(self.target(), [("updateMany", [flt, {"$set": {path: value}}])])
        must = [["updateMany", "updateOne"], ["$set"], [path], ["true" if value else "false"]] + (got[4] if got else [])
        return Scenario("create", "set_flag", "crud", pt, en, self.default_editor(), output, ["updateMany", "updateOne"], must, self.name,
                        f"{'Ativa' if value else 'Desativa'} {self.name}.", f"{'Activates' if value else 'Deactivates'} {self.name}.")

    def s_percent_change(self) -> Scenario:
        r = self.rng
        f = r.choice([x for x in self.cond_fields() if x.type == "money"])
        path, lp, le = self.path(f), self.label(f, "pt"), self.label(f, "en")
        pct = r.choice([5, 10, 15, 20])
        up = r.random() < .6
        factor = f"{1 + pct / 100 if up else 1 - pct / 100:.2f}".rstrip("0").rstrip(".")
        flt, cpt, cen, fmust = self.filter(1) if r.random() < .6 else ({}, "", "", [])
        cp, ce = self.coll("pt"), self.coll("en")
        if up:
            pt = r.choice([f"aumente {lp} em {pct}%", f"dá um aumento de {pct}% em {lp}", f"reajuste {lp} em +{pct}%"])
            en = r.choice([f"increase {le} by {pct}%", f"raise {le} by {pct}%", f"bump {le} up {pct}%"])
        else:
            pt = r.choice([f"reduza {lp} em {pct}%", f"aplique {pct}% de desconto em {lp}", f"diminua {lp} em {pct}%"])
            en = r.choice([f"reduce {le} by {pct}%", f"cut {le} by {pct}%", f"apply a {pct}% discount to {le}"])
        pt += f" de {cp} onde {cpt}" if flt else f" de todos os {cp}"
        en += f" for {ce} where {cen}" if flt else f" for all {ce}"
        output = self.chain(self.target(), [("updateMany", [flt, {"$mul": {path: Raw(factor)}}])])
        must = [["updateMany"], ["$mul"], [path], [factor]] + fmust
        return Scenario("create", "percent_change", "crud", pt, en, self.default_editor(), output, ["updateMany"], must, self.name,
                        f"Altera {path} em {pct}%.", f"Changes {path} by {pct}%.")

    def s_latest_n(self) -> Scenario:
        r = self.rng
        f = next((x for x in self.cond_fields() if x.type == "date" and x.path.split(".")[-1].lower() in CREATED), None)
        if f is None:
            raise ValueError("coleção sem data de criação")
        path, n = self.path(f), r.choice([3, 5, 10, 20])
        cp, ce = self.coll("pt"), self.coll("en")
        people = self.name.lower() in PEOPLE_COLLECTIONS
        pt = r.choice([f"os últimos {n} {cp} cadastrados", f"quem foram os últimos {n} {cp} a se cadastrar?"] if people else
                      [f"os {n} {cp} mais recentes", f"últimos {n} {cp} criados"])
        en = r.choice([f"who are the last {n} {ce} to sign up?", f"the {n} most recently registered {ce}"] if people else
                      [f"the {n} newest {ce}", f"latest {n} {ce}"])
        if self.json:
            output = self.pipeline([{"$sort": {path: -1}}, {"$limit": n}])
        else:
            output = self.chain(self.target(), [("find", [{}]), ("sort", [{path: -1}]), ("limit", [n])])
        must = [[f"{path}:-1", f'"{path}":-1'], [f"limit({n})", f"$limit:{n}", f'"$limit":{n}']]
        return Scenario("create", "latest_n", "crud", pt, en, self.default_editor(), output, ["find", "aggregate"], must, self.name,
                        f"Últimos {n} {self.name} por {path}.", f"Latest {n} {self.name} by {path}.")

    def s_index_speed(self) -> Scenario:
        r = self.rng
        pool = [x for x in self.cond_fields() if x.type in ("enum", "date", "ref", "code", "int", "money", "bool")]
        fields = r.sample(pool, r.choice([1, 2]) if len(pool) > 1 else 1)
        spec = {self.path(x): (-1 if x.type == "date" else 1) for x in fields}
        names_pt = " e ".join(self.label(x, "pt") for x in fields)
        names_en = " and ".join(self.label(x, "en") for x in fields)
        pt = r.choice([f"deixa a busca por {names_pt} mais rápida", f"as consultas por {names_pt} estão lentas, resolve",
                       f"acelere as consultas que filtram por {names_pt}"])
        en = r.choice([f"make lookups by {names_en} faster", f"queries by {names_en} are slow, fix that",
                       f"speed up queries that filter on {names_en}"])
        output = self.chain(self.target(), [("createIndex", [spec])])
        return Scenario("create", "index_speed", "indexes", pt, en, self.default_editor(), output, ["createIndex"],
                        [["createIndex"]] + [[self.path(x)] for x in fields], self.name, f"Índice em {self.name}.", f"Index on {self.name}.")

    CREATE_WEIGHTS = {"find_filter": .14, "recent_days": .08, "count": .07, "group_count": .1, "group_metric": .06, "top_n": .06, "distinct": .03,
                      "lookup": .05, "search": .11, "index": .06, "update": .04, "delete": .03, "insert": .03, "dsl_connection": .07, "ejson": .07}
    # Pesos da rodada de reforço (v1.1): as famílias novas ganham peso, as originais continuam presentes para não esquecer.
    CREATE_WEIGHTS_V11 = {**{k: v * .5 for k, v in CREATE_WEIGHTS.items()}, "temporal": .3, "top_groups": .2}
    # v1.2: linguagem livre; famílias originais e v1.1 continuam presentes com peso menor.
    CREATE_WEIGHTS_V12 = {**{k: v * .4 for k, v in CREATE_WEIGHTS.items()}, "temporal": .1, "top_groups": .06, "natural_filter": .24,
                          "set_flag": .05, "percent_change": .04, "latest_n": .05, "index_speed": .04}
    JSON_ONLY_UNSUPPORTED = {"index", "update", "delete", "insert", "dsl_connection", "set_flag", "percent_change", "index_speed"}

    def create(self) -> Scenario:
        version = self.cfg.get("create_weights")
        weights = dict(self.CREATE_WEIGHTS_V12 if version == "v1.2" else self.CREATE_WEIGHTS_V11 if version == "v1.1" else self.CREATE_WEIGHTS)
        if self.json:
            weights = {k: v for k, v in weights.items() if k not in self.JSON_ONLY_UNSUPPORTED}
        if self.mode != "console":
            weights.pop("dsl_connection", None)
        for _ in range(12):
            name = pick(self.rng, weights)
            try:
                return getattr(self, "s_" + name)()
            except (IndexError, ValueError, KeyError):
                continue
        raise ValueError("nenhuma intenção de criação aplicável")

    # -- intenções: alteração ------------------------------------------------------------------------------------------------------------------
    def modify(self) -> Scenario:
        r = self.rng
        kinds = {"add_filter": .3, "add_sort_limit": .22, "add_projection": .12, "to_aggregate": .14, "to_count": .1, "change_connection": .12}
        if self.json:
            kinds = {"add_filter": .45, "add_sort_limit": .35, "to_count": .2}
        elif self.mode != "console":
            kinds.pop("change_connection")
        if self.cfg.get("create_weights") == "v1.2":
            kinds = {**{k: v * .5 for k, v in kinds.items()}, **{k: w for k, w in self.MODIFY_V12.items()}}
        for _ in range(12):
            try:
                kind = pick(r, kinds)
                return self.modify_v12(kind) if kind in self.MODIFY_V12 else self.modify_kind(kind)
            except (IndexError, ValueError, KeyError, StopIteration):
                continue
        raise ValueError("nenhuma alteração aplicável")

    MODIFY_V12 = {"natural_sort": .15, "pipeline_add_match": .2, "pipeline_add_group": .15}

    def modify_v12(self, kind: str) -> Scenario:
        """Alterações v1.2: ordenação em linguagem natural e edição de pipelines existentes preservando os estágios que já estão no editor."""
        r = self.rng
        cp = self.coll("pt")
        if kind == "natural_sort":
            flt, _, _, fmust = self.filter(1) if r.random() < .6 else ({}, "", "", [])
            sort, spt, sen, smust = self.natural_sort()
            if next(iter(sort)) in flt:
                raise ValueError("ordenação no campo filtrado")
            if self.json:
                editor, output = self.pipeline([{"$match": flt}]), self.pipeline([{"$match": flt}, {"$sort": sort}])
            else:
                editor = self.chain(self.target(), [("find", [flt])])
                output = self.chain(self.target(), [("find", [flt]), ("sort", [sort])])
            spt = agree_pt(spt, cp)
            pt = r.choice(["ordena {s}", "coloca {s}", "{s}, por favor", "deixa {s}"]).format(s=spt)
            en = r.choice(["sort it {s}", "order them {s}", "{s} please", "show {s}"]).format(s=sen)
            must = [["sort", "$sort"]] + smust + ([[next(iter(flt))]] if flt else [])
            return Scenario("modify", kind, "crud", pt, en, editor, output, ["find", "aggregate"], must, self.name)
        group = self.group_field()
        gpath = self.path(group)
        if kind == "pipeline_add_match":
            metric = [x for x in self.cond_fields() if x.type in ("money", "int") and x is not group]
            use_sum = bool(metric) and r.random() < .5
            number_field = r.choice(metric) if use_sum else None
            accumulator = {"$sum": "$" + self.path(number_field)} if number_field else {"$sum": 1}
            key = "total" if number_field else "count"
            existing: list[dict] = [{"$group": {"_id": "$" + gpath, key: accumulator}}] + ([{"$sort": {key: -1}}] if r.random() < .4 else [])
            flt, npt, nen, fmust = self.natural_filter(1)
            if gpath in flt:
                raise ValueError("filtro no campo agrupado")
            editor, output = self.pipeline(existing), self.pipeline([{"$match": flt}] + existing)
            pt = r.choice(["considera só {x}", "filtra antes: só {x}", "só entra {x}", "aplica isso só pra {x}"]).format(x=npt)
            en = r.choice(["only include {x}", "just consider {x}", "filter it first to {x}", "limit this to {x}"]).format(x=nen)
            must = [["$match"]] + fmust + [["$group"], ["$" + gpath]]
            return Scenario("modify", kind, "aggregation", pt, en, editor, output, ["aggregate"], must, self.name)
        # pipeline_add_group: agrega depois do $match que já está no editor
        flt, _, _, fmust = self.natural_filter(1) if r.random() < .6 else self.filter(1)
        if gpath in flt:
            raise ValueError("agrupamento no campo filtrado")
        metric = [x for x in self.cond_fields() if x.type in ("money", "int") and x is not group and self.path(x) not in flt]
        number_field = r.choice(metric) if metric and r.random() < .45 else None
        accumulator = {"$sum": "$" + self.path(number_field)} if number_field else {"$sum": 1}
        key = "total" if number_field else "count"
        stages = [{"$match": flt}, {"$group": {"_id": "$" + gpath, key: accumulator}}] + ([{"$sort": {key: -1}}] if r.random() < .3 else [])
        editor, output = self.pipeline([{"$match": flt}]), self.pipeline(stages)
        lg_pt, lg_en = self.label(group, "pt"), self.label(group, "en")
        if number_field:
            lm_pt, lm_en = self.label(number_field, "pt"), self.label(number_field, "en")
            pt = r.choice(["agora agrupa por {g} e soma {m}", "soma {m} por {g}", "total de {m} por {g}"]).format(g=lg_pt, m=lm_pt)
            en = r.choice(["now group by {g} and sum {m}", "total {m} per {g}", "sum {m} by {g}"]).format(g=lg_en, m=lm_en)
        else:
            pt = r.choice(["agora agrupa por {g} e conta", "conta por {g}", "quantos por {g}?"]).format(g=lg_pt)
            en = r.choice(["now group by {g} and count", "count per {g}", "how many per {g}?"]).format(g=lg_en)
        must = [["$match"]] + [[k] for k in flt] + [["$group"], ["$" + gpath], ["$sum"]] + ([["$" + self.path(number_field)]] if number_field else [])
        return Scenario("modify", kind, "aggregation", pt, en, editor, output, ["aggregate"], must, self.name)

    def modify_kind(self, kind: str) -> Scenario:
        r = self.rng
        default = r.random() < .4
        base: dict[str, Any] = {} if default else self.filter(1)[0]
        base_limit = 100 if default else (r.choice([10, 20, 50]) if r.random() < .5 else 0)
        cp, ce = self.coll("pt"), self.coll("en")
        if self.json:
            base_stages: list[dict] = ([{"$match": base}] if base or r.random() < .5 else []) + ([{"$limit": base_limit}] if base_limit else [])
            editor = render(base_stages or [{"$match": {}}], self.style)
            base_stages = base_stages or [{"$match": {}}]
        else:
            base_calls = [("find", [base])] + ([("limit", [base_limit])] if base_limit else [])
            editor = self.chain(self.target(), base_calls)
        if kind == "add_filter":
            flt, cpt, cen, must = self.filter(1)
            key = next(iter(flt))
            if key in base:
                raise ValueError("campo já filtrado")
            merged = {**base, **flt}
            if self.json:
                stages = [dict(s) for s in base_stages]
                if "$match" in stages[0]:
                    stages[0] = {"$match": {**stages[0]["$match"], **flt}}
                else:
                    stages.insert(0, {"$match": flt})
                output = render(stages, self.style)
            else:
                output = self.chain(self.target(), [("find", [merged])] + ([("limit", [base_limit])] if base_limit else []))
            pt = r.choice(["filtre só {x}", "adicione um filtro: {x}", "mostre apenas os que têm {x}", "acrescente a condição {x}"]).format(x=cpt)
            en = r.choice(["only {x}", "add a filter for {x}", "keep only documents with {x}", "also require {x}"]).format(x=cen)
            ops = ["find", "aggregate"]
        elif kind == "add_sort_limit":
            f = r.choice([x for x in self.cond_fields() if x.numeric or x.type in ("date", "name", "code")])
            direction = r.choice([1, -1])
            limit = r.choice([5, 10, 20, 50])
            with_limit = r.random() < .7
            word = ("decrescente", "descending") if direction == -1 else ("crescente", "ascending")
            if self.json:
                stages = [s for s in base_stages if "$limit" not in s] + [{"$sort": {self.path(f): direction}}] + ([{"$limit": limit}] if with_limit else [])
                output = render(stages, self.style)
            else:
                calls = [("find", [base]), ("sort", [{self.path(f): direction}])]
                if with_limit:
                    calls.append(("limit", [limit]))
                elif base_limit:
                    calls.append(("limit", [base_limit]))
                output = self.chain(self.target(), calls)
            pt = f"ordene por {self.label(f, 'pt')} {word[0]}"
            en = f"sort by {self.label(f, 'en')} {word[1]}"
            must = [["sort"], [self.path(f)], [f"{self.path(f)}:{direction}", f'"{self.path(f)}":{direction}']]
            if with_limit:
                lp, le, alts = self.limit_phrase(limit)
                pt, en, must = pt + lp, en + le, must + [alts]
            ops = ["find", "aggregate"]
        elif kind == "add_projection":
            fields = r.sample([x for x in self.cond_fields()], min(3, len(self.cond_fields())))[: r.randint(1, 3)]
            projection: dict[str, Any] = {self.path(x): 1 for x in fields}
            no_id = r.random() < .4
            if no_id:
                projection = {"_id": 0, **projection}
            output = self.chain(self.target(), [("find", [base, projection])] + ([("limit", [base_limit])] if base_limit else []))
            names_pt = ", ".join(self.label(x, "pt") for x in fields)
            names_en = ", ".join(self.label(x, "en") for x in fields)
            pt = r.choice(["retorne só {x}", "mostre apenas os campos {x}", "traga somente {x}"]).format(x=names_pt) + (" sem o _id" if no_id else "")
            en = r.choice(["return only {x}", "project {x}", "only include {x}"]).format(x=names_en) + (" without _id" if no_id else "")
            must = [[self.path(x)] for x in fields] + ([["_id:0", '"_id":0']] if no_id else [])
            ops = ["find", "aggregate"]
        elif kind == "to_aggregate":
            stages = [{"$match": base}] + ([{"$limit": base_limit}] if base_limit else [])
            output = self.chain(self.target(), [("aggregate", [stages])])
            pt = r.choice(["converta para aggregate", "reescreva usando pipeline de agregação", "transforme em aggregate com $match"])
            en = r.choice(["convert this to an aggregation pipeline", "rewrite it using aggregate", "turn it into an aggregate with $match"])
            must = [["aggregate"], ["$match"]] + ([[f"$limit:{base_limit}"]] if base_limit else [])
            ops = ["aggregate"]
        elif kind == "to_count":
            if self.json:
                stages = [s for s in base_stages if "$limit" not in s] + [{"$count": "total"}]
                output = render(stages, self.style)
            else:
                output = self.chain(self.target(), [("countDocuments", [base])])
            pt = r.choice(["em vez de listar, conte", "troque por uma contagem", "só quero a quantidade"])
            en = r.choice(["count instead of listing", "change it to a count", "I only need the number of documents"])
            must = [["countDocuments", "$count"]]
            ops = ["countDocuments", "aggregate"]
        else:
            conn, database = self.g.connection, self.g.database
            target = [f"getConnection({self.q(conn)})", f".getDatabase({self.q(database)})", f".getCollection({self.q(self.name)})"]
            output = self.chain(target, [("find", [base])] + ([("limit", [base_limit])] if base_limit else []))
            pt = r.choice(["execute isso na conexão {k}, banco {d}", "use getConnection para {k}/{d}", "aponte para a conexão {k} e o banco {d}"]).format(k=conn, d=database)
            en = r.choice(["run this against connection {k}, database {d}", "use getConnection for {k}/{d}", "point it to connection {k} and database {d}"]).format(k=conn, d=database)
            must = [[f'getConnection("{conn}")'], [f'getDatabase("{database}")'], [self.name]]
            ops = ["find"]
        scenario = Scenario("modify", kind, "slop_dsl" if kind == "change_connection" else ("aggregation" if kind == "to_aggregate" else "crud"),
                            pt, en, editor, output, ops, must, self.name)
        if not self.json and r.random() < self.cfg.get("multi_statement_probability", .15):
            other_key = r.choice([k for k in self.g.colls if k != self.g.coll_key])
            other = self.chain(self.target(self.g.cname(other_key)), [("find", [{}]), ("limit", [r.choice([10, 20])])])
            if r.random() < .5:
                scenario.editor, scenario.output = other + "\n" + scenario.editor, other + "\n" + scenario.output
            else:
                scenario.editor, scenario.output = scenario.editor + "\n" + other, scenario.output + "\n" + other
            scenario.must.append([self.g.cname(other_key)])
            scenario.pt += f" (só na consulta de {self.name})"
            scenario.en += f" (only in the {self.name} query)"
        return scenario

    # -- correção ---------------------------------------------------------------------------------------------------------------------------
    def correction(self) -> Scenario:
        r = self.rng
        for _ in range(16):
            try:
                base = self.create()
            except ValueError:
                continue
            corruptions = self.corruptions(base)
            if not corruptions:
                continue
            name, broken, semantic, hint_pt, hint_en = r.choice(corruptions)
            pt = r.choice(["corrija a query", "corrija o código", "tem algum erro aqui? corrija", "isso não funciona, arrume", "conserte essa consulta",
                           "corrija o erro"])
            en = r.choice(["fix this query", "fix the code", "something is wrong here, fix it", "this doesn't work, fix it", "correct this query"])
            if hint_pt and (semantic or r.random() < .35):
                pt, en = pt + f" — {hint_pt}", en + f" — {hint_en}"
            return Scenario("correction", name, base.category, pt, en, broken, base.output, base.operations, base.must, base.collection,
                            base.desc_pt, base.desc_en, semantic=semantic)
        raise ValueError("nenhuma correção aplicável")

    def corruptions(self, base: Scenario) -> list[tuple[str, str, bool, str, str]]:
        r = self.rng
        text = base.output
        out: list[tuple[str, str, bool, str, str]] = []

        def add(name: str, broken: str | None, semantic: bool = False, hint_pt: str = "", hint_en: str = "") -> None:
            if broken is not None and broken != text:
                out.append((name, broken, semantic, hint_pt, hint_en))

        if "ConnectionPool" in text:
            add("pool_typo", text.replace("ConnectionPool", "ConnectionPull", 1), hint_pt="ConnectionPull is not defined", hint_en="ConnectionPull is not defined")
        if ".getDatabase(" in text:
            add("get_database_case", text.replace(".getDatabase(", ".GetDatabase(", 1), hint_pt="GetDatabase não existe", hint_en="GetDatabase is not a function")
        if ".getCollection(" in text:
            add("get_collection_case", text.replace(".getCollection(", ".GetCollection(", 1))
            add("collection_method", text.replace(".getCollection(", ".collection(", 1), hint_pt="db.collection is not a function", hint_en="db.collection is not a function")
        operator = re.search(r'(?<![\w$])("?)\$(gte|lte|gt|lt|in|nin|ne|exists|all|size)\b', text)
        if operator:
            add("missing_dollar", text[:operator.start()] + operator.group(1) + text[operator.start() + len(operator.group(1)) + 1:], semantic=True,
                hint_pt="o filtro não retorna nada", hint_en="the filter matches nothing")
        stage_typos = {"$group": "$grup", "$match": "$macth", "$project": "$projcet", "$lookup": "$lokup", "$sort": "$srot", "$limit": "$limt",
                       "$unwind": "$unwnd", "$search": "$serach", "$count": "$cont"}
        present = [s for s in stage_typos if re.search(r'(?<![\w$])"?' + re.escape(s) + r'"?\s*:', text)]
        if present:
            stage = r.choice(present)
            add("stage_typo", text.replace(stage, stage_typos[stage], 1), hint_pt="estágio desconhecido", hint_en="unrecognized pipeline stage")
        method_typos = {".find(": ".fnd(", ".aggregate(": ".agregate(", ".countDocuments(": ".countDocument(", ".limit(": ".limt(", ".sort(": ".sortt(",
                        ".distinct(": ".distinc(", ".createIndex(": ".createIndx(", ".deleteMany(": ".deletemany(", ".updateMany(": ".updatemany("}
        methods = [m for m in method_typos if m in text]
        if methods:
            method = r.choice(methods)
            typo = method_typos[method].strip(".(")
            add("method_typo", text.replace(method, method_typos[method], 1), hint_pt=f"{typo} não é uma função", hint_en=f"{typo} is not a function")
        closer = max(text.rfind(")"), text.rfind("}"), text.rfind("]"))
        if closer > 0:
            add("missing_close", text[:closer] + text[closer + 1:], hint_pt="SyntaxError", hint_en="SyntaxError")
        if self.mode == "console":
            count = re.search(r"\.countDocuments\((.*)\)$", text, re.DOTALL)
            if count:
                add("count_cursor", text[:count.start()] + f".find({count.group(1)}).count()", hint_pt="count não existe no Console", hint_en="count is not available in the Console")
            if ".insertOne(" in text:
                add("legacy_insert", text.replace(".insertOne(", ".insert(", 1), hint_pt="insert não é suportado", hint_en="insert is not supported")
        date = re.search(r'ISODate\("(\d{4}-\d{2}-\d{2})T00:00:00Z"\)', text)
        if date:
            add("string_date", text[:date.start()] + f'"{date.group(1)}"' + text[date.end():], semantic=True,
                hint_pt="o filtro de data não retorna nada", hint_en="the date filter returns nothing")
        if base.stages and len(base.stages) >= 2 and "$search" in base.stages[0]:
            rotated = base.stages[1:2] + base.stages[:1] + base.stages[2:]
            add("search_not_first", self.pipeline(rotated), hint_pt="$search precisa ser o primeiro estágio", hint_en="$search must be the first stage")
        return out

    # -- explicação ---------------------------------------------------------------------------------------------------------------------------
    def explain(self) -> Scenario:
        r = self.rng
        if r.random() < .45:
            for _ in range(10):
                try:
                    base = self.create()
                except ValueError:
                    continue
                if not base.desc_pt:
                    continue
                pt = r.choice(["o que essa query faz?", "explique esse código", "explica rapidinho o que isso faz", "o que essa consulta retorna?"])
                en = r.choice(["what does this query do?", "explain this code", "briefly explain what this does", "what does this query return?"])
                scenario = Scenario("explain", "explain_query", "chat_technical", pt, en, base.output, base.output, base.operations,
                                    base.must + [["//"]], base.collection, base.desc_pt, base.desc_en)
                scenario.extra = {"comment_pt": base.desc_pt, "comment_en": base.desc_en}
                return scenario
        from .glossary import CONCEPTS

        for _ in range(12):
            concept = r.choice(CONCEPTS)
            if self.mode not in concept.modes:
                continue
            try:
                code, must, operations = concept.build(self)
            except (IndexError, ValueError, KeyError):
                continue
            scenario = Scenario("explain", "concept:" + concept.id, "chat_technical", r.choice(concept.pt), r.choice(concept.en),
                                self.rng.choice(["", self.default_editor()]) if not self.json else "", code, operations, must + [["//"]], self.name)
            scenario.extra = {"comment_pt": concept.answer_pt, "comment_en": concept.answer_en}
            return scenario
        raise ValueError("nenhuma explicação aplicável")

    # -- exemplo final ------------------------------------------------------------------------------------------------------------------------
    def example(self, kind: str, language: str) -> dict:
        r = self.rng
        scenario = getattr(self, kind)()
        lang = "pt" if language == "pt-BR" else "en"
        nl = "\r\n" if r.random() < .4 else "\n"
        output = scenario.output
        if scenario.kind == "explain":
            comment = scenario.extra["comment_pt" if lang == "pt" else "comment_en"]
            output = "// " + comment + "\n" + output
        editor = scenario.editor.replace("\n", nl)
        output = output.replace("\n", nl)
        instruction = scenario.pt if lang == "pt" else scenario.en
        # Rótulos que já começam com o conector ("with fullName", "total") geram "with with"/"com com"/"total total".
        instruction = DOUBLED.sub(r"\1", instruction)
        if lang == "pt" and r.random() < .15:
            instruction = deaccent(instruction)
        if r.random() < .45:
            instruction = instruction[:1].upper() + instruction[1:]
        if r.random() < .15 and not instruction.endswith("?"):
            instruction += "."
        fields: list[str] = []
        if self.natural or r.random() < .55:
            fields = ["_id"]
            for f in self.g.coll.fields:
                parts = self.g.fpath(f).split(".")
                for i in range(1, len(parts) + 1):
                    candidate = ".".join(parts[:i])
                    if candidate not in fields:
                        fields.append(candidate)
        profile = self.g.connection if r.random() < .92 else ""
        database = self.g.database if r.random() < .95 else ""
        if self.coll_natural or r.random() < .75:
            collection = self.name
        else:
            collection = "" if r.random() < .6 else self.g.cname(r.choice(list(self.g.colls)))
        visual = f"{profile or 'Sem conexão'} › {database or 'Escolha um banco'}" + ("" if self.mode == "console" or not collection else f" › {collection}")
        additional = ([f"conexão: {profile}"] if profile else []) + [f"destino visual: {visual}"]
        if fields:
            additional.append("campos conhecidos nos resultados: " + ", ".join(fields[:128]))
        additional.append("não executar consultas nem alterar o Explorer durante a revisão")
        header = r.choice(["Console", "Console 1", "Console 2", collection or "Console", database or "Console"] if self.mode == "console" else
                          (["Script 1", "script.js", "relatorio.js", "Script"] if self.mode == "script" else ["Agregação 1", "Agregação", collection or "Agregação"]))
        ctx = chat_context(instruction, editor, database, collection, infer_operation(editor, self.mode), header, "\n".join(additional))
        return {
            "type": "correction" if scenario.kind == "correction" else "chat", "language": language, "category": scenario.category,
            "subcategory": f"{scenario.kind}:{scenario.intent}", "dialect": MODE_DIALECT[self.mode], "mode": self.mode, "domain": self.domain,
            "input": instruction, "output": output, "chat_context": ctx,
            "intent": {"kind": scenario.kind, "name": scenario.intent, "operations": scenario.operations, "must_include": scenario.must,
                       "collection": scenario.collection},
            "editor_expected_invalid": scenario.kind == "correction" and not scenario.semantic, "semantic_correction": scenario.semantic,
            "natural_labels": self.natural, "natural_collection": self.coll_natural,
        }
