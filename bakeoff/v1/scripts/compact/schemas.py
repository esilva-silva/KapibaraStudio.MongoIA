"""Domínios e schemas 100% sintéticos (sem dados, nomes de bancos ou coleções de clientes reais).

Notação de campo: ``caminho:tipo[:argumento]``; ``itens[].campo`` indica array de subdocumentos.
Tipos: code, enum, money, int, float, bool, date, oid, ref, uuid, name, email, city, state, country, text, word,
tags, geo, version.
"""
from __future__ import annotations

import random
import re
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .jsrender import DateV, Money, Oid, UuidV

DOMAINS: dict[str, dict] = {
    "commerce": {
        "databases": ["ShopDb", "commerce", "storefront", "CakeShop", "retail", "marketplace"],
        "search_index": ["default", "products_search", "catalog"],
        "collections": {
            "orders": [
                "orderNumber:code:ORD", "status:enum:pending|paid|shipped|delivered|cancelled|refunded", "total:money:10-5000",
                "currency:enum:BRL|USD|EUR", "createdAt:date", "updatedAt:date", "customer.id:ref:customers", "customer.name:name",
                "customer.email:email", "items[].sku:code:SKU", "items[].name:text:product", "items[].quantity:int:1-12",
                "items[].price:money:5-900", "shipping.city:city", "shipping.state:state", "shipping.country:country",
                "paymentMethod:enum:card|pix|boleto|paypal", "tags:tags:gift|express|promo|b2b|marketplace", "discount:float:0-0.3",
            ],
            "customers": [
                "name:name", "email:email", "active:bool", "tier:enum:bronze|silver|gold|platinum", "createdAt:date",
                "address.city:city", "address.state:state", "address.zip:code:ZIP", "totalSpent:money:0-90000",
                "lastOrderAt:date", "preferences.newsletter:bool", "segment:enum:retail|wholesale|vip",
            ],
            "products": [
                "sku:code:SKU", "name:text:product", "category:enum:cakes|breads|drinks|electronics|books|toys|garden",
                "price:money:2-3000", "stock:int:0-5000", "rating:float:1-5", "tags:tags:new|sale|organic|imported|bestseller",
                "active:bool", "description:text:description", "brand:enum:Acme|Nimbus|Orion|Vertex|Aurora",
                "attributes.color:enum:red|blue|green|black|white", "attributes.size:enum:S|M|L|XL", "createdAt:date",
            ],
            "reviews": [
                "productId:ref:products", "customerId:ref:customers", "rating:int:1-5", "title:text:review",
                "comment:text:review", "createdAt:date", "verified:bool", "helpfulVotes:int:0-400",
            ],
        },
    },
    "saas": {
        "databases": ["platform", "SlopSaaS", "tenantdb", "CoreDb", "workspace"],
        "search_index": ["default", "tickets_search"],
        "collections": {
            "projects": [
                "System.Id:code:", "Account.Id:code:", "name:text:project", "status:enum:active|archived|draft",
                "ownerId:ref:users", "members[].userId:ref:users", "members[].role:enum:owner|editor|viewer", "createdAt:date",
                "updatedAt:date", "settings.visibility:enum:private|internal|public", "budget:money:1000-250000", "dueDate:date",
                "tags:tags:internal|client|urgent|research",
            ],
            "users": [
                "username:word", "email:email", "role:enum:admin|editor|viewer|billing", "active:bool", "lastLoginAt:date",
                "createdAt:date", "profile.displayName:name", "profile.locale:enum:pt-BR|en-US|es-ES", "mfaEnabled:bool",
                "loginCount:int:0-9000", "accountId:ref:accounts",
            ],
            "events": [
                "type:enum:login|logout|page_view|click|purchase|error", "userId:ref:users", "sessionId:uuid", "timestamp:date",
                "properties.path:enum:/home|/billing|/settings|/projects|/reports", "properties.durationMs:int:5-60000",
                "device.os:enum:windows|linux|macos|android|ios", "device.browser:enum:chrome|firefox|edge|safari",
            ],
            "subscriptions": [
                "accountId:ref:accounts", "plan:enum:free|starter|pro|enterprise", "status:enum:trialing|active|past_due|canceled",
                "seats:int:1-500", "mrr:money:0-40000", "startedAt:date", "renewsAt:date", "billing.interval:enum:monthly|yearly",
            ],
            "tickets": [
                "number:int:1000-99999", "subject:text:subject", "status:enum:open|pending|solved|closed",
                "priority:enum:low|normal|high|urgent", "assigneeId:ref:users", "requesterId:ref:users", "createdAt:date",
                "resolvedAt:date", "tags:tags:billing|bug|feature|access|performance", "channel:enum:email|chat|phone",
            ],
        },
    },
    "logistics": {
        "databases": ["logistics", "FleetDb", "warehouse", "Shipping"],
        "search_index": ["default"],
        "collections": {
            "shipments": [
                "trackingCode:code:TRK", "status:enum:created|in_transit|out_for_delivery|delivered|returned",
                "origin.city:city", "destination.city:city", "destination.location:geo", "weightKg:float:0.1-900",
                "carrier:enum:FastShip|RouteX|CargoOne|Loggi", "estimatedDelivery:date", "deliveredAt:date",
                "events[].status:enum:created|in_transit|delivered", "events[].at:date", "events[].city:city",
                "warehouseCode:code:WH",
            ],
            "warehouses": [
                "code:code:WH", "name:text:warehouse", "city:city", "capacity:int:1000-90000", "location:geo", "active:bool",
            ],
            "vehicles": [
                "plate:code:ABC", "type:enum:van|truck|motorcycle", "capacityKg:int:100-30000", "driverId:ref:drivers",
                "lastMaintenanceAt:date", "odometerKm:int:0-900000", "status:enum:available|on_route|maintenance",
            ],
            "inventory": [
                "sku:code:SKU", "warehouseCode:code:WH", "quantity:int:0-20000", "reserved:int:0-2000", "updatedAt:date",
                "lot:code:LOT",
            ],
        },
    },
    "iot": {
        "databases": ["telemetry", "IotHub", "sensors", "plant"],
        "search_index": ["default"],
        "collections": {
            "devices": [
                "deviceId:code:dev-", "model:enum:TX-100|TX-200|HX-9|EnviroPro", "firmware:version", "status:enum:online|offline|maintenance",
                "location:geo", "site:enum:plant-a|plant-b|warehouse-3|lab", "installedAt:date", "lastSeenAt:date",
                "tags:tags:critical|outdoor|indoor|pilot",
            ],
            "readings": [
                "deviceId:code:dev-", "metric:enum:temperature|humidity|pressure|co2|voltage", "value:float:-20-1200",
                "unit:enum:C|%|hPa|ppm|V", "timestamp:date", "quality:enum:good|suspect|bad",
            ],
            "alerts": [
                "deviceId:code:dev-", "severity:enum:info|warning|critical", "message:text:alert", "acknowledged:bool",
                "createdAt:date", "resolvedAt:date", "rule:enum:high_temp|low_battery|offline|threshold",
            ],
        },
    },
    "hr": {
        "databases": ["hr", "PeopleDb", "company", "CompanyDb"],
        "search_index": ["default", "employees_search"],
        "collections": {
            "employees": [
                "employeeCode:code:E", "fullName:name", "email:email",
                "department:enum:engineering|sales|finance|support|marketing|operations",
                "title:enum:analyst|engineer|manager|director|intern|specialist", "salary:money:2000-45000", "hiredAt:date",
                "active:bool", "managerId:ref:employees", "skills:tags:mongodb|csharp|python|sql|excel|negotiation",
                "address.city:city", "contract.type:enum:clt|pj|intern",
            ],
            "departments": [
                "name:enum:engineering|sales|finance|support|marketing|operations", "code:code:DEP", "budget:money:10000-9000000",
                "headcount:int:1-800", "managerId:ref:employees",
            ],
            "timesheets": [
                "employeeId:ref:employees", "date:date", "hours:float:0.5-12", "projectCode:code:PRJ", "approved:bool",
            ],
        },
    },
    "content": {
        "databases": ["cms", "blog", "ContentDb", "newsroom"],
        "search_index": ["default", "articles_search", "autocomplete"],
        "collections": {
            "articles": [
                "title:text:article", "slug:word", "body:text:body", "authorId:ref:authors", "status:enum:draft|published|archived",
                "publishedAt:date", "tags:tags:mongodb|cloud|design|career|ai|security", "views:int:0-900000", "likes:int:0-40000",
                "category:enum:tech|business|lifestyle|science", "language:enum:pt|en|es", "summary:text:summary",
            ],
            "comments": [
                "articleId:ref:articles", "authorName:name", "text:text:comment", "createdAt:date", "approved:bool", "score:int:-50-500",
            ],
            "authors": [
                "name:name", "bio:text:bio", "email:email", "followers:int:0-200000", "verified:bool", "joinedAt:date",
            ],
        },
    },
    "finance": {
        "databases": ["finance", "billing", "LedgerDb", "accounting"],
        "search_index": ["default"],
        "collections": {
            "invoices": [
                "number:code:INV-", "customerId:ref:customers", "amount:money:10-80000", "currency:enum:BRL|USD|EUR",
                "status:enum:draft|open|paid|void|overdue", "issuedAt:date", "dueAt:date", "paidAt:date",
                "lines[].description:text:line", "lines[].quantity:int:1-100", "lines[].unitPrice:money:1-5000", "taxRate:float:0-0.27",
            ],
            "payments": [
                "invoiceId:ref:invoices", "amount:money:10-80000", "method:enum:card|pix|boleto|transfer",
                "status:enum:pending|succeeded|failed|refunded", "processedAt:date", "gateway:enum:stripe|adyen|pagarme|internal",
                "attempts:int:1-5",
            ],
            "accounts": [
                "code:code:ACC", "name:text:account", "type:enum:asset|liability|equity|revenue|expense", "balance:money:-500000-900000",
                "currency:enum:BRL|USD|EUR", "updatedAt:date",
            ],
        },
    },
    "travel": {
        "databases": ["travel", "BookingDb", "tourism"],
        "search_index": ["default", "hotels_search"],
        "collections": {
            "bookings": [
                "code:code:BK", "status:enum:confirmed|cancelled|pending", "hotelId:ref:hotels", "guestName:name", "checkIn:date",
                "checkOut:date", "nights:int:1-21", "totalPrice:money:80-20000", "rooms:int:1-4", "channel:enum:web|app|agency",
            ],
            "hotels": [
                "name:text:hotel", "city:city", "stars:int:1-5", "amenities:tags:wifi|pool|gym|spa|parking|breakfast",
                "location:geo", "rating:float:1-10", "pricePerNight:money:60-3000", "description:text:description",
            ],
            "flights": [
                "flightNumber:code:LA", "origin:enum:GRU|GIG|BSB|CNF|POA|REC|SSA", "destination:enum:GRU|GIG|BSB|LIS|MIA|EZE|SCL",
                "departureAt:date", "arrivalAt:date", "status:enum:scheduled|boarding|departed|arrived|delayed|cancelled",
                "seatsAvailable:int:0-280", "price:money:150-9000",
            ],
        },
    },
    # Estilo do pedido original (Projects/Account.Id/System.Id/CreatedOn), nomes PascalCase; totalmente sintético.
    "workspace": {
        "databases": ["Connect", "Workspace", "PortalDb", "Tenancy"],
        "search_index": ["default", "projects_search"],
        "collections": {
            "Projects": [
                "System.Id:code:", "Account.Id:code:", "Name:text:project", "Status:enum:Active|Archived|Draft|Suspended",
                "CreatedOn:date", "UpdatedOn:date", "Owner.Id:ref:Users", "Tags:tags:internal|client|urgent|research",
                "Budget:money:1000-250000", "Members[].UserId:ref:Users", "Members[].Role:enum:Owner|Editor|Viewer", "DueOn:date",
                "Description:text:description",
            ],
            "Accounts": [
                "System.Id:code:", "Name:text:account", "Plan:enum:Free|Team|Business|Enterprise", "Active:bool", "CreatedOn:date",
                "Billing.Country:country", "Billing.City:city", "Seats:int:1-900", "Mrr:money:0-60000",
            ],
            "Users": [
                "Account.Id:code:", "Email:email", "DisplayName:name", "Role:enum:Admin|Member|Guest", "Active:bool",
                "LastLoginOn:date", "CreatedOn:date", "LoginCount:int:0-9000",
            ],
            "Tasks": [
                "Project.Id:ref:Projects", "Account.Id:code:", "Title:text:task", "State:enum:Todo|Doing|Done|Blocked",
                "Priority:enum:Low|Normal|High|Critical", "Assignee.Id:ref:Users", "CreatedOn:date", "CompletedOn:date",
                "EstimateHours:float:0.5-80", "Labels:tags:bug|feature|chore|docs",
            ],
        },
    },
    # Domínio reservado ao benchmark (nunca aparece em treino/validação): mede generalização por contexto.
    "education": {
        "databases": ["school", "EduDb", "learning", "Academy"],
        "search_index": ["default", "courses_search"],
        "collections": {
            "courses": [
                "code:code:CS", "title:text:course", "category:enum:programming|data|design|languages|business",
                "credits:int:1-8", "level:enum:beginner|intermediate|advanced", "active:bool", "instructorId:ref:instructors",
                "tags:tags:online|live|certificate|free", "price:money:0-2500", "rating:float:1-5", "description:text:description",
            ],
            "enrollments": [
                "courseId:ref:courses", "studentId:ref:students", "status:enum:enrolled|completed|dropped", "progress:int:0-100",
                "enrolledAt:date", "completedAt:date", "grade:float:0-10",
            ],
            "students": [
                "name:name", "email:email", "cohort:enum:2025-A|2025-B|2026-A", "gpa:float:0-10", "active:bool",
                "country:country", "interests:tags:ai|web|mobile|data|cloud",
            ],
        },
    },
}

FIRST = ["Ana", "Bruno", "Carla", "Diego", "Eduarda", "Felipe", "Gabriela", "Hugo", "Isabela", "João", "Karina", "Lucas",
         "Marina", "Nicolas", "Olivia", "Pedro", "Rafaela", "Samuel", "Tatiana", "Vitor", "Alice", "Ben", "Chloe", "Daniel"]
LAST = ["Silva", "Souza", "Oliveira", "Santos", "Lima", "Pereira", "Costa", "Rocha", "Almeida", "Ferreira", "Gomes",
        "Martins", "Araujo", "Barbosa", "Ribeiro", "Smith", "Garcia", "Miller", "Lopez", "Nguyen"]
CITIES = ["São Paulo", "Rio de Janeiro", "Curitiba", "Belo Horizonte", "Recife", "Porto Alegre", "Salvador", "Fortaleza",
          "Campinas", "Florianópolis", "Lisbon", "Madrid", "Austin", "Toronto", "Berlin", "Buenos Aires"]
STATES = ["SP", "RJ", "PR", "MG", "PE", "RS", "BA", "CE", "SC", "GO", "DF"]
COUNTRIES = ["BR", "US", "PT", "ES", "AR", "CA", "DE", "MX"]
WORDS = ["alpha", "atlas", "nova", "delta", "orbit", "pixel", "vector", "cobalt", "ember", "lumen", "quartz", "sierra"]
PHRASES = {
    "product": ["chocolate cake", "sourdough bread", "wireless mouse", "cold brew coffee", "garden hose", "kids puzzle", "notebook stand"],
    "description": ["handmade with organic ingredients", "fast charging and long battery life", "ideal for small spaces",
                    "limited edition collection", "best seller of the season"],
    "review": ["great quality", "arrived late", "exactly as described", "would buy again", "packaging could be better"],
    "project": ["billing revamp", "mobile onboarding", "data migration", "search relevance", "partner portal"],
    "subject": ["cannot reset password", "invoice missing", "slow dashboard", "export fails", "access request"],
    "warehouse": ["north hub", "central depot", "coastal storage", "cross dock"],
    "alert": ["temperature above threshold", "device offline", "battery low", "pressure spike detected"],
    "article": ["mongodb aggregation tips", "designing indexes", "atlas search in practice", "scaling read workloads",
                "schema design patterns"],
    "body": ["this article explains how to model data", "a practical guide to pipelines", "lessons learned in production"],
    "summary": ["short guide", "quick overview", "deep dive"],
    "comment": ["very helpful", "thanks for sharing", "can you add examples", "great explanation"],
    "bio": ["backend engineer", "data enthusiast", "technical writer", "cloud architect"],
    "line": ["consulting hours", "license fee", "support plan", "setup service"],
    "account": ["cash", "accounts receivable", "revenue", "payroll expenses"],
    "hotel": ["ocean view resort", "downtown suites", "mountain lodge", "airport inn"],
    "course": ["intro to mongodb", "data modeling", "advanced aggregation", "python for data", "ux fundamentals"],
    "task": ["review pull request", "update onboarding docs", "fix login timeout", "prepare quarterly report", "migrate cron jobs"],
}
SEARCH_TERMS = ["mongo", "cake", "fast", "index", "coffee", "billing", "offline", "resort", "data", "guide", "wireless", "atlas"]

EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass
class Field:
    path: str  # caminho lógico com ponto (itens[] vira itens)
    type: str
    arg: str = ""
    array_parent: str | None = None  # "items" para items[].sku

    @property
    def enum(self) -> list[str]:
        return self.arg.split("|") if self.type in ("enum", "tags") else []

    def range(self) -> tuple[float, float]:
        match = re.match(r"^(-?[\d.]+)-(-?[\d.]+)$", self.arg or "")
        return (float(match[1]), float(match[2])) if match else (0.0, 100.0)

    @property
    def numeric(self) -> bool:
        return self.type in ("money", "int", "float")


@dataclass
class Collection:
    domain: str
    name: str
    fields: list[Field] = field(default_factory=list)

    def by_type(self, *types: str) -> list[Field]:
        return [f for f in self.fields if f.type in types]


def parse_field(spec: str) -> Field:
    path, type_, *rest = spec.split(":")
    arg = ":".join(rest)
    parent = None
    if "[]" in path:
        parent = path.split("[]")[0]
        path = path.replace("[]", "")
    return Field(path, type_, arg, parent)


def load_domain(name: str) -> dict[str, Collection]:
    return {cname: Collection(name, cname, [parse_field(s) for s in specs])
            for cname, specs in DOMAINS[name]["collections"].items()}


# -- estilos de nome ----------------------------------------------------------------------------------------------
def _split_camel(name: str) -> list[str]:
    return re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", name) or [name]


def style_segment(segment: str, style: str) -> str:
    if segment == "_id" or not segment or segment[0].isupper() and style == "camel":
        return segment
    parts = _split_camel(segment)
    if style == "pascal":
        return "".join(p[:1].upper() + p[1:] for p in parts) if not segment[0].isupper() else segment
    if style == "snake":
        return "_".join(p.lower() for p in parts)
    return segment


def style_path(path: str, style: str) -> str:
    return ".".join(style_segment(s, style) for s in path.split("."))


def style_collection(name: str, style: str) -> str:
    parts = _split_camel(name)
    if style == "pascal":
        return "".join(p[:1].upper() + p[1:] for p in parts)
    if style == "snake":
        return "_".join(p.lower() for p in parts)
    if style == "kebab":
        return "-".join(p.lower() for p in parts)
    return name


# -- valores sintéticos -----------------------------------------------------------------------------------------------
def oid(rng: random.Random) -> Oid:
    return Oid("".join(rng.choice("0123456789abcdef") for _ in range(24)))


def iso_date(rng: random.Random, day_precision: bool | None = None) -> str:
    moment = EPOCH + timedelta(days=rng.randint(0, 900), seconds=rng.randint(0, 86399))
    if day_precision if day_precision is not None else rng.random() < 0.6:
        return moment.strftime("%Y-%m-%dT00:00:00Z")
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def sample_value(f: Field, rng: random.Random, date_form: str = "isodate"):
    t = f.type
    if t == "enum":
        return rng.choice(f.enum)
    if t == "tags":
        return rng.choice(f.enum)
    if t in ("money", "float"):
        lo, hi = f.range()
        value = rng.uniform(lo, hi)
        if t == "money":
            return Money(round(value, rng.choice([0, 0, 2])))
        return round(value, rng.choice([1, 2]))
    if t == "int":
        lo, hi = f.range()
        return rng.randint(int(lo), int(hi))
    if t == "bool":
        return rng.random() < 0.6
    if t == "date":
        return DateV(iso_date(rng), date_form)
    if t in ("oid", "ref"):
        return oid(rng)
    if t == "uuid":
        return UuidV(str(_uuid.UUID(int=rng.getrandbits(128), version=4)))
    if t == "name":
        return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    if t == "email":
        return f"{rng.choice(FIRST).lower().replace('ã', 'a').replace('ç', 'c')}.{rng.choice(LAST).lower()}@example.com"
    if t == "city":
        return rng.choice(CITIES)
    if t == "state":
        return rng.choice(STATES)
    if t == "country":
        return rng.choice(COUNTRIES)
    if t == "text":
        return rng.choice(PHRASES.get(f.arg, WORDS))
    if t == "word":
        return rng.choice(WORDS) + ("-" + str(rng.randint(1, 99)) if rng.random() < 0.4 else "")
    if t == "code":
        digits = "".join(rng.choice("0123456789") for _ in range(rng.choice([3, 4, 5, 6])))
        return f"{f.arg}{digits}" if f.arg else digits
    if t == "version":
        return f"{rng.randint(1, 4)}.{rng.randint(0, 12)}.{rng.randint(0, 20)}"
    if t == "geo":
        return {"type": "Point", "coordinates": [round(rng.uniform(-60, -34), 4), round(rng.uniform(-30, -3), 4)]}
    raise ValueError(f"tipo desconhecido {t}")


CONNECTION_NAMES = ["local", "dev-cluster", "staging", "Development", "Production", "analytics-replica", "developercluster",
                    "qa", "reporting", "Sandbox", "Production Cluster", "legacy-rs", "atlas-dev", "test"]
ENV_KEYS = ["CUSTOMER_ID", "TENANT_ID", "ACCOUNT_ID", "REGION", "START_DATE", "PROJECT_ID", "DEFAULT_LIMIT", "STATUS_FILTER"]
