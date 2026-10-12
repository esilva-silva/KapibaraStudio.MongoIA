"""Glossário bilíngue curto (chat técnico): perguntas conceituais → comentário // de uma linha + exemplo de código aplicado à coleção.

Respostas escritas para este projeto (conceitos públicos do MongoDB e da DSL do Slop Studio), curtas e objetivas, sem Markdown.
Cada builder recebe o ChatGenerator e devolve (código, elementos obrigatórios, operações aceitas).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import schemas as S
from .jsrender import IDENT, DateV, DecimalV, Raw


@dataclass
class Concept:
    id: str
    pt: list[str]
    en: list[str]
    answer_pt: str
    answer_en: str
    build: Callable[[Any], tuple[str, list[list[str]], list[str]]]
    modes: tuple[str, ...] = ("console", "script")


def fields(gen: Any, *types: str) -> list[S.Field]:
    return [f for f in gen.cond_fields() if f.type in types]


def arrays(gen: Any) -> list[str]:
    return sorted({f.array_parent for f in gen.g.coll.fields if f.array_parent})


def leaf(gen: Any, field: S.Field) -> str:
    return S.style_segment(field.path.split(".")[-1], gen.g.name_style)


def b_match_filter(gen: Any):
    r = gen.rng
    flt, _, _, must = gen.filter(1)
    parents = [p for p in arrays(gen) if any(f.array_parent == p and f.numeric for f in gen.g.coll.fields)]
    tags = fields(gen, "tags")
    if parents:
        parent = r.choice(parents)
        sub = r.choice([f for f in gen.g.coll.fields if f.array_parent == parent and f.numeric])
        path = gen.g.ppath(parent)
        expr = {"$filter": {"input": "$" + path, "as": "item", "cond": {"$gte": ["$$item." + leaf(gen, sub), gen.g.value(sub)]}}}
    elif tags:
        t = r.choice(tags)
        path = gen.path(t)
        expr = {"$filter": {"input": "$" + path, "as": "tag", "cond": {"$ne": ["$$tag", str(gen.g.value(t))]}}}
    else:
        raise IndexError
    return gen.pipeline([{"$match": flt}, {"$project": {path: expr}}]), must + [["$match"], ["$filter"]], ["aggregate"]


def b_lookup(gen: Any):
    scenario = gen.s_lookup()
    return scenario.output, scenario.must, scenario.operations


def b_unwind(gen: Any):
    parents = arrays(gen)
    tags = fields(gen, "tags")
    path = gen.g.ppath(gen.rng.choice(parents)) if parents else gen.path(gen.rng.choice(tags))
    stages = [{"$unwind": {"path": "$" + path, "preserveNullAndEmptyArrays": True}}, {"$limit": 20}]
    return gen.pipeline(stages), [["$unwind"], ["preserveNullAndEmptyArrays"], [path]], ["aggregate"]


def b_group_sum(gen: Any):
    group = gen.group_field()
    number = gen.rng.choice([f for f in gen.cond_fields() if f.numeric])
    stages = [{"$group": {"_id": "$" + gen.path(group), "total": {"$sum": "$" + gen.path(number)}}}]
    return gen.pipeline(stages), [["$group"], ["$sum"], ["$" + gen.path(number)]], ["aggregate"]


def b_ttl(gen: Any):
    date = gen.rng.choice(fields(gen, "date"))
    code = gen.chain(gen.target(), [("createIndex", [{gen.path(date): 1}, {"expireAfterSeconds": 2592000}])])
    return code, [["createIndex"], ["expireAfterSeconds"], [gen.path(date)]], ["createIndex"]


def b_esr(gen: Any):
    equality = gen.rng.choice(fields(gen, "enum", "code", "bool"))
    sort = gen.rng.choice(fields(gen, "date"))
    rng_fields = [f for f in fields(gen, "int", "money", "float") if f is not equality]
    spec: dict[str, Any] = {gen.path(equality): 1, gen.path(sort): -1}
    if rng_fields:
        spec[gen.path(gen.rng.choice(rng_fields))] = 1
    code = gen.chain(gen.target(), [("createIndex", [spec])])
    return code, [["createIndex"], [gen.path(equality)], [gen.path(sort)]], ["createIndex"]


def b_count_estimated(gen: Any):
    flt, _, _, must = gen.filter(1)
    code = gen.chain(gen.target(), [("countDocuments", [flt])]) + "\n" + gen.chain(gen.target(), [("estimatedDocumentCount", [])])
    return code, must + [["countDocuments"], ["estimatedDocumentCount"]], ["countDocuments", "estimatedDocumentCount"]


def b_elem_match(gen: Any):
    parents = [p for p in arrays(gen) if sum(1 for f in gen.g.coll.fields if f.array_parent == p) >= 2]
    parent = gen.rng.choice(parents)
    subs = gen.rng.sample([f for f in gen.g.coll.fields if f.array_parent == parent], 2)
    condition = {leaf(gen, f): ({"$gte": gen.g.value(f)} if f.numeric else gen.g.value(f)) for f in subs}
    code = gen.chain(gen.target(), [("find", [{gen.g.ppath(parent): {"$elemMatch": condition}}]), ("limit", [20])])
    return code, [["$elemMatch"], [gen.g.ppath(parent)]], ["find"]


def b_in_all(gen: Any):
    t = gen.rng.choice(fields(gen, "tags"))
    values = [str(v) for v in gen.g.distinct_values(t, 2)]
    code = (gen.chain(gen.target(), [("find", [{gen.path(t): {"$in": values}}])]) + "\n"
            + gen.chain(gen.target(), [("find", [{gen.path(t): {"$all": values}}])]))
    return code, [["$in"], ["$all"], [gen.path(t)]], ["find"]


def b_set_unset(gen: Any):
    flt, _, _, must = gen.filter(1)
    a, b = gen.rng.sample([f for f in gen.cond_fields() if f.type in ("enum", "bool", "int", "code", "text") and gen.path(f) not in flt], 2)
    update = {"$set": {gen.path(a): gen.g.value(a)}, "$unset": {gen.path(b): ""}}
    return gen.chain(gen.target(), [("updateMany", [flt, update])]), must + [["$set"], ["$unset"]], ["updateMany"]


def b_upsert(gen: Any):
    flt, _, _, must = gen.filter(1)
    a = gen.rng.choice([f for f in gen.cond_fields() if f.type in ("enum", "bool", "int", "code", "text") and gen.path(f) not in flt])
    code = gen.chain(gen.target(), [("updateOne", [flt, {"$set": {gen.path(a): gen.g.value(a)}}, {"upsert": True}])])
    return code, must + [["upsert"], ["$set"]], ["updateOne"]


def b_search(gen: Any):
    scenario = gen.s_search()
    return scenario.output, scenario.must, scenario.operations


def b_facet(gen: Any):
    group = gen.group_field()
    stages = [{"$facet": {gen.g.out("byGroup"): [{"$group": {"_id": "$" + gen.path(group), "count": {"$sum": 1}}}],
                          gen.g.out("total"): [{"$count": "count"}]}}]
    return gen.pipeline(stages), [["$facet"], ["$group"], ["$count"]], ["aggregate"]


def b_date_filter(gen: Any):
    date = gen.rng.choice(fields(gen, "date"))
    code = gen.chain(gen.target(), [("find", [{gen.path(date): {"$gte": DateV("2026-01-01T00:00:00Z")}}])])
    return code, [["ISODate", "$date"], [gen.path(date)]], ["find"]


def b_decimal(gen: Any):
    money = gen.rng.choice(fields(gen, "money"))
    others = [f for f in gen.cond_fields() if f.type in ("enum", "code", "name", "text") and f is not money][:1]
    doc: dict[str, Any] = {gen.path(f): gen.g.value(f) for f in others}
    doc[gen.path(money)] = DecimalV("19.90")
    return gen.chain(gen.target(), [("insertOne", [doc])]), [["NumberDecimal"], [gen.path(money)]], ["insertOne"]


def b_ejson(gen: Any):
    oid = S.oid(gen.rng)
    code = gen.chain(gen.target(), [("find", [Raw(f"EJSON.parse('{{\"_id\": {{\"$oid\": \"{oid.hex}\"}}}}')")])])
    return code, [["EJSON.parse"], ["$oid"]], ["find"]


def b_pool(gen: Any):
    conn, database, name = gen.g.connection, gen.g.database, gen.name
    if all(IDENT.match(s) for s in (conn, database, name)):
        pool = f"ConnectionPool.{conn}.{database}.{name}"
    else:
        pool = "ConnectionPool" + "".join(f"[{gen.q(s)}]" for s in (conn, database, name))
    explicit = f"getConnection({gen.q(conn)}).getDatabase({gen.q(database)}).getCollection({gen.q(name)})"
    code = f"{pool}.find({{}}).limit(10)\n{explicit}.find({{}}).limit(10)"
    return code, [["ConnectionPool"], ["getConnection"], ["getDatabase"]], ["find"]


def b_projection(gen: Any):
    chosen = gen.rng.sample(gen.cond_fields(), min(2, len(gen.cond_fields())))
    projection = {"_id": 0, **{gen.path(f): 1 for f in chosen}}
    return gen.chain(gen.target(), [("find", [{}, projection])]), [["_id:0", '"_id":0']] + [[gen.path(f)] for f in chosen], ["find"]


def b_pagination(gen: Any):
    code = gen.chain(gen.target(), [("find", [{}]), ("sort", [{"_id": 1}]), ("skip", [20]), ("limit", [20])])
    return code, [["skip(20)"], ["limit(20)"], ["sort("]], ["find"]


def b_expr(gen: Any):
    a, b = gen.rng.sample([f for f in gen.cond_fields() if f.numeric], 2)
    code = gen.chain(gen.target(), [("find", [{"$expr": {"$gt": ["$" + gen.path(a), "$" + gen.path(b)]}}])])
    return code, [["$expr"], ["$gt"], [gen.path(a)], [gen.path(b)]], ["find"]


def b_date_trunc(gen: Any):
    date = gen.rng.choice(fields(gen, "date"))
    stages = [{"$group": {"_id": {"$dateTrunc": {"date": "$" + gen.path(date), "unit": "month"}}, "count": {"$sum": 1}}}, {"$sort": {"_id": 1}}]
    return gen.pipeline(stages), [["$dateTrunc"], ["$group"], [gen.path(date)]], ["aggregate"]


def b_array_filters(gen: Any):
    parents = [p for p in arrays(gen) if any(f.array_parent == p and f.numeric for f in gen.g.coll.fields)
               and any(f.array_parent == p and not f.numeric for f in gen.g.coll.fields)]
    parent = gen.rng.choice(parents)
    number = gen.rng.choice([f for f in gen.g.coll.fields if f.array_parent == parent and f.numeric])
    key = gen.rng.choice([f for f in gen.g.coll.fields if f.array_parent == parent and not f.numeric])
    update = {"$set": {f"{gen.g.ppath(parent)}.$[item].{leaf(gen, number)}": gen.g.value(number)}}
    code = gen.chain(gen.target(), [("updateMany", [{}, update, {"arrayFilters": [{f"item.{leaf(gen, key)}": gen.g.value(key)}]}])])
    return code, [["arrayFilters"], ["$[item]"]], ["updateMany"]


def b_vector(gen: Any):
    stages = [{"$vectorSearch": {"index": "vector_index", "path": gen.g.out("embedding"), "queryVector": Raw("queryVector"),
                                 "numCandidates": 100, "limit": 10}}]
    code = "const queryVector = [0.12, -0.08, 0.33, 0.51]\n" + gen.pipeline(stages)
    return code, [["$vectorSearch"], ["numCandidates"], ["queryVector"]], ["aggregate"]


def b_explain(gen: Any):
    flt, _, _, must = gen.filter(1)
    code = gen.chain(gen.target(), [("find", [flt]), ("explain", ["executionStats"])])
    return code, must + [["explain"], ["executionStats"]], ["find"]


CONCEPTS = [
    Concept("match_vs_filter", ["qual a diferença entre $match e $filter?", "$match e $filter fazem a mesma coisa?"],
            ["what is the difference between $match and $filter?", "do $match and $filter do the same thing?"],
            "$match filtra documentos do pipeline; $filter filtra elementos de um array dentro de cada documento.",
            "$match filters pipeline documents; $filter filters array elements inside each document.", b_match_filter),
    Concept("lookup", ["como faço join entre coleções?", "para que serve o $lookup?"], ["how do I join two collections?", "what is $lookup for?"],
            "$lookup faz um left outer join com outra coleção do mesmo banco e grava os relacionados em um array.",
            "$lookup performs a left outer join with another collection in the same database and stores matches in an array.", b_lookup),
    Concept("unwind", ["o que o $unwind faz?", "como transformar cada item do array em um documento?"],
            ["what does $unwind do?", "how do I turn each array element into a document?"],
            "$unwind gera um documento por elemento do array; preserveNullAndEmptyArrays mantém documentos sem itens.",
            "$unwind outputs one document per array element; preserveNullAndEmptyArrays keeps documents without items.", b_unwind),
    Concept("group_sum", ["como somar valores por grupo?", "como funciona o $group com $sum?"], ["how do I sum values per group?", "how does $group with $sum work?"],
            "$group agrupa pela expressão em _id; acumuladores como $sum calculam o valor de cada grupo.",
            "$group groups by the _id expression; accumulators such as $sum compute a value per group.", b_group_sum),
    Concept("ttl", ["como apagar documentos automaticamente depois de um tempo?", "o que é índice TTL?"],
            ["how do I delete documents automatically after some time?", "what is a TTL index?"],
            "Índice TTL remove documentos automaticamente expireAfterSeconds após o valor do campo de data.",
            "A TTL index deletes documents automatically expireAfterSeconds after the date field value.", b_ttl),
    Concept("esr", ["qual a ordem dos campos num índice composto?", "como montar um índice composto?"],
            ["what field order should a compound index use?", "how should I build a compound index?"],
            "Em índice composto use igualdade primeiro, depois ordenação e por último intervalos (regra ESR).",
            "In a compound index put equality fields first, then sort fields, then range fields (ESR rule).", b_esr),
    Concept("count_vs_estimated", ["countDocuments ou estimatedDocumentCount?", "qual a diferença entre countDocuments e estimatedDocumentCount?"],
            ["countDocuments or estimatedDocumentCount?", "what is the difference between countDocuments and estimatedDocumentCount?"],
            "countDocuments conta com filtro e precisão; estimatedDocumentCount usa metadados, é rápido e não aceita filtro.",
            "countDocuments counts matching documents accurately; estimatedDocumentCount uses metadata, is fast and takes no filter.", b_count_estimated),
    Concept("elem_match", ["quando usar $elemMatch?", "como filtrar um item do array com duas condições?"],
            ["when should I use $elemMatch?", "how do I match one array element with two conditions?"],
            "$elemMatch exige que um mesmo elemento do array satisfaça todas as condições.",
            "$elemMatch requires a single array element to satisfy all conditions.", b_elem_match),
    Concept("in_vs_all", ["qual a diferença entre $in e $all?", "$in ou $all para arrays?"], ["what is the difference between $in and $all?", "$in or $all for arrays?"],
            "$in casa se o campo tiver algum dos valores; $all exige que o array contenha todos eles.",
            "$in matches when the field has any of the values; $all requires the array to contain all of them.", b_in_all),
    Concept("set_unset", ["como alterar um campo e remover outro?", "diferença entre $set e $unset"], ["how do I change one field and remove another?", "$set vs $unset"],
            "$set define ou substitui campos; $unset remove campos do documento.", "$set adds or replaces fields; $unset removes fields from the document.", b_set_unset),
    Concept("upsert", ["o que é upsert?", "como inserir se não existir?"], ["what is an upsert?", "how do I insert when nothing matches?"],
            "Com upsert: true o update insere um documento novo quando nenhum corresponde ao filtro.",
            "With upsert: true the update inserts a new document when none matches the filter.", b_upsert),
    Concept("search_vs_regex", ["quando usar Atlas Search em vez de $regex?", "$search ou $regex?"], ["when should I use Atlas Search instead of $regex?", "$search or $regex?"],
            "$search usa um índice Atlas Search com relevância e fuzzy; $regex varre valores sem análise de texto.",
            "$search uses an Atlas Search index with relevance and fuzzy matching; $regex scans values without text analysis.", b_search, ("console", "script", "aggregation")),
    Concept("facet", ["para que serve o $facet?", "como calcular vários resultados numa agregação só?"], ["what is $facet for?", "how do I compute several results in one aggregation?"],
            "$facet executa vários sub-pipelines sobre os mesmos documentos e devolve tudo em um documento.",
            "$facet runs several sub-pipelines over the same input and returns everything in one document.", b_facet, ("console", "script", "aggregation")),
    Concept("date_filter", ["como filtrar por data?", "posso filtrar datas com texto?"], ["how do I filter by date?", "can I filter dates using strings?"],
            "Datas devem ser comparadas com valores Date (ISODate); texto não casa com campos do tipo Date.",
            "Compare dates with Date values (ISODate); strings do not match Date fields.", b_date_filter),
    Concept("decimal", ["qual tipo usar para dinheiro?", "como gravar valores monetários?"], ["which type should I use for money?", "how do I store currency values?"],
            "Use NumberDecimal (Decimal128) para valores monetários e evitar arredondamento de double.",
            "Use NumberDecimal (Decimal128) for money to avoid double rounding errors.", b_decimal),
    Concept("ejson_parse", ["como usar Extended JSON no Console?", "o que o EJSON.parse faz?"], ["how do I use Extended JSON in the Console?", "what does EJSON.parse do?"],
            "EJSON.parse converte Extended JSON ($oid, $date, $numberDecimal) em tipos BSON no Console.",
            "EJSON.parse turns Extended JSON ($oid, $date, $numberDecimal) into BSON types in the Console.", b_ejson, ("console",)),
    Concept("pool", ["qual a diferença entre ConnectionPool e getConnection?", "como consultar outra conexão no Slop Studio?"],
            ["what is the difference between ConnectionPool and getConnection?", "how do I query another connection in Slop Studio?"],
            "ConnectionPool.Conexao.Banco.Colecao equivale a getConnection(nome).getDatabase(banco).getCollection(colecao); nomes com espaço usam colchetes.",
            "ConnectionPool.Connection.Database.Collection equals getConnection(name).getDatabase(db).getCollection(coll); names with spaces use brackets.",
            b_pool, ("console",)),
    Concept("projection", ["como retornar só alguns campos?", "como funciona a projeção?"], ["how do I return only some fields?", "how does projection work?"],
            "A projeção escolhe os campos retornados: 1 inclui, 0 exclui; _id vem por padrão.",
            "A projection picks returned fields: 1 includes, 0 excludes; _id is included by default.", b_projection),
    Concept("pagination", ["como paginar resultados?", "como usar skip e limit?"], ["how do I paginate results?", "how do skip and limit work?"],
            "skip descarta documentos e limit limita o retorno; ordene por um campo estável para paginar.",
            "skip discards documents and limit caps the result; sort by a stable field to paginate.", b_pagination),
    Concept("expr", ["como comparar dois campos do mesmo documento?", "para que serve o $expr?"], ["how do I compare two fields of the same document?", "what is $expr for?"],
            "$expr permite usar expressões de agregação no filtro, como comparar dois campos do documento.",
            "$expr lets a query use aggregation expressions, such as comparing two fields of the document.", b_expr),
    Concept("date_trunc", ["como agrupar por mês?", "o que faz o $dateTrunc?"], ["how do I group by month?", "what does $dateTrunc do?"],
            "$dateTrunc trunca datas para a unidade (dia, mês) e facilita agrupar séries temporais.",
            "$dateTrunc truncates dates to a unit (day, month), which makes grouping time series easy.", b_date_trunc, ("console", "script", "aggregation")),
    Concept("array_filters", ["como atualizar só alguns itens de um array?", "para que serve arrayFilters?"],
            ["how do I update only some array elements?", "what is arrayFilters for?"],
            "arrayFilters define quais elementos do array o operador posicional $[nome] deve atualizar.",
            "arrayFilters selects which array elements the filtered positional operator $[name] updates.", b_array_filters),
    Concept("vector_search", ["como fazer busca vetorial?", "o que é $vectorSearch?"], ["how do I run a vector search?", "what is $vectorSearch?"],
            "$vectorSearch busca por similaridade de embeddings em um índice vetorial do Atlas e deve ser o primeiro estágio.",
            "$vectorSearch finds documents by embedding similarity using an Atlas vector index and must be the first stage.", b_vector),
    Concept("explain", ["como ver se a consulta usa índice?", "como analisar o plano da query?"], ["how do I check whether a query uses an index?", "how do I inspect the query plan?"],
            "explain(\"executionStats\") mostra o plano escolhido, os índices usados e os documentos examinados.",
            "explain(\"executionStats\") shows the chosen plan, the indexes used and the documents examined.", b_explain, ("script",)),
]
