"""Validação estrutural MongoDB sobre a AST (sem executar código).

Resultado:
- ``validatable=False`` quando o código usa construções fora do subconjunto conhecido;
- ``valid=False`` com lista de erros quando há operador/estágio/API inexistente ou argumento malformado.
Dialetos: ``console`` (API real do Console Jint da IDE), ``mongosh`` (shell completa) e ``json``
(modo Agregação: array JSON de estágios).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import mongo_catalog as C
from .jsparse import JsSyntaxError, Node, Unsupported, key_name, parse, walk


@dataclass
class Validation:
    validatable: bool
    valid: bool
    errors: list[str] = field(default_factory=list)
    operations: int = 0


class _Ctx:
    def __init__(self, dialect: str) -> None:
        self.dialect = dialect
        self.api = C.DIALECT_API["mongosh" if dialect == "json" else dialect]
        self.errors: list[str] = []
        self.operations = 0
        self.bindings: dict[str, str] = {}  # nome -> "connection" | "database" | "collection" | "cursor"

    def err(self, message: str) -> None:
        if len(self.errors) < 20:
            self.errors.append(message)


# -- literais ---------------------------------------------------------------------------------------------
BSON_CONSTRUCTORS = {"ObjectId", "ISODate", "NumberLong", "NumberDecimal", "NumberInt", "UUID", "CGUUID", "JUUID",
                     "GUUID", "Date", "Timestamp", "BinData", "Decimal128", "Int32", "Long", "Double", "MinKey", "MaxKey",
                     "RegExp", "String", "Number", "Boolean"}


EJSON_KEYS = {"$oid", "$date", "$numberLong", "$numberInt", "$numberDouble", "$numberDecimal", "$binary", "$uuid",
              "$regularExpression", "$timestamp", "$minKey", "$maxKey", "$symbol", "$code"}


def _is_ejson_literal(keys: list[str | None]) -> bool:
    return len(keys) == 1 and keys[0] in EJSON_KEYS


def _unparen(node: Node) -> Node:
    while node.type == "ParenthesizedExpression":
        node = node.expression
    return node


def _obj_props(node: Node) -> list[tuple[str | None, Node, Node]]:
    return [(key_name(p), p.value, p) for p in node.properties if p.type == "Property"]


def _is_string(node: Node) -> bool:
    node = _unparen(node)
    return (node.type == "Literal" and isinstance(node.get("value"), str) and node.get("regex") is None) or node.type == "TemplateLiteral"


def _is_number(node: Node) -> bool:
    node = _unparen(node)
    if node.type == "UnaryExpression" and node.operator == "-":
        node = node.argument
    return node.type == "Literal" and isinstance(node.get("value"), (int, float)) and not isinstance(node.get("value"), bool)


def _number(node: Node) -> float | None:
    node = _unparen(node)
    sign = 1
    if node.type == "UnaryExpression" and node.operator == "-":
        sign, node = -1, node.argument
    value = node.get("value") if node.type == "Literal" else None
    return sign * value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _is_regex(node: Node) -> bool:
    node = _unparen(node)
    return (node.type == "Literal" and node.get("regex") is not None) or (
        node.type in ("CallExpression", "NewExpression") and node.callee.type == "Identifier" and node.callee.name == "RegExp")


def _is_opaque_value(node: Node) -> bool:
    """Valores dinâmicos aceitos em qualquer posição de valor (variáveis, chamadas, BSON, expressões JS)."""
    node = _unparen(node)
    return node.type in ("Identifier", "MemberExpression", "CallExpression", "NewExpression", "TemplateLiteral",
                         "BinaryExpression", "ConditionalExpression", "LogicalExpression", "UnaryExpression",
                         "ArrowFunctionExpression", "FunctionExpression")


# -- filtros ------------------------------------------------------------------------------------------------
def validate_filter(node: Node, ctx: _Ctx, where: str) -> None:
    node = _unparen(node)
    if node.type != "ObjectExpression":
        if not (node.type in ("Identifier", "MemberExpression", "CallExpression")):
            ctx.err(f"{where}: filtro deve ser objeto")
        return
    for key, value, _ in _obj_props(node):
        if key is None:
            continue
        if key.startswith("$"):
            if key not in C.TOP_LEVEL_QUERY:
                ctx.err(f"{where}: operador de topo desconhecido {key}")
            elif key in C.LOGICAL_OPERATORS:
                value = _unparen(value)
                if value.type != "ArrayExpression" or not value.elements:
                    ctx.err(f"{where}: {key} exige array não vazio")
                else:
                    for item in value.elements:
                        if item is not None:
                            validate_filter(item, ctx, where + "." + key)
            elif key == "$expr":
                validate_expression(value, ctx, where + ".$expr")
            elif key == "$text":
                value = _unparen(value)
                if value.type != "ObjectExpression" or "$search" not in {k for k, _, _ in _obj_props(value)}:
                    ctx.err(f"{where}: $text exige {{ $search }}")
        else:
            validate_field_condition(value, ctx, f"{where}.{key}")


def validate_field_condition(node: Node, ctx: _Ctx, where: str) -> None:
    node = _unparen(node)
    if node.type != "ObjectExpression":
        return  # igualdade literal / regex / valor dinâmico
    props = _obj_props(node)
    keys = [k for k, _, _ in props if k is not None]
    dollar = [k for k in keys if k.startswith("$")]
    if not dollar or _is_ejson_literal(keys):
        return  # igualdade com subdocumento ou literal Extended JSON
    if len(dollar) != len(keys):
        ctx.err(f"{where}: mistura operadores e campos")
        return
    for key, value, _ in props:
        value = _unparen(value)
        if key not in C.QUERY_OPERATORS:
            ctx.err(f"{where}: operador de consulta desconhecido {key}")
        elif key in ("$in", "$nin", "$all") and value.type != "ArrayExpression" and not _is_opaque_value(value):
            ctx.err(f"{where}: {key} exige array")
        elif key == "$exists" and not (value.type == "Literal" and value.get("value") in (True, False, 0, 1)):
            ctx.err(f"{where}: $exists exige booleano")
        elif key == "$regex" and not (_is_string(value) or _is_regex(value) or _is_opaque_value(value)):
            ctx.err(f"{where}: $regex exige string ou regex")
        elif key == "$options" and ("$regex" not in keys or not _is_string(value)):
            ctx.err(f"{where}: $options exige $regex e string")
        elif key == "$size" and not (_is_number(value) or _is_opaque_value(value)):
            ctx.err(f"{where}: $size exige número")
        elif key == "$elemMatch":
            if value.type != "ObjectExpression":
                ctx.err(f"{where}: $elemMatch exige objeto")
            else:
                inner = [k for k, _, _ in _obj_props(value) if k]
                if inner and all(k.startswith("$") for k in inner):
                    validate_field_condition(value, ctx, where + ".$elemMatch")
                else:
                    validate_filter(value, ctx, where + ".$elemMatch")
        elif key == "$not" and not (value.type == "ObjectExpression" or _is_regex(value)):
            ctx.err(f"{where}: $not exige objeto de operador ou regex")
        elif key == "$not" and value.type == "ObjectExpression":
            validate_field_condition(value, ctx, where + ".$not")


# -- expressões de agregação ------------------------------------------------------------------------------
def validate_expression(node: Node, ctx: _Ctx, where: str, depth: int = 0) -> None:
    node = _unparen(node)
    if depth > 40:
        return
    if node.type == "ArrayExpression":
        for item in node.elements:
            if item is not None and item.type != "SpreadElement":
                validate_expression(item, ctx, where, depth + 1)
    elif node.type == "ObjectExpression":
        if _is_ejson_literal([k for k, _, _ in _obj_props(node)]):
            return
        for key, value, _ in _obj_props(node):
            if key is None:
                continue
            if key.startswith("$"):
                if key not in C.EXPRESSION_OPERATORS:
                    ctx.err(f"{where}: operador de expressão desconhecido {key}")
                    continue
                if key == "$literal":
                    continue
            validate_expression(value, ctx, f"{where}.{key}", depth + 1)


def validate_projection(node: Node, ctx: _Ctx, where: str) -> None:
    node = _unparen(node)
    if node.type != "ObjectExpression":
        if not _is_opaque_value(node) and not (node.type == "Literal" and node.get("value") is None):
            ctx.err(f"{where}: projeção deve ser objeto")
        return
    for key, value, _ in _obj_props(node):
        if key is not None and key.startswith("$"):
            ctx.err(f"{where}: campo de projeção não pode começar com $ ({key})")
        value = _unparen(value)
        if value.type == "ObjectExpression":
            for inner, inner_value, _ in _obj_props(value):
                if inner and inner.startswith("$") and inner not in C.EXPRESSION_OPERATORS | {"$slice", "$elemMatch"}:
                    ctx.err(f"{where}.{key}: operador de projeção desconhecido {inner}")
            validate_expression(value, ctx, f"{where}.{key}")


# -- pipeline -------------------------------------------------------------------------------------------------
def validate_pipeline(node: Node, ctx: _Ctx, where: str, nested: bool = False) -> None:
    node = _unparen(node)
    if node.type != "ArrayExpression":
        if not _is_opaque_value(node):
            ctx.err(f"{where}: pipeline deve ser array")
        return
    for index, stage in enumerate(node.elements):
        if stage is None:
            ctx.err(f"{where}[{index}]: estágio vazio")
            continue
        stage = _unparen(stage)
        if stage.type in ("Identifier", "MemberExpression", "CallExpression", "SpreadElement"):
            continue
        if stage.type != "ObjectExpression":
            ctx.err(f"{where}[{index}]: estágio deve ser objeto")
            continue
        props = _obj_props(stage)
        if len(props) != 1 or props[0][0] is None:
            ctx.err(f"{where}[{index}]: estágio deve ter exatamente uma chave")
            continue
        name, value, _ = props[0]
        at = f"{where}[{index}].{name}"
        if name not in C.STAGES:
            ctx.err(f"{at}: estágio desconhecido")
            continue
        if ctx.dialect == "console" and name in C.CONSOLE_FORBIDDEN_STAGES:
            ctx.err(f"{at}: estágio de escrita rejeitado pelo Console da IDE")
        if name in C.FIRST_STAGE_ONLY and (index != 0 or nested and name != "$search"):
            if index != 0:
                ctx.err(f"{at}: deve ser o primeiro estágio")
        validate_stage(name, value, ctx, at)


def _require_keys(node: Node, required: set[str], ctx: _Ctx, where: str, allowed: set[str] | None = None) -> dict[str, Node]:
    node = _unparen(node)
    if node.type != "ObjectExpression":
        ctx.err(f"{where}: exige objeto")
        return {}
    props = {k: v for k, v, _ in _obj_props(node) if k is not None}
    missing = required - props.keys()
    if missing:
        ctx.err(f"{where}: faltando {sorted(missing)}")
    if allowed is not None:
        extra = set(props) - allowed
        if extra:
            ctx.err(f"{where}: chaves desconhecidas {sorted(extra)}")
    return props


def validate_stage(name: str, value: Node, ctx: _Ctx, where: str) -> None:
    value = _unparen(value)
    if name == "$match":
        validate_filter(value, ctx, where)
    elif name == "$group":
        props = _require_keys(value, {"_id"}, ctx, where)
        for key, expr in props.items():
            if key == "_id":
                validate_expression(expr, ctx, where + "._id")
                continue
            expr = _unparen(expr)
            inner = _obj_props(expr) if expr.type == "ObjectExpression" else []
            if len(inner) != 1 or inner[0][0] not in C.ACCUMULATORS:
                ctx.err(f"{where}.{key}: campo de $group exige um acumulador")
            else:
                validate_expression(inner[0][1], ctx, f"{where}.{key}")
    elif name in ("$project",):
        validate_projection(value, ctx, where)
    elif name in ("$set", "$addFields"):
        if value.type != "ObjectExpression":
            ctx.err(f"{where}: exige objeto")
        else:
            validate_expression(value, ctx, where)
    elif name == "$unset":
        if not (_is_string(value) or (value.type == "ArrayExpression" and all(e is not None and _is_string(e) for e in value.elements))):
            ctx.err(f"{where}: exige string ou array de strings")
    elif name == "$sort":
        if value.type != "ObjectExpression" or not value.properties:
            ctx.err(f"{where}: exige objeto não vazio")
        else:
            for key, direction, _ in _obj_props(value):
                direction = _unparen(direction)
                if _number(direction) not in (1, -1) and not (direction.type == "ObjectExpression"):
                    ctx.err(f"{where}.{key}: direção deve ser 1 ou -1")
    elif name in ("$limit", "$skip"):
        number = _number(value)
        if number is None and not _is_opaque_value(value):
            ctx.err(f"{where}: exige inteiro")
        elif number is not None and (number != int(number) or number < (1 if name == "$limit" else 0)):
            ctx.err(f"{where}: inteiro fora do intervalo")
    elif name == "$count":
        if not _is_string(value):
            ctx.err(f"{where}: exige string")
    elif name == "$lookup":
        props = _require_keys(value, {"from", "as"}, ctx, where,
                              {"from", "localField", "foreignField", "as", "let", "pipeline"})
        if props and not ({"localField", "foreignField"} <= props.keys() or "pipeline" in props):
            ctx.err(f"{where}: exige localField/foreignField ou pipeline")
        if "pipeline" in props:
            validate_pipeline(props["pipeline"], ctx, where + ".pipeline", nested=True)
    elif name == "$unwind":
        if _is_string(value):
            raw = _unparen(value).get("value")
            if isinstance(raw, str) and not raw.startswith("$"):
                ctx.err(f"{where}: caminho deve começar com $")
        else:
            props = _require_keys(value, {"path"}, ctx, where, {"path", "includeArrayIndex", "preserveNullAndEmptyArrays"})
            path = props.get("path")
            if path is not None and _is_string(path) and not str(_unparen(path).get("value", "$")).startswith("$"):
                ctx.err(f"{where}.path: caminho deve começar com $")
    elif name == "$facet":
        if value.type != "ObjectExpression" or not value.properties:
            ctx.err(f"{where}: exige objeto não vazio")
        else:
            for key, pipeline, _ in _obj_props(value):
                validate_pipeline(pipeline, ctx, f"{where}.{key}", nested=True)
                pipeline = _unparen(pipeline)
                if pipeline.type == "ArrayExpression":
                    for stage in pipeline.elements:
                        if stage is not None and stage.type == "ObjectExpression":
                            inner = key_name(stage.properties[0]) if stage.properties and stage.properties[0].type == "Property" else None
                            if inner in ("$facet", "$out", "$merge", "$search", "$searchMeta"):
                                ctx.err(f"{where}.{key}: {inner} não permitido dentro de $facet")
    elif name in ("$replaceRoot",):
        props = _require_keys(value, {"newRoot"}, ctx, where, {"newRoot"})
        if "newRoot" in props:
            validate_expression(props["newRoot"], ctx, where + ".newRoot")
    elif name in ("$replaceWith", "$sortByCount"):
        validate_expression(value, ctx, where)
    elif name == "$sample":
        _require_keys(value, {"size"}, ctx, where, {"size"})
    elif name == "$search":
        validate_search(value, ctx, where)
    elif name == "$searchMeta":
        validate_search(value, ctx, where, meta=True)
    elif name == "$unionWith":
        if not _is_string(value):
            props = _require_keys(value, {"coll"}, ctx, where, {"coll", "pipeline"})
            if "pipeline" in props:
                validate_pipeline(props["pipeline"], ctx, where + ".pipeline", nested=True)
    elif name == "$bucket":
        _require_keys(value, {"groupBy", "boundaries"}, ctx, where, {"groupBy", "boundaries", "default", "output"})


def validate_search(node: Node, ctx: _Ctx, where: str, meta: bool = False) -> None:
    node = _unparen(node)
    if node.type != "ObjectExpression":
        ctx.err(f"{where}: exige objeto")
        return
    props = {k: v for k, v, _ in _obj_props(node) if k is not None}
    operators = [k for k in props if k in C.SEARCH_OPERATORS]
    unknown = [k for k in props if k not in C.SEARCH_OPERATORS and k not in C.SEARCH_OPTIONS]
    if unknown:
        ctx.err(f"{where}: chaves desconhecidas {unknown}")
    if len(operators) != 1 and not (meta and "facet" in props):
        ctx.err(f"{where}: exige exatamente um operador Atlas Search")
    if "index" in props and not _is_string(props["index"]):
        ctx.err(f"{where}.index: exige string")
    for op in operators:
        validate_search_operator(op, props[op], ctx, f"{where}.{op}")


def validate_search_operator(op: str, node: Node, ctx: _Ctx, where: str) -> None:
    node = _unparen(node)
    if node.type != "ObjectExpression":
        ctx.err(f"{where}: exige objeto")
        return
    props = {k: v for k, v, _ in _obj_props(node) if k is not None}
    if op == "compound":
        if not props.keys() & {"must", "mustNot", "should", "filter"}:
            ctx.err(f"{where}: compound exige must/should/filter/mustNot")
        for key, value in props.items():
            if key not in C.COMPOUND_CLAUSES:
                ctx.err(f"{where}: cláusula desconhecida {key}")
                continue
            if key in ("minimumShouldMatch", "score"):
                continue
            value = _unparen(value)
            clauses = value.elements if value.type == "ArrayExpression" else [value]
            for clause in clauses:
                clause = _unparen(clause) if clause is not None else None
                if clause is None or clause.type != "ObjectExpression":
                    ctx.err(f"{where}.{key}: cláusula deve ser objeto de operador")
                    continue
                inner = [(k, v) for k, v, _ in _obj_props(clause) if k is not None]
                if len(inner) != 1 or inner[0][0] not in C.SEARCH_OPERATORS:
                    ctx.err(f"{where}.{key}: exige um operador Atlas Search por cláusula")
                else:
                    validate_search_operator(inner[0][0], inner[0][1], ctx, f"{where}.{key}.{inner[0][0]}")
        return
    required = C.SEARCH_OPERATOR_REQUIRED.get(op, set())
    missing = required - props.keys()
    if missing:
        ctx.err(f"{where}: faltando {sorted(missing)}")
    extra = set(props) - C.SEARCH_OPERATOR_KEYS
    if extra:
        ctx.err(f"{where}: chaves desconhecidas {sorted(extra)}")
    if op == "range" and not props.keys() & {"gt", "gte", "lt", "lte"}:
        ctx.err(f"{where}: range exige gt/gte/lt/lte")


# -- updates, documentos e índices ------------------------------------------------------------------------------
def validate_update(node: Node, ctx: _Ctx, where: str) -> None:
    node = _unparen(node)
    if node.type == "ArrayExpression":
        validate_pipeline(node, ctx, where)
        for stage in node.elements:
            if stage is not None and stage.type == "ObjectExpression" and stage.properties:
                name = key_name(stage.properties[0])
                if name not in {"$set", "$addFields", "$unset", "$project", "$replaceRoot", "$replaceWith"}:
                    ctx.err(f"{where}: estágio {name} não permitido em update com pipeline")
        return
    if node.type != "ObjectExpression":
        if not _is_opaque_value(node):
            ctx.err(f"{where}: update deve ser objeto ou pipeline")
        return
    props = _obj_props(node)
    if not props:
        ctx.err(f"{where}: update vazio")
    for key, value, _ in props:
        if key is None:
            continue
        if not key.startswith("$"):
            ctx.err(f"{where}: update exige operadores ($set...), encontrado campo {key}")
            continue
        if key not in C.UPDATE_OPERATORS:
            ctx.err(f"{where}: operador de update desconhecido {key}")
            continue
        value = _unparen(value)
        if value.type != "ObjectExpression":
            ctx.err(f"{where}.{key}: exige objeto")
            continue
        for field_name, field_value, _ in _obj_props(value):
            if field_name and field_name.startswith("$"):
                ctx.err(f"{where}.{key}: campo inválido {field_name}")
            field_value = _unparen(field_value)
            if key in ("$push", "$addToSet") and field_value.type == "ObjectExpression":
                inner = [k for k, _, _ in _obj_props(field_value) if k]
                if any(k.startswith("$") for k in inner) and not set(inner) <= C.UPDATE_MODIFIERS:
                    ctx.err(f"{where}.{key}.{field_name}: modificador desconhecido")
            if key in ("$inc", "$mul") and not (_is_number(field_value) or _is_opaque_value(field_value)):
                ctx.err(f"{where}.{key}.{field_name}: exige número")
            if key == "$pull" and field_value.type == "ObjectExpression":
                inner = [k for k, _, _ in _obj_props(field_value) if k]
                if inner and all(k.startswith("$") for k in inner):
                    validate_field_condition(field_value, ctx, f"{where}.$pull.{field_name}")
                else:
                    validate_filter(field_value, ctx, f"{where}.$pull.{field_name}")


def validate_document(node: Node, ctx: _Ctx, where: str) -> None:
    node = _unparen(node)
    if node.type != "ObjectExpression":
        if not _is_opaque_value(node):
            ctx.err(f"{where}: documento deve ser objeto")
        return
    for key, _, _ in _obj_props(node):
        if key is not None and key.startswith("$"):
            ctx.err(f"{where}: documento não pode ter chave de topo {key}")


def validate_options(node: Node, allowed: set[str], ctx: _Ctx, where: str) -> None:
    node = _unparen(node)
    if node.type != "ObjectExpression":
        if not _is_opaque_value(node):
            ctx.err(f"{where}: opções devem ser objeto")
        return
    for key, _, _ in _obj_props(node):
        if key is not None and key not in allowed:
            ctx.err(f"{where}: opção desconhecida {key}")


def validate_index_keys(node: Node, ctx: _Ctx, where: str) -> None:
    node = _unparen(node)
    if node.type != "ObjectExpression" or not node.properties:
        ctx.err(f"{where}: chave de índice deve ser objeto não vazio")
        return
    for key, value, _ in _obj_props(node):
        value = _unparen(value)
        literal = _number(value) if _is_number(value) else value.get("value") if value.type == "Literal" else None
        if literal not in C.INDEX_KEY_VALUES:
            ctx.err(f"{where}.{key}: tipo de índice inválido")
        if key and key.startswith("$") and key != "$**":
            ctx.err(f"{where}: campo de índice inválido {key}")


# -- cadeias de chamadas ---------------------------------------------------------------------------------------------
def _call_chain(node: Node) -> tuple[Node, list[tuple[str, list[Node] | None, Node]]]:
    """Desmonta ``a.b(x).c`` em raiz + [(nome, args|None, nó)]."""
    steps: list[tuple[str, list[Node] | None, Node]] = []
    current = _unparen(node)
    while True:
        if current.type == "CallExpression" and current.callee.type == "MemberExpression":
            member = current.callee
            name = member.property.name if not member.computed else _static_string(member.property)
            steps.append((name if name is not None else "[]", current.arguments, current))
            current = _unparen(member.object)
        elif current.type == "MemberExpression":
            name = current.property.name if not current.computed else _static_string(current.property)
            steps.append((name if name is not None else "[]", None, current))
            current = _unparen(current.object)
        else:
            break
    steps.reverse()
    return current, steps


def _static_string(node: Node) -> str | None:
    node = _unparen(node)
    return node.get("value") if node.type == "Literal" and isinstance(node.get("value"), str) else None


def _string_arg(args: list[Node] | None, index: int = 0) -> bool:
    return args is not None and len(args) > index and (_is_string(args[index]) or _is_opaque_value(args[index]))


def classify_chain(root: Node, steps: list[tuple[str, list[Node] | None, Node]], ctx: _Ctx) -> None:
    """Percorre a cadeia validando APIs e argumentos por tipo de objeto."""
    api = ctx.api
    kind: str | None
    if root.type == "Identifier":
        name = root.name
        if name == "db":
            kind = "database"
        elif name == "getConnection" and ctx.dialect == "console":
            kind = "getConnection"
        elif name == "ConnectionPool" and ctx.dialect == "console":
            kind = "pool"
        elif name in ctx.bindings:
            kind = ctx.bindings[name]
        else:
            kind = None
    else:
        kind = None
    if root.type == "CallExpression" and root.callee.type == "Identifier":
        callee = root.callee.name
        if callee == "getConnection" and ctx.dialect == "console":
            if not _string_arg(root.arguments) or len(root.arguments) != 1:
                ctx.err("getConnection exige um nome de conexão")
            kind = "connection"
        elif callee in ("getConnection", "ConnectionPool") and ctx.dialect != "console":
            ctx.err(f"{callee} só existe no Console da IDE")
    if kind is None:
        return
    for name, args, node in steps:
        if kind == "getConnection":
            kind = None
            break
        if kind == "pool":
            if args is not None:
                ctx.err("ConnectionPool.<conexão> não é função")
                return
            kind = "connection"
            continue
        if kind == "connection":
            if args is None:
                kind = "database"
            elif name == "getDatabase":
                if len(args) != 1 or not _string_arg(args):
                    ctx.err("getDatabase exige um nome de banco")
                kind = "database"
            else:
                ctx.err(f"conexão não possui método {name}()" + (" (use getDatabase)" if name.lower() == "getdatabase" else ""))
                return
            continue
        if kind == "database":
            if args is None:
                if name in api["database"]:
                    ctx.err(f"{name} é método e exige chamada")
                    return
                kind = "collection"
            elif name in ("getCollection", "getSiblingDB"):
                if len(args) != 1 or not _string_arg(args):
                    ctx.err(f"{name} exige um nome")
                kind = "collection" if name == "getCollection" else "database"
            elif name in api["database"]:
                ctx.operations += 1
                if name == "aggregate":
                    kind = "cursor"
                else:
                    return
            else:
                ctx.err(f"banco não possui método {name}()")
                return
            continue
        if kind == "collection":
            if args is None:
                if ctx.dialect == "mongosh" and name not in api["collection"]:
                    kind = "collection"  # db.a.b é coleção "a.b" no mongosh
                    continue
                ctx.err(f"acesso inválido .{name} em coleção")
                return
            if name not in api["collection"]:
                ctx.err(f"coleção não possui método {name}()" + ("" if ctx.dialect == "mongosh" else " no Console da IDE"))
                return
            ctx.operations += 1
            validate_collection_call(name, args, ctx)
            if name in ("find", "aggregate"):
                kind = "cursor"
            elif name == "explain":
                kind = "explain"
            else:
                kind = "value"
            continue
        if kind == "explain":
            if args is None or name not in C.EXPLAIN_METHODS:
                ctx.err(f"explain() não possui {name}")
                return
            validate_collection_call(name, args, ctx)
            kind = "cursor"
            continue
        if kind == "cursor":
            if args is None:
                ctx.err(f"cursor não possui propriedade {name}")
                return
            if name not in api["cursor"]:
                ctx.err(f"cursor não possui método {name}()")
                return
            validate_cursor_call(name, args, ctx)
            kind = "cursor" if name not in ("toArray", "forEach", "map", "itcount", "count", "size", "hasNext", "next", "explain") else "value"
            continue
        return


def validate_collection_call(name: str, args: list[Node], ctx: _Ctx) -> None:
    count = len(args)
    at = f"{name}()"
    update_options = C.CONSOLE_UPDATE_OPTIONS if ctx.dialect == "console" else C.UPDATE_OPTIONS
    if name in ("find", "findOne"):
        if count > (2 if ctx.dialect == "console" and name == "findOne" else 3):
            ctx.err(f"{at}: argumentos demais")
        if count >= 1:
            validate_filter(args[0], ctx, at + ".filter")
        if count >= 2:
            validate_projection(args[1], ctx, at + ".projection")
    elif name in ("countDocuments", "deleteOne", "deleteMany"):
        if count < (0 if name == "countDocuments" else 1):
            ctx.err(f"{at}: filtro obrigatório")
        if count >= 1:
            validate_filter(args[0], ctx, at + ".filter")
    elif name == "distinct":
        if count < 1 or not _is_string(args[0]):
            ctx.err(f"{at}: exige nome de campo")
        if count >= 2:
            validate_filter(args[1], ctx, at + ".filter")
    elif name == "aggregate":
        if count < 1:
            ctx.err(f"{at}: pipeline obrigatório")
        else:
            validate_pipeline(args[0], ctx, at)
    elif name in ("updateOne", "updateMany", "findOneAndUpdate"):
        if count < 2:
            ctx.err(f"{at}: exige filtro e update")
            return
        validate_filter(args[0], ctx, at + ".filter")
        validate_update(args[1], ctx, at + ".update")
        if count >= 3:
            validate_options(args[2], C.FIND_AND_MODIFY_OPTIONS if name == "findOneAndUpdate" else update_options, ctx, at + ".options")
    elif name in ("replaceOne", "findOneAndReplace"):
        if count < 2:
            ctx.err(f"{at}: exige filtro e documento")
            return
        validate_filter(args[0], ctx, at + ".filter")
        validate_document(args[1], ctx, at + ".replacement")
    elif name == "insertOne":
        if count < 1:
            ctx.err(f"{at}: documento obrigatório")
        else:
            validate_document(args[0], ctx, at)
    elif name == "insertMany":
        if count < 1:
            ctx.err(f"{at}: array obrigatório")
            return
        docs = _unparen(args[0])
        if docs.type != "ArrayExpression":
            if not _is_opaque_value(docs):
                ctx.err(f"{at}: exige array")
        else:
            for doc in docs.elements:
                if doc is not None:
                    validate_document(doc, ctx, at)
    elif name == "bulkWrite":
        ops = _unparen(args[0]) if count else None
        if ops is None or ops.type != "ArrayExpression":
            ctx.err(f"{at}: exige array de operações")
            return
        for op in ops.elements:
            op = _unparen(op) if op is not None else None
            if op is None or op.type != "ObjectExpression" or len(op.properties) != 1:
                ctx.err(f"{at}: operação deve ter uma chave")
                continue
            kind = key_name(op.properties[0])
            body = op.properties[0].value
            required = {"insertOne": {"document"}, "updateOne": {"filter", "update"}, "updateMany": {"filter", "update"},
                        "replaceOne": {"filter", "replacement"}, "deleteOne": {"filter"}, "deleteMany": {"filter"}}.get(kind or "")
            if required is None:
                ctx.err(f"{at}: operação desconhecida {kind}")
                continue
            props = _require_keys(body, required, ctx, f"{at}.{kind}", required | {"upsert", "arrayFilters", "hint", "collation"})
            if "filter" in props:
                validate_filter(props["filter"], ctx, f"{at}.{kind}.filter")
            if "update" in props:
                validate_update(props["update"], ctx, f"{at}.{kind}.update")
            if "document" in props:
                validate_document(props["document"], ctx, f"{at}.{kind}.document")
    elif name == "createIndex":
        if count < 1:
            ctx.err(f"{at}: chaves obrigatórias")
            return
        validate_index_keys(args[0], ctx, at)
        if count >= 2:
            validate_options(args[1], C.INDEX_OPTIONS, ctx, at + ".options")
    elif name == "createIndexes":
        specs = _unparen(args[0]) if count else None
        if specs is None or specs.type != "ArrayExpression":
            ctx.err(f"{at}: exige array de chaves")
        else:
            for spec in specs.elements:
                if spec is not None:
                    validate_index_keys(spec, ctx, at)
    elif name == "dropIndex":
        if count != 1 or not (_is_string(args[0]) or _unparen(args[0]).type == "ObjectExpression"):
            ctx.err(f"{at}: exige nome ou chave do índice")
    elif name == "createSearchIndex":
        if count < 1:
            ctx.err(f"{at}: definição obrigatória")
    elif name in ("estimatedDocumentCount", "drop", "getIndexes", "stats", "dropIndexes"):
        pass


def validate_cursor_call(name: str, args: list[Node], ctx: _Ctx) -> None:
    at = f".{name}()"
    if name == "sort":
        if len(args) != 1:
            ctx.err(f"{at}: exige um objeto")
        else:
            validate_stage("$sort", args[0], ctx, at)
    elif name in ("limit", "skip", "batchSize", "maxTimeMS"):
        number = _number(args[0]) if args else None
        if len(args) != 1 or (number is None and not _is_opaque_value(args[0])) or (number is not None and (number < 0 or number != int(number))):
            ctx.err(f"{at}: exige inteiro não negativo")
    elif name == "project":
        if len(args) != 1:
            ctx.err(f"{at}: exige um objeto")
        else:
            validate_projection(args[0], ctx, at)
    elif name in ("toArray", "itcount", "hasNext", "next", "pretty", "size"):
        if args:
            ctx.err(f"{at}: não recebe argumentos")
    elif name in ("forEach", "map"):
        if len(args) != 1 or _unparen(args[0]).type not in ("ArrowFunctionExpression", "FunctionExpression", "Identifier", "MemberExpression"):
            ctx.err(f"{at}: exige função")


def _collect_bindings(program: Node, ctx: _Ctx) -> None:
    for node in walk(program):
        if node.type == "VariableDeclarator" and node.id.type == "Identifier" and node.init is not None:
            root, steps = _call_chain(node.init)
            probe = _Ctx(ctx.dialect)
            probe.bindings = dict(ctx.bindings)
            kind = _chain_kind(root, steps, probe)
            if kind:
                ctx.bindings[node.id.name] = kind


def _chain_kind(root: Node, steps: list[tuple[str, list[Node] | None, Node]], ctx: _Ctx) -> str | None:
    """Tipo resultante de uma cadeia (para variáveis como ``const orders = db.getCollection("orders")``)."""
    if root.type == "Identifier":
        kind = {"db": "database", "ConnectionPool": "pool"}.get(root.name) or ctx.bindings.get(root.name)
    elif root.type == "CallExpression" and root.callee.type == "Identifier" and root.callee.name == "getConnection":
        kind = "connection"
    else:
        return None
    for name, args, _ in steps:
        if kind == "pool" and args is None:
            kind = "connection"
        elif kind == "connection":
            kind = "database" if (args is None or name == "getDatabase") else None
        elif kind == "database":
            if args is None:
                kind = "collection"
            elif name == "getCollection":
                kind = "collection"
            elif name == "getSiblingDB":
                kind = "database"
            else:
                return None
        elif kind == "collection":
            kind = "cursor" if args is not None and name in ("find", "aggregate") else ("collection" if args is None else None)
        elif kind == "cursor":
            kind = "cursor" if args is not None and name in C.MONGOSH_CURSOR_METHODS - {"toArray", "forEach", "map", "itcount", "next", "hasNext", "count", "size"} else None
        else:
            return None
        if kind is None:
            return None
    return kind


def validate_js(code: str, dialect: str) -> Validation:
    try:
        program = parse(code)
    except (JsSyntaxError, Unsupported, RecursionError):
        return Validation(False, False)
    ctx = _Ctx(dialect)
    _collect_bindings(program, ctx)
    handled: set[int] = set()
    for node in walk(program):
        if id(node) in handled or node.type not in ("CallExpression", "MemberExpression"):
            continue
        root, steps = _call_chain(node)
        for _, _, inner in steps:
            handled.add(id(inner))
            if inner.type == "CallExpression":
                handled.add(id(inner.callee))
                callee = inner.callee
                while callee.type == "ParenthesizedExpression":
                    callee = callee.expression
                    handled.add(id(callee))
        if steps:
            classify_chain(root, steps, ctx)
        if node.type == "CallExpression" and node.callee.type == "Identifier":
            callee = node.callee.name
            if callee in ("getConnection", "ConnectionPool") and dialect != "console":
                ctx.err(f"{callee} só existe no Console da IDE")
            if callee == "ObjectId" and node.arguments:
                raw = _static_string(node.arguments[0])
                if raw is not None and (len(raw) != 24 or any(c not in "0123456789abcdefABCDEF" for c in raw)):
                    ctx.err("ObjectId exige 24 dígitos hexadecimais")
    for node in walk(program):
        if node.type == "MemberExpression" and not node.computed and node.object.type == "Identifier" \
                and node.object.name in ("ConnectionPull",):
            ctx.err("ConnectionPull não existe (use ConnectionPool)")
        if node.type == "Identifier" and node.name == "ConnectionPull":
            ctx.err("ConnectionPull não existe (use ConnectionPool)")
    return Validation(True, not ctx.errors, ctx.errors, ctx.operations)


def validate_json_pipeline(text: str) -> Validation:
    """Modo Agregação da IDE: texto JSON (Extended JSON) com array de estágios."""
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return Validation(True, False, [f"JSON inválido: {exc.msg}"])
    try:
        program = parse("(" + text + ")")
    except (JsSyntaxError, Unsupported):
        return Validation(False, False)
    ctx = _Ctx("json")
    statement = program.body[0] if program.body else None
    if statement is None or statement.type != "ExpressionStatement":
        return Validation(True, False, ["pipeline ausente"])
    validate_pipeline(statement.expression, ctx, "pipeline")
    return Validation(True, not ctx.errors, ctx.errors, 1)


def validate(code: str, dialect: str) -> Validation:
    return validate_json_pipeline(code) if dialect == "json" else validate_js(code, dialect)
