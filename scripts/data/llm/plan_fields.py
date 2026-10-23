#!/usr/bin/env python3
"""Checagem de CAMPOS/COLECOES de um comando de leitura REALMENTE enviado ao servidor (find/aggregate/count/distinct), com fluxo de
campos entre estagios. Opera sobre o BSON capturado pelo proxy (variaveis, funcoes e loops do JS ja foram resolvidos pelo mongosh).

check_command(cmd: dict, schema_info) -> {"unknown_fields": [...], "unknown_collections": [...], "stages": [...], "ops": [...], "unsupported": [...]}

Modelo: `avail` = {caminho: "schema"|"opaque"} (None = desconhecido/permissivo: depois de $replaceRoot, $unionWith, $densify...). Um caminho e valido se
estiver em `avail`, se algum ancestral for "opaque" ou se for ancestral de algum caminho de `avail` (objeto pai). Segmentos numericos (posicao em
array) sao ignorados. `$$var` e valores literais nunca sao tratados como campos."""
from __future__ import annotations

import re

NUM_SEG = re.compile(r"^\d+$")
LOGICAL = {"$and", "$or", "$nor"}
EXPR_ARG_STAGES = {"$replaceRoot", "$replaceWith"}


def _norm(path: str) -> str:
    return ".".join(s for s in path.split(".") if not NUM_SEG.match(s) and s != "$")


class Ctx:
    def __init__(self, info, coll_name):
        self.info = info
        self.coll_name = coll_name
        self.unknown_fields: set = set()
        self.unknown_colls: set = set()
        self.stages: list = []
        self.ops: set = set()
        self.unsupported: list = []

    def coll_fields(self, name):
        for c in self.info.schema.colls:
            if c.name == name or c.key == name:
                return {p: "schema" for p in c.fields}
        return None


def valid(path: str, avail) -> bool:
    if avail is None:
        return True
    p = _norm(path)
    if not p or p in avail or p == "_id":
        return True
    parts = p.split(".")
    for i in range(1, len(parts)):
        if avail.get(".".join(parts[:i])) == "opaque":
            return True
    pre = p + "."
    return any(k.startswith(pre) for k in avail)


def need(ctx: Ctx, path: str, avail, where=""):
    if isinstance(path, str) and path and not path.startswith("$") and not valid(path, avail):
        ctx.unknown_fields.add(_norm(path))


def refs(ctx: Ctx, expr, avail):
    """Verifica "$campo" em qualquer lugar de uma expressao de agregacao."""
    if isinstance(expr, str):
        if expr.startswith("$") and not expr.startswith("$$") and len(expr) > 1:
            need(ctx, expr[1:], avail)
    elif isinstance(expr, list):
        for x in expr:
            refs(ctx, x, avail)
    elif isinstance(expr, dict):
        for k, v in expr.items():
            if k == "$literal":
                continue
            if isinstance(k, str) and k.startswith("$"):
                ctx.ops.add(k)
            if k in ("$map", "$filter", "$reduce", "$let") and isinstance(v, dict):
                for kk in ("input", "initialValue", "vars"):
                    if kk in v:
                        if kk == "vars" and isinstance(v[kk], dict):
                            for vv in v[kk].values():
                                refs(ctx, vv, avail)
                        else:
                            refs(ctx, v[kk], avail)
                for kk in ("in", "cond", "limit"):
                    if kk in v:
                        refs(ctx, v[kk], avail)
                continue
            refs(ctx, v, avail)


def match_doc(ctx: Ctx, q, avail, prefix: str = ""):
    """Filtro de consulta ($match / find / count): chaves sao campos; $expr e expressao."""
    if not isinstance(q, dict):
        return
    for k, v in q.items():
        if k in LOGICAL:
            for sub in v if isinstance(v, list) else []:
                match_doc(ctx, sub, avail, prefix)
            ctx.ops.add(k)
        elif k == "$expr":
            refs(ctx, v, avail)
            ctx.ops.add(k)
        elif k in ("$text", "$where", "$comment", "$jsonSchema", "$natural"):
            ctx.ops.add(k)
        elif isinstance(k, str) and k.startswith("$"):
            ctx.ops.add(k)
        else:
            path = (prefix + "." + k) if prefix else k
            need(ctx, path, avail)
            op_dict(ctx, v, path, avail)


def op_dict(ctx: Ctx, v, path: str, avail):
    if isinstance(v, dict):
        for ok, ov in v.items():
            if isinstance(ok, str) and ok.startswith("$"):
                ctx.ops.add(ok)
                if ok == "$elemMatch" and isinstance(ov, dict):
                    # subchaves relativas ao elemento do array (ou operadores diretos)
                    for ek, ev in ov.items():
                        if isinstance(ek, str) and ek.startswith("$"):
                            ctx.ops.add(ek)
                            if ek in LOGICAL:
                                for sub in ev if isinstance(ev, list) else []:
                                    match_doc(ctx, sub, avail, path)
                            elif ek == "$expr":
                                refs(ctx, ev, avail)
                            else:
                                op_dict(ctx, {ek: ev}, path, avail)
                        else:
                            need(ctx, path + "." + ek, avail)
                            op_dict(ctx, ev, path + "." + ek, avail)
                elif ok == "$not":
                    op_dict(ctx, ov, path, avail)
                elif ok == "$exists" or ok == "$type":
                    pass
            else:
                # sub-documento literal comparado por igualdade ({ field: { a: 1 } }): nao ha o que checar
                pass


def project_stage(ctx: Ctx, spec, avail):
    """$project: retorna o novo avail."""
    if not isinstance(spec, dict):
        return avail
    incl = {}
    excl = set()
    computed = {}
    has_incl = False
    for k, v in spec.items():
        if isinstance(v, bool) or (isinstance(v, (int, float)) and not isinstance(v, bool)):
            if v:
                need(ctx, k, avail)
                incl[k] = "opaque"
                if k != "_id":
                    has_incl = True
            else:
                excl.add(k)
                if k != "_id":
                    need(ctx, k, avail)
        elif isinstance(v, str) and v.startswith("$") and not v.startswith("$$"):
            refs(ctx, v, avail)
            computed[k] = "opaque"
            has_incl = True
        else:
            refs(ctx, v, avail)
            computed[k] = "opaque"
            has_incl = True
    if has_incl or computed:
        out = {"_id": "schema"} if "_id" not in excl else {}
        out.update(incl)
        out.update(computed)
        return out
    if avail is None:
        return None
    return {k: v for k, v in avail.items() if k not in excl and not any(k.startswith(e + ".") for e in excl)}


def addfields_stage(ctx: Ctx, spec, avail):
    if not isinstance(spec, dict):
        return avail
    for v in spec.values():
        refs(ctx, v, avail)
    if avail is None:
        return None
    out = dict(avail)
    for k in spec:
        out[_norm(k)] = "opaque"
    return out


def group_stage(ctx: Ctx, spec, avail):
    if not isinstance(spec, dict):
        return avail
    out = {"_id": "opaque" if isinstance(spec.get("_id"), (dict, list)) else "schema"}
    if "_id" not in spec:
        ctx.unsupported.append("group_sem_id")
    refs(ctx, spec.get("_id"), avail)
    for k, v in spec.items():
        if k == "_id":
            continue
        refs(ctx, v, avail)
        out[k] = "opaque"
    return out


def run_pipeline(ctx: Ctx, pipeline, avail, depth=0):
    if not isinstance(pipeline, list) or depth > 6:
        return None
    for st in pipeline:
        if not isinstance(st, dict) or not st:
            continue
        name, spec = next(iter(st.items()))
        ctx.stages.append(name)
        if name == "$match":
            match_doc(ctx, spec, avail)
        elif name in ("$project",):
            avail = project_stage(ctx, spec, avail)
        elif name in ("$addFields", "$set"):
            avail = addfields_stage(ctx, spec, avail)
        elif name == "$unset":
            for f in ([spec] if isinstance(spec, str) else spec if isinstance(spec, list) else []):
                need(ctx, f, avail)
            if avail is not None:
                rm = set([spec] if isinstance(spec, str) else spec if isinstance(spec, list) else [])
                avail = {k: v for k, v in avail.items() if k not in rm}
        elif name == "$group":
            avail = group_stage(ctx, spec, avail)
        elif name == "$sort":
            for k in spec if isinstance(spec, dict) else []:
                if not (isinstance(spec[k], dict)):          # {$meta: ...}
                    need(ctx, k, avail)
        elif name in ("$limit", "$skip", "$sample"):
            pass
        elif name == "$unwind":
            p = spec if isinstance(spec, str) else (spec.get("path") if isinstance(spec, dict) else None)
            if isinstance(p, str) and p.startswith("$"):
                need(ctx, p[1:], avail)
            if isinstance(spec, dict) and spec.get("includeArrayIndex") and avail is not None:
                avail = {**avail, spec["includeArrayIndex"]: "schema"}
        elif name == "$lookup":
            if not isinstance(spec, dict):
                continue
            frm = spec.get("from")
            ff = ctx.coll_fields(frm) if isinstance(frm, str) else None
            if isinstance(frm, str) and ff is None:
                ctx.unknown_colls.add(frm)
            if "localField" in spec:
                need(ctx, spec["localField"], avail)
            if "foreignField" in spec and ff is not None:
                need(ctx, spec["foreignField"], ff)
            if isinstance(spec.get("let"), dict):
                for v in spec["let"].values():
                    refs(ctx, v, avail)
            if "pipeline" in spec:
                run_pipeline(ctx, spec["pipeline"], ff, depth + 1)
            if avail is not None and isinstance(spec.get("as"), str):
                avail = {**avail, _norm(spec["as"]): "opaque"}
        elif name == "$graphLookup":
            if isinstance(spec, dict):
                frm = spec.get("from")
                ff = ctx.coll_fields(frm) if isinstance(frm, str) else None
                if isinstance(frm, str) and ff is None:
                    ctx.unknown_colls.add(frm)
                refs(ctx, spec.get("startWith"), avail)
                if ff is not None:
                    for kk in ("connectFromField", "connectToField"):
                        if isinstance(spec.get(kk), str):
                            need(ctx, spec[kk], ff)
                if avail is not None and isinstance(spec.get("as"), str):
                    avail = {**avail, _norm(spec["as"]): "opaque"}
        elif name == "$count":
            avail = {spec: "schema"} if isinstance(spec, str) else avail
        elif name == "$sortByCount":
            refs(ctx, spec, avail)
            avail = {"_id": "schema", "count": "schema"}
        elif name in ("$bucket", "$bucketAuto"):
            if isinstance(spec, dict):
                refs(ctx, spec.get("groupBy"), avail)
                out = {"_id": "opaque"}
                outs = spec.get("output")
                if isinstance(outs, dict):
                    for k, v in outs.items():
                        refs(ctx, v, avail)
                        out[k] = "opaque"
                else:
                    out["count"] = "schema"
                avail = out
        elif name == "$facet":
            out = {}
            if isinstance(spec, dict):
                for k, sub in spec.items():
                    run_pipeline(ctx, sub, avail, depth + 1)
                    out[k] = "opaque"
            avail = out
        elif name == "$setWindowFields":
            if isinstance(spec, dict):
                refs(ctx, spec.get("partitionBy"), avail)
                sb = spec.get("sortBy")
                if isinstance(sb, dict):
                    for k in sb:
                        need(ctx, k, avail)
                outs = spec.get("output")
                if isinstance(outs, dict):
                    for k, v in outs.items():
                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                if kk != "window":
                                    refs(ctx, {kk: vv}, avail)
                    if avail is not None:
                        avail = {**avail, **{_norm(k): "opaque" for k in outs}}
        elif name in EXPR_ARG_STAGES:
            refs(ctx, spec.get("newRoot") if isinstance(spec, dict) and name == "$replaceRoot" else spec, avail)
            avail = None
        elif name == "$unionWith":
            coll = spec if isinstance(spec, str) else spec.get("coll") if isinstance(spec, dict) else None
            ff = ctx.coll_fields(coll) if isinstance(coll, str) else None
            if isinstance(coll, str) and ff is None:
                ctx.unknown_colls.add(coll)
            if isinstance(spec, dict) and "pipeline" in spec:
                run_pipeline(ctx, spec["pipeline"], ff, depth + 1)
            avail = None
        elif name == "$redact":
            refs(ctx, spec, avail)
        elif name in ("$densify", "$fill", "$geoNear"):
            avail = None
        else:
            ctx.unsupported.append(name)
            avail = None
    return avail


def check_command(cmd: dict, info) -> dict:
    name = next(iter(cmd))
    coll = cmd[name] if isinstance(cmd.get(name), str) else None
    ctx = Ctx(info, coll)
    base = ctx.coll_fields(coll) if coll else None
    if coll and base is None:
        ctx.unknown_colls.add(coll)
    n = name.lower()
    if n == "find":
        match_doc(ctx, cmd.get("filter"), base)
        proj = cmd.get("projection")
        if isinstance(proj, dict):
            project_stage(ctx, proj, base)
        sort = cmd.get("sort")
        if isinstance(sort, dict):
            for k in sort:
                if not isinstance(sort[k], dict):
                    need(ctx, k, base)
        ctx.stages.append("find")
    elif n == "aggregate":
        run_pipeline(ctx, cmd.get("pipeline"), base)
        ctx.stages.insert(0, "aggregate")
    elif n == "count":
        match_doc(ctx, cmd.get("query"), base)
        ctx.stages.append("count")
    elif n == "distinct":
        need(ctx, cmd.get("key", ""), base)
        match_doc(ctx, cmd.get("query"), base)
        ctx.stages.append("distinct")
    return {"unknown_fields": sorted(ctx.unknown_fields), "unknown_collections": sorted(ctx.unknown_colls), "stages": ctx.stages,
            "ops": sorted(ctx.ops), "unsupported": ctx.unsupported, "command": n}
