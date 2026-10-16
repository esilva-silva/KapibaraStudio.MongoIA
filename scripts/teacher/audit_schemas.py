#!/usr/bin/env python3
"""Auditoria (SOMENTE RELATORIO) dos schemas de dominios com mais erros de campo do professor (it.1 -> it.2).

  python -I scripts/teacher/audit_schemas.py --domains saas iot gaming logistics \
      --predictions Repos/KapiTeacher-Mongo-7B/reports/eval/sft_v2c-final/predictions.jsonl \
      --out Repos/KapiTeacher-Mongo-7B/reports/schema_audit_v2d.md

Le data/schemas/<dominio>/*.json (via gen_snippets.load_schemas) e as predicoes do adapter; NAO escreve em data/schemas nem nos
fixtures. Para cada predicao com `exec.field_hallucination` classifica o nome inventado contra o schema da colecao:
variante de estilo (caixa/underscore), erro de aninhamento (folha existe em outro caminho), quase-igual (edicao <= 2),
campo de colecao irma, campo de outro schema do dominio, ou invencao.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))
from gen_snippets import load_schemas  # noqa: E402

GENERIC_CONTAINERS = {"meta", "info", "extra", "attrs", "attributes", "stats", "data", "details", "props", "properties", "settings",
                      "config", "options", "flags", "profile", "social", "prefs", "metrics", "tags", "labels", "seo", "audit"}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def lev(a: str, b: str, limit: int = 3) -> int:
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def classify(bad: str, coll, sch, siblings: dict, domain_leaves: dict) -> str:
    paths = list(coll.fields)
    leaves = {p.rsplit(".", 1)[-1] for p in paths}
    bad_leaf = bad.rsplit(".", 1)[-1]
    if bad in coll.fields:
        return "existe_no_schema(erro de tipo/estrutura)"
    if any(norm(p) == norm(bad) for p in paths):
        return "variante_de_estilo (caixa/underscore)"
    if bad_leaf in leaves:
        return "aninhamento_errado (folha existe em outro caminho)"
    if any(norm(l) == norm(bad_leaf) for l in leaves):
        return "variante_de_estilo (caixa/underscore)"
    if any(lev(norm(bad_leaf), norm(l)) <= 2 for l in leaves if len(l) > 3):
        return "quase_igual (edicao <= 2)"
    if "." in bad and ".".join(bad.split(".")[:-1]) in coll.fields:
        return "filho_inventado_de_objeto_existente"
    if bad_leaf in siblings:
        return "campo_de_colecao_irma"
    if bad_leaf in domain_leaves:
        return "vocabulario_de_outro_schema_do_dominio"
    return "invencao"


def audit_schema(sch) -> dict:
    out = {"collections": len(sch.colls), "fields": 0, "max_depth": 0, "rare": [], "similar": [], "generic": [], "deep": [], "style": sch.name_style}
    leaf_to_paths: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    norm_to_names: dict[str, set] = collections.defaultdict(set)
    for coll in sch.colls:
        for path, f in coll.fields.items():
            out["fields"] += 1
            depth = path.count(".") + 1
            out["max_depth"] = max(out["max_depth"], depth)
            leaf = path.rsplit(".", 1)[-1]
            leaf_to_paths[leaf].append((coll.name, path))
            norm_to_names[norm(leaf)].add(leaf)
            if depth >= 4:
                out["deep"].append(f"{coll.name}.{path}")
            if (f.presence or 1.0) < 0.35 and path != "_id":
                out["rare"].append(f"{coll.name}.{path} (p={f.presence})")
            if leaf.lower() in GENERIC_CONTAINERS and f.t in ("object", "array"):
                inner = sorted(p.split(".")[-1] for p in coll.fields if p.startswith(path + ".") and p.count(".") == path.count(".") + 1)
                out["generic"].append(f"{coll.name}.{path} -> {{{', '.join(inner[:6])}}}")
    for leaf, where in leaf_to_paths.items():
        in_coll = collections.defaultdict(list)
        for cname, path in where:
            in_coll[cname].append(path)
        for cname, paths in in_coll.items():
            if len(paths) > 1:
                out["similar"].append(f"`{leaf}` repetido em {cname}: {', '.join(paths)}")
    for key, names in norm_to_names.items():
        if len(names) > 1:
            out["similar"].append(f"variantes de estilo no schema: {sorted(names)}")
    # nomes quase iguais na MESMA colecao (edicao <= 2)
    for coll in sch.colls:
        leaves = sorted({p.rsplit(".", 1)[-1] for p in coll.fields if len(p.rsplit(".", 1)[-1]) > 3})
        for i, a in enumerate(leaves):
            for b in leaves[i + 1:]:
                if norm(a) != norm(b) and lev(norm(a), norm(b)) <= 2:
                    out["similar"].append(f"{coll.name}: `{a}` ~ `{b}`")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+", default=["saas", "iot", "gaming", "logistics"])
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    schemas = load_schemas(ROOT / "data" / "schemas")
    by_id = {s.full_id: s for s in schemas}
    preds = [json.loads(l) for l in open(a.predictions, encoding="utf-8")]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from teacher_measures import context_kinds_by_id  # noqa: E402
    missing = {p["id"] for p in preds if not p.get("context_kind")}
    if missing:
        joined = context_kinds_by_id(missing, ROOT / "data" / "validated" / "v2c")
        for p in preds:
            p.setdefault("context_kind", joined.get(p["id"]))
    L = ["# Auditoria de schemas para a 2a iteracao do professor (saas, iot, gaming, logistics)", "",
         f"Gerada por `scripts/teacher/audit_schemas.py` a partir de `{a.predictions}` ({len(preds)} predicoes do adapter `sft_v2c`) e de "
         "`data/schemas/<dominio>/*.json`. **Somente relatorio**: nenhum schema ou fixture foi alterado.", "",
         "Como ler: os schemas de `eval`/`test` do val v2c sao os que geram as predicoes; os de treino (mesmo dominio, estilos diferentes) "
         "sao o que o modelo viu. Taxas por schema tem n pequeno (~15-30 predicoes): indicio, nao medida.", ""]
    total_by_domain = collections.Counter(p["domain"] for p in preds)
    for domain in a.domains:
        dom_schemas = [s for s in schemas if s.domain == domain]
        dom_leaves = {p.rsplit(".", 1)[-1] for s in dom_schemas for c in s.colls for p in c.fields}
        dp = [p for p in preds if p["domain"] == domain]
        bad = [p for p in dp if p["exec"]["field_hallucination"]]
        ctxs = collections.Counter(p.get("context_kind") or "?" for p in bad)
        L += [f"## {domain}", "",
              f"- Predicoes no val: {len(dp)}; campo inexistente: **{len(bad)} ({100 * len(bad) / max(1, len(dp)):.1f}%)** "
              f"(geral {100 * sum(p['exec']['field_hallucination'] for p in preds) / len(preds):.1f}%). Schemas do dominio: "
              f"{', '.join(f'{s.schema_id} ({s.split_role or s.holdout and 'holdout' or '-'}, {s.name_style})' for s in dom_schemas)}.", ""]
        cls = collections.Counter()
        rows = []
        for p in bad:
            sch = by_id.get(p["schema_id"])
            coll = next((c for c in sch.colls if c.name == p["collection"]), None) if sch else None
            if coll is None:
                continue
            name = str(p["exec"]["error"]).split(": ", 1)[-1].strip()
            siblings = {x.rsplit(".", 1)[-1] for c in sch.colls if c is not coll for x in c.fields}
            kind = classify(name, coll, sch, siblings, dom_leaves)
            cls[kind] += 1
            rows.append((p["schema_id"], p["collection"], p["cut_kind"], name, kind))
        if cls:
            L += ["| tipo do nome inventado | n |", "|---|---:|", *[f"| {k} | {v} |" for k, v in cls.most_common()], ""]
        if rows:
            L += ["| schema | colecao | corte | nome inventado | tipo |", "|---|---|---|---|---|",
                  *[f"| {r[0]} | {r[1]} | {r[2]} | `{r[3]}` | {r[4]} |" for r in sorted(rows)[:25]], ""]
        per_schema = collections.Counter(r[0] for r in rows)
        per_coll = collections.Counter((r[0], r[1]) for r in rows)
        if per_schema:
            L += ["Falhas por schema: " + ", ".join(f"{k} ({v})" for k, v in per_schema.most_common()) + ". Colecoes mais afetadas: " +
                  ", ".join(f"{k[0]}.{k[1]} ({v})" for k, v in per_coll.most_common(6)) + ".", ""]
        for s in dom_schemas:
            au = audit_schema(s)
            L += [f"### {s.full_id} ({au['style']}; {au['collections']} colecoes, {au['fields']} campos, aninhamento max {au['max_depth']})", ""]
            for title, key, cap in (("Nomes ambiguos/similares", "similar", 8), ("Contenedores genericos (inner keys imprevisiveis)", "generic", 8),
                                    ("Aninhamento profundo (>= 4 niveis)", "deep", 6), ("Campos opcionais raros (presenca < 0,35)", "rare", 8)):
                items = au[key]
                if items:
                    L.append(f"- **{title}** ({len(items)}): " + "; ".join(items[:cap]) + (" ..." if len(items) > cap else ""))
            L.append("")
    L += ["## Sintese e recomendacoes (nao aplicadas)", "",
          "1. Contenedores genericos (`meta`, `info`, `extra`, `attrs`, `stats`, `seo`, `social`...) cujo conteudo muda por colecao concentram "
          "as invencoes `x.y`: o modelo sabe que o pai existe e inventa o filho. Em contexto `full` o `RESULT FIELDS` os lista; sem ele "
          "(none/irrelevant) nao ha como acertar: a v2d so treina esses cortes quando o caminho esta visivel no documento.",
          "2. Variantes de estilo (camel x snake, `validUntil` vs `valid_until`) entre schemas do mesmo dominio e a mesma folha em caminhos "
          "diferentes (`at`, `total`, `name`) sao fonte de `aninhamento_errado`; mitigar com exemplos de PropertyKey/FieldReferenceString "
          "(sobre-amostrados x2 na v2d), nao mudando os schemas (decisao: somente relatorio).",
          "3. Campos opcionais raros: so entram em `FieldSchema`/`RESULT FIELDS` por amostragem (`schema_subset`); quando o modelo os cita "
          "sem estarem visiveis, o corte e descartado nos contextos sem schema (v2d).", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"auditoria: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
