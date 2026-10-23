#!/usr/bin/env python3
"""Relatorio do lote v3: reports/datasets/schemas-v3-report.md (estatisticas lidas de data/schemas e do manifest).

  python scripts/data/report_schemas_v3.py [--validation validation.json]

`validation.json` (opcional, escrito a mao/por script): {"seed_seconds":..,"check":"OK ...","idempotent":true,"pytest":"..",
"dump_mb":..,"old_hash_intact":..}. Sem ele, a secao de validacao fica como "nao informada".
"""
import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3 = "v3_"


def load():
    out = []
    for p in sorted((ROOT / "data" / "schemas").glob("*/*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_path"] = p
        out.append(d)
    return out


def walk(fields, in_arr=False):
    for k, s in fields.items():
        yield k, s, in_arr
        if s["t"] == "object" and "map" not in s and "geo" not in s:
            yield from walk(s["f"], in_arr)
        elif s["t"] == "object" and "map" in s:
            yield from walk({"#v": s["map"]["value"]}, True)
        elif s["t"] == "array":
            it = s["item"]
            while it["t"] == "array":
                it = it["item"]
            if it["t"] == "object" and "map" not in it and "geo" not in it:
                yield from walk(it["f"], True)


def alt_kind(s):
    a = s["alt"][0]
    if a.get("as") == "string":
        return "ref ObjectId|string"
    if a.get("g") == "isodate":
        return "Date|string ISO"
    if a.get("g") == "numstr":
        return "numero|string numerica"
    if a["t"] in ("double", "int32"):
        return "int|double" if s["t"] != "double" else "double|int"
    if a.get("g") == "enum":
        return "bool|string Y/N"
    if a.get("g") == "word":
        return "array|escalar"
    return a["t"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", default="")
    ap.add_argument("--out", default=str(ROOT / "reports" / "datasets" / "schemas-v3-report.md"))
    a = ap.parse_args()
    allsc = load()
    v3 = [s for s in allsc if s["schema_id"].startswith(V3)]
    old = [s for s in allsc if not s["schema_id"].startswith(V3)]
    man = json.loads((ROOT / "data" / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
    ment = [e for e in man["entries"] if e["schema_id"].startswith(V3)]
    docs_by_schema = defaultdict(int)
    for e in ment:
        docs_by_schema[e["schema_id"]] += e["count"]

    L = []
    w = L.append
    w("# Schemas v3 (realistas e dificeis) - relatorio")
    w("")
    w(f"Gerado por `scripts/data/report_schemas_v3.py` a partir de `data/schemas` e `data/fixtures/manifest.json`. "
      f"Seed do lote: 20261022 (`build_schemas_v3.py`); os {len(old)} schemas antigos (seed 20261008) ficam intactos.")
    w("")
    w("## 1. Contagens")
    w("")
    roles = Counter(s["split_role"] for s in v3)
    w(f"- schemas v3: **{len(v3)}** (train {roles['train']}, eval {roles['eval']}, hom {roles['hom']}); "
      f"hom = {100 * roles['hom'] / len(v3):.1f}% (minimo pedido: 25%). Total no repositorio: {len(allsc)} schemas.")
    ncol = sum(len(s["collections"]) for s in v3)
    w(f"- colecoes: **{ncol}** ({min(len(s['collections']) for s in v3)}-{max(len(s['collections']) for s in v3)} por schema); "
      f"documentos semeados: **{sum(e['count'] for e in ment):,}** em {len({e['db'] for e in ment})} bancos `kapi_fx_<dominio>_v3_*` "
      f"(limite: 2.000.000).".replace(",", "."))
    w("")
    w("| dominio | schemas | train | eval | hom | colecoes | docs |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    bydom = defaultdict(list)
    for s in v3:
        bydom[s["domain"]].append(s)
    tot = [0] * 6
    for dom in sorted(bydom):
        ss = bydom[dom]
        r = Counter(x["split_role"] for x in ss)
        nc = sum(len(x["collections"]) for x in ss)
        nd = sum(docs_by_schema[x["schema_id"]] for x in ss)
        new = " (novo)" if dom in ("telecom", "adtech", "retail-pos", "manufacturing") else ""
        hold = " (holdout)" if ss[0]["holdout"] else ""
        w(f"| {dom}{new}{hold} | {len(ss)} | {r['train']} | {r['eval']} | {r['hom']} | {nc} | {nd:,} |".replace(",", "."))
        for i, v in enumerate((len(ss), r["train"], r["eval"], r["hom"], nc, nd)):
            tot[i] += v
    w(f"| **total** | {tot[0]} | {tot[1]} | {tot[2]} | {tot[3]} | {tot[4]} | {tot[5]:,} |".replace(",", "."))
    w("")
    w("Papeis: `hom` = homologacao (domínios inteiros `telecom` e `adtech`; nunca em train/val/test/bench dos conjuntos antigos); "
      "`eval` = education/legal (holdout, so test/bench) + ~15% dos demais por hash; `train` = o resto (cada dominio mantem >= 1 train).")
    w("")

    # --------------------------------------------------------- heterogeneidade
    w("## 2. Estatisticas de heterogeneidade e dificuldade")
    w("")
    cols = [(s, c) for s in v3 for c in s["collections"]]
    alt_kinds = Counter()
    n_fields = n_alt = n_map = n_geo = n_aa = n_rare = 0
    deep = Counter()
    maxd = []
    idtypes = Counter()
    kinds = Counter()
    idx_kinds = Counter()
    n_idx = 0
    n_val = n_warn = 0
    n_search = 0
    for s, c in cols:
        for k, f, in_arr in walk(c["fields"]):
            if k == "#v":
                continue
            n_fields += 1
            if f.get("alt"):
                n_alt += 1
                alt_kinds[alt_kind(f)] += 1
            if "map" in f:
                n_map += 1
            if "geo" in f:
                n_geo += 1
            if f["t"] == "array" and f["item"]["t"] == "array":
                n_aa += 1
            if isinstance(f["p"], float) and f["p"] < 0.1:
                n_rare += 1
        deep[c["max_depth"]] += 1
        idtypes[c["v3"]["id_type"]] += 1
        kinds[c["v3"]["kind"]] += 1
        n_val += "validator" in c
        n_warn += c.get("validationAction") == "warn"
        n_search += bool(c.get("search_indexes"))
        for ix in c["indexes"] + c.get("extra_indexes", []):
            n_idx += 1
            dirs = {d for _, d in ix["keys"]}
            if len(ix["keys"]) > 1:
                idx_kinds["composto"] += 1
            if ix.get("unique"):
                idx_kinds["unico"] += 1
            if "expireAfterSeconds" in ix:
                idx_kinds["TTL"] += 1
            if "partialFilterExpression" in ix:
                idx_kinds["parcial"] += 1
            if ix.get("sparse"):
                idx_kinds["sparse"] += 1
            if ix.get("collation"):
                idx_kinds["collation (case-insensitive)"] += 1
            if "hashed" in dirs:
                idx_kinds["hashed"] += 1
            if "2dsphere" in dirs:
                idx_kinds["2dsphere"] += 1
            if "text" in dirs:
                idx_kinds["text"] += 1
            if any(k.endswith(".$**") for k, _ in ix["keys"]):
                idx_kinds["wildcard"] += 1
    w(f"- campos (todos os niveis): {n_fields:,}; **{n_alt:,} com 2+ tipos no mesmo campo** ({100 * n_alt / n_fields:.1f}%; `alt`); "
      f"{sum(1 for _, c in cols if c['v3']['alt_fields'])}/{len(cols)} colecoes tem ao menos um campo heterogeneo.".replace(",", "."))
    w("- tipos de heterogeneidade: " + ", ".join(f"{k}: {v}" for k, v in alt_kinds.most_common()) + ".")
    w(f"- mapas dinamicos (chaves variaveis): {n_map}; GeoJSON Point: {n_geo}; arrays de arrays: {n_aa}; "
      f"campos raros (presentes em < 10% dos docs): {n_rare}.")
    d5 = sum(v for k, v in deep.items() if k >= 5)
    w(f"- profundidade (`max_depth`, doc=1): " + ", ".join(f"{k}: {deep[k]}" for k in sorted(deep)) +
      f"; **{d5}/{len(cols)} colecoes ({100 * d5 / len(cols):.0f}%) com profundidade >= 5**; "
      f"{sum(1 for s in v3 if max(c['max_depth'] for c in s['collections']) >= 5)}/{len(v3)} schemas tem alguma colecao com >= 5.")
    w("- `_id` por colecao: " + ", ".join(f"{k}: {v}" for k, v in idtypes.most_common()) +
      " (referencias entre colecoes seguem o tipo do `_id` alvo, e ~12% delas aparecem tambem como string: `$lookup` exige conversao).")
    w("- tipos de colecao: " + ", ".join(f"{k}: {v}" for k, v in kinds.most_common()) +
      " (time-series e capped criados no servidor 8.0.32; `options` no JSON).")
    w(f"- validators `$jsonSchema`: {n_val}/{len(cols)} colecoes ({n_warn} com `validationAction: warn`; unioes viram `bsonType: [..]`); "
      f"Atlas Search/Vector (so definicao estatica): {n_search} colecoes.")
    w(f"- indices: {n_idx} no total; " + ", ".join(f"{k}: {v}" for k, v in idx_kinds.most_common()) + ".")
    n_pt = sum(1 for s in v3 if s["language"] == "pt-BR")
    sty = Counter(s["name_style"] for s in v3)
    w(f"- nomes bagunçados: estilo dominante {dict(sty)}; {n_pt} schemas pt-BR sem acento; em cada colecao ~15% dos nomes de campo "
      "fogem do estilo (outro estilo, abreviacoes `cust`/`qty`/`addr`, prefixo de tipo `dt_`/`is_`/`fk_`, sufixos `_v2`/`_old`, `ID`, "
      "`_` inicial, MAIUSCULAS) e ha campos legados duplicados (`campo_v2`, `old_campo`, `legacy_campo`); nomes de colecao com "
      "`tbl_`/`col_`, `_v2`/`_old`/`_bkp`, hifen ou espaco.")
    w("")
    w("Todo schema traz `realism_notes` (lista de strings: o que foi aplicado por colecao) e cada colecao `v3` "
      "(contadores: alt_fields, map_fields, geo_fields, rare_fields, array_of_arrays, id_type, kind).")
    w("")

    # --------------------------------------------------------- aditivo
    w("## 3. O que foi aditivo")
    w("")
    manifest_old = [e for e in man["entries"] if not e["schema_id"].startswith(V3)]
    bad = 0
    for e in manifest_old:
        sp = next(s["_path"] for s in old if s["schema_id"] == e["schema_id"])
        if hashlib.sha256(sp.read_bytes()).hexdigest() != e["schema_sha256"]:
            bad += 1
    w(f"- schemas antigos: {len(old)} arquivos e {len(manifest_old)} entradas de manifest; **{bad} com `schema_sha256` divergente** do arquivo atual.")
    try:
        head = subprocess.run(["git", "show", "HEAD:data/fixtures/manifest.json"], cwd=ROOT, capture_output=True, check=True).stdout
        hm = json.loads(head)["entries"]
        key = lambda e: (e["db"], e["collection"])  # noqa: E731
        cur = {key(e): e for e in manifest_old}
        same = sum(1 for e in hm if cur.get(key(e)) == e)
        w(f"- manifest no `HEAD` x atual para os schemas antigos: {same}/{len(hm)} entradas **identicas** (incl. `docs_sha256`).")
    except Exception as ex:  # noqa: BLE001
        w(f"- comparacao com `HEAD` indisponivel ({ex.__class__.__name__}).")
    w("- novos arquivos: `scripts/data/{build_schemas_v3,schemas_v3_lib,report_schemas_v3}.py`, 104 `data/schemas/*/v3_*.json`, "
      "bancos `kapi_fx_*_v3_*`. Alterados de forma compativel: `seed_fixtures.py` (novos construtos so ativam com chaves ausentes nos "
      "schemas antigos; fluxo de RNG inalterado), `check_fixtures.py` (hash canonico p/ time-series, checagem de `options`), "
      "`split_roles.py` (papel `hom`, `assign_roles_v3`, `is_hom`, `is_trainable`), `configs/dataset.yaml` (`schema_split.v3`), "
      "`env/dump_fixtures.ps1` (`-SchemaPrefix`), testes de split.")
    w("- formato do schema: `format_version: 2` nos v3; construtos novos sempre mantem um `t` primario valido para leitores antigos "
      "(`alt` = variantes extras; `map` em objeto com `f: {}`; `geo` em objeto GeoJSON; `_id` por colecao; `options`).")
    w("")
    w("## 4. Validacao")
    w("")
    if a.validation and Path(a.validation).exists():
        v = json.loads(Path(a.validation).read_text(encoding="utf-8"))
        for k, t in v.items():
            w(f"- {k}: {t}")
    else:
        w("- nao informada.")
    w("")
    w("## 5. Papel `hom` x geradores existentes")
    w("")
    w("Geradores que leem `split_role` e so conhecem `train|eval` (arquivos de outro agente; **nao editados** aqui) e precisam tratar `hom`:")
    w("")
    w("- `scripts/data/gen_snippets.py`: `SPLIT_ROLES = (\"train\",\"eval\")` -> `hom` vira `None` e cai no split v1 por hash (poderia ir para train!). "
      "Precisa filtrar `split_role == \"hom\"` (excluir de train/val/test/bench e emitir so no conjunto de homologacao).")
    w("- `scripts/data/gen_chat.py` (`schema_role`, pool de snippets), `scripts/data/gen_fim.py` / `gen_editor_context.py` (herdam do snippet): "
      "rejeitar registros cujo `split_role` seja `hom` fora do conjunto de homologacao.")
    w("- `scripts/data/validate_pipeline.py` e `scripts/data/decontaminate.py`: isolamento train x eval deve incluir `hom` como terceiro conjunto "
      "(decontaminar hom x train).")
    w("- `scripts/distill/build_prompt_pool.py` (`(rec.get('split_role') or 'train') != 'train'` -> ja exclui `hom`, seguro), "
      "`scripts/teacher/build_teacher_corpus.py` (usa `split_role`/`schema_split_role` p/ isolamento; conferir que `hom` nunca entra no treino), "
      "`scripts/teacher/audit_schemas.py` (so rotulo), `scripts/student/build_student_corpus.py` (propaga o campo).")
    w("- Use `split_roles.role_for(...)`/`is_trainable(...)`/`is_hom(...)`: `is_trainable` so e True para `train` (e `None` legado).")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"escrito {a.out} ({len(L)} linhas)")


if __name__ == "__main__":
    sys.exit(main())
