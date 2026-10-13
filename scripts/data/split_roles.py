"""split_role por schema (train|eval) - Fase 03.

Regra (documentada em configs/dataset.yaml, bloco `schema_split`):
  * dominios holdout (education, legal) => "eval" sempre;
  * demais dominios: por dominio, schemas ordenados por sha256(f"{seed}:{schema_id}"); os primeiros
    round(eval_fraction * n) (minimo 1 se n >= min_schemas_for_eval) sao "eval", o resto "train";
  * cobertura de estilo: cada (name_style, language) presente em dominios elegiveis com >= 2 schemas deve
    aparecer nos dois lados; se faltar, troca-se o schema de menor hash com outro do mesmo dominio.

Interface para os geradores:
    from split_roles import role_for
    role_for("atlas_saas" | "saas/atlas_saas" | "data/schemas/saas/atlas_saas.json" | "saas/atlas_saas::tpl")
    -> "train" | "eval" | None (desconhecido)
O papel e lido do JSON do schema (campo `split_role`), com fallback em data/fixtures/manifest.json.
CLI: python scripts/data/split_roles.py [--json] [--check]
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = ROOT / "data" / "schemas"
MANIFEST = ROOT / "data" / "fixtures" / "manifest.json"
SEED = 20261008
EVAL_FRACTION = 0.25
MIN_SCHEMAS_FOR_EVAL = 6
HOLDOUT_DOMAINS = ("education", "legal")


def _h(seed: int, schema_id: str) -> str:
    return hashlib.sha256(f"{seed}:{schema_id}".encode()).hexdigest()


def assign_roles(schemas: list[dict], seed: int = SEED, eval_fraction: float = EVAL_FRACTION) -> dict[str, str]:
    """schemas: dicts com schema_id, domain, holdout, name_style, language. Retorna {schema_id: role}."""
    roles: dict[str, str] = {}
    by_dom: dict[str, list[dict]] = defaultdict(list)
    for s in schemas:
        by_dom[s["domain"]].append(s)
    for dom, items in by_dom.items():
        items.sort(key=lambda s: _h(seed, s["schema_id"]))
        if any(s.get("holdout") for s in items) or dom in HOLDOUT_DOMAINS:
            for s in items:
                roles[s["schema_id"]] = "eval"
            continue
        n = len(items)
        k = int(round(eval_fraction * n))
        if n >= MIN_SCHEMAS_FOR_EVAL:
            k = max(1, k)
        k = min(k, n - 1) if n > 1 else 0
        for i, s in enumerate(items):
            roles[s["schema_id"]] = "eval" if i < k else "train"
    # cobertura de estilo nos dois lados
    elig = [s for s in schemas if not s.get("holdout") and s["domain"] not in HOLDOUT_DOMAINS]
    style = lambda s: (s.get("name_style"), s.get("language"))  # noqa: E731
    for _ in range(10):
        changed = False
        for st in sorted({style(s) for s in elig}, key=str):
            grp = [s for s in elig if style(s) == st]
            if len(grp) < 2:
                continue
            have = {roles[s["schema_id"]] for s in grp}
            if len(have) == 2:
                continue
            miss = "eval" if have == {"train"} else "train"
            # candidato do estilo (menor hash) cujo dominio permaneca valido apos a troca
            for c in sorted(grp, key=lambda s: _h(seed, s["schema_id"])):
                dom_items = [x for x in elig if x["domain"] == c["domain"]]
                ev = sum(roles[x["schema_id"]] == "eval" for x in dom_items)
                if miss == "eval" and ev >= max(1, round(eval_fraction * len(dom_items)) + 1):
                    continue
                if miss == "train" and (len(dom_items) - ev) <= 1:
                    continue
                roles[c["schema_id"]] = miss
                changed = True
                break
        if not changed:
            break
    return roles


# ---------------------------------------------------------------------------- consulta
_CACHE: dict[str, str] | None = None


def _load() -> dict[str, str]:
    global _CACHE
    if _CACHE is None:
        m: dict[str, str] = {}
        if MANIFEST.exists():
            try:
                for e in json.loads(MANIFEST.read_text(encoding="utf-8")).get("entries", []):
                    if e.get("split_role"):
                        m[e["schema_id"]] = e["split_role"]
            except (OSError, ValueError):
                pass
        for p in sorted(SCHEMAS_DIR.glob("*/*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if d.get("split_role"):
                m[d["schema_id"]] = d["split_role"]
        _CACHE = m
    return _CACHE


def _sid(x) -> str:
    s = str(x).split("::", 1)[0].replace("\\", "/")
    if s.endswith(".json"):
        return s.rsplit("/", 1)[-1][:-5]
    return s.rsplit("/", 1)[-1]


def role_for(schema_id_or_path) -> str | None:
    """"train" | "eval" | None (schema desconhecido / sem papel)."""
    return _load().get(_sid(schema_id_or_path))


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Resumo de split_role por dominio")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true", help="falha se algum schema nao tiver papel")
    a = ap.parse_args(argv)
    rows = []
    for p in sorted(SCHEMAS_DIR.glob("*/*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append(d)
    cnt: dict[str, Counter] = defaultdict(Counter)
    for d in rows:
        cnt[d["domain"]][d.get("split_role") or "None"] += 1
    if a.json:
        print(json.dumps({k: dict(v) for k, v in cnt.items()}, indent=1, sort_keys=True))
    else:
        print(f"{'dominio':22s} {'train':>5s} {'eval':>5s}  holdout")
        for dom in sorted(cnt):
            c = cnt[dom]
            ho = "sim" if any(d["holdout"] for d in rows if d["domain"] == dom) else ""
            print(f"{dom:22s} {c['train']:5d} {c['eval']:5d}  {ho}" + (f"  SEM PAPEL={c['None']}" if c["None"] else ""))
        t = sum(c["train"] for c in cnt.values()); e = sum(c["eval"] for c in cnt.values())
        print(f"{'TOTAL':22s} {t:5d} {e:5d}")
        sty: dict[tuple, Counter] = defaultdict(Counter)
        for d in rows:
            sty[(d.get("name_style"), d.get("language"))][d.get("split_role")] += 1
        for k in sorted(sty, key=str):
            print(f"  estilo {k[0]}/{k[1]}: train={sty[k]['train']} eval={sty[k]['eval']}")
    if a.check and any(c["None"] for c in cnt.values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
