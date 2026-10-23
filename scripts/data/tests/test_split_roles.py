"""split_role por schema (Fase 03 + lote v3): holdout sempre eval, determinismo, cobertura, papel `hom`, ausencia de `secret`."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import split_roles as sr  # noqa: E402

ALL = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(sr.SCHEMAS_DIR.glob("*/*.json"))]
SCHEMAS = [s for s in ALL if not s["schema_id"].startswith(sr.V3_PREFIX)]   # lote v2 (110 schemas; papeis train|eval)
V3 = [s for s in ALL if s["schema_id"].startswith(sr.V3_PREFIX)]              # lote v3 (train|eval|hom)
HOLDOUT = {s["domain"] for s in SCHEMAS if s["holdout"]}


def _meta(s):
    return {k: s.get(k) for k in ("schema_id", "domain", "holdout", "name_style", "language")}


def test_every_schema_has_role():
    assert SCHEMAS
    assert len(SCHEMAS) == 110
    assert all(s.get("split_role") in ("train", "eval") for s in SCHEMAS)   # o lote v2 nunca recebe `hom`
    assert all(s.get("split_role") in sr.ROLES for s in ALL)


def test_holdout_always_eval():
    assert HOLDOUT == set(sr.HOLDOUT_DOMAINS)
    for s in SCHEMAS:
        if s["holdout"] or s["domain"] in HOLDOUT:
            assert s["split_role"] == "eval", s["schema_id"]
        assert sr.role_for(s["schema_id"]) == s["split_role"]


def test_deterministic_and_matches_files():
    metas = [_meta(s) for s in ALL]
    a = sr.assign_roles(metas)
    b = sr.assign_roles(list(reversed(metas)))
    assert a == b
    assert a == {s["schema_id"]: s["split_role"] for s in ALL}
    # aditivo: os papeis do lote v2 independem do v3
    assert sr.assign_roles([_meta(s) for s in SCHEMAS]) == {s["schema_id"]: s["split_role"] for s in SCHEMAS}


def test_eval_per_eligible_domain_and_both_sides():  # lote v2
    by = defaultdict(list)
    for s in SCHEMAS:
        by[s["domain"]].append(s["split_role"])
    for dom, roles in by.items():
        if dom in HOLDOUT:
            continue
        if len(roles) >= sr.MIN_SCHEMAS_FOR_EVAL:
            assert "eval" in roles, dom
        assert "train" in roles, dom
    elig = [s for s in SCHEMAS if s["domain"] not in HOLDOUT]
    frac = sum(s["split_role"] == "eval" for s in elig) / len(elig)
    assert 0.15 <= frac <= 0.35


def test_name_styles_on_both_sides():
    elig = [s for s in SCHEMAS if s["domain"] not in HOLDOUT]
    sty = defaultdict(set)
    for s in elig:
        sty[(s["name_style"], s["language"])].add(s["split_role"])
    for k, v in sty.items():
        n = sum((s["name_style"], s["language"]) == k for s in elig)
        if n >= 2:
            assert v == {"train", "eval"}, k


def test_role_for_interface():
    s = SCHEMAS[0]
    assert sr.role_for(f"{s['domain']}/{s['schema_id']}") == s["split_role"]
    assert sr.role_for(f"data/schemas/{s['domain']}/{s['schema_id']}.json") == s["split_role"]
    assert sr.role_for(f"{s['domain']}/{s['schema_id']}::tpl3") == s["split_role"]
    assert sr.role_for("nao_existe_xyz") is None


def _names(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("fields", "f"):
                if isinstance(v, dict):
                    out.update(v.keys())
            _names(v, out)
    elif isinstance(node, list):
        for v in node:
            _names(v, out)


def test_no_secret_fields_left():
    pat = re.compile(r"secret|passw|pwd|api[_-]?key|access[_-]?token", re.I)
    for s in SCHEMAS:
        names: set = set()
        _names(s["collections"], names)
        bad = [n for n in names if pat.search(n)]
        assert not bad, (s["schema_id"], bad)
        assert '"secret"' not in json.dumps(s), s["schema_id"]


# ----------------------------------------------------------------------------- lote v3 / papel hom
def test_v3_batch_size_and_roles():
    assert len(V3) >= 80
    cnt = defaultdict(int)
    for s in V3:
        cnt[s["split_role"]] += 1
        assert s["format_version"] == 2 and s["seed"] == sr.V3_SEED
        assert 4 <= len(s["collections"]) <= 14, s["schema_id"]
        assert s.get("realism_notes"), s["schema_id"]
    assert cnt["hom"] >= 0.25 * len(V3)
    assert cnt["train"] >= 0.5 * len(V3)
    assert cnt["eval"] >= 0.10 * len(V3)


def test_hom_domains_entirely_hom():
    doms = defaultdict(set)
    for s in V3:
        doms[s["domain"]].add(s["split_role"])
    hom_doms = {d for d, r in doms.items() if r == {"hom"}}
    assert set(sr.HOM_DOMAINS) <= hom_doms and len(hom_doms) >= 2
    for s in V3:
        assert (s["split_role"] == "hom") == (s["domain"] in sr.HOM_DOMAINS), s["schema_id"]
        if s["domain"] in sr.HOLDOUT_DOMAINS:
            assert s["split_role"] == "eval"
    for d, r in doms.items():
        if d not in sr.HOM_DOMAINS and d not in sr.HOLDOUT_DOMAINS:
            assert "train" in r, d


def test_hom_never_trainable():
    for s in V3:
        r = sr.role_for(s["schema_id"])
        assert r == s["split_role"]
        assert sr.is_hom(s["schema_id"]) == (r == "hom")
        assert sr.is_trainable(s["schema_id"]) == (r == "train")
    assert not sr.is_trainable("v3_ion_tel")
    assert sr.role_for("telecom/v3_ion_tel::tpl1") == "hom"
    assert sr.role_for("nao_existe_xyz") is None and sr.is_trainable("nao_existe_xyz")   # None = comportamento v1 legado


def test_assign_roles_v3_deterministic_and_stable():
    metas = [_meta(s) for s in V3]
    a = sr.assign_roles_v3(metas)
    assert a == sr.assign_roles_v3(list(reversed(metas)))
    assert a == {s["schema_id"]: s["split_role"] for s in V3}
    toy = [dict(schema_id=f"v3_t{i}_x", domain="saas", holdout=False) for i in range(5)] +           [dict(schema_id="v3_h1_y", domain="telecom", holdout=False), dict(schema_id="v3_e1_z", domain="legal", holdout=True)]
    r = sr.assign_roles_v3(toy)
    assert r["v3_h1_y"] == "hom" and r["v3_e1_z"] == "eval"
    assert list(r.values()).count("train") >= 1
