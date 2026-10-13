"""split_role por schema (Fase 03): holdout sempre eval, determinismo, cobertura, ausencia de `secret`."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import split_roles as sr  # noqa: E402

SCHEMAS = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(sr.SCHEMAS_DIR.glob("*/*.json"))]
HOLDOUT = {s["domain"] for s in SCHEMAS if s["holdout"]}


def _meta(s):
    return {k: s.get(k) for k in ("schema_id", "domain", "holdout", "name_style", "language")}


def test_every_schema_has_role():
    assert SCHEMAS
    assert all(s.get("split_role") in ("train", "eval") for s in SCHEMAS)


def test_holdout_always_eval():
    assert HOLDOUT == set(sr.HOLDOUT_DOMAINS)
    for s in SCHEMAS:
        if s["holdout"] or s["domain"] in HOLDOUT:
            assert s["split_role"] == "eval", s["schema_id"]
        assert sr.role_for(s["schema_id"]) == s["split_role"]


def test_deterministic_and_matches_files():
    metas = [_meta(s) for s in SCHEMAS]
    a = sr.assign_roles(metas)
    b = sr.assign_roles(list(reversed(metas)))
    assert a == b
    assert a == {s["schema_id"]: s["split_role"] for s in SCHEMAS}


def test_eval_per_eligible_domain_and_both_sides():
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
