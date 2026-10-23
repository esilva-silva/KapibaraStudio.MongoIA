"""Determinismo com seed e isolamento hom x train."""
import json
import os
import stat

import pytest

import gen_llm_queries as G
import homolog_guard as hg
import llm_to_fim as L
import prompts as P
from common import assert_roles_disjoint, load_cfg, load_schema_infos

SCHEMA = {"schema_id": "SID", "domain": "saas", "holdout": False, "split_role": "ROLE", "name_style": "snake", "seed": 1, "format_version": 1,
          "language": "en", "description": "t",
          "collections": [{"key": "tenants", "name": "tenants", "doc_count": 10,
                           "fields": {"name": {"t": "string", "p": 1.0, "n": 0.0}, "plan": {"t": "string", "p": 1.0, "n": 0.0, "values": ["a", "b"]},
                                      "seats": {"t": "int32", "p": 1.0, "n": 0.0, "lo": 1, "hi": 5}}, "indexes": []}]}


def _write_schemas(root):
    for sid, role in (("s_train", "train"), ("s_hom", "hom"), ("s_eval", "eval"), ("s_none", None)):
        d = root / "saas"
        d.mkdir(parents=True, exist_ok=True)
        raw = dict(SCHEMA, schema_id=sid, split_role=role)
        (d / f"{sid}.json").write_text(json.dumps(raw), encoding="utf-8")
    return root


def test_roles_filter_and_disjoint(tmp_path):
    root = _write_schemas(tmp_path / "schemas")
    train = load_schema_infos(("train",), root, require_fixture=False)
    hom = load_schema_infos(("hom",), root, require_fixture=False)
    both = load_schema_infos(("train", "hom"), root, require_fixture=False)
    assert [i.schema_id for i in train] == ["s_train"] and [i.schema_id for i in hom] == ["s_hom"]
    assert {i.schema_id for i in both} == {"s_train", "s_hom"}          # eval e sem papel nunca entram
    assert_roles_disjoint(train, hom)
    with pytest.raises(SystemExit):
        assert_roles_disjoint(train, both)                                 # train e hom misturados


def test_plan_deterministic_and_seed_sensitive(tmp_path):
    infos = load_schema_infos(("train",), _write_schemas(tmp_path / "s"), require_fixture=False)
    cfg = load_cfg()
    p1 = [c["call_id"] for c in G.build_plan(infos, 40, cfg, 7)]
    p2 = [c["call_id"] for c in G.build_plan(infos, 40, cfg, 7)]
    assert p1 == p2 and len(p1) == -(-40 // int(cfg["queries_per_call"]))
    j1 = G.make_job(G.build_plan(infos, 8, cfg, 7)[0], cfg, 7, None)
    j2 = G.make_job(G.build_plan(infos, 8, cfg, 7)[0], cfg, 7, None)
    j3 = G.make_job(G.build_plan(infos, 8, cfg, 7)[0], cfg, 8, None)
    assert j1["messages"] == j2["messages"] and j1["seed"] == j2["seed"] and j1["specs"] == j2["specs"]
    assert j1["messages"] != j3["messages"]
    assert len(P.TASKS) >= 25


def test_cuts_deterministic():
    import gen_fim as gf
    from gen_snippets import load_config
    dscfg = load_config()
    info = load_schema_infos(("train",), require_fixture=False)[0]
    coll = info.schema.colls[0]
    q = {"id": "llmq-x", "text": f"db.{coll.name}.find({{ _id: {{ $exists: true }} }}).sort({{ _id: 1 }}).limit(5)", "qkey": "k",
         "collection": coll.name, "task": "t", "validation": {"commands": ["find"]}}
    tc = gf.middle_tokens_for(dscfg)
    a = L.cuts_for_snippet(L.snippet_from_query(q, info, 3), dscfg, tc, 3, 6)
    b = L.cuts_for_snippet(L.snippet_from_query(q, info, 3), dscfg, tc, 3, 6)
    assert a and a == b


def test_train_build_skips_hom_schema(tmp_path):
    """Consulta de schema hom entregue ao build de papel train e descartada (nunca produz exemplo de treino)."""
    hom = [i for i in load_schema_infos(("hom",))]
    if not hom:
        pytest.skip("ainda nao ha schemas hom semeados")
    cfg = load_cfg()
    q = {"id": "llmq-h", "text": "db.x.find()", "qkey": "k", "collection": hom[0].schema.colls[0].name, "schema_id": hom[0].full_id, "task": "t",
         "validation": {"commands": ["find"]}}
    st = L.build([q], "train", tmp_path, cfg, do_decontam=False)
    assert st["snippets"] == 0 and st["skipped"].get("schema_fora_do_papel") == 1


def test_homolog_guard_blocks_reads_and_hash_reuse(tmp_path, monkeypatch):
    hdir = tmp_path / "homolog"
    (hdir / "v1" / "items").mkdir(parents=True)
    item = {"id": "i1", "prefix": "db.a.find({ ", "suffix": " })", "middle": "x: 1", "author": "llm", "split": "hom", "split_role": "hom",
            "cut_kind": "key", "context_kind": "full", "domain": "saas", "schema_id": "saas/s_hom"}
    (hdir / "v1" / "items" / "items.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
    monkeypatch.setenv("KAPI_HOMOLOG_DIR", str(hdir))
    import homolog_tool as ht
    monkeypatch.setattr(ht, "VERSIONED_MANIFEST", tmp_path / "pub.json")
    man = ht.seal(hdir / "v1", quiet=True)
    assert man["n_items"] == 1 and (tmp_path / "pub.json").exists()
    assert ht.verify(hdir / "v1")
    hg.enforce()                                       # gerador de treino (guard ativo) nao le o diretorio
    try:
        with pytest.raises(hg.HomologAccessError):
            open(hdir / "v1" / "items" / "items.jsonl", encoding="utf-8")
        with pytest.raises(hg.HomologAccessError):
            os.listdir(hdir)
        with pytest.raises(hg.HomologAccessError):
            (hdir / "v1" / "items" / "items.jsonl").read_text()
        with hg.authorized("teste", "pytest"):         # leitura autorizada e registrada
            assert open(hdir / "v1" / "items" / "items.jsonl", encoding="utf-8").read()
        assert (hdir / "v1" / "access.log.jsonl").exists()
        with pytest.raises(hg.HomologAccessError):     # reaproveitar item/prompt selado em dados de treino e recusado
            hg.check_record(item)
        with pytest.raises(hg.HomologAccessError):
            hg.check_text(item["prefix"] + "\x00" + item["suffix"])
        hg.check_text("outro prefixo\x00outro sufixo")
    finally:
        hg.uninstall()
    p = hdir / "v1" / "items" / "items.jsonl"          # selo detecta adulteracao
    os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
    p.write_text(json.dumps({**item, "middle": "y: 2"}) + "\n", encoding="utf-8")
    assert not ht.verify(hdir / "v1")
