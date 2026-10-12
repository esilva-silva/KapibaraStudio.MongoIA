"""Testes rapidos dos geradores (gen_snippets / gen_fim / validate_snippets)."""
import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import gen_fim  # noqa: E402
import gen_snippets as gs  # noqa: E402
import validate_snippets as vs  # noqa: E402

ROOT = gs.ROOT
N = 250


@pytest.fixture(scope="module")
def cfg():
    return gs.load_config()


@pytest.fixture(scope="module")
def schemas(cfg):
    sc = gs.load_schemas(ROOT / cfg["schemas_dir"])
    assert sc, "sem schemas em data/schemas"
    return sc


@pytest.fixture(scope="module")
def snips(cfg, schemas):
    return gs.generate(cfg, schemas, N)


def test_deterministic_by_seed(cfg, schemas, snips):
    again = gs.generate(cfg, schemas, N)
    assert json.dumps(again, sort_keys=True) == json.dumps(snips, sort_keys=True)
    other = copy.deepcopy(cfg)
    other["seed"] = cfg["seed"] + 1
    diff = gs.generate(other, schemas, 40)
    assert [r["text"] for r in diff] != [r["text"] for r in snips[:40]]


def test_fim_deterministic_and_reconstructs(cfg, snips):
    a = gen_fim.generate_fim(snips, cfg, 300)
    b = gen_fim.generate_fim(snips, cfg, 300)
    assert a == b and len(a) == 300
    by_id = {r["id"]: r for r in snips}
    for r in a:
        assert r["prefix"] + r["middle"] + r["suffix"] == by_id[r["snippet_id"]]["text"]
        assert r["prefix"].strip() and r["suffix"].strip() and r["middle"].strip()
        assert r["approx_middle_tokens"] <= cfg["fim"]["max_tokens"]
        assert "<|" not in r["prefix"] + r["middle"] + r["suffix"]
        for k in ("id", "kind", "prefix", "suffix", "middle", "domain", "schema_id", "collection", "cut_kind",
                  "fields_used", "group", "split_key"):
            assert k in r
    assert {r["cut_kind"] for r in a} >= {"key", "value", "operator", "stage"}


def test_no_nonexistent_fields(snips, schemas):
    by = {s.full_id: s for s in schemas}
    for r in snips:
        sch = by[r["schema_id"]]
        coll = next(c for c in sch.colls if c.name == r["collection"])
        res = gs.analyze_plan(r["plan"], sch, coll)
        assert res["missing"] == [], (r["text"], res["missing"])
        for p in r["fields_used"]:
            if "::" in p:
                ck, path = p.split("::", 1)
                assert path in sch.by_key[ck].fields
            else:
                assert p in coll.fields, (r["id"], p)


def test_checker_detects_bad_field(snips, schemas):
    by = {s.full_id: s for s in schemas}
    r = next(x for x in snips if x["op"] == "find" and x["plan"].get("filter"))
    plan = copy.deepcopy(r["plan"])
    k = next(iter(plan["filter"]))
    plan["filter"]["campoInexistenteXYZ"] = plan["filter"].pop(k)
    sch = by[r["schema_id"]]
    coll = next(c for c in sch.colls if c.name == r["collection"])
    assert "campoInexistenteXYZ" in gs.analyze_plan(plan, sch, coll)["missing"]


def test_holdout_only_test_bench(cfg, schemas):
    hold = set(cfg["holdout_domains"])
    sc = [s for s in schemas if s.domain in hold]
    assert sc, "schemas holdout ausentes"
    recs = gs.generate(cfg, sc, 120)
    assert recs and all(r["split"] in ("test", "bench") and r["holdout"] for r in recs)
    fims = gen_fim.generate_fim(recs, cfg, 150)
    assert fims and all(r["split"] in ("test", "bench") for r in fims)
    allr = gs.generate(cfg, schemas, N)
    assert all(r["split"] in ("test", "bench") for r in allr if r["domain"] in hold)


def test_groups_never_cross_splits(snips):
    g2s = {}
    for r in snips:
        assert g2s.setdefault(r["group_id"], r["split"]) == r["split"]
    keys = {}
    for r in snips:
        keys.setdefault(r["split_key"], set()).add(r["split"])
    assert all(len(v) == 1 for v in keys.values())


def test_style_variation(snips):
    crlf = sum(r["style"]["crlf"] for r in snips) / len(snips)
    assert 0.40 <= crlf <= 0.70
    assert {r["style"]["quote"] for r in snips} == {'"', "'"}
    assert {r["style"]["indent"] for r in snips} == {2, 4}
    assert any("getCollection(" in r["text"] for r in snips)
    assert any('db["' in r["text"] or "db['" in r["text"] for r in snips)


class FakeCursor:
    def sort(self, *a):
        return self

    def skip(self, *a):
        return self

    def limit(self, *a):
        return self

    def __iter__(self):
        return iter([])


class FakeColl:
    """Colecao que registra chamadas; qualquer metodo fora da leitura estoura."""

    def __init__(self, log):
        self.log = log

    def find(self, *a, **k):
        self.log.append("find")
        return FakeCursor()

    def aggregate(self, *a, **k):
        self.log.append("aggregate")
        return []

    def count_documents(self, *a, **k):
        self.log.append("count_documents")
        return 0

    def distinct(self, *a, **k):
        self.log.append("distinct")
        return []

    def __getattr__(self, name):
        raise AssertionError(f"metodo nao-leitura chamado: {name}")


def test_zero_writes_executed(snips):
    log = []

    class DB:
        def __getitem__(self, k):
            return FakeColl(log)

    assert any(r["kind"] == "write" for r in snips), "amostra sem escritas"
    for r in snips:
        v = vs.validate_record(lambda name: DB(), r, 1000, 10)
        if r["kind"] == "write":
            assert v["exec"] == "skipped"
        else:
            assert v["exec"] in ("ok", "blocked")
    assert log and set(log) <= {"find", "aggregate", "count_documents", "distinct"}
    with pytest.raises(vs.Blocked):
        vs.execute_plan(FakeColl(log), {"op": "updateMany", "filter": {}, "update": {"$set": {"a": 1}}}, 100, 10)
    for stage in ("$out", "$merge", "$function", "$where"):
        with pytest.raises(vs.Blocked):
            vs.execute_plan(FakeColl(log), {"op": "aggregate", "pipeline": [{"$match": {}}, {stage: "x"}]}, 100, 10)
    with pytest.raises(vs.Blocked):
        vs.execute_plan(FakeColl(log), {"op": "aggregate",
                                        "pipeline": [{"$lookup": {"from": "a", "pipeline": [{"$out": "z"}], "as": "q"}}]}, 100, 10)


def test_db_prefix_enforced():
    r = {"id": "x", "op": "find", "kind": "read", "db": "admin", "collection": "system.users",
         "plan": {"op": "find", "filter": {}}}

    def boom(name):
        raise AssertionError("tocou no driver")
    assert vs.validate_record(boom, r, 100, 10)["exec"] == "blocked"


def _mongo():
    try:
        c, g = vs.make_client("mongodb://127.0.0.1:27017")
        c.admin.command("ping")
        return c, g
    except Exception:
        return None, None


def test_live_exec_reads_no_writes(snips):
    c, g = _mongo()
    if c is None:
        pytest.skip("MongoDB local indisponivel")
    seeded = {n for n in c.list_database_names() if n.startswith("kapi_fx_")}
    reads = [r for r in snips if r["kind"] == "read" and r["db"] in seeded][:80]
    if not reads:
        pytest.skip("sem bancos semeados")
    res = [vs.validate_record(lambda n: c[n], r, 2000, 101) for r in reads]
    bad = [x for x in res if x["exec"] == "error"]
    assert len(bad) <= max(1, len(res) // 50), bad[:3]
    assert g.violations == []
    r = dict(reads[0])
    r["plan"] = {"op": "find", "filter": {"a": {"$naoExisteOp": 1}}}
    assert vs.validate_record(lambda n: c[n], r, 2000, 10)["exec"] == "error"


# ---- split_key sem vazamento (snippets e FIM) + guarda do conjunto cego ----------------------

def test_fim_split_key_no_leak_and_holdout(cfg, snips):
    fims = gen_fim.generate_fim(snips, cfg, 600)
    keys = {}
    for r in fims:
        keys.setdefault(r["split_key"], set()).add(r["split"])
        if r["domain"] in cfg["holdout_domains"]:
            assert r["split"] in ("test", "bench")
    assert all(len(v) == 1 for v in keys.values())
    by_id = {r["id"]: r for r in snips}
    assert all(by_id[r["snippet_id"]]["split"] == r["split"] for r in fims)


def test_generated_v1_split_integrity():
    d = ROOT / "data" / "generated" / "v1"
    if not (d / "snippets.jsonl").exists() or not (d / "fim.jsonl").exists():
        pytest.skip("v1 nao gerada")
    hold = set(gs.load_config()["holdout_domains"])
    seen = {}
    for name in ("snippets.jsonl", "fim.jsonl"):
        for line in open(d / name, encoding="utf-8"):
            r = json.loads(line)
            seen.setdefault(r["split_key"], set()).add(r["split"])
            if r["domain"] in hold:
                assert r["split"] in ("test", "bench")
                assert r["split"] not in ("train", "val")
    assert all(len(v) == 1 for v in seen.values())


def test_blind_guard_applied_in_pipeline(tmp_path, monkeypatch, cfg, snips):
    import blind_guard as bg
    victim = gen_fim.generate_fim(snips, cfg, 50)[0]
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    ph = bg.prompt_sha256(bg.prompt_text({"kind": "autocomplete", "prefix": victim["prefix"], "suffix": victim["suffix"]}))
    sh = bg.prompt_sha256(snips[0]["text"])
    (sealed / "manifest.json").write_text(json.dumps({"item_sha256": {}, "prompt_sha256": [ph, sh]}), encoding="utf-8")
    monkeypatch.setenv("KAPI_BLIND_DIR", str(tmp_path))
    out = gen_fim.generate_fim(snips, cfg, 300)
    assert out and all((r["prefix"], r["suffix"]) != (victim["prefix"], victim["suffix"]) for r in out)
    assert not gs.blind_clean(snips[0], snips[0]["text"])
    assert gs.blind_clean(snips[1], snips[1]["text"])
