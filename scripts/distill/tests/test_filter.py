"""Filtro: rejeita campo inexistente, nunca executa escrita, rejeita truncadas/vazias, aplica a decontaminacao oficial."""
import json
import re

import pytest

import build_prompt_pool as bpp
import filter_teacher as ft
from _distill_common import ROOT, load_config

VALIDATED = ROOT / "data" / "validated" / "v2c" / "fim-00000.jsonl"
pytestmark = pytest.mark.skipif(not VALIDATED.exists(), reason="data/validated/v2c ausente")

READ_OPS = {"find", "findOne", "aggregate", "countDocuments", "distinct"}


class FakeExecutor:
    """Registra o que o filtro pediria ao driver; falha se alguma escrita/DDL chegar aqui."""
    violations: list = []

    def __init__(self):
        self.calls = []

    def run(self, db, collection, plan):
        assert plan["op"] in READ_OPS, f"ESCRITA chegou ao executor: {plan['op']}"
        self.calls.append((db, collection, plan["op"]))
        return {"exec": "ok", "n_docs": 1}


@pytest.fixture(scope="module")
def rows():
    out = []
    with open(VALIDATED, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 7:
                continue
            r = json.loads(line)
            if r["split"] == "train":
                out.append(bpp.compact(r))
    return out


def cand(pred, stop="eos", k=0):
    return {"idx": 0, "id": "x", "k": k, "mode": "greedy", "pred": pred, "raw": pred, "stop": stop, "gen_tokens": 3, "kept_tokens": 3,
            "logprob_mean": -0.1, "logprob_min": -0.5}


@pytest.fixture(scope="module")
def validator():
    import yaml
    cfg = load_config()
    dcfg = yaml.safe_load((ROOT / cfg["dataset_config"]).read_text(encoding="utf-8"))
    ex = FakeExecutor()
    v = ft.Validator(dcfg, cfg["filter"], executor=ex)
    v.fake = ex
    yield v
    v.close()


def key_row(rows):
    for r in rows:
        if r["cut_kind"] == "key" and r["gold_exec"] == "ok" and r["dialect"] == "console" and re.fullmatch(r"[A-Za-z_][\w.]*", r["gold"] or ""):
            return r
    pytest.skip("sem corte key simples")


def test_gold_as_prediction_is_valid_and_agrees(validator, rows):
    r = key_row(rows)
    v = validator.validate(r, cand(r["gold"]))
    assert v["ok"], v
    assert v["agrees_with_gold"] and v["gold_exact"] and v["exec_kind"] == "live"
    assert validator.fake.calls and all(op in READ_OPS for _, _, op in validator.fake.calls)


def test_nonexistent_field_is_rejected(validator, rows):
    r = key_row(rows)
    v = validator.validate(r, cand("ZzCampoInexistente"))
    assert not v["ok"] and v["stage"] == "fields" and v["reason"].startswith("campo/estrutura invalido"), v


def test_valid_but_different_field_is_kept_and_flagged(validator, rows):
    r = key_row(rows)
    sch, coll = validator.coll(r["schema_id"], r["collection"])
    others = [p for p, f in coll.fields.items() if p != r["gold"] and "." not in p and f.t == "string" and not f.via_array]
    if not others:
        pytest.skip("sem outro campo string")
    v = validator.validate(r, cand(others[0]))
    if not v["ok"]:
        pytest.skip(f"campo alternativo invalido para este corte ({v['reason']})")
    assert v["agrees_with_gold"] is False and v["gold_exact"] is False


def test_incomplete_and_empty_are_rejected(validator, rows):
    r = key_row(rows)
    assert validator.validate(r, cand(r["gold"], stop="max"))["stage"] == "semantic"
    assert validator.validate(r, cand("   "))["stage"] == "semantic"
    assert validator.validate(r, cand("x```y"))["stage"] == "semantic"
    assert validator.validate(r, cand(r["gold"] + "<|fim_pad|>"))["stage"] == "semantic"


def test_budget_over_96_tokens_is_rejected(validator, rows):
    r = key_row(rows)
    v = validator.validate(r, cand("campo " * 200))
    assert not v["ok"] and v["stage"] == "budget"


def test_writes_are_never_sent_to_the_driver(validator, rows):
    before = len(validator.fake.calls)
    w = [r for r in rows if r["gold_exec"] == "skipped_write" and r["cut_kind"] in ("key", "value", "operator")][:15]
    if not w:
        pytest.skip("sem escritas na amostra")
    kinds = set()
    for r in w:
        v = validator.validate(r, cand(r["gold"]))      # FakeExecutor.run falharia o teste se uma escrita chegasse
        if v["ok"]:
            kinds.add(v["exec_kind"])
    assert len(validator.fake.calls) == before
    assert kinds <= {"skipped_write"}


def test_atlas_search_is_static_only(validator, rows):
    a = [r for r in rows if r["category"] == "atlas_search" and r["gold_exec"] == "skipped_static"][:10]
    if not a:
        pytest.skip("sem atlas na amostra")
    before = len(validator.fake.calls)
    for r in a:
        v = validator.validate(r, cand(r["gold"]))
        if v["ok"]:
            assert v["exec_kind"] == "skipped_static"
    assert len(validator.fake.calls) == before


def test_rank_is_independent_of_gold():
    rank = load_config()["filter"]["rank"]
    assert rank == ["stages_passed"]

    def mk(stages, k, exact=False):
        return ({"logprob_mean": -1.0, "k": k}, {"stages_passed": stages, "gold_exact": exact, "plan_equal_gold": exact})

    # mais etapas vencem, mesmo contra um candidato que bate o gold
    cs = [mk(7, 0), mk(5, 1, exact=True)]
    assert max(cs, key=lambda cv: ft.rank_key(rank, cv[1], cv[0]))[0]["k"] == 0
    # empate: a greedy vence, o gold nao influencia
    cs = [mk(7, 1, exact=True), mk(7, 0)]
    assert max(cs, key=lambda cv: ft.rank_key(rank, cv[1], cv[0]))[0]["k"] == 0
    w = load_config()["filter"]["weights"]
    assert w["agrees_with_gold"] == w["disagrees_with_gold"]


def test_stages_passed_counts_validator_stages(validator, rows):
    r = key_row(rows)
    assert validator.validate(r, cand(r["gold"]))["stages_passed"] == len(ft.VALIDATION_STAGES)
    assert validator.validate(r, cand("ZzCampoInexistente"))["stages_passed"] == ft.VALIDATION_STAGES.index("fields")
    assert validator.validate(r, cand("   "))["stages_passed"] == 0


def test_decontam_rule_is_the_official_one():
    import yaml
    cfg = load_config()
    dcfg = yaml.safe_load((ROOT / cfg["dataset_config"]).read_text(encoding="utf-8"))
    assert cfg["filter"]["decontam"]["ngram_min_content"] == dcfg["decontam"]["ngram_min_content"] == 2


def mongo_up():
    try:
        import pymongo
        c = pymongo.MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=1500)
        c.admin.command("ping")
        return any(n.startswith("kapi_fx_") for n in c.list_database_names())
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not mongo_up(), reason="MongoDB local indisponivel")
def test_live_guard_sees_zero_writes_and_decontam_removes_eval_duplicate(tmp_path, rows):
    """Ponta a ponta com o MongoDB real: prompts de treino com pred=gold (aceitos) + 1 prompt cujo documento e identico a um de val
    (removido pela decontaminacao) + 1 candidata com campo inexistente; escritas ficam estaticas; guard do driver sem violacoes."""
    out = tmp_path / "run"
    pool = out / "pool"
    gen = out / "gen"
    pool.mkdir(parents=True)
    gen.mkdir()
    cache = ROOT / "data" / "distilled" / "v2c"
    docs = cache / "eval_docs.jsonl"
    if not docs.exists():
        pytest.skip("eval_docs.jsonl ausente (rode build_prompt_pool.py)")
    ev = None
    with open(docs, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["split"] == "val" and d["type"] == "fim" and d["cut_kind"] == "key" and d["dialect"] == "console":
                ev = d
                break
    assert ev
    sel = [r for r in rows if r["gold_exec"] == "ok" and r["cut_kind"] == "key" and r["dialect"] == "console"
           and re.fullmatch(r"[A-Za-z_][\w.]*", r["gold"] or "")][:2]
    sel += [r for r in rows if r["gold_exec"] == "skipped_write" and r["cut_kind"] == "key"][:1]
    leak = dict(sel[0], id="leak-1", prefix=ev["prefix"], suffix=ev["suffix"], gold=ev["middle"], schema_id=ev["schema_id"],
                collection=ev["collection"], snippet_id="snip-leak", dialect=ev["dialect"], editor=None, gold_exec="ok")
    bad = dict(sel[0], id="bad-1", snippet_id="snip-bad")
    prompts = sel + [leak, bad]
    with open(pool / "pool-00000.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in prompts:
            f.write(json.dumps(r) + "\n")
    preds = [r["gold"] for r in prompts[:-1]] + ["ZzCampoInexistente"]
    with open(gen / "gen-00000.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i, (r, p) in enumerate(zip(prompts, preds)):
            f.write(json.dumps({**cand(p), "idx": i, "id": r["id"], "origin": "base"}) + "\n")
    rc = ft.main(["--pool", str(pool), "--gen", str(gen), "--out", str(out), "--eval-cache", str(cache), "--workers", "1", "--name", "pytest"])
    assert rc == 0
    rep = json.loads((out / "filter_report.json").read_text(encoding="utf-8"))
    assert rep["write_commands_seen_by_guard"] == []
    assert set(rep["exec_ops_requested"]) <= READ_OPS
    acc = [json.loads(line) for line in (out / "accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    rej = [json.loads(line) for line in (out / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "leak-1" not in {a["id"] for a in acc} and "bad-1" not in {a["id"] for a in acc}
    assert any(r["id"] == "leak-1" and r["stage"] == "decontam" for r in rej)
    assert any(r["id"] == "bad-1" and r["stage"] == "fields" for r in rej)
    assert all(a["label_source"] == "teacher_seq" and a["middle"] == a["target"] and "weight" in a for a in acc)
    (ROOT / "reports" / "distill" / "filter-report-pytest.md").unlink(missing_ok=True)
