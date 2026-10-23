"""Testes simples do avaliador de homologacao: determinismo, escrita so em data/homolog/results e reports/, uso do guard."""
import json
import re
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[1]
for p in (str(EVAL), str(EVAL.parent / "data" / "llm")):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_homolog as eh  # noqa: E402
import homolog_guard as hg  # noqa: E402


@pytest.fixture()
def hdir(tmp_path, monkeypatch):
    base = tmp_path / "homolog"
    (base / "v1" / "items").mkdir(parents=True)
    items = [{"id": f"it-{i}", "prefix": "db.c.find({", "suffix": "})", "middle": f"a: {i}", "dialect": "console", "collection": "c"} for i in range(30)]
    (base / "v1" / "items" / "items.jsonl").write_text("\n".join(json.dumps(x) for x in items) + "\n", encoding="utf-8")
    monkeypatch.setenv("KAPI_HOMOLOG_DIR", str(base))
    yield base
    hg.uninstall()


def test_write_paths_restricted(tmp_path):
    with pytest.raises(PermissionError):
        eh.check_write_path(tmp_path / "x.json")
    with pytest.raises(PermissionError):
        eh.check_write_path(eh.ROOT / "data" / "generated" / "x.json")
    with pytest.raises(PermissionError):
        eh.check_write_path(eh.RESULTS / ".." / "v1" / "items" / "items.jsonl")        # escapa de results/ por ..
    assert eh.check_write_path(eh.RESULTS / "m" / "predictions.jsonl")
    assert eh.check_write_path(eh.REPORT_DIR / "homolog-v1-eval.json")
    with pytest.raises(PermissionError):
        eh.write_text(tmp_path / "fora.txt", "x")
    assert not (tmp_path / "fora.txt").exists()


def test_load_items_goes_through_guard_and_is_deterministic(hdir):
    hg.enforce()
    with pytest.raises(hg.HomologAccessError):          # leitura direta e recusada com o guard ativo
        open(hdir / "v1" / "items" / "items.jsonl", encoding="utf-8")
    a = eh.load_items("teste", "pytest")
    b = eh.load_items("teste", "pytest")
    assert [r["id"] for r in a] == [r["id"] for r in b] and len(a) == 30
    assert [r["id"] for r in a] == sorted((r["id"] for r in a), key=eh.order_key)
    assert [r["id"] for r in eh.load_items("teste", "pytest", 5)] == [r["id"] for r in a[:5]]
    with pytest.raises(hg.HomologAccessError):
        eh.load_items("", "pytest")                      # sem motivo, sem leitura
    hg.uninstall()
    log = [json.loads(line) for line in (hdir / "v1" / "access.log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(log) == 3 and all(e["who"] == "pytest" and e["reason"] == "teste" for e in log)      # toda leitura deixa rastro


def test_all_phases_enforce_guard():
    src = Path(eh.__file__).read_text(encoding="utf-8")
    for fn in ("phase_gen", "phase_score", "phase_report"):
        body = src.split(f"def {fn}(", 1)[1].split("\ndef ", 1)[0]
        assert "hg.enforce()" in body, fn
    assert not re.search(r"^\s*(?!#).*open\([^)]*items\.jsonl", src, re.M) or "hg.authorized" in src     # itens so via load_items


def test_exec_text_and_key_deterministic():
    it = {"prefix": '[{"$match":{"a":', "suffix": "}}]\r\n", "dialect": "json", "collection": "c1"}
    doc, text = eh.build_exec_text(it, "1")
    assert doc == '[{"$match":{"a":1}}]\n' and "\r" not in doc
    assert text == 'db.getCollection("c1").aggregate(EJSON.parse(' + json.dumps(doc) + "));"
    it2 = {"prefix": "db.c.find({a:", "suffix": "})", "dialect": "console", "collection": "c"}
    assert eh.build_exec_text(it2, "1")[1] == "db.c.find({a:1})"
    assert eh.exec_key("db1", "x") == eh.exec_key("db1", "x") != eh.exec_key("db2", "x")


def _pred(i, norm, empty=False, syntax=True):
    return {"id": f"p{i}", "cut_kind": "key" if i % 2 else "value", "context_kind": ("full", "none", "irrelevant")[i % 3], "domain": "telecom" if i % 2 else "adtech",
            "author": "llm" if i % 4 < 2 else "grammar",
            "metrics": {"normalized_match": norm, "exact_match": norm, "edit_similarity": 0.5, "accepted_prefix_tokens": 1.0, "syntax_valid": syntax,
                        "mongo_valid": True, "empty": empty}, "reference_checks": {"syntax_valid": True}}


def _exec(i, ok=True, bad_field=False):
    return {"id": f"p{i}", "status": "ok" if ok else "error", "error": "", "n_docs": 1, "unknown_fields": ["zz"] if bad_field else [], "unknown_collections": []}


def test_aggregation_order_independent_and_semantics():
    preds = [_pred(i, i % 3 == 0, empty=(i == 5)) for i in range(60)]
    exe = [_exec(i, ok=(i % 5 != 0), bad_field=(i % 7 == 0)) for i in range(60)]
    ref = [_exec(i, ok=(i % 10 != 1)) for i in range(60)]
    rows = eh.merge_rows(preds, exe, ref)
    s1 = eh.summarize(rows)
    s2 = eh.summarize(list(reversed(rows)))
    assert s1 == s2                                                         # ordem dos itens nao importa; bootstrap com seed fixa
    by = {r["id"]: r for r in rows}
    assert by["p5"]["exec_valid"] is False                                   # vazio nunca conta como exec-valido
    assert by["p7"]["exec_valid"] is False and by["p7"]["field_hallucination"] is True      # campo inexistente reprova
    assert by["p3"]["exec_valid"] is True
    o = s1["overall"]
    assert o["n"] == 60 and o["exec_over_ceiling"] == round(o["exec_valid"] / o["ceiling_exec_valid"], 4)
    assert set(s1["by_context_kind"]) == {"full", "none", "irrelevant"} and sum(v["n"] for v in s1["by_author"].values()) == 60
    assert eh.wilson(0.5, 100)[0] < 0.5 < eh.wilson(0.5, 100)[1]


def test_gpu_counter_verdict_rejects_igpu():
    ok = eh.parse_gpu_counters({"mem": [{"i": "luid_0x0_0x1_phys_0", "v": 2.5 * 2**30}, {"i": "luid_0x0_0x2_phys_0", "v": 0.3 * 2**30}],
                                "eng": [{"i": "pid_9_luid_0x0_0x1_phys_0_eng_0_engtype_3D", "v": 55.0}, {"i": "pid_9_luid_0x0_0x2_phys_0_eng_0_engtype_3D", "v": 0.0}]})
    assert ok["verdict"] == "ok" and ok["largest_adapter"] == "luid_0x0_0x1_phys_0"
    bad = eh.parse_gpu_counters({"mem": [{"i": "luid_0x0_0x1_phys_0", "v": 2.5 * 2**30}, {"i": "luid_0x0_0x2_phys_0", "v": 0.3 * 2**30}],
                                 "eng": [{"i": "pid_9_luid_0x0_0x2_phys_0_eng_0_engtype_3D", "v": 70.0}]})
    assert bad["verdict"] == "REPROVADO"
    assert eh.parse_gpu_counters({"mem": [], "eng": []})["verdict"] == "inconclusivo"
