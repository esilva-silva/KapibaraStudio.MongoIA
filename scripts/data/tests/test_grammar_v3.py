"""Gramatica dos schemas v3: papel train/eval separados, hom nunca, seeds fixas (determinismo), contexto conservador, referencia da homologacao sem lista vazia."""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
ROOT = DATA.parents[1]
for p in (DATA, DATA / "llm", ROOT / "scripts" / "distill", ROOT / "scripts" / "teacher"):
    sys.path.insert(0, str(p))
site = os.environ.get("KAPI_TOKENIZERS_SITE") or r"C:\SlopStudio.MongoAI-artifacts\.venv\Lib\site-packages"
if Path(site).exists() and site not in sys.path:
    sys.path.append(site)

import build_grammar_v3 as bg  # noqa: E402
import gen_fim as gf  # noqa: E402
import gen_snippets as gs  # noqa: E402
import homolog_guard as hg  # noqa: E402
import homolog_refs  # noqa: E402
import llm_to_fim as l2f  # noqa: E402
from jsonl_shards import iter_jsonl  # noqa: E402

pytestmark = pytest.mark.skipif(not any((ROOT / "data" / "schemas").glob("*/v3_*.json")), reason="sem schemas v3")


def test_stage_has_only_requested_role_and_never_hom(tmp_path):
    tr = bg.stage_schemas("train", tmp_path)
    ev = bg.stage_schemas("eval", tmp_path)
    for d, role in ((tr, "train"), (ev, "eval")):
        files = sorted(d.glob("*/*.json"))
        assert files and all(p.name.startswith("v3_") for p in files)
        for p in files:
            raw = json.loads(p.read_text(encoding="utf-8"))
            assert raw["split_role"] == role and raw["domain"] not in ("telecom", "adtech")
    assert not ({p.name for p in tr.glob("*/*.json")} & {p.name for p in ev.glob("*/*.json")})      # nenhum schema nos dois lados


def test_derived_config_is_train_conservative(tmp_path):
    import yaml
    cfg = yaml.safe_load(bg.derived_config("train", tmp_path).read_text(encoding="utf-8"))
    assert "eval_counts" not in cfg["fim"] and cfg["fim"]["decontam_prefilter"] is False
    assert abs(sum(cfg["fim"]["category_targets"].values()) - 1.0) < 1e-9
    assert cfg["seed"] == bg.SEEDS["train"] != bg.SEEDS["eval"]
    assert cfg["decontam"]["ngram_min_content"] == 2
    assert cfg["snippets"]["extra_categories"]["atlas_search"] >= 0.1                   # mais cobertura estrutural


def _gen_small(tmp_path, tag, role="train", n_snip=700, n_cut=500):
    out = tmp_path / tag
    out.mkdir()
    stage = bg.stage_schemas(role, out)
    cfgp = bg.derived_config(role, out)
    snip = bg.run_step_snippets(role, out, stage, cfgp, n_snip, False)
    val = out / f"validation_{role}.jsonl"
    with open(val, "w", encoding="utf-8") as f:                                           # sem Mongo no teste: tudo `skipped_static`
        for r in iter_jsonl(snip):
            f.write(json.dumps({"id": r["id"], "exec": "ok"}) + "\n")
    fim = bg.run_step_fim(role, out, snip, val, cfgp, 1, n_cut, False)
    ctx, st = bg.run_step_context(role, out, stage, snip, fim, cfgp, False)
    return out, snip, fim, ctx, st


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    root = tmp_path_factory.mktemp("g3")
    return _gen_small(root, "a"), _gen_small(root, "b")


def test_fixed_seeds_make_snippets_fim_and_context_deterministic(small):
    (_, s1, f1, c1, _), (_, s2, f2, c2, _) = small
    assert sha(s1) == sha(s2) and sha(f1) == sha(f2) and sha(c1) == sha(c2)


def test_records_are_train_grammar_v3_gold_non_hom(small):
    (_, _, _, ctx, st), _ = small
    recs = list(iter_jsonl(ctx))
    assert recs and len({r["id"] for r in recs}) == len(recs)
    for r in recs:
        assert r["split"] == "train" and r["split_role"] == "train" and r["source"] == "grammar-v3" and r["label_source"] == "gold"
        assert r["author"] == "grammar" and r["domain"] not in ("telecom", "adtech") and r["schema_id"].split("/", 1)[1].startswith("v3_")
        assert r["target"] == r["middle"]


def test_blind_contexts_are_at_least_25pct_and_conservative(small):
    (out, _, _, ctx, st), _ = small
    recs = list(iter_jsonl(ctx))
    share = sum(1 for r in recs if r["context_kind"] != "full") / len(recs)
    assert share >= 0.25 and st["none_or_irrelevant_share"] == pytest.approx(share, abs=1e-3)
    stage = out / "_schemas_train"
    names = {}
    for s in gs.load_schemas(stage):
        shim = type("S", (), {"schema_id": s.schema_id})
        for c in s.colls:
            names[(s.full_id, c.name)] = l2f.schema_names(shim, c)
    for r in recs:
        if r["context_kind"] != "full":                      # alvo conservador: nenhum nome do schema ausente do texto visivel
            assert l2f.field_need(r["middle"], r["prefix"] + r["suffix"], names[(r["schema_id"], r["collection"])]) <= 1
    assert {r["context_kind"] for r in recs} == {"full", "none", "irrelevant"}


def test_cut_kinds_and_categories_cover_the_structural_targets(small):
    (_, _, fim, _, _), _ = small
    recs = list(iter_jsonl(fim))
    cats = {r["category"] for r in recs}
    assert {"atlas_search", "json_ejson", "dsl", "indexes"} <= cats
    assert len({r["cut_kind"] for r in recs}) == 9


# -- referencia da homologacao: nunca lista vazia silenciosa --------------------------------------------------------------------------
def _fake_homolog(tmp_path, with_items=True):
    base = tmp_path / "homolog" / "v1" / "v1"
    (base / "sealed").mkdir(parents=True)
    (base / "sealed" / "manifest.json").write_text(json.dumps({"prompt_sha256": ["x"], "item_sha256": {"a": "b"}}), encoding="utf-8")
    if with_items:
        (base / "items").mkdir()
        rows = [{"id": "h1", "type": "fim", "kind": "find", "prefix": "db.a.find({ ", "middle": "k", "suffix": " })", "snippet_id": "s1", "collection": "a"},
                {"id": "h2", "type": "fim", "kind": "find", "prefix": "db.a.find({ ", "middle": "q", "suffix": " })", "snippet_id": "s1", "collection": "a"}]
        (base / "items" / "items.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path / "homolog"


def test_homolog_refs_reads_the_real_layout_and_logs_access(tmp_path, monkeypatch):
    monkeypatch.setenv("KAPI_HOMOLOG_DIR", str(_fake_homolog(tmp_path)))
    docs = homolog_refs.homolog_docs()
    assert len(docs) == 1 and docs[0][0] == "homolog" and docs[0][1]["snippet_id"] == "s1"      # 1 por snippet
    log = tmp_path / "homolog" / "v1" / "access.log.jsonl"
    with hg.authorized("teste: ler o log de acesso do homolog tmp", "test_grammar_v3"):     # fora do `authorized` o guard recusa ate a leitura do log
        assert log.exists() and "homolog_refs" in log.read_text(encoding="utf-8")
    with pytest.raises(hg.HomologAccessError):
        open(log, encoding="utf-8")                                                       # e o guard esta ativo neste processo


def test_homolog_refs_fails_loudly_when_sealed_but_items_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KAPI_HOMOLOG_DIR", str(_fake_homolog(tmp_path, with_items=False)))
    with pytest.raises(SystemExit):
        homolog_refs.homolog_docs()
