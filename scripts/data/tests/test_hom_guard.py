"""Guarda do papel `hom` (homologacao): nunca em train/val/test/bench; validate/decontam falham; corpora abortam; regressao dos antigos."""
import hashlib
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
ROOT = DATA.parents[1]
for p in (DATA, ROOT / "scripts" / "distill", ROOT / "scripts" / "teacher", ROOT / "scripts" / "student"):
    sys.path.insert(0, str(p))
site = os.environ.get("KAPI_TOKENIZERS_SITE") or r"C:\SlopStudio.MongoAI-artifacts\.venv\Lib\site-packages"
if Path(site).exists() and site not in sys.path:
    sys.path.append(site)

import decontaminate as dc  # noqa: E402
import gen_snippets as gs  # noqa: E402
import hom_guard as hg  # noqa: E402
import split_roles as sr  # noqa: E402

HOM_IDS = sorted(k for k, v in sr._load().items() if v == "hom")
pytestmark = pytest.mark.skipif(not HOM_IDS, reason="sem schemas hom (v3) no repositorio")
OLD_PIN_500 = "7d33d9420fd0e061c12e8125e2e9004324d6a436583257a6e559aa2ca7255e64"   # 500 snippets, seed 20261008, 110 schemas antigos


def _hom_rec(i="x1", split="train", **kw):
    sid = HOM_IDS[0]
    dom = "telecom" if sid.endswith("_tel") else "adtech"
    r = {"id": i, "type": "fim", "split": split, "domain": dom, "schema_id": f"{dom}/{sid}", "prefix": "db.a.find({ ", "middle": "k", "suffix": ": 1 })"}
    r.update(kw)
    return r


def _ok_rec(i="o1", split="train"):
    return {"id": i, "type": "fim", "split": split, "domain": "saas", "schema_id": "saas/atlas_saas", "prefix": "p", "middle": "m", "suffix": "s"}


# ------------------------------------------------------------------ (a) geradores
def test_load_schemas_excludes_hom_by_default():
    sch = gs.load_schemas(ROOT / "data" / "schemas")
    assert sch and not [s for s in sch if sr.role_for(s.schema_id) == "hom" or s.domain in sr.HOM_DOMAINS]
    assert not ({s.domain for s in sch} & {"telecom", "adtech"})
    with_hom = gs.load_schemas(ROOT / "data" / "schemas", include_hom=True)
    assert {s.schema_id for s in with_hom if sr.role_for(s.schema_id) == "hom"} == set(HOM_IDS)


def test_generated_snippets_never_hom_in_any_split():
    cfg = gs.load_config()
    cfg["seed"] = 20261008
    sch = gs.load_schemas(ROOT / "data" / "schemas")
    seen = set()
    for r in gs.generate_iter(cfg, sch, 1500):
        assert not hg.record_is_hom(r), r["schema_id"]
        assert r["split"] in ("train", "val", "test", "bench")
        seen.add(r["split"])
    assert "train" in seen


def test_include_hom_marks_homolog_split():
    cfg = gs.load_config()
    cfg["seed"] = 20261008
    sch = [s for s in gs.load_schemas(ROOT / "data" / "schemas", include_hom=True) if sr.role_for(s.schema_id) == "hom"][:6]
    recs = list(gs.generate_iter(cfg, sch, 60))
    assert recs and all(r["split"] == "homolog" for r in recs)


def test_drop_hom_filters_stream():
    recs = [_ok_rec("a"), _hom_rec("b"), _ok_rec("c", "val"), _hom_rec("d", "bench")]
    assert [r["id"] for r in hg.drop_hom(recs)] == ["a", "c"]
    assert len(list(hg.drop_hom(recs, include_hom=True))) == 4


def test_v3_train_and_eval_schemas_accepted():
    sch = gs.load_schemas(ROOT / "data" / "schemas")
    roles = {sr.role_for(s.schema_id) for s in sch if s.schema_id.startswith("v3_")}
    assert roles == {"train", "eval"}


# ------------------------------------------------------------------ (b) validate_pipeline / decontaminate
def test_decontaminate_main_fails_with_hom_in_input(tmp_path):
    f = tmp_path / "in.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in (_ok_rec(), _hom_rec())) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        dc.main(["--in", str(f), "--report-dir", str(tmp_path / "rep"), "--no-homolog"])
    assert e.value.code not in (0, None) and "hom" in str(e.value.code)


def test_validate_pipeline_fails_with_hom_in_training_input(tmp_path):
    import validate_pipeline as vp
    f = tmp_path / "fim.jsonl"
    f.write_text(json.dumps(_hom_rec()) + "\n", encoding="utf-8")
    chat = tmp_path / "chat.jsonl"
    chat.write_text("", encoding="utf-8")
    a = Namespace(config=str(gs.CONFIG_PATH), ver="t", fim=str(f), chat=str(chat), snippets=str(tmp_path / "none.jsonl"), include_snippets=False,
                  out_dir=str(tmp_path / "o"), splits_dir=str(tmp_path / "s"), report=str(tmp_path / "r.md"), decontam_dir=str(tmp_path / "d"),
                  exec="recorded", tokenizer="off", no_node=True, workers=1, legacy=[], ngram_min_content=None, audit_literal=False,
                  analysis_ngram_min_content=0, shard_size=0, compare=None, dedup_jaccard=dc.DEDUP_JACCARD, limit=None)
    with pytest.raises(SystemExit) as e:
        vp.run_pipeline(a)
    assert "hom" in str(e.value.code)


def test_gate_g27_hom_present_in_validate_pipeline():
    src = (DATA / "validate_pipeline.py").read_text(encoding="utf-8")
    assert 'g["G2.7-hom"]' in src and "hom_guard.record_is_hom" in src


def test_load_homolog_uses_same_official_rule(tmp_path):
    d = tmp_path / "v1"
    d.mkdir()
    rec = {"id": "h1", "snippet_id": "s", "instruction": "lista usuarios", "output": "db.users.find({ age: { $gt: 3 } })", "context": {}}
    (d / "homolog.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    ev = dc.load_homolog(d)
    assert ev and ev[0][0] == "homolog"
    train = [{"id": "t1", "snippet_id": "s2", "instruction": "lista usuarios", "output": "db.users.find({ age: { $gt: 3 } })", "split": "train"}]
    res = dc.run(train, ev, seed=1, min_content=2)
    assert "t1" in res["removed"]                      # copia exata do homolog e removida do treino
    assert dc.load_homolog(tmp_path / "nao_existe") == []


# ------------------------------------------------------------------ (c) corpora professor / aluno / pool
def test_pool_isolation_rejects_hom():
    pytest.importorskip("numpy")
    import build_prompt_pool as bpp
    iso = bpp.Isolation({"ids": [], "group_ids": [], "schema_ids": [], "snippet_ids": [], "prompt_sha256": [], "prefix_suffix_sha256": []})
    assert iso.violations(_hom_rec(snippet_id="z"))[0] == "hom_schema"


def test_pool_main_aborts_with_hom(tmp_path):
    pytest.importorskip("numpy")
    import yaml
    import build_prompt_pool as bpp
    src = tmp_path / "src"
    src.mkdir()
    (src / "fim-00000.jsonl").write_text(json.dumps(_hom_rec("fim-1", prefix="p", suffix="s")) + "\n", encoding="utf-8")
    cfg = bpp.load_config()
    cfg["source_dir"] = str(src)
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        bpp.main(["--config", str(cp), "--workers", "1", "--target", "10", "--out-dir", str(tmp_path / "o"), "--no-rerender"])
    assert "hom" in str(e.value.code)


def test_student_load_accepted_aborts_with_hom(tmp_path):
    pytest.importorskip("numpy")
    import build_student_corpus as bsc
    p = tmp_path / "acc.jsonl"
    p.write_text(json.dumps(_hom_rec("a1")) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        bsc.load_accepted(p, None, None, set(), {}, set(), None)
    assert "hom" in str(e.value.code)


def test_teacher_scan_flags_hom(tmp_path):
    pytest.importorskip("numpy")
    import build_teacher_corpus as btc
    p = tmp_path / "fim-00000.jsonl"
    p.write_text(json.dumps(_hom_rec("a1")) + "\n", encoding="utf-8")
    out = btc._scan_shard(str(p))
    assert out["hom"] and not out["train"] and not out["counts"]


# ------------------------------------------------------------------ (d) regressao: schemas antigos identicos
def test_old_schemas_output_unchanged():
    cfg = gs.load_config()
    cfg["seed"] = 20261008
    sch = [s for s in gs.load_schemas(ROOT / "data" / "schemas") if not s.schema_id.startswith("v3_")]
    assert len(sch) == 110
    h = hashlib.sha256()
    for r in gs.generate_iter(cfg, sch, 500):
        h.update(json.dumps(r, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    assert h.hexdigest() == OLD_PIN_500
