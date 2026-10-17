"""Isolamento G2.7 do pool de prompts: so train, nunca ids/grupos/schemas/snippets/prompts de val/test/bench, gold fora do prompt."""
import json
from pathlib import Path

import pytest

import build_prompt_pool as bpp
from _distill_common import ROOT, BlindChecker

VALIDATED = ROOT / "data" / "validated" / "v2c"


def rec(i, split, group, schema="s/train1", **kw):
    r = {"id": f"fim-{i}", "type": "fim", "split": split, "prefix": f"db.c.find({{ k{i}", "suffix": f": 1 }})", "middle": "SEGREDO",
         "domain": "commerce", "schema_id": schema, "collection": "c", "cut_kind": "key", "group_id": group, "snippet_id": f"snip-{i}",
         "category": "crud", "holdout": False, "split_role": "train" if split == "train" else "eval", "context_kind": "full",
         "editor": None, "language": "Mongo Console JavaScript", "prompt_sha256": f"sha{i}", "exec": "ok", "dialect": "console",
         "validation": {"x": 1}, "facts": [{"kind": "x"}], "target": "SEGREDO", "known_names": []}
    r.update(kw)
    return r


def test_compact_drops_label_and_keeps_gold_separate():
    c = bpp.compact(rec(1, "train", "g1"))
    assert "middle" not in c and "target" not in c and "facts" not in c and "validation" not in c
    assert c["gold"] == "SEGREDO" and c["gold_exec"] == "ok" and c["origin"] == "base"


def make_index(tmp_path):
    eval_recs = [rec(100, "val", "g-eval", schema="s/eval1"), rec(101, "test", "g-eval2", schema="s/eval2")]
    parts = []
    idx = {"ids": [], "groups": set(), "schemas": set(), "snippets": set(), "prompt_sha": set(), "ps_sha": set()}
    for r in eval_recs:
        idx["ids"].append(r["id"])
        idx["groups"].add(r["group_id"])
        idx["schemas"].add(r["schema_id"])
        idx["snippets"].add(r["snippet_id"])
        idx["prompt_sha"].add(r["prompt_sha256"])
        idx["ps_sha"].add(bpp.ps_sha(r["prefix"], r["suffix"]))
    return bpp.build_eval_index([idx], {"groups": {"g-chat-eval"}}, ["education", "legal"])


def test_isolation_flags_every_leak_kind(tmp_path):
    iso = bpp.Isolation(make_index(tmp_path), BlindChecker(tmp_path / "nao-existe"))
    ok = bpp.compact(rec(1, "train", "g1"))
    assert iso.violations(ok) == []
    assert "split_not_train" in iso.violations(bpp.compact(rec(2, "val", "g1")))
    assert "eval_group" in iso.violations(bpp.compact(rec(3, "train", "g-eval")))
    assert "eval_group" in iso.violations(bpp.compact(rec(3, "train", "g-chat-eval")))
    assert "eval_schema" in iso.violations(bpp.compact(rec(4, "train", "g4", schema="s/eval1")))
    assert "eval_snippet" in iso.violations(bpp.compact(rec(5, "train", "g5", snippet_id="snip-100")))
    assert "eval_id" in iso.violations(bpp.compact(rec(100, "train", "g6")))
    assert "eval_prompt_sha" in iso.violations(bpp.compact(rec(7, "train", "g7", prompt_sha256="sha101")))
    assert "eval_prefix_suffix" in iso.violations(bpp.compact(rec(8, "train", "g8", prefix="db.c.find({ k100")))
    assert "holdout_domain" in iso.violations(bpp.compact(rec(9, "train", "g9", domain="legal")))
    assert "schema_role_eval" in iso.violations(bpp.compact(rec(10, "train", "g10", split_role="eval")))
    child = bpp.compact(rec(11, "train", "g11"))
    child["parent_id"] = "fim-100"                       # re-renderizado de um corte de avaliacao
    assert "eval_id" in iso.violations(child)


def test_blind_prompt_is_blocked(tmp_path):
    import blind_guard as bg
    blind = tmp_path / "blind"
    (blind / "sealed").mkdir(parents=True)
    r = rec(1, "train", "g1")
    (blind / "sealed" / "manifest.json").write_text(json.dumps({"prompt_sha256": [bg.prompt_sha256(r["prefix"] + "\x00" + r["suffix"])]}))
    iso = bpp.Isolation(make_index(tmp_path), BlindChecker(blind))
    assert "blind" in iso.violations(bpp.compact(r))
    assert BlindChecker(blind).sealed


def test_scan_shard_separates_train_and_eval(tmp_path):
    p = tmp_path / "fim-00000.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in (rec(1, "train", "g1"), rec(2, "val", "g2"), rec(3, "test", "g3"), rec(4, "bench", "g4"), rec(5, "train", "g5")):
            f.write(json.dumps(r) + "\n")
    out = bpp.scan_shard((str(p), 1.0, "salt", "train"))
    assert {c["id"] for _, c in out["train"]} == {"fim-1", "fim-5"}
    assert {d["id"] for d in out["eval_docs"]} == {"fim-2", "fim-3", "fim-4"}
    assert all(c["split"] == "train" for _, c in out["train"])
    # a amostragem por hash nao depende da ordem de leitura
    a = bpp.scan_shard((str(p), 0.5, "salt", "train"))
    b = bpp.scan_shard((str(p), 0.5, "salt", "train"))
    assert [c["id"] for _, c in a["train"]] == [c["id"] for _, c in b["train"]]


def test_main_on_synthetic_source_drops_leaks_and_never_uses_eval(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rows = [rec(i, "train", f"g{i}", schema="s/train1") for i in range(1, 40)]
    rows += [rec(100, "val", "g-eval", schema="s/eval1"), rec(101, "test", "g-eval2", schema="s/eval2")]
    rows += [rec(200, "train", "g-eval", schema="s/train1")]                 # grupo de treino IGUAL ao de val: vazamento
    rows += [rec(201, "train", "g201", schema="s/eval1")]                    # schema de avaliacao no treino: vazamento
    with open(src / "fim-00000.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (src / "manifest.json").write_text(json.dumps({"counts": {"by_type_split": {"fim.train": 41}}}))
    out = tmp_path / "out"
    rc = bpp.main(["--source-dir", str(src), "--out-dir", str(out), "--eval-cache", str(out), "--target", "30", "--no-rerender", "--workers", "1"])
    assert rc == 0
    pool = [json.loads(l) for l in (out / "pool" / "pool-00000.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(pool) == 30
    assert all(p["split"] == "train" for p in pool)
    ids = {p["id"] for p in pool}
    assert not ids & {"fim-100", "fim-101", "fim-200", "fim-201"}
    assert all(p["gold"] == "SEGREDO" and "middle" not in p for p in pool)
    man = json.loads((out / "pool" / "pool.manifest.json").read_text(encoding="utf-8"))
    assert man["isolation_G2_7"]["violations_in_pool"] == 0
    assert man["isolation_G2_7"]["dropped_by_reason"].get("eval_group", 0) >= 1
    assert man["isolation_G2_7"]["dropped_by_reason"].get("eval_schema", 0) >= 1


@pytest.mark.skipif(not (VALIDATED / "fim-00000.jsonl").exists(), reason="data/validated/v2c ausente")
def test_real_pool_is_train_only_and_isolated(tmp_path):
    out = tmp_path / "out"
    rc = bpp.main(["--source-dir", str(VALIDATED), "--out-dir", str(out), "--eval-cache", str(out), "--max-shards", "1", "--target", "300",
                   "--no-rerender", "--workers", "1"])
    assert rc == 0
    pool = [json.loads(l) for l in (out / "pool" / "pool-00000.jsonl").read_text(encoding="utf-8").splitlines()]
    index = json.loads((out / "eval_index.json").read_text(encoding="utf-8"))
    assert len(pool) == 300 and all(p["split"] == "train" and not p["holdout"] for p in pool)
    ev = set(index["ids"]), set(index["group_ids"]), set(index["schema_ids"])
    assert not any(p["id"] in ev[0] or p["group_id"] in ev[1] or p["schema_id"] in ev[2] for p in pool)
    assert all("middle" not in p and "target" not in p and "prompt" not in p for p in pool)
