"""Corpus dos alunos: isolamento G2.7, determinismo, mesmos ids/ordem/linhas entre distilado e controle (gold), balanco estratificado."""
import json
import math
from collections import Counter
from pathlib import Path

import pytest

import build_student_corpus as bsc

ROOT = Path(__file__).resolve().parents[3]
ACCEPTED = ROOT / "data" / "distilled" / "v2c-final" / "accepted.jsonl"
EVAL_INDEX = ROOT / "data" / "distilled" / "v2c" / "eval_index.json"
needs_data = pytest.mark.skipif(not (ACCEPTED.exists() and EVAL_INDEX.exists() and (ROOT / "data" / "teacher" / "v2d" / "val").exists()),
                                reason="dados da destilacao/corpus do professor ausentes")


# -- funcoes puras ---------------------------------------------------------------------------------------------------------------------
def test_allocate_is_proportional_capped_and_exact():
    share = {"a": 0.5, "b": 0.3, "c": 0.2}
    out = bsc.allocate(100, share, {"a": 1000, "b": 1000, "c": 1000})
    assert out == {"a": 50, "b": 30, "c": 20}
    capped = bsc.allocate(100, share, {"a": 10, "b": 1000, "c": 1000})      # 'a' estoura o teto: o resto e redistribuido na proporcao b:c = 3:2
    assert capped["a"] == 10 and sum(capped.values()) == 100 and abs(capped["b"] - 54) <= 1
    assert sum(bsc.allocate(10 ** 6, share, {"a": 5, "b": 7, "c": 9}).values()) == 21     # nunca passa da disponibilidade


def test_selection_is_deterministic_nested_and_label_independent():
    recs = [{"id": f"r{i}", "cut_kind": ("k", "v")[i % 2], "category": "c", "context_kind": ("full", "none")[i % 3 == 0]} for i in range(600)]
    share = {s: 0.25 for s in {bsc.stratum_of(r) for r in recs}}
    t1 = bsc.select_ids(recs, 300, share, 7, "p")
    again = bsc.select_ids(list(reversed(recs)), 300, share, 7, "p")
    assert t1 == again                                                    # independe da ordem de entrada
    caps = Counter(t1.values())
    t0 = bsc.select_ids(recs, 120, share, 7, "p", caps)
    assert set(t0) <= set(t1)                                             # T0 subconjunto de T1
    for r in recs:
        r["gold"], r["target"] = "x", "y"                                 # trocar rotulos nao muda a selecao nem o componente
    assert bsc.select_ids(recs, 300, share, 7, "p") == t1
    comps = [bsc.component_of(r["id"], True, 7, 0.3, 0.07) for r in recs]
    assert comps == [bsc.component_of(r["id"], True, 7, 0.3, 0.07) for r in recs]
    assert bsc.component_of("zzz", False, 7, 1.0, 0.0) == "fim_ctx"       # completion so se elegivel


def test_isolation_flags_every_leak_kind():
    index = {"ids": ["ev1"], "group_ids": ["g_eval"], "schema_ids": ["s_eval"], "snippet_ids": ["sn_eval"], "prompt_sha256": ["p_eval"],
             "prefix_suffix_sha256": [], "chat_group_ids": [], "holdout_domains": ["legal"]}
    iso = bsc.Isolation(index)
    base = {"id": "t1", "split": "train", "group_id": "g", "schema_id": "s", "snippet_id": "sn", "prompt_sha256": "p", "domain": "saas",
            "prefix": "a", "suffix": "b", "holdout": False, "split_role": "train"}
    assert iso.violations(base) == []
    for patch, why in (({"split": "val"}, "split_not_train"), ({"group_id": "g_eval"}, "eval_group"), ({"schema_id": "s_eval"}, "eval_schema"),
                       ({"snippet_id": "sn_eval"}, "eval_snippet"), ({"prompt_sha256": "p_eval"}, "eval_prompt_sha"), ({"domain": "legal"}, "holdout_domain"),
                       ({"id": "ev1"}, "eval_id"), ({"parent_id": "ev1"}, "eval_id")):
        assert why in iso.violations({**base, **patch}), why


@needs_data
def test_real_accepted_prefix_is_isolated_and_train_only():
    """Os aceitos passam em TODAS as checagens G2.7 do construtor (60k primeiros aqui; o construtor roda os 270.993 e aborta se houver 1)."""
    from _distill_common import BlindChecker

    index = json.loads(EVAL_INDEX.read_text(encoding="utf-8"))
    iso = bsc.Isolation(index, BlindChecker())
    splits = ROOT / "data" / "splits" / "v2c"
    train = bsc.read_ids(splits / "train.ids.txt")
    others = {n: bsc.read_ids(splits / f"{n}.ids.txt") for n in ("val", "test", "bench")}
    content = set()
    with (ROOT / "data" / "distilled" / "v2c" / "eval_docs.jsonl").open(encoding="utf-8") as h:
        for line in h:
            d = json.loads(line)
            if d.get("type", "fim") == "fim":
                content.add(bsc.content_hash(d["prefix"], d["suffix"], d["middle"]))
    recs, stats = bsc.load_accepted(ACCEPTED, 60000, iso, train, others, content, BlindChecker())
    assert stats["read"] == 60000 and stats.get("isolation_dropped", 0) == 0 and len(recs) == 60000
    assert all(r["split"] == "train" and not r["holdout"] for r in recs)


def test_loader_drops_a_tampered_record(tmp_path):
    """Um registro que e de val (id em val.ids) ou cujo grupo esta na avaliacao e descartado e contado."""
    from _distill_common import BlindChecker

    index = {"ids": [], "group_ids": ["g_eval"], "schema_ids": [], "snippet_ids": [], "prompt_sha256": [], "prefix_suffix_sha256": [],
             "chat_group_ids": [], "holdout_domains": []}
    iso = bsc.Isolation(index, BlindChecker())
    rec = {"id": "a", "split": "train", "group_id": "g_eval", "schema_id": "s", "snippet_id": "x", "prompt_sha256": "p", "domain": "d", "prefix": "p",
           "suffix": "s", "target": "t", "gold": "g", "holdout": False, "split_role": "train", "label_source": "teacher_seq"}
    ok = {**rec, "id": "b", "group_id": "g_ok"}
    f = tmp_path / "acc.jsonl"
    f.write_text(json.dumps(rec) + "\n" + json.dumps(ok) + "\n", encoding="utf-8")
    recs, stats = bsc.load_accepted(f, None, iso, {"a", "b"}, {"val": {"a"}, "test": set(), "bench": set()}, set(), BlindChecker())
    assert [r["id"] for r in recs] == ["b"] and stats["isolation_dropped"] == 1 and stats["violation:eval_group"] == 1 and stats["violation:id_in_val"] == 1


# -- mini-corpus completo (renderizacao, empacotamento, bracos) -----------------------------------------------------------------------
@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    if not (ACCEPTED.exists() and (ROOT / "data" / "replay" / "v1" / "manifest.json").exists() and (ROOT / "data" / "teacher" / "v2d" / "val").exists()):
        pytest.skip("dados ausentes")
    root = tmp_path_factory.mktemp("stu")
    out = {}
    for tag in ("a", "b"):
        rc = bsc.main(["--limit", "3000", "--workers", "2", "--out-root", str(root / tag), "--report", str(root / tag / "r.md")])
        assert rc == 0
        out[tag] = root / tag
    return out


def test_mini_build_is_deterministic(mini):
    for tier in ("t1", "t0"):
        for arm in ("distill", "gold"):
            ma = json.loads((mini["a"] / arm / tier / "manifest.json").read_text(encoding="utf-8"))
            mb = json.loads((mini["b"] / arm / tier / "manifest.json").read_text(encoding="utf-8"))
            assert [s["sha256"] for s in ma["shards"]] == [s["sha256"] for s in mb["shards"]]
            assert [s["meta_sha256"] for s in ma["shards"]] == [s["meta_sha256"] for s in mb["shards"]]
            assert ma["verification"]["ids_sha256"] == mb["verification"]["ids_sha256"]


def test_arms_have_same_ids_order_rows_and_differ_only_in_teacher_labels(mini):
    import numpy as np

    for tier in ("t1", "t0"):
        ver = bsc.verify_pair(mini["a"], tier, ROOT / "data" / "splits" / "v2c")
        assert ver["n_issues"] == 0 and ver["same_ids_same_order"] and ver["same_rows"]
        assert ver["in_val_ids"] == ver["in_test_ids"] == ver["in_bench_ids"] == 0
        man = {a: json.loads((mini["a"] / a / tier / "manifest.json").read_text(encoding="utf-8")) for a in ("distill", "gold")}
        assert man["distill"]["counts"]["examples"] == man["gold"]["counts"]["examples"]
        differ = Counter()
        for sd, sg in zip(man["distill"]["shards"], man["gold"]["shards"]):
            with np.load(mini["a"] / "distill" / tier / sd["file"]) as d, np.load(mini["a"] / "gold" / tier / sg["file"]) as g:
                assert d["row_ptr"].tolist() == g["row_ptr"].tolist()
                assert d["loss_start"].tolist() == g["loss_start"].tolist()          # mesmo prompt em cada exemplo
                for i in range(min(len(d["loss_start"]), 400)):
                    pd_ = d["ids"][d["ex_offsets"][i]: d["ex_offsets"][i] + d["loss_start"][i]]
                    pg = g["ids"][g["ex_offsets"][i]: g["ex_offsets"][i] + g["loss_start"][i]]
                    assert pd_.tolist() == pg.tolist()
            metas_d = [json.loads(x) for x in (mini["a"] / "distill" / tier / sd["meta_file"]).read_text(encoding="utf-8").splitlines()]
            metas_g = [json.loads(x) for x in (mini["a"] / "gold" / tier / sg["meta_file"]).read_text(encoding="utf-8").splitlines()]
            for md, mg in zip(metas_d, metas_g):
                assert md["id"] == mg["id"] and md["component"] == mg["component"] and md["n_prompt"] == mg["n_prompt"]
                differ[(md["component"], md["label_hash"] != mg["label_hash"])] += 1
        for comp in ("fim_raw", "replay_fim", "replay_causal"):
            assert differ[(comp, True)] == 0                                          # componentes sem professor: rotulo identico
        assert differ[("fim_ctx", True)] > 0                                          # com professor: rotulos diferem onde o professor discorda
        assert man["distill"]["label_source"].startswith("teacher_seq") and man["gold"]["label_source"].startswith("gold")


def test_labels_are_the_teacher_target_and_the_gold_middle(mini):
    import numpy as np
    from render_contract import Tokenizer

    tok = Tokenizer(ROOT / "BaseRepos" / "Qwen2.5-Coder-0.5B" / "tokenizer.json")
    acc = {}
    with ACCEPTED.open(encoding="utf-8") as h:
        for n, line in enumerate(h):
            if n >= 3000:
                break
            r = json.loads(line)
            acc[r["id"]] = r
    checked = Counter()
    d, g = mini["a"] / "distill" / "t1", mini["a"] / "gold" / "t1"
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    for sh in man["shards"]:
        with np.load(d / sh["file"]) as zd, np.load(g / sh["file"]) as zg:
            metas = [json.loads(x) for x in (d / sh["meta_file"]).read_text(encoding="utf-8").splitlines()]
            for i, m in enumerate(metas):
                if m["component"] != "fim_ctx" or m["id"] not in acc:
                    continue
                td = tok.decode(zd["ids"][zd["ex_offsets"][i] + zd["loss_start"][i]: zd["ex_offsets"][i + 1] - 1].tolist())
                tg = tok.decode(zg["ids"][zg["ex_offsets"][i] + zg["loss_start"][i]: zg["ex_offsets"][i + 1] - 1].tolist())
                assert td == acc[m["id"]]["target"] and tg == acc[m["id"]]["gold"]
                assert zd["ids"][zd["ex_offsets"][i + 1] - 1] == bsc.EOS_ID
                checked[td == tg] += 1
    assert checked[True] > 0 and checked[False] > 0


def test_t0_is_subset_of_t1_and_replay_is_isolated(mini):
    sel = {t: {json.loads(line)["id"] for line in (mini["a"] / "selection" / f"{t}.jsonl").read_text(encoding="utf-8").splitlines()} for t in ("t0", "t1")}
    mongo_t0 = {i for i in sel["t0"] if not i.startswith("rpl-")}
    mongo_t1 = {i for i in sel["t1"] if not i.startswith("rpl-")}
    assert mongo_t0 <= mongo_t1
    man = json.loads((mini["a"] / "distill" / "t1" / "manifest.json").read_text(encoding="utf-8"))
    rep = man["replay"]
    assert rep["isolation"]["package_overlap_with_generic_eval"] == [] and rep["isolation"]["chunks_containing_eval_cut"] == 0
    assert math.isclose(rep["token_share_achieved"], 0.12, abs_tol=0.03)
