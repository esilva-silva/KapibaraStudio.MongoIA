"""Corpus v3 dos alunos: isolamento hom / G2.7, determinismo, proveniencia por exemplo, balanceamento, fidelidade ao schema, T0 subconjunto de T1."""
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "scripts" / "data" / "llm",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_student_corpus_v3 as v3  # noqa: E402
import build_student_corpus as bsc  # noqa: E402

CFG = ROOT / "configs" / "student_corpus_v3.yaml"
GRAMMAR = ROOT / "data" / "validated" / "v3g" / "manifest.json"
LLM = ROOT / "data" / "generated" / "v3" / "train" / "fim_ctx.jsonl"
needs_data = pytest.mark.skipif(
    not (GRAMMAR.exists() and LLM.exists() and (ROOT / "data" / "distilled" / "v2c-final" / "accepted.jsonl").exists()
         and (ROOT / "data" / "teacher" / "v2d" / "val").exists() and (ROOT / "data" / "replay" / "v1" / "manifest.json").exists()),
    reason="dados do corpus v3 (gramatica validada, LLM, destilado, replay) ausentes")


# -- funcoes puras ---------------------------------------------------------------------------------------------------------------------
NAMES = frozenset({"orders", "status", "customer_id", "customer", "total", "_id", "shop"})


def test_label_need_levels():
    assert v3.label_need("$sum", "x", NAMES) == 0                                   # sem nomes do schema
    assert v3.label_need("status: 'A'", "db.orders.find({ status: ", NAMES) == 1     # nome visivel
    assert v3.label_need("customer_id", "db.orders.find({ ", NAMES) == 2             # campo do schema ausente do texto: adivinhar
    assert v3.label_need("customer.total", "db.orders.find({ customer: 1 ", NAMES) == 2   # um dos segmentos ausente
    assert v3.label_need("_id", "db.orders.find({ ", NAMES, frozenset({"_id"})) == 0  # `_id` ignorado (existe em todo schema)


def test_header_text_follows_context_kind():
    rec = {"context_kind": "full", "editor": {"known_names": ["orders"], "result_fields": ["customer_id"], "recent_commands": ["db.x.find()"], "input_panel": ""}}
    assert "customer_id" in v3.header_text(rec) and "orders" in v3.header_text(rec)
    assert v3.header_text({**rec, "context_kind": "none"}) == ""
    assert v3.header_text({"context_kind": "full", "editor": None}) == ""


def test_source_targets_cover_every_source_and_sum_to_one():
    cfg = v3.merge_cfg(CFG)
    dims = {"cut_kind": {"key": 3, "value": 2}, "category": {"crud": 2, "aggregation": 2, "atlas_search": 1}}
    t = v3.source_targets(cfg, dims, {"llm": {"crud", "aggregation"}})
    assert set(t) == set(v3.SOURCES)
    for s in v3.SOURCES:
        assert abs(sum(t[s].values()) - 1.0) < 1e-9
    assert all(k[1] in ("crud", "aggregation") for k in t["llm"])                  # o LLM so cobre as categorias que tem
    assert {k[1] for k in t["grammar"]} == set(cfg["composition"]["grammar_category"])
    ctx = Counter()
    for k, v in t["accepted"].items():
        ctx[k[2]] += v
    want = cfg["composition"]["context_kind_by_source"]["accepted"]
    assert abs(ctx["none"] + ctx["irrelevant"] - (want["none"] + want["irrelevant"])) < 1e-9


def test_rebalance_pulls_global_marginals_toward_the_pool_target():
    tgt = {"cut_kind": {"key": 0.5, "value": 0.5}, "category": {"crud": 0.6, "dsl": 0.4}}
    got = {"cut_kind": {"key": 0.7, "value": 0.3}, "category": {"crud": 0.6, "dsl": 0.4}}
    adj, dev = v3.rebalance({}, tgt, got)
    assert dev == pytest.approx(20.0) and adj["cut"]["key"] < 1.0 < adj["cut"]["value"]
    assert adj["cat"]["crud"] == pytest.approx(1.0)
    cfg = v3.merge_cfg(CFG)
    dims = {"cut_kind": {"key": 1, "value": 1}, "category": {"crud": 1, "dsl": 1}}
    base = v3.source_targets(cfg, dims, {"llm": {"crud", "dsl"}})
    moved = v3.source_targets(cfg, dims, {"llm": {"crud", "dsl"}}, adj)
    for s in v3.SOURCES:
        assert abs(sum(moved[s].values()) - 1.0) < 1e-9
        k = next(k for k in base[s] if k[0] == "key" and k[2] == "full" and k[1] == "crud")
        v = next(x for x in base[s] if x[0] == "value" and x[2] == "full" and x[1] == "crud")
        assert moved[s][k] / moved[s][v] < base[s][k] / base[s][v]


def _fake_by_src(n=400):
    by = {s: {} for s in v3.SOURCES}
    for s, off in (("accepted", 0), ("llm", 1000), ("grammar", 2000)):
        for k in range(n):
            st = ("key" if k % 2 else "value", "crud", ("full", "none", "irrelevant")[k % 3])
            by[s].setdefault(st, []).append(f"{s}-{k:04d}")
    return by


def test_select_respects_new_share_context_target_and_is_deterministic():
    cfg = v3.merge_cfg(CFG)
    comp = cfg["composition"]
    by = _fake_by_src()
    dims = {"cut_kind": {"key": 1, "value": 1}, "category": {"crud": 1}}
    tg = v3.source_targets(cfg, dims, {"llm": {"crud"}})
    a = v3.select(by, 600, comp, tg, None)
    b = v3.select(by, 600, comp, tg, None)
    assert a == b
    n = sum(len(v) for v in a.values())
    assert n == 600
    new = (len(a["llm"]) + len(a["grammar"])) / n
    assert abs(new - comp["new_share"]) < 0.01
    allids = [i for s in v3.SOURCES for i in a[s]]
    assert len(set(allids)) == len(allids)
    # T0: subconjunto de T1 (teto = selecao de T1)
    caps = {s: dict(Counter(st for st, ids in by[s].items() for i in ids if i in set(a[s]))) for s in v3.SOURCES}
    small = v3.select(by, 300, comp, tg, caps)
    for s in v3.SOURCES:
        assert set(small[s]) <= set(a[s])


def test_calibrate_hits_token_budget_and_is_monotonic():
    cfg = v3.merge_cfg(CFG)
    by = _fake_by_src()
    dims = {"cut_kind": {"key": 1, "value": 1}, "category": {"crud": 1}}
    tg = v3.source_targets(cfg, dims, {"llm": {"crud"}})
    tok = {i: 100 + (hash(i) % 7) for s in v3.SOURCES for ids in by[s].values() for i in ids}
    n, ch = v3.calibrate(by, tok, 30000, cfg["composition"], tg, None)
    got = sum(tok[i] for s in v3.SOURCES for i in ch[s])
    assert 30000 <= got < 30000 + 700
    n2, _ = v3.calibrate(by, tok, 60000, cfg["composition"], tg, None)
    assert n2 > n


# -- isolamento: hom nunca entra; G2.7 por registro ----------------------------------------------------------------------------------
def _loader_stub(**kw):
    ld = v3.Loader.__new__(v3.Loader)
    from collections import Counter as C

    from _distill_common import BlindChecker
    from build_prompt_pool import Isolation
    index = {"ids": ["ev1"], "group_ids": ["g_eval"], "schema_ids": ["finance/eval_s"], "snippet_ids": ["sn_eval"], "prompt_sha256": [],
             "prefix_suffix_sha256": [], "chat_group_ids": [], "holdout_domains": ["education", "legal"]}
    ld.iso = Isolation(index, BlindChecker())
    ld.trainable = {"hr/ok_s", "gaming/ok_g"}
    ld.v3_eval_snips, ld.v3_eval_ids = {"sn_v3eval"}, set()
    ld.eval_content = set()
    ld.sealed_items, ld.sealed_prompts = set(), set()
    ld.names, ld.schema_gen = {("hr/ok_s", "c"): frozenset({"name"})}, {"hr/ok_s": "base"}
    ld.ignore, ld.keep_unobs, ld.seed = frozenset(), 0.0, 1
    ld.stats = {s: C() for s in v3.SOURCES}
    ld.seen_content = set()
    return ld


def _rec(**kw):
    r = {"id": "r1", "split": "train", "holdout": False, "domain": "hr", "split_role": "train", "group_id": "g_ok", "schema_id": "hr/ok_s",
         "snippet_id": "sn1", "prompt_sha256": "p", "prefix": "db.c.find({ ", "suffix": " })", "gold": "x: 1", "target": "x: 1", "collection": "c",
         "context_kind": "none", "editor": None, "cut_kind": "key", "category": "crud", "agrees_with_gold": True, "origin": "llm", "language": "x"}
    r.update(kw)
    return r


def test_hom_schema_record_aborts_the_loader():
    ld = _loader_stub()
    for rec in (_rec(schema_id="telecom/v3_x", domain="telecom"), _rec(domain="adtech"), _rec(split_role="hom")):
        with pytest.raises(SystemExit) as e:
            ld._admit(rec, "llm", "gold", "llm-qwen3-14b", lambda r: r["gold"])
        assert "hom-guard" in str(e.value)


@pytest.mark.parametrize("patch,reason", [({"split": "val"}, "split_not_train"), ({"group_id": "g_eval"}, "eval_group"),
                                          ({"snippet_id": "sn_eval"}, "eval_snippet"), ({"id": "ev1"}, "eval_id"),
                                          ({"schema_id": "hr/not_train"}, "schema_not_train_role"), ({"snippet_id": "sn_v3eval"}, "v3_eval_overlap"),
                                          ({"holdout": True}, "holdout_domain"), ({"split_role": "eval"}, "schema_role_eval")])
def test_g27_violations_are_dropped_and_counted(patch, reason):
    ld = _loader_stub()
    assert ld._admit(_rec(**patch), "llm", "gold", "llm-qwen3-14b", lambda r: r["gold"]) is None
    assert ld.stats["llm"][f"violation:{reason}"] == 1 and ld.stats["llm"]["isolation_dropped"] == 1


def test_clean_record_is_admitted_with_provenance_and_content_dedup():
    ld = _loader_stub()
    out = ld._admit(_rec(), "llm", "gold", "llm-qwen3-14b", lambda r: r["gold"])
    assert out and out["source"] == "llm-qwen3-14b" and out["label_source"] == "gold" and out["schema_gen"] == "base"
    assert ld._admit(_rec(id="r2"), "llm", "gold", "llm-qwen3-14b", lambda r: r["gold"]) is None     # mesmo conteudo
    assert ld.stats["llm"]["duplicate_content"] == 1


def test_sealed_homolog_prompt_hash_and_eval_pair_are_soft_drops_not_aborts():
    ld = _loader_stub()
    import homolog_guard as hg
    ld.sealed_prompts = {hg.prompt_sha256("db.c.find({ \x00 })")}
    assert ld._admit(_rec(), "llm", "gold", "llm-qwen3-14b", lambda r: r["gold"]) is None
    assert ld.stats["llm"]["soft_dropped:homolog_hash"] == 1 and ld.stats["llm"]["isolation_dropped"] == 0     # contado, nao aborta
    ld.iso.ps_sha = {bsc.ps_sha("db.c.find({ ", " })")}
    assert ld._admit(_rec(id="r9", prefix="db.c.find({ ", suffix=" })", gold="x: 1", target="x: 1"), "llm", "gold", "t", lambda r: r["gold"]) is None
    assert ld.stats["llm"]["soft_dropped:eval_prefix_suffix"] == 1 and ld.stats["llm"]["isolation_dropped"] == 0


def test_fidelity_filters_unobservable_names():
    ld = _loader_stub()
    # none/irrelevant: rotulo com nome do schema ausente do texto -> descartado
    ld2 = _loader_stub()
    ld2.names = {("hr/ok_s", "c"): frozenset({"salary"})}
    assert ld2._admit(_rec(gold="salary: 1", target="salary: 1"), "llm", "gold", "t", lambda r: r["gold"]) is None
    assert ld2.stats["llm"]["fidelity_dropped:unobservable_blind"] == 1
    # full com keep=0 e nome ausente do prompt inteiro -> descartado; presente no cabecalho -> mantido
    ed = {"known_names": [], "result_fields": ["salary"], "recent_commands": [], "input_panel": ""}
    ld3 = _loader_stub()
    ld3.names = {("hr/ok_s", "c"): frozenset({"salary"})}
    keep = ld3._admit(_rec(gold="salary: 1", target="salary: 1", context_kind="full", editor=ed), "llm", "gold", "t", lambda r: r["gold"])
    assert keep is not None and keep["need_prompt"] == 1
    ld4 = _loader_stub()
    ld4.names = {("hr/ok_s", "c"): frozenset({"salary"})}
    ed2 = {"known_names": [], "result_fields": [], "recent_commands": [], "input_panel": ""}
    assert ld4._admit(_rec(gold="salary: 1", target="salary: 1", context_kind="full", editor=ed2), "llm", "gold", "t", lambda r: r["gold"]) is None
    assert ld4.stats["llm"]["fidelity_dropped:unobservable_full"] == 1


# -- dados reais ---------------------------------------------------------------------------------------------------------------------
@needs_data
def test_real_new_sources_are_train_only_non_hom_and_have_gold_labels():
    n = 0
    with LLM.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            assert r["split"] == "train" and r["split_role"] == "train" and r["author"] == "llm" and r["domain"] not in ("telecom", "adtech")
            n += 1
            if n >= 3000:
                break
    for p in sorted((ROOT / "data" / "validated" / "v3g").glob("fim*.jsonl"))[:1]:
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                assert r["split"] == "train" and r["split_role"] == "train" and r["domain"] not in ("telecom", "adtech") and r["source"] == "grammar-v3"
                assert r["schema_id"].split("/", 1)[1].startswith("v3_")
                if i >= 3000:
                    break


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    root = tmp_path_factory.mktemp("stu3")
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    cfg["tiers"]["t1"]["tokens_per_epoch"] = 1_500_000
    cfg["tiers"]["t0"]["tokens_per_epoch"] = 900_000
    cfgp = root / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    cfg_abs = cfg
    cfg_abs["base_config"] = str(ROOT / cfg["base_config"])
    out = {}
    for tag in ("a", "b"):
        rc = v3.main(["--config", str(cfgp), "--limit", "6000", "--workers", "2", "--out-root", str(root / tag), "--report", str(root / tag / "r.md"),
                      "--no-decontam-check"])
        assert rc == 0
        out[tag] = root / tag
    return out


@needs_data
def test_mini_build_is_deterministic(mini):
    for tier in ("t1", "t0"):
        ma = json.loads((mini["a"] / tier / "manifest.json").read_text(encoding="utf-8"))
        mb = json.loads((mini["b"] / tier / "manifest.json").read_text(encoding="utf-8"))
        assert [s["sha256"] for s in ma["shards"]] == [s["sha256"] for s in mb["shards"]]
        assert [s["meta_sha256"] for s in ma["shards"]] == [s["meta_sha256"] for s in mb["shards"]]
        assert ma["verification"]["ids_sha256"] == mb["verification"]["ids_sha256"]


@needs_data
def test_mini_provenance_and_verification(mini):
    cfg = v3.merge_cfg(CFG)
    for tier in ("t1", "t0"):
        ver = v3.verify_v3(mini["a"], tier, cfg)
        assert ver["n_issues"] == 0, ver["issues"]
        man = json.loads((mini["a"] / tier / "manifest.json").read_text(encoding="utf-8"))
        metas = [json.loads(line) for sh in man["shards"] for line in (mini["a"] / tier / sh["meta_file"]).read_text(encoding="utf-8").splitlines()]
        for m in metas:
            assert m["source"] and m["label_source"]
            if m["src"] == "replay":
                assert m["source"] == "replay-js" and m["label_source"] == "replay"
            else:
                assert m["split_role"] == "train" and m["domain"] not in ("telecom", "adtech")
                if m["source"] in ("llm-qwen3-14b", "grammar-v3"):
                    assert m["label_source"] == "gold" and not m["label_differs"]
        assert {m["source"] for m in metas} >= {"accepted-v2c-teacher", "llm-qwen3-14b", "grammar-v3", "replay-js"}


@needs_data
def test_mini_balance_composition_and_t0_subset(mini):
    cfg = v3.merge_cfg(CFG)
    comp = cfg["composition"]
    bal = {}
    for tier in ("t1", "t0"):
        man = json.loads((mini["a"] / tier / "manifest.json").read_text(encoding="utf-8"))
        metas = [json.loads(line) for sh in man["shards"] for line in (mini["a"] / tier / sh["meta_file"]).read_text(encoding="utf-8").splitlines()]
        mongo = [m for m in metas if m["src"] != "replay"]
        bal[tier] = {m["id"] for m in mongo}
        new = sum(1 for m in mongo if m["src"] in ("llm", "grammar")) / len(mongo)
        assert abs(new - comp["new_share"]) < 0.04
        blind = sum(1 for m in mongo if m["ctx"] in ("none", "irrelevant")) / len(mongo)
        assert blind >= 0.25
        cut = Counter(m["kind"] for m in mongo)
        assert len(cut) == 9 and max(cut.values()) / len(mongo) < 0.25
        pool = man["selection"]["rebalance"]["pool_marginals"]["cut_kind"]
        assert set(pool) == set(cut) and man["selection"]["rebalance"]["max_dev_pp_by_iteration"]
        rep_tokens = sum(s["tokens"] for s in man["shards"])
        assert abs(man["replay"]["token_share_achieved"] - 0.12) < 0.03 and rep_tokens > 0
        assert man["schedule"]["total_steps"] == int(man["schedule"]["steps_per_epoch"] * man["schedule"]["train_epochs"])
    assert bal["t0"] <= bal["t1"]


@needs_data
def test_verify_detects_tampering(mini, tmp_path):
    import shutil
    cfg = v3.merge_cfg(CFG)
    shutil.copytree(mini["a"] / "t0", tmp_path / "t0")
    man = json.loads((tmp_path / "t0" / "manifest.json").read_text(encoding="utf-8"))
    meta = tmp_path / "t0" / man["shards"][0]["meta_file"]
    lines = meta.read_text(encoding="utf-8").splitlines()
    m0 = json.loads(lines[0])
    m0.pop("source")
    lines[0] = json.dumps(m0)
    m1 = json.loads(lines[1])
    m1["schema_id"] = "telecom/v3_x"
    m1["domain"] = "telecom"
    m1["src"] = "llm"
    lines[1] = json.dumps(m1)
    meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ver = v3.verify_v3(tmp_path, "t0", cfg)
    assert ver["n_issues"] >= 2
    assert any("proveniencia" in i for i in ver["issues"]) and any("hom" in i for i in ver["issues"])
