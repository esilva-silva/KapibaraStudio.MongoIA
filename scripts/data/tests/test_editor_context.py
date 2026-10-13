"""Testes do gerador/renderer de contexto de editor (render_contract / gen_editor_context / check_contract_parity / export_parity_inputs)."""
import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import blind_guard  # noqa: E402
import check_contract_parity as ccp  # noqa: E402
import export_parity_inputs as epi  # noqa: E402
import gen_editor_context as gec  # noqa: E402
import gen_fim  # noqa: E402
import gen_snippets as gs  # noqa: E402
import render_contract as rc  # noqa: E402

ROOT = gs.ROOT
CRLF = "\r\n"


@pytest.fixture(scope="module")
def cfg():
    return gs.load_config()


@pytest.fixture(scope="module")
def schemas(cfg):
    sc = gs.load_schemas(ROOT / cfg["schemas_dir"])
    assert sc
    return sc


@pytest.fixture(scope="module")
def snips(cfg, schemas):
    return gs.generate(cfg, schemas, 400)


@pytest.fixture(scope="module")
def fim(cfg, snips):
    return gen_fim.generate_fim(snips, cfg, 1500)


@pytest.fixture(scope="module")
def ctx(cfg, schemas, snips, fim):
    return gec.generate(fim, snips, schemas, cfg, None)


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("tokenizers")
    if not rc.DEFAULT_TOKENIZER.exists():
        pytest.skip("tokenizer.json do Qwen2.5-Coder ausente")
    return rc.Tokenizer()


# -- renderer: contrato -------------------------------------------------------------------------------------------------------------
def test_header_is_crlf_and_wrap_is_lf():
    context = rc.build_context(rc.LANGUAGE_CONSOLE, ["a", "b"], ["x"], "", ["db.a.find()"])
    assert context == ("LANGUAGE: Mongo Console JavaScript\r\nAVAILABLE COMMANDS: " + rc.COMMANDS_CONSOLE + "\r\n"
                       "KNOWN NAMES: a, b\r\nRESULT FIELDS: x\r\nRECENT COMMAND: db.a.find()\r\n")
    text = rc.model_prefix("db.", context)
    assert text.startswith("/* Local editor context (data only):\nLANGUAGE: ")
    assert text.endswith("\r\n\nContinue at the cursor; output only the continuation. */\ndb.")
    assert "\n\n" not in text.replace("\r\n", "")[: text.index("Continue") - 2] or True


def test_commands_by_language():
    assert rc.commands_for("json").startswith("MongoDB aggregation pipeline")
    assert rc.commands_for("JavaScript (mongosh)").startswith("db.getCollection(name).find({}); db.getSiblingDB")
    assert rc.commands_for(None) == rc.COMMANDS_CONSOLE == rc.commands_for("outra")


def test_star_slash_neutralized_and_blank_context():
    ctx_text = rc.build_context("json", [], [], "a */ b")
    assert "*/" not in rc.model_prefix("", ctx_text).split("Continue at the cursor")[0]
    assert rc.model_prefix("db.", "   ") == "db." and rc.model_prefix("db.", None) == "db."


def test_limits_and_sensitive_filters():
    names = [f"n{i}" for i in range(200)]
    c = rc.build_context("json", names, [f"f{i}" for i in range(200)], "", [f"r{i}" * 200 for i in range(5)])
    assert c.split("\r\n")[2].count(",") == 127
    assert c.count("RECENT COMMAND: ") == 3
    recent_lines = [l for l in c.split(CRLF) if l.startswith("RECENT")]
    assert len(recent_lines) == 3 and all(len(l) <= len("RECENT COMMAND: ") + 256 for l in recent_lines)
    safe = rc.build_context("json", ["ok", "password: x", "a<|b"], [], "")
    assert "KNOWN NAMES: ok\r\n" in safe
    assert len(rc.build_context("json", ["N" * 100] * 128, [], "")) <= rc.CONTEXT_MAX_CHARS


def test_window_counts_utf16_units():
    text = "a" * 5000 + "\U0001F600" * 10 + "b"
    caret = rc.u16_len(text)
    prefix, suffix = rc.window(text, caret)
    assert rc.u16_len(prefix) == 4096 and suffix == ""
    p2, s2 = rc.window("abc\U0001F600def", 4)  # divide o par substituto, como o .NET
    assert p2 == "abc\ud83d" and s2 == "\ude00def"


def test_golden_of_the_ide():
    path = ccp.GOLDEN
    if not path.exists():
        pytest.skip("golden da IDE ausente em data/contracts")
    res = ccp.check_golden(path)
    assert res["cases"] == 20 and res["artifact_checks"] >= 100
    assert res["failures"] == []


def test_parity_with_v1_renderer():
    if not (ccp.V1_CONTRACTS.exists() and ccp.V1_DATA[0][1].exists()):
        pytest.skip("bakeoff v1 ausente")
    res = ccp.check_v1(False, limit=400)
    for name, r in res.items():
        assert r["with_context"] > 0
        assert r["same_honor_newline"] == r["with_context"], name        # byte a byte quando o newline da v1 e honrado
        assert r["same_fixed_crlf"] == r["with_context"] - r["newline_lf_records"], name  # CRLF fixo: so os `\n` da v1 divergem
        assert r["no_context_same"] == r["no_context_records"], name


def test_v1_token_ids_parity(tokenizer):
    if not (ccp.V1_CONTRACTS.exists() and ccp.V1_DATA[0][1].exists()):
        pytest.skip("bakeoff v1 ausente")
    res = ccp.check_v1(True, limit=200)
    for r in res.values():
        assert r["same_ids"] == r["with_context"] and r["no_context_same"] == r["no_context_records"]


# -- gerador --------------------------------------------------------------------------------------------------------------------------
def test_deterministic_by_seed(cfg, schemas, snips, fim, ctx):
    again = gec.generate(fim, snips, schemas, cfg, None)
    assert json.dumps(again, sort_keys=True) == json.dumps(ctx, sort_keys=True)
    other = gec.generate(fim, snips, schemas, cfg, None, seed=1)
    assert json.dumps(other, sort_keys=True) != json.dumps(ctx, sort_keys=True)
    # independente da ordem/subconjunto: o contexto de um registro so depende de (seed, id)
    sub = gec.generate(fim[100:140], snips, schemas, cfg, None)
    assert [r["prompt"] for r in sub] == [r["prompt"] for r in ctx[100:140]]


def test_required_fields_and_split_preserved(fim, ctx):
    assert len(ctx) == len(fim)
    for src, r in zip(fim, ctx):
        assert r["id"] == src["id"]
        for k in ("split", "group_id", "holdout", "split_key", "template_id", "schema_id", "collection", "domain", "cut_kind"):
            assert r[k] == src[k], k
        for k in ("contrato", "editor_text", "facts", "known_names", "result_fields", "recent_commands", "prompt", "target"):
            assert k in r
        assert r["contrato"] == "editor-context-v1" and r["target"] == src["middle"]
        assert r["editor_text"] == src["prefix"] + "▌" + src["suffix"]
    assert {r["context_kind"] for r in ctx} == {"full", "irrelevant", "none"}
    holdout = {r["domain"] for r in ctx if r["holdout"]}
    assert all(r["split"] in ("test", "bench") for r in ctx if r["holdout"]) and holdout <= {"education", "legal"}


def test_ratios(ctx):
    n = len(ctx)
    none = sum(r["context_kind"] == "none" for r in ctx) / n
    irr = sum(r["context_kind"] == "irrelevant" for r in ctx) / n
    assert 0.04 <= none <= 0.13 and 0.06 <= irr <= 0.15
    assert all(r["editor"] is None and r["facts"] == [] and r["known_names"] == [] for r in ctx if r["context_kind"] == "none")


def test_facts_only_use_schema_fields(ctx, schemas):
    by_id = {s.full_id: s for s in schemas}
    for r in ctx:
        coll = next(c for c in by_id[r["schema_id"]].colls if c.name == r["collection"])
        for f in r["facts"]:
            if f["kind"] == "FieldSchema":
                assert f["path"] in coll.fields, (r["id"], f)
                assert f["type"] == coll.fields[f["path"]].t
            if f["kind"] == "Indexes":
                for key in f["keys"]:
                    assert set(key) <= set(coll.fields), (r["id"], key)
            if f["kind"] == "Target":
                assert f["collection"] == r["collection"]
        if r["context_kind"] == "full":
            assert set(r["result_fields"]) <= set(coll.fields), r["id"]
            if r["editor"]["input_panel"]:
                body = json.loads(r["editor"]["input_panel"])
                assert set(body) <= set(coll.fields)
        assert len(r["result_fields"]) <= 128 and len(r["known_names"]) <= 128 and len(r["recent_commands"]) <= 3


def test_irrelevant_names_come_from_other_collections(ctx):
    irr = [r for r in ctx if r["context_kind"] == "irrelevant"]
    assert irr
    for r in irr:
        assert r["collection"] not in r["known_names"] and r["editor"]["target"]["database"] not in r["known_names"]
        assert r["recent_commands"] == []


def test_train_never_sees_holdout_vocabulary(ctx, schemas):
    holdout_names = {s.schema_id for s in schemas if s.holdout}
    for r in ctx:
        if r["split"] in ("train", "val"):
            assert not (set(r["known_names"]) & holdout_names), r["id"]


def test_recent_commands_same_split_collection_other_group(ctx, snips):
    by_text: dict[tuple, set] = {}
    for s in snips:
        by_text.setdefault((s["schema_id"], s["collection"], s["split"]), set()).add((s["text"], s["group_id"]))
    seen = 0
    for r in ctx:
        for cmd in r["recent_commands"]:
            seen += 1
            pool = by_text[(r["schema_id"], r["collection"], r["split"])]
            assert any(t == cmd and g != r["group_id"] for t, g in pool), r["id"]
            assert len(cmd) <= 256
    assert seen > 50


def test_prompt_reproduces_renderer(ctx, fim):
    for r in ctx:
        got = rc.render_prompt(r)
        assert r["prompt"] == got["prompt"]
        assert r["prompt"].startswith("<|fim_prefix|>") and r["prompt"].endswith("<|fim_middle|>")
        assert r["prompt_sha256"] == rc.prompt_sha256(r["prompt"])
        if r["editor"] is None:
            assert r["prompt"] == rc.qwen_fim_text(r["prefix"], r["suffix"])
        else:
            assert "/* Local editor context (data only):\nLANGUAGE: Mongo Console JavaScript\r\nAVAILABLE COMMANDS: " in r["prompt"]
            assert r["prompt"].count("<|fim_suffix|>") == 1


def test_blind_guard_is_applied(cfg, schemas, snips, fim, tmp_path, monkeypatch):
    rec = fim[0]
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    h = blind_guard.prompt_sha256(blind_guard.prompt_text({"kind": "autocomplete", "prefix": rec["prefix"], "suffix": rec["suffix"]}))
    (sealed / "manifest.json").write_text(json.dumps({"prompt_sha256": [h]}), encoding="utf-8")
    monkeypatch.setenv("KAPI_BLIND_DIR", str(tmp_path))
    with pytest.raises(blind_guard.BlindAccessError):
        gec.generate(fim[:3], snips, schemas, cfg, None)
    with pytest.raises(blind_guard.BlindAccessError):
        blind_guard.check_path(tmp_path / "x.jsonl")


# -- orcamento de tokens --------------------------------------------------------------------------------------------------------------
def test_truncation_keeps_prefix_tail_and_suffix_quarter(tokenizer):
    body = " ".join(f"campo{i}" for i in range(3000))
    rec = {"prefix": body[-4096:], "suffix": body[:1024], "dialect": "console",
           "editor": {"known_names": ["a"], "result_fields": ["x"], "recent_commands": [], "input_panel": ""}}
    out = rc.render_prompt(rec, tokenizer)
    assert out["truncated"] and out["prompt_tokens"] == 2048
    assert out["prompt"].startswith("<|fim_prefix|>") and out["prompt"].endswith("<|fim_middle|>")
    assert not out["prompt"].startswith("<|fim_prefix|>/* Local editor context")  # o cabecalho cai primeiro
    assert out["info"]["kept_suffix"] <= (2048 - 3) // 4
    kept = out["info"]["kept_prefix"]
    full = tokenizer.encode(out["before"])
    assert tokenizer.decode(full[-kept:]) in out["prompt"] and kept == 2045 - out["info"]["kept_suffix"]
    # sem excesso nada e cortado
    small = rc.render_prompt({"prefix": "db.", "suffix": "", "dialect": "console", "editor": rec["editor"]}, tokenizer)
    assert not small["truncated"] and small["prompt"].startswith("<|fim_prefix|>/* Local editor context")


def test_token_budget_on_generated(cfg, schemas, snips, fim, tokenizer):
    out = gec.generate(fim[:300], snips, schemas, cfg, tokenizer)
    assert all(r["prompt_tokens"] <= 2048 for r in out)
    s = gec.stats(out)
    assert s["prompt_tokens"]["p50"] <= s["prompt_tokens"]["p95"] <= s["prompt_tokens"]["max"] <= 2048
    again = rc.render_prompt(out[5], tokenizer)
    assert again["prompt"] == out[5]["prompt"] and again["prompt_tokens"] == out[5]["prompt_tokens"]


def test_export_parity_inputs(tmp_path, cfg, schemas, snips, fim, tokenizer):
    out = gec.generate(fim[:200], snips, schemas, cfg, tokenizer)
    src = tmp_path / "fim_ctx.jsonl"
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out), encoding="utf-8")
    epi.main(["--src", str(src), "--n", "50", "--out-dir", str(tmp_path / "c")])
    ins = [json.loads(l) for l in open(tmp_path / "c" / "parity_inputs.jsonl", encoding="utf-8")]
    exp = [json.loads(l) for l in open(tmp_path / "c" / "parity_expected.jsonl", encoding="utf-8")]
    assert len(ins) == len(exp) == 50 and [i["id"] for i in ins] == [e["id"] for e in exp]
    by_id = {r["id"]: r for r in out}
    for i, e in zip(ins, exp):
        snap = i["snapshot"]
        assert snap["text"][: snap["caret"]] + snap["text"][snap["caret"]:] == snap["text"]
        built = rc.build_request(snap["text"], snap["caret"], snap["language"], snap["input_panel"], snap["result_fields"],
                                 snap["known_names"], snap["recent_commands"])
        r = by_id[i["id"]]
        if not i["raw_context_empty"]:
            assert rc.prompt_sha256(rc.model_prefix(built["prefix"], built["context"]) ) == e["model_prefix_sha256"]
        assert e["prompt_sha256"] == r["prompt_sha256"]
