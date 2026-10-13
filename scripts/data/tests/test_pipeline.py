"""Testes do pipeline de validacao (validate_pipeline.py) e da decontaminacao (decontaminate.py)."""
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import blind_guard as bg  # noqa: E402
import chat_validate as cv  # noqa: E402
import decontaminate as dc  # noqa: E402
import gen_snippets as gs  # noqa: E402
import render_contract as rc  # noqa: E402
import validate_pipeline as vp  # noqa: E402

ROOT = gs.ROOT
GEN = ROOT / "data" / "generated" / "v1"


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def cfg():
    return gs.load_config()


@pytest.fixture(scope="module")
def schemas(cfg):
    return gs.load_schemas(ROOT / cfg["schemas_dir"])


@pytest.fixture(scope="module")
def snips(cfg, schemas):
    return [s for s in gs.generate(cfg, schemas, 60) if s["kind"] == "read" and s["op"] == "find"]


@pytest.fixture(scope="module")
def ctx(cfg, schemas):
    c = vp.Ctx(cfg, schemas, {}, exec_mode="recorded", tok_mode="off", node=cv.node_available())
    yield c
    c.close()


def mk_fim(snip, prefix, middle, suffix, **kw):
    r = {"id": "fim-test-" + hashlib.sha1((prefix + middle + suffix).encode()).hexdigest()[:8], "type": "fim", "kind": snip["op"],
         "task": "autocomplete", "dialect": "console", "prefix": prefix, "middle": middle, "suffix": suffix,
         "domain": snip["domain"], "schema_id": snip["schema_id"], "collection": snip["collection"], "cut_kind": "key",
         "group_id": snip["group_id"], "split": "train", "holdout": False, "snippet_id": None, "editor": None, "exec": "ok",
         "read_only": True}
    r.update(kw)
    r["prompt"] = rc.render_prompt(r)["prompt"]
    return r


def run_stages(r, ctx):
    e = vp.stages_1_to_7(r, "fim", ctx)
    return e


def string_field(schemas, snip):
    sch = next(s for s in schemas if s.full_id == snip["schema_id"])
    coll = next(c for c in sch.colls if c.name == snip["collection"])
    for p, f in coll.fields.items():
        if f.t == "string" and not f.via_array and "." not in p:
            return p
    pytest.skip("sem campo string")


# --------------------------------------------------------------------------- etapa 1
@pytest.mark.parametrize("text,frag", [
    ('db.a.find({ uri: "mongodb://user:pw@host/db" })', "URI"),
    ("db.a.find({}) // <|fim_prefix|>", "<|"),
    ("```js\ndb.a.find({})\n```", "```"),
    ('db.a.find({ email: "joao@gmail.com" })', "e-mail"),
    ('db.a.find({ cpf: "123.456.789-09" })', "CPF"),
    ('db.a.find({ c: "4111111111111111" })', "cartao"),
    ("token sk-abcdefghijklmnopqrstuvwxyz0123", "token"),
])
def test_sanitize_rejects(text, frag):
    why = vp.sanitize_text(text)
    assert why and frag.lower() in why.lower(), why


def test_sanitize_accepts_clean():
    assert vp.sanitize_text('db.users.find({ email: "ana.silva1@example.com", n: 4111111111111112 })') is None
    assert vp.sanitize_text("db.orders.find({ _id: ObjectId('507f1f77bcf86cd799439011') })") is None


# --------------------------------------------------------------------------- etapas 2-7 (cada uma rejeita o caso certo)
def test_valid_record_passes_all_stages(snips, ctx):
    s = snips[0]
    t = s["text"]
    e = run_stages(mk_fim(s, t[:10], t[10:14], t[14:], snippet_id=None), ctx)
    assert e.failed is None, e.failed
    assert [e.val[k]["status"] for k in ("sanitize", "syntax", "api", "fields", "exec", "semantic", "budget")][:4] == ["ok"] * 4


def test_stage_sanitize(snips, ctx):
    s = snips[0]
    e = run_stages(mk_fim(s, "// mongodb://u:p@h/x\n" + s["text"][:5], s["text"][5:9], s["text"][9:]), ctx)
    assert e.failed and e.failed[0] == "sanitize"


@pytest.mark.skipif(not cv.node_available(), reason="node ausente")
def test_stage_syntax(snips, ctx):
    s = snips[0]
    e = run_stages(mk_fim(s, "db." + s["collection"] + ".find({ a: ", "}}(", ")"), ctx)
    assert e.failed and e.failed[0] == "syntax"


def test_stage_api(snips, ctx):
    s = snips[0]
    c = s["collection"]
    e = run_stages(mk_fim(s, "db." + c + ".find({}).", "pretty", "()"), ctx)
    assert e.failed and e.failed[0] == "api", e.failed
    e = run_stages(mk_fim(s, "db." + c + ".find({ $contains: ", "1", " })"), ctx)
    assert e.failed and e.failed[0] == "api", e.failed


def test_stage_fields(snips, ctx):
    s = snips[0]
    e = run_stages(mk_fim(s, "db." + s["collection"] + ".find({ campo_inexistente_xyz: ", "1", " })"), ctx)
    assert e.failed and e.failed[0] == "fields", e.failed


def test_stage_exec(snips, ctx):
    s = snips[0]
    t = s["text"]
    e = run_stages(mk_fim(s, t[:10], t[10:14], t[14:], exec="error"), ctx)
    assert e.failed and e.failed[0] == "exec"


def test_stage_semantic_no_context(snips, ctx):
    s = snips[0]
    e = run_stages(mk_fim(s, "", s["text"], ""), ctx)
    assert e.failed and e.failed[0] == "semantic"


def test_stage_semantic_plan_mismatch(snips, ctx):
    a, b = snips[0], snips[1]
    # documento valido para a colecao de `a`, mas o plano de referencia e de outro snippet
    ctx2 = vp.Ctx(ctx.cfg, list(ctx.by_schema.values()), {"snip-x": {**b, "id": "snip-x", "text": a["text"]}},
                  exec_mode="recorded", tok_mode="off", node=False)
    e = vp.stages_1_to_7(mk_fim(a, a["text"][:10], a["text"][10:14], a["text"][14:], snippet_id="snip-x", schema_id=a["schema_id"]),
                         "fim", ctx2)
    assert e.failed and e.failed[0] == "semantic", e.failed


def test_stage_budget(snips, schemas, ctx):
    s = snips[0]
    f = string_field(schemas, s)
    big = ", ".join('"valor-numero-%03d-longo"' % i for i in range(40))
    e = run_stages(mk_fim(s, "db." + s["collection"] + ".find({ " + f + ": { $in: [", big, "] } })"), ctx)
    assert e.failed and e.failed[0] == "budget", e.failed


# --------------------------------------------------------------------------- etapa 8
def lite_fim(i, prefix, middle, suffix, snippet, split="train"):
    return {"id": f"f{i}", "split": split, "snippet_id": snippet, "collection": "c", "prefix": prefix, "middle": middle,
            "suffix": suffix, "type": "fim"}


def ent(r):
    class R(dict):
        pass
    return vp.Entry({**r, "group_id": "g"}, "fim")


def test_dedup_exact_and_near():
    base = " ".join(f"{{ campo{j}: {j} }}," for j in range(60))
    docs = {
        "f1": lite_fim(1, "db.c.find(", "{ a: 1 }", ")", "s1"),
        "f2": lite_fim(2, "db.c.find(", "{ a:   1 }", ")", "s2"),                      # so espacos: exato
        "f3": lite_fim(3, "db.c.aggregate([" + base, "{ z: 1 }", "])", "s3"),
        "f4": lite_fim(4, "db.c.aggregate([" + base.replace("campo30: 30", "campo30: 31"), "{ z: 1 }", "])", "s4"),  # quase igual
        "f5": lite_fim(5, "db.c.aggregate([" + base, "{ z: 1 }", "])", "s3"),         # outro corte? mesmo doc e mesmo texto => exato
        "f6": lite_fim(6, "db.c.aggregate([", "{ y: 2 }", "])", "s6"),
    }
    ents = [vp.Entry({**d, "group_id": "g", "kind": "find"}, "fim") for d in docs.values()]
    for e in ents:
        e.val["sanitize"] = {"status": "ok"}
    st = vp.stage_dedup(ents, docs, 20261008, 0.85)
    failed = {e.id: e.failed[1] for e in ents if e.failed}
    assert "f2" in failed and "exata" in failed["f2"]
    assert "f4" in failed and "near-dup" in failed["f4"]
    assert "f5" in failed
    assert "f1" not in failed and "f3" not in failed and "f6" not in failed
    assert st["exact"] >= 2 and st["near_dup"] >= 1


def test_dedup_keeps_distinct_cuts_of_same_doc():
    a = lite_fim(1, "db.c.find({ a: ", "1", " })", "s1")
    b = lite_fim(2, "db.c.find({ ", "a: 1", " })", "s1")
    ents = [vp.Entry({**d, "group_id": "g", "kind": "find"}, "fim") for d in (a, b)]
    vp.stage_dedup(ents, {"f1": a, "f2": b}, 1, 0.85)
    assert all(e.failed is None for e in ents)


# --------------------------------------------------------------------------- etapa 9 (decontaminacao)
SHARED = ("db.pedidos.aggregate([ { $match: { status_pedido: { $in: ['x'] } } }, { $group: { _id: '$cliente_nome', total_geral: "
          "{ $sum: '$valor_total' } } }, { $sort: { total_geral: -1 } } ])")


def rec(i, text, split, **kw):
    return {"id": f"r{i}", "split": split, "snippet_id": f"s{i}", "collection": "pedidos", "prefix": text[:20], "middle": text[20:30],
            "suffix": text[30:], "type": "fim", **kw}


def test_decontam_removes_shared_13gram(tmp_path):
    # train e teste so compartilham um 13-grama (literais diferentes, restante diferente)
    ev = rec(1, "db.a.find({}) ; " + SHARED + " ; fim A", "test")
    tr = rec(2, "db.b.count() ; " + SHARED.replace("'x'", "'y'") + " ; outro fim", "train")
    other = rec(3, "db.zzz.find({ k_unico: 1 }).limit(3)", "train")
    res = dc.run([tr, other], [("test", ev)], seed=1, blind=tmp_path)
    assert tr["id"] in res["removed"] and "ngram13" in res["removed"][tr["id"]]
    assert other["id"] not in res["removed"]
    assert res["blind"] == "not_sealed"
    assert tr["id"] in sum(res["examples"].values(), [])


def test_decontam_approx_key_and_minhash(tmp_path):
    ev = {"id": "e1", "split": "val", "snippet_id": "e1", "collection": "pedidos", "type": "fim", "prefix": "db.pedidos.find({ a: 1, b: 'x' }).sort({ c: 1 })",
          "middle": "", "suffix": ""}
    tr = {**ev, "id": "t1", "split": "train", "snippet_id": "t1", "prefix": "db.pedidos.find({ a: 99, b: 'zz' }).sort({ c: -1 })"}
    res = dc.run([tr], [("val", ev)], seed=1, blind=tmp_path)
    assert "approx_key" in res["removed"]["t1"]


def test_decontam_minhash_threshold(tmp_path):
    base = " ".join(f"campo_{j} : {{ $gte : {j} }} ," for j in range(40))
    ev = {"id": "e1", "split": "bench", "snippet_id": "e1", "type": "fim", "prefix": "db.x.find({" + base, "middle": "", "suffix": "})"}
    near = {"id": "t1", "split": "train", "snippet_id": "t1", "type": "fim", "prefix": "db.y.find({" + base.replace("campo_5 ", "campo_five "),
            "middle": "", "suffix": "})"}
    far = {"id": "t2", "split": "train", "snippet_id": "t2", "type": "fim", "prefix": "db.q.find({ nada: 1 })", "middle": "", "suffix": ""}
    res = dc.run([near, far], [("bench", ev)], seed=1, blind=tmp_path)
    assert "t1" in res["removed"] and "t2" not in res["removed"]


def test_decontam_blind_hash_via_guard(tmp_path):
    prefix, suffix = "db.segredo.find({ k: ", " })"
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "manifest.json").write_text(json.dumps({"prompt_sha256": [bg.prompt_sha256(prefix + "\x00" + suffix)]}), encoding="utf-8")
    tr = {"id": "t1", "split": "train", "snippet_id": "t1", "type": "fim", "prefix": prefix, "middle": "1", "suffix": suffix}
    res = dc.run([tr], [], seed=1, blind=tmp_path)
    assert res["blind"] == "sealed" and res["removed"].get("t1") == ["blind"]


def test_decontam_never_reads_blind_dir():
    src = (HERE.parent / "decontaminate.py").read_text(encoding="utf-8")
    assert "data/eval/blind" not in src.replace("`data/eval/blind`", "") or "blind_guard" in src
    assert "open(" not in src.split("def blind_status")[1].split("def build_index")[0]


def test_decontam_deterministic():
    ev = [("test", rec(1, "db.a.find({}) ; " + SHARED, "test"))]
    tr = [rec(i, "x" * i + SHARED + "y" * i, "train") for i in range(2, 8)]
    a = dc.run(tr, ev, seed=5)
    b = dc.run(tr, ev, seed=5)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------- pipeline completo (amostra)
needs_data = pytest.mark.skipif(not (GEN / "fim_ctx.jsonl").exists() or not (GEN / "chat.jsonl").exists(),
                                reason="data/generated/v1 ausente")


def run_small(tmp, limit=300, **kw):
    args = Namespace(config=str(gs.CONFIG_PATH), ver="v1", fim=None, chat=None, snippets=None, include_snippets=False,
                     out_dir=str(tmp / "out"), splits_dir=str(tmp / "splits"), report=str(tmp / "rep.md"),
                     decontam_dir=str(tmp / "dc"), exec="recorded", tokenizer="off", no_node=True, legacy=[], dedup_jaccard=0.85,
                     ngram_min_content=0, limit=limit)
    for k, v in kw.items():
        setattr(args, k, v)
    return vp.run_pipeline(args)


@needs_data
def test_pipeline_deterministic_and_no_leak(tmp_path):
    a = run_small(tmp_path / "a")
    b = run_small(tmp_path / "b")

    def sums(base):
        out = {}
        for rel in ("out/fim.jsonl", "out/chat.jsonl", "out/rejected.jsonl", "splits/train.ids.txt", "splits/val.ids.txt",
                    "splits/test.ids.txt", "splits/bench.ids.txt"):
            out[rel] = hashlib.sha256((base / rel).read_bytes()).hexdigest()
        return out
    assert sums(tmp_path / "a") == sums(tmp_path / "b")
    # manifesto sem timestamp igual
    ma, mb = a["manifest"], b["manifest"]
    for k in ("counts", "rejections", "decontam", "seed", "config"):
        assert ma[k] == mb[k]
    # nenhum vazamento: ids disjuntos e 2a passada de decontaminacao vazia
    ids = {s: set((tmp_path / "a" / "splits" / f"{s}.ids.txt").read_text().split()) for s in ("train", "val", "test", "bench")}
    assert ids["train"] and ids["test"]
    for x in ids:
        for y in ids:
            if x < y:
                assert not (ids[x] & ids[y]), (x, y)
    recs = [json.loads(l) for p in ("fim", "chat") for l in open(tmp_path / "a" / "out" / f"{p}.jsonl", encoding="utf-8")]
    groups: dict = {}
    for r in recs:
        groups.setdefault(r.get("group_id"), set()).add(r["split"])
        assert set(r["validation"]) == set(vp.STAGES)
        assert all(v["status"] in ("ok", "skipped") for v in r["validation"].values())
        assert not r.get("holdout") or r["split"] in ("test", "bench")
    assert all(len(v) == 1 for v in groups.values())
    train = [vp.lite_record(r, "fim" if "prefix" in r else "chat") for r in recs if r["split"] == "train"]
    evals = [(r["split"], vp.lite_record(r, "fim" if "prefix" in r else "chat")) for r in recs if r["split"] != "train"]
    assert dc.run(train, evals, seed=ma["seed"])["train_removed"] == 0


@needs_data
def test_pipeline_report_and_manifest(tmp_path):
    out = run_small(tmp_path, limit=120)
    mf = out["manifest"]
    assert mf["seed"] and len(mf["generator_commit"]) >= 7 and len(mf["config"]["sha256"]) == 64
    assert (tmp_path / "rep.md").exists() and (tmp_path / "out" / "manifest.json").exists()
    assert {"train", "val", "test", "bench"} <= set(mf["counts"])
    dcs = list((tmp_path / "dc").glob("*.json"))
    assert len(dcs) == 1
    rep = json.loads(dcs[0].read_text(encoding="utf-8"))
    assert rep["blind"] in ("not_sealed", "sealed") and set(rep["by_rule"]) == set(dc.RULES)
    assert all(len(v) <= 20 for v in rep["examples"].values())
