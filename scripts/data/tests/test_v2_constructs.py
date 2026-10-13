"""Ciclo 3: construcoes novas (getConnection/getSiblingDB/ENV.get/console.log/cursor em variavel, createIndex/dropIndex,
$search/$searchMeta/$vectorSearch, dialeto json) aceitas pelo pipeline SEM relaxar a seguranca; multi-turno; shards; MinHash."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import chat_validate as cv  # noqa: E402
import decontaminate as dc  # noqa: E402
import gen_fim  # noqa: E402
import gen_snippets as gs  # noqa: E402
import jsonl_shards as js  # noqa: E402
import render_contract as rc  # noqa: E402
import validate_pipeline as vp  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return gs.load_config()


@pytest.fixture(scope="module")
def schemas(cfg):
    return gs.load_schemas(gs.ROOT / cfg["schemas_dir"])


@pytest.fixture(scope="module")
def snips(cfg, schemas):
    return gs.generate(cfg, schemas, 1500)


def by_cat(snips, cat, **kw):
    out = [s for s in snips if s.get("category") == cat and all(s.get(k) == v for k, v in kw.items())]
    assert out, f"sem snippet da categoria {cat}"
    return out


def sch_coll(schemas, s):
    sch = next(x for x in schemas if x.full_id == s["schema_id"])
    return sch, next(c for c in sch.colls if c.name == s["collection"])


class Boom:
    """Executor que falha se o pipeline tentar executar algo (prova de que skipped_* nunca chega ao driver)."""
    violations: list = []

    def run(self, *a, **k):
        raise AssertionError("o executor foi chamado para um documento skipped_*")


def stage_ctx(cfg, schemas, snips_dict=None):
    c = vp.Ctx(cfg, schemas, snips_dict or {}, exec_mode="recorded", tok_mode="off", node=cv.node_available())
    c.exec_mode = "live"          # forca o ramo live; o executor falso explode se for usado
    c.executor = Boom()
    return c


# --------------------------------------------------------------------------- parser do Console
@pytest.mark.parametrize("text,root,database,conn", [
    ("getConnection('Prod Cluster').getDatabase('shop').getCollection('orders').find({ a: 1 })", "conn", "shop", "Prod Cluster"),
    ("getConnection(ENV.get('DEFAULT_CONNECTION')).getDatabase('shop').getCollection('orders').find({}).limit(3).toArray()",
     "conn", "shop", "ENV:DEFAULT_CONNECTION"),
    ("db.getSiblingDB('shop').orders.countDocuments({ a: { $gt: 1 } })", "sibling", "shop", None),
    ("db.getSiblingDB('shop').getCollection('orders').find({})", "sibling", "shop", None),
    ("ConnectionPool['Prod Cluster']['shop']['orders'].find({})", "pool", "shop", "Prod Cluster"),
    ("console.log(db.orders.find({ a: 1 }).sort({ a: 1 }).toArray());", "dot", None, None),
    ("console.log(getConnection('local').getDatabase('shop').getCollection('orders').aggregate([{ $match: { a: 1 } }]))",
     "conn", "shop", "local"),
])
def test_parser_accepts_new_console_constructs(text, root, database, conn):
    plan, meta = cv.parse_console(text)
    assert meta["collection"] == "orders" and meta["root_form"] == root
    assert meta.get("database") == database and meta.get("connection") == conn
    assert plan["op"] in ("find", "aggregate", "countDocuments")


def test_parser_cursor_in_variable():
    plan, meta = cv.parse_console("const cursor = db.orders.find({ a: 1 });\ncursor.sort({ b: -1 }).limit(5).toArray();")
    assert plan == {"op": "find", "filter": {"a": 1}, "sort": {"b": -1}, "limit": 5} and meta["collection"] == "orders"
    plan, _ = cv.parse_console("const conn = getConnection('x');\nconst cur = db.orders.find({});\ncur")
    assert plan["op"] == "find"
    # variavel de cursor + metodo de escrita nao passa
    with pytest.raises(cv.ChatParseError):
        cv.parse_console("const cur = db.orders.find({});\ncur.drop()")


@pytest.mark.parametrize("text", [
    "console.error(db.orders.find({}))",                  # so console.log
    "console.log(1)",                                     # sem consulta
    "ENV.get('X')",                                       # ENV nao e consulta
    "getConnection('a').getDatabase('b').dropDatabase()",  # metodo fora do subconjunto
    "db.getSiblingDB('b').orders.find({}).hint({ a: 1 })",  # .hint() fora do cursor do Console (spec: sort/skip/limit/project/toArray)
    "getConnection(1).getDatabase('b').getCollection('c').find({})",
    "db.orders.find({}).forEach(printjson)",
])
def test_parser_still_rejects(text):
    with pytest.raises(cv.ChatParseError):
        cv.parse_console(text)


def test_hint_stays_out_of_cursor_methods():
    assert "hint" not in cv.CURSOR_METHODS
    with pytest.raises(cv.ChatParseError) as e:
        cv.parse_console("db.orders.find({}).hint({ a: 1 })")
    assert e.value.stage == "api"


def test_chat_parser_never_accepts_writes_or_atlas():
    for t in ("db.orders.createIndex({ a: 1 })", "db.orders.dropIndex('a_1')", "db.orders.updateOne({}, { $set: { a: 1 } })"):
        with pytest.raises(cv.ChatParseError):
            cv.parse_console(t)
    plan = {"op": "aggregate", "pipeline": [{"$search": {"text": {"query": "x", "path": "a"}}}]}
    assert cv.check_ops(plan) is not None              # chat: $search continua fora da lista
    assert cv.check_ops(plan, allow_atlas=True) is None


# --------------------------------------------------------------------------- documentos FIM (analyze_text)
def test_analyze_every_generated_category(snips, schemas, cfg):
    """Todo snippet gerado (todas as categorias) passa nas etapas estaticas com o seu dialeto/exec_mode."""
    node = cv.NodeChecker()
    try:
        seen = set()
        for s in snips:
            sch, coll = sch_coll(schemas, s)
            an = vp.analyze_text(s["text"], sch, coll, node.check(s["text"]), allow_write=True, expect_plan=s["plan"],
                                 dialect=s.get("dialect", "console"))
            for k in ("syntax", "api", "fields", "semantic"):
                assert an[k][0] == "ok", (s["category"], s["text"][:200], k, an[k])
            seen.add(s["category"])
        assert {"atlas_search", "json_ejson", "indexes", "dsl", "aggregation", "crud", "operators"} <= seen
    finally:
        node.close()


def test_dsl_forms_covered(snips):
    forms = {(s.get("extra") or {}).get("dsl_form") for s in by_cat(snips, "dsl")}
    assert {"conn", "sibling"} <= forms
    assert any((s.get("extra") or {}).get("uses_env") for s in by_cat(snips, "dsl"))
    assert any((s.get("extra") or {}).get("cursor_var") for s in by_cat(snips, "dsl"))
    assert any((s.get("extra") or {}).get("console_log") for s in by_cat(snips, "dsl"))


def test_writes_and_atlas_never_reach_executor(cfg, schemas, snips):
    ctx = stage_ctx(cfg, schemas)
    try:
        picks = by_cat(snips, "atlas_search")[:5] + by_cat(snips, "indexes")[:5] + \
            [s for s in snips if s["op"] in ("updateOne", "updateMany", "deleteOne", "deleteMany")][:5]
        assert len(picks) >= 12
        for s in picks:
            t = s["text"]
            fim = {"id": "fim-x-" + s["id"], "type": "fim", "kind": s["op"], "dialect": s.get("dialect", "console"),
                   "prefix": t[:8], "middle": t[8:14], "suffix": t[14:], "domain": s["domain"], "schema_id": s["schema_id"],
                   "collection": s["collection"], "cut_kind": "key", "group_id": s["group_id"], "split": "train", "holdout": False,
                   "snippet_id": s["id"], "editor": None, "exec": s["exec_mode"], "read_only": s["kind"] == "read"}
            fim["prompt"] = rc.render_prompt(fim)["prompt"]
            ctx.snips = {s["id"]: s}
            e = vp.stages_1_to_7(fim, "fim", ctx)
            assert e.failed is None, (s["category"], e.failed, t[:160])
            st = e.val["exec"]
            assert st["status"] == "skipped" and st["reason"].startswith(("skipped_static", "skipped_write")), st
            want = "skipped_static" if s["category"] == "atlas_search" else "skipped_write"
            assert st["reason"].startswith(want), (s["category"], st)
    finally:
        ctx.close()


def test_atlas_without_exec_mode_still_not_executed(cfg, schemas, snips):
    """Defesa em profundidade: sem `exec_mode` no snippet/registro, um estagio $search ainda e pulado (nunca vai ao driver)."""
    ctx = stage_ctx(cfg, schemas)
    try:
        s = by_cat(snips, "atlas_search")[0]
        t = s["text"]
        fim = {"id": "fim-y", "type": "fim", "kind": "aggregate", "dialect": "console", "prefix": t[:8], "middle": t[8:14], "suffix": t[14:],
               "domain": s["domain"], "schema_id": s["schema_id"], "collection": s["collection"], "cut_kind": "key",
               "group_id": s["group_id"], "split": "train", "holdout": False, "snippet_id": None, "editor": None, "exec": "ok"}
        fim["prompt"] = rc.render_prompt(fim)["prompt"]
        e = vp.stages_1_to_7(fim, "fim", ctx)
        assert e.failed is None, e.failed
        assert e.val["exec"]["reason"].startswith("skipped_static")
    finally:
        ctx.close()


def test_check_atlas_rules():
    ok = {"op": "aggregate", "pipeline": [{"$vectorSearch": {"index": "i", "path": "embedding", "queryVector": [0.1, 0.2],
                                                              "numCandidates": 50, "limit": 5}}, {"$limit": 3}]}
    assert cv.check_atlas(ok) is None
    late = {"op": "aggregate", "pipeline": [{"$match": {}}, {"$search": {"text": {"query": "a", "path": "b"}}}]}
    assert cv.check_atlas(late)[0] == "api"
    nov = {"op": "aggregate", "pipeline": [{"$vectorSearch": {"index": "i", "path": "p", "queryVector": [1], "limit": 3}}]}
    assert cv.check_atlas(nov)[0] == "api"          # sem numCandidates/exact
    bad_qv = {"op": "aggregate", "pipeline": [{"$vectorSearch": {"index": "i", "path": "p", "queryVector": ["a"], "numCandidates": 3,
                                                                  "limit": 3}}]}
    assert cv.check_atlas(bad_qv)[0] == "api"


def test_json_dialect(schemas, snips):
    s = by_cat(snips, "json_ejson")[0]
    sch, coll = sch_coll(schemas, s)
    an = vp.analyze_text(s["text"], sch, coll, None, allow_write=True, expect_plan=s["plan"], dialect="json")
    assert all(an[k][0] == "ok" for k in ("syntax", "api", "fields", "semantic")), an
    assert an["plan"]["op"] == "aggregate"
    # JSON invalido / $out / $oid malformado
    bad = vp.analyze_text(s["text"] + " }", sch, coll, None, allow_write=True, dialect="json")
    assert bad["syntax"][0] == "fail"
    out = vp.analyze_text('[{"$out": "x"}]', sch, coll, None, allow_write=True, dialect="json")
    assert out["api"][0] == "fail"
    oid = vp.analyze_text('[{"$match": {"_id": {"$oid": "zz"}}}]', sch, coll, None, allow_write=True, dialect="json")
    assert oid["api"][0] == "fail"
    # o dialeto console nao aceita um array solto
    con = vp.analyze_text(s["text"], sch, coll, None, allow_write=True, dialect="console")
    assert con["syntax"][0] == "fail"


def test_index_plans_fields_checked(schemas, snips):
    s = by_cat(snips, "indexes", op="createIndex")[0]
    sch, coll = sch_coll(schemas, s)
    ok = vp.analyze_text(s["text"], sch, coll, None, allow_write=True, expect_plan=s["plan"])
    assert ok["plan"]["op"] == "createIndex" and ok["fields"][0] == "ok"
    bad = vp.analyze_text(f"db.{coll.name}.createIndex({{ campo_inexistente_zz: 1 }})", sch, coll, None, allow_write=True)
    assert bad["fields"][0] == "fail"
    d = vp.analyze_text(f"db.{coll.name}.dropIndex('x_1')", sch, coll, None, allow_write=True)
    assert d["plan"] == {"op": "dropIndex", "index": "x_1"} and d["api"][0] == "ok"


# --------------------------------------------------------------------------- chat multi-turno
def test_multiturn_maps_to_chat_component():
    assert vp.mixture_component("chat_multiturn") == "chat"
    assert vp.mixture_component("explain") == "chat"
    assert vp.mixture_component("correction") == "correction"
    assert vp.mixture_component("negative") == "negative"


def test_history_counts_in_budget_and_is_validated(cfg, schemas, snips):
    s = [x for x in snips if x["op"] == "find" and x["kind"] == "read" and x["category"] in ("crud", "operators")][0]
    sch, coll = sch_coll(schemas, s)
    text = s["text"]
    base = {"id": "chat-mt-1", "type": "chat", "kind": "chat_multiturn", "language": "pt-BR", "domain": s["domain"],
            "schema_id": s["schema_id"], "collection": s["collection"], "group_id": s["group_id"], "split": "train", "holdout": False,
            "snippet_id": s["id"], "instruction": "agora ordene", "output": text,
            "context": {"connection": "local", "database": sch.schema_id, "collection": coll.name, "selection": None,
                        "editor": text, "known_collections": []},
            "history": [{"role": "user", "content": "liste", "editor": ""}, {"role": "assistant", "content": text}],
            "validation": {"syntax": "ok", "api": "ok", "fields": "ok", "exec": "ok", "nonempty": True, "plan_equal": True}}
    ctx = vp.Ctx(cfg, schemas, {s["id"]: s}, exec_mode="recorded", tok_mode="off", node=False)
    e = vp.stages_1_to_7(base, "chat", ctx)       # sem KeyError e sem rejeicao
    assert e.failed is None, e.failed
    short = e.val["budget"]["prompt_tokens"]
    # historico longo estoura o prompt de 2048 tokens (conta no orcamento)
    big = dict(base, id="chat-mt-2", history=[{"role": "user", "content": "x " * 7000, "editor": ""},
                                              {"role": "assistant", "content": text}])
    e2 = vp.stages_1_to_7(big, "chat", ctx)
    assert e2.failed and e2.failed[0] == "budget", e2.failed
    assert vp.approx_tokens(vp.chat_prompt_text(big)) > short
    # codigo invalido no historico reprova na etapa de sintaxe/API
    badh = dict(base, id="chat-mt-3", history=[{"role": "user", "content": "liste", "editor": ""},
                                               {"role": "assistant", "content": "db." + coll.name + ".findAll({})"}])
    e3 = vp.stages_1_to_7(badh, "chat", ctx)
    assert e3.failed and e3.failed[0] == "api" and "historico" in e3.failed[1], e3.failed


# --------------------------------------------------------------------------- shards / minhash / decontam-analysis
def test_shard_writer_roundtrip(tmp_path):
    w = js.ShardWriter(tmp_path, "fim", 3)
    for i in range(8):
        w.write({"id": i, "t": "ã"})
    r = w.close()
    assert [f["records"] for f in r["files"]] == [3, 3, 2] and r["records"] == 8
    paths = js.resolve_inputs(tmp_path / "fim.jsonl")
    assert [p.name for p in paths] == ["fim-00000.jsonl", "fim-00001.jsonl", "fim-00002.jsonl"]
    assert [x["id"] for x in js.iter_jsonl(paths)] == list(range(8))
    for f in r["files"]:
        assert hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest() == f["sha256"]
    # reescrever apaga shards velhos
    w2 = js.ShardWriter(tmp_path, "fim", 5)
    for i in range(2):
        w2.write({"id": i})
    w2.close()
    assert len(js.resolve_inputs(tmp_path / "fim.jsonl")) == 1


def test_minhash_numpy_equals_python():
    mh = dc.MinHasher(128, 20261008)
    toks = [f"t{i % 37}" for i in range(150)]
    sh = mh.shingles(toks)
    py = tuple(min((a * x + b) % dc._P for x in sh) for a, b in mh.ab)
    assert mh.signature(toks) == py


def test_decontam_analysis_is_not_applied(tmp_path):
    stages = "[{ $limit: 5 }, { $skip: 2 }, { $limit: 1 }, { $skip: 1 }, { $limit: 7 }]"
    own_e = "; ".join(f"db.pe.find({{ ev_{i}: {i} }})" for i in range(14))      # parte propria (nao compartilhada)
    own_t = "; ".join(f"db.pt.find({{ tr_{i}: {i} }})" for i in range(14))
    ev = {"id": "e1", "split": "test", "snippet_id": "e1", "collection": "c", "type": "fim", "cut_kind": "stage",
          "prefix": own_e + "; db.c.aggregate(" + stages, "middle": "", "suffix": ")"}
    tr = {**ev, "id": "t1", "split": "train", "snippet_id": "t1", "collection": "d",
          "prefix": own_t + "; db.d.aggregate(" + stages.replace("5", "9")}
    lit = dc.run([tr], [("test", ev)], seed=1, blind=tmp_path, analysis_k=(2,), min_content=0)
    ana = lit["analysis"]["2"]
    assert lit["train_removed"] == 1 and "ngram13" in lit["removed"]["t1"]    # regra literal remove (estrutura da linguagem)
    assert ana["removed"] == 0 and ana["kept"] == 1                          # analise k=2: nao removeria
    assert ana["in_by_cut_kind"] == {"stage": 1}
    strict = dc.run([tr], [("test", ev)], seed=1, blind=tmp_path, min_content=2)
    assert strict["train_removed"] == 0                                       # e o resultado aplicado so muda com --ngram-min-content


# --------------------------------------------------------------------------- gen_fim: contagens fixas por split
def test_gen_fim_fixed_eval_counts(cfg, snips):
    c = json.loads(json.dumps(cfg))
    c["fim"]["eval_counts"] = {"val": 30, "test": 20, "bench": 10}
    c["fim"]["eval_margin"] = 1.0
    c["fim"]["eval_max_fraction"] = 0.5
    rep: dict = {}
    recs = list(gen_fim.iter_fim(snips, c, 400, None, False, None, rep))
    assert len(recs) == 400
    from collections import Counter
    cnt = Counter(r["split"] for r in recs)
    assert cnt["val"] == 30 and cnt["test"] == 20 and cnt["bench"] == 10 and cnt["train"] == 340
    # cada corte vem de um snippet do mesmo split
    sp = {s["id"]: s["split"] for s in snips}
    assert all(sp[r["snippet_id"]] == r["split"] for r in recs)
    # 2a execucao identica (determinismo)
    again = list(gen_fim.iter_fim(snips, c, 400, None, False, None))
    assert [r["id"] for r in recs] == [r["id"] for r in again]
    # alvo de categoria com sobrevivencia: a categoria que perde mais ganha peso bruto
    fc = dict(c["fim"], category_survival={"atlas_search": 0.25}, resample_for_decontam=True)
    w0 = gen_fim.category_weights(["atlas_search", "crud"], dict(fc, category_survival={}))
    w1 = gen_fim.category_weights(["atlas_search", "crud"], fc)
    assert w1["atlas_search"] > w0["atlas_search"]


def test_waterfill_respects_capacity_and_redistributes():
    q = gen_fim.waterfill(100, {"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 10, "b": 1000, "c": 1000})
    assert q["a"] == 10 and sum(q.values()) == 100
    assert q["b"] / q["c"] == pytest.approx(1.5, rel=0.05)
    assert sum(gen_fim.waterfill(100, {"a": 1.0}, {"a": 7}).values()) == 7          # oferta menor que o pedido
    assert gen_fim.waterfill(10, {"a": 1.0, "b": 0.0}, {"a": 100, "b": 100}) == {"a": 10, "b": 0}


def test_gen_fim_prefilter_matches_pipeline_decontam_and_cap(cfg, snips):
    c = json.loads(json.dumps(cfg))
    c["fim"].update({"eval_counts": {"val": 40, "test": 30, "bench": 20}, "eval_margin": 1.0, "eval_max_fraction": 0.4,
                     "decontam_prefilter": True, "max_cuts_per_snippet": 2, "decontam_legacy": []})
    rep: dict = {}
    recs = list(gen_fim.iter_fim(lambda: iter(snips), c, 600, None, False, None, rep))
    tr = [r for r in recs if r["split"] == "train"]
    ev = [r for r in recs if r["split"] != "train"]
    assert tr and ev
    assert rep["decontam_prefilter"]["train_snippets_dropped"] > 0
    from collections import Counter
    assert max(Counter(r["snippet_id"] for r in tr).values()) <= 2                  # teto de cortes por snippet
    # a etapa 9 do pipeline (decontaminate.run, regra literal) nao remove nenhum corte de treino
    lite = lambda r: {"id": r["id"], "split": r["split"], "snippet_id": r["snippet_id"], "collection": r["collection"],  # noqa: E731
                      "prefix": r["prefix"], "middle": r["middle"], "suffix": r["suffix"], "type": "fim", "cut_kind": r["cut_kind"]}
    res = dc.run([lite(r) for r in tr], [(r["split"], lite(r)) for r in ev], seed=c["seed"])
    assert res["train_removed"] == 0, res["by_rule"]


def test_chat_pool_survivors_and_caps(tmp_path):
    import gen_chat
    rows, vals = [], []
    for i in range(30):
        sp = "train" if i < 20 else "val"
        rows.append({"id": f"s{i}", "kind": "read", "op": "find", "plan": {"op": "find", "filter": {}}, "schema_id": "d/x", "collection": "c",
                     "domain": "d", "db": "kapi_fx_x", "template_id": "t", "group_id": f"g{i}", "split_key": "k", "split": sp,
                     "holdout": False, "text": "db.c.find({})", "spans": []})
        vals.append({"id": f"s{i}", "exec": "ok", "empty": False})
    rows.append({**rows[0], "id": "w1", "kind": "write", "op": "updateOne"})
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (tmp_path / "v.jsonl").write_text("\n".join(json.dumps(v) for v in vals) + "\n", encoding="utf-8")
    surv = {f"s{i}" for i in range(0, 20, 2)} | {"w1"}
    keep, val = gen_chat.load_chat_pool(tmp_path / "s.jsonl", tmp_path / "v.jsonl", 0, 0, "x", surv)
    ids = {r["id"] for r in keep}
    assert {r["id"] for r in keep if r["split"] == "train"} == surv                      # treino so os sobreviventes
    assert {f"s{i}" for i in range(20, 30)} <= ids                                       # avaliacao nao e filtrada
    assert set(keep[0]) <= set(gen_chat.CHAT_SNIP_KEYS) and "text" not in keep[0]        # registro enxuto
    k1, _ = gen_chat.load_chat_pool(tmp_path / "s.jsonl", tmp_path / "v.jsonl", 5, 3, "x", None)
    a = sorted(r["id"] for r in k1)
    k2, _ = gen_chat.load_chat_pool(tmp_path / "s.jsonl", tmp_path / "v.jsonl", 5, 3, "x", None)
    assert sorted(r["id"] for r in k2) == a                                              # teto deterministico
    assert sum(1 for r in k1 if r["split"] == "train" and r["kind"] == "read") == 5
    assert sum(1 for r in k1 if r["split"] == "val") == 3


def test_pipeline_reads_and_writes_shards(tmp_path):
    from argparse import Namespace
    src = gs.ROOT / "data" / "generated" / "v1"
    if not (src / "fim_ctx.jsonl").exists() or not (src / "chat.jsonl").exists():
        pytest.skip("data/generated/v1 ausente")
    lines = [l for _, l in zip(range(240), open(src / "fim_ctx.jsonl", encoding="utf-8"))]
    d = tmp_path / "gen"
    d.mkdir()
    (d / "fim_ctx-00000.jsonl").write_text("".join(lines[:100]), encoding="utf-8")
    (d / "fim_ctx-00001.jsonl").write_text("".join(lines[100:]), encoding="utf-8")
    chat = [l for _, l in zip(range(60), open(src / "chat.jsonl", encoding="utf-8"))]
    (d / "chat.jsonl").write_text("".join(chat), encoding="utf-8")
    args = Namespace(config=str(gs.CONFIG_PATH), ver="vx", fim=str(d / "fim_ctx.jsonl"), chat=str(d / "chat.jsonl"), snippets=None,
                     include_snippets=False, out_dir=str(tmp_path / "out"), splits_dir=str(tmp_path / "sp"), report=str(tmp_path / "r.md"),
                     decontam_dir=str(tmp_path / "dc"), exec="recorded", tokenizer="off", no_node=True, legacy=[], dedup_jaccard=0.85,
                     ngram_min_content=0, analysis_ngram_min_content=2, shard_size=70, compare=None, limit=0)
    out = vp.run_pipeline(args)
    outs = sorted(p.name for p in (tmp_path / "out").glob("fim-*.jsonl"))
    assert outs and all(n.startswith("fim-0000") for n in outs)
    kept = sum(1 for p in (tmp_path / "out").glob("fim-*.jsonl") for _ in open(p, encoding="utf-8"))
    assert kept == sum(v for k, v in out["manifest"]["counts"]["by_type_split"].items() if k.startswith("fim."))
    assert out["manifest"]["inputs_records"] == {"fim": 240, "chat": 60}
    assert "analysis" in out["manifest"]["decontam"] and "2" in (out["manifest"]["decontam"]["analysis"] or {})
    assert "survival_by_cut_kind" in out["manifest"]["decontam"]
    rep = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "Auditoria paralela" in rep and "Sobrevivencia por cut_kind" in rep
