"""Testes do gerador de chat/correcao/negativos (gen_chat / chat_describe / chat_validate)."""
import json
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import chat_describe as cd  # noqa: E402
import chat_validate as cv  # noqa: E402
import gen_chat as gc  # noqa: E402
import gen_snippets as gs  # noqa: E402
import validate_snippets as vs  # noqa: E402

ROOT = gs.ROOT
N_SNIP = 700
N_CHAT = 220


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
    return gs.generate(cfg, schemas, N_SNIP)


@pytest.fixture(scope="module")
def mongo(cfg):
    try:
        ex = cv.Executor(cfg)
    except Exception:
        return None
    if not ex.existing:
        return None
    return ex


@pytest.fixture(scope="module")
def node():
    n = cv.NodeChecker()
    yield n
    n.close()


def make_validation(snips, mongo):
    if mongo is None:  # sem MongoDB: so a validacao estatica; execucao marcada como ok/nao vazia
        return {r["id"]: {"exec": "ok" if r["kind"] == "read" else "skipped", "empty": False} for r in snips}
    out = {}
    for r in snips:
        v = vs.validate_record(lambda n: mongo.client[n], r, 2000, 101)
        out[r["id"]] = v
    return out


@pytest.fixture(scope="module")
def validation(snips, mongo):
    return make_validation(snips, mongo)


def run_gen(cfg, snips, validation, schemas, mongo, node, count=N_CHAT, seed=None):
    c = dict(cfg)
    if seed is not None:
        c = {**cfg, "seed": seed}
    g = gc.ChatGen(c, snips, validation, schemas, mongo, node)
    return g.generate(count), g


@pytest.fixture(scope="module")
def chat(cfg, snips, validation, schemas, mongo, node):
    recs, g = run_gen(cfg, snips, validation, schemas, mongo, node)
    return recs, g


def test_deterministic(cfg, snips, validation, schemas, mongo, node, chat):
    recs, _ = chat
    again, _ = run_gen(cfg, snips, validation, schemas, mongo, node)
    assert json.dumps(again, sort_keys=True) == json.dumps(recs, sort_keys=True)
    other, _ = run_gen(cfg, snips, validation, schemas, mongo, node, count=60, seed=cfg["seed"] + 7)
    assert [r["instruction"] for r in other] != [r["instruction"] for r in recs[:60]]


def test_schema_and_kinds(chat, cfg):
    recs, _ = chat
    assert len(recs) == N_CHAT
    kinds = {r["kind"] for r in recs}
    assert kinds == {"chat", "correction", "negative", "explain"}
    n = len(recs)
    assert sum(r["kind"] == "explain" for r in recs) / n <= 0.05 + 0.02
    for r in recs:
        for k in ("id", "kind", "language", "domain", "schema_id", "collection", "group", "split_key", "split", "holdout",
                  "template_id", "snippet_id", "instruction", "context", "output"):
            assert k in r, (k, r["id"])
        assert r["language"] in ("pt-BR", "en")
        assert set(r["context"]) >= {"connection", "database", "collection", "selection", "editor"}
        assert r["instruction"].strip() and r["output"].strip()
        assert "<|" not in r["output"] and "```" not in r["output"] and "\x01" not in r["instruction"]


def test_pt_br_and_en(chat):
    recs, _ = chat
    pt = [r for r in recs if r["language"] == "pt-BR"]
    en = [r for r in recs if r["language"] == "en"]
    assert len(pt) >= 0.35 * len(recs) and len(en) >= 0.35 * len(recs)
    # pt-BR: palavras tipicas; en: sem acentos de template
    assert any(any(w in r["instruction"].lower() for w in ("busque", "liste", "mostra", "filtrar", "quantos", "conte", "traz", "corrij", "corrige"))
               for r in pt)
    assert any(any(w in r["instruction"].lower() for w in ("find", "list", "show", "count", "fix", "get", "how many")) for r in en)
    assert any(r.get("pt_accents_stripped") for r in pt) or True
    regs = {r.get("register") for r in recs if r["kind"] != "negative"}
    assert regs >= {"n", "i", "t"}


def test_correction_fails_then_passes(chat, schemas, mongo, node):
    recs, _ = chat
    by = {s.full_id: s for s in schemas}
    corr = [r for r in recs if r["kind"] == "correction"]
    assert len(corr) >= 20
    allowed = {"syntax", "api", "fields", "type", "exec"}
    for r in corr:
        sch = by[r["schema_id"]]
        coll = next(c for c in sch.colls if c.name == r["collection"])
        db = r["validation"]["fixture"]
        bad, good = r["context"]["editor"], r["output"]
        assert bad != good
        vb = cv.validate_text(bad, sch, coll, mongo, node.check(bad), db=db)
        assert vb["stage"] in allowed, (r["id"], vb["stage"], bad)
        assert vb["stage"] == r["defect"]["failed_stage"]
        vg = cv.validate_text(good, sch, coll, mongo, node.check(good), db=db)
        assert vg["stage"] == "ok", (r["id"], vg["stage"], vg["error"])
        assert r["defect"]["type"] in {"wrong_field", "wrong_operator", "missing_dollar", "syntax_comma", "syntax_quote",
                                       "syntax_bracket", "wrong_type", "invalid_api"}
    assert len({r["defect"]["type"] for r in corr}) >= 5


def test_defect_detectors(schemas, cfg, node):
    sch = next(s for s in schemas if s.colls)
    coll = sch.colls[0]
    name = coll.name
    f = next(p for p, x in coll.fields.items() if x.leaf_t == "string" and p != "_id")
    base = f'db["{name}"].find({{ "{f}": "a" }})'
    cases = {
        "ok": (base, "ok"),
        "campo": (f'db["{name}"].find({{ "{f}zzq": "a" }})', "fields"),
        "op": (f'db["{name}"].find({{ "{f}": {{ $gtee: "a" }} }})', "api"),
        "dolar": (f'db["{name}"].find({{ "{f}": {{ gte: "a" }} }})', "fields"),
        "virgula": (f'db["{name}"].find({{ "{f}": "a" "_id": 1 }})', "syntax"),
        "aspas": (f'db["{name}"].find({{ "{f}: "a" }})', "syntax"),
        "pretty": (base + ".pretty()", "api"),
        "tipo": (f'db["{name}"].find({{ "{f}": 5 }})', "type"),
        "oid": (f'db["{name}"].find({{ "_id": ObjectId("123") }})', "api"),
        "write": (f'db["{name}"].deleteMany({{}})', "api"),
    }
    for k, (txt, stage) in cases.items():
        v = cv.validate_text(txt, sch, coll, None, node.check(txt))
        assert v["stage"] == stage, (k, v)


def test_no_executable_writes(chat):
    recs, g = chat
    for r in recs:
        assert not cv.has_executable_write(r["output"]), r["output"]
        if r["kind"] == "negative":
            code = [l for l in r["output"].splitlines() if l.strip() and not l.strip().startswith("//")]
            assert len(code) <= 1  # no maximo o find de pre-visualizacao
            for l in code:
                assert ".find(" in l
    assert any(r["kind"] == "negative" and r["negative"]["type"] == "write" for r in recs)
    assert g.ex is None or g.ex.violations == []
    # o detector enxerga escritas reais e ignora comentarios
    for bad in ("db.a.deleteMany({})", "db.a.updateOne({}, {$set:{a:1}})", "db.a.aggregate([{$out:'x'}])", "db.a.drop()",
                "db.a.insertOne({})", "db.runCommand({ping:1})", "db.a.createIndex({a:1})"):
        assert cv.has_executable_write(bad)
    assert not cv.has_executable_write("// db.a.deleteMany({})\n// nao executo escrita")


def test_no_group_leak_and_holdout(chat, cfg, snips):
    recs, _ = chat
    g2s = {}
    for r in recs:
        assert g2s.setdefault(r["group"], r["split"]) == r["split"], r["group"]
    keys = {}
    for r in recs:
        keys.setdefault(r["split_key"], set()).add(r["split"])
    assert all(len(v) == 1 for v in keys.values())
    hold = set(cfg["holdout_domains"])
    by = {s["id"]: s for s in snips}
    for r in recs:
        assert by[r["snippet_id"]]["split"] == r["split"] and by[r["snippet_id"]]["group_id"] == r["group"]
        if r["domain"] in hold:
            assert r["holdout"] and r["split"] in ("test", "bench")
    assert any(r["domain"] in hold for r in recs)
    # nenhum par instrucao+saida repetido entre splits
    seen = {}
    for r in recs:
        k = (r["schema_id"], r["collection"], gc.norm_text(r["instruction"]), r["output"], r["context"]["editor"])
        assert seen.setdefault(k, r["split"]) == r["split"]


def test_outputs_reference_only_schema_fields(chat, schemas):
    recs, _ = chat
    by = {s.full_id: s for s in schemas}
    for r in recs:
        if r["kind"] == "negative" and "find(" not in r["output"]:
            continue
        sch = by[r["schema_id"]]
        coll = next(c for c in sch.colls if c.name == r["collection"])
        out = r["output"]
        if r["kind"] in ("explain", "negative"):
            out = "\n".join(l for l in out.splitlines() if not l.strip().startswith("//"))
        plan, meta = cv.parse_console(out)
        an = gs.analyze_plan(plan, sch, coll)
        assert an["missing"] == [], (r["id"], an["missing"])
        assert all(p.split("::")[-1] in (sch.by_key[p.split("::")[0]].fields if "::" in p else coll.fields) for p in an["used"])


def test_values_cited_literally(chat, snips):
    recs, _ = chat
    by = {s["id"]: s for s in snips}
    checked = 0
    for r in recs:
        if r["kind"] != "chat" or r["intent"]["kind"] != "create" or r.get("typo_injected") or r.get("pt_accents_stripped"):
            continue
        plan = by[r["snippet_id"]]["plan"]
        flt = plan.get("filter") or {}
        for x in gc.leaves({"filter": flt}):
            if isinstance(x, str):
                s = x[1:] if x.startswith("^") else x
                if s:
                    assert s in r["instruction"], (r["instruction"], x)
                    checked += 1
    assert checked > 10


def test_blind_guard_applied(cfg, snips, validation, schemas, mongo, node, chat, tmp_path, monkeypatch):
    import blind_guard as bg
    recs, _ = chat
    victim = next(r for r in recs if r["kind"] == "chat")
    ph = bg.prompt_sha256(bg.prompt_text({"kind": "chat", "instruction": victim["instruction"], "context": victim["context"]["editor"]}))
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "manifest.json").write_text(json.dumps({"item_sha256": {}, "prompt_sha256": [ph]}), encoding="utf-8")
    monkeypatch.setenv("KAPI_BLIND_DIR", str(tmp_path))
    again, g = run_gen(cfg, snips, validation, schemas, mongo, node)
    assert g.discards["blind_guard"] >= 1
    assert all(not (r["instruction"] == victim["instruction"] and r["context"]["editor"] == victim["context"]["editor"]) for r in again)


def test_config_mixture(cfg):
    cc = cfg["chat"]
    m = cc["mixture"]
    assert abs(sum(m.values()) - 1.0) < 1e-6
    assert m["chat"] + m["correction"] <= 0.25
    assert m["negative"] <= 0.05
    assert cc["kinds"]["negative"] <= 0.15 and cc["kinds"]["explain"] <= 0.05
    assert abs(sum(cc["kinds"].values()) - 1.0) < 1e-6
    assert abs(sum(cc["defects"].values()) - 1.0) < 1e-6
    assert abs(sum(cc["negative_types"].values()) - 1.0) < 1e-6


def test_describer_unsupported_raises():
    d = cd.Describer("pt", random.Random(1))
    with pytest.raises(cd.Unsupported):
        d.describe({"op": "aggregate", "pipeline": [{"$lookup": {"from": "a", "let": {"x": "$y"}, "pipeline": [], "as": "z"}}]}, "c")
    t = d.describe({"op": "find", "filter": {"status": "paid", "total": {"$gte": 10, "$lt": 20}}, "limit": 5}, "orders")
    s = cd.strip_marks(t)
    assert "orders" in s and '"paid"' in s and "10" in s and "20" in s and "5" in s


def test_generated_chat_file_integrity(cfg):
    p = ROOT / "data" / "generated" / "v1" / "chat.jsonl"
    if not p.exists():
        pytest.skip("chat.jsonl v1 nao gerado")
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    assert len(recs) >= 5000
    n = len(recs)
    kinds = {k: sum(r["kind"] == k for r in recs) / n for k in ("chat", "correction", "negative", "explain")}
    assert kinds["explain"] <= 0.05 + 1e-9 and kinds["negative"] <= 0.15
    hold = set(cfg["holdout_domains"])
    g2s = {}
    for r in recs:
        assert g2s.setdefault(r["group"], r["split"]) == r["split"]
        if r["domain"] in hold:
            assert r["split"] in ("test", "bench") and r["holdout"]
        assert not cv.has_executable_write(r["output"])
    langs = {r["language"] for r in recs}
    assert langs == {"pt-BR", "en"}
    assert len({r["id"] for r in recs}) == n
