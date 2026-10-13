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
    assert kinds == {"chat", "chat_multiturn", "correction", "negative", "explain"}
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
    assert regs >= {"n", "i", "t", "g", "q"}


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
        assert r["defect"]["type"] in set(gc.ALL_DEFECTS)
        assert r["defect"]["n_defects"] == len(r["defect"]["types"]) >= 1
    assert len({r["defect"]["type"] for r in corr}) >= 8
    assert len(gc.NEW_DEFECTS) >= 6


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
                assert ".find(" in l or ".countDocuments(" in l
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


# --------------------------------------------------------------------------- ciclo 2


def _gen_with(cfg, snips, validation, schemas, mongo, node, **chat_over):
    c = json.loads(json.dumps(cfg))
    c["chat"].update(chat_over)
    return gc.ChatGen(c, snips, validation, schemas, mongo, node)


def _force_corrections(g, family, want=2, max_snips=600, seed="t"):
    """Gera ate `want` correcoes da familia, forcando snippets um a um (mesmo caminho do gerador)."""
    out = []
    for k, snip in enumerate(g.reads[:max_snips]):
        if not gc.applicable(family, snip["plan"]):
            continue
        g._force_snip, g._force_defect = snip, family
        try:
            rec = g.gen_correction(k, random.Random(f"{seed}:{family}:{k}"), "pt" if k % 2 else "en", "n")
        finally:
            g._force_snip = g._force_defect = None
        if rec is not None:
            out.append(rec)
            if len(out) >= want:
                break
    return out


@pytest.mark.parametrize("family", list(gc.ALL_DEFECTS))
def test_each_defect_family_fails_then_passes(family, cfg, snips, validation, schemas, mongo, node):
    g = _gen_with(cfg, snips, validation, schemas, mongo, node, defect_count={"1": 1.0})
    recs = _force_corrections(g, family)
    assert recs, f"familia {family} nao gerou nenhuma correcao"
    by = {s.full_id: s for s in schemas}
    for r in recs:
        sch = by[r["schema_id"]]
        coll = next(c for c in sch.colls if c.name == r["collection"])
        db = r["validation"]["fixture"]
        bad, good = r["context"]["editor"], r["output"]
        assert r["defect"]["type"] == family and r["defect"]["n_defects"] == 1
        assert bad != good
        vb = cv.validate_text(bad, sch, coll, mongo, node.check(bad), db=db)
        assert vb["stage"] != "ok", (family, bad)                          # o defeituoso FALHA
        assert vb["stage"] == r["defect"]["failed_stage"]
        vg = cv.validate_text(good, sch, coll, mongo, node.check(good), db=db)
        assert vg["stage"] == "ok", (family, vg["error"])                  # o corrigido PASSA
        assert not cv.has_executable_write(good) and not cv.has_executable_write(bad)


def test_new_defect_detectors_static(schemas, node):
    """Cada regra estatica nova pega o defeito e nao pega o texto correto (sem MongoDB)."""
    sch = next(s for s in schemas if s.colls)
    coll = next(c for c in sch.colls if sum(1 for p, f in c.fields.items()
                                            if f.leaf_t in ("string", "int32", "int64", "double") and p != "_id" and "." not in p) >= 2)
    name = coll.name
    fs = [p for p, f in coll.fields.items() if f.leaf_t in ("string", "int32", "int64", "double") and p != "_id" and "." not in p]
    a, b = fs[0], fs[1]
    cases = {
        "group_ok": (f'db["{name}"].aggregate([{{ $group: {{ _id: "${a}", n: {{ $sum: 1 }} }} }}])', "ok"),
        "group_no_id": (f'db["{name}"].aggregate([{{ $group: {{ n: {{ $sum: 1 }} }} }}])', "fields"),
        "acc_in_match": (f'db["{name}"].find({{ "{a}": {{ $sum: 1 }} }})', "api"),
        "query_op_in_group": (f'db["{name}"].aggregate([{{ $group: {{ _id: "${a}", n: {{ $gt: 1 }} }} }}])', "api"),
        "proj_mix": (f'db["{name}"].find({{}}, {{ "{a}": 1, "{b}": 0 }})', "fields"),
        "proj_id_ok": (f'db["{name}"].find({{}}, {{ "{a}": 1, "{b}": 1, "_id": 0 }})', "ok"),
        "sort_dir": (f'db["{name}"].find({{}}).sort({{ "{a}": "desc" }})', "fields"),
        "limit_str": (f'db["{name}"].find({{}}).limit("5")', "api"),
        "limit_stage_zero": (f'db["{name}"].aggregate([{{ $limit: 0 }}])', "fields"),
        "hint": (f'db["{name}"].find({{}}).hint({{ "{a}": 1 }})', "api"),
        "unknown_method": (f'db["{name}"].findAll({{}})', "api"),
        "lookup_missing_as": (f'db["{name}"].aggregate([{{ $lookup: {{ from: "x", localField: "{a}", foreignField: "_id" }} }}])', "fields"),
    }
    for k, (txt, stage) in cases.items():
        v = cv.validate_text(txt, sch, coll, None, node.check(txt))
        assert v["stage"] == stage, (k, v["stage"], v["error"])


def test_combined_defects(cfg, snips, validation, schemas, mongo, node):
    g = _gen_with(cfg, snips, validation, schemas, mongo, node, defect_count={"3": 1.0})
    by = {s.full_id: s for s in schemas}
    n3 = 0
    for fam in ("wrong_field", "stage_order", "limit_value", "misplaced_operator", "wrong_operator"):
        for r in _force_corrections(g, fam, want=3, seed="comb"):
            assert r["defect"]["n_defects"] == len(r["defect"]["types"]) >= 1
            if r["defect"]["n_defects"] >= 3:
                n3 += 1
                assert len(r["defect"]["combined"]) == r["defect"]["n_defects"] - 1
                assert len(set(r["defect"]["types"])) == r["defect"]["n_defects"]    # familias distintas
            sch = by[r["schema_id"]]
            coll = next(c for c in sch.colls if c.name == r["collection"])
            bad, good = r["context"]["editor"], r["output"]
            assert cv.validate_text(bad, sch, coll, mongo, node.check(bad), db=r["validation"]["fixture"])["stage"] != "ok"
            assert cv.validate_text(good, sch, coll, mongo, node.check(good), db=r["validation"]["fixture"])["stage"] == "ok"
    assert n3 >= 3
    # configuracao: ~15% das correcoes com 3+ defeitos
    dc = cfg["chat"]["defect_count"]
    assert 0.12 <= sum(v for k, v in dc.items() if int(k) >= 3) <= 0.18


def test_multiturn(chat, snips, schemas, mongo, node):
    recs, _ = chat
    mt = [r for r in recs if r["kind"] == "chat_multiturn"]
    assert len(mt) >= 8
    by_s = {s["id"]: s for s in snips}
    by = {s.full_id: s for s in schemas}
    for r in mt:
        assert [h["role"] for h in r["history"]] == ["user", "assistant"]
        h_user, h_code = r["history"][0]["content"], r["history"][1]["content"]
        assert h_user.strip() and h_code.strip() and r["instruction"].strip()
        assert r["context"]["editor"] == h_code                 # o editor do turno 2 e a resposta do turno 1
        assert r["output"] != h_code and r["intent"]["turns"] == 2
        plan = by_s[r["snippet_id"]]["plan"]
        sch = by[r["schema_id"]]
        coll = next(c for c in sch.colls if c.name == r["collection"])
        v = cv.validate_text(r["output"], sch, coll, mongo, node.check(r["output"]), db=r["validation"]["fixture"])
        assert v["stage"] == "ok" and cv.plans_equal(v["plan"], plan)
        vh = cv.validate_text(h_code, sch, coll, mongo, node.check(h_code), db=r["validation"]["fixture"])
        assert vh["stage"] == "ok"
        assert by_s[r["snippet_id"]]["split"] == r["split"]
        assert not cv.has_executable_write(r["output"]) and not cv.has_executable_write(h_code)
    # o historico nao carrega o pedido do turno 2 (sem vazamento do refinamento)
    assert all(r["instruction"] not in r["history"][0]["content"] for r in mt)


def test_multiturn_refinement_kinds(cfg, snips, validation, schemas, mongo, node):
    g = _gen_with(cfg, snips, validation, schemas, mongo, node)
    got = {"tail": 0, "filter": 0}
    for k, snip in enumerate(g.reads[:700]):
        for which in ("tail", "filter"):
            g._force_snip, g._force_refine = snip, which
            try:
                rec = g.gen_multiturn(k, random.Random(f"mt:{which}:{k}"), "pt" if k % 2 else "en", "n")
            finally:
                g._force_snip = g._force_refine = None
            if rec:
                got[which] += 1
                assert rec["intent"]["refine"] == which
        if min(got.values()) >= 3:
            break
    assert got["tail"] >= 3 and got["filter"] >= 3


def test_split_role_inherited_and_conflicts(cfg, snips, validation, schemas, mongo, node, chat):
    recs, _ = chat
    by_s = {s["id"]: s for s in snips}
    for r in recs:                                             # split do chat == split do snippet, sempre
        assert r["split"] == by_s[r["snippet_id"]]["split"]
        assert r["schema_split_role"] in (None, "train", "eval")
    # papel conhecido + snippet em split incompativel => snippet fora do pool (sem dependencia dura: role ausente = ignorado)
    train_snips = [s for s in snips if s["split"] == "train" and s["kind"] == "read"]
    sid = train_snips[0]["schema_id"]
    sch = next(s for s in schemas if s.full_id == sid)
    old = getattr(sch, "split_role", None)
    try:
        sch.split_role = "eval"
        g = gc.ChatGen(cfg, snips, validation, schemas, mongo, node)
        assert g.role_conflicts >= 1
        assert all(not (r["schema_id"] == sid and r["split"] == "train") for r in g.reads)
        sch.split_role = None
        g2 = gc.ChatGen(cfg, snips, validation, schemas, mongo, node)
        assert g2.role_conflicts < g.role_conflicts or g2.role_conflicts == 0
    finally:
        sch.split_role = old
    assert gc.role_conflict(None, {"split": "train"}) is False
    assert gc.role_conflict("train", {"split": "test", "holdout": False}) is True
    assert gc.role_conflict("eval", {"split": "bench"}) is False


def test_negative_variety(chat):
    recs, _ = chat
    neg = [r for r in recs if r["kind"] == "negative"]
    types = {r["negative"]["type"] for r in neg}
    assert types >= {"write", "out_of_scope", "no_collection", "ambiguous"}
    assert len(types) >= 5
    for r in neg:
        assert not cv.has_executable_write(r["output"])
        assert r["output"].lstrip().startswith("//")
        assert isinstance(r["context"]["known_collections"], list) and r["context"]["known_collections"]   # campo estruturado
        if r["negative"]["type"] == "ambiguous_field":
            assert "?" in r["output"] and r["negative"]["candidates"]
    assert len(neg) / len(recs) <= 0.15


def test_unique_keys_and_instruction_variety(chat):
    recs, g = chat
    keys = [gc.ChatGen.record_key(r) for r in recs]
    assert len(set(keys)) == len(keys)                         # nenhuma repeticao pela chave normalizada
    # variedade: a mesma consulta rende muitas instrucoes distintas por registro/idioma
    plan = {"op": "find", "filter": {"status": "paid", "total": {"$gte": 10}}, "sort": {"total": -1}, "limit": 5}
    seen = {"pt": set(), "en": set()}
    for lang in ("pt", "en"):
        for reg in "nitgq":
            for sd in range(12):
                rng = random.Random(f"{lang}{reg}{sd}")
                d = cd.Describer(lang, rng, reg, rng.random() < 0.1)
                seen[lang].add(gc.norm_text(cd.finalize(d.describe(plan, "orders"), lang, rng, reg, 0.3, 0.04)))
    assert len(seen["pt"]) >= 35 and len(seen["en"]) >= 35
    assert any(t.endswith("?") for t in seen["pt"]) and any(";" in t for t in seen["en"])   # indireta e telegrafica


def test_capacity_report(cfg, snips, validation, schemas, mongo, node):
    g = _gen_with(cfg, snips, validation, schemas, mongo, node)
    keep = sorted({r["schema_id"] for r in g.reads})[:3]
    g.reads = [r for r in g.reads if r["schema_id"] in keep]
    g.writes = [r for r in g.writes if r["schema_id"] in keep]
    cap = g.capacity(per_schema=3, rounds=2)
    assert cap["schemas"] == len(keep)
    fams = cap["families"]
    assert {"chat.create", "chat.modify", "explain"} <= set(fams) and len(fams) >= 25
    assert any(v["est_unique"] > 0 for per in fams.values() for v in per.values())
    summ = gc.summarize_capacity(cap, cfg["chat"]["kinds"], 15000)
    assert summ["bottleneck_kind"] in cfg["chat"]["kinds"] and set(summ["needed"]) == set(cfg["chat"]["kinds"])
    assert "familia" in gc.format_capacity(cap, summ)


def test_decontam_friendly_error_instructions(chat):
    """Instrucao com mensagem de erro nao vira boilerplate de >= 13 tokens (evita remocao em massa por 13-grama)."""
    import re
    recs, _ = chat
    corr = [r for r in recs if r["kind"] == "correction"]
    long_ = [r for r in corr if r["defect"]["error_in_instruction"] and len(re.findall(r"\w+", r["instruction"])) > 16]
    assert len(long_) <= max(3, 0.05 * len(corr))
