"""Resiliencia do validador: BSON invalido/gigante vira rejeicao (nunca crash), travas viram timeout (bisseccao), retomada por lote e heartbeat."""
import base64
import json
import struct
import subprocess
import time

import pytest

import run_pipeline
import validate_text as V
from common import load_cfg, load_schema_infos


# ------------------------------------------------------------------------------------------------------------ BSON auxiliar
def _bad_uuid_find() -> bytes:
    """find com UUID subtipo 4 de 11 bytes (o que `UUID('8131f50c-...')` truncado gera no mongosh): invalido para o decoder do pymongo."""
    def el(t, name, val):
        return bytes([t]) + name.encode() + b"\x00" + val

    flt = el(0x05, "externalId", struct.pack("<i", 11) + b"\x04" + b"\x81" * 11)
    flt = struct.pack("<i", len(flt) + 5) + flt + b"\x00"
    body = b"\x02find\x00" + struct.pack("<i", 8) + b"tenants\x00" + b"\x03filter\x00" + flt
    return struct.pack("<i", len(body) + 5) + body + b"\x00"


def _good_find() -> bytes:
    from bson import encode
    return encode({"find": "tenants", "filter": {"plan": "team"}, "$db": "x"})


def _cmd(raw: bytes, name="find"):
    return {"name": name, "db": "x", "bson": base64.b64encode(raw).decode()}


def test_decode_invalid_bson_is_reject_not_crash():
    with pytest.raises(V.ExecReject) as e:
        V.decode_command(_cmd(_bad_uuid_find()))
    assert e.value.kind == "bson_decode"
    with pytest.raises(V.ExecReject) as e:
        V.decode_command(_cmd(_good_find()[:-7]))          # BSON truncado no meio
    assert e.value.kind == "bson_decode"
    with pytest.raises(V.ExecReject) as e:
        V.decode_command({"name": "find", "db": "x", "bson": "@@nao-e-base64@@"})
    assert e.value.kind == "bson_decode"
    with pytest.raises(V.ExecReject) as e:
        V.decode_command({"name": "find", "db": "x", "bson": None, "too_large": 9_000_000})   # saida/comando enorme nao e truncado: e rejeitado
    assert e.value.kind == "cmd_too_large"
    assert V.decode_command(_cmd(_good_find()))["find"] == "tenants"


# ------------------------------------------------------------------------------------------------------------ funil com WSL simulado
@pytest.fixture
def env(tmp_path, monkeypatch):
    infos = load_schema_infos(roles=("train",))
    info = next((i for i in infos if i.schema_id == "tenantly"), None)
    if info is None:
        pytest.skip("tenantly nao semeado")
    cfg = load_cfg()
    cfg["execution"].update(batch_size=3, workers=2)

    class FakeMongo:
        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

    import pymongo
    monkeypatch.setattr(pymongo, "MongoClient", FakeMongo)
    monkeypatch.setattr(V, "fingerprint", lambda m, dbs: {d: "h" for d in dbs})
    monkeypatch.setattr(V, "opcounters", lambda m: {"insert": 0, "update": 0, "delete": 0, "query": 0, "command": 0, "getmore": 0})
    base = {"schema_id": info.full_id, "schema_short": info.schema_id, "domain": info.domain, "db": info.db, "collection": "tenants",
            "split_role": "train", "persona": "t", "task": "t"}
    rows = [{**base, "id": f"q{i}", "qkey": f"k{i}", "text": f"db.tenants.find({{ plan: 'p{i}' }}).limit({i + 1})"} for i in range(10)]
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return dict(cfg=cfg, infos=infos, raw=raw, out=tmp_path / "out", tmp=tmp_path)


def _ok_result(i, raw=None):
    return {"id": i, "status": "ok", "error": None, "n_docs": 2, "n_reads": 1, "ms": 5, "commands": [_cmd(raw or _good_find())]}


def _fake_run_batch(calls, behave=None):
    def fake(args):
        bdir, items, cfg = args
        calls.append([it["id"] for it in items])
        res = {}
        for it in items:
            r = (behave or {}).get(it["id"]) or _ok_result(it["id"])
            res[it["id"]] = r
        return {"dir": bdir, "rc": 0, "stderr": "", "seconds": 0.1, "results": res, "proxy": {"allowed": {"find": len(items)}, "blocked": {}, "queries": len(items)}}
    return fake


def _rejected(out):
    return {json.loads(l)["id"]: json.loads(l) for l in open(out / "queries.rejected.jsonl", encoding="utf-8")}


def test_bad_bson_and_timeout_are_rejected_batch_survives(env, monkeypatch):
    calls = []
    behave = {"q4": _ok_result("q4", _bad_uuid_find()),
              "q5": {"id": "q5", "status": "timeout", "error": "lote_travou_prazo_duro", "n_docs": 0, "commands": []},
              "q6": _ok_result("q6", _good_find()[:-9])}
    monkeypatch.setattr(V, "run_batch", _fake_run_batch(calls, behave))
    summ = V.validate(env["raw"], env["out"], env["cfg"], infos=env["infos"])      # nao levanta
    rej = _rejected(env["out"])
    assert rej["q4"]["stage"] == "exec" and rej["q4"]["reason"].startswith("bson_decode")
    assert rej["q6"]["reason"].startswith("bson_decode")
    assert rej["q5"]["reason"].startswith("timeout")
    assert summ["rejections_by_reason"]["exec:bson_decode"] == 2 and summ["rejections_by_reason"]["exec:timeout"] == 1
    assert summ["readonly_proof"]["dbHash_before_equals_after"] is True


def test_hung_batch_is_bisected_to_single_timeout(env, monkeypatch, tmp_path):
    """run_batch_once devolve None (lote travou/morreu) para todo lote com a consulta venenosa: bisseccao isola so ela."""
    seen = []

    def once(bdir, items, cfg):
        ids = [it["id"] for it in items]
        seen.append(ids)
        if "q7" in ids:
            return None
        return {"dir": bdir, "rc": 0, "stderr": "", "seconds": 0.1, "results": {i: _ok_result(i) for i in ids}, "proxy": {"allowed": {}, "blocked": {}}}

    monkeypatch.setattr(V, "run_batch_once", once)
    out = V.run_batch((tmp_path / "b0", [{"id": f"q{i}", "db": "x", "text": "t"} for i in range(8)], env["cfg"]))
    assert {i: r["status"] for i, r in out["results"].items()} == {**{f"q{i}": "ok" for i in range(8)}, "q7": "timeout"}
    assert len(out["results"]) == 8


def test_resume_skips_validated_batches_and_heartbeat_advances(env, monkeypatch):
    beats, calls = [], []
    state = {"n": 0}

    base = _fake_run_batch(calls)

    def dying(args):
        state["n"] += 1
        if state["n"] > 2:                      # a partir do 3o lote o orquestrador "cai" (excecao no lote: nada e persistido)
            raise RuntimeError("wsl sumiu")
        return base(args)

    monkeypatch.setattr(V, "run_batch", dying)
    V.validate(env["raw"], env["out"], env["cfg"], infos=env["infos"], workers=1, progress=beats.append)
    done = [json.loads(l) for l in open(env["out"] / "done.idx", encoding="utf-8")]
    assert len(done) == 2                                                   # so os lotes concluidos ficaram persistidos
    persisted = {json.loads(l)["id"] for l in open(env["out"] / "exec_results.jsonl", encoding="utf-8")}
    assert persisted == {i for c in calls for i in c} and len(persisted) == 6
    bd = [b["batches_done"] for b in beats if b.get("stage") == "exec"]
    assert bd == sorted(bd) and bd[-1] >= 2 and len(set(bd)) >= 3           # heartbeat por lote, monotonico
    assert beats[-1]["exec_ok"] >= 6

    # retomada: so executa as consultas que faltam (4), nenhuma das 6 ja validadas
    calls2 = []
    monkeypatch.setattr(V, "run_batch", _fake_run_batch(calls2))
    summ = V.validate(env["raw"], env["out"], env["cfg"], infos=env["infos"], workers=1)
    ran = {i for c in calls2 for i in c}
    assert ran.isdisjoint(persisted) and len(ran) == 4
    assert summ["funnel"]["exec_resumed"] == 6
    assert summ["accepted"] == 10 - sum(1 for l in open(env["out"] / "queries.rejected.jsonl", encoding="utf-8"))

    # terceira execucao: tudo persistido, nenhum lote roda
    calls3 = []
    monkeypatch.setattr(V, "run_batch", _fake_run_batch(calls3))
    V.validate(env["raw"], env["out"], env["cfg"], infos=env["infos"], workers=1)
    assert calls3 == []


def test_pipeline_heartbeat_is_atomic_and_fresh(tmp_path):
    run_pipeline.heartbeat(tmp_path, phase="validate", batches_done=1)
    t1 = json.loads((tmp_path / "heartbeat.json").read_text())
    time.sleep(0.02)
    run_pipeline.heartbeat(tmp_path, phase="validate", batches_done=2, accepted_provisional=7)
    t2 = json.loads((tmp_path / "heartbeat.json").read_text())
    assert t2["ts"] > t1["ts"] and t2["batches_done"] == 2 and t2["accepted_provisional"] == 7
    assert not (tmp_path / "heartbeat.json.tmp").exists()


# ------------------------------------------------------------------------------------------------------------ integracao real (WSL)
def _wsl_ok():
    try:
        return subprocess.run(["wsl", "-d", "Ubuntu", "-e", "bash", "-c", "which mongosh python3"], capture_output=True, timeout=30).returncode == 0
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _wsl_ok(), reason="WSL/mongosh indisponivel")
def test_real_host_hanging_query_times_out_and_others_survive(env):
    cfg = env["cfg"]
    cfg["execution"].update(per_query_timeout_s=4, batch_timeout_s=60)
    info = env["infos"][0]
    items = [
        {"id": "a", "db": info.db, "text": "db.tenants.find({ plan: 'team' }).limit(2)"},
        {"id": "hang", "db": info.db, "text": "while (true) { }"},
        {"id": "big", "db": info.db, "text": "print('x'.repeat(5000000)); db.tenants.find().limit(1)"},
        {"id": "b", "db": info.db, "text": "db.tenants.find({ plan: 'free' }).limit(2)"},
    ]
    t0 = time.time()
    out = V.run_batch((env["tmp"] / "rb", items, cfg))
    st = {i: r["status"] for i, r in out["results"].items()}
    assert st["hang"] == "timeout" and st["a"] == "ok" and st["b"] == "ok"
    assert st["big"] in ("ok", "error", "timeout")        # saida enorme: sem crash nem BSON truncado
    assert time.time() - t0 < 120
    # nenhum mongosh orfao desse lote
    ps = subprocess.run(["wsl", "-d", "Ubuntu", "-e", "bash", "-c", f"pgrep -f {V.batch_tag(env['tmp'] / 'rb')} | wc -l"], capture_output=True, text=True)
    assert ps.stdout.strip() in ("0", "1")                 # (o proprio pgrep pode aparecer)
