"""Validador: campos inexistentes (plan_fields), reescrita de db, proxy somente-leitura e execucao ponta a ponta (se Mongo/WSL disponiveis)."""
import json
import socket
import struct
import subprocess

import pytest
from bson import encode

import plan_fields
import ro_proxy
from common import load_cfg, load_schema_infos
from validate_text import rewrite_db, validate


def _info():
    infos = [i for i in load_schema_infos(roles=("train",), require_fixture=False) if i.schema_id == "tenantly"]
    if not infos:
        pytest.skip("schema tenantly ausente")
    return infos[0]


def test_unknown_field_in_find_and_pipeline():
    info = _info()
    ok = plan_fields.check_command({"find": "tenants", "filter": {"plan": "team", "settings.limits.api_rpm": {"$gt": 1}}, "sort": {"seats": -1}}, info)
    assert ok["unknown_fields"] == []
    bad = plan_fields.check_command({"find": "tenants", "filter": {"plan": "team", "plano_inexistente": 1}, "sort": {"zzz": 1}}, info)
    assert bad["unknown_fields"] == ["plano_inexistente", "zzz"]
    pipe = [{"$match": {"plan": "team"}}, {"$group": {"_id": "$plan", "total": {"$sum": "$seats"}}}, {"$sort": {"total": -1}},
            {"$project": {"total": 1}}]
    assert plan_fields.check_command({"aggregate": "tenants", "pipeline": pipe}, info)["unknown_fields"] == []
    # campo calculado em $group usado depois e valido; campo que nao existe nem foi calculado, nao
    pipe2 = [{"$group": {"_id": "$plan", "total": {"$sum": "$seats"}}}, {"$match": {"seats": {"$gt": 3}}}]
    assert plan_fields.check_command({"aggregate": "tenants", "pipeline": pipe2}, info)["unknown_fields"] == ["seats"]
    pipe3 = [{"$lookup": {"from": "users", "localField": "_id", "foreignField": "tenant_id", "as": "u"}}, {"$unwind": "$u"},
             {"$match": {"u.role": "admin"}}, {"$lookup": {"from": "nao_existe", "localField": "_id", "foreignField": "x", "as": "z"}}]
    r = plan_fields.check_command({"aggregate": "tenants", "pipeline": pipe3}, info)
    assert r["unknown_fields"] == [] and r["unknown_collections"] == ["nao_existe"]
    nested = plan_fields.check_command({"find": "tenants", "filter": {"settings.limits.nope": 1}}, info)
    assert nested["unknown_fields"] == ["settings.limits.nope"]


def test_rewrite_db():
    assert rewrite_db("db.getSiblingDB('tenantly').users.find()", "tenantly", "kapi_fx_saas_tenantly") == \
        "db.getSiblingDB('kapi_fx_saas_tenantly').users.find()"
    assert rewrite_db('db.users.find({ n: "tenantly" })', "tenantly", "kapi_fx_saas_tenantly") == 'db.users.find({ n: "tenantly" })'


def _op_msg(doc: dict, req_id=7) -> bytes:
    body = struct.pack("<I", 0) + b"\x00" + encode(doc)
    return struct.pack("<iiii", 16 + len(body), req_id, 0, ro_proxy.OP_MSG) + body


def test_proxy_blocks_writes_and_injects_max_time():
    px = ro_proxy.RoProxy(allowed_dbs={"kapi_fx_x"}, max_time_ms=3000)
    try:
        v = px.inspect(_op_msg({"insert": "c", "documents": [{"a": 1}], "$db": "kapi_fx_x"}))
        assert v[0] == "block"
        for name in ("update", "delete", "createIndexes", "drop", "dropDatabase", "create", "findAndModify", "renameCollection", "collMod"):
            assert px.inspect(_op_msg({name: "c", "$db": "kapi_fx_x"}))[0] == "block", name
        assert px.inspect(_op_msg({"find": "c", "filter": {}, "$db": "outro_banco"}))[0] == "block"
        assert px.inspect(_op_msg({"aggregate": "c", "pipeline": [{"$out": "z"}], "cursor": {}, "$db": "kapi_fx_x"}))[0] == "block"
        assert px.inspect(_op_msg({"aggregate": "c", "pipeline": [{"$merge": {"into": "z"}}], "cursor": {}, "$db": "kapi_fx_x"}))[0] == "block"
        v = px.inspect(_op_msg({"find": "c", "filter": {"a": 1}, "$db": "kapi_fx_x"}))
        assert v[0] == "pass"
        injected = v[1]
        assert b"maxTimeMS" in injected and struct.unpack_from("<i", injected, 0)[0] == len(injected)
        assert px.inspect(_op_msg({"find": "c", "filter": {}, "maxTimeMS": 600000, "$db": "kapi_fx_x"}))[0] == "block"
        assert px.stats.blocked.get("insert") == 1
    finally:
        px.close()


def _wsl_mongo_ok() -> bool:
    try:
        r = subprocess.run(["wsl", "-d", "Ubuntu", "-e", "bash", "-c", "which mongosh && python3 --version"], capture_output=True, timeout=60)
        s = socket.create_connection(("127.0.0.1", 27017), timeout=3)
        s.close()
        return r.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _wsl_mongo_ok(), reason="MongoDB/WSL indisponiveis")
def test_end_to_end_rejects_missing_field_and_writes(tmp_path):
    cfg = load_cfg()
    infos = load_schema_infos(roles=("train",))
    info = next((i for i in infos if i.schema_id == "tenantly"), None)
    if info is None:
        pytest.skip("tenantly nao semeado")
    base = {"schema_id": info.full_id, "schema_short": info.schema_id, "domain": info.domain, "db": info.db, "collection": "tenants",
            "split_role": "train", "persona": "t", "task": "t"}
    qs = [
        "db.tenants.find({ plan: 'team' }).limit(3)",                                                   # ok
        "db.tenants.find({ campo_que_nao_existe: 1 })",                                                 # executa vazio, campo inexistente
        "db.tenants.aggregate([{ $group: { _id: '$plan', n: { $sum: 1 } } }, { $sort: { n: -1 } }])",   # ok
        "db.tenants.find({ plan: { $in: ['free'] } }).sort({ nada: 1 })",                               # campo inexistente no sort
        "db.tenants.find({ plan: 'x' }).limit(2",                                                       # sintaxe
        "db.tenants.insertOne({ a: 1 })",                                                               # bloqueio estatico
    ]
    raw = tmp_path / "raw.jsonl"
    with open(raw, "w", encoding="utf-8") as f:
        for i, t in enumerate(qs):
            f.write(json.dumps({**base, "id": f"t{i}", "text": t, "qkey": f"k{i}"}) + "\n")
    cfg["execution"]["batch_size"] = 10
    cfg["execution"]["workers"] = 1
    summ = validate(raw, tmp_path / "out", cfg, infos=infos)
    acc = {json.loads(l)["id"] for l in open(tmp_path / "out" / "queries.validated.jsonl", encoding="utf-8")}
    rej = {json.loads(l)["id"]: json.loads(l)["stage"] for l in open(tmp_path / "out" / "queries.rejected.jsonl", encoding="utf-8")}
    assert acc == {"t0", "t2"}, (acc, rej)
    assert rej["t1"] in ("fields", "empty") and rej["t3"] == "fields" and rej["t4"] == "syntax" and rej["t5"] == "static"
    assert summ["readonly_proof"]["dbHash_before_equals_after"] is True
    assert summ["readonly_proof"]["proxy_write_commands_forwarded"] == {}
