import copy
import json
from pathlib import Path

import pytest
import yaml

import check_gates as cg
import compare_models as cm
import gates_to_ide_json as g2i

ROOT = Path(__file__).resolve().parents[3]
GATES = ROOT / "configs" / "gates.yaml"
FIX = Path(__file__).parent / "fixtures"


def fx(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def run(py, kl, tier="T0", gates=GATES, tmp=None):
    def w(obj, n):
        p = tmp / n
        p.write_text(json.dumps(obj), encoding="utf-8")
        return str(p)

    args = ["--tier", tier, "--gates", str(gates), "--measures", w(py, "py.json"),
            "--out", str(tmp / "out" / "gates.json")]
    if kl is not None:
        args += ["--measures-kapilab", w(kl, "kl.json")]
    code = cg.main(args)
    return code, json.loads((tmp / "out" / "gates.json").read_text(encoding="utf-8"))


def by_id(doc):
    return {g["id"]: g for g in doc["gates"]}


def test_real_yaml_todo_blocks_approval(tmp_path):
    # o gate todo de paridade tokenizer/prompt/greedy (sem limiar na doc) impede aprovar: exit 7
    code, doc = run(fx("measures_T0_pass.json"), fx("measures_T0_kapilab.json"), tmp=tmp_path)
    assert code == 7
    assert doc["blocking_unevaluated"] == ["common.parity.tokenizer_prompt_greedy"]
    assert doc["blocking_failed"] == []


def _gates_without_todo(tmp_path):
    cfg = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    cfg["common"]["gates"] = [g for g in cfg["common"]["gates"] if not g.get("todo")]
    p = tmp_path / "gates_notodo.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_all_pass_exit_0(tmp_path):
    gp = _gates_without_todo(tmp_path)
    code, doc = run(fx("measures_T0_pass.json"), fx("measures_T0_kapilab.json"), gates=gp, tmp=tmp_path)
    assert code == 0, doc["blocking_failed"] + doc["blocking_unevaluated"]
    assert doc["approved"] and doc["qualified"]
    g = by_id(doc)
    assert g["T0.apt_vs_base"]["status"] == "passed"
    assert g["T0.cpu.ttft_mean_ms"]["owner"] == "kapilab"
    assert g["common.harness.effective_provider"]["status"] == "passed"
    assert g["common.parity.int4.delta_apt"]["status"] == "not_applicable"
    assert doc["schema"] == "kapi-gates-v1"


def test_fail_exit_6(tmp_path):
    py = fx("measures_T0_pass.json")
    py["metrics"]["syntax_valid"] = 0.90  # < 0.92
    code, doc = run(py, fx("measures_T0_kapilab.json"), gates=_gates_without_todo(tmp_path), tmp=tmp_path)
    assert code == 6
    assert "T0.syntax_valid" in doc["blocking_failed"]
    assert by_id(doc)["T0.syntax_valid"]["passed"] is False


def test_noninferior_ci_and_group_all(tmp_path):
    py = fx("measures_T0_pass.json")
    py["metrics"]["T0.apt_vs_base"] = {"delta": 0.0, "ci_low": -0.06, "ci_high": 0.05}  # < -0.05
    code, doc = run(py, fx("measures_T0_kapilab.json"), tmp=tmp_path)
    assert code == 6
    grp = {g["group"]: g for g in doc["groups"]}
    assert grp["T0.apt_vs_base_all"]["status"] == "failed"
    assert "group:T0.apt_vs_base_all" in doc["blocking_failed"]


def test_unevaluated_exit_7_never_passes(tmp_path):
    py = fx("measures_T0_pass.json")
    del py["metrics"]["syntax_valid"]
    code, doc = run(py, fx("measures_T0_kapilab.json"), gates=_gates_without_todo(tmp_path), tmp=tmp_path)
    assert code == 7
    r = by_id(doc)["T0.syntax_valid"]
    assert r["status"] == "unevaluated" and r["passed"] is None
    assert not doc["approved"]


def test_no_kapilab_measures_is_unevaluated(tmp_path):
    code, doc = run(fx("measures_T0_pass.json"), None, tmp=tmp_path)
    assert code == 7
    assert by_id(doc)["T0.cpu.ttft_mean_ms"]["status"] == "unevaluated"


def test_python_only_owners(tmp_path):
    p = tmp_path / "py.json"
    p.write_text(json.dumps(fx("measures_T0_pass.json")), encoding="utf-8")
    code = cg.main(["--tier", "T0", "--gates", str(_gates_without_todo(tmp_path)), "--measures", str(p), "--owners", "python",
                    "--out", str(tmp_path / "o" / "g.json")])
    assert code == 0


def test_todo_and_null_threshold_unevaluated(tmp_path):
    cfg = {"tiers": {"TX": {"gates": [
        {"id": "TX.todo", "metric": "m", "op": "==", "threshold": None, "owner": "python", "blocking": True, "todo": True},
        {"id": "TX.nullthr", "metric": "n", "op": ">=", "threshold": None, "owner": "python", "blocking": False},
        {"id": "TX.ok", "metric": "o", "op": ">=", "threshold": 1, "owner": "python", "blocking": True}]}},
        "common": {"gates": []}}
    gp = tmp_path / "g.yaml"
    gp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    code, doc = run({"metrics": {"m": 0, "n": 5, "o": 2}}, None, tier="TX", gates=gp, tmp=tmp_path)
    g = by_id(doc)
    assert g["TX.todo"]["status"] == "unevaluated"
    assert g["TX.nullthr"]["status"] == "unevaluated"
    assert g["TX.ok"]["status"] == "passed"
    assert code == 7  # so o bloqueante todo conta


def test_inactive_ignored(tmp_path):
    cfg = {"tiers": {"TX": {"gates": [
        {"id": "TX.dead", "metric": "m", "op": "<=", "threshold": 1, "owner": "python", "blocking": True, "inactive": True},
        {"id": "TX.ok", "metric": "o", "op": ">=", "threshold": 1, "owner": "python", "blocking": True}]}},
        "common": {"gates": []}}
    gp = tmp_path / "g.yaml"
    gp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    code, doc = run({"metrics": {"m": 999, "o": 2}}, None, tier="TX", gates=gp, tmp=tmp_path)
    assert code == 0
    assert by_id(doc)["TX.dead"]["status"] == "inactive"


@pytest.mark.parametrize("env,reason", [
    ({"used_fallback": True}, "used_fallback"),
    ({"allow_gpu_parallel": True}, "allow_gpu_parallel"),
    ({"machine_load_pct": 30}, "machine_not_idle"),
    ({"genai": "0.17.1", "package_min_genai": "0.15.2"}, "genai_mismatch"),
])
def test_qualified_false(tmp_path, env, reason):
    kl = copy.deepcopy(fx("measures_T0_kapilab.json"))
    kl["env"].update(env)
    code, doc = run(fx("measures_T0_pass.json"), kl, tmp=tmp_path)
    assert doc["qualified"] is False
    assert any(reason in r for r in doc["qualified_reasons"])
    assert by_id(doc)["T0.cpu.ttft_mean_ms"]["status"] == "unqualified"
    assert by_id(doc)["T0.syntax_valid"]["status"] == "passed"  # python nao e afetado
    assert code == 7 and not doc["approved"]


def test_superior_holm(tmp_path):
    cfg = {"statistics": {"superiority": {"alpha": 0.05}}, "tiers": {"TX": {"gates": [
        {"id": "TX.a", "metric": "pa", "op": "superior", "threshold": 0.05, "unit": "p", "owner": "python", "blocking": True},
        {"id": "TX.b", "metric": "pb", "op": "superior", "threshold": 0.05, "unit": "p", "owner": "python", "blocking": True}]}},
        "common": {"gates": []}}
    gp = tmp_path / "g.yaml"
    gp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    # 0.03 e 0.04: bruto passaria os dois; Holm ajusta (0.06) -> reprova
    code, doc = run({"metrics": {"pa": 0.03, "pb": 0.04}}, None, tier="TX", gates=gp, tmp=tmp_path)
    assert code == 6
    code, doc = run({"metrics": {"pa": 0.01, "pb": 0.02}}, None, tier="TX", gates=gp, tmp=tmp_path)
    assert code == 0


def test_any_group(tmp_path):
    py = fx("measures_T0_pass.json")
    kl = fx("measures_T0_kapilab.json")
    cfg = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    code, doc = run(py, kl, tier="T2", tmp=tmp_path)  # T2: nenhuma medida python -> grupo any nao avaliado
    grp = {g["group"]: g for g in doc["groups"]}
    assert grp["T2.gain_vs_t1_any"]["status"] == "unevaluated"
    assert by_id(doc)["T2.14B.dml_int4.ttft_mean_ms"]["status"] == "inactive"
    assert code == 7


def test_real_yaml_loads_all_tiers():
    for t in ("T0", "T1", "T2", "T3", "T4"):
        assert cg.load_gates(GATES, t)


def test_ide_json_only_kapilab(tmp_path):
    raw = GATES.read_bytes()
    doc = g2i.build(yaml.safe_load(raw), raw)
    ids = [g["id"] for t in doc["tiers"].values() for s in t["scenarios"].values() for g in s] + \
          [g["id"] for g in doc["common"]]
    cfg = yaml.safe_load(raw)
    allg = [g for t in cfg["tiers"].values() for g in t["gates"]] + cfg["common"]["gates"] + cfg["common"]["parity"]
    kap = {g["id"]: g for g in allg if g.get("owner") == "kapilab"}
    assert ids and all(i in kap for i in ids)
    assert all(not kap[i].get("inactive") for i in ids)
    assert "T2.14B.dml_int4.ttft_mean_ms" in doc["excluded"]["inactive"]
    assert "T0.cpu.ttft_p95_ms" in doc["tiers"]["T0"]["scenarios"]["T0.cpu"][1]["id"]
    for g in doc["common"]:
        assert set(g) == {"id", "metric", "op", "threshold"}


def test_compare_models_aligns_and_groups(tmp_path):
    def rec(i, ok, grp, ttft):
        return {"id": f"s{i}", "bench_group": grp, "ttft_ms": ttft,
                "metrics": {"exact_match": ok, "edit_distance": 10 - 5 * ok}}
    A = [rec(i, i % 4 == 0, "g1" if i < 60 else "g2", 10.0 + i % 7) for i in range(80)]
    B = [rec(i, i % 2 == 0, "g1" if i < 60 else "g2", 9.0 + i % 7) for i in range(10, 90)]
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    pa.write_text("\n".join(json.dumps(r) for r in A), encoding="utf-8")
    pb.write_text("\n".join(json.dumps(r) for r in B), encoding="utf-8")
    res = cm.run(pa, pb, n_resamples=300)
    assert res["n_aligned"] == 70 and res["only_in_a"] == 10 and res["only_in_b"] == 10
    em = next(r for r in res["overall"] if r["metric"] == "exact_match")
    assert em["better"] + em["worse"] + em["equal"] == 70
    assert em["better"] > em["worse"] and "p_holm" in em
    assert set(res["groups"]) == {"g1"}  # g2 tem n < 50 (so 10 alinhados)
    assert "g2" in res["groups_skipped"]
    assert "ttft_ms" in {r["metric"] for r in res["overall"]}
    assert "| exact_match |" in cm.to_markdown(res)
