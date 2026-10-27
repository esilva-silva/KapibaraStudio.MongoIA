"""Regras de decisao do retreino v3 (decide_v3.decide): GO / INCONCLUSIVO / SEM_MELHORA / REJEITADO e o portao do T1."""
import copy
import json
from pathlib import Path

import pytest

import decide_v3 as dv


def _model(fim, eoc, full_fields):
    return {"overall": {"normalized_match": fim, "exec_over_ceiling": eoc, "nonexistent_fields_rate": 0.1},
            "by_context_kind": {"full": {"nonexistent_fields_rate": full_fields}}}


def _hom(tier="t0", cur_full=0.1535, new_full=0.08, ci_low=0.03, fim_new=0.335, eoc_new=0.75):
    base = dv.BASE_KEY[tier]
    return {"models": {base: _model(0.126, 0.395, 0.07), f"{tier}_distill": _model(0.331, 0.744, cur_full), f"{tier}_v3": _model(fim_new, eoc_new, new_full)},
            "pairs": {f"{tier}_v3_vs_{tier}_distill": {"context_kind": {
                "overall": [{"metric": "normalized_match", "delta": 0.004, "ci_low": -0.004, "ci_high": 0.01},
                            {"metric": "exec_valid", "delta": 0.01, "ci_low": 0.0, "ci_high": 0.02}],
                "groups": {"full": {"metrics": [{"metric": "no_nonexistent_field", "delta": 0.07, "ci_low": ci_low, "ci_high": 0.1}]}}}}}}


def _summ(fim, ex, ceil=0.6255):
    return {"headline": {"normalized_match": fim, "exec_valid": ex}, "reference_ceiling": {"exec_valid": ceil}}


def _gen(v):
    return {"headline": {"normalized_match": v}}


VAL = {"current": _summ(0.403, 0.582), "new": _summ(0.40, 0.58)}
GEN = {"new": _gen(0.50), "current": _gen(0.518), "base": _gen(0.34)}


def test_go_when_fields_drop_confirmed_and_guards_hold():
    d = dv.decide("t0", _hom(), VAL, GEN)
    assert d["verdict"] == "GO" and d["go"] and all(d["checks"].values())
    assert d["numbers"]["full_nonexistent_pct"]["v3"] == pytest.approx(8.0)


def test_ideal_flag_uses_base_plus_one_pp():
    assert dv.decide("t0", _hom(new_full=0.075), VAL, GEN)["ideal_met"] is True
    assert dv.decide("t0", _hom(new_full=0.10), VAL, GEN)["ideal_met"] is False


def test_inconclusive_when_ci_includes_zero():
    d = dv.decide("t0", _hom(ci_low=-0.01), VAL, GEN)
    assert d["verdict"] == "INCONCLUSIVO" and not d["go"]


def test_no_improvement_when_fields_do_not_drop():
    d = dv.decide("t1", _hom("t1", cur_full=0.0919, new_full=0.095, ci_low=-0.02), VAL, GEN)
    assert d["verdict"] == "SEM_MELHORA" and not d["go"] and "NAO gastar mais GPU" in d["action"]


def test_rejected_when_hom_fim_or_exec_loses_more_than_one_pp():
    assert dv.decide("t0", _hom(fim_new=0.320), VAL, GEN)["verdict"] == "REJEITADO"          # -1,1 pp de FIM acc
    assert dv.decide("t0", _hom(eoc_new=0.73), VAL, GEN)["verdict"] == "REJEITADO"           # -1,4 pp de exec / teto
    assert dv.decide("t0", _hom(fim_new=0.322), VAL, GEN)["checks"]["C3_hom_guardrail_loss_le_1pp"] is True   # -0,9 pp: dentro


def test_rejected_on_val_or_generic_regression():
    bad_val = {"current": _summ(0.403, 0.582), "new": _summ(0.37, 0.58)}
    assert dv.decide("t0", _hom(), bad_val, GEN)["verdict"] == "REJEITADO"
    bad_gen = {"new": _gen(0.30), "current": _gen(0.518), "base": _gen(0.34)}               # -4 pp vs zero-shot (>3)
    assert dv.decide("t0", _hom(), VAL, bad_gen)["verdict"] == "REJEITADO"


def test_missing_model_aborts():
    h = _hom()
    del h["models"]["t0_v3"]
    with pytest.raises(SystemExit):
        dv.decide("t0", h, VAL, GEN)


def test_cli_writes_gate_only_on_go(tmp_path, monkeypatch):
    hom = tmp_path / "hom.json"
    ev = tmp_path / "Repos" / "KapiCoder-Mongo-Nano" / "reports" / "eval"
    for name, s in (("sft_distill_v1", VAL["current"]), ("sft_v3", VAL["new"])):
        (ev / name).mkdir(parents=True)
        (ev / name / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    for name, g in (("sft_v3", GEN["new"]), ("sft_distill_v1", GEN["current"]), (dv.BASE_RUN["t0"], GEN["base"])):
        (ev / f"{name}-generic-js").mkdir(parents=True)
        (ev / f"{name}-generic-js" / "summary.json").write_text(json.dumps(g), encoding="utf-8")
    monkeypatch.setattr(dv, "ROOT", tmp_path)
    out = tmp_path / "out"
    hom.write_text(json.dumps(_hom()), encoding="utf-8")
    assert dv.main(["--tier", "t0", "--hom-json", str(hom), "--out-dir", str(out)]) == 0
    assert (out / "v3_gate_t0.go").exists() and json.loads((out / "v3_decision_t0.json").read_text())["verdict"] == "GO"
    hom.write_text(json.dumps(_hom(new_full=0.16, ci_low=-0.05)), encoding="utf-8")
    dv.main(["--tier", "t0", "--hom-json", str(hom), "--out-dir", str(out)])
    assert not (out / "v3_gate_t0.go").exists()          # o portao some se uma reavaliacao nao for GO


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_required_generic_metric_never_approves(value):
    gen = copy.deepcopy(GEN)
    gen["new"]["headline"]["normalized_match"] = value
    with pytest.raises(ValueError, match="obrigatoria"):
        dv.decide("t0", _hom(), VAL, gen)


def test_missing_generic_or_ceiling_never_approves():
    with pytest.raises(ValueError):
        dv.decide("t0", _hom(), VAL, {})
    val = copy.deepcopy(VAL)
    val["new"]["reference_ceiling"] = {}
    with pytest.raises(ValueError):
        dv.decide("t0", _hom(), val, GEN)


def test_exact_guard_boundaries_and_explicit_gain_sign():
    d = dv.decide("t0", _hom(fim_new=0.321, eoc_new=0.734), VAL, GEN)
    assert d["checks"]["C3_hom_guardrail_loss_le_1pp"]
    improved = dv.decide("t0", _hom(fim_new=0.4234, eoc_new=0.8617), VAL, GEN)
    assert improved["numbers"]["comparisons"]["hom_fim_acc"]["delta_pp"] == pytest.approx(9.24)
    assert "positivo significa ganho" in dv.render_md(improved)


@pytest.mark.parametrize("changes", [{"ci_high": None}, {"ci_low": True}, {"ci_high": 0.01}, {"delta": float("nan")}])
def test_incomplete_or_invalid_paired_interval_is_not_confirmed(changes):
    hom = _hom()
    metric = hom["pairs"]["t0_v3_vs_t0_distill"]["context_kind"]["groups"]["full"]["metrics"][0]
    metric.update(changes)
    result = dv.decide("t0", hom, VAL, GEN)
    assert not result["checks"]["C2_improvement_confirmed_ci95"]
    assert not result["go"]
