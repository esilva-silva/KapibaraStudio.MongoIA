"""train_student sem GPU: plano de epocas determinista, cobertura das linhas, fingerprint, configs pareadas, agenda de LR."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "data" / "student" / "distill" / "t0"
needs_corpus = pytest.mark.skipif(not (CORPUS / "manifest.json").exists(), reason="corpus dos alunos ausente (build_student_corpus.py)")

sys.path.insert(0, str(ROOT / "scripts" / "teacher"))


@needs_corpus
def test_epoch_plan_is_deterministic_and_covers_all_rows():
    import train_student as ts

    a = ts.EpochCorpus(CORPUS, None, 2, 2, 20261018)
    b = ts.EpochCorpus(CORPUS, None, 2, 2, 20261018)
    assert a.total_steps == 2 * a.steps_per_epoch
    n = a.steps_per_epoch * 2
    first = [a.plan_index(s, k) for s in range(a.steps_per_epoch) for k in range(2)]
    second = [a.plan_index(a.steps_per_epoch + s, k) for s in range(a.steps_per_epoch) for k in range(2)]
    assert first == list(range(n))                               # epoca 1 = ordem do manifesto (ja embaralhada)
    assert sorted(second) == list(range(n)) and second != first   # epoca 2 = permutacao das mesmas linhas
    assert second == [b.plan_index(b.steps_per_epoch + s, k) for s in range(b.steps_per_epoch) for k in range(2)]


@needs_corpus
def test_distill_and_gold_have_the_same_plan():
    import train_student as ts

    gold = CORPUS.parents[1] / "gold" / "t0"
    if not (gold / "manifest.json").exists():
        pytest.skip("gold ausente")
    d, g = ts.EpochCorpus(CORPUS, None, 2, 2, 1), ts.EpochCorpus(gold, None, 2, 2, 1)
    assert d.steps_per_epoch == g.steps_per_epoch and d.rows == g.rows
    step = 5000 % d.total_steps
    assert [d.plan_index(step, k) for k in range(2)] == [g.plan_index(step, k) for k in range(2)]


@needs_corpus
def test_fingerprint_changes_with_hyperparameters():
    import train_student as ts
    from train_teacher import load_config

    path = ROOT / "Repos" / "KapiCoder-Mongo-Nano" / "configs" / "sft_distill_v1.yaml"
    cfg = load_config(path, [])
    corpus = ts.EpochCorpus(CORPUS, None, 2, 2, cfg["run"]["seed"])
    fp1 = ts.fingerprint(cfg, corpus, "m", 2)
    cfg2 = load_config(path, ["optim.lr=3.0e-5"])
    assert ts.fingerprint(cfg2, corpus, "m", 2)["combined"] != fp1["combined"]
    assert ts.fingerprint(cfg, corpus, "m", 2)["combined"] == fp1["combined"]


def test_arm_configs_differ_only_in_name_arm_and_corpus():
    import yaml

    for repo in ("KapiCoder-Mongo-Nano", "KapiCoder-Mongo-Small"):
        c = {a: yaml.safe_load((ROOT / "Repos" / repo / "configs" / f"sft_{a}_v1.yaml").read_text(encoding="utf-8")) for a in ("distill", "gold")}
        assert c["distill"]["run"]["arm"] == "distill" and c["gold"]["run"]["arm"] == "gold"
        for cfg in c.values():
            cfg["run"].pop("name"), cfg["run"].pop("arm"), cfg["data"].pop("corpus_dir")
        assert c["distill"] == c["gold"], repo


def test_lr_schedule_is_cosine_with_warmup():
    from train_teacher import lr_at

    o = {"lr": 1e-4, "warmup_ratio": 0.03, "min_lr_ratio": 0.1}
    assert lr_at(0, 1000, o) < lr_at(29, 1000, o) <= 1e-4
    assert abs(lr_at(30, 1000, o) - 1e-4) < 1e-9
    assert abs(lr_at(999, 1000, o) - 1e-5) < 2e-7


def test_measures_build_unevaluated_without_v1_baseline_and_never_approve():
    """eval_student: sem o baseline v1 re-medido, os gates de APT/exec-match ficam ausentes (unevaluated), nunca aprovados."""
    sys.path.insert(0, str(ROOT / "scripts" / "student"))
    import eval_student as es

    row = {"id": "x", "cut_kind": "key", "domain": "d", "context_kind": "full", "ttft_ms": 1.0, "generated_tokens": 3, "prediction": "name",
           "metrics": {"normalized_match": True, "exact_match": True, "accepted_prefix_tokens": 1.0, "edit_similarity": 1.0, "syntax_valid": True,
                       "mongo_valid": True, "empty": False}, "exec": {"exec_valid": True, "field_hallucination": False}}
    summary = {"n": 1, "run_name": "t", "dataset": {"sample_ids_sha256": "s"}, "headline": {"syntax_valid": 1.0, "mongo_valid": 1.0, "nonexistent_fields_rate": 0.0,
               "normalized_match": 1.0, "exact_match": 1.0, "accepted_prefix_tokens": 1.0, "exec_valid": 1.0, "empty_rate": 0.0},
               "reference_ceiling": {"exec_valid": 1.0}, "exec_score": {"write_commands_seen_by_guard": []}}
    m = es.build_measures(summary, [row], tier="t0")["metrics"]
    assert m["syntax_valid"] == 1.0 and m["exec_valid_vs_ceiling"] == 1.0 and m["write_or_destructive_executions"] == 0
    assert "T0.apt_vs_base" not in m and "exec_match_autocomplete" not in m and "apt_delta_per_group_min" not in m
