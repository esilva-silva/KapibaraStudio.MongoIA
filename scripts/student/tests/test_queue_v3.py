"""Fila e configs do retreino v3: hiperparametros iguais ao sft_distill_v1, run-name distinto, inicio do zero, jobs enfileirados sem iniciar."""
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
TIERS = {"t0": "KapiCoder-Mongo-Nano", "t1": "KapiCoder-Mongo-Small"}
ALLOWED_DIFF = {("run", "name"), ("run", "arm"), ("data", "corpus_dir"), ("budget", "hours")}


def _flat(d, p=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v, p + (k,))
        else:
            yield p + (k,), v


@pytest.mark.parametrize("tier", ["t0", "t1"])
def test_sft_v3_keeps_base_hyperparameters_and_starts_from_base(tier):
    cdir = ROOT / "Repos" / TIERS[tier] / "configs"
    old = dict(_flat(yaml.safe_load((cdir / "sft_distill_v1.yaml").read_text(encoding="utf-8"))))
    new = dict(_flat(yaml.safe_load((cdir / "sft_v3.yaml").read_text(encoding="utf-8"))))
    assert set(old) == set(new)
    changed = {k for k in old if old[k] != new[k]}
    assert changed == ALLOWED_DIFF, changed
    assert new[("run", "name")] == "sft_v3" and new[("run", "arm")] == "v3" and new[("data", "corpus_dir")] == f"data/student/v3/{tier}"
    assert new[("model", "path")] == old[("model", "path")] and str(new[("model", "path")]).startswith("BaseRepos/Qwen2.5-Coder-")   # do zero, do Qwen base
    text = (cdir / "sft_v3.yaml").read_text(encoding="utf-8")
    assert "init_from" not in text and "resume_from" not in text and "adapter_path" not in text          # nunca do adapter/modelo anterior


def _jobs():
    cfg = yaml.safe_load((ROOT / "configs" / "gpu_queue.yaml").read_text(encoding="utf-8"))
    return {j["id"]: j for j in cfg["jobs"]}


def test_v3_jobs_are_queued_not_started_with_expected_dependencies():
    jobs = _jobs()
    for jid in ("J20-T0-v3", "J21-T1-v3", "J22-T0-v3-eval", "J23-T1-v3-eval"):
        assert jid in jobs, jid
        assert jobs[jid]["status"] in ("ready", "waiting")
    assert jobs["J20-T0-v3"]["depends_on"] == []
    assert jobs["J22-T0-v3-eval"]["depends_on"] == ["J20-T0-v3"]
    assert "J22-T0-v3-eval" in jobs["J21-T1-v3"]["depends_on"]                       # o T1 so depois da decisao do T0
    assert "reports/student/v3_gate_t0.go" in jobs["J21-T1-v3"]["requires_files"]    # sem GO no T0 a fila NAO inicia o T1
    assert jobs["J23-T1-v3-eval"]["depends_on"] == ["J21-T1-v3"]
    st = ROOT / "reports" / "gpu_queue" / "state.json"
    if st.exists():
        state = json.loads(st.read_text(encoding="utf-8"))
        assert not any(j in state for j in ("J20-T0-v3", "J21-T1-v3", "J22-T0-v3-eval", "J23-T1-v3-eval")), "algum job v3 ja foi iniciado"


def test_v3_job_commands_point_to_v3_configs_and_scripts():
    jobs = _jobs()
    t0, t1 = jobs["J20-T0-v3"]["command"], jobs["J21-T1-v3"]["command"]
    assert "Repos/KapiCoder-Mongo-Nano/configs/sft_v3.yaml" in t0 and "--total-steps" not in t0
    assert "Repos/KapiCoder-Mongo-Small/configs/sft_v3.yaml" in t1 and "--total-steps" in t1          # T1: corte em ~1,5 epoca
    assert int(t1[t1.index("--total-steps") + 1]) > 1000
    for jid, tier in (("J22-T0-v3-eval", "t0"), ("J23-T1-v3-eval", "t1")):
        cmd = jobs[jid]["command"]
        assert "scripts/student/eval_v3.py" in cmd and tier in cmd
        assert (ROOT / "scripts" / "student" / "eval_v3.py").exists()
    assert jobs["J20-T0-v3"]["done_marker"].endswith("merged/sft_v3/model.safetensors")
