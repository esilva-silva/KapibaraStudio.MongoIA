import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detect_gpus as dg  # noqa: E402
import microbench as mb  # noqa: E402

MB = dg.load_config()["microbench"]


def test_evaluate_accepts_7800xt_reference_numbers():
    # [M] T1 7800 XT: TTFT 148 ms, 104 tok/s; CPU T0 int4 285 ms (referencia do T0)
    assert mb.evaluate("t1", {"ttft_ms_p50": 148, "decode_tok_s": 104, "adapter_verified": True}, {"ttft_ms_p50": 697}, MB)["accept"]
    assert mb.evaluate("t0", {"ttft_ms_p50": 109, "decode_tok_s": 166, "adapter_verified": True}, {"ttft_ms_p50": 285}, MB)["accept"]


def test_evaluate_rejects_slow_igpu_floor():
    r = mb.evaluate("t0", {"ttft_ms_p50": 700, "decode_tok_s": 20, "adapter_verified": True}, {"ttft_ms_p50": 285}, MB)
    assert not r["accept"] and len(r["reasons"]) >= 2


def test_evaluate_requires_clear_win_over_cpu():
    # GPU 260 ms passa o teto absoluto (350) mas nao ganha 30% da CPU de 285 ms
    assert not mb.evaluate("t0", {"ttft_ms_p50": 260, "decode_tok_s": 90, "adapter_verified": True}, {"ttft_ms_p50": 285}, MB)["accept"]


def test_evaluate_failure():
    assert not mb.evaluate("t0", {"ok": False}, None, MB)["accept"]


def test_cache_key_changes_with_driver_and_genai():
    ad = {"vendor_id": 0x1002, "device_id": 0x747E, "subsys_id": 1, "revision": 0xC8}
    k = mb.cache_key(ad, "32.0.31041.1004", "0.15.2", "abc", "26100")
    assert k == mb.cache_key(ad, "32.0.31041.1004", "0.15.2", "abc", "26100")
    assert k != mb.cache_key(ad, "32.0.31042.1", "0.15.2", "abc", "26100")
    assert k != mb.cache_key(ad, "32.0.31041.1004", "0.17.1", "abc", "26100")


# --- fase 2: calibragem com dados reais (reports/hwmap/phase2_measurements.md Â§C) ---
def test_phase2_measured_7800xt_accepted_and_igpu_rejected():
    cpu_t0 = {"ttft_ms_p50": 231}   # baseline unico: T0 int4 na CPU do mesmo PC
    assert mb.evaluate("t0", {"ttft_ms_p50": 127, "decode_tok_s": 124, "adapter_verified": True}, cpu_t0, MB)["accept"]      # pior execucao boa da 7800 XT
    assert mb.evaluate("t1", {"ttft_ms_p50": 181, "decode_tok_s": 79.4, "adapter_verified": True}, cpu_t0, MB)["accept"]     # T1 7800 XT (170/231 = 0,74 > 0,7: precisa do teto proprio do T1)
    assert not mb.evaluate("t0", {"ttft_ms_p50": 1347, "decode_tok_s": 21, "adapter_verified": True}, cpu_t0, MB)["accept"]  # iGPU 2 CU
    assert not mb.evaluate("t1", {"ttft_ms_p50": 3488, "decode_tok_s": 8.1, "adapter_verified": True}, cpu_t0, MB)["accept"]


def test_phase2_thresholds_have_margin_over_worst_good_run():
    assert MB["accept_ttft_ms_p50_max"] >= 1.5 * 127 and MB["t1_accept_ttft_ms_p50_max"] >= 1.3 * 181
    assert MB["accept_decode_tok_s_min"] <= 124 / 2 + 1 and MB["t1_accept_decode_tok_s_min"] <= 79 / 2 + 1


def test_warmup_limit_rejects_only_above_four_times_ceiling():
    for tier, key in (("t0", "accept_ttft_ms_p50_max"), ("t1", "t1_accept_ttft_ms_p50_max")):
        limit = MB[key] * MB["abort_warmup_ttft_factor"]
        assert not mb.warmup_rejected(tier, limit, MB)
        assert mb.warmup_rejected(tier, limit + 0.01, MB)
        assert mb.warmup_rejected(tier, float("nan"), MB)


def test_missing_vram_counter_is_not_adapter_verification():
    result = mb.evaluate("t0", {"ttft_ms_p50": 100, "decode_tok_s": 100, "adapter_verified": False}, {"ttft_ms_p50": 300}, MB)
    assert not result["accept"]
    assert "contador" in result["reasons"][0]


@pytest.mark.parametrize("tier", ["t0", "t1"])
@pytest.mark.parametrize("key", ["ttft_ms_p50", "decode_tok_s"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf"), 0, -1, True, "100"])
def test_invalid_required_gpu_metrics_never_pass(tier, key, value):
    gpu = {"ttft_ms_p50": 100, "decode_tok_s": 100, "adapter_verified": True}
    gpu[key] = value
    result = mb.evaluate(tier, gpu, {"ttft_ms_p50": 300}, MB)
    assert not result["accept"]
    assert any(key in reason for reason in result["reasons"])


@pytest.mark.parametrize("tier", ["t0", "t1"])
@pytest.mark.parametrize("cpu", [None, {}, {"ttft_ms_p50": None}, {"ttft_ms_p50": float("nan")},
                                 {"ttft_ms_p50": float("inf")}, {"ttft_ms_p50": 0}, {"ttft_ms_p50": -1},
                                 {"ttft_ms_p50": True}, {"ttft_ms_p50": "300"}, {"ttft_ms_p50": 300, "ok": False}])
def test_cpu_baseline_is_required_and_must_be_positive_and_finite(tier, cpu):
    result = mb.evaluate(tier, {"ttft_ms_p50": 100, "decode_tok_s": 100, "adapter_verified": True}, cpu, MB)
    assert not result["accept"]
    assert any("baseline CPU" in reason for reason in result["reasons"])


@pytest.mark.parametrize("key", ["ttft_ms_p50", "decode_tok_s"])
def test_missing_gpu_metric_never_passes(key):
    gpu = {"ttft_ms_p50": 100, "decode_tok_s": 100, "adapter_verified": True}
    del gpu[key]
    assert not mb.evaluate("t0", gpu, {"ttft_ms_p50": 300}, MB)["accept"]


@pytest.mark.parametrize("proof", [None, False, 1, "true"])
def test_adapter_verification_must_be_the_boolean_true(proof):
    gpu = {"ttft_ms_p50": 100, "decode_tok_s": 100, "adapter_verified": proof}
    result = mb.evaluate("t0", gpu, {"ttft_ms_p50": 300}, MB)
    assert not result["accept"]
    assert "contador" in result["reasons"][0]


def test_missing_adapter_proof_never_passes():
    result = mb.evaluate("t0", {"ttft_ms_p50": 100, "decode_tok_s": 100}, {"ttft_ms_p50": 300}, MB)
    assert not result["accept"]
    assert "contador" in result["reasons"][0]


def test_timing_only_executor_does_not_claim_adapter_verification(monkeypatch):
    class Config:
        def __init__(self, pkg):
            pass

        def clear_providers(self):
            pass

        def append_provider(self, provider):
            pass

    class Params:
        def __init__(self, model):
            pass

        def set_search_options(self, **options):
            pass

    class Generator:
        def __init__(self, model, params):
            pass

        def append_tokens(self, ids):
            pass

        def generate_next_token(self):
            pass

        def is_done(self):
            return True

    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(array=lambda ids, **kwargs: ids, int32=int))
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", SimpleNamespace(
        Config=Config, Model=lambda cfg: object(), Tokenizer=lambda model: SimpleNamespace(encode=lambda text: [1]),
        GeneratorParams=Params, Generator=Generator))
    monkeypatch.setattr(mb, "warmup_rejected", lambda *args: False)
    result = mb.run_candidate("synthetic-package", "dml", prompt_tokens=1, decode_tokens=2, repeats=1)
    assert result["ok"]
    assert result["adapter_verified"] is False
