import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detect_gpus as dg  # noqa: E402
import microbench as mb  # noqa: E402

MB = dg.load_config()["microbench"]


def test_evaluate_accepts_7800xt_reference_numbers():
    # [M] T1 7800 XT: TTFT 148 ms, 104 tok/s; CPU T0 int4 285 ms (referencia do T0)
    assert mb.evaluate("t1", {"ttft_ms_p50": 148, "decode_tok_s": 104}, {"ttft_ms_p50": 697}, MB)["accept"]
    assert mb.evaluate("t0", {"ttft_ms_p50": 109, "decode_tok_s": 166}, {"ttft_ms_p50": 285}, MB)["accept"]


def test_evaluate_rejects_slow_igpu_floor():
    r = mb.evaluate("t0", {"ttft_ms_p50": 700, "decode_tok_s": 20}, {"ttft_ms_p50": 285}, MB)
    assert not r["accept"] and len(r["reasons"]) >= 2


def test_evaluate_requires_clear_win_over_cpu():
    # GPU 260 ms passa o teto absoluto (350) mas nao ganha 30% da CPU de 285 ms
    assert not mb.evaluate("t0", {"ttft_ms_p50": 260, "decode_tok_s": 90}, {"ttft_ms_p50": 285}, MB)["accept"]


def test_evaluate_failure():
    assert not mb.evaluate("t0", {"ok": False}, None, MB)["accept"]


def test_cache_key_changes_with_driver_and_genai():
    ad = {"vendor_id": 0x1002, "device_id": 0x747E, "subsys_id": 1, "revision": 0xC8}
    k = mb.cache_key(ad, "32.0.31041.1004", "0.15.2", "abc", "26100")
    assert k == mb.cache_key(ad, "32.0.31041.1004", "0.15.2", "abc", "26100")
    assert k != mb.cache_key(ad, "32.0.31042.1", "0.15.2", "abc", "26100")
    assert k != mb.cache_key(ad, "32.0.31041.1004", "0.17.1", "abc", "26100")


# --- fase 2: calibragem com dados reais (reports/hwmap/phase2_measurements.md §C) ---
def test_phase2_measured_7800xt_accepted_and_igpu_rejected():
    cpu_t0 = {"ttft_ms_p50": 231}   # baseline unico: T0 int4 na CPU do mesmo PC
    assert mb.evaluate("t0", {"ttft_ms_p50": 127, "decode_tok_s": 124}, cpu_t0, MB)["accept"]      # pior execucao boa da 7800 XT
    assert mb.evaluate("t1", {"ttft_ms_p50": 181, "decode_tok_s": 79.4}, cpu_t0, MB)["accept"]     # T1 7800 XT (170/231 = 0,74 > 0,7: precisa do teto proprio do T1)
    assert not mb.evaluate("t0", {"ttft_ms_p50": 1347, "decode_tok_s": 21}, cpu_t0, MB)["accept"]  # iGPU 2 CU
    assert not mb.evaluate("t1", {"ttft_ms_p50": 3488, "decode_tok_s": 8.1}, cpu_t0, MB)["accept"]


def test_phase2_thresholds_have_margin_over_worst_good_run():
    assert MB["accept_ttft_ms_p50_max"] >= 1.5 * 127 and MB["t1_accept_ttft_ms_p50_max"] >= 1.3 * 181
    assert MB["accept_decode_tok_s_min"] <= 124 / 2 + 1 and MB["t1_accept_decode_tok_s_min"] <= 79 / 2 + 1
