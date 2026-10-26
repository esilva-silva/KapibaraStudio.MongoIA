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
