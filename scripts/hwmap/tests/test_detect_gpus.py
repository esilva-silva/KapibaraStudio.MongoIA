"""Testes de detect_gpus.py com saidas simuladas (nao tocam o hardware). Rodar:
  env\\.venv\\Scripts\\python.exe -m pytest scripts/hwmap/tests -q -p no:cacheprovider
Os device_id simulados sao [E] (ver configs/gpu_compat.yaml); os de 7800 XT / iGPU Raphael sao [M]."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detect_gpus as dg  # noqa: E402

CFG = dg.load_config()
CPU12 = {"physical_cores": 12, "ram_gb": 32.0}
CPU4 = {"physical_cores": 4, "ram_gb": 8.0}


def ad(name, ven, dev, vram, shared=16000, software=False, luid="0x00000000_0x00000001"):
    return {"name": name, "vendor_id": ven, "device_id": dev, "dedicated_vram_mb": vram, "shared_mb": shared,
            "is_software": software, "luid": luid, "driver_version": "1.0"}


WARP = ad("Microsoft Basic Render Driver", 0x1414, 0x8C, 0, software=True, luid="0x00000000_0x000000ff")

# nome, adaptador, variante esperada, necessita micro-benchmark?
CASES = [
    ("RTX 3060", ad("NVIDIA GeForce RTX 3060", 0x10DE, 0x2504, 12288), "t1-dml-fp16", False),
    ("GTX 1060 6GB (Pascal)", ad("NVIDIA GeForce GTX 1060 6GB", 0x10DE, 0x1C03, 6144), "t1-dml-fp16", True),
    ("GTX 1650 4GB", ad("NVIDIA GeForce GTX 1650", 0x10DE, 0x1F82, 4096), "t0-dml-fp16", False),
    ("RX 580 8GB (Polaris)", ad("AMD Radeon RX 580 Series", 0x1002, 0x67DF, 8192), "t1-dml-fp16", True),
    ("RX Vega 56", ad("Radeon RX Vega", 0x1002, 0x687F, 8176), "t1-dml-fp16", False),
    ("RX 6600", ad("AMD Radeon RX 6600", 0x1002, 0x73FF, 8176), "t1-dml-fp16", False),
    ("RX 7800 XT", ad("AMD Radeon RX 7800 XT", 0x1002, 0x747E, 16177), "t1-dml-fp16", False),
    ("Arc A770", ad("Intel(R) Arc(TM) A770 Graphics", 0x8086, 0x56A0, 16288), "t1-dml-fp16", True),
    ("Arc B580", ad("Intel(R) Arc(TM) B580 Graphics", 0x8086, 0xE20B, 12200), "t1-dml-fp16", True),
]
IGPU_CASES = [
    # iGPU sozinha: Vega 8, UHD 630 e Iris Xe ficam na CPU; 780M e Core Ultra sao candidatas GPU com micro-benchmark
    ("iGPU Vega 8", ad("AMD Radeon(TM) Graphics", 0x1002, 0x1638, 512), "t0-cpu-int4", False),
    ("iGPU 780M", ad("AMD Radeon(TM) 780M Graphics", 0x1002, 0x15BF, 512), "t0-dml-fp16", True),
    ("UHD 630", ad("Intel(R) UHD Graphics 630", 0x8086, 0x3E92, 128), "t0-cpu-int4", False),
    ("Iris Xe", ad("Intel(R) Iris(R) Xe Graphics", 0x8086, 0x9A49, 128), "t0-cpu-int4", False),
    ("Core Ultra iGPU", ad("Intel(R) Arc(TM) Graphics", 0x8086, 0x7D55, 128), "t0-dml-fp16", True),
    ("Basic Render Driver", WARP, "t0-cpu-int4", False),
]


@pytest.mark.parametrize("label,a,variant,mb", CASES + IGPU_CASES, ids=[c[0] for c in CASES + IGPU_CASES])
def test_single_adapter(label, a, variant, mb):
    r = dg.decide([a, WARP] if a is not WARP else [a], CPU12, CFG)
    assert r["recommended_variant"] == variant, r["reason"]
    assert r["needs_microbench"] is mb
    assert r["fallback_chain"][-1] == "t0-cpu-int4"
    assert r["confidence"] in ("alta", "media", "baixa")
    if variant != "t0-cpu-int4":
        assert r["chosen_adapter"]["name"] == a["name"]


def test_this_pc_ignores_igpu_and_warp_regardless_of_order():
    gpu = ad("AMD Radeon RX 7800 XT", 0x1002, 0x747E, 16177, luid="0x00000000_0x00012602")
    igpu = ad("AMD Radeon(TM) Graphics", 0x1002, 0x164E, 485, luid="0x00000000_0x00014830")
    for order in ([gpu, igpu, WARP], [igpu, gpu, WARP], [WARP, igpu, gpu]):
        r = dg.decide(order, CPU12, CFG)
        assert r["recommended_variant"] == "t1-dml-fp16"
        assert r["chosen_adapter"]["luid"] == "0x00000000_0x00012602"
        assert r["chosen_adapter"]["device_id"] == "0x747E"
        names = {i["name"] for i in r["ignored"]}
        assert {"AMD Radeon(TM) Graphics", "Microsoft Basic Render Driver"} <= names


def test_dgpu_preferred_over_larger_igpu_carveout():
    # Strix Halo com 64 GB de carve-out nao pode vencer uma dGPU elegivel
    strix = ad("AMD Radeon(TM) 8060S Graphics", 0x1002, 0x1586, 65536)
    gpu = ad("NVIDIA GeForce RTX 3060", 0x10DE, 0x2504, 12288, luid="0x00000000_0x00000002")
    assert dg.decide([strix, gpu], CPU12, CFG)["chosen_adapter"]["name"] == "NVIDIA GeForce RTX 3060"


def test_two_dgpus_picks_more_vram():
    a = ad("NVIDIA GeForce RTX 3060", 0x10DE, 0x2504, 12288, luid="0x00000000_0x0000000a")
    b = ad("NVIDIA GeForce RTX 4090", 0x10DE, 0x2684, 24564, luid="0x00000000_0x0000000b")
    assert dg.decide([a, b], CPU12, CFG)["chosen_adapter"]["luid"] == "0x00000000_0x0000000b"


def test_vram_below_t0_goes_cpu():
    r = dg.decide([ad("NVIDIA GeForce GTX 1050", 0x10DE, 0x1C81, 2048)], CPU12, CFG)
    assert r["recommended_variant"] == "t0-cpu-int4"


def test_pre_pascal_unsupported():
    r = dg.decide([ad("NVIDIA GeForce GTX 970", 0x10DE, 0x13C2, 4096)], CPU12, CFG)
    assert r["recommended_variant"] == "t0-cpu-int4"
    assert r["adapters"][0]["supported"] is False


def test_gtx1650_sits_at_t0_not_t1():
    r = dg.decide([ad("NVIDIA GeForce GTX 1650", 0x10DE, 0x1F82, 4096)], CPU12, CFG)
    assert r["recommended_variant"] == "t0-dml-fp16"
    assert r["fallback_chain"] == ["t0-dml-fp16", "t0-cpu-int4"]


def test_rx580_not_misclassified_as_rdna1():
    r = dg.decide([ad("AMD Radeon RX 580 Series", 0x1002, 0x67DF, 8192)], CPU12, CFG)
    assert r["adapters"][0]["rule"] == "amd-polaris"
    assert r["adapters"][0]["supported"] == "unknown"


def test_iris_xe_is_microbench_candidate_only():
    r = dg.decide([ad("Intel(R) Iris(R) Xe Graphics", 0x8086, 0x9A49, 128)], CPU12, CFG)
    assert r["recommended_variant"] == "t0-cpu-int4"
    assert r["microbench_candidates"] and r["microbench_candidates"][0]["variant"] == "t0-dml-fp16"


def test_unknown_vendor_and_no_adapters():
    r = dg.decide([ad("Qualcomm Adreno", 0x5143, 0x43, 0)], CPU12, CFG)
    assert r["recommended_variant"] == "t0-cpu-int4"
    assert dg.decide([], CPU12, CFG)["recommended_variant"] == "t0-cpu-int4"


@pytest.mark.parametrize("cpu,allowed", [({"physical_cores": 12, "ram_gb": 64}, True), ({"physical_cores": 8, "ram_gb": 32}, False),
                                         ({"physical_cores": 16, "ram_gb": 8}, False)])
def test_t1_cpu_int8_rule(cpu, allowed):
    r = dg.decide([], cpu, CFG)
    assert r["cpu_options"]["t1_cpu_int8_allowed"] is allowed
    assert r["recommended_variant"] == "t0-cpu-int4"  # autocomplete na CPU e sempre T0 int4


def test_config_sanity():
    ids = [r["id"] for r in CFG["rules"]]
    assert len(ids) == len(set(ids))
    for r in CFG["rules"]:
        assert r["supported"] in (True, False, "unknown"), r["id"]
        assert r["confidence"] in ("alta", "media", "baixa"), r["id"]
        assert r["evidence"], r["id"]
        assert r["gpu_variant"] in ("auto_vram", "cpu", "t0-dml-fp16", "t1-dml-fp16"), r["id"]
        assert r["vendor_id"] in (0x10DE, 0x1002, 0x8086, 0x1414), r["id"]


def test_live_enumeration_smoke():
    """No Windows real: nao quebra e devolve JSON-serializavel (nao assere o hardware)."""
    import json
    import os
    if os.name != "nt":
        pytest.skip("Windows only")
    ads = dg.enumerate_dxgi()
    if not ads:
        pytest.skip("DXGI indisponivel")
    json.dumps(dg.decide(ads, CPU12, CFG))
