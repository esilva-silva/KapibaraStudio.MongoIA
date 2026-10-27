"""Testes de detect_gpus.py com saidas simuladas (nao tocam o hardware). Rodar:
  env\\.venv\\Scripts\\python.exe -m pytest scripts/hwmap/tests -q -p no:cacheprovider
Regras testadas: decisoes D1-D5 (2026-10-26), configs/gpu_compat.yaml schema 2. Os device_id simulados sao [E] (ver o YAML); os de 7800 XT / iGPU Raphael sao [M]."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detect_gpus as dg  # noqa: E402

CFG = dg.load_config()
TH = CFG["thresholds"]
CPU12 = {"physical_cores": 12, "ram_gb": 32.0}
CPU4 = {"physical_cores": 4, "ram_gb": 8.0}
FULL = 2048   # contexto cheio dos pacotes


def ad(name, ven, dev, vram, shared=16000, software=False, luid="0x00000000_0x00000001", idx=None):
    return {"name": name, "vendor_id": ven, "device_id": dev, "dedicated_vram_mb": vram, "shared_mb": shared,
            "is_software": software, "luid": luid, "driver_version": "1.0", "dxgi_index": idx}


WARP = ad("Microsoft Basic Render Driver", 0x1414, 0x8C, 0, software=True, luid="0x00000000_0x000000ff")

# nome, adaptador, variante esperada, necessita micro-benchmark?   (os 15 casos simulados originais, com as regras novas)
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
    # D3: TODA iGPU vai para a CPU nesta versao (igpu_policy cpu_always_until_measured), inclusive 780M, Core Ultra e Iris Xe
    ("iGPU Vega 8", ad("AMD Radeon(TM) Graphics", 0x1002, 0x1638, 512), "t0-cpu-int4", False),
    ("iGPU 780M", ad("AMD Radeon(TM) 780M Graphics", 0x1002, 0x15BF, 512), "t0-cpu-int4", False),
    ("UHD 630", ad("Intel(R) UHD Graphics 630", 0x8086, 0x3E92, 128), "t0-cpu-int4", False),
    ("Iris Xe", ad("Intel(R) Iris(R) Xe Graphics", 0x8086, 0x9A49, 128), "t0-cpu-int4", False),
    ("Core Ultra iGPU", ad("Intel(R) Arc(TM) Graphics", 0x8086, 0x7D55, 128), "t0-cpu-int4", False),
    ("Basic Render Driver", WARP, "t0-cpu-int4", False),
]
# casos adicionais pedidos (D1/D2/D3)
EXTRA = [
    ("RX 6600 8GB", ad("AMD Radeon RX 6600", 0x1002, 0x73FF, 8176), "t1-dml-fp16", False),
    ("RX 6500 XT 4GB", ad("AMD Radeon RX 6500 XT", 0x1002, 0x743F, 4096), "t0-dml-fp16", False),
    ("GTX 1650 4GB (TU117)", ad("NVIDIA GeForce GTX 1650", 0x10DE, 0x1F82, 4096), "t0-dml-fp16", False),
    ("RTX 3050 6GB", ad("NVIDIA GeForce RTX 3050", 0x10DE, 0x2507, 6144), "t1-dml-fp16", True),
    ("GTX 1060 6GB", ad("NVIDIA GeForce GTX 1060 6GB", 0x10DE, 0x1C03, 6144), "t1-dml-fp16", True),
    ("iGPU 780M (extra)", ad("AMD Radeon(TM) 780M Graphics", 0x1002, 0x15BF, 512), "t0-cpu-int4", False),
]
ALL = CASES + IGPU_CASES + EXTRA


@pytest.mark.parametrize("label,a,variant,mb", ALL, ids=[c[0] for c in ALL])
def test_single_adapter(label, a, variant, mb):
    r = dg.decide([a, WARP] if a is not WARP else [a], CPU12, CFG)
    assert r["recommended_variant"] == variant, r["reason"]
    assert r["needs_microbench"] is mb
    assert r["fallback_chain"][-1] == "t0-cpu-int4"
    assert r["confidence"] in ("alta", "media", "baixa")
    assert isinstance(r["context_cap_tokens"], int) and r["context_cap_tokens"] > 0
    if variant != "t0-cpu-int4":
        assert r["chosen_adapter"]["name"] == a["name"]
        assert r["context_cap_tokens"] == dg.cap_for(CFG, variant, a["dedicated_vram_mb"])
    else:
        assert r["context_cap_tokens"] == TH["cpu_context_cap_tokens"]


# ---------------------------------------------------------------- D1: VRAM e teto de contexto
def test_d1_t1_full_context_needs_8gb():
    for vram in (8176, 8192, 12288, 16177):
        r = dg.decide([ad("AMD Radeon RX 6600", 0x1002, 0x73FF, vram)], CPU12, CFG)
        assert r["recommended_variant"] == "t1-dml-fp16" and r["context_cap_tokens"] == FULL and r["context_capped"] is False


@pytest.mark.parametrize("vram", [6100, 6144, 7000, 8149])
def test_d1_6_to_8gb_t1_only_capped_and_needs_microbench(vram):
    r = dg.decide([ad("AMD Radeon RX 6600", 0x1002, 0x73FF, vram)], CPU12, CFG)
    assert r["recommended_variant"] == "t1-dml-fp16"
    assert r["context_capped"] is True and r["needs_microbench"] is True
    assert CFG["context_cap_min_useful_tokens"] <= r["context_cap_tokens"] < FULL
    assert "t1-dml-fp16" in r["fallback_chain"] and r["fallback_chain"][-1] == "t0-cpu-int4"


@pytest.mark.parametrize("vram", [3500, 4096, 5000, 6099])
def test_d1_below_6gb_goes_t0_with_proportional_cap(vram):
    r = dg.decide([ad("AMD Radeon RX 6600", 0x1002, 0x73FF, vram)], CPU12, CFG)
    assert r["recommended_variant"] == "t0-dml-fp16"
    assert r["context_cap_tokens"] == dg.cap_for(CFG, "t0-dml-fp16", vram)
    assert r["fallback_chain"] == ["t0-dml-fp16", "t0-cpu-int4"]


@pytest.mark.parametrize("vram", [0, 1024, 2048, 3499])
def test_d1_below_3_5gb_goes_cpu(vram):
    r = dg.decide([ad("AMD Radeon RX 6600", 0x1002, 0x73FF, vram)], CPU12, CFG)
    assert r["recommended_variant"] == "t0-cpu-int4" and r["chosen_adapter"] is None


def test_d1_caps_are_monotonic_in_vram_and_t1_needs_more_than_t0():
    prev = {"t1-dml-fp16": 0, "t0-dml-fp16": 0}
    for vram in range(0, 20000, 100):
        for v in prev:
            c = dg.cap_for(CFG, v, vram) or 0
            assert c >= prev[v], (v, vram)
            prev[v] = c
    assert dg.cap_for(CFG, "t1-dml-fp16", TH["t1_full_min_vram_mb"]) == FULL
    # T1 sempre pede mais VRAM do que o T0 para o mesmo teto
    for cap in (768, 1024, 1536, FULL):
        need = {v: min((m for m, c in CFG["context_cap_by_vram"][v] if c >= cap), default=None) for v in ("t1-dml-fp16", "t0-dml-fp16")}
        if need["t1-dml-fp16"] and need["t0-dml-fp16"]:
            assert need["t1-dml-fp16"] > need["t0-dml-fp16"], cap


def test_d1_table_matches_thresholds():
    t1 = CFG["context_cap_by_vram"]["t1-dml-fp16"]
    t0 = CFG["context_cap_by_vram"]["t0-dml-fp16"]
    assert min(m for m, _ in t1) == TH["t1_capped_min_vram_mb"] and TH["t1_capped_min_vram_mb"] <= TH["t1_full_min_vram_mb"]
    assert min(m for m, _ in t0) == TH["t0_min_vram_mb"]
    assert all(c >= CFG["context_cap_min_useful_tokens"] for _, c in t0 + t1)


# ---------------------------------------------------------------- D2: Pascal / Polaris
def test_d2_pascal_unknown_microbench_no_fp32_and_failure_goes_cpu():
    r = dg.decide([ad("NVIDIA GeForce GTX 1060 6GB", 0x10DE, 0x1C03, 6144)], CPU12, CFG)
    assert r["adapters"][0]["rule"] == "nv-pascal" and r["adapters"][0]["supported"] == "unknown"
    assert r["needs_microbench"] is True
    assert r["fallback_chain"] == ["t1-dml-fp16", "t0-cpu-int4"]     # reprovou o micro-benchmark => CPU T0 int4, sem T0 DML
    assert not any("fp32" in v for v in r["fallback_chain"])
    r8 = dg.decide([ad("NVIDIA GeForce GTX 1080", 0x10DE, 0x1B80, 8192)], CPU12, CFG)
    assert r8["recommended_variant"] == "t1-dml-fp16" and r8["needs_microbench"] is True and r8["context_cap_tokens"] == FULL


def test_d2_no_fp32_variant_anywhere_in_config():
    for rule in CFG["rules"]:
        assert "fp32" not in str(rule["gpu_variant"]) and "fp32" not in str(rule.get("fallback", ""))
    assert "fp32" not in str(CFG["thresholds"]["cpu_default_variant"])
    assert any(i["id"] == "dml-fp32" for i in CFG["variants_not_exported"]["items"])


def test_rx580_not_misclassified_as_rdna1():
    r = dg.decide([ad("AMD Radeon RX 580 Series", 0x1002, 0x67DF, 8192)], CPU12, CFG)
    assert r["adapters"][0]["rule"] == "amd-polaris"
    assert r["adapters"][0]["supported"] == "unknown"


# ---------------------------------------------------------------- D3: iGPU sempre CPU
def test_d3_igpu_policy_and_every_igpu_rule_is_cpu():
    assert CFG["igpu_policy"] == "cpu_always_until_measured"
    integrated = [r for r in CFG["rules"] if r.get("integrated") or r.get("kind") == "igpu"]
    assert integrated
    for r in integrated:
        assert r["gpu_variant"] == "cpu", r["id"]
        assert "candidate_variant" not in r


@pytest.mark.parametrize("a", [ad("AMD Radeon(TM) 780M Graphics", 0x1002, 0x15BF, 512), ad("AMD Radeon(TM) 680M", 0x1002, 0x1681, 512),
                               ad("AMD Radeon(TM) 8060S Graphics", 0x1002, 0x1586, 65536), ad("Intel(R) Arc(TM) Graphics", 0x8086, 0x7D55, 128),
                               ad("Intel(R) Iris(R) Xe Graphics", 0x8086, 0x9A49, 128), ad("AMD Radeon(TM) Graphics", 0x1002, 0x164E, 485)])
def test_d3_larger_igpus_not_candidates(a):
    r = dg.decide([a], CPU12, CFG)
    assert r["recommended_variant"] == "t0-cpu-int4" and r["chosen_adapter"] is None
    assert r["microbench_candidates"] == []
    assert r["igpu_policy"] == "cpu_always_until_measured"
    assert r["adapters"][0]["variant_if_chosen"] == "cpu"
    assert r["igpus_pending_measurement"] and r["igpus_pending_measurement"][0]["name"] == a["name"]


def test_this_pc_ignores_igpu_and_warp_regardless_of_order():
    gpu = ad("AMD Radeon RX 7800 XT", 0x1002, 0x747E, 16177, luid="0x00000000_0x00012602")
    igpu = ad("AMD Radeon(TM) Graphics", 0x1002, 0x164E, 485, luid="0x00000000_0x00014830")
    for order in ([gpu, igpu, WARP], [igpu, gpu, WARP], [WARP, igpu, gpu]):
        r = dg.decide(order, CPU12, CFG)
        assert r["recommended_variant"] == "t1-dml-fp16" and r["context_cap_tokens"] == FULL
        assert r["chosen_adapter"]["luid"] == "0x00000000_0x00012602"
        assert r["chosen_adapter"]["device_id"] == "0x747E"
        names = {i["name"] for i in r["ignored"]}
        assert {"AMD Radeon(TM) Graphics", "Microsoft Basic Render Driver"} <= names


def test_dgpu_preferred_over_larger_igpu_carveout():
    # Strix Halo com 64 GB de carve-out nao pode vencer uma dGPU elegivel
    strix = ad("AMD Radeon(TM) 8060S Graphics", 0x1002, 0x1586, 65536)
    gpu = ad("NVIDIA GeForce RTX 3060", 0x10DE, 0x2504, 12288, luid="0x00000000_0x00000002")
    assert dg.decide([strix, gpu], CPU12, CFG)["chosen_adapter"]["name"] == "NVIDIA GeForce RTX 3060"


# ---------------------------------------------------------------- notebooks hibridos (iGPU + dGPU): escolhe a dGPU por vendor/device/VRAM
@pytest.mark.parametrize("order", ["igpu_first", "dgpu_first"])
def test_hybrid_laptop_780m_plus_rtx3050_picks_dgpu_with_cap(order):
    igpu = ad("AMD Radeon(TM) 780M Graphics", 0x1002, 0x15BF, 512, luid="0x00000000_0x0000a001", idx=0 if order == "igpu_first" else 1)
    dgpu = ad("NVIDIA GeForce RTX 3050 Laptop GPU", 0x10DE, 0x25A2, 6144, luid="0x00000000_0x0000a002", idx=1 if order == "igpu_first" else 0)
    ads = [igpu, dgpu] if order == "igpu_first" else [dgpu, igpu]
    r = dg.decide(ads + [WARP], CPU12, CFG)
    assert r["chosen_adapter"]["luid"] == "0x00000000_0x0000a002"
    assert r["recommended_variant"] == "t1-dml-fp16" and r["context_capped"] is True and r["needs_microbench"] is True
    assert r["adapter_selection"]["multi_adapter_machine"] is True
    assert r["adapter_selection"]["effective_adapter_check_required"] is True
    assert r["adapter_selection"]["chosen_is_dxgi_default"] is (order == "dgpu_first")   # o indice nunca decide, so informa (PR-5)
    assert "AMD Radeon(TM) 780M Graphics" in {i["name"] for i in r["ignored"]}


def test_hybrid_small_dgpu_still_beats_igpu_and_below_min_goes_cpu():
    igpu = ad("Intel(R) UHD Graphics 630", 0x8086, 0x3E92, 128, luid="0x00000000_0x0000b001", idx=0)
    mx = ad("NVIDIA GeForce GTX 1650 with Max-Q", 0x10DE, 0x1F91, 4096, luid="0x00000000_0x0000b002", idx=1)
    r = dg.decide([igpu, mx], CPU12, CFG)
    assert r["chosen_adapter"]["luid"] == "0x00000000_0x0000b002" and r["recommended_variant"] == "t0-dml-fp16"
    weak = ad("NVIDIA GeForce GTX 1050", 0x10DE, 0x1C81, 2048, luid="0x00000000_0x0000b003", idx=1)
    r2 = dg.decide([igpu, weak], CPU12, CFG)
    assert r2["recommended_variant"] == "t0-cpu-int4" and r2["chosen_adapter"] is None   # a iGPU NUNCA assume


def test_two_dgpus_picks_more_vram():
    a = ad("NVIDIA GeForce RTX 3060", 0x10DE, 0x2504, 12288, luid="0x00000000_0x0000000a")
    b = ad("NVIDIA GeForce RTX 4090", 0x10DE, 0x2684, 24564, luid="0x00000000_0x0000000b")
    assert dg.decide([a, b], CPU12, CFG)["chosen_adapter"]["luid"] == "0x00000000_0x0000000b"


# ---------------------------------------------------------------- fronteira e CPU (D1, D4)
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
    assert r["context_cap_tokens"] == dg.cap_for(CFG, "t0-dml-fp16", 4096) < FULL


def test_unknown_vendor_and_no_adapters():
    r = dg.decide([ad("Qualcomm Adreno", 0x5143, 0x43, 0)], CPU12, CFG)
    assert r["recommended_variant"] == "t0-cpu-int4"
    r0 = dg.decide([], CPU12, CFG)
    assert r0["recommended_variant"] == "t0-cpu-int4" and r0["context_cap_tokens"] == TH["cpu_context_cap_tokens"]


@pytest.mark.parametrize("cpu,allowed", [({"physical_cores": 12, "ram_gb": 64}, True), ({"physical_cores": 8, "ram_gb": 32}, False),
                                         ({"physical_cores": 16, "ram_gb": 8}, False)])
def test_d4_t1_cpu_int8_is_optional_slow_never_default(cpu, allowed):
    r = dg.decide([], cpu, CFG)
    o = r["cpu_options"]
    assert o["t1_cpu_int8_allowed"] is allowed
    assert o["t1_cpu_int8_is_default"] is False and o["t1_cpu_int8_user_choice_only"] is True
    assert o["t1_cpu_int8_declared_slow"] is True and o["t1_cpu_int8_meets_ttft_gate"] is False
    assert r["recommended_variant"] == "t0-cpu-int4"  # autocomplete na CPU e sempre T0 int4
    assert r["fallback_chain"] == ["t0-cpu-int4"]


def test_d4_config_flags():
    assert TH["cpu_default_variant"] == "t0-cpu-int4" and TH["cpu_optional_variant"] == "t1-cpu-int8"
    assert TH["cpu_t1_int8_is_default"] is False and TH["cpu_t1_int8_meets_ttft_gate"] is False
    assert TH["cpu_t1_int8_min_physical_cores"] == 12 and TH["cpu_t1_int8_min_ram_gb"] == 16


# ---------------------------------------------------------------- D5, esquema e sanidade
def test_d5_variants_not_exported_and_decisions():
    ids = {i["id"] for i in CFG["variants_not_exported"]["items"]}
    assert {"cuda", "tensorrt-rtx", "openvino", "migraphx", "gguf-llama-cpp", "t1-dml-int4", "t0-dml-int4", "dml-fp32"} <= ids
    assert CFG["variants_not_exported"]["decision_ref"] == "D5"
    for d in ("D1", "D2", "D3", "D4", "D5"):
        assert CFG["decisions"][d]["status"] == "DECIDIDO" and CFG["decisions"][d]["date"] == "2026-10-26"


def test_schema_version_2_and_migration_documented():
    assert CFG["schema_version"] == dg.SCHEMA == 2
    assert CFG["migration"]["from"] == 1 and CFG["migration"]["to"] == 2 and CFG["migration"]["changes"]
    r = dg.decide([ad("AMD Radeon RX 7800 XT", 0x1002, 0x747E, 16177)], CPU12, CFG)
    assert r["schema_version"] == 2


def test_old_schema_is_rejected(tmp_path):
    old = tmp_path / "old.yaml"
    old.write_text("schema_version: 1\nrules: []\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        dg.load_config(old)


def test_config_sanity():
    ids = [r["id"] for r in CFG["rules"]]
    assert len(ids) == len(set(ids))
    for r in CFG["rules"]:
        assert r["supported"] in (True, False, "unknown"), r["id"]
        assert r["confidence"] in ("alta", "media", "baixa"), r["id"]
        assert r["evidence"], r["id"]
        assert r["gpu_variant"] in ("auto_vram", "cpu", "t0-dml-fp16", "t1-dml-fp16"), r["id"]
        assert r["vendor_id"] in (0x10DE, 0x1002, 0x8086, 0x1414), r["id"]
        if r["gpu_variant"] == "auto_vram":
            assert r.get("decision_ref") in ("D1", "D2"), r["id"]
        if r.get("integrated"):
            assert r.get("decision_ref") == "D3", r["id"]


def test_evidence_m_only_where_measured():
    """[M] so aparece em regras/limiares com medida neste PC: as regras [E] nao podem comecar a evidencia com 'M ' sem citar arquivo/medida."""
    for r in CFG["rules"]:
        for ev in r["evidence"]:
            if ev.startswith("M "):
                assert any(k in ev for k in ("reports/", "llama.cpp", "PNPDeviceID", "D2", "D3")), (r["id"], ev)


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


@pytest.mark.parametrize("cores,threads", [(8,8), (8,16), (8,12), (24,32), (12,24)])
def test_physical_cores_are_wmi_measurements_not_half_threads(monkeypatch, cores, threads):
    monkeypatch.setattr(dg.os, "cpu_count", lambda: threads)
    assert dg.detect_cpu({"cpu": [{"NumberOfCores": cores, "Name": "fixture"}]})["physical_cores"] == cores


def test_unknown_topology_is_not_invented(monkeypatch):
    monkeypatch.setattr(dg.os, "cpu_count", lambda: 16)
    assert dg.detect_cpu(None)["physical_cores"] is None


def test_missing_device_id_can_use_name_or_unknown_rule():
    adapter = ad("AMD Radeon RX 7800 XT", 0x1002, None, 16368)
    assert dg.decide([adapter], CPU12, CFG)["chosen_adapter"]["device_id"] is None
    adapter["name"] = "Unknown adapter"
    assert dg.decide([adapter], CPU12, CFG)["needs_microbench"]


@pytest.mark.parametrize("reverse", [False, True])
def test_wmi_controllers_without_luids_are_still_a_multi_adapter_machine(reverse):
    wmi = {"video": [
        {"Name": "NVIDIA GeForce GTX 1650", "PNPDeviceID": "PCI\\VEN_10DE&DEV_1F82", "AdapterRAM": 4096 * 1024 * 1024},
        {"Name": "Intel(R) UHD Graphics 630", "PNPDeviceID": "PCI\\VEN_8086&DEV_3E92", "AdapterRAM": 128 * 1024 * 1024}
    ]}
    adapters = dg.adapters_from_wmi(wmi)
    assert all(adapter["luid"] is None for adapter in adapters)
    if reverse:
        adapters.reverse()
    result = dg.decide(adapters, CPU12, CFG)
    assert result["chosen_adapter"]["name"] == "NVIDIA GeForce GTX 1650"
    assert result["adapter_selection"]["multi_adapter_machine"] is True
    assert result["adapter_selection"]["effective_adapter_check_required"] is True


def test_single_wmi_controller_without_luid_is_not_a_multi_adapter_machine():
    adapters = dg.adapters_from_wmi({"video": [
        {"Name": "NVIDIA GeForce GTX 1650", "PNPDeviceID": "PCI\\VEN_10DE&DEV_1F82", "AdapterRAM": 4096 * 1024 * 1024}
    ]})
    result = dg.decide(adapters, CPU12, CFG)
    assert result["adapter_selection"]["multi_adapter_machine"] is False
