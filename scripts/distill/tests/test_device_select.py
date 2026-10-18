"""Selecao do dispositivo do llama-server por NOME (nao por indice VulkanN) e guarda de sanidade (saida simulada de --list-devices)."""
import pytest

import generate_teacher as gt

OUT_NORMAL = """ggml_vulkan: Found 2 Vulkan devices:
Available devices:
  Vulkan0: AMD Radeon RX 7800 XT (16368 MiB, 15409 MiB free)
  Vulkan1: AMD Radeon(TM) Graphics (48545 MiB, 46118 MiB free)
"""
OUT_SWAPPED = """Available devices:
  Vulkan0: AMD Radeon(TM) Graphics (48545 MiB, 46118 MiB free)
  Vulkan1: AMD Radeon RX 7800 XT (16368 MiB, 15409 MiB free)
"""


def test_parse_devices():
    d = gt.parse_devices(OUT_NORMAL)
    assert [x["id"] for x in d] == ["Vulkan0", "Vulkan1"]
    assert d[0]["name"] == "AMD Radeon RX 7800 XT" and d[0]["total_mib"] == 16368 and d[0]["free_mib"] == 15409
    assert d[1]["name"] == "AMD Radeon(TM) Graphics"


def test_selection_follows_name_not_index():
    assert gt.select_device(gt.parse_devices(OUT_NORMAL))["id"] == "Vulkan0"
    assert gt.select_device(gt.parse_devices(OUT_SWAPPED))["id"] == "Vulkan1"


def test_custom_pattern():
    assert gt.select_device(gt.parse_devices(OUT_SWAPPED), "graphics")["id"] == "Vulkan0"


def test_fallback_largest_vram_without_graphics():
    devs = gt.parse_devices("  Vulkan0: AMD Radeon(TM) Graphics (48545 MiB, 1 MiB free)\n  Vulkan1: NVIDIA RTX 4070 (12282 MiB, 1 MiB free)\n"
                            "  Vulkan2: AMD Radeon RX 6900 XT (16368 MiB, 1 MiB free)\n")
    assert gt.select_device(devs)["id"] == "Vulkan2"


def test_fails_clearly():
    with pytest.raises(SystemExit):
        gt.select_device([])
    with pytest.raises(SystemExit, match="dGPU"):
        gt.select_device(gt.parse_devices("  Vulkan0: AMD Radeon(TM) Graphics (48545 MiB, 1 MiB free)\n"))


def test_parse_ignores_noise():
    assert gt.parse_devices("load_backend: loaded Vulkan\nbuild: 11693\n") == []
