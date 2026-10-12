"""Ambiente de execução: caches fora do repositório e exigência de GPU ROCm (sem fallback para CPU)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


class GpuUnavailableError(RuntimeError):
    pass


def configure_caches(artifacts_root: str | Path) -> dict[str, str]:
    """Direciona caches e temporários pesados para o artifacts_root (antes de importar torch/transformers).

    Não sobrescreve variáveis já definidas pelo usuário.
    """
    root = Path(artifacts_root)
    cache = root / "cache"
    tmp = root / "tmp"
    values = {
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface" / "hub",
        "TORCH_HOME": cache / "torch",
        "TRITON_CACHE_DIR": cache / "triton",
        "MIOPEN_USER_DB_PATH": cache / "miopen",
        "MIOPEN_CUSTOM_CACHE_DIR": cache / "miopen",
        "PIP_CACHE_DIR": cache / "pip",
        "TMP": tmp,
        "TEMP": tmp,
        "TMPDIR": tmp,
    }
    applied: dict[str, str] = {}
    for key, path in values.items():
        if not os.environ.get(key) or key in ("TMP", "TEMP", "TMPDIR") and not os.environ.get("MONGOAI_KEEP_SYSTEM_TEMP"):
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
        applied[key] = os.environ[key]
    # ROCm/Windows: sem esta flag o SDPA cai no kernel "math" (O(n²) de memória). Medido na RX 7800 XT (2×32×2048, bf16):
    # efficient 0.037 s / 0.5 GB vs math 0.087 s / 4.47 GB, diferença máx. 0.0156. Opt-out: MONGOAI_ROCM_EXPERIMENTAL_ATTENTION=0.
    if os.environ.get("MONGOAI_ROCM_EXPERIMENTAL_ATTENTION", "1") == "1":
        os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    applied["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # o modelo base é local; nada é baixado implicitamente
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tempfile.tempdir = os.environ["TMP"]
    return applied


def gpu_report(expect_device: str = "", strict: bool = True) -> dict[str, Any]:
    """Valida e descreve a GPU. Com ``strict``, levanta GpuUnavailableError em qualquer falha."""
    report: dict[str, Any] = {}

    def fail(message: str) -> None:
        report.setdefault("failures", []).append(message)
        if strict:
            raise GpuUnavailableError(message)

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        fail(f"torch não importa: {type(exc).__name__}: {exc}")
        return report

    report["torch"] = torch.__version__
    report["hip"] = getattr(torch.version, "hip", None)
    report["rocm_aotriton_experimental_attention"] = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0")
    report["cuda_toolkit"] = torch.version.cuda
    if "+cpu" in torch.__version__:
        fail(f"torch CPU-only instalado ({torch.__version__}); instale o wheel ROCm via scripts/setup_env.ps1")
    if not report["hip"]:
        fail("torch.version.hip ausente: este torch não é um build ROCm")
    available = torch.cuda.is_available()
    report["cuda_is_available"] = available
    if not available:
        fail("torch.cuda.is_available() retornou False (driver AMD/ROCm não detectado pelo torch)")
        return report
    report["device_count"] = torch.cuda.device_count()
    devices = []
    for i in range(report["device_count"]):
        p = torch.cuda.get_device_properties(i)
        devices.append({"index": i, "name": torch.cuda.get_device_name(i), "gcn_arch": getattr(p, "gcnArchName", None),
                        "total_vram_gb": round(p.total_memory / 2**30, 2)})
    report["devices"] = devices
    # Seleciona explicitamente a GPU pelo nome (a iGPU do Ryzen também aparece no ROCm) ou por MONGOAI_GPU_INDEX.
    forced = os.environ.get("MONGOAI_GPU_INDEX")
    if forced is not None:
        index = int(forced)
    elif expect_device:
        matches = [d["index"] for d in devices if expect_device.lower() in d["name"].lower()]
        index = matches[0] if matches else -1
    else:
        index = max(devices, key=lambda d: d["total_vram_gb"])["index"]
    if index < 0 or index >= len(devices):
        fail(f"GPU esperada {expect_device!r} não encontrada entre {[d['name'] for d in devices]}")
        return report
    torch.cuda.set_device(index)
    device = f"cuda:{index}"
    name = devices[index]["name"]
    report["selected_index"] = index
    report["device_name"] = name
    report["gcn_arch"] = devices[index]["gcn_arch"]
    report["total_vram_gb"] = devices[index]["total_vram_gb"]
    if expect_device and expect_device.lower() not in name.lower():
        fail(f"GPU inesperada: {name!r} (esperado conter {expect_device!r})")
    try:
        free, total = torch.cuda.mem_get_info(index)
        report["free_vram_gb"] = round(free / 2**30, 2)
    except Exception as exc:  # noqa: BLE001
        report["free_vram_gb"] = f"unavailable: {exc}"
    try:
        report["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        a = torch.randn(1024, 1024, device=device, dtype=torch.float32)
        b = torch.randn(1024, 1024, device=device, dtype=torch.float32)
        c = a @ b
        torch.cuda.synchronize(index)
        expected = (a.cpu() @ b.cpu())
        report["tensor_allocation_ok"] = a.device.type == "cuda"
        report["matmul_ok"] = bool(torch.allclose(c.cpu(), expected, rtol=1e-3, atol=1e-2))
        if report["bf16_supported"]:
            x = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
            report["matmul_bf16_ok"] = bool(torch.isfinite((x @ x).float()).all().item())
        if not report["matmul_ok"]:
            fail("multiplicação de matrizes na GPU divergiu do resultado de referência")
    except GpuUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001
        fail(f"falha ao executar na GPU: {type(exc).__name__}: {exc}")
    return report


def video_controllers() -> list[dict]:
    """Nome/driver das GPUs via WMI (Windows)."""
    import json
    import platform
    import subprocess

    if platform.system() != "Windows":
        return []
    command = ("Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate,VideoProcessor "
               "| ConvertTo-Json -Compress")
    try:
        output = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True,
                                timeout=60, check=False).stdout.strip()
        data = json.loads(output) if output else []
        return data if isinstance(data, list) else [data]
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def environment_report(gpu: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ambiente completo para relatórios de treino/benchmark: Python, bibliotecas, GPU, driver e VRAM."""
    import importlib
    import platform
    import sys

    def version(name: str) -> str:
        try:
            return getattr(importlib.import_module(name), "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            return f"unavailable: {type(exc).__name__}"

    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "libraries": {n: version(n) for n in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "tokenizers",
                                               "safetensors", "onnxruntime", "onnxruntime_genai")},
        "video_controllers": video_controllers(),
        "gpu": gpu,
    }


def require_gpu(expect_device: str = "") -> dict[str, Any]:
    """Chamado por treino/benchmark/merge em GPU antes de carregar pesos."""
    return gpu_report(expect_device=expect_device, strict=True)


def limit_vram(gpu: dict[str, Any], fraction: float, reserve_gb: float = 0.9) -> dict[str, Any]:
    """Limita o alocador à VRAM dedicada realmente livre.

    No Windows, o driver permite que alocações ROCm transbordem para a memória compartilhada do sistema (medido: pico de 28,6 GB
    numa RX 7800 XT de 16 GB; com o professor 6.7B, 3,3 GB compartilhados e GPU ociosa paginando pelo PCIe), o que é um offload
    implícito e muito lento. O teto é o menor entre `fraction` do total e a VRAM livre no início menos `reserve_gb` (desktop, outros
    processos e overhead do driver). O excesso vira OutOfMemoryError explícito.
    """
    import torch

    index = gpu["selected_index"]
    free, total = torch.cuda.mem_get_info(index)
    cap = min(fraction * total, max(0.0, free - reserve_gb * 2**30))
    effective = cap / total
    torch.cuda.set_per_process_memory_fraction(effective, index)
    return {"requested_fraction": fraction, "effective_fraction": round(effective, 4), "max_vram_gb": round(cap / 2**30, 2),
            "free_at_start_gb": round(free / 2**30, 2), "reserve_gb": reserve_gb, "shared_memory_spill": "bloqueado"}
