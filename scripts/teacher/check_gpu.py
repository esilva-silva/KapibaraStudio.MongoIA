#!/usr/bin/env python3
"""Verificacao rapida de ambiente para a fila da GPU (gpu_queue.ps1): GPU esperada responde a matmul bf16. Exit 0 ok, 2 falha."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bakeoff" / "v1" / "scripts"))


def main() -> int:
    try:
        from compact.env import configure_caches, gpu_report

        configure_caches(ROOT)
        report = gpu_report("7800 XT", strict=True)
    except Exception as exc:  # noqa: BLE001
        print(f"check_gpu: FALHA {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"check_gpu: ok {report.get('device_name')} {report.get('gcn_arch')} livre {report.get('free_vram_gb')} GB "
          f"torch {report.get('torch')} hip {report.get('hip')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
