#!/usr/bin/env python3
"""Passo 1 da exportacao ONNX (Fase 07 §5/§6): roda o builder do onnxruntime-genai 0.15.2 (venv ROCm, CPU) para um tier e uma variante.

  python -I scripts/export/build_variants.py --tier t0 --variant int4 [--variant dml-fp16 ...] [--force]

Saida: Repos/KapiCoder-Mongo-<Nano|Small>/exports/KapiCoder-Mongo-<Nano|Small>-ONNX-<variante>/ (gitignored). O pacote so recebe o complemento
(tokenizer byte a byte, metadata kapi, LICENSE/NOTICE/README, export_manifest.json) em package_variant.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIERS = {
    "t0": {"family": "KapiCoder-Mongo-Nano", "merged": "Repos/KapiCoder-Mongo-Nano/merged/sft_distill_v1"},
    "t1": {"family": "KapiCoder-Mongo-Small", "merged": "Repos/KapiCoder-Mongo-Small/merged/sft_distill_v1"},
}
# variante -> (precisao, EP, extra_options). Fase 07 §5.2/§5.3 + contornos da sonda (reports/probe/onnx_export_t0_t1.md).
VARIANTS = {
    "fp32-cpu": ("fp32", "cpu", []),
    "int4": ("int4", "cpu", ["algo_config=k_quant", "matmul_mixed_precision=last_matmul:int8", "block_size=32", "accuracy_level=4"]),
    "int4-rtn": ("int4", "cpu", ["algo_config=rtn", "block_size=32", "accuracy_level=4"]),
    "int4-kquantmixed": ("int4", "cpu", ["algo_config=k_quant_mixed"]),
    # `-p int8` padrao gera grafo invalido no 0.15.2 (lm_head.MatMul.weight_Q4 inexistente) -> contorno: so MatMul, embedding fora do quantizador
    "int8": ("int8", "cpu", ["op_types_to_quantize=MatMul", "shared_embeddings=false", "block_size=32", "accuracy_level=4"]),
    # int8 + embedding INT4 (Gather so suporta 4 bits no ORT 1.30): variante para caber no gate de disco do T0 (<= 700 MB); pos-processo quantize_gather.py
    "int8-emb4": ("int8", "cpu", ["op_types_to_quantize=MatMul", "shared_embeddings=false", "block_size=32", "accuracy_level=4"]),
    "dml-fp16": ("fp16", "dml", []),
    "dml-int4": ("int4", "dml", ["op_types_to_quantize=MatMul", "shared_embeddings=false", "block_size=32"]),
}
PY = Path("C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe")


def package_dir(tier: str, variant: str) -> Path:
    fam = TIERS[tier]["family"]
    return ROOT / "Repos" / fam / "exports" / f"{fam}-ONNX-{variant}"


def build(tier: str, variant: str, force: bool = False) -> dict:
    prec, ep, extra = VARIANTS[variant]
    out = package_dir(tier, variant)
    if out.exists() and (out / "genai_config.json").exists() and not force:
        print(f"[skip] {out.name} ja existe")
        return {}
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cache = ROOT / "tmp" / f"export_cache_{tier}_{variant}"   # um cache por build (o builder apaga o diretorio ao terminar)
    cache.mkdir(parents=True, exist_ok=True)
    cmd = [str(PY), "-I", "-m", "onnxruntime_genai.models.builder", "-i", str(ROOT / TIERS[tier]["merged"]), "-o", str(out), "-p", prec, "-e", ep,
           "-c", str(cache)]
    if extra:
        cmd += ["--extra_options", *extra]
    print("$", " ".join(cmd), flush=True)
    t = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"})
    info = {"command": " ".join(cmd), "returncode": r.returncode, "seconds": round(time.time() - t, 1)}
    (out / ".build_info.json").write_text(json.dumps(info), encoding="utf-8")
    if r.returncode:
        raise SystemExit(f"builder falhou ({variant}): {r.returncode}")
    if variant == "int8-emb4":
        cmd2 = [str(PY), "-I", str(ROOT / "scripts" / "export" / "quantize_gather.py"), str(out)]
        r2 = subprocess.run(cmd2, cwd=ROOT)
        info["command"] += " && " + " ".join(cmd2)
        if r2.returncode:
            raise SystemExit("quantize_gather falhou")
        (out / ".build_info.json").write_text(json.dumps(info), encoding="utf-8")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS), required=True)
    ap.add_argument("--variant", action="append", choices=list(VARIANTS), required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for v in a.variant:
        build(a.tier, v, a.force)


if __name__ == "__main__":
    main()
