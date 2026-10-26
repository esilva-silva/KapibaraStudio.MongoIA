#!/usr/bin/env python3
"""Passo 1 da exportacao ONNX (Fase 07 §5/§6): roda o builder do onnxruntime-genai 0.15.2 (venv ROCm, CPU) para um tier e uma variante.

  python -I scripts/export/build_variants.py --tier t0 --variant int4 [--variant dml-fp16 ...] [--force]

Saida: Repos/KapiCoder-Mongo-<Nano|Small>/exports/KapiCoder-Mongo-<Nano|Small>-ONNX-<variante>/ (gitignored). O pacote so recebe o complemento
(tokenizer byte a byte, metadata kapi, LICENSE/NOTICE/README, export_manifest.json) em package_variant.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Geracao dos modelos: "" = legado (merged/sft_distill_v1, dataset v2c); KAPI_EXPORT_GEN=v3 = retreino v3 (merged/sft_v3, corpus v3).
# A geracao v3 usa pacotes `<familia>-v3-ONNX-<variante>` e diretorios de trabalho/relatorio separados (work_v3, eval_v3, results_v3_*, gates_v3):
# nada do legado e sobrescrito. Defina a variavel ANTES de iniciar qualquer script (os subprocessos herdam).
GEN = os.environ.get("KAPI_EXPORT_GEN", "").strip().lower()
if GEN not in ("", "v3"):
    raise SystemExit(f"KAPI_EXPORT_GEN invalido: {GEN!r} (use vazio ou v3)")
V3 = GEN == "v3"
RUN = "sft_v3" if V3 else "sft_distill_v1"          # nome do run (merged/<RUN>, reports/eval/<RUN>)
TIERS = {
    "t0": {"family": "KapiCoder-Mongo-Nano", "merged": f"Repos/KapiCoder-Mongo-Nano/merged/{RUN}"},
    "t1": {"family": "KapiCoder-Mongo-Small", "merged": f"Repos/KapiCoder-Mongo-Small/merged/{RUN}"},
}
EXPORT = ROOT / "reports" / "export"
WORK = EXPORT / ("work_v3" if V3 else "work")                  # prompts, saidas brutas do runtime, latencia
EVAL_DIR = EXPORT / ("eval_v3" if V3 else "eval")               # predicoes/summary por tier/variante
GATES_DIR = EXPORT / ("gates_v3" if V3 else "gates")
HF_EVAL = {t: ROOT / "Repos" / TIERS[t]["family"] / "reports" / "eval" / RUN for t in TIERS}    # predicoes do HF bf16 (n=2000, val v2c)


def results_path(tier: str) -> Path:
    return EXPORT / (f"results_v3_{tier}.json" if V3 else f"results_{tier}.json")
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


def package_dir(tier: str, variant: str, gen: str | None = None) -> Path:
    """Diretorio do pacote. `gen` (None = a geracao ativa por KAPI_EXPORT_GEN; "" = legado; "v3")."""
    fam = TIERS[tier]["family"]
    v3 = V3 if gen is None else gen == "v3"
    return ROOT / "Repos" / fam / "exports" / (f"{fam}-v3-ONNX-{variant}" if v3 else f"{fam}-ONNX-{variant}")


def merged_dir(tier: str, gen: str | None = None) -> Path:
    fam = TIERS[tier]["family"]
    v3 = V3 if gen is None else gen == "v3"
    return ROOT / "Repos" / fam / "merged" / ("sft_v3" if v3 else "sft_distill_v1")


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
