#!/usr/bin/env python3
"""Exporta pacotes EXPERIMENTAIS (fase 2 do mapa de GPU) com o builder 0.15.2 da IDE, sem tocar nos pacotes v3 existentes.

  python -I scripts/hwmap/build_exp.py --tier t0 --variant dml-fp32 --variant dml-int4

Saida: Repos/KapiCoder-Mongo-<Nano|Small>/exports/<familia>-v3-ONNX-<variante>-exp (gitignored). Equivale ao `build_variants.build`, mas com o
sufixo `-exp`, a variante `dml-fp32` (nao existe no build_variants) e `search.max_length = 2048` aplicado no genai_config (como o package_variant).
Nao aplica o complemento do package_variant (metadata kapi): o pacote serve so para medir.
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
os.environ["KAPI_EXPORT_GEN"] = "v3"
sys.path.insert(0, str(ROOT / "scripts" / "export"))
import build_variants as bv  # noqa: E402

# O builder 0.15.2 declara FP32+DML invalido ("Valid precision + execution provider combinations": so FP16 DML e INT4 DML); `-p fp32 -e dml` cai no caminho de
# Attention empacotada e gera grafo invalido ("Graph output (present.0.key) does not exist"). Contorno experimental: forcar o caminho MultiHeadAttention (com repeat_kv) por
# monkeypatch em memoria (nao altera o pacote pip). `dml-fp16-mha` = controle com o MESMO caminho de atencao em fp16 (isola o efeito da precisao).
MHA_PATCH = ("import runpy,sys,os,importlib.util; d=os.path.join(os.path.dirname(importlib.util.find_spec('onnxruntime_genai').origin),'models'); sys.path.insert(0,d); "
             "from builders import base; base.Model.is_packed_attn_supported = lambda self: False; base.Model.is_gqa_supported = (lambda self: False) if os.environ.get('KAPI_NO_GQA') else base.Model.is_gqa_supported; sys.argv=['builder']+sys.argv[1:]; "
             "runpy.run_module('onnxruntime_genai.models.builder', run_name='__main__')")
EXP = {"dml-fp32": ("fp32", "dml", []), "dml-fp16-mha": ("fp16", "dml", []), "dml-int4": bv.VARIANTS["dml-int4"], "dml-fp16": bv.VARIANTS["dml-fp16"]}


def exp_dir(tier: str, variant: str) -> Path:
    return Path(str(bv.package_dir(tier, variant.replace("-mha", "") if variant == "dml-fp16-mha" else variant)) + ("-mha-exp" if variant == "dml-fp16-mha" else "-exp"))


def build(tier: str, variant: str, force: bool) -> None:
    prec, ep, extra = EXP[variant]
    out = exp_dir(tier, variant)
    if (out / "genai_config.json").exists() and not force:
        print("[skip]", out.name)
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cache = ROOT / "tmp" / f"export_cache_exp_{tier}_{variant}"
    cache.mkdir(parents=True, exist_ok=True)
    head = [str(bv.PY), "-I", "-c", MHA_PATCH] if variant in ("dml-fp32", "dml-fp16-mha") else [str(bv.PY), "-I", "-m", "onnxruntime_genai.models.builder"]
    os.environ["KAPI_NO_GQA"] = "1" if variant == "dml-fp16-mha" else ""
    cmd = [*head, "-i", str(ROOT / bv.TIERS[tier]["merged"]), "-o", str(out), "-p", prec, "-e", ep, "-c", str(cache)]
    if extra:
        cmd += ["--extra_options", *extra]
    print("$", " ".join(cmd), flush=True)
    t = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    info = {"command": " ".join(cmd), "returncode": r.returncode, "seconds": round(time.time() - t, 1), "experimental": True}
    (out / ".build_info.json").write_text(json.dumps(info), encoding="utf-8")
    if r.returncode:
        raise SystemExit(f"builder falhou ({variant}): {r.returncode}")
    cfgp = out / "genai_config.json"
    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
    cfg["search"]["max_length"] = 2048
    if variant in ("dml-fp32", "dml-fp16-mha"):   # MHA sem past_present_share_buffer => sem graph capture do DML
        for po in cfg["model"]["decoder"]["session_options"].get("provider_options", []):
            if "dml" in po:
                po["dml"]["enable_graph_capture"] = "0"
    cfgp.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.iterdir() if p.is_file()) / 2**20
    print(f"[ok] {out.name} {size:.0f} MB em {info['seconds']} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(bv.TIERS), required=True)
    ap.add_argument("--variant", action="append", choices=list(EXP), required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for v in a.variant:
        build(a.tier, v, a.force)


if __name__ == "__main__":
    main()
