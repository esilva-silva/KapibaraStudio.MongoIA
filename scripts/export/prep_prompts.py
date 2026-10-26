#!/usr/bin/env python3
"""Prepara os prompts do contrato editor-context-v1 (ids) e a referencia HF bf16 token a token para a paridade do pacote ONNX.

Venv ROCm (GPU RX 7800 XT via compact.env.require_gpu + gpu_lock). Saida: reports/export/work/<tier>_prompts.json
  - amostra de 500 ids: Random(seed).sample dos 2.000 ids de base-v2c/sample_ids.json; os 200 primeiros = conjunto de paridade/latencia
  - por exemplo: ids do prompt (build_ids de eval_teacher, ctx 2048), suffix[:1024], e para os 200 primeiros os tokens greedy do HF bf16
    (mesmo Generator do eval_teacher: limite 96, paradas do contrato, parada por sufixo).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
sys.path.insert(0, str(ROOT / "scripts" / "teacher"))
sys.path.insert(0, str(ROOT / "scripts" / "student"))
import eval_teacher as et  # noqa: E402
from _common import Generator, gpu_lock, load_hf  # noqa: E402

from build_variants import TIERS as _T, WORK  # noqa: E402
TIERS = {k: v["merged"] for k, v in _T.items()}
SAMPLE = ROOT / "Repos/KapiTeacher-Mongo-7B/reports/eval/base-v2c/sample_ids.json"
SEED = 20261020
N_QUALITY, N_PARITY = 500, 200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS), required=True)
    a = ap.parse_args()
    et.DATA_VERSION = "v2c"
    ids = json.loads(SAMPLE.read_text(encoding="utf-8"))["ids"]
    chosen = random.Random(SEED).sample(ids, N_QUALITY)
    rows = {r["id"]: r for r in et.read_rows("editor-context-v1", "val")}
    from compact.contracts import contract_for, postprocess  # noqa: F401

    model_dir = (ROOT / TIERS[a.tier]).resolve()
    out = []
    with gpu_lock(f"prep_prompts:{a.tier}"):
        model, tok, device, gpu, _ = load_hf(model_dir, "bf16")
        contract = contract_for(tok)
        gen = Generator(model, tok, device)
        print("GPU:", gpu.get("device_name"), flush=True)
        for k, rid in enumerate(chosen):
            row = rows[rid]
            pids = et.build_ids(contract, row, "editor-context-v1")
            item = {"id": rid, "prompt_ids": pids, "suffix": row["suffix"][:1024], "prompt_text": tok.decode(pids, skip_special_tokens=False)}
            if k < N_PARITY:
                res = gen.generate(pids, et.MAX_NEW_TOKENS, contract.stops, item["suffix"])
                item["hf_tokens"] = res["tokens"]
                item["hf_stop"] = res["stop"]
            out.append(item)
            if (k + 1) % 50 == 0:
                print(k + 1, flush=True)
        stops = sorted(contract.stops)
        del model
    dest = WORK / f"{a.tier}_prompts.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"tier": a.tier, "model": str(model_dir), "seed": SEED, "stops": stops, "max_new": et.MAX_NEW_TOKENS,
                                "n_parity": N_PARITY, "items": out}, ensure_ascii=False), encoding="utf-8")
    print("saved", dest, len(out))


if __name__ == "__main__":
    main()
