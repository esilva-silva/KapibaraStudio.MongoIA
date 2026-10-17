#!/usr/bin/env python3
"""Funde o adapter LoRA do KapiTeacher no Qwen2.5-Coder-7B Base em bf16 (CPU; nao usa a GPU) - Fase 02 §3.3/§4.

  C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe -I scripts/teacher/merge_teacher.py \
      --adapter Repos/KapiTeacher-Mongo-7B/adapters/sft_v2c/final --out Repos/KapiTeacher-Mongo-7B/merged/KapiTeacher-Mongo-7B-v1

Carrega o base em bf16 (~15 GB de RAM), aplica o adapter, `merge_and_unload()`, grava safetensors (shards de 5 GB), tokenizer,
LICENSE Apache-2.0 do Qwen, aviso de modificacao e `merge_manifest.json` (sha256 dos arquivos, do adapter e revisao do base).
Observacao: o adapter foi treinado sobre o base NF4; fundir em bf16 muda levemente os pesos efetivos -> validar a paridade
(eval_teacher no merged bf16 x NF4+adapter) antes de usar o merged na destilacao/GGUF.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import library_versions, lock_entry, sha256_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=str(ROOT / "BaseRepos" / "Qwen2.5-Coder-7B"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="KapiTeacher-Mongo-7B-v1")
    ap.add_argument("--corpus-manifest", help="manifest.json do corpus de treino (hash gravado no merge_manifest)")
    a = ap.parse_args()
    # CPU puro: esconde a GPU (a iGPU gfx1036 do Ryzen aparece como cuda:0 e quebra o rocBLAS; o PEFT carregaria o adapter nela)
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base, adapter, out = Path(a.base).resolve(), Path(a.adapter).resolve(), Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} ja existe e nao esta vazio")
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": "cpu"}, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter, torch_device="cpu")
    model = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
        shutil.copy2(base / name, out / name)  # copia byte a byte: tokenizer.json identico ao do base e dos students (G2.1)
    shutil.copy2(base / "LICENSE", out / "LICENSE")
    entry = lock_entry(base) or {}
    (out / "MODIFICATION_NOTICE.md").write_text(
        f"# {a.name}\n\nDerivado de Qwen/Qwen2.5-Coder-7B (revisao {entry.get('revision')}), licenca Apache-2.0 (ver LICENSE).\n"
        "Modificado: adaptacao de dominio MongoDB por QLoRA (LoRA r=64) e fusao do adapter em bf16. Nao e um produto da Alibaba Cloud/Qwen.\n",
        encoding="utf-8", newline="\n")
    files = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "merge_manifest.json"}
    manifest = {"name": a.name, "base": {"path": str(base), "repo_id": entry.get("repo_id"), "revision": entry.get("revision")},
                "adapter": {"path": str(adapter), "adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                            "adapter_config": json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))},
                "adapter_train_state_fingerprint": (json.loads((adapter / "train_state.json").read_text(encoding="utf-8")).get("fingerprint")
                                                    if (adapter / "train_state.json").exists() else None),
                "corpus_manifest": ({"path": a.corpus_manifest, "sha256": sha256_file(Path(a.corpus_manifest))} if a.corpus_manifest else None),
                "dtype": "bfloat16", "files": files, "libraries": library_versions(), "merged_utc": time.strftime("%FT%TZ", time.gmtime()),
                "seconds": round(time.time() - started, 1)}
    (out / "merge_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"merge ok em {manifest['seconds']} s -> {out} ({len(files)} arquivos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
