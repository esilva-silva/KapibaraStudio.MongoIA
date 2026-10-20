#!/usr/bin/env python3
"""Passo 2 da exportacao ONNX: completa o pacote gerado pelo builder (Fase 07 §6.3/§6.4/§7.6; Fase 08 C2).

  python -I scripts/export/package_variant.py --tier t0 --variant int4 [--all]

- tokenizer/generation_config: copia BYTE A BYTE do merged (o builder re-serializa; os originais substituem os dele); remove chat_template.jinja
  (so pacotes de agente o carregam; o conteudo continua dentro do tokenizer_config.json)
- genai_config.json: `model.type` permanece `qwen2` (adapter da IDE); ajusta apenas search (greedy, max_length = contexto recomendado 2048)
- kapibarastudio-model.json (metadata `kapi`, schema `kapilab-kapi-model-v1`): contextContract editor-context-v1, paradas, contexto 2048, 32 tokens,
  runtime.minimumGenAi 0.15.2, proveniencia (base + revisao, dataset v2c, licenca)
- LICENSE (Apache-2.0), NOTICE.md (aviso de modificacao), README.md
- kapicoder_manifest.json (schema `kapilab-kapicoder-manifest-v1`; lista path+sha256 de todos os arquivos, exceto ele e export_manifest.json) e
  export_manifest.json (sha256+tamanho de TODOS os arquivos exceto ele mesmo; ferramenta/versao/comando/proveniencia)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from build_variants import TIERS, VARIANTS, package_dir  # noqa: E402

META_SCHEMA = "kapilab-kapi-model-v1"
MANIFEST_SCHEMA = "kapilab-kapicoder-manifest-v1"
VERSION = "2.0.0-preview.1"
STOP_TEXTS = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>", "<|im_end|>", "<|fim_pad|>"]
PARAMS = {"t0": "0.5B", "t1": "1.5B"}
HW = {"cpu": ["cpu"], "dml": ["gpu"]}
TOKEN_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def dump(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def package(tier: str, variant: str) -> Path:
    t, (prec, ep, extra) = TIERS[tier], VARIANTS[variant]
    pkg = package_dir(tier, variant)
    merged = ROOT / t["merged"]
    lock = json.loads((ROOT / "Repos" / t["family"] / "model.lock.json").read_text(encoding="utf-8"))
    if (pkg / ".build_info.json").exists():
        info = json.loads((pkg / ".build_info.json").read_text(encoding="utf-8"))
    elif (pkg / "export_manifest.json").exists():   # reexecucao: recupera comando/duracao do manifesto anterior
        old = json.loads((pkg / "export_manifest.json").read_text(encoding="utf-8"))["build"]
        info = {"command": old.get("command"), "seconds": old.get("build_seconds")}
    else:
        info = {}
    export_info = json.loads((merged / "export_info.json").read_text(encoding="utf-8"))
    for name in TOKEN_FILES:
        shutil.copyfile(merged / name, pkg / name)
    (pkg / "chat_template.jinja").unlink(missing_ok=True)

    # genai_config: model.type fica qwen2; so search muda
    cfg_path = pkg / "genai_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["model"]["type"] == "qwen2", cfg["model"]["type"]
    cfg["search"]["do_sample"] = False
    cfg["search"]["max_length"] = 2048
    write(cfg_path, dump(cfg))

    fam = t["family"]
    base = lock["base"]
    display = f"{fam} ONNX {variant.upper()}"
    meta = {
        "schema": META_SCHEMA,
        "name": display,
        "version": VERSION,
        "architecture": "qwen2",
        "parameters": PARAMS[tier],
        "domain": ["mongodb", "mongosh", "json", "kapibara-dsl"],
        "capabilities": ["autocomplete", "fim"],
        "hardware": HW[ep],
        "contextContract": "editor-context-v1",
        "recommendedContextTokens": 2048,
        "recommendedCompletionTokens": 32,
        "generation": {"autocomplete": {"maxTokens": 32, "temperature": 0}},
        "runtime": {"minimumGenAi": "0.15.2"},
        "stopSequences": STOP_TEXTS,
        "tier": lock["tier"],
        "variant": variant,
        "license": "Apache-2.0",
        "modification": "Modelo derivado e MODIFICADO de " + base["repo_id"] + " (ajuste fino de dominio MongoDB, destilacao do KapiTeacher-Mongo-7B; "
                        "exportado e quantizado para ONNX). Nao e um produto da Alibaba Cloud/Qwen.",
        "base": {"repo_id": base["repo_id"], "revision": base["revision"], "license": "Apache-2.0"},
        "datasetVersion": lock["dataset_version"],
        "training": {"run": lock["arms"]["distill"]["run"], "method": "SFT sequence-level FIM editor-context-v1, distilado do KapiTeacher-Mongo-7B",
                     "mergedWeightsSha256": export_info["files_sha256"]["model.safetensors"]},
        "manifest": "kapicoder_manifest.json",
        "status": "candidate-unqualified (medidas proxy do Python; o gate oficial e do KapiLab)",
    }
    write(pkg / "kapibarastudio-model.json", dump(meta))

    shutil.copyfile(ROOT / "Repos" / fam / "LICENSE", pkg / "LICENSE")
    write(pkg / "NOTICE.md", f"""# NOTICE

{fam} ONNX `{variant}` ({VERSION}) e um trabalho derivado e **modificado** de `{base['repo_id']}` (revisao `{base['revision']}`), licenciado sob
Apache License 2.0 (ver `LICENSE`).

Modificacoes: ajuste fino completo (SFT) para autocomplete FIM de MongoDB no contrato `editor-context-v1` (dataset `{lock['dataset_version']}`),
com alvos destilados do KapiTeacher-Mongo-7B (derivado de Qwen/Qwen2.5-Coder-7B, Apache-2.0; os pesos do professor nao fazem parte deste pacote);
exportacao para ONNX e quantizacao ({VARIANTS[variant][0]}, provider {VARIANTS[variant][1]}) com onnxruntime-genai 0.15.2.
Este pacote nao e um produto da Alibaba Cloud/Qwen. Os dados sinteticos (schemas ficticios) nao contem dados pessoais; o replay de codigo vem de
pacotes npm com licencas MIT/Apache-2.0/BSD/ISC.
""")
    write(pkg / "README.md", f"""# {display}

Autocomplete FIM de MongoDB para o Kapibara Studio ({PARAMS[tier]}, tier {lock['tier']}), pacote ONNX `{variant}` para {'CPU' if ep == 'cpu' else 'GPU DirectML'}.

- Contrato de contexto: `editor-context-v1` (QwenFimPromptBuilder + cabecalho do editor), contexto 2.048 tokens, ate 32 tokens gerados, greedy.
- Paradas: {', '.join(f'`{s}`' for s in STOP_TEXTS)}.
- Runtime: onnxruntime-genai >= 0.15.2 (`model.type = qwen2`, adapter Qwen da IDE).
- Base: `{base['repo_id']}` @ `{base['revision']}` (Apache-2.0). Dataset `{lock['dataset_version']}`. Licenca Apache-2.0 (`LICENSE`, `NOTICE.md`).
- Integridade: `kapicoder_manifest.json` (SHA-256 de cada arquivo) e `export_manifest.json` (ferramenta, comando, tamanhos).
- Estado: **candidato nao qualificado**. Medidas de paridade/latencia do Python sao PROXY; o gate oficial vem do KapiLab/harness da IDE.
""")

    # manifests
    build = {"tool": "onnxruntime_genai.models.builder", "tool_version": "0.15.2", "precision": prec, "execution_provider": ep,
             "extra_options": extra, "command": info.get("command"), "build_seconds": info.get("seconds")}
    (pkg / ".build_info.json").unlink(missing_ok=True)
    manifest_files = sorted(p for p in pkg.iterdir() if p.is_file() and p.name not in ("kapicoder_manifest.json", "export_manifest.json"))
    write(pkg / "kapicoder_manifest.json", dump({
        "schema": MANIFEST_SCHEMA, "package": pkg.name, "version": VERSION, "created_utc": time.strftime("%FT%TZ", time.gmtime()),
        "files": [{"path": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in manifest_files]}))
    all_files = sorted(p for p in pkg.iterdir() if p.is_file() and p.name != "export_manifest.json")
    write(pkg / "export_manifest.json", dump({
        "schema": "kapi-export-manifest-v1", "package": pkg.name, "tier": lock["tier"], "variant": variant, "created_utc": time.strftime("%FT%TZ", time.gmtime()),
        "source": {"merged_dir": t["merged"], "files_sha256": export_info["files_sha256"], "combined_sha256": export_info["combined_sha256"],
                   "dtype": export_info["dtype"], "base": base, "dataset_version": lock["dataset_version"]},
        "build": build, "total_bytes": sum(p.stat().st_size for p in all_files),
        "files": [{"path": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in all_files]}))
    print(f"[ok] {pkg.name}: {sum(p.stat().st_size for p in all_files) / 2**20:.1f} MB, {len(all_files)} arquivos")
    return pkg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS), required=True)
    ap.add_argument("--variant", action="append", choices=list(VARIANTS))
    ap.add_argument("--all", action="store_true", help="todas as variantes ja construidas do tier")
    a = ap.parse_args()
    variants = [v for v in VARIANTS if (package_dir(a.tier, v) / "genai_config.json").exists()] if a.all else (a.variant or [])
    for v in variants:
        package(a.tier, v)


if __name__ == "__main__":
    main()
