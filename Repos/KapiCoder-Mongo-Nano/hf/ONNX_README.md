---
license: apache-2.0
base_model: esilva/KapiCoder-Mongo-Nano
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
- pt
tags:
- onnx
- onnxruntime-genai
- mongodb
- code
- fim
- autocomplete
- qwen2
- synthetic-data
---

# KapiCoder-Mongo-Nano-ONNX (0.5B)

> **Status: PREVIEW / experimental, private preview, not qualified.** ONNX builds of [`esilva/KapiCoder-Mongo-Nano`](https://huggingface.co/esilva/KapiCoder-Mongo-Nano) (version v3, tag `v0.1.0-preview`)
> for `onnxruntime-genai`, made for the *Kapibara Studio* editor. Read the model card of the safetensors repository first: the model, its limits and its evaluation are described there.
> **This repository has no evaluation of its own**: the numbers in the main card were measured on the bf16 safetensors weights, not on these builds.

## What is here

One self-contained folder per variant (each can be downloaded and used on its own):

| Folder | Build |
|---|---|
| `cpu-int4/` | INT4, CPU (onnxruntime-genai, `cpu` provider) |
| `cpu-int8/` | INT8, CPU |
| `dml-fp16/` | FP16, GPU through DirectML |

Each folder holds `model.onnx` (with `model.onnx.data` when the weights are external), `genai_config.json`, the tokenizer files, `kapibarastudio-model.json`, `kapicoder_manifest.json` (SHA-256 of every file) and `export_manifest.json` (tool, version, options; local paths were removed). Builds were produced with `onnxruntime-genai` 0.15.2 (`model.type=qwen2`) from the safetensors weights of the main repository.

## Notes before using

- **The `kapi` metadata serves the Kapibara Studio application.** `kapibarastudio-model.json` (schema `kapilab-kapi-model-v1`) and `kapicoder_manifest.json` (schema `kapilab-kapicoder-manifest-v1`) are read by that editor to load and verify a model; other applications can ignore them.
- **EOS problem in onnxruntime-genai 0.15.2, also affecting the IDE.** The step that generates `<|endoftext|>` does not append that token to the sequence, so code that reads the last element of `GetSequence(0)` repeats the previous token (for example `1` becomes `11`). Read `get_next_tokens()` or check `is_done()` first. The packages are published as exported, without a workaround.
- **Quantised and converted builds can differ from the bf16 weights**, in accuracy and in the text they generate. Measure on your own prompts before relying on one. Latency depends on the host and was not measured for this card.
- The prompt format is the `editor-context-v1` header described in the main model card. The model only suggests text and never executes anything; validate field names before running a suggestion.

## License, modification notice and attribution

- License: **Apache License 2.0** (see `LICENSE`).
- **Modification notice:** derived from `Qwen/Qwen2.5-Coder-0.5B` (revision `8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301`) by fine-tuning for MongoDB fill-in-the-middle autocomplete (see the main repository) and converted to ONNX (and, where the folder name says so, quantised). Not a product of Alibaba Cloud or the Qwen team and not endorsed by them. See `NOTICE.md`.
- Attribution: base model **Qwen2.5-Coder** by the Qwen team (Hui et al., arXiv:2409.12186), Apache-2.0. Replay and data attribution are in the main repository (`REPLAY_ATTRIBUTION.md`).

---

## Resumo em português (pt-BR)

**Status: PREVIEW / experimental, preview privado, não qualificado.** Builds ONNX do `esilva/KapiCoder-Mongo-Nano` (versão v3) para o `onnxruntime-genai`, feitos para o editor Kapibara Studio.
Este repositório não tem avaliação própria: os números do card principal vêm dos pesos safetensors bf16. Pastas autocontidas por variante: `cpu-int4/`, `cpu-int8/`, `dml-fp16/`.
Os arquivos `kapibarastudio-model.json` e `kapicoder_manifest.json` (metadado `kapi`) servem ao Kapibara Studio; outros usos podem ignorá-los. O ORT GenAI 0.15.2 tem um problema de EOS (repete o último token se a aplicação ler `GetSequence(0)[^1]`), que afeta também a IDE.
Licença Apache-2.0 com aviso de modificação (derivado do Qwen2.5-Coder-0.5B, sem endosso da equipe Qwen).
