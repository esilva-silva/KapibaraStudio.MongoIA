---
license: apache-2.0
base_model: esilva/KapiCoder-Mongo-Small
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

# KapiCoder-Mongo-Small-ONNX (1.5B)

> **Status: PREVIEW / experimental, public preview, not qualified.** ONNX builds of [`esilva/KapiCoder-Mongo-Small`](https://huggingface.co/esilva/KapiCoder-Mongo-Small) (version v3, tag `v0.1.0-preview`)
> for `onnxruntime-genai`, made for the *Kapibara Studio* editor. Read the model card of the safetensors repository first: the model, its limits and its evaluation are described there.
> **This repository has no evaluation of its own**: the numbers in the main card were measured on the bf16 safetensors weights, not on these builds.

## What is here

One self-contained folder per variant (each can be downloaded and used on its own):

| Folder | Build |
|---|---|
| `cpu-int8/` | INT8, CPU (onnxruntime-genai, `cpu` provider) |
| `dml-fp16/` | FP16, GPU through DirectML |

INT4 builds are not offered at this size: INT4 builds of the previous version lost accuracy against the bf16 weights.
Each folder holds `model.onnx` (with `model.onnx.data` when the weights are external), `genai_config.json`, the tokenizer files, `kapibarastudio-model.json`, `kapicoder_manifest.json` (SHA-256 of every file) and `export_manifest.json` (tool, version, options; local paths were removed). Builds were produced with `onnxruntime-genai` 0.15.2 (`model.type=qwen2`) from the safetensors weights of the main repository.

## Notes before using

- **The `kapi` metadata serves the Kapibara Studio application.** `kapibarastudio-model.json` (schema `kapilab-kapi-model-v1`) and `kapicoder_manifest.json` (schema `kapilab-kapicoder-manifest-v1`) are read by that editor to load and verify a model; other applications can ignore them.
- **EOS integration requirement (GenAI 0.15.2):** EOS may finish generation without appending a token. The corrected IDE consumes a token only when `GetSequence(0)` grows after `GenerateNextToken`; this prevents repeating the preceding token and preserves a valid last token. Checking `IsDone()` alone after generating can discard a valid token. Python references use `get_next_tokens()`. Use the corrected IDE runtime; these are the standard exports, without alternate EOS packages.
- **Quantised and converted builds can differ from the bf16 weights**, in accuracy and in the text they generate. Measure on your own prompts before relying on one. Latency depends on the host and was not measured for this card.
- The prompt format is the `editor-context-v1` header described in the main model card. The model only suggests text and never executes anything; validate field names before running a suggestion.

## License, modification notice and attribution

- License: **Apache License 2.0** (see `LICENSE`).
- **Modification notice:** derived from `Qwen/Qwen2.5-Coder-1.5B` (revision `df3ce67c0e24480f20468b6ef2894622d69eb73b`) by fine-tuning for MongoDB fill-in-the-middle autocomplete (see the main repository) and converted to ONNX (and, where the folder name says so, quantised). Not a product of Alibaba Cloud or the Qwen team and not endorsed by them. See `NOTICE.md`.
- Attribution: base model **Qwen2.5-Coder** by the Qwen team (Hui et al., arXiv:2409.12186), Apache-2.0. Replay and data attribution are in the main repository (`REPLAY_ATTRIBUTION.md`).

---

## Resumo em português (pt-BR)

**Status: PREVIEW / experimental, preview público, não qualificado.** Builds ONNX do `esilva/KapiCoder-Mongo-Small` (versão v3) para o `onnxruntime-genai`, feitos para o editor Kapibara Studio.
Este repositório não tem avaliação própria: os números do card principal vêm dos pesos safetensors bf16. Pastas autocontidas por variante: `cpu-int8/`, `dml-fp16/`.
Os arquivos `kapibarastudio-model.json` e `kapicoder_manifest.json` (metadado `kapi`) servem ao Kapibara Studio; outros usos podem ignorá-los. A IDE corrigida consome somente tokens efetivamente anexados à sequência, evitando duplicação no EOS do GenAI 0.15.2 sem perder um último token válido. Usar apenas `IsDone()` após gerar não basta. Os pacotes permanecem os exports padrão.
Licença Apache-2.0 com aviso de modificação (derivado do Qwen2.5-Coder-1.5B, sem endosso da equipe Qwen).


## Installation in the IDE

The remote catalog uses this dedicated ONNX repository and downloads one complete variant folder, pinned to a Hub commit and verified by file hashes. Nano `cpu-int4` is the default; Small `cpu-int8` is an optional CPU alternative. Keep `editor-context-v1` and greedy autocomplete (32 output tokens, recommended context 2048). Prefix cache remains disabled pending latency and memory qualification. DirectML folders are experimental: successful packaging does not establish adapter selection or GPU performance.

For an offline preview, select the desired complete variant folder through the local ONNX model directory setting. Do not select the Safetensors root or mix files from different variants. Public download requires publication of this staged repository; a local staging folder is not a published release.
