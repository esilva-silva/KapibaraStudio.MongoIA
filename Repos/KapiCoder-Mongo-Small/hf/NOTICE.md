# NOTICE

KapiCoder-Mongo-Small is a derivative work of `Qwen/Qwen2.5-Coder-1.5B` (revision `df3ce67c0e24480f20468b6ef2894622d69eb73b`),
licensed under the Apache License, Version 2.0 (see `LICENSE`).

**Modifications:** LoRA fine-tuning (r=64) merged into the weights, for MongoDB fill-in-the-middle autocomplete in the `editor-context-v1` format, with labels distilled from a 7B teacher and validated by read-only execution; export to safetensors bf16 and ONNX.
Weights are published in safetensors bf16 and, optionally, as ONNX packages built with onnxruntime-genai 0.15.2.

This model is not a product of Alibaba Cloud or the Qwen team and is not endorsed by them. "Qwen" is a name of its owners.
The teacher used to produce training labels (KapiTeacher-Mongo-7B, derived from Qwen2.5-Coder-7B, Apache-2.0) is not part of this repository.
The training data are synthetic (fictitious schemas); a small share of generic JavaScript/TypeScript replay text comes from npm packages under
MIT, Apache-2.0, BSD and ISC licenses, listed in `REPLAY_ATTRIBUTION.md`; that code is not redistributed.

---

# AVISO (pt-BR)

KapiCoder-Mongo-Small e um trabalho derivado e **modificado** de `Qwen/Qwen2.5-Coder-1.5B` (revisao `df3ce67c0e24480f20468b6ef2894622d69eb73b`), licenciado sob a Apache License 2.0 (veja `LICENSE`).
Modificacoes: ajuste fino LoRA (r=64) mesclado nos pesos, para autocomplete FIM de MongoDB no formato `editor-context-v1`, com rotulos destilados de um professor de 7B e validados por execucao somente leitura; exportacao para safetensors bf16 e ONNX.
Nao e um produto da Alibaba Cloud nem da equipe Qwen e nao tem endosso deles.
