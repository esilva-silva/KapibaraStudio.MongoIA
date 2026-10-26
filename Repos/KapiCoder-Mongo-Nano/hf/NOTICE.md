# NOTICE

KapiCoder-Mongo-Nano (version v3, tag `v0.1.0-preview`) is a derivative work of `Qwen/Qwen2.5-Coder-0.5B` (revision `8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301`),
licensed under the Apache License, Version 2.0 (see `LICENSE`).

**Modifications:** full fine-tuning (SFT) for MongoDB fill-in-the-middle autocomplete in the `editor-context-v1` format. Training data: synthetic MongoDB examples whose labels come partly from a 7B teacher
(KapiTeacher-Mongo-7B, derived from Qwen2.5-Coder-7B) validated by read-only execution, partly from queries written by Qwen3-14B (Apache-2.0) and from a grammar, each with the reference as label;
a share of generic JavaScript/TypeScript replay text. Export to safetensors bf16.
Weights are published in safetensors bf16 only; ONNX builds, if any, are published in a separate repository.

This model is not a product of Alibaba Cloud or the Qwen team and is not endorsed by them. "Qwen" is a name of its owners.
The teacher used to produce part of the training labels is not part of this repository, and neither is the training data.
The training data are synthetic (fictitious schemas); a small share of generic JavaScript/TypeScript replay text comes from npm packages under
MIT, Apache-2.0, BSD and ISC licenses, listed in `REPLAY_ATTRIBUTION.md`; that code is not redistributed.

---

# AVISO (pt-BR)

KapiCoder-Mongo-Nano (versão v3, tag `v0.1.0-preview`) é um trabalho derivado e **modificado** de `Qwen/Qwen2.5-Coder-0.5B` (revisão `8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301`), licenciado sob a Apache License 2.0 (veja `LICENSE`).
Modificações: ajuste fino completo (SFT) para autocomplete FIM de MongoDB no formato `editor-context-v1`, com rótulos em parte de um professor de 7B validados por execução somente leitura, em parte de consultas escritas pelo Qwen3-14B (Apache-2.0) e de uma gramática, mais texto de replay de JavaScript/TypeScript; exportação para safetensors bf16.
Não é um produto da Alibaba Cloud nem da equipe Qwen e não tem endosso deles. O professor e os dados de treino não fazem parte deste repositório.
