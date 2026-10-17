# Merge do KapiTeacher-Mongo-7B (sft_v2d/final) - 2026-10-16

- Base: `BaseRepos/Qwen2.5-Coder-7B` (Qwen/Qwen2.5-Coder-7B, rev `0396a76181e127dfc13e5c5ec48a8cee09938b02`, sha `74ac207b...a30042`).
- Adapter: `adapters/sft_v2d/final`, `adapter_model.safetensors` sha256 `3d534a7ff414ec1f0647c9f888b5c0535084a9635f7131d180debae57ea09bf3`. Corpus `data/teacher/v2d/manifest.json` sha256 `679a4ecaca8871f0eb4b564f861da727046a9fa883010bbb50756e51912d36c7`.
- Merge: `scripts/teacher/merge_teacher.py --adapter ... --out Repos/KapiTeacher-Mongo-7B/merged/sft_v2d-bf16 --corpus-manifest data/teacher/v2d/manifest.json` (CPU, bf16, 57 s). Saida 15 GB (4 shards), `merge_manifest.json` com hashes de todos os arquivos.
- Disco: C: 726 GB livres antes do merge (>= 200 GB).
- Validacao estrutural: 339 tensores, mesmas chaves/shapes, todos BF16; tokenizer.json/tokenizer_config/vocab/merges/generation_config identicos ao base (sha256); IDs FIM: prefix 151659, middle 151660, suffix 151661, pad 151662, eos 151643, im_end 151645. config.json difere so em campos de versao do transformers (dtype, sliding_window=None, layer_types), sem efeito.
- Paridade (merged carregado em NF4 vs base NF4 + adapter, mesmo contrato editor-context-v1, greedy, n=300 do val v2c, ids de `sft_v2d-final/sample_ids.json`; saida em `reports/eval/sft_v2d-merged-nf4`):
  saidas identicas 63,0% (300; @32 tokens 63,7%); em 50 amostras 58-64%. Diferencas sao de valores literais (ObjectId/datas/digitos), esperadas: o adapter foi treinado sobre o NF4 do base, e o merged re-quantiza (W_bf16+delta -> NF4), alterando o ruido de quantizacao.
- Metricas equivalentes (n=300, merged-NF4 x adapter-NF4): exact_match 0,4067 x 0,4067; edit_sim 0,6545 x 0,6543; syntax 0,990 x 0,993; mongo_valid 0,987; exec_valid 0,580 x 0,587; campos inexistentes 2,0% x 2,0%. (Referencia n=2000 do adapter: EM 0,4095, exec_valid 0,576.)
- bf16 puro nao cabe em 16 GB (15,2 GB); a comparacao com bf16 exato exigiria CPU/offload e nao foi feita. Para a destilacao (J13) usar o merged em NF4 e filtrar pelo validador, ou GGUF Q8_0 quando houver llama.cpp.
- GGUF: NAO gerado. Ausentes no disco: llama-server, llama-cli, llama-quantize, convert_hf_to_gguf.py e o pacote python `gguf` (so existe `BaseRepos/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf`, um modelo pronto). Nada foi baixado (exige aprovacao). Necessario: release llama.cpp Windows (`win-vulkan` ou `win-rocm`) + `convert_hf_to_gguf.py` da mesma tag + `pip install gguf numpy sentencepiece` (ou o repo llama.cpp). Comandos prontos:
  ```
  python convert_hf_to_gguf.py Repos\KapiTeacher-Mongo-7B\merged\sft_v2d-bf16 --outtype f16 --outfile Repos\KapiTeacher-Mongo-7B\gguf\sft_v2d-f16.gguf
  llama-quantize gguf\sft_v2d-f16.gguf gguf\sft_v2d-Q8_0.gguf Q8_0      (e Q5_K_M)
  llama-server -m gguf\sft_v2d-Q8_0.gguf -ngl 99 -c 8192 -np 8 --cont-batching
  ```
  Alternativa agora: HF NF4 + batch (vazao medida pelo agente de destilacao); depois validar top-1 >= 97% do GGUF contra NF4+adapter.
