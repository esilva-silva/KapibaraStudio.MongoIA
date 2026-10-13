# Exportabilidade ONNX T0/T1 (Qwen2.5-Coder 0.5B e 1.5B Base)

Builder: `onnxruntime-genai` **0.15.2** (mesma versao da IDE), ja instalado no venv ROCm (CPU). DML executado pelo `onnxruntime-genai-winml` 0.15.2 do venv existente `C:\IA\models\SlopStudio.MongoAI-Compact\.venv-winml` (nada instalado; o `onnxruntime` do venv principal so tem CPU/Azure). Pacotes em `C:\SlopStudio.MongoAI-artifacts\probe-onnx\` (fora do repo).

Paridade: 50 prompts FIM do val de `data/generated/v1/fim_ctx.jsonl` (campo `prompt`, ~234 tokens), greedy, 32 tokens novos, contra HF **fp32** (referencia limpa); entre parenteses, contra HF bf16 (o bf16 ja difere do fp32 em ~12-24% dos prompts). TTFT/tok/s: CPU 4 threads; DML na RX 7800 XT; batch 1. "EM mid" = acerto exato do `middle` real (base sem ajuste, n=50, ruido alto).

## 0.5B

| variante | tamanho | paridade greedy % (vs bf16) | prefixo medio % | EM mid % | TTFT medio / p95 (ms) | tok/s | RAM pico | notas |
|---|---|---|---|---|---|---|---|---|
| CPU fp32 | 1,9 GB | **100** (76) | 100 | 10 | 423 / 630 | 26 | 2,5 GB | prova que o grafo esta correto |
| CPU int8 (`-p int8` padrao) | 556 MB | - | - | - | - | - | - | **modelo invalido** (no `lm_head.MatMul.weight_Q4` inexistente; bug do builder 0.15.2) |
| CPU int8 `op_types_to_quantize=MatMul shared_embeddings=false` | 1,1 GB | 76 (66) | 86 | 10 | 396 / 595 | 76 | 0,8 GB | carrega e gera |
| CPU int4 padrao (RTN) | 322 MB | 16 | 35 | 4 | 438 / 673 | 110 | 0,6 GB | embedding quantizada (Gather) |
| CPU int4 MatMul-only (emb. fp32) | 841 MB | 16 | 31 | 10 | 442 / 658 | 112 | 0,6 GB | |
| CPU int4 `algo_config=k_quant_mixed` | 428 MB | 28 (28) | 42 | 8 | 428 / 655 | 96 | 0,8 GB | melhor int4 |
| DML fp16 | 964 MB | **94** (78) | 96 | 10 | **104 / 125** | 118 | 0,56 GB RSS (pesos na GPU) | |
| DML int4 (MatMul, emb. fp16) | 548 MB | 16 | 32 | 8 | 129 / 150 | 132 | 0,62 GB RSS | |

## 1.5B (feito porque o 0.5B passou: fp32 e fp16 exportam e batem)

| variante | tamanho | paridade greedy % (vs bf16) | prefixo medio % | EM mid % | TTFT medio / p95 (ms) | tok/s | RAM pico | notas |
|---|---|---|---|---|---|---|---|---|
| CPU fp32 | 5,8 GB | **100** (88) | 100 | 18 | 1.248 / 1.911 | 9,7 | 6,8 GB | |
| CPU int4 padrao | 956 MB | 34 (32) | 61 | 18 | 1.184 / 1.803 | 46 | 1,6 GB | |
| CPU int4 k_quant_mixed | 1,2 GB | 48 (50) | 65 | 12 | 1.143 / 1.734 | 39 | 1,9 GB | |
| DML fp16 | 2,9 GB | **100** (88) | 100 | 18 | **133 / 159** | 69 | 0,56 GB RSS | |
| DML int4 (MatMul, emb. fp16) | 1,3 GB | 36 (36) | 62 | 16 | 152 / 173 | 94 | 0,61 GB RSS | |

VRAM dedicada do DML nao foi medida (o contador `GPU Process Memory` retornou 0); estimar ~tamanho do pacote + KV.

## Compatibilidade com a IDE

- `genai_config.json`: `model.type = "qwen2"`, `context_length` 32768, 24 camadas / 14 heads / 2 KV (0.5B), `past_present_share_buffer: true`, `provider_options: []`. O `QwenCoderModelAdapter` (`CanHandle` somente `qwen2`) **aceitaria** os pacotes; tokenizer.json/vocab/merges sao emitidos. Idem para o 1.5B. DML nao muda o `type`; o provider e escolhido em runtime (WinML).
- Falta o `slopstudio-model.json` (hardware/contexto recomendado), que o builder nao gera.

## Conclusoes e problemas

1. Exportacao validada: fp32 CPU = 100% e fp16 DML = 94% (0.5B) / 100% (1.5B) de paridade greedy. DML reduz o TTFT 4-9x frente a CPU (104-133 ms).
2. **int4 (RTN ou k-quant) perde muito em paridade greedy (16-48%)**, embora o EM contra o `middle` real fique proximo (4-18%; n=50 e base zero-shot, sem poder estatistico). A decisao de int4 deve vir de APT/qualidade na Fase 06 apos o ajuste fino, nao deste numero. Alternativa segura: DML fp16 (964 MB / 2,9 GB) ou CPU int8 (0.5B: 76%).
3. `-p int8` puro do builder 0.15.2 gera grafo invalido na CPU (usar `op_types_to_quantize=MatMul` ou pos-quantizacao como na v1).
4. Com `tie_word_embeddings`, o int4 padrao usa `GatherBlockQuantized` (DML nao executa): usar `op_types_to_quantize=MatMul shared_embeddings=false` (exportado e testado aqui).

## Comandos reprodutiveis

```
PY=C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe
O=C:/SlopStudio.MongoAI-artifacts/probe-onnx
W=C:/IA/models/SlopStudio.MongoAI-Compact/.venv-winml/Scripts/python.exe
M=BaseRepos/Qwen2.5-Coder-0.5B   # ou 1.5B (prefixo q15)
B="$PY -I -m onnxruntime_genai.models.builder -i $M"
$B -o $O/q05-fp32-cpu -p fp32 -e cpu
$B -o $O/q05-int4-cpu -p int4 -e cpu
$B -o $O/q05-int4-kquantmixed-cpu -p int4 -e cpu --extra_options algo_config=k_quant_mixed
$B -o $O/q05-int8-matmulonly-cpu -p int8 -e cpu --extra_options op_types_to_quantize=MatMul shared_embeddings=false
$B -o $O/q05-fp16-dml -p fp16 -e dml
$B -o $O/q05-int4-dml -p int4 -e dml --extra_options op_types_to_quantize=MatMul shared_embeddings=false
$PY -I scripts/probe/onnx_parity.py ref --model $M --dtype float32 --out $O/res/ref_q05_fp32.json
$PY -I scripts/probe/onnx_parity.py run --pkg $O/q05-int4-cpu --provider cpu --ref $O/res/ref_q05_fp32.json --out $O/res/run.json
$W scripts/probe/onnx_parity.py run --pkg $O/q05-fp16-dml --provider dml --ref $O/res/ref_q05_fp32.json --out $O/res/run_dml.json   # sem -I (sitecustomize do WinML)
$PY -I scripts/probe/onnx_parity.py cmp --ref $O/res/ref_q05_fp32.json --run $O/res/run.json
```

Scripts: `scripts/probe/onnx_parity.py`; logs de exportacao: `logs/probe_onnx_export_*.log`.
