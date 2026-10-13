# Sonda T4: atencao e QLoRA do Qwen3-8B na RX 7800 XT (gfx1101, ROCm/Windows)

Dados: `attention_t4.json`, `qlora_t4_4k8k.json`, `qlora_t4_12k16k.json`, `qlora_t4_12k16k_r32_cap14.8.json`.
Venv: `C:\SlopStudio.MongoAI-artifacts\.venv` (torch 2.10.0+rocm7.14, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`). Qwen3-8B: 32 heads, 8 KV, head_dim 128, 36 camadas.

## 1. Atencao (SDPA, fwd+bwd, bf16, batch 1, causal, 1 camada)

| backend | 4k | 8k | 12k | 16k |
|---|---|---|---|---|
| FLASH (AOTriton) | 85 ms / 0,38 GB | 273 ms / 0,75 GB | 572 ms / 1,13 GB | 1.018 ms / 1,51 GB |
| EFFICIENT | 74 ms / 0,31 GB | 266 ms / 0,63 GB | 579 ms / 0,94 GB | 1.019 ms / 1,25 GB |
| MATH | 175 ms / 8,4 GB | 5.994 ms / 32,7 GB (transborda para memoria compartilhada) | OOM | OOM |

- **Kernel eficiente existe no gfx1101** (flash e mem-efficient): memoria O(L), tempo ~quadratico. Com `enable_gqa=True` so o FLASH despacha (EFFICIENT: "No available kernel"; usar `repeat_interleave` de K/V).
- Com MATH a atencao nao cabe acima de 4k. `torch.profiler` nao expoe kernels no ROCm/Windows; o despacho foi inferido pela escala de memoria.
- Atencao a 16k: 36 camadas x ~1,0 s (+ recomputo do fwd) = ~45-50 s por micro-batch so de atencao.

## 2. QLoRA NF4 + gradient checkpointing (reaproveita `bakeoff/scripts/probe_qlora.py`; PagedAdamW8bit, CE em blocos)

| micro-batch | LoRA | teto VRAM | passos | tok/s | pico alocado | pico reservado | status |
|---|---|---|---|---|---|---|---|
| 4k | r64 | 14,14 GB | 20 | **268** | 8,85 GB | 9,79 GB | ok |
| 8k | r64 | 14,14 GB | 20 | **161** | 11,41 GB | 13,37 GB | ok |
| 12k | r64 | 14,14 GB | 8 | - | 13,8 GB | 14,12 GB | **OOM** |
| 12k | r32 | 14,84 GB | 6 | **100** | 13,71 GB | 14,35 GB | ok (sem folga) |
| 16k | r64 | 14,14 GB | 8 | - | 13,7 GB | 14,04 GB | **OOM** |
| 16k | r32 | 14,84 GB | 6 | - | 14,25 GB | - | **OOM** |

Sem offload de checkpoints para a RAM nem MLP em blocos (nao implementados na sonda), 16k nao cabe; 12k cabe so com r32 e teto de 14,8 GB.

## 3. Recomendacao

1. **16k direto: reprovado** na receita atual (OOM mesmo com r32 e teto 14,8 GB). Exigiria offload assincrono de checkpoints + MLP em blocos (§6.4c), ainda nao validados, e ficaria na faixa de ~65-70 tok/s (estimativa).
2. **Plano B: treinar a 8k** (161 tok/s, 11,4 GB, folga de ~3 GB), buckets 2k/4k/8k (4k a 268 tok/s), com **extensao curta a 12k, LoRA r32** (100 tok/s, no limite da VRAM) para uma fatia pequena dos dados e gate de qualidade por faixa de contexto.
3. **Tempo**: os tok/s medidos ficam ~40% abaixo da estimativa do doc (380-450 ate 8k). 80 M tokens (70% <=8k a ~200 tok/s, 30% a 12k a 100) = ~145 h; tudo a 8k = ~85-130 h. Reduzir o orcamento de tokens ou aceitar 5-6 dias de GPU; atualizar §8/§9 do doc 05.

Reproducao: `scripts/probe/probe_attention.py` e `scripts/probe/probe_qlora_t4.py` (flags `--micro-tokens`, `--rank`, `--vram-fraction`, `--reserve-gb`); logs em `logs/probe_attention.log`, `logs/probe_qlora_t4*.log`.
