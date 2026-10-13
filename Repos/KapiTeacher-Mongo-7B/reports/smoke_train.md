# Smoke test do treino do KapiTeacher-Mongo-7B (QLoRA NF4, corpus v2c)

Data: 2026-10-13 · GPU: RX 7800 XT (gfx1101), torch 2.10.0+rocm7.14, transformers 4.57.6, peft 0.17.1, bitsandbytes 0.50.2,
AOTriton experimental ligado (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, SDPA flash causal). Teto de VRAM 14,14 GB (`limit_vram`).
Script `scripts/teacher/train_teacher.py`, run `smoke` (`--run-name smoke`), corpus `data/teacher/v2c` (manifest sha256
`4e0a067c65187641`), shard 0 apenas (30 linhas = 117.040 tokens reais). Arquivos de evidencia nesta pasta: `smoke/metrics.jsonl`,
`smoke/eval_history.jsonl`, `smoke/run_manifest.json`. GPU usada: ~13 min (limite 15). Checkpoints/adapters do smoke ja removidos.

Configuracao do smoke (override do `sft_v2c.yaml`): 1 linha de 4.096 tokens por passo (`optim.tokens_per_step=4096`, acumulacao 1; o
job real usa 12), 30 passos, checkpoint a cada 8 passos, eval a cada 14 (val_loss em 800 exemplos + 6 FIM greedy; final: 16 FIM).
Hiperparametros iguais ao job real: LoRA r=64/alpha=128/dropout 0,05 em q,k,v,o,gate,up,down (161,5 M parametros), PagedAdamW8bit
lr 1e-4 (cosseno ate 1e-5, warmup 3%), grad clip 1,0, CE em blocos de 1.024 so no alvo, gradient checkpointing.

## Resultado

| item | resultado |
|---|---|
| Treina | sim: 30 passos, 117.040 tokens reais, sem OOM, sem paginacao (teto de 14,14 GB) |
| Loss finita | sim (verificacao por passo e abort se nao finita); media dos 5 primeiros passos 0,80 -> 5 ultimos 0,42 (ruidosa: cada passo e uma linha de composicao diferente, chat ~0,2 e FIM ~1,0) |
| Loss de validacao (800 ex., split val) | passo 14: 0,5507 (FIM 1,034 / chat 0,235) -> passo 28: 0,5247 (0,999 / 0,215) -> passo 30: 0,5225 (0,996 / 0,213): cai monotonicamente |
| Throughput (tokens reais/s) | **506 em media** (mediana 509; faixa 455-520; 7,8 s por passo de 4.096 tokens); padded 507 tok/s (padding 0,2%) |
| VRAM de pico | **8,7 GB alocados**, 10,1-10,9 GB reservados (pesos NF4 + embed/lm_head: 5,2 GB); folga de ~3,3 GB sob o teto |
| Checkpoint | atomico (`.tmp` -> rename + `COMPLETE`): adapter 646 MB (fp32) + optimizer 8-bit + RNG; 0,6-0,8 s por checkpoint; ~0,93 GB cada |
| Kill + resume | processo morto com `taskkill /F` no passo 19 (ultimo checkpoint: passo 16); `--resume auto` retomou do passo 16 e **reproduziu a loss dos passos 17-19 identica a 5 casas** (0,5594 / 0,4615 / 1,3048), confirmando adapter, optimizer, RNG e posicao no plano de linhas |
| Lock | `tmp/gpu.lock` obsoleto (pid morto) foi tomado sem intervencao; fingerprint (modelo, corpus, hiperparametros) conferido na retomada |
| Eval periodico | funciona: val_loss em 800 ex. = ~170 s; geracao greedy FIM NF4+LoRA ~3 s/exemplo (16 ex. = 48 s); EM/syntax/mongo_valid calculados (n tiny: 16 FIM, EM 0,19, syntax 0,81) |
| Parada por orcamento / pausa | implementadas (`budget.hours`, `tmp/gpu.pause` -> codigo 75); nao exercitadas no smoke |

## Observacoes e limites do smoke

- O throughput medido (506 tok/s) e **maior** que o da sonda J03b (360 tok/s a 4k): a sonda usava sequencias empacotadas de 4.096 tokens
  com atencao sobre toda a janela; aqui cada exemplo atende so a si mesmo (~270 tokens em media, linhas de n x L <= 4.096), com atencao
  causal sem mascara (kernel flash) e padding de 0,2%. O numero vale para a forma real do corpus. O smoke usou acumulacao 1 (otimizador
  a cada linha); com 12 linhas por passo o custo do otimizador cai, entao 506 tok/s e uma estimativa conservadora.
- 30 passos nao mostram convergencia (so tendencia): a decisao de qualidade e do val_loss/EM do job completo (G2.2 em eval_teacher).
- Pico de VRAM visto cresce de 8,1 para 8,7 GB conforme linhas de formatos diferentes passam (maior n x L de logits); o teto de 14,1 GB
  deixa margem. Nao houve uso de memoria compartilhada (alocador limitado: transbordo viraria OOM).
- Nao medido aqui: acumulacao 12, duracao de 20 h (calor/clock sustentado), WDDM/TDR em carga longa.
