# Smoke do treino dos alunos T0 Nano e T1 Small (2026-10-18)

`scripts/student/train_student.py`, GPU RX 7800 XT (teto de VRAM aplicado 14,14 GB, spill para memoria compartilhada bloqueado), corpus `data/student/distill/{t0,t1}`
(micro-batch 16.384 tokens, 2 linhas/passo = ~32,8k tokens, checkpointing de ativacoes, CE so no alvo). Dados brutos: `reports/student/smoke_metrics.json`
(os checkpoints, modelos e pastas `smoke_*` foram apagados). Numeros medidos nos primeiros 10-20 passos do plano real (linhas aleatorias, nao enviesadas).

## Medidas

| variante | tok/s reais (media / mediana) | s/passo | VRAM pico alocada / reservada | observacao |
|---|---:|---:|---:|---|
| **T0 full FT** (mestre fp32 + autocast bf16, AdamW fp32, LR 2e-5) | **8.231 / 8.537** | 3,5 | **11,5 / 12,8 GB** | 20 passos + retomada; cabe com folga |
| T0 LoRA r=64 (LR 2e-4) | 6.322 / 6.443 | 4,3 | 4,4 / 5,5 GB | 10 passos; ~O desempenho da v1 (6,8 k) |
| **T1 LoRA r=64** (LR 1,5e-4) | **2.259 / 2.290** | 13,0 | **9,2 / 10,3 GB** | 20 passos + retomada + merge bf16; v1: 2,4 k |
| T1 full FT, mestre fp32 + AdamW 8-bit | - | - | OOM no 1o passo (11,2 GB alocados + 594 MiB > teto 14,1 GB) | pesos 6,2 GB + grad 6,2 GB + estados 3,1 GB nao cabem |
| T1 full FT, bf16 puro + AdamW 8-bit | 830 (passo 2) | 38 | 12,8 / 14,0 GB; **OOM no 3o passo** | thrash do alocador; alem disso bf16 puro perde as atualizacoes pequenas (LR 1e-5) |

Conclusao do metodo: **T0 = full FT** (mais rapido que LoRA em ~30% e cabe: 11,5 GB); **T1 = LoRA r=64/alpha=128** (full FT nao cabe em 16 GB sem perder precisao).
A receita A/B do T0 (LoRA x full FT, Fase 05 §6.2) fica decidida pela medida de custo; a comparacao de qualidade LoRA x full FT do T0 nao foi treinada
(seria +~4 h; ver README_job do Nano, `--set model.method=lora --set optim.lr=2.0e-4 --set optim.weight_decay=0`).

## Retomada e determinismo (T0)

Perna 1: passos 1-12, checkpoint `step-000012` (modelo fp32 + estados do AdamW + RNG, 5 s); perna 2: `--resume auto` -> passos 13-20 (fingerprint aceito).
Perna 3: nova retomada explicita do mesmo `step-000012` em outra pasta de run -> passos 13-16 com **loss identica a 5 casas** (1,24593 / 0,90854 / 1,20721 / 0,96294)
e tok/s equivalente. T1: perna 1 (12 passos) + perna 2 (retomada do `step-000012`, passos 13-20, eval, adapter `final`, merge bf16 em `merged/`) sem divergencia.
Eval periodico: T0 20 FIM + 10 Generic-JS em 15 s; T1 em 53 s (val_loss incluso) -> com 100+50 exemplos ~1-2 min por avaliacao.
O smoke de avaliacao (`eval_student.py run`, n=60 + Generic-JS 500, com Mongo no WSL) passou ponta a ponta: summary, measures.json e gates.json do T0.

## Horas estimadas (otimista = tok/s medio medido; conservador = pior caso)

| job | tokens (2 epocas) | otimista | conservador | com evals/ckpt/export (otim. / cons.) | budget.hours |
|---|---:|---:|---:|---:|---:|
| J14 T0 distilado (full FT) | 88,3 M | 3,0 h (8,2 k tok/s) | 3,9 h (6,3 k = LoRA) | 3,3 / 4,2 h | 9 |
| J10b T0 gold (opcional) | 88,4 M | 3,0 h | 3,9 h | 3,3 / 4,2 h | 9 |
| J15 T1 distilado (LoRA) | 127,4 M | 15,5 h (2,29 k) | 17,7 h (2,0 k) | 16,0 / 18,2 h | 22 |
| J10 T1 gold (controle) | 127,5 M | 15,5 h | 17,7 h | 16,0 / 18,2 h | 22 |
| avaliacoes (J16a, J17a, J16, J17, J10e; +J10be) | - | - | - | 3,3 h (+0,5) | - |

**Total dos 3 treinos obrigatorios + avaliacoes: 38,6 h otimista, 43,9 h conservador.** Com o J10b opcional: 42,4 / 47,7 h.
Proposta de corte se passar de ~40 h: (A) T1 com 1,5 epoca nos DOIS bracos (`--total-steps 3422`, mantem o pareamento): -3,9 h (otim.) a -4,4 h (cons.) por braco
(-7,8 a -8,8 h no total), ficando em ~30,8 h / ~35,1 h; (B) nao rodar o J10b (-3,8 h); (C) so se necessario, 1 epoca no T1 nos dois bracos (-15,5 h otim. / -17,7 h cons.),
perdendo a receita de 2 epocas da Fase 05. Nao cortar o J10 (controle do T1): sem ele o G2.4 do professor nao tem referencia.
