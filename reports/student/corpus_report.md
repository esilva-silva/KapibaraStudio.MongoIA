# Corpus dos alunos T0 Nano / T1 Small (autocomplete FIM, editor-context-v1)

Gerado 2026-10-18T18:09:15 por `scripts/student/build_student_corpus.py` (config `configs/student_corpus.yaml`, seed 20261018, git 31c30db5dcf4, 87 s). Dados em `data/student/{distill,gold}/{t0,t1}/` (gitignored).

## Bracos e pareamento

- **distill**: `fim_ctx` e `completion` rotulados pelo professor (alvo `teacher_seq` de `data/distilled/v2c-final/accepted.jsonl`, sem usar o gold para escolher candidata; `agrees_with_gold` so nos metadados).
- **gold (controle/J10)**: os MESMOS ids, prompts, componentes, ordem e linhas (micro-batches), com o `middle` original como rotulo.
- `fim_raw` (gold) e `replay` (codigo generico) sao IDENTICOS nos dois bracos. Linhas empacotadas por `prompt + max(alvo distilado, alvo gold)` -> mesma composicao de lote nos dois bracos.

## T1 KapiCoder-Mongo-Small (`BaseRepos/Qwen2.5-Coder-1.5B`)

- **229,834 exemplos** (217,306 Mongo + 12,528 replay), **63,702,244 tokens/epoca** (media 277.2/ex.; alvo com perda 8,459,904 = 13.28%), 4,563 linhas de <= 16384 tokens, **2,281 passos/epoca** (accum 2, ~32,8k tokens/passo) -> 4,562 passos em 2 epocas (127,404,488 tokens).
- Gold: 63,725,889 tokens/epoca (a diferenca vem so do comprimento dos alvos do professor).
- Exemplos Mongo cujo rotulo do professor difere do gold: **117,585 (54.11%)**; concordam com o gold (informativo): 43.91%.
- Replay: 12,528 trechos, 7,718,144 tokens = **12.12%** dos tokens (alvo 12.00%); 4429 arquivos de 555 pacotes; por tipo {'causal': 6228, 'fim': 6300}.

| componente | exemplos | % exemplos | tokens (distill) | alvo c/ perda (distill) | rotulo difere do gold |
|---|---:|---:|---:|---:|---:|
| completion | 57,132 | 24.86% | 14,602,878 | 2,073,280 | 32,792 |
| fim_ctx | 149,170 | 64.90% | 40,291,892 | 2,307,436 | 84,793 |
| fim_raw | 11,004 | 4.79% | 1,089,330 | 170,590 | 0 |
| replay_causal | 6,228 | 2.71% | 3,839,239 | 3,839,239 | 0 |
| replay_fim | 6,300 | 2.74% | 3,878,905 | 69,359 | 0 |

Mistura alvo (sobre o Mongo): fim_ctx 0.68, completion 0.27, fim_raw 0.05; obtido: {'completion': 0.2629, 'fim_ctx': 0.6865, 'fim_raw': 0.0506}.

Balanco da selecao Mongo (estratos cut_kind x categoria x context_kind) contra o POOL de 300k (alvo; ele proprio a <= 0,3 p.p. do treino em cut_kind/categoria/dominio/papel do cursor, context_kind rebalanceado para ~29% none+irrelevant):

| dimensao | desvio max. selecao x pool (p.p.) | desvio max. aceito bruto x pool (p.p.) |
|---|---:|---:|
| cut_kind | 0.42 | 1.164 |
| category | 0.151 | 1.146 |
| context_kind | 0.674 | 2.76 |
| domain | 0.146 | 0.11 |
| cursor_role | 0.42 | 1.164 |
| origin | 0.267 | 1.023 |

context_kind: full 71.92%, irrelevant 14.70%, none 13.37%  |  cut_kind: key 16.28%, member 10.10%, multiline 7.49%, operator 13.62%, partial 10.13%, rest 11.93%, stage 12.52%, stmt_start 3.64%, value 14.29%

Descartes na renderizacao: {'completion_gt_96': 1860}; fracao elegivel a completion 0.8626.

### Verificacao independente (T1)

- ids/componentes/ordem iguais nos dois bracos: **True**; linhas iguais: **True**; problemas: **0**; `ids_sha256` 9937b346325f0595...
- Mongo: 217,306 ids; em train.ids.txt (base): 217,306; em val/test/bench: 0/0/0.
- Replay isolamento: {'package_overlap_with_generic_eval': [], 'file_sha256_overlap_with_generic_eval': 0, 'chunks_containing_eval_cut': 0, 'chunks_matching_mongo_eval_content': 0, 'chunks_dropped_containing_eval_cut': 249, 'source_manifest_isolation': {'package_overlap_train_x_eval': [], 'file_sha256_overlap_train_x_eval': 0, 'normalized_content_overlap': 0}}

## T0 KapiCoder-Mongo-Nano (`BaseRepos/Qwen2.5-Coder-0.5B`)

- **159,865 exemplos** (151,180 Mongo + 8,685 replay), **44,164,423 tokens/epoca** (media 276.3/ex.; alvo com perda 5,903,432 = 13.37%), 3,169 linhas de <= 16384 tokens, **1,584 passos/epoca** (accum 2, ~32,8k tokens/passo) -> 3,168 passos em 2 epocas (88,328,846 tokens).
- Gold: 44,185,000 tokens/epoca (a diferenca vem so do comprimento dos alvos do professor).
- Exemplos Mongo cujo rotulo do professor difere do gold: **82,047 (54.27%)**; concordam com o gold (informativo): 43.79%.
- Replay: 8,685 trechos, 5,351,686 tokens = **12.12%** dos tokens (alvo 12.00%); 4429 arquivos de 555 pacotes; por tipo {'causal': 4315, 'fim': 4370}.

| componente | exemplos | % exemplos | tokens (distill) | alvo c/ perda (distill) | rotulo difere do gold |
|---|---:|---:|---:|---:|---:|
| completion | 39,938 | 24.98% | 10,165,468 | 1,450,319 | 23,001 |
| fim_ctx | 103,581 | 64.79% | 27,884,759 | 1,613,970 | 59,046 |
| fim_raw | 7,661 | 4.79% | 762,510 | 120,961 | 0 |
| replay_causal | 4,315 | 2.70% | 2,669,959 | 2,669,959 | 0 |
| replay_fim | 4,370 | 2.73% | 2,681,727 | 48,223 | 0 |

Mistura alvo (sobre o Mongo): fim_ctx 0.68, completion 0.27, fim_raw 0.05; obtido: {'completion': 0.2642, 'fim_ctx': 0.6852, 'fim_raw': 0.0507}.

Balanco da selecao Mongo (estratos cut_kind x categoria x context_kind) contra o POOL de 300k (alvo; ele proprio a <= 0,3 p.p. do treino em cut_kind/categoria/dominio/papel do cursor, context_kind rebalanceado para ~29% none+irrelevant):

| dimensao | desvio max. selecao x pool (p.p.) | desvio max. aceito bruto x pool (p.p.) |
|---|---:|---:|
| cut_kind | 0.042 | 1.164 |
| category | 0.089 | 1.146 |
| context_kind | 0.036 | 2.76 |
| domain | 0.147 | 0.11 |
| cursor_role | 0.106 | 1.164 |
| origin | 0.055 | 1.023 |

context_kind: full 71.29%, irrelevant 15.10%, none 13.61%  |  cut_kind: key 16.14%, member 10.09%, multiline 7.88%, operator 13.50%, partial 10.04%, rest 11.83%, stage 12.49%, stmt_start 3.88%, value 14.16%

Descartes na renderizacao: {'completion_gt_96': 1284}; fracao elegivel a completion 0.8626.

### Verificacao independente (T0)

- ids/componentes/ordem iguais nos dois bracos: **True**; linhas iguais: **True**; problemas: **0**; `ids_sha256` 628510727e813061...
- Mongo: 151,180 ids; em train.ids.txt (base): 151,180; em val/test/bench: 0/0/0.
- Replay isolamento: {'package_overlap_with_generic_eval': [], 'file_sha256_overlap_with_generic_eval': 0, 'chunks_containing_eval_cut': 0, 'chunks_matching_mongo_eval_content': 0, 'chunks_dropped_containing_eval_cut': 249, 'source_manifest_isolation': {'package_overlap_train_x_eval': [], 'file_sha256_overlap_train_x_eval': 0, 'normalized_content_overlap': 0}}

## Notas de dimensionamento

- Volume pela receita da Fase 05 §6.2 em EXEMPLOS (T0 ~160k, T1 ~230k); em tokens ficou abaixo dos ~62 M / ~100 M estimados na spec porque o prompt `editor-context-v1` medido tem ~270 tokens (nao ~370) e o alvo ~13: ~277 tokens/ex. em media. A GPU custa por token, nao por exemplo: ver `configs/gpu_queue.yaml` (horas medidas no smoke).
- Mistura: a spec previa 5-23% de chat/correcao por tier; com o chat fora de escopo (Q1) esse percentual foi para FIM/completion, e o replay sobe de 3-7% para ~12% dos tokens (faixa 10-15% pedida). O replay de origem tem so ~7,2k trechos (4,45 M tokens): para chegar a 12% o mesmo trecho entra ate 3x com outro sorteio FIM/causal e corte (`max_variants`); as variantes 0 de todos os trechos vem antes de qualquer variante 1.
- `fim_raw` (5%) usa o gold nos dois bracos e `completion` usa rotulo + sufixo original: o efeito do professor fica isolado em fim_ctx + completion (~90% dos exemplos Mongo). Onde o professor concorda com o gold (~44%), os dois bracos sao identicos token a token.
- T0 e subconjunto de T1 (mesmo teto por estrato, mesma ordem de prioridade); o replay do T0 e prefixo do replay do T1.

## Isolamento G2.7 (por registro, aborta em qualquer violacao)

Registros lidos do aceito: 270993; descartados por isolamento: 0 (esperado 0). Checagens: `Isolation` (split, holdout, papel do schema, id/parent_id, grupo, schema, snippet, prompt_sha256, prefixo+sufixo, cego), id em `train.ids.txt` e fora de `val/test/bench.ids.txt`, hash de conteudo (prefixo+sufixo+middle/alvo) fora de `eval_docs.jsonl` para os DOIS rotulos, val do treino (copiado de `data/teacher/v2d/val`) sem ids do treino dos alunos, replay x `generic_js_fim_eval`/monitor (pacote, sha de arquivo, corte, conteudo). Testes: `scripts/student/tests/`.

Pool de balanco: 300,000 prompts. Nenhum arquivo de `data/eval/blind` e lido (so hashes do manifesto via `blind_guard`).

