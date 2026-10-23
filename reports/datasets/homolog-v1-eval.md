# Avaliacao no conjunto de homologacao v1 (6.030 cortes FIM)

Gerado por `scripts/eval/eval_homolog.py` (gen: venv ROCm; score: `validate_text` sobre o texto real; report). Numeros agregados em `reports/datasets/homolog-v1-eval.json`; predicoes por id ficam em `data/homolog/results/` (gitignored). Conjunto selado `aca0aae5...` (28 schemas `hom`, telecom/adtech, nunca em treino); leitura so por `homolog_guard.authorized` (cada leitura/escrita em `access.log.jsonl`).

## Protocolo
- Contrato `editor-context-v1` (prompt reconstruido = prompt selado em 100/100 itens conferidos), greedy, 96 tokens, contexto 2048, padding do vocab mascarado; Professor = Qwen2.5-Coder-7B + adapter `sft_v2d/final`, NF4. Todas as geracoes na RX 7800 XT: torch reporta o dispositivo e os contadores do Windows (tabela T5) mostram 2,6 a 9,0 GB dedicados e 52 a 107% de uso do pid so no adaptador grande; os outros dois adaptadores (iGPU com ~0,34 GB dedicados e um adaptador virtual com 0) ficaram em 0% de uso.
- exec-valid: `prefix+predicao+suffix` executa no mongosh (WSL) atras do proxy somente-leitura (status ok, comando de leitura enviado), sem campo/colecao fora do schema no plano capturado, predicao nao vazia. Itens `dialect=json` (804, so `grammar`) executam como `aggregate(EJSON.parse(texto))`. Teto = mesma validacao sobre o `middle`: 93,1% (llm 100%; grammar json 100%; grammar console 82,7% — 418 itens referenciam `ConnectionPool`/`getConnection`, helpers que o mongosh nao define, e nenhum modelo pode passa-los). Campos inexistentes no teto: 0.
- Prova de leitura apenas (por execucao de score): `dbHash` antes = depois, nenhum comando de escrita encaminhado, `maxTimeMS` 3000; o `opcounters` global mostra updates de outro processo (semeadura paralela), nao do proxy.
- Intervalos de confianca: Wilson (FIM) e bootstrap de itens (exec ÷ teto); comparacoes pareadas com `compare_models` (McNemar exato + bootstrap pareado, Holm).

## Interpretacao (n=6.030; sem superlativos)
**Onde cai.** Os alunos treinados perdem em relacao ao val v2c: FIM acc T0 40,3 → 33,1 (−7,2 pp), gold 41,1 → 32,4 (−8,8), T1 41,0 → 37,0 (−4,1), professor 41,0 → 37,0 (−3,9). A queda maior esta em exec ÷ teto (T0 −18,6 pp; T1 −10,7; professor −9,5) e em campos inexistentes com contexto `full` (T0 2,1 → 15,3%; T1 1,6 → 9,2%; professor 1,9 → 11,3%). Ressalva: o teto do val v2c vem de outro validador (`chat_validate`), entao o Δ de exec ÷ teto mistura generalizacao e diferenca de validador; FIM acc e campos inexistentes sao a leitura mais limpa. As bases subiram (0,5B 7,6 → 12,6; 1,5B 14,2 → 18,7): a homologacao e mais facil para quem nao treinou (consultas mais curtas/regulares), logo a lacuna dos treinados e se anda subestimada em relacao ao poder de memorizacao do val. Os cortes `stage` e `multiline` ficam em 0–4% de FIM acc para todos (alvo longo demais para 96 tokens/igualdade exata), mas o exec ÷ teto neles segue 54–77% nos treinados.
**Distilado × gold (T0).** FIM acc +0,7 pp (IC95 [+0,1, +1,4], McNemar-Holm p=0,48: sem evidencia); exec-valid +3,1 pp [+2,2, +3,9] (p<1e-9) e campos inexistentes 15,1% × 18,1%. O ganho do distilado e em executar/campos validos, concentrado em `grammar` (+4,2 pp exec) mais que em `llm` (+1,8) e em `full` (+4,0) mais que em `irrelevant` (+1,3, IC cruza 0). Direcao igual a do val, magnitude pequena.
**T1 × T0.** T1 ganha de T0 distilado em tudo: FIM acc +3,9 pp [+3,2, +4,5], exec-valid +9,3 [+8,4, +10,2], campos inexistentes 10,1% × 15,1%, sintaxe 99,0 × 95,6; e a lacuna de T1 e a metade da de T0. Na homologacao o T1 vale o custo adicional para exec/campos; o ganho em FIM acc e modesto (cerca de 4 pp).
**T1 × professor.** FIM acc igual (−0,1 pp [−0,8, +0,6]); exec-valid T1 +1,7 pp [+0,9, +2,6] (p=7e-4) e campos inexistentes 10,1% × 11,8%. O 7B NF4+LoRA nao supera o T1 aqui: empate em FIM acc e diferenca pequena em execucao, a favor do T1. O T0 fica 3,9 pp (FIM) e 7,6 pp (exec) abaixo do professor.
**grammar × llm.** Bases: caem em `grammar` (8,5% / 12,6%) e vao melhor em `llm` (17,4% / 25,8%): estruturas de gramatica sao as mais dificeis zero-shot. Treinados: o ganho sobre a base em FIM acc e maior em `grammar` (T0 +26,2 pp) que em `llm` (+14,0), e o exec ÷ teto dos treinados e mais alto em `grammar` (T0 82,8% × 66,1%; T1 90,8 × 78,1). Isto e coerente com o aprendizado de estrutura conhecida (o corpus de treino e da mesma gramatica, schemas diferentes), e com generalizacao menor para o estilo livre do LLM: em `llm` T1 e professor empatam em FIM (36,2 × 37,1) e T0 cai a 31,4%. Como `grammar` vem de templates da mesma familia dos de treino, nao e um teste independente de estilo; o `llm` e o teste de estilo novo — e nele o exec ÷ teto cai 10–15 pp em relacao a `grammar`.
**Contexto.** FIM acc e maior sem schema visivel (none/irrelevant ~40–46%) que com `full` (28–32%) porque os cortes `full` foram alocados a alvos com mais campos/valores (field_need); nao comparar entre celulas como se a dificuldade fosse igual. Campos inexistentes nao diminuem com `full` nos alunos (T0 15,3% em full × 14,7% em none), sinal de que a falha vem de nomes de campos de schemas nunca vistos, nao so de contexto ausente — hipotese, nao testada aqui.

## Limites
- Autor Qwen3-14B / gramatica: mesma familia dos alunos, NAO consultas humanas; mede estilo/estrutura novos, nao uso real. Itens `grammar` compartilham templates com o treino.
- Executar nao e acertar a intencao: exec-valid conta consultas que rodam e citam so campos existentes; o resultado pode ser vazio ou diferente do desejado. Exec ÷ teto e razao de medias; o teto de 93,1% inclui 418 itens inexecutaveis por helpers indefinidos.
- Campos inexistentes so existem quando o comando chega ao servidor (plano capturado); predicoes que falham antes (sintaxe/static, ~6% no T0) nao entram no numerador.
- Celulas pequenas: menor `cut_kind` tem n=269 (`stmt_start`); `author × context` 546–2121; IC95 da celula nao e mostrado nas tabelas por grupo (apenas no geral). Comparacoes por grupo no JSON so para n>=50, sem correcao de multiplicidade entre grupos.
- Greedy unico, uma semente de bootstrap; sem variancia entre treinos; avaliador comparavel ao val apenas em FIM acc/edit sim./sintaxe.

## Tabelas

<!-- TABLES:BEGIN (gerado por scripts/eval/eval_homolog.py report; nao edite) -->
### T1. Homologacao v1 (n=6030), geral

| modelo | n | FIM acc (norm.) IC95 | exact | edit sim. | sintaxe | mongo-style | exec-valid | teto | exec ÷ teto IC95 | campos inex. | campos inex. (full) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Qwen2.5-Coder-0.5B base | 6030 | 12.6 [11.8, 13.5] | 12.4 | 29.9 | 56.1 | 49.3 | 36.8 | 93.1 | 39.5 [38.2, 40.8] | 8.0 | 7.0 |
| T0 Nano distilado | 6030 | 33.1 [31.9, 34.3] | 33.0 | 58.9 | 95.6 | 92.5 | 69.3 | 93.1 | 74.4 [73.2, 75.6] | 15.1 | 15.3 |
| T0 controle gold | 6030 | 32.4 [31.2, 33.6] | 32.4 | 58.4 | 95.9 | 92.8 | 66.2 | 93.1 | 71.2 [69.9, 72.4] | 18.1 | 18.7 |
| Qwen2.5-Coder-1.5B base | 6030 | 18.7 [17.8, 19.7] | 18.6 | 37.3 | 67.7 | 60.5 | 49.1 | 93.1 | 52.8 [51.5, 54.1] | 8.2 | 7.1 |
| T1 Small distilado | 6030 | 37.0 [35.8, 38.2] | 36.9 | 61.9 | 99.0 | 96.2 | 78.6 | 93.1 | 84.4 [83.4, 85.4] | 10.1 | 9.2 |
| Professor v2d (7B NF4+LoRA) | 6030 | 37.0 [35.8, 38.3] | 36.9 | 62.2 | 98.9 | 96.1 | 76.8 | 93.1 | 82.6 [81.5, 83.6] | 11.8 | 11.3 |

Celulas dos grupos abaixo: FIM acc / exec ÷ teto / campos inexistentes (em %).

### T2. Por context_kind

| context_kind (n) | Qwen2.5-Coder-0.5B base | T0 Nano distilado | T0 controle gold | Qwen2.5-Coder-1.5B base | T1 Small distilado | Professor v2d (7B NF4+LoRA) |
|---|---:|---:|---:|---:|---:|---:|
| full (3720) | 9.5 / 38.1 / 7.0 | 28.3 / 74.0 / 15.3 | 28.2 / 69.7 / 18.7 | 14.0 / 48.7 / 7.1 | 31.5 / 85.3 / 9.2 | 31.8 / 82.2 / 11.3 |
| irrelevant (1155) | 16.4 / 38.9 / 8.4 | 41.1 / 75.3 / 14.5 | 39.6 / 73.9 / 17.1 | 24.9 / 56.5 / 9.6 | 45.5 / 83.9 / 11.3 | 45.1 / 83.1 / 12.7 |
| none (1155) | 19.0 / 44.6 / 10.9 | 40.6 / 75.1 / 14.7 | 38.6 / 73.1 / 17.3 | 28.0 / 62.1 / 10.0 | 46.0 / 82.3 / 11.6 | 45.9 / 83.1 / 12.5 |

### T2. Por author

| author (n) | Qwen2.5-Coder-0.5B base | T0 Nano distilado | T0 controle gold | Qwen2.5-Coder-1.5B base | T1 Small distilado | Professor v2d (7B NF4+LoRA) |
|---|---:|---:|---:|---:|---:|---:|
| grammar (3220) | 8.5 / 33.1 / 6.4 | 34.6 / 82.8 / 13.9 | 34.4 / 78.0 / 17.4 | 12.6 / 44.0 / 7.6 | 37.6 / 90.8 / 7.2 | 37.0 / 87.8 / 9.2 |
| llm (2810) | 17.4 / 46.0 / 9.9 | 31.4 / 66.1 / 16.4 | 30.0 / 64.3 / 18.9 | 25.8 / 61.6 / 8.8 | 36.2 / 78.1 / 13.3 | 37.1 / 77.4 / 14.7 |

### T2. Por domain

| domain (n) | Qwen2.5-Coder-0.5B base | T0 Nano distilado | T0 controle gold | Qwen2.5-Coder-1.5B base | T1 Small distilado | Professor v2d (7B NF4+LoRA) |
|---|---:|---:|---:|---:|---:|---:|
| adtech (3057) | 13.4 / 40.7 / 7.4 | 33.7 / 76.2 / 13.7 | 32.8 / 72.2 / 17.3 | 18.6 / 53.2 / 8.2 | 37.6 / 85.7 / 9.3 | 38.3 / 83.9 / 10.5 |
| telecom (2973) | 11.9 / 38.3 / 8.7 | 32.5 / 72.6 / 16.5 | 31.9 / 70.0 / 19.0 | 18.9 / 52.4 / 8.2 | 36.3 / 83.1 / 10.8 | 35.8 / 81.3 / 13.1 |

### T2. Por cut_kind

| cut_kind (n) | Qwen2.5-Coder-0.5B base | T0 Nano distilado | T0 controle gold | Qwen2.5-Coder-1.5B base | T1 Small distilado | Professor v2d (7B NF4+LoRA) |
|---|---:|---:|---:|---:|---:|---:|
| key (959) | 8.8 / 35.1 / 8.1 | 27.6 / 72.4 / 18.8 | 26.2 / 68.0 / 23.2 | 18.6 / 54.9 / 5.6 | 35.7 / 84.4 / 11.6 | 33.1 / 81.0 / 14.5 |
| member (600) | 35.7 / 52.2 / 5.2 | 75.8 / 87.1 / 8.2 | 74.5 / 84.3 / 9.5 | 45.8 / 64.6 / 7.7 | 78.5 / 92.5 / 4.2 | 80.8 / 94.0 / 3.8 |
| multiline (480) | 2.1 / 32.4 / 13.5 | 1.9 / 54.4 / 35.0 | 0.4 / 44.6 / 43.5 | 2.5 / 43.7 / 14.4 | 2.7 / 67.8 / 25.0 | 3.5 / 64.3 / 27.5 |
| operator (841) | 14.0 / 40.6 / 6.7 | 43.4 / 81.1 / 8.1 | 43.5 / 79.3 / 10.9 | 23.4 / 54.0 / 5.6 | 45.5 / 89.6 / 6.1 | 47.2 / 88.2 / 8.1 |
| partial (600) | 22.5 / 37.4 / 11.7 | 54.8 / 73.4 / 18.0 | 54.7 / 73.3 / 19.3 | 33.3 / 51.8 / 12.2 | 60.8 / 80.8 / 13.3 | 61.2 / 82.0 / 12.0 |
| rest (720) | 6.7 / 33.7 / 4.4 | 26.1 / 69.5 / 11.0 | 26.4 / 67.7 / 14.3 | 7.8 / 51.0 / 5.3 | 28.7 / 81.1 / 7.6 | 28.5 / 78.0 / 12.2 |
| stage (720) | 1.4 / 38.1 / 14.3 | 3.6 / 63.0 / 26.0 | 1.9 / 54.6 / 31.5 | 3.1 / 43.6 / 13.6 | 4.0 / 76.6 / 15.8 | 5.1 / 70.3 / 20.6 |
| stmt_start (269) | 9.7 / 44.9 / 5.6 | 16.0 / 62.4 / 20.1 | 15.6 / 62.7 / 18.2 | 12.3 / 64.3 / 10.0 | 23.8 / 79.1 / 15.6 | 27.1 / 80.2 / 12.3 |
| value (841) | 13.8 / 45.0 / 4.0 | 37.6 / 92.3 / 1.9 | 37.1 / 91.9 / 2.3 | 18.7 / 53.0 / 4.9 | 42.2 / 97.7 / 1.1 | 40.0 / 97.4 / 0.8 |

### T2b. author × context_kind

| author/context (n) | Qwen2.5-Coder-0.5B base | T0 Nano distilado | T0 controle gold | Qwen2.5-Coder-1.5B base | T1 Small distilado | Professor v2d (7B NF4+LoRA) |
|---|---:|---:|---:|---:|---:|---:|
| grammar/full (2121) | 6.7 / 31.7 / 5.9 | 29.0 / 81.3 / 15.2 | 29.1 / 75.2 / 19.2 | 9.9 / 39.8 / 7.1 | 32.2 / 90.6 / 7.3 | 31.6 / 86.8 / 9.6 |
| grammar/irrelevant (553) | 10.1 / 33.5 / 6.3 | 44.5 / 86.1 / 11.2 | 44.7 / 83.5 / 13.9 | 13.7 / 45.3 / 9.2 | 46.3 / 91.0 / 6.9 | 45.8 / 88.0 / 9.8 |
| grammar/none (546) | 13.7 / 38.1 / 8.2 | 46.3 / 85.0 / 11.9 | 44.7 / 83.4 / 14.1 | 22.0 / 58.9 / 8.1 | 49.8 / 91.1 / 7.5 | 48.9 / 91.4 / 7.3 |
| llm/full (1599) | 13.2 / 45.6 / 8.5 | 27.3 / 65.5 / 15.6 | 26.9 / 63.3 / 18.0 | 19.4 / 59.0 / 7.2 | 30.6 / 79.2 / 11.8 | 32.0 / 77.0 / 13.5 |
| llm/irrelevant (602) | 22.3 / 43.4 / 10.3 | 38.0 / 66.5 / 17.6 | 35.0 / 66.1 / 20.1 | 35.0 / 65.6 / 10.0 | 44.7 / 78.1 / 15.4 | 44.5 / 79.2 / 15.4 |
| llm/none (609) | 23.6 / 49.6 / 13.3 | 35.5 / 67.3 / 17.2 | 33.2 / 65.0 / 20.2 | 33.3 / 64.5 / 11.8 | 42.5 / 75.4 / 15.3 | 43.2 / 76.7 / 17.1 |

### T3. Lacuna de generalizacao (val v2c sintetico → homologacao)

| modelo | FIM acc val → hom (Δ pp) | edit sim. val → hom (Δ) | sintaxe val → hom (Δ) | exec ÷ teto val → hom (Δ pp) | campos inex. full val → hom (Δ pp) |
|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B base | 7.6 → 12.6 (+5.0) | 22.1 → 29.9 (+7.8) | 49.8 → 56.1 (+6.3) | 34.1 → 39.5 (+5.5) | 4.1 → 7.0 (+2.9) |
| T0 Nano distilado | 40.3 → 33.1 (-7.2) | 64.8 → 58.9 (-6.0) | 99.5 → 95.6 (-3.9) | 93.0 → 74.4 (-18.6) | 2.1 → 15.3 (+13.3) |
| T0 controle gold | 41.1 → 32.4 (-8.8) | 66.2 → 58.4 (-7.8) | 99.7 → 95.9 (-3.8) | 90.3 → 71.2 (-19.2) | 2.8 → 18.7 (+15.9) |
| Qwen2.5-Coder-1.5B base | 14.2 → 18.7 (+4.5) | 30.6 → 37.3 (+6.7) | 66.7 → 67.7 (+1.0) | 50.5 → 52.8 (+2.3) | 3.8 → 7.1 (+3.4) |
| T1 Small distilado | 41.0 → 37.0 (-4.1) | 65.7 → 61.9 (-3.8) | 99.6 → 99.0 (-0.5) | 95.1 → 84.4 (-10.7) | 1.6 → 9.2 (+7.6) |
| Professor v2d (7B NF4+LoRA) | 41.0 → 37.0 (-3.9) | 65.4 → 62.2 (-3.2) | 99.6 → 98.9 (-0.7) | 92.1 → 82.6 (-9.5) | 1.9 → 11.3 (+9.4) |

(O teto do val v2c vem do validador `chat_validate` e o da homologacao do `validate_text`: a razao exec ÷ teto cruza dois validadores; compare tendencias, nao a segunda casa.)

### T4. Comparacao pareada (mesmos 100% dos itens; Δ em pp, IC95 bootstrap pareado; p McNemar-Holm)

| B − A (A → B) | n | Δ normalized_match [IC95] | Δ exec_valid [IC95] | Δ edit_similarity [IC95] | Δ no_nonexistent_field [IC95] | McNemar p (FIM / exec) |
|---|---:|---|---|---|---|---|
| T0 Nano distilado − T0 controle gold | 6030 | +0.7 [+0.1, +1.4] | +3.1 [+2.2, +3.9] | +0.005 [-0.000, +0.010] | +3.1 [+2.3, +3.9] | 0.483 / 1.73e-10 (Holm) |
| T0 Nano distilado − Qwen2.5-Coder-0.5B base | 6030 | +20.5 [+19.3, +21.7] | +32.5 [+31.0, +34.0] | +0.290 [+0.279, +0.301] | -7.0 [-8.1, -6.1] | 6.41e-242 / 3.66e-321 (Holm) |
| T0 controle gold − Qwen2.5-Coder-0.5B base | 6030 | +19.8 [+18.6, +20.9] | +29.4 [+27.9, +31.0] | +0.285 [+0.274, +0.296] | -10.1 [-11.2, -9.0] | 3.65e-224 / 5.49e-261 (Holm) |
| T1 Small distilado − T0 Nano distilado | 6030 | +3.9 [+3.2, +4.5] | +9.3 [+8.4, +10.2] | +0.030 [+0.025, +0.035] | +5.0 [+4.2, +5.8] | 2.56e-30 / 1.14e-85 (Holm) |
| T1 Small distilado − T0 controle gold | 6030 | +4.6 [+3.8, +5.4] | +12.4 [+11.4, +13.4] | +0.035 [+0.029, +0.041] | +8.1 [+7.2, +9.0] | 7.47e-31 / 2.04e-135 (Holm) |
| T1 Small distilado − Qwen2.5-Coder-1.5B base | 6030 | +18.2 [+17.1, +19.4] | +29.4 [+28.0, +30.9] | +0.246 [+0.235, +0.256] | -1.9 [-2.8, -1.0] | 3.46e-193 / 1.91e-314 (Holm) |
| T1 Small distilado − Professor v2d (7B NF4+LoRA) | 6030 | -0.1 [-0.8, +0.6] | +1.7 [+0.9, +2.6] | -0.003 [-0.008, +0.002] | +1.7 [+1.0, +2.5] | 1 / 0.000653 (Holm) |
| T0 Nano distilado − Professor v2d (7B NF4+LoRA) | 6030 | -3.9 [-4.7, -3.2] | -7.6 [-8.6, -6.6] | -0.033 [-0.039, -0.027] | -3.3 [-4.2, -2.4] | 4.94e-23 / 5.42e-50 (Holm) |
| T0 controle gold − Professor v2d (7B NF4+LoRA) | 6030 | -4.7 [-5.5, -3.9] | -10.6 [-11.6, -9.7] | -0.038 [-0.044, -0.032] | -6.4 [-7.3, -5.5] | 1.62e-29 / 2.88e-100 (Holm) |

#### T4 por author: Δ normalized_match (pp)

| B − A | author=grammar | author=llm |
|---|---|---|
| T0 Nano distilado − T0 controle gold | +0.2 [-0.7, +1.1] | +1.4 [+0.4, +2.3] |
| T0 Nano distilado − Qwen2.5-Coder-0.5B base | +26.2 [+24.7, +27.8] | +14.0 [+12.2, +15.7] |
| T0 controle gold − Qwen2.5-Coder-0.5B base | +26.0 [+24.4, +27.6] | +12.6 [+10.9, +14.3] |
| T1 Small distilado − T0 Nano distilado | +3.0 [+2.2, +3.8] | +4.9 [+3.9, +5.9] |
| T1 Small distilado − T0 controle gold | +3.2 [+2.1, +4.2] | +6.2 [+5.1, +7.4] |
| T1 Small distilado − Qwen2.5-Coder-1.5B base | +25.0 [+23.4, +26.6] | +10.5 [+8.8, +12.2] |
| T1 Small distilado − Professor v2d (7B NF4+LoRA) | +0.6 [-0.2, +1.5] | -0.9 [-2.0, +0.1] |
| T0 Nano distilado − Professor v2d (7B NF4+LoRA) | -2.4 [-3.3, -1.5] | -5.8 [-7.0, -4.5] |
| T0 controle gold − Professor v2d (7B NF4+LoRA) | -2.5 [-3.6, -1.6] | -7.1 [-8.4, -5.8] |

#### T4 por author: Δ exec_valid (pp)

| B − A | author=grammar | author=llm |
|---|---|---|
| T0 Nano distilado − T0 controle gold | +4.2 [+3.0, +5.3] | +1.8 [+0.5, +3.1] |
| T0 Nano distilado − Qwen2.5-Coder-0.5B base | +43.3 [+41.3, +45.2] | +20.1 [+17.7, +22.4] |
| T0 controle gold − Qwen2.5-Coder-0.5B base | +39.1 [+37.0, +41.1] | +18.3 [+15.9, +20.6] |
| T1 Small distilado − T0 Nano distilado | +6.9 [+5.8, +8.0] | +12.1 [+10.5, +13.6] |
| T1 Small distilado − T0 controle gold | +11.1 [+9.9, +12.3] | +13.8 [+12.3, +15.4] |
| T1 Small distilado − Qwen2.5-Coder-1.5B base | +40.7 [+38.8, +42.5] | +16.5 [+14.5, +18.6] |
| T1 Small distilado − Professor v2d (7B NF4+LoRA) | +2.6 [+1.7, +3.5] | +0.7 [-0.6, +2.1] |
| T0 Nano distilado − Professor v2d (7B NF4+LoRA) | -4.3 [-5.5, -3.1] | -11.3 [-13.0, -9.7] |
| T0 controle gold − Professor v2d (7B NF4+LoRA) | -8.5 [-9.7, -7.3] | -13.1 [-14.7, -11.5] |

#### T4 por context_kind: Δ normalized_match (pp)

| B − A | context_kind=full | context_kind=irrelevant | context_kind=none |
|---|---|---|---|
| T0 Nano distilado − T0 controle gold | +0.1 [-0.6, +0.9] | +1.5 [-0.3, +3.2] | +2.0 [+0.4, +3.5] |
| T0 Nano distilado − Qwen2.5-Coder-0.5B base | +18.8 [+17.4, +20.3] | +24.7 [+21.7, +27.7] | +21.6 [+18.8, +24.5] |
| T0 controle gold − Qwen2.5-Coder-0.5B base | +18.7 [+17.3, +20.1] | +23.2 [+20.3, +26.1] | +19.7 [+16.7, +22.6] |
| T1 Small distilado − T0 Nano distilado | +3.3 [+2.5, +4.0] | +4.3 [+2.8, +5.8] | +5.4 [+3.6, +7.2] |
| T1 Small distilado − T0 controle gold | +3.4 [+2.5, +4.3] | +5.8 [+3.8, +7.8] | +7.4 [+5.5, +9.4] |
| T1 Small distilado − Qwen2.5-Coder-1.5B base | +17.6 [+16.2, +19.0] | +20.6 [+17.7, +23.5] | +18.0 [+15.0, +21.0] |
| T1 Small distilado − Professor v2d (7B NF4+LoRA) | -0.3 [-1.0, +0.5] | +0.3 [-1.4, +2.1] | +0.1 [-1.6, +1.7] |
| T0 Nano distilado − Professor v2d (7B NF4+LoRA) | -3.5 [-4.4, -2.6] | -4.0 [-5.8, -2.1] | -5.3 [-7.4, -3.4] |
| T0 controle gold − Professor v2d (7B NF4+LoRA) | -3.6 [-4.6, -2.7] | -5.5 [-7.4, -3.5] | -7.3 [-9.3, -5.4] |

#### T4 por context_kind: Δ exec_valid (pp)

| B − A | context_kind=full | context_kind=irrelevant | context_kind=none |
|---|---|---|---|
| T0 Nano distilado − T0 controle gold | +4.0 [+2.8, +5.1] | +1.3 [-0.4, +3.1] | +1.9 [+0.3, +3.5] |
| T0 Nano distilado − Qwen2.5-Coder-0.5B base | +33.1 [+31.0, +35.0] | +34.4 [+31.0, +37.7] | +28.7 [+25.3, +31.9] |
| T0 controle gold − Qwen2.5-Coder-0.5B base | +29.1 [+27.1, +31.1] | +33.1 [+29.6, +36.5] | +26.8 [+23.4, +30.0] |
| T1 Small distilado − T0 Nano distilado | +10.5 [+9.2, +11.8] | +8.1 [+6.3, +10.0] | +6.8 [+4.8, +8.7] |
| T1 Small distilado − T0 controle gold | +14.4 [+13.1, +15.8] | +9.4 [+7.4, +11.4] | +8.7 [+6.8, +10.6] |
| T1 Small distilado − Qwen2.5-Coder-1.5B base | +33.8 [+31.9, +35.6] | +25.9 [+22.9, +29.0] | +19.0 [+15.8, +22.1] |
| T1 Small distilado − Professor v2d (7B NF4+LoRA) | +2.8 [+1.8, +4.0] | +0.7 [-1.0, +2.4] | -0.8 [-2.4, +0.9] |
| T0 Nano distilado − Professor v2d (7B NF4+LoRA) | -7.6 [-9.0, -6.3] | -7.4 [-9.5, -5.4] | -7.5 [-9.6, -5.5] |
| T0 controle gold − Professor v2d (7B NF4+LoRA) | -11.6 [-12.9, -10.3] | -8.7 [-10.7, -6.7] | -9.4 [-11.5, -7.4] |

### T5. Procedencia

| execucao | dispositivo (torch) | contadores do Windows | GB dedicados por adaptador | uso do pid por adaptador (%) | prompt = prompt selado (100 amostras) |
|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B base | AMD Radeon RX 7800 XT | ok | {'luid_0x00000000_0x00012602_phys_0': 2.604, 'luid_0x00000000_0x000147cb_phys_0': 0.0, 'luid_0x00000000_0x00014830_phys_0': 0.332} | {'luid_0x00000000_0x00012602_phys_0': 56.7, 'luid_0x00000000_0x00014830_phys_0': 0.0} | 100/100 |
| T0 Nano distilado | AMD Radeon RX 7800 XT | ok | {'luid_0x00000000_0x00012602_phys_0': 2.592, 'luid_0x00000000_0x000147cb_phys_0': 0.0, 'luid_0x00000000_0x00014830_phys_0': 0.338} | {'luid_0x00000000_0x00012602_phys_0': 52.3, 'luid_0x00000000_0x00014830_phys_0': 0.0} | 100/100 |
| T0 controle gold | AMD Radeon RX 7800 XT | ok | {'luid_0x00000000_0x00012602_phys_0': 2.592, 'luid_0x00000000_0x000147cb_phys_0': 0.0, 'luid_0x00000000_0x00014830_phys_0': 0.338} | {'luid_0x00000000_0x00012602_phys_0': 51.6, 'luid_0x00000000_0x00014830_phys_0': 0.0} | 100/100 |
| Qwen2.5-Coder-1.5B base | AMD Radeon RX 7800 XT | ok | {'luid_0x00000000_0x00012602_phys_0': 4.937, 'luid_0x00000000_0x000147cb_phys_0': 0.0, 'luid_0x00000000_0x00014830_phys_0': 0.421} | {'luid_0x00000000_0x00012602_phys_0': 88.4, 'luid_0x00000000_0x00014830_phys_0': 0.0} | 100/100 |
| T1 Small distilado | AMD Radeon RX 7800 XT | ok | {'luid_0x00000000_0x00012602_phys_0': 4.822, 'luid_0x00000000_0x000147cb_phys_0': 0.0, 'luid_0x00000000_0x00014830_phys_0': 0.412} | {'luid_0x00000000_0x00012602_phys_0': 89.1, 'luid_0x00000000_0x00014830_phys_0': 0.0} | 100/100 |
| Professor v2d (7B NF4+LoRA) | AMD Radeon RX 7800 XT | ok | {'luid_0x00000000_0x00012602_phys_0': 8.997, 'luid_0x00000000_0x000147cb_phys_0': 0.0, 'luid_0x00000000_0x00014830_phys_0': 0.337} | {'luid_0x00000000_0x00012602_phys_0': 107.4, 'luid_0x00000000_0x00014830_phys_0': 0.0} | 100/100 |
<!-- TABLES:END -->
