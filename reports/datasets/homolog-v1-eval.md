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
| Qwen2.5-Coder-1.5B base | 6030 | 18.7 [17.8, 19.7] | 18.6 | 37.3 | 67.7 | 60.5 | 49.1 | 93.1 | 52.8 [51.5, 54.1] | 8.2 | 7.1 |
| T1 Small distilado | 6030 | 37.0 [35.8, 38.2] | 36.9 | 61.9 | 99.0 | 96.2 | 78.6 | 93.1 | 84.4 [83.4, 85.4] | 10.1 | 9.2 |
| T1 Small v3 (corpus v3) | 6030 | 45.4 [44.1, 46.6] | 45.2 | 70.1 | 99.7 | 97.2 | 83.1 | 93.1 | 89.3 [88.4, 90.1] | 7.9 | 7.5 |

Celulas dos grupos abaixo: FIM acc / exec ÷ teto / campos inexistentes (em %).

### T2. Por context_kind

| context_kind (n) | Qwen2.5-Coder-1.5B base | T1 Small distilado | T1 Small v3 (corpus v3) |
|---|---:|---:|---:|
| full (3720) | 14.0 / 48.7 / 7.1 | 31.5 / 85.3 / 9.2 | 38.3 / 88.8 / 7.5 |
| irrelevant (1155) | 24.9 / 56.5 / 9.6 | 45.5 / 83.9 / 11.3 | 55.5 / 90.0 / 8.2 |
| none (1155) | 28.0 / 62.1 / 10.0 | 46.0 / 82.3 / 11.6 | 58.1 / 89.9 / 8.7 |

### T2. Por author

| author (n) | Qwen2.5-Coder-1.5B base | T1 Small distilado | T1 Small v3 (corpus v3) |
|---|---:|---:|---:|
| grammar (3220) | 12.6 / 44.0 / 7.6 | 37.6 / 90.8 / 7.2 | 38.4 / 91.9 / 5.3 |
| llm (2810) | 25.8 / 61.6 / 8.8 | 36.2 / 78.1 / 13.3 | 53.4 / 86.6 / 10.8 |

### T2. Por domain

| domain (n) | Qwen2.5-Coder-1.5B base | T1 Small distilado | T1 Small v3 (corpus v3) |
|---|---:|---:|---:|
| adtech (3057) | 18.6 / 53.2 / 8.2 | 37.6 / 85.7 / 9.3 | 45.8 / 89.5 / 7.7 |
| telecom (2973) | 18.9 / 52.4 / 8.2 | 36.3 / 83.1 / 10.8 | 45.0 / 89.0 / 8.0 |

### T2. Por cut_kind

| cut_kind (n) | Qwen2.5-Coder-1.5B base | T1 Small distilado | T1 Small v3 (corpus v3) |
|---|---:|---:|---:|
| key (959) | 18.6 / 54.9 / 5.6 | 35.7 / 84.4 / 11.6 | 44.0 / 91.3 / 6.9 |
| member (600) | 45.8 / 64.6 / 7.7 | 78.5 / 92.5 / 4.2 | 83.5 / 92.9 / 4.0 |
| multiline (480) | 2.5 / 43.7 / 14.4 | 2.7 / 67.8 / 25.0 | 11.2 / 73.3 / 19.0 |
| operator (841) | 23.4 / 54.0 / 5.6 | 45.5 / 89.6 / 6.1 | 57.6 / 93.4 / 5.2 |
| partial (600) | 33.3 / 51.8 / 12.2 | 60.8 / 80.8 / 13.3 | 65.7 / 85.9 / 11.3 |
| rest (720) | 7.8 / 51.0 / 5.3 | 28.7 / 81.1 / 7.6 | 40.0 / 89.9 / 6.8 |
| stage (720) | 3.1 / 43.6 / 13.6 | 4.0 / 76.6 / 15.8 | 10.6 / 82.0 / 13.9 |
| stmt_start (269) | 12.3 / 64.3 / 10.0 | 23.8 / 79.1 / 15.6 | 37.2 / 89.0 / 8.9 |
| value (841) | 18.7 / 53.0 / 4.9 | 42.2 / 97.7 / 1.1 | 49.6 / 98.0 / 1.1 |

### T2b. author × context_kind

| author/context (n) | Qwen2.5-Coder-1.5B base | T1 Small distilado | T1 Small v3 (corpus v3) |
|---|---:|---:|---:|
| grammar/full (2121) | 9.9 / 39.8 / 7.1 | 32.2 / 90.6 / 7.3 | 32.3 / 91.1 / 5.3 |
| grammar/irrelevant (553) | 13.7 / 45.3 / 9.2 | 46.3 / 91.0 / 6.9 | 47.9 / 93.7 / 4.9 |
| grammar/none (546) | 22.0 / 58.9 / 8.1 | 49.8 / 91.1 / 7.5 | 52.0 / 93.5 / 5.7 |
| llm/full (1599) | 19.4 / 59.0 / 7.2 | 30.6 / 79.2 / 11.8 | 46.2 / 86.2 / 10.4 |
| llm/irrelevant (602) | 35.0 / 65.6 / 10.0 | 44.7 / 78.1 / 15.4 | 62.5 / 87.0 / 11.3 |
| llm/none (609) | 33.3 / 64.5 / 11.8 | 42.5 / 75.4 / 15.3 | 63.5 / 87.0 / 11.5 |

### T3. Lacuna de generalizacao (val v2c sintetico → homologacao)

| modelo | FIM acc val → hom (Δ pp) | edit sim. val → hom (Δ) | sintaxe val → hom (Δ) | exec ÷ teto val → hom (Δ pp) | campos inex. full val → hom (Δ pp) |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B base | 14.2 → 18.7 (+4.5) | 30.6 → 37.3 (+6.7) | 66.7 → 67.7 (+1.0) | 50.5 → 52.8 (+2.3) | 3.8 → 7.1 (+3.4) |
| T1 Small distilado | 41.0 → 37.0 (-4.1) | 65.7 → 61.9 (-3.8) | 99.6 → 99.0 (-0.5) | 95.1 → 84.4 (-10.7) | 1.6 → 9.2 (+7.6) |
| T1 Small v3 (corpus v3) | 41.0 → 45.4 (+4.4) | 65.4 → 70.1 (+4.7) | 99.8 → 99.7 (-0.1) | 96.2 → 89.3 (-6.9) | 1.2 → 7.5 (+6.3) |

(O teto do val v2c vem do validador `chat_validate` e o da homologacao do `validate_text`: a razao exec ÷ teto cruza dois validadores; compare tendencias, nao a segunda casa.)

### T4. Comparacao pareada (mesmos 100% dos itens; Δ em pp, IC95 bootstrap pareado; p McNemar-Holm)

| B − A (A → B) | n | Δ normalized_match [IC95] | Δ exec_valid [IC95] | Δ edit_similarity [IC95] | Δ no_nonexistent_field [IC95] | McNemar p (FIM / exec) |
|---|---:|---|---|---|---|---|
| T1 Small distilado − Qwen2.5-Coder-1.5B base | 6030 | +18.2 [+17.1, +19.4] | +29.4 [+28.0, +30.9] | +0.246 [+0.235, +0.256] | -1.9 [-2.8, -1.0] | 3.46e-193 / 1.91e-314 (Holm) |
| T1 Small v3 (corpus v3) − T1 Small distilado | 6030 | +8.4 [+7.6, +9.3] | +4.5 [+3.7, +5.3] | +0.082 [+0.076, +0.089] | +2.2 [+1.5, +2.9] | 1.05e-87 / 4.59e-27 (Holm) |
| T1 Small v3 (corpus v3) − Qwen2.5-Coder-1.5B base | 6030 | +26.6 [+25.4, +27.8] | +33.9 [+32.6, +35.3] | +0.328 [+0.318, +0.338] | +0.3 [-0.5, +1.1] | 0 / 0 (Holm) |

#### T4 por author: Δ normalized_match (pp)

| B − A | author=grammar | author=llm |
|---|---|---|
| T1 Small distilado − Qwen2.5-Coder-1.5B base | +25.0 [+23.4, +26.6] | +10.5 [+8.8, +12.2] |
| T1 Small v3 (corpus v3) − T1 Small distilado | +0.7 [-0.0, +1.6] | +17.2 [+15.7, +18.7] |
| T1 Small v3 (corpus v3) − Qwen2.5-Coder-1.5B base | +25.7 [+24.2, +27.3] | +27.7 [+25.9, +29.4] |

#### T4 por author: Δ exec_valid (pp)

| B − A | author=grammar | author=llm |
|---|---|---|
| T1 Small distilado − Qwen2.5-Coder-1.5B base | +40.7 [+38.8, +42.5] | +16.5 [+14.5, +18.6] |
| T1 Small v3 (corpus v3) − T1 Small distilado | +1.0 [+0.2, +1.9] | +8.4 [+7.0, +9.8] |
| T1 Small v3 (corpus v3) − Qwen2.5-Coder-1.5B base | +41.7 [+39.8, +43.6] | +25.0 [+23.0, +27.0] |

#### T4 por context_kind: Δ normalized_match (pp)

| B − A | context_kind=full | context_kind=irrelevant | context_kind=none |
|---|---|---|---|
| T1 Small distilado − Qwen2.5-Coder-1.5B base | +17.6 [+16.2, +19.0] | +20.6 [+17.7, +23.5] | +18.0 [+15.0, +21.0] |
| T1 Small v3 (corpus v3) − T1 Small distilado | +6.7 [+5.8, +7.7] | +10.0 [+8.0, +12.2] | +12.1 [+10.0, +14.3] |
| T1 Small v3 (corpus v3) − Qwen2.5-Coder-1.5B base | +24.3 [+22.8, +25.8] | +30.6 [+27.8, +33.5] | +30.1 [+27.3, +33.0] |

#### T4 por context_kind: Δ exec_valid (pp)

| B − A | context_kind=full | context_kind=irrelevant | context_kind=none |
|---|---|---|---|
| T1 Small distilado − Qwen2.5-Coder-1.5B base | +33.8 [+31.9, +35.6] | +25.9 [+22.9, +29.0] | +19.0 [+15.8, +22.1] |
| T1 Small v3 (corpus v3) − T1 Small distilado | +3.3 [+2.2, +4.2] | +5.8 [+4.1, +7.5] | +7.1 [+5.3, +8.9] |
| T1 Small v3 (corpus v3) − Qwen2.5-Coder-1.5B base | +37.0 [+35.2, +38.8] | +31.7 [+28.8, +34.6] | +26.1 [+23.3, +28.9] |

### T5. Procedencia

| execucao | dispositivo (torch) | contadores do Windows | GB dedicados por adaptador | uso do pid por adaptador (%) | prompt = prompt selado (100 amostras) |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B base | AMD Radeon RX 7800 XT | ok | {'luid_0x00000000_0x00012602_phys_0': 4.937, 'luid_0x00000000_0x000147cb_phys_0': 0.0, 'luid_0x00000000_0x00014830_phys_0': 0.421} | {'luid_0x00000000_0x00012602_phys_0': 88.4, 'luid_0x00000000_0x00014830_phys_0': 0.0} | 100/100 |
| T1 Small distilado | AMD Radeon RX 7800 XT | ok | {'luid_0x00000000_0x00012602_phys_0': 4.822, 'luid_0x00000000_0x000147cb_phys_0': 0.0, 'luid_0x00000000_0x00014830_phys_0': 0.412} | {'luid_0x00000000_0x00012602_phys_0': 89.1, 'luid_0x00000000_0x00014830_phys_0': 0.0} | 100/100 |
| T1 Small v3 (corpus v3) | AMD Radeon RX 7800 XT | ok | {'luid_0x00000000_0x00012602_phys_0': 3.564, 'luid_0x00000000_0x000147cb_phys_0': 0.0, 'luid_0x00000000_0x00014830_phys_0': 0.393} | {'luid_0x00000000_0x00012602_phys_0': 74.0, 'luid_0x00000000_0x00014830_phys_0': 0.0} | 100/100 |
<!-- TABLES:END -->
