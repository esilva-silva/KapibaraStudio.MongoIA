# Dataset v2 - relatorio de validacao (pipeline de 9 etapas)

Gerado por `scripts/data/validate_pipeline.py` (`validate-pipeline/1`), seed 20261008, commit `6c1023cdee78` (arvore com alteracoes). Execucao (etapa 5): `live`; orcamento (etapa 7): `qwen2.5-coder tokenizer.json`; sintaxe JS: `node vm.Script`.

## 1. Funil por etapa

| etapa | entrada | aprovados | rejeitados | fim (ent->saida) | chat (ent->saida) |
|---|---:|---:|---:|---|---|
| sanitize | 1240000 | 1239990 | 10 | 1200000->1199990 | 40000->40000 |
| syntax | 1239990 | 1239990 | 0 | 1199990->1199990 | 40000->40000 |
| api | 1239990 | 1239990 | 0 | 1199990->1199990 | 40000->40000 |
| fields | 1239990 | 1239990 | 0 | 1199990->1199990 | 40000->40000 |
| exec | 1239990 | 1239990 | 0 | 1199990->1199990 | 40000->40000 |
| semantic | 1239990 | 1239990 | 0 | 1199990->1199990 | 40000->40000 |
| budget | 1239990 | 1239981 | 9 | 1199990->1199990 | 40000->39991 |
| dedup | 1239981 | 1214211 | 25770 | 1199990->1174831 | 39991->39380 |
| decontam | 1214211 | 1158502 | 55709 | 1174831->1125770 | 39380->32732 |

Entrada: 1240000 registros (1200000 fim_ctx + 40000 chat). Saida: 1158502 (train 1134248, val 13290, test 6657, bench 4307).

### Principais motivos de rejeicao

| etapa | motivo | n |
|---|---|---:|
| decontam | coincide com avaliacao: ngram# | 53283 |
| dedup | near-dup MinHash J~# >= # | 18506 |
| dedup | duplicata exata | 7264 |
| decontam | coincide com avaliacao: approx_key | 2282 |
| decontam | coincide com avaliacao: ngram#,approx_key | 100 |
| decontam | coincide com avaliacao: exact,minhash | 43 |
| sanitize | numero de cartao (Luhn) | 10 |
| budget | resposta com # tokens > # | 9 |
| decontam | coincide com avaliacao: ngram#,minhash | 1 |

## 2. Decontaminacao (etapa 9)

- Treino de entrada: 1189957 registros; removidos: **55709** (4.7%); mantidos 1134248.
- Documentos de avaliacao indexados: 35032 {'val': 12386, 'test': 6431, 'bench': 4215, 'legacy-v1': 12000}. Conjunto cego: **not_sealed**.

| regra | removidos (qualquer coincidencia) | removidos (1a regra) |
|---|---:|---:|
| blind | 0 | 0 |
| exact | 43 | 43 |
| ngram13 | 53384 | 53384 |
| approx_key | 2382 | 2282 |
| minhash | 44 | 0 |

Sensibilidade do 13-grama (registros de treino removidos se o 13-grama exigisse >= k tokens de conteudo): k=0: 53384, k=1: 45754, k=2: 9399, k=3: 749, k=4: 201 (aplicado: k=0, 0 = regra literal da spec).

Parametros: {'ngram': 13, 'minhash_perm': 128, 'shingle_k': 5, 'bands': 25, 'rows': 5, 'jaccard': 0.7, 'seed': 20261008, 'ngram_min_content': 0}. Relatorio completo: `reports/decontam/61c4085cd0d57126.json`. Verificacao pos-remocao (2a passada treino x avaliacao): 0 registros ainda coincidentes.

### Sobrevivencia por cut_kind (train FIM, regra literal) - base para `fim.decontam_survival`

| cut_kind | entrada (etapas 1-8) | mantidos | sobrevivencia |
|---|---:|---:|---:|
| key | 207164 | 198569 | 0.959 |
| member | 142492 | 136812 | 0.960 |
| multiline | 77660 | 74474 | 0.959 |
| operator | 187508 | 179329 | 0.956 |
| partial | 142220 | 136601 | 0.961 |
| rest | 159810 | 152625 | 0.955 |
| stage | 43802 | 40303 | 0.920 |
| stmt_start | 21857 | 21151 | 0.968 |
| value | 171618 | 165206 | 0.963 |

### Sobrevivencia por categoria (train FIM, regra literal) - base para `fim.category_survival`

| categoria | entrada (etapas 1-8) | mantidos | sobrevivencia |
|---|---:|---:|---:|
| aggregation | 292013 | 268190 | 0.918 |
| crud | 326067 | 320689 | 0.984 |
| dsl | 194842 | 186926 | 0.959 |
| indexes | 86930 | 86921 | 1.000 |
| json_ejson | 53296 | 50672 | 0.951 |
| operators | 200983 | 191672 | 0.954 |

### Analise paralela (NAO aplicada ao treino): 13-grama com >= 2 tokens de conteudo

Removeria 11770 de 1189957 registros de treino (1.0%) contra 55709 (4.7%) na regra literal; mantidos 1178187 contra 1134248. Uso: so diagnostico (quanto da remocao e estrutura da linguagem e nao conteudo de schema); o treino usa a regra literal da spec.

| cut_kind | entrada | removidos (analise) | sobrevivencia (analise) | sobrevivencia (literal) |
|---|---:|---:|---:|---:|
| chat | 18607 | 502 | 0.973 | - |
| chat_multiturn | 3273 | 80 | 0.976 | - |
| correction | 8597 | 251 | 0.971 | - |
| explain | 1860 | 53 | 0.972 | - |
| key | 207164 | 1701 | 0.992 | 0.959 |
| member | 142492 | 1605 | 0.989 | 0.960 |
| multiline | 77660 | 599 | 0.992 | 0.959 |
| negative | 3489 | 122 | 0.965 | - |
| operator | 187508 | 1410 | 0.992 | 0.956 |
| partial | 142220 | 1563 | 0.989 | 0.961 |
| rest | 159810 | 1777 | 0.989 | 0.955 |
| stage | 43802 | 881 | 0.980 | 0.920 |
| stmt_start | 21857 | 124 | 0.994 | 0.968 |
| value | 171618 | 1102 | 0.994 | 0.963 |

## 3. Por split, tipo e domino

| split | tipo | entrada | saida | rejeitados |
|---|---|---:|---:|---:|
| bench | chat | 859 | 857 | 2 |
| bench | fim | 3450 | 3450 | 0 |
| test | chat | 911 | 907 | 4 |
| test | fim | 5750 | 5750 | 0 |
| train | chat | 36429 | 29178 | 7251 |
| train | fim | 1179300 | 1105070 | 74230 |
| val | chat | 1801 | 1790 | 11 |
| val | fim | 11500 | 11500 | 0 |

| dominio | holdout | entrada | saida | train | val | test | bench |
|---|---|---:|---:|---:|---:|---:|---:|
| commerce | nao | 90527 | 85045 | 83571 | 1105 | 197 | 172 |
| content | nao | 81217 | 75196 | 73688 | 1089 | 212 | 207 |
| education | sim | 2542 | 2541 | 0 | 0 | 1682 | 859 |
| energy | nao | 119168 | 111233 | 109685 | 1140 | 250 | 158 |
| finance | nao | 91979 | 85265 | 83672 | 1178 | 259 | 156 |
| gaming | nao | 148969 | 138281 | 136826 | 1052 | 208 | 195 |
| healthcare-synthetic | nao | 104377 | 97685 | 96100 | 1173 | 220 | 192 |
| hr | nao | 105619 | 99700 | 98172 | 1126 | 220 | 182 |
| iot | nao | 106777 | 99411 | 97908 | 1094 | 236 | 173 |
| legal | sim | 3449 | 3448 | 0 | 0 | 2254 | 1194 |
| logistics | nao | 116211 | 108744 | 107264 | 1043 | 235 | 202 |
| saas | nao | 89404 | 83700 | 82216 | 1068 | 213 | 203 |
| travel | nao | 99322 | 92543 | 90955 | 1108 | 240 | 240 |
| workspace | nao | 80439 | 75710 | 74191 | 1114 | 231 | 174 |

| kind (fim: cut_kind; chat: kind) | tipo | entrada | saida |
|---|---|---:|---:|
| chat | chat | 20000 | 16708 |
| chat_multiturn | chat | 3600 | 2634 |
| correction | chat | 10000 | 8163 |
| explain | chat | 2000 | 1709 |
| negative | chat | 4400 | 3518 |
| key | fim | 212603 | 201552 |
| member | fim | 149153 | 138681 |
| multiline | fim | 80395 | 76346 |
| operator | fim | 193211 | 182136 |
| partial | fim | 149194 | 138454 |
| rest | fim | 167096 | 155192 |
| stage | fim | 47949 | 43544 |
| stmt_start | fim | 23309 | 21948 |
| value | fim | 177090 | 167917 |

## 4. Gates de dataset (spec §6) medidos sobre a saida

| gate | resultado | medicao |
|---|---|---|
| G3.1 paridade de contrato | NAO MEDIDO aqui | depende de ContractProbe/IDE (`data/contracts/parity_inputs.jsonl`) |
| G3.2 validade | PASSA | 1158502/1158502 com syntax/api/fields ok; exec ok 875398, skipped 283104 (escritas e respostas sem codigo); exec-valid das leituras na entrada do pipeline: 100.0%; chat/correcao com equivalencia ok |
| G3.3 volume e distribuicao | FALHA | train validado 1134248 (fim 1105070 + chat 29178; alvo >= 1.000.000: ok); cut_kind +-2pp: FALHA; categorias +-2pp: FALHA; piso 4% (atlas/json/dsl/indexes): FALHA; idiomas +-2pp: ok; dialetos {'console': 1054398, 'json': 50672}; categorias ausentes: atlas_search |
| G3.4 vazamento | PASSA | grupos compartilhados entre splits: 0; treino x (val/test/bench/legacy-v1) restantes: 0; blind: not_sealed; legacy-v1 conferido: 12000 docs |
| G3.5 privacidade | PASSA | 0 ocorrencias de padroes proibidos na saida |
| G3.6 blind | PENDENTE | conjunto cego nao selado (`blind: not_sealed`); 400 itens escritos/revisados por pessoa |
| G3.7 revisao humana | PENDENTE | 300 exemplos aleatorios do train (amostrar de `data/splits/v2/train.ids.txt`) |
| G3.8 reproducibilidade | COBERTO POR TESTE (2 execucoes => mesmos sha256) | sha256 de saidas no `manifest.json`; mesmo seed/config/commit => mesmos hashes |
| G3.9 licencas | PENDENTE (licenses.md ausente) | fontes geradas (`source_license: generated`); `data/sources/licenses.md` |

### Gates adicionais da tarefa

- **Chat+correcao <= 25%** (pool de treino validado): 2.3%; na mistura-alvo da config: 21.0% -> PASSA. Pool por componente: {'autocomplete': 1105070, 'chat': 19203, 'correction': 6764, 'negative': 3211}; alvo: {'autocomplete': 0.76, 'chat': 0.15, 'correction': 0.06, 'negative': 0.03}; maximo sem repeticao: 107033 exemplos (limitante: `negative`).
- **Piso de grupos pequenos (4%)** (indexes, json_ejson, dsl, atlas_search): categorias abaixo de 4%: ['atlas_search']; cut_kind abaixo de 4%: ['stmt_start', 'stage'] -> FALHA.
- **Categorias (train FIM; alvo = `fim.category_targets`, +-2pp)**: aggregation 24.3% (alvo 27.0%), atlas_search 0.0% (alvo 13.0%), crud 29.0% (alvo 18.0%), dsl 16.9% (alvo 14.0%), indexes 7.9% (alvo 6.0%), json_ejson 4.6% (alvo 11.0%), operators 17.3% (alvo 11.0%).
- **Cobertura**: dominios no train 12; operadores distintos no train 63; operadores raros (<20 docs): ['$floor']. cut_kind (train, alvo +-2pp): key 18.0% (alvo 16%), value 14.9% (alvo 14%), operator 16.2% (alvo 14%), stage 3.6% (alvo 12%), member 12.4% (alvo 10%), partial 12.4% (alvo 10%), rest 13.8% (alvo 12%), multiline 6.7% (alvo 8%), stmt_start 1.9% (alvo 4%).
- **G2.7 (consulta ao manifesto)**: ids de val/test/bench no train = 0; holdout em train/val = 0 -> PASSA.
- **0 escritas executaveis**: outputs de chat com escrita executavel = 0; comandos de escrita vistos pelo servidor = 0; documentos FIM que sao escritas (sem execucao) = 277308 -> PASSA.

## 5. Falhas e ajustes propostos (sem alterar geradores)

- G3.3 cut_kind fora de +-2pp apos o pipeline: ['operator', 'stage', 'member', 'partial', 'stmt_start']. Ajuste: recalibrar `fim.decontam_survival` com a sobrevivencia medida e reamostrar (a decontaminacao remove desproporcionalmente alguns cut_kind).
- G3.3 categorias fora de +-2pp apos o pipeline: ['aggregation', 'atlas_search', 'crud', 'dsl', 'json_ejson', 'operators']. Ajuste: `fim.category_survival` com a sobrevivencia medida.
- G3.3 piso de 4%: alguma categoria pequena (atlas_search/json_ejson/dsl/indexes) ficou abaixo do piso apos a decontaminacao.
- G3.3 categorias ausentes no train: ['atlas_search'].
- Mistura: a componente `negative` limita a mistura-alvo a 107033 exemplos sem repeticao (pool: {'autocomplete': 1105070, 'chat': 19203, 'correction': 6764, 'negative': 3211}).
- Etapa 1: 10 registros rejeitados por padrao proibido (campos `secret`/`token`/`password` nos schemas).
- Etapa 7: 9 registros acima do limite de tokens (prompt 2048 / resposta 96 fim, 256 chat).
- G3.6: conjunto cego nao selado; a decontaminacao contra blind so passa a valer apos `blind_tool.py seal`.

## 5b. Comparacao v1 x v2

| metrica | v1 | v2 |
|---|---:|---:|
| train (fim+chat) | 8548 | 1134248 |
| val (fim+chat) | 3084 | 13290 |
| test (fim+chat) | 12337 | 6657 |
| bench (fim+chat) | 4995 | 4307 |
| train fim | 7392 | 1105070 |
| train chat | 1156 | 29178 |
| rejeitados em sanitize | 42 | 10 |
| rejeitados em syntax | 0 | 0 |
| rejeitados em api | 0 | 0 |
| rejeitados em fields | 0 | 0 |
| rejeitados em exec | 0 | 0 |
| rejeitados em semantic | 0 | 0 |
| rejeitados em budget | 885 | 9 |
| rejeitados em dedup | 124 | 25770 |
| rejeitados em decontam | 35985 | 55709 |
| removidos na decontaminacao (treino) | 35985 | 55709 |

| cut_kind (train FIM) | v1 | v2 |
|---|---:|---:|
| key | 19.4% | 18.0% |
| value | 16.4% | 14.9% |
| operator | 14.3% | 16.2% |
| stage | 2.8% | 3.6% |
| member | 12.7% | 12.4% |
| partial | 13.4% | 12.4% |
| rest | 11.0% | 13.8% |
| multiline | 6.2% | 6.7% |
| stmt_start | 3.9% | 1.9% |

| categoria (train FIM) | v1 | v2 |
|---|---:|---:|
| aggregation | 11.4% | 24.3% |
| atlas_search | 0.0% | 0.0% |
| crud | 88.6% | 29.0% |
| dsl | 0.0% | 16.9% |
| indexes | 0.0% | 7.9% |
| json_ejson | 0.0% | 4.6% |
| operators | 0.0% | 17.3% |

## 6. Limites

- Orcamento de chat usa renderizacao aproximada do `local-agent-fim-v0` (spec §4.4), nao o ContractProbe.
- Etapa 5 em FIM executa o plano do snippet (o documento reconstruido e igual ao texto e ao plano); escritas nunca sao executadas.
- Dedup exato por split usa tokens com literais preservados; near-dup (Jaccard >= 0,85) so no treino, decidido por documento (cortes distintos do mesmo snippet nao sao duplicatas).
- `limite por grupo` (max_per_group) e nomes de pessoas (regras v1) nao estao na config atual e nao foram aplicados.
- Atlas Search (`$search`/`$searchMeta`/`$vectorSearch`) so tem validacao ESTATICA (sem atlas-local): `exec: skipped_static`; createIndex/dropIndex e update/delete sao `skipped_write` (0 escritas chegam ao driver).
- `.hint()` NAO e metodo de cursor do Console (spec §2: sort, skip, limit, project, toArray): continua rejeitado na etapa API.
- Multi-turno (`history`) nao tem renderizacao no contrato `local-agent-fim-v0`: o orcamento de tokens (etapa 7) usa uma aproximacao (turnos `Usuario:`/`Assistente:` antes do editor).

## 7. Execucao do ciclo de volume v2 (tempo, disco, iteracao)

- **Cadeia** (seed 20261008, 110 schemas role-aware): `gen_snippets` 3.000.000 pedidos -> 2.997.498 unicos (~18 min, 5,1 GB) ->
  `validate_snippets` (exec so leitura, maxTimeMS 2000, kapi_fx_*; 2.470.101 ok, 347.599 skipped_write, 179.761 skipped_static, 37 erro;
  0 comandos de escrita no servidor; ~18 min) -> `gen_fim` (avaliacao primeiro, indice de decontaminacao, so snippets de treino que
  sobrevivem; ~17 min) -> `gen_editor_context` (shards de 100 mil; ~13 min) -> `gen_chat` 40.000 (3 min) -> `validate_pipeline` (~30 min).
  Total ~1,7 h de relogio + piloto 10x (v2s1: 200 mil snippets / 600 mil cortes; pipeline 10,6 min). Disco: generated/v2 14 GB,
  generated/v2b 6,2 GB, validated/v2 5,3 GB, splits 61 MB (C: 763 GB livres).
- **Piloto 10x (sem pre-filtro, regra literal):** 85,2% do treino removido na etapa 9 (508.418 de 596.528); a analise `>= 2` tokens de
  conteudo removeria 37,1%. Sobrevivencia por categoria: aggregation 3,0%, atlas_search 0,0%, json_ejson 0,7%, dsl 12,7%, operators 21,0%,
  crud 32,5%, indexes 83,0%; por cut_kind: stage 1,8%, multiline 11,8%, stmt_start 13,8%, key 18,8%. Causa: 13-gramas de ESTRUTURA da
  linguagem ($match/$group/$sort..., placeholders) coincidem com a avaliacao; atlas_search nao tem nenhum documento de treino sem
  13-grama comum com o val/test/bench de atlas_search.
- **Decisao de gerador:** `fim.decontam_prefilter` aplica a MESMA regra literal (indice `decontaminate`: exact, 13-grama, approx_key,
  MinHash >= 0,7) por snippet antes de amostrar os cortes de treino (1.965.025 snippets de treino -> 303.432 sobreviventes), e
  `fim.eval_counts` fixa val/test/bench em 10.000/5.000/3.000 (+15%). A etapa 9 do pipeline continua sendo o portao: removeu mais 55.709
  (4,7%) e a 2a passada tem 0 coincidencias. As proporcoes brutas (`snippets.kinds`/`extra_categories`) foram ajustadas pela
  sobrevivencia medida (mais aggregate/json_ejson).
- **Iteracao de reamostragem (uma):** antes = `max_cuts_per_snippet: 8`, count 1,5 M; depois = 16 e 1,2 M (fim.manifest v2 -> v2b).
  Treino FIM (apos pipeline): antes 1.399.116 / depois 1.105.070. aggregation 9,7% -> 24,3% (alvo 27%); json_ejson 1,8% -> 4,6% (11%);
  stage 1,5% -> 3,6% (12%); multiline 4,6% -> 6,7% (8%); stmt_start 1,4% -> 1,9% (4%); crud 36,4% -> 29,0% (18%); atlas_search 0,0% (13%).
  Continua fora de +-2 pp: o limite e a OFERTA de documentos de agregacao/json/atlas que sobrevivem a regra literal
  (aggregation 19 mil, json_ejson 3 mil, atlas_search 0 snippets), nao o numero de cortes.
- **Mistura:** chat+correcao+negativo = 29.178 treino (limite do chat 40 mil registros, pedido do usuario) contra 1.105.070 de
  autocomplete: 2,3% (alvo da config 24%; teto 25% respeitado). Mistura-alvo sem repeticao: 107.033 exemplos (limitante `negative`).
- **Pendencias:** G3.1 (ContractProbe/IDE; entradas em `data/contracts/v2/parity_inputs.jsonl`, 2.000 registros), G3.6 (blind nao
  selado), G3.7 (300 exemplos humanos de `data/splits/v2/train.ids.txt`), G3.9 (licenses.md); decisao do usuario sobre a regra de
  decontaminacao (literal x >= 2 tokens de conteudo) e sobre atlas_search (so existe no val/test/bench; 0% no treino).
- `.hint()` permanece fora do cursor do Console (spec: sort, skip, limit, project, toArray); a familia `hint_index` do chat segue
  gerando a entrada defeituosa que falha na etapa API.
