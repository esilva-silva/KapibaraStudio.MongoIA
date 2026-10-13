# Dataset v2 - relatorio de validacao (pipeline de 9 etapas)

Gerado por `scripts/data/validate_pipeline.py` (`validate-pipeline/1`), seed 20261008, commit `6c1023cdee78` (arvore com alteracoes). Execucao (etapa 5): `live`; orcamento (etapa 7): `qwen2.5-coder tokenizer.json`; sintaxe JS: `node vm.Script`.

## 1. Funil por etapa

| etapa | entrada | aprovados | rejeitados | fim (ent->saida) | chat (ent->saida) |
|---|---:|---:|---:|---|---|
| sanitize | 1540000 | 1539987 | 13 | 1500000->1499987 | 40000->40000 |
| syntax | 1539987 | 1539987 | 0 | 1499987->1499987 | 40000->40000 |
| api | 1539987 | 1539987 | 0 | 1499987->1499987 | 40000->40000 |
| fields | 1539987 | 1539987 | 0 | 1499987->1499987 | 40000->40000 |
| exec | 1539987 | 1539987 | 0 | 1499987->1499987 | 40000->40000 |
| semantic | 1539987 | 1539987 | 0 | 1499987->1499987 | 40000->40000 |
| budget | 1539987 | 1539978 | 9 | 1499987->1499987 | 40000->39991 |
| dedup | 1539978 | 1507646 | 32332 | 1499987->1468266 | 39991->39380 |
| decontam | 1507646 | 1452548 | 55098 | 1468266->1419816 | 39380->32732 |

Entrada: 1540000 registros (1500000 fim_ctx + 40000 chat). Saida: 1452548 (train 1428294, val 13290, test 6657, bench 4307).

### Principais motivos de rejeicao

| etapa | motivo | n |
|---|---|---:|
| decontam | coincide com avaliacao: ngram# | 51309 |
| dedup | near-dup MinHash J~# >= # | 23430 |
| dedup | duplicata exata | 8902 |
| decontam | coincide com avaliacao: approx_key | 3612 |
| decontam | coincide com avaliacao: ngram#,approx_key | 133 |
| decontam | coincide com avaliacao: exact,minhash | 43 |
| sanitize | numero de cartao (Luhn) | 13 |
| budget | resposta com # tokens > # | 9 |
| decontam | coincide com avaliacao: ngram#,minhash | 1 |

## 2. Decontaminacao (etapa 9)

- Treino de entrada: 1483392 registros; removidos: **55098** (3.7%); mantidos 1428294.
- Documentos de avaliacao indexados: 35032 {'val': 12386, 'test': 6431, 'bench': 4215, 'legacy-v1': 12000}. Conjunto cego: **not_sealed**.

| regra | removidos (qualquer coincidencia) | removidos (1a regra) |
|---|---:|---:|
| blind | 0 | 0 |
| exact | 43 | 43 |
| ngram13 | 51443 | 51443 |
| approx_key | 3745 | 3612 |
| minhash | 44 | 0 |

Sensibilidade do 13-grama (registros de treino removidos se o 13-grama exigisse >= k tokens de conteudo): k=0: 51443, k=1: 44819, k=2: 6686, k=3: 421, k=4: 105 (aplicado: k=0, 0 = regra literal da spec).

Parametros: {'ngram': 13, 'minhash_perm': 128, 'shingle_k': 5, 'bands': 25, 'rows': 5, 'jaccard': 0.7, 'seed': 20261008, 'ngram_min_content': 0}. Relatorio completo: `reports/decontam/a3cebdfaf269b67b.json`. Verificacao pos-remocao (2a passada treino x avaliacao): 0 registros ainda coincidentes.

### Sobrevivencia por cut_kind (train FIM, regra literal) - base para `fim.decontam_survival`

| cut_kind | entrada (etapas 1-8) | mantidos | sobrevivencia |
|---|---:|---:|---:|
| key | 275350 | 267151 | 0.970 |
| member | 194535 | 188060 | 0.967 |
| multiline | 66431 | 64212 | 0.967 |
| operator | 235994 | 227569 | 0.964 |
| partial | 180562 | 174492 | 0.966 |
| rest | 210320 | 202395 | 0.962 |
| stage | 21898 | 20225 | 0.924 |
| stmt_start | 20274 | 19658 | 0.970 |
| value | 242202 | 235354 | 0.972 |

### Sobrevivencia por categoria (train FIM, regra literal) - base para `fim.category_survival`

| categoria | entrada (etapas 1-8) | mantidos | sobrevivencia |
|---|---:|---:|---:|
| aggregation | 148251 | 136298 | 0.919 |
| crud | 518226 | 509777 | 0.984 |
| dsl | 300084 | 288266 | 0.961 |
| indexes | 134695 | 134690 | 1.000 |
| json_ejson | 26648 | 25336 | 0.951 |
| operators | 319662 | 304749 | 0.953 |

### Analise paralela (NAO aplicada ao treino): 13-grama com >= 2 tokens de conteudo

Removeria 10404 de 1483392 registros de treino (0.7%) contra 55098 (3.7%) na regra literal; mantidos 1472988 contra 1428294. Uso: so diagnostico (quanto da remocao e estrutura da linguagem e nao conteudo de schema); o treino usa a regra literal da spec.

| cut_kind | entrada | removidos (analise) | sobrevivencia (analise) | sobrevivencia (literal) |
|---|---:|---:|---:|---:|
| chat | 18607 | 502 | 0.973 | - |
| chat_multiturn | 3273 | 80 | 0.976 | - |
| correction | 8597 | 251 | 0.971 | - |
| explain | 1860 | 53 | 0.972 | - |
| key | 275350 | 1157 | 0.996 | 0.970 |
| member | 194535 | 1969 | 0.990 | 0.967 |
| multiline | 66431 | 322 | 0.995 | 0.967 |
| negative | 3489 | 122 | 0.965 | - |
| operator | 235994 | 1128 | 0.995 | 0.964 |
| partial | 180562 | 1877 | 0.990 | 0.966 |
| rest | 210320 | 1605 | 0.992 | 0.962 |
| stage | 21898 | 416 | 0.981 | 0.924 |
| stmt_start | 20274 | 101 | 0.995 | 0.970 |
| value | 242202 | 821 | 0.997 | 0.972 |

## 3. Por split, tipo e domino

| split | tipo | entrada | saida | rejeitados |
|---|---|---:|---:|---:|
| bench | chat | 859 | 857 | 2 |
| bench | fim | 3450 | 3450 | 0 |
| test | chat | 911 | 907 | 4 |
| test | fim | 5750 | 5750 | 0 |
| train | chat | 36429 | 29178 | 7251 |
| train | fim | 1479300 | 1399116 | 80184 |
| val | chat | 1801 | 1790 | 11 |
| val | fim | 11500 | 11500 | 0 |

| dominio | holdout | entrada | saida | train | val | test | bench |
|---|---|---:|---:|---:|---:|---:|---:|
| commerce | nao | 110519 | 104675 | 103201 | 1105 | 197 | 172 |
| content | nao | 102232 | 95845 | 94337 | 1089 | 212 | 207 |
| education | sim | 2542 | 2541 | 0 | 0 | 1682 | 859 |
| energy | nao | 148818 | 140234 | 138686 | 1140 | 250 | 158 |
| finance | nao | 117022 | 109296 | 107703 | 1178 | 259 | 156 |
| gaming | nao | 183243 | 172020 | 170565 | 1052 | 208 | 195 |
| healthcare-synthetic | nao | 129167 | 122049 | 120464 | 1173 | 220 | 192 |
| hr | nao | 129839 | 123567 | 122039 | 1126 | 220 | 182 |
| iot | nao | 131561 | 123722 | 122219 | 1094 | 236 | 173 |
| legal | sim | 3449 | 3448 | 0 | 0 | 2254 | 1194 |
| logistics | nao | 146517 | 138210 | 136730 | 1043 | 235 | 202 |
| saas | nao | 109578 | 103604 | 102120 | 1068 | 213 | 203 |
| travel | nao | 125525 | 118337 | 116749 | 1108 | 240 | 240 |
| workspace | nao | 99988 | 95000 | 93481 | 1114 | 231 | 174 |

| kind (fim: cut_kind; chat: kind) | tipo | entrada | saida |
|---|---|---:|---:|
| chat | chat | 20000 | 16708 |
| chat_multiturn | chat | 3600 | 2634 |
| correction | chat | 10000 | 8163 |
| explain | chat | 2000 | 1709 |
| negative | chat | 4400 | 3518 |
| key | fim | 282853 | 270134 |
| member | fim | 202425 | 189929 |
| multiline | fim | 69066 | 66084 |
| operator | fim | 241511 | 230376 |
| partial | fim | 187767 | 176345 |
| rest | fim | 219555 | 204962 |
| stage | fim | 25810 | 23466 |
| stmt_start | fim | 21678 | 20455 |
| value | fim | 249335 | 238065 |

## 4. Gates de dataset (spec §6) medidos sobre a saida

| gate | resultado | medicao |
|---|---|---|
| G3.1 paridade de contrato | NAO MEDIDO aqui | depende de ContractProbe/IDE (`data/contracts/parity_inputs.jsonl`) |
| G3.2 validade | PASSA | 1452548/1452548 com syntax/api/fields ok; exec ok 1008746, skipped 443802 (escritas e respostas sem codigo); exec-valid das leituras na entrada do pipeline: 100.0%; chat/correcao com equivalencia ok |
| G3.3 volume e distribuicao | FALHA | train validado 1428294 (fim 1399116 + chat 29178; alvo >= 1.000.000: ok); cut_kind +-2pp: FALHA; categorias +-2pp: FALHA; piso 4% (atlas/json/dsl/indexes): FALHA; idiomas +-2pp: ok; dialetos {'console': 1373780, 'json': 25336}; categorias ausentes: atlas_search |
| G3.4 vazamento | PASSA | grupos compartilhados entre splits: 0; treino x (val/test/bench/legacy-v1) restantes: 0; blind: not_sealed; legacy-v1 conferido: 12000 docs |
| G3.5 privacidade | PASSA | 0 ocorrencias de padroes proibidos na saida |
| G3.6 blind | PENDENTE | conjunto cego nao selado (`blind: not_sealed`); 400 itens escritos/revisados por pessoa |
| G3.7 revisao humana | PENDENTE | 300 exemplos aleatorios do train (amostrar de `data/splits/v2/train.ids.txt`) |
| G3.8 reproducibilidade | COBERTO POR TESTE (2 execucoes => mesmos sha256) | sha256 de saidas no `manifest.json`; mesmo seed/config/commit => mesmos hashes |
| G3.9 licencas | PENDENTE (licenses.md ausente) | fontes geradas (`source_license: generated`); `data/sources/licenses.md` |

### Gates adicionais da tarefa

- **Chat+correcao <= 25%** (pool de treino validado): 1.8%; na mistura-alvo da config: 21.0% -> PASSA. Pool por componente: {'autocomplete': 1399116, 'chat': 19203, 'correction': 6764, 'negative': 3211}; alvo: {'autocomplete': 0.76, 'chat': 0.15, 'correction': 0.06, 'negative': 0.03}; maximo sem repeticao: 107033 exemplos (limitante: `negative`).
- **Piso de grupos pequenos (4%)** (indexes, json_ejson, dsl, atlas_search): categorias abaixo de 4%: ['atlas_search', 'json_ejson']; cut_kind abaixo de 4%: ['stmt_start', 'stage'] -> FALHA.
- **Categorias (train FIM; alvo = `fim.category_targets`, +-2pp)**: aggregation 9.7% (alvo 27.0%), atlas_search 0.0% (alvo 13.0%), crud 36.4% (alvo 18.0%), dsl 20.6% (alvo 14.0%), indexes 9.6% (alvo 6.0%), json_ejson 1.8% (alvo 11.0%), operators 21.8% (alvo 11.0%).
- **Cobertura**: dominios no train 12; operadores distintos no train 63; operadores raros (<20 docs): ['$floor']. cut_kind (train, alvo +-2pp): key 19.1% (alvo 16%), value 16.8% (alvo 14%), operator 16.3% (alvo 14%), stage 1.5% (alvo 12%), member 13.4% (alvo 10%), partial 12.5% (alvo 10%), rest 14.5% (alvo 12%), multiline 4.6% (alvo 8%), stmt_start 1.4% (alvo 4%).
- **G2.7 (consulta ao manifesto)**: ids de val/test/bench no train = 0; holdout em train/val = 0 -> PASSA.
- **0 escritas executaveis**: outputs de chat com escrita executavel = 0; comandos de escrita vistos pelo servidor = 0; documentos FIM que sao escritas (sem execucao) = 438006 -> PASSA.

## 5. Falhas e ajustes propostos (sem alterar geradores)

- G3.3 cut_kind fora de +-2pp apos o pipeline: ['key', 'value', 'operator', 'stage', 'member', 'partial', 'rest', 'multiline', 'stmt_start']. Ajuste: recalibrar `fim.decontam_survival` com a sobrevivencia medida e reamostrar (a decontaminacao remove desproporcionalmente alguns cut_kind).
- G3.3 categorias fora de +-2pp apos o pipeline: ['aggregation', 'atlas_search', 'crud', 'dsl', 'indexes', 'json_ejson', 'operators']. Ajuste: `fim.category_survival` com a sobrevivencia medida.
- G3.3 piso de 4%: alguma categoria pequena (atlas_search/json_ejson/dsl/indexes) ficou abaixo do piso apos a decontaminacao.
- G3.3 categorias ausentes no train: ['atlas_search'].
- Mistura: a componente `negative` limita a mistura-alvo a 107033 exemplos sem repeticao (pool: {'autocomplete': 1399116, 'chat': 19203, 'correction': 6764, 'negative': 3211}).
- Etapa 1: 13 registros rejeitados por padrao proibido (campos `secret`/`token`/`password` nos schemas).
- Etapa 7: 9 registros acima do limite de tokens (prompt 2048 / resposta 96 fim, 256 chat).
- G3.6: conjunto cego nao selado; a decontaminacao contra blind so passa a valer apos `blind_tool.py seal`.

## 5b. Comparacao v1 x v2

| metrica | v1 | v2 |
|---|---:|---:|
| train (fim+chat) | 8548 | 1428294 |
| val (fim+chat) | 3084 | 13290 |
| test (fim+chat) | 12337 | 6657 |
| bench (fim+chat) | 4995 | 4307 |
| train fim | 7392 | 1399116 |
| train chat | 1156 | 29178 |
| rejeitados em sanitize | 42 | 13 |
| rejeitados em syntax | 0 | 0 |
| rejeitados em api | 0 | 0 |
| rejeitados em fields | 0 | 0 |
| rejeitados em exec | 0 | 0 |
| rejeitados em semantic | 0 | 0 |
| rejeitados em budget | 885 | 9 |
| rejeitados em dedup | 124 | 32332 |
| rejeitados em decontam | 35985 | 55098 |
| removidos na decontaminacao (treino) | 35985 | 55098 |

| cut_kind (train FIM) | v1 | v2 |
|---|---:|---:|
| key | 19.4% | 19.1% |
| value | 16.4% | 16.8% |
| operator | 14.3% | 16.3% |
| stage | 2.8% | 1.5% |
| member | 12.7% | 13.4% |
| partial | 13.4% | 12.5% |
| rest | 11.0% | 14.5% |
| multiline | 6.2% | 4.6% |
| stmt_start | 3.9% | 1.4% |

| categoria (train FIM) | v1 | v2 |
|---|---:|---:|
| aggregation | 11.4% | 9.7% |
| atlas_search | 0.0% | 0.0% |
| crud | 88.6% | 36.4% |
| dsl | 0.0% | 20.6% |
| indexes | 0.0% | 9.6% |
| json_ejson | 0.0% | 1.8% |
| operators | 0.0% | 21.8% |

## 6. Limites

- Orcamento de chat usa renderizacao aproximada do `local-agent-fim-v0` (spec §4.4), nao o ContractProbe.
- Etapa 5 em FIM executa o plano do snippet (o documento reconstruido e igual ao texto e ao plano); escritas nunca sao executadas.
- Dedup exato por split usa tokens com literais preservados; near-dup (Jaccard >= 0,85) so no treino, decidido por documento (cortes distintos do mesmo snippet nao sao duplicatas).
- `limite por grupo` (max_per_group) e nomes de pessoas (regras v1) nao estao na config atual e nao foram aplicados.
- Atlas Search (`$search`/`$searchMeta`/`$vectorSearch`) so tem validacao ESTATICA (sem atlas-local): `exec: skipped_static`; createIndex/dropIndex e update/delete sao `skipped_write` (0 escritas chegam ao driver).
- `.hint()` NAO e metodo de cursor do Console (spec §2: sort, skip, limit, project, toArray): continua rejeitado na etapa API.
- Multi-turno (`history`) nao tem renderizacao no contrato `local-agent-fim-v0`: o orcamento de tokens (etapa 7) usa uma aproximacao (turnos `Usuario:`/`Assistente:` antes do editor).
