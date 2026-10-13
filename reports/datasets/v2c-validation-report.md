# Dataset v2c - relatorio de validacao (pipeline de 9 etapas)

Gerado por `scripts/data/validate_pipeline.py` (`validate-pipeline/1`), seed 20261008, commit `6c1023cdee78` (arvore com alteracoes). Execucao (etapa 5): `live`; orcamento (etapa 7): `qwen2.5-coder tokenizer.json`; sintaxe JS: `node vm.Script`.

## 1. Funil por etapa

| etapa | entrada | aprovados | rejeitados | fim (ent->saida) | chat (ent->saida) |
|---|---:|---:|---:|---|---|
| sanitize | 1400000 | 1399983 | 17 | 1100000->1099985 | 300000->299998 |
| syntax | 1399983 | 1399983 | 0 | 1099985->1099985 | 299998->299998 |
| api | 1399983 | 1399983 | 0 | 1099985->1099985 | 299998->299998 |
| fields | 1399983 | 1399983 | 0 | 1099985->1099985 | 299998->299998 |
| exec | 1399983 | 1399983 | 0 | 1099985->1099985 | 299998->299998 |
| semantic | 1399983 | 1399983 | 0 | 1099985->1099985 | 299998->299998 |
| budget | 1399983 | 1399400 | 583 | 1099985->1099985 | 299998->299415 |
| dedup | 1399400 | 1388707 | 10693 | 1099985->1096067 | 299415->292640 |
| decontam | 1388707 | 1337899 | 50808 | 1096067->1069613 | 292640->268286 |

Entrada: 1400000 registros (1100000 fim_ctx + 300000 chat). Saida: 1337899 (train 1247125, val 45381, test 28455, bench 16938).

### Principais motivos de rejeicao

| etapa | motivo | n |
|---|---|---:|
| decontam | coincide com avaliacao: ngram# | 46878 |
| dedup | duplicata exata | 7184 |
| dedup | near-dup MinHash J~# >= # | 3509 |
| decontam | coincide com avaliacao: approx_key | 3357 |
| budget | resposta com # tokens > # | 583 |
| decontam | coincide com avaliacao: exact,minhash | 313 |
| decontam | coincide com avaliacao: ngram#,approx_key | 199 |
| decontam | coincide com avaliacao: minhash | 46 |
| sanitize | numero de cartao (Luhn) | 17 |
| decontam | coincide com avaliacao: exact,approx_key,minhash | 10 |
| decontam | coincide com avaliacao: approx_key,minhash | 4 |
| decontam | coincide com avaliacao: ngram#,approx_key,minhash | 1 |

## 2. Decontaminacao (etapa 9)

- Treino de entrada: 1297933 registros; removidos: **50808** (3.9%); mantidos 1247125.
- Documentos de avaliacao indexados: 91247 {'val': 38076, 'test': 25118, 'bench': 16053, 'legacy-v1': 12000}. Conjunto cego: **not_sealed**.

| regra | removidos (qualquer coincidencia) | removidos (1a regra) |
|---|---:|---:|
| blind | 0 | 0 |
| exact | 323 | 323 |
| ngram13 | 47078 | 47078 |
| approx_key | 3571 | 3361 |
| minhash | 374 | 46 |

Sensibilidade do 13-grama (registros de treino removidos se o 13-grama exigisse >= k tokens de conteudo): k=0: 1093279, k=1: 850992, k=2: 47078, k=3: 3281, k=4: 693 (APLICADO: k=2; 0 = regra literal da spec).

Parametros: {'ngram': 13, 'minhash_perm': 128, 'shingle_k': 5, 'bands': 25, 'rows': 5, 'jaccard': 0.7, 'seed': 20261008, 'ngram_min_content': 2}. Relatorio completo: `reports/decontam/8885372255e21e4b.json`. Verificacao pos-remocao (2a passada treino x avaliacao): 0 registros ainda coincidentes.

### Sobrevivencia por cut_kind (train FIM, regra oficial) - base para `fim.decontam_survival`

| cut_kind | entrada (etapas 1-8) | mantidos | sobrevivencia |
|---|---:|---:|---:|
| key | 164910 | 161081 | 0.977 |
| member | 102758 | 100273 | 0.976 |
| multiline | 82656 | 80717 | 0.977 |
| operator | 137553 | 134225 | 0.976 |
| partial | 102750 | 100068 | 0.974 |
| rest | 124076 | 120751 | 0.973 |
| stage | 129410 | 124754 | 0.964 |
| stmt_start | 39343 | 38544 | 0.980 |
| value | 143613 | 140202 | 0.976 |

### Sobrevivencia por categoria (train FIM, regra oficial) - base para `fim.category_survival`

| categoria | entrada (etapas 1-8) | mantidos | sobrevivencia |
|---|---:|---:|---:|
| aggregation | 292732 | 279540 | 0.955 |
| atlas_search | 129508 | 128974 | 0.996 |
| crud | 181987 | 180016 | 0.989 |
| dsl | 145445 | 141137 | 0.970 |
| indexes | 47427 | 47425 | 1.000 |
| json_ejson | 115266 | 111410 | 0.967 |
| operators | 114704 | 112113 | 0.977 |

### Auditoria paralela (NAO aplicada ao treino): regra LITERAL da spec (qualquer 13-grama comum)

Removeria 1096330 de 1297933 registros de treino (84.5%) contra 50808 (3.9%) na regra OFICIAL (13-grama com >= 2 tokens de conteudo); mantidos 201603 contra 1247125. A diferenca e estrutura da linguagem (sem conteudo de schema); os schemas sao disjuntos por split (`split_role`), entao nao ha vazamento de conteudo. exact, approx_key, MinHash e blind valem nas duas.

| cut_kind | entrada | removidos (auditoria) | sobrevivencia (auditoria) | sobrevivencia (oficial) |
|---|---:|---:|---:|---:|
| chat | 139003 | 118764 | 0.146 | - |
| chat_multiturn | 25136 | 22892 | 0.089 | - |
| correction | 68974 | 62407 | 0.095 | - |
| explain | 13671 | 11602 | 0.151 | - |
| key | 164910 | 136076 | 0.175 | 0.977 |
| member | 102758 | 81669 | 0.205 | 0.976 |
| multiline | 82656 | 69995 | 0.153 | 0.977 |
| negative | 24080 | 3242 | 0.865 | - |
| operator | 137553 | 122108 | 0.112 | 0.976 |
| partial | 102750 | 84079 | 0.182 | 0.974 |
| rest | 124076 | 104330 | 0.159 | 0.973 |
| stage | 129410 | 126432 | 0.023 | 0.964 |
| stmt_start | 39343 | 34182 | 0.131 | 0.980 |
| value | 143613 | 118552 | 0.175 | 0.976 |

## 3. Por split, tipo e domino

| split | tipo | entrada | saida | rejeitados |
|---|---|---:|---:|---:|
| bench | chat | 5513 | 5439 | 74 |
| bench | fim | 11500 | 11499 | 1 |
| test | chat | 5516 | 5455 | 61 |
| test | fim | 23000 | 23000 | 0 |
| train | chat | 277872 | 246510 | 31362 |
| train | fim | 1031000 | 1000615 | 30385 |
| val | chat | 11099 | 10882 | 217 |
| val | fim | 34500 | 34499 | 1 |

| dominio | holdout | entrada | saida | train | val | test | bench |
|---|---|---:|---:|---:|---:|---:|---:|
| commerce | nao | 92605 | 88674 | 83135 | 3777 | 1005 | 757 |
| content | nao | 90240 | 85486 | 80081 | 3631 | 954 | 820 |
| education | sim | 10629 | 10599 | 0 | 0 | 7196 | 3403 |
| energy | nao | 133706 | 126458 | 120807 | 3957 | 961 | 733 |
| finance | nao | 116266 | 111432 | 105799 | 3943 | 1014 | 676 |
| gaming | nao | 149129 | 141095 | 135714 | 3691 | 946 | 744 |
| healthcare-synthetic | nao | 115326 | 110330 | 104663 | 3985 | 937 | 745 |
| hr | nao | 119241 | 114566 | 108979 | 3784 | 977 | 826 |
| iot | nao | 119647 | 114582 | 109101 | 3819 | 978 | 684 |
| legal | sim | 14116 | 14078 | 0 | 0 | 9551 | 4527 |
| logistics | nao | 134581 | 129236 | 123898 | 3585 | 964 | 789 |
| saas | nao | 100733 | 97728 | 92286 | 3761 | 895 | 786 |
| travel | nao | 106250 | 100105 | 94607 | 3661 | 1067 | 770 |
| workspace | nao | 97531 | 93530 | 88055 | 3787 | 1010 | 678 |

| kind (fim: cut_kind; chat: kind) | tipo | entrada | saida |
|---|---|---:|---:|
| chat | chat | 150000 | 137856 |
| chat_multiturn | chat | 27000 | 24627 |
| correction | chat | 75000 | 67340 |
| explain | chat | 15000 | 13595 |
| negative | chat | 33000 | 24868 |
| key | fim | 176721 | 172128 |
| member | fim | 110336 | 107171 |
| multiline | fim | 88361 | 86241 |
| operator | fim | 147003 | 143554 |
| partial | fim | 110219 | 106957 |
| rest | fim | 133096 | 129072 |
| stage | fim | 138161 | 133391 |
| stmt_start | fim | 42116 | 41271 |
| value | fim | 153987 | 149828 |

## 4. Gates de dataset (spec §6) medidos sobre a saida

| gate | resultado | medicao |
|---|---|---|
| G3.1 paridade de contrato | NAO MEDIDO aqui | depende de ContractProbe/IDE (`data/contracts/parity_inputs.jsonl`) |
| G3.2 validade | PASSA | 1337899/1337899 com syntax/api/fields ok; exec ok 1034477, skipped 303422 (escritas e respostas sem codigo); exec-valid das leituras na entrada do pipeline: 100.0%; chat/correcao com equivalencia ok |
| G3.3 volume e distribuicao | PASSA | train validado 1247125 (fim 1000615 + chat 246510; alvo >= 1.000.000: ok); cut_kind +-2pp: ok; categorias +-2pp: ok; piso 4% (atlas/json/dsl/indexes): ok; idiomas +-2pp: ok; dialetos {'console': 889205, 'json': 111410}; categorias ausentes: nenhuma |
| G3.4 vazamento | PASSA | grupos compartilhados entre splits: 0; treino x (val/test/bench/legacy-v1) restantes: 0; blind: not_sealed; legacy-v1 conferido: 12000 docs |
| G3.5 privacidade | PASSA | 0 ocorrencias de padroes proibidos na saida |
| G3.6 blind | PENDENTE | conjunto cego nao selado (`blind: not_sealed`); 400 itens escritos/revisados por pessoa |
| G3.7 revisao humana | PENDENTE | 300 exemplos aleatorios do train (amostrar de `data/splits/v2c/train.ids.txt`) |
| G3.8 reproducibilidade | COBERTO POR TESTE (2 execucoes => mesmos sha256) | sha256 de saidas no `manifest.json`; mesmo seed/config/commit => mesmos hashes |
| G3.9 licencas | PENDENTE (licenses.md ausente) | fontes geradas (`source_license: generated`); `data/sources/licenses.md` |

### Gates adicionais da tarefa

- **Chat+correcao <= 25%** (pool de treino validado): 17.9%; na mistura-alvo da config: 21.0% -> PASSA. Pool por componente: {'autocomplete': 1000615, 'chat': 162418, 'correction': 61485, 'negative': 22607}; alvo: {'autocomplete': 0.76, 'chat': 0.15, 'correction': 0.06, 'negative': 0.03}; maximo sem repeticao: 753566 exemplos (limitante: `negative`).
- **Piso de grupos pequenos (4%)** (indexes, json_ejson, dsl, atlas_search): categorias abaixo de 4%: nenhuma; cut_kind abaixo de 4%: ['stmt_start'] -> PASSA.
- **Categorias (train FIM; alvo = `fim.category_targets`, +-2pp)**: aggregation 27.9% (alvo 27.0%), atlas_search 12.9% (alvo 13.0%), crud 18.0% (alvo 18.0%), dsl 14.1% (alvo 14.0%), indexes 4.7% (alvo 6.0%), json_ejson 11.1% (alvo 11.0%), operators 11.2% (alvo 11.0%).
- **Cobertura**: dominios no train 12; operadores distintos no train 68; operadores raros (<20 docs): ['$floor']. cut_kind (train, alvo +-2pp): key 16.1% (alvo 16%), value 14.0% (alvo 14%), operator 13.4% (alvo 14%), stage 12.5% (alvo 12%), member 10.0% (alvo 10%), partial 10.0% (alvo 10%), rest 12.1% (alvo 12%), multiline 8.1% (alvo 8%), stmt_start 3.9% (alvo 4%).
- **G2.7 (consulta ao manifesto)**: ids de val/test/bench no train = 0; holdout em train/val = 0 -> PASSA.
- **0 escritas executaveis**: outputs de chat com escrita executavel = 0; comandos de escrita vistos pelo servidor = 0; documentos FIM que sao escritas (sem execucao) = 144067 -> PASSA.

## 5. Falhas e ajustes propostos (sem alterar geradores)

- Mistura: a componente `negative` limita a mistura-alvo a 753566 exemplos sem repeticao (pool: {'autocomplete': 1000615, 'chat': 162418, 'correction': 61485, 'negative': 22607}).
- Etapa 1: 17 registros rejeitados por padrao proibido (campos `secret`/`token`/`password` nos schemas).
- Etapa 7: 583 registros acima do limite de tokens (prompt 2048 / resposta 96 fim, 256 chat).
- G3.6: conjunto cego nao selado; a decontaminacao contra blind so passa a valer apos `blind_tool.py seal`.

## 5b. Comparacao v1 x v2

| metrica | v1 | v2 |
|---|---:|---:|
| train (fim+chat) | 8548 | 1247125 |
| val (fim+chat) | 3084 | 45381 |
| test (fim+chat) | 12337 | 28455 |
| bench (fim+chat) | 4995 | 16938 |
| train fim | 7392 | 1000615 |
| train chat | 1156 | 246510 |
| rejeitados em sanitize | 42 | 17 |
| rejeitados em syntax | 0 | 0 |
| rejeitados em api | 0 | 0 |
| rejeitados em fields | 0 | 0 |
| rejeitados em exec | 0 | 0 |
| rejeitados em semantic | 0 | 0 |
| rejeitados em budget | 885 | 583 |
| rejeitados em dedup | 124 | 10693 |
| rejeitados em decontam | 35985 | 50808 |
| removidos na decontaminacao (treino) | 35985 | 50808 |

| cut_kind (train FIM) | v1 | v2 |
|---|---:|---:|
| key | 19.4% | 16.1% |
| value | 16.4% | 14.0% |
| operator | 14.3% | 13.4% |
| stage | 2.8% | 12.5% |
| member | 12.7% | 10.0% |
| partial | 13.4% | 10.0% |
| rest | 11.0% | 12.1% |
| multiline | 6.2% | 8.1% |
| stmt_start | 3.9% | 3.9% |

| categoria (train FIM) | v1 | v2 |
|---|---:|---:|
| aggregation | 11.4% | 27.9% |
| atlas_search | 0.0% | 12.9% |
| crud | 88.6% | 18.0% |
| dsl | 0.0% | 14.1% |
| indexes | 0.0% | 4.7% |
| json_ejson | 0.0% | 11.1% |
| operators | 0.0% | 11.2% |

## 6. Limites

- Orcamento de chat usa renderizacao aproximada do `local-agent-fim-v0` (spec §4.4), nao o ContractProbe.
- Etapa 5 em FIM executa o plano do snippet (o documento reconstruido e igual ao texto e ao plano); escritas nunca sao executadas.
- Dedup exato por split usa tokens com literais preservados; near-dup (Jaccard >= 0,85) so no treino, decidido por documento (cortes distintos do mesmo snippet nao sao duplicatas).
- `limite por grupo` (max_per_group) e nomes de pessoas (regras v1) nao estao na config atual e nao foram aplicados.
- Atlas Search (`$search`/`$searchMeta`/`$vectorSearch`) so tem validacao ESTATICA (sem atlas-local): `exec: skipped_static`; createIndex/dropIndex e update/delete sao `skipped_write` (0 escritas chegam ao driver).
- `.hint()` NAO e metodo de cursor do Console (spec §2: sort, skip, limit, project, toArray): continua rejeitado na etapa API.
- Multi-turno (`history`) nao tem renderizacao no contrato `local-agent-fim-v0`: o orcamento de tokens (etapa 7) usa uma aproximacao (turnos `Usuario:`/`Assistente:` antes do editor).

## 7. v2c: regra oficial B x auditoria literal, v1 x v2 x v2c

**Decisao (opcao B):** `decontam.ngram_min_content: 2` e a regra OFICIAL (decontaminate.py, gen_fim pre-filtro e etapa 9 usam a
mesma). Um 13-grama comum so e contaminacao com >= 2 tokens de conteudo (campo/colecao/constante). Justificativa: schemas disjuntos
por split (`split_role`) bloqueiam vazamento de conteudo; 13-gramas estruturais ($match/$group/$sort, pontuacao, placeholders, API do
Console e argumentos de estagio/Atlas Search) sao a gramatica e coincidem por construcao. Ajuste feito junto: o vocabulario de
argumentos de estagios do MQL/Atlas (`index/text/query/path/compound/...`) passou a ser estrutural (`_STRUCT`); sem isso todo `$search`
tinha >= 2 "tokens de conteudo" e atlas_search ficava com 0 documentos de treino mesmo em k=2. exact, approx_key, MinHash >= 0,7 e
blind (blind_guard) inalterados.

| treino FIM+chat que chega a etapa 9 | literal (auditoria, k=0) | oficial (k=2) |
|---|---:|---:|
| removidos | 1.096.330 (84% dos documentos de treino ja pre-filtrados) | 50.808 (3,9%) |
| piloto v2s1 sem pre-filtro (600 mil cortes) | 85,2% removidos (aggregation 97%, atlas_search 100%, json_ejson 99,3%) | 37,1% (k=2 antes do ajuste de vocabulario) |

| | v1 | v2 (literal) | v2c (oficial) |
|---|---:|---:|---:|
| train fim+chat | 8.548 | 1.134.248 | 1.247.125 (fim 1.000.615 + chat 246.510) |
| val / test / bench (fim+chat) | 3.084 / 12.337 / 4.995 | 13.290 / 6.657 / 4.307 | 45.381 / 28.455 / 16.938 |
| aggregation / atlas_search / json_ejson | 11% / 0% / 0% | 24,3% / 0% / 4,6% | 27,9% / 12,9% / 11,1% |
| crud / operators / dsl / indexes | 89% / 0 / 0 / 0 | 29,0 / 17,3 / 16,9 / 7,9% | 18,0 / 11,2 / 14,1 / 4,7% |
| stage / multiline / stmt_start | 2,8 / 6,2 / 3,9% | 3,6 / 6,7 / 1,9% | 12,5 / 8,1 / 3,9% |
| G3.3 | FALHA | FALHA | PASSA |

- **Resample:** nao foi necessaria nova iteracao; a unica geracao v2c (max_cuts_per_snippet 8, count 1,1 M, eval 30k/20k/10k +15%)
  ja cai em +-2 pp para cut_kind e categorias (min: indexes 4,7% vs alvo 6,0%, piso 4% ok). A iteracao da v2 (max_cuts 16) nao entra.
- **Avaliacao:** val 34.499 + 10.882 chat, test 23.000 + 5.455, bench 11.499 + 5.439; por categoria cada split tem n >= 649
  (bench: indexes 649, atlas 1.432, aggregation 3.244). Por `group_id` (schema::template) so 10/2/0 grupos de val/test/bench
  chegam a n >= 50 (18.735/14.325/8.039 grupos): o criterio n >= 50 vale por categoria (e por dominio), nao por template.
- **Mistura (pool de treino):** autocomplete 80,2% / chat 13,0% / correcao 4,9% / negativo 1,8% (alvo da config 76/15/6/3; chat+correcao
  17,9% <= 25%). Gargalo: negativos (22.607 treino -> teto 753 mil sem repeticao) e os 300 mil registros de chat pedidos (o gerador nao teve
  shortfall: 0 repeticoes em 300 mil; pool de 405 mil leituras/398 mil planos). Para fechar 24% sobre 1,0 M de autocomplete seriam
  ~320 mil chat/correcao/negativo; a cota de negativos (11% dos kinds) e a primeira a faltar.
- **Paralelismo (deterministico):** gen_fim pre-filtro, gen_editor_context e etapas 1-7 do pipeline em processos spawn (`par.py`,
  janela limitada, resultado na ordem da entrada). Verificado 1 x N workers: sha256 identico por shard (fim_ctx 6.000 registros, 1 x 6;
  pipeline 3.000 registros, 1 x 4: fim.jsonl, chat.jsonl, rejected.jsonl, ids). Tempos v2c: fim 15 min, fim_ctx 3,5 min,
  chat 13 min, pipeline 31 min.
- **Apagado do disco:** data/generated/v2b (6,2 GB) e, em data/generated/v2, fim-*.jsonl, fim_ctx-*.jsonl, chat.jsonl e manifestos
  (superseded; ~8 GB). Mantidos: data/generated/v2/snippets.jsonl + validation.jsonl (entrada da v2c), data/validated/v2 (baseline
  literal) e v1. C: 738 GB livres.
