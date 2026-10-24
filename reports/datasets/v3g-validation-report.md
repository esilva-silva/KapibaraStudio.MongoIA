# Dataset v3g - relatorio de validacao (pipeline de 9 etapas)

Gerado por `scripts/data/validate_pipeline.py` (`validate-pipeline/1`), seed 20261024, commit `2271398ef8f6` (arvore com alteracoes). Execucao (etapa 5): `live`; orcamento (etapa 7): `qwen2.5-coder tokenizer.json`; sintaxe JS: `node vm.Script`.

## 1. Funil por etapa

| etapa | entrada | aprovados | rejeitados | fim (ent->saida) | chat (ent->saida) |
|---|---:|---:|---:|---|---|
| sanitize | 152000 | 151997 | 3 | 152000->151997 | 0->0 |
| syntax | 151997 | 151997 | 0 | 151997->151997 | 0->0 |
| api | 151997 | 151997 | 0 | 151997->151997 | 0->0 |
| fields | 151997 | 151997 | 0 | 151997->151997 | 0->0 |
| exec | 151997 | 151995 | 2 | 151997->151995 | 0->0 |
| semantic | 151995 | 151995 | 0 | 151995->151995 | 0->0 |
| budget | 151995 | 151995 | 0 | 151995->151995 | 0->0 |
| dedup | 151995 | 151876 | 119 | 151995->151876 | 0->0 |
| decontam | 151876 | 151177 | 699 | 151876->151177 | 0->0 |

Entrada: 152000 registros (152000 fim_ctx + 0 chat). Saida: 151177 (train 151177, val 0, test 0, bench 0).

### Principais motivos de rejeicao

| etapa | motivo | n |
|---|---|---:|
| decontam | coincide com avaliacao: ngram# | 579 |
| dedup | near-dup MinHash J~# >= # | 87 |
| decontam | coincide com avaliacao: minhash | 70 |
| decontam | coincide com avaliacao: approx_key | 47 |
| dedup | duplicata exata | 32 |
| sanitize | numero de cartao (Luhn) | 3 |
| decontam | coincide com avaliacao: ngram#,approx_key | 2 |
| exec | OperationFailure: PlanExecutor error during aggregation :: caused by :: can't $subtract string from int, full  | 2 |
| decontam | coincide com avaliacao: exact,minhash | 1 |

## 2. Decontaminacao (etapa 9)

- Treino de entrada: 151876 registros; removidos: **699** (0.5%); mantidos 151177.
- Documentos de avaliacao indexados: 84510 {'homolog': 4170, 'legacy-v1': 80340}. Conjunto cego: **not_sealed**.

| regra | removidos (qualquer coincidencia) | removidos (1a regra) |
|---|---:|---:|
| blind | 0 | 0 |
| exact | 1 | 1 |
| ngram13 | 581 | 581 |
| approx_key | 49 | 47 |
| minhash | 71 | 70 |

Sensibilidade do 13-grama (registros de treino removidos se o 13-grama exigisse >= k tokens de conteudo): k=0: 125289, k=1: 81671, k=2: 581, k=3: 15, k=4: 0 (APLICADO: k=2; 0 = regra literal da spec).

Parametros: {'ngram': 13, 'minhash_perm': 128, 'shingle_k': 5, 'bands': 25, 'rows': 5, 'jaccard': 0.7, 'seed': 20261024, 'ngram_min_content': 2}. Relatorio completo: `reports/decontam/e9ca99ccecd342fb.json`. Verificacao pos-remocao (2a passada treino x avaliacao): 0 registros ainda coincidentes.

### Sobrevivencia por cut_kind (train FIM, regra oficial) - base para `fim.decontam_survival`

| cut_kind | entrada (etapas 1-8) | mantidos | sobrevivencia |
|---|---:|---:|---:|
| key | 24634 | 24522 | 0.996 |
| member | 15406 | 15338 | 0.996 |
| multiline | 12321 | 12270 | 0.996 |
| operator | 19884 | 19812 | 0.996 |
| partial | 15429 | 15359 | 0.996 |
| rest | 18533 | 18439 | 0.995 |
| stage | 18734 | 18617 | 0.994 |
| stmt_start | 5361 | 5343 | 0.997 |
| value | 21574 | 21477 | 0.996 |

### Sobrevivencia por categoria (train FIM, regra oficial) - base para `fim.category_survival`

| categoria | entrada (etapas 1-8) | mantidos | sobrevivencia |
|---|---:|---:|---:|
| aggregation | 27805 | 27661 | 0.995 |
| atlas_search | 30617 | 30365 | 0.992 |
| crud | 15442 | 15394 | 0.997 |
| dsl | 24706 | 24614 | 0.996 |
| indexes | 10055 | 10022 | 0.997 |
| json_ejson | 27805 | 27704 | 0.996 |
| operators | 15446 | 15417 | 0.998 |

### Auditoria paralela (NAO aplicada ao treino): regra LITERAL da spec (qualquer 13-grama comum)

Removeria 125315 de 151876 registros de treino (82.5%) contra 699 (0.5%) na regra OFICIAL (13-grama com >= 2 tokens de conteudo); mantidos 26561 contra 151177. A diferenca e estrutura da linguagem (sem conteudo de schema); os schemas sao disjuntos por split (`split_role`), entao nao ha vazamento de conteudo. exact, approx_key, MinHash e blind valem nas duas.

| cut_kind | entrada | removidos (auditoria) | sobrevivencia (auditoria) | sobrevivencia (oficial) |
|---|---:|---:|---:|---:|
| key | 24634 | 19601 | 0.204 | 0.996 |
| member | 15406 | 11283 | 0.268 | 0.996 |
| multiline | 12321 | 10043 | 0.185 | 0.996 |
| operator | 19884 | 17382 | 0.126 | 0.996 |
| partial | 15429 | 12049 | 0.219 | 0.996 |
| rest | 18533 | 14952 | 0.193 | 0.995 |
| stage | 18734 | 18299 | 0.023 | 0.994 |
| stmt_start | 5361 | 4563 | 0.149 | 0.997 |
| value | 21574 | 17143 | 0.205 | 0.996 |

## 3. Por split, tipo e domino

| split | tipo | entrada | saida | rejeitados |
|---|---|---:|---:|---:|
| train | fim | 152000 | 151177 | 823 |

| dominio | holdout | entrada | saida | train | val | test | bench |
|---|---|---:|---:|---:|---:|---:|---:|
| commerce | nao | 6803 | 6783 | 6783 | 0 | 0 | 0 |
| content | nao | 9682 | 9616 | 9616 | 0 | 0 | 0 |
| energy | nao | 7210 | 7207 | 7207 | 0 | 0 | 0 |
| finance | nao | 7730 | 7692 | 7692 | 0 | 0 | 0 |
| gaming | nao | 12202 | 12160 | 12160 | 0 | 0 | 0 |
| healthcare-synthetic | nao | 10865 | 10843 | 10843 | 0 | 0 | 0 |
| hr | nao | 9363 | 9222 | 9222 | 0 | 0 | 0 |
| iot | nao | 12487 | 12369 | 12369 | 0 | 0 | 0 |
| logistics | nao | 11630 | 11558 | 11558 | 0 | 0 | 0 |
| manufacturing | nao | 21654 | 21583 | 21583 | 0 | 0 | 0 |
| retail-pos | nao | 18424 | 18337 | 18337 | 0 | 0 | 0 |
| saas | nao | 9359 | 9309 | 9309 | 0 | 0 | 0 |
| travel | nao | 7060 | 7001 | 7001 | 0 | 0 | 0 |
| workspace | nao | 7531 | 7497 | 7497 | 0 | 0 | 0 |

| kind (fim: cut_kind; chat: kind) | tipo | entrada | saida |
|---|---|---:|---:|
| key | fim | 24650 | 24522 |
| member | fim | 15424 | 15338 |
| multiline | fim | 12325 | 12270 |
| operator | fim | 19885 | 19812 |
| partial | fim | 15455 | 15359 |
| rest | fim | 18562 | 18439 |
| stage | fim | 18738 | 18617 |
| stmt_start | fim | 5369 | 5343 |
| value | fim | 21592 | 21477 |

## 4. Gates de dataset (spec §6) medidos sobre a saida

| gate | resultado | medicao |
|---|---|---|
| G3.1 paridade de contrato | NAO MEDIDO aqui | depende de ContractProbe/IDE (`data/contracts/parity_inputs.jsonl`) |
| G3.2 validade | PASSA | 151177/151177 com syntax/api/fields ok; exec ok 102673, skipped 48504 (escritas e respostas sem codigo); exec-valid das leituras na entrada do pipeline: 100.0%; chat/correcao com equivalencia ok |
| G3.3 volume e distribuicao | FALHA | train validado 151177 (fim 151177 + chat 0; alvo >= 1.000.000: FALHA); cut_kind +-2pp: ok; categorias +-2pp: ok; piso 4% (atlas/json/dsl/indexes): ok; idiomas +-2pp: FALHA; dialetos {'json': 27704, 'console': 123473}; categorias ausentes: nenhuma |
| G3.4 vazamento | PASSA | grupos compartilhados entre splits: 0; treino x (val/test/bench/legacy-v1) restantes: 0; blind: not_sealed; legacy-v1 conferido: 80340 docs |
| G3.5 privacidade | PASSA | 0 ocorrencias de padroes proibidos na saida |
| G3.6 blind | PENDENTE | conjunto cego nao selado (`blind: not_sealed`); 400 itens escritos/revisados por pessoa |
| G3.7 revisao humana | PENDENTE | 300 exemplos aleatorios do train (amostrar de `data/splits/v3g/train.ids.txt`) |
| G3.8 reproducibilidade | COBERTO POR TESTE (2 execucoes => mesmos sha256) | sha256 de saidas no `manifest.json`; mesmo seed/config/commit => mesmos hashes |
| G3.9 licencas | PENDENTE (licenses.md ausente) | fontes geradas (`source_license: generated`); `data/sources/licenses.md` |

### Gates adicionais da tarefa

- **Chat+correcao <= 25%** (pool de treino validado): 0.0%; na mistura-alvo da config: 21.0% -> PASSA. Pool por componente: {'autocomplete': 151177}; alvo: {'autocomplete': 0.76, 'chat': 0.15, 'correction': 0.06, 'negative': 0.03}; maximo sem repeticao: 0 exemplos (limitante: `chat`).
- **Piso de grupos pequenos (4%)** (indexes, json_ejson, dsl, atlas_search): categorias abaixo de 4%: nenhuma; cut_kind abaixo de 4%: ['stmt_start'] -> PASSA.
- **Categorias (train FIM; alvo = `fim.category_targets`, +-2pp)**: aggregation 18.3% (alvo 18.0%), atlas_search 20.1% (alvo 20.0%), crud 10.2% (alvo 10.0%), dsl 16.3% (alvo 16.0%), indexes 6.6% (alvo 8.0%), json_ejson 18.3% (alvo 18.0%), operators 10.2% (alvo 10.0%).
- **Cobertura**: dominios no train 14; operadores distintos no train 68; operadores raros (<20 docs): ['$map']. cut_kind (train, alvo +-2pp): key 16.2% (alvo 16%), value 14.2% (alvo 14%), operator 13.1% (alvo 14%), stage 12.3% (alvo 12%), member 10.2% (alvo 10%), partial 10.2% (alvo 10%), rest 12.2% (alvo 12%), multiline 8.1% (alvo 8%), stmt_start 3.5% (alvo 4%).
- **G2.7-hom**: schemas hom no train/val/test/bench = 0; grupos hom = 0; conjunto data/homolog/v1 usado na decontaminacao: 4170 registros -> PASSA.
- **G2.7 (consulta ao manifesto)**: ids de val/test/bench no train = 0; holdout em train/val = 0 -> PASSA.
- **0 escritas executaveis**: outputs de chat com escrita executavel = 0; comandos de escrita vistos pelo servidor = 0; documentos FIM que sao escritas (sem execucao) = 18139 -> PASSA.

## 5. Falhas e ajustes propostos (sem alterar geradores)

- G3.3 volume: train validado 151177 (fim 151177 + chat 0) < 1.000.000. Gargalo medido: a decontaminacao literal remove 0% do treino que chega a etapa 9 (13-gramas de estrutura da linguagem sao compartilhados com a avaliacao). Ajuste: mais volume bruto (a sobrevivencia e ~constante por documento) ou mais planos distintos; ver `Sobrevivencia por cut_kind`.
- Mistura: a componente `chat` limita a mistura-alvo a 0 exemplos sem repeticao (pool: {'autocomplete': 151177}).
- Etapa 1: 3 registros rejeitados por padrao proibido (campos `secret`/`token`/`password` nos schemas).
- G3.6: conjunto cego nao selado; a decontaminacao contra blind so passa a valer apos `blind_tool.py seal`.

## 6. Limites

- Orcamento de chat usa renderizacao aproximada do `local-agent-fim-v0` (spec §4.4), nao o ContractProbe.
- Etapa 5 em FIM executa o plano do snippet (o documento reconstruido e igual ao texto e ao plano); escritas nunca sao executadas.
- Dedup exato por split usa tokens com literais preservados; near-dup (Jaccard >= 0,85) so no treino, decidido por documento (cortes distintos do mesmo snippet nao sao duplicatas).
- `limite por grupo` (max_per_group) e nomes de pessoas (regras v1) nao estao na config atual e nao foram aplicados.
- Atlas Search (`$search`/`$searchMeta`/`$vectorSearch`) so tem validacao ESTATICA (sem atlas-local): `exec: skipped_static`; createIndex/dropIndex e update/delete sao `skipped_write` (0 escritas chegam ao driver).
- `.hint()` NAO e metodo de cursor do Console (spec §2: sort, skip, limit, project, toArray): continua rejeitado na etapa API.
- Multi-turno (`history`) nao tem renderizacao no contrato `local-agent-fim-v0`: o orcamento de tokens (etapa 7) usa uma aproximacao (turnos `Usuario:`/`Assistente:` antes do editor).
