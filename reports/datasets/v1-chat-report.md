# Dataset v1 - chat curto / correcao / negativos / explain (relatorio)

Gerado com `gen_chat.py --ver v1` (seed 20261008, configs/dataset.yaml `chat:`), 6000 registros em 12 s. Formato estruturado independente de contrato (`instruction`, `context`, `output`); nenhum renderizador `local-agent-fim-v0` aplicado.

## Totais por kind / idioma / split

| kind | n | % | pt-BR | en | train | val | test | bench |
|---|---|---|---|---|---|---|---|---|
| chat | 3720 | 62.0% | 1870 | 1850 | 2552 | 198 | 687 | 283 |
| correction | 1320 | 22.0% | 656 | 664 | 944 | 62 | 218 | 96 |
| negative | 660 | 11.0% | 329 | 331 | 447 | 27 | 135 | 51 |
| explain | 300 | 5.0% | 138 | 162 | 203 | 10 | 65 | 22 |

Idiomas: pt-BR 2993 (49.9%), en 3007 (50.1%). Registro: {'n': 2654, 'i': 1856, 't': 1490}. pt-BR com acentos removidos: 387 de 2993 (config `pt_deaccent` = 0.3); erros de digitacao injetados: 168.

Chat: intents {'create': 2950, 'modify': 770}; referencia ambigua a colecao em 310 (8.3%) dos chat 'create'.

## Por dominio e split

| dominio | n | train | val | test | bench |
|---|---|---|---|---|---|
| commerce | 346 | 289 | 9 | 39 | 9 |
| content | 366 | 283 | 28 | 39 | 16 |
| education | 323 | 0 | 0 | 197 | 126 |
| energy | 488 | 399 | 20 | 57 | 12 |
| finance | 411 | 321 | 18 | 64 | 8 |
| gaming | 587 | 467 | 43 | 65 | 12 |
| healthcare-synthetic | 465 | 362 | 28 | 59 | 16 |
| hr | 396 | 294 | 25 | 65 | 12 |
| iot | 424 | 331 | 23 | 56 | 14 |
| legal | 428 | 0 | 0 | 254 | 174 |
| logistics | 532 | 428 | 32 | 63 | 9 |
| saas | 402 | 317 | 26 | 44 | 15 |
| travel | 462 | 367 | 24 | 58 | 13 |
| workspace | 370 | 288 | 21 | 45 | 16 |

Dominios holdout (education, legal) ficam so em test/bench; violacoes de holdout: 0. Split e `group` herdados do snippet (nenhum grupo cruza splits).

## Validacao (exec-valid)

- Execucoes de `output`/referencia tentadas: 5915; ok: 5915 (**exec-valid 100.00%**; sintaxe node + API + campos/tipos + execucao em kapi_fx_* + plano reconstruido == plano de origem).
- Registros de codigo com resultado nao vazio: 5340/5340 (100.0%); vazios sao descartados.
- Snippets de leitura com resultado dependente de ordem instavel ($group/$sortByCount + $limit/$skip sem $sort) excluidos do pool de chat: 768.
- Comandos de escrita vistos pelo servidor durante a validacao: 0 [].
- Outputs com escrita executavel (escrita/DDL/$out/$merge fora de comentario): 0.

## Correcao: defeitos e falha do input defeituoso

| defeito | n | % |
|---|---|---|
| wrong_field | 258 | 19.5% |
| wrong_operator | 205 | 15.5% |
| missing_dollar | 202 | 15.3% |
| wrong_type | 180 | 13.6% |
| invalid_api | 178 | 13.5% |
| syntax_comma | 120 | 9.1% |
| syntax_bracket | 100 | 7.6% |
| syntax_quote | 77 | 5.8% |

Etapa em que o input defeituoso falhou: fields=465, api=406, syntax=297, type=148, exec=4. Tentativas de defeito: 1362; todo par mantido tem input defeituoso que falha e corrigido que passa (`semantic` = valido mas com plano diferente do pretendido: 0).

## Negativos

| tipo | n |
|---|---|
| write | 229 |
| out_of_scope | 200 |
| ambiguous | 121 |
| no_collection | 110 |

Recusas de escrita/exclusao sao comentarios `//`; 35% trazem uma consulta `find` de leitura equivalente (executada e validada).

## Descartes (geracao)

| motivo | n |
|---|---|
| describer_unsupported | 98 |
| negative_unsupported | 93 |
| defect_not_failing | 42 |
| explain_long | 9 |
| explain_unsupported | 6 |
| correction_no_defect | 6 |
| duplicate | 2 |
| instruction_too_long | 1 |

Shortfall de quota: {}.

## Mistura alvo (config `chat.mixture`, fracao do treino total)

| componente | fracao |
|---|---|
| autocomplete | 0.76 |
| chat | 0.15 |
| correction | 0.06 |
| negative | 0.03 |

chat+explain+correcao = 21% (limite 25%); negativos = 3%.

## Limites conhecidos

- Instrucoes vem de templates parametrizados pelo plano (nenhum LLM): paraphrases variam, mas a cobertura e a de `gen_snippets` (estagios e operadores do plano); construcoes nao descritas (`$lookup` com `let/pipeline`, `$expr` com variaveis, etc.) sao descartadas. Risco v1: saturar benchmark programatico e falhar em pedido livre -> o blind escrito a mao continua sendo o gate.
- Valores citados aparecem literalmente; nomes de campo aparecem como no schema (sem sinonimos de dominio).
- Validacao sem ContractProbe/Acornima: sintaxe por `node vm.Script`, API por parser do subconjunto Console proprio; $search/$vectorSearch e estagios fora da allowlist nao entram.
- Correcao: defeitos sao edicoes textuais do output (8 familias); `semantic` e raro por construcao. Mensagens de erro de sintaxe vem do node (V8), nao do Acornima da IDE.
- Negativos: `known_collections` no contexto e necessario para `no_collection`; o contrato `local-agent-fim-v0` atual nao carrega essa lista (decisao pendente).
- Referencia ambigua so de colecao; nao ha chat multi-turno nem uso de `selection` (sempre null) nesta fatia.
- Dominios holdout so alimentam test/bench; train de chat depende dos mesmos grupos do FIM (sem vazamento entre eles).


---

## Apendice - ciclo 2 (variedade de correcao, instrucoes, multi-turno, negativos, capacidade)

Codigo: `scripts/data/{gen_chat,chat_describe,chat_validate}.py`, testes `scripts/data/tests/test_chat.py`, config `chat:` de `configs/dataset.yaml`.
Medicoes abaixo vieram de execucoes de descarte (6000 e 16000 registros, seed 20261008, fora de `data/generated/`); o volume final **nao** foi gerado e as secoes acima continuam sendo o relatorio do ciclo 1.

### 1. Familias de defeito (19 = 8 antigas + 11 novas)

Cada par mantido tem input defeituoso que FALHA em alguma etapa (syntax / api / fields / type / exec) e saida corrigida que passa em todas.
Regras estaticas novas em `chat_validate.check_structure` (as mesmas que o servidor rejeitaria; 0 falso positivo atribuivel as regras nos 17,2 mil snippets de leitura do v1; as 14 reprovacoes sao campos renomeados nos schemas novos):

| familia nova | defeito injetado | etapa que falha | mensagem tipo Console na instrucao |
|---|---|---|---|
| misplaced_operator | operador de agregacao em filtro (`$gt` -> `$sum`) ou de consulta em acumulador | api | `unknown operator` / `unknown group operator` |
| group_no_id | `$group` sem `_id` | fields | `a group specification must include an _id` |
| type_mismatch | string -> `ObjectId(...)`, bool -> `"true"`, data -> epoch numerico | type | (silenciosa: "volta vazio") |
| lookup_field | `$lookup` com `localField`/`foreignField`/`from` inexistente | fields | (silenciosa) |
| stage_order | dois estagios adjacentes trocados (campo usado antes de existir) | fields | (silenciosa) |
| unknown_method | `.findAll`, `.select`, `.agg`, `.pipeline`, `.length`, `.unique`... | api | `TypeError: X is not a function` |
| hint_index | `.hint({campo:1})` / `.hint("campo_1")` com indice inexistente | api | `hint provided does not correspond to an existing index` |
| projection_mix | projecao mista inclusao/exclusao | fields | `Cannot do exclusion on field X in inclusion projection` |
| sort_direction | direcao `"desc"`/`"asc"`/`2` | fields | `$sort key ordering must be 1 ... or -1` |
| limit_value | `$limit` texto/zero, `.limit("10")`, `$skip` negativo | api/fields | `limit requires an integer` / `the limit must be positive` |
| wrong_collection | colecao inexistente em `db[...]`/`db.x` | fields | (silenciosa) |

Combinacao: `chat.defect_count` = {1: 0.71, 2: 0.12, 3: 0.13, 4: 0.04}; medido em 6000 registros (1500 correcoes): 1 defeito 72,5%, 2 defeitos 13,3%, **3 defeitos 10,7%, 4 defeitos 3,5% (>=3: 14,2%)**; familias distintas dentro do mesmo registro (`defect.types`, `defect.combined`).
Distribuicao por familia (6000 registros): wrong_field 8,9%, wrong_type 7,8%, wrong_operator 7,7%, missing_dollar 7,2%, type_mismatch 6,6%, stage_order 5,8%, misplaced_operator 5,6%, unknown_method 5,1%, lookup_field 5,1%, projection_mix 4,7%, wrong_collection 4,7%, invalid_api 4,7%, limit_value 4,6%, hint_index 4,2%, syntax_* 3,2-3,7%, group_no_id 3,3%, sort_direction 3,3%.
Taxa de sucesso por familia (defeito mantido / tentativas de registro): 100% em 15 familias; wrong_type 91%, type_mismatch 98%, lookup_field 96% (sem `ObjectId`/data/numero ou sem `$lookup` o gerador escolhe outro snippet; ate 4 por registro).
Etapa em que falha (correcoes): fields 38%, api 36%, syntax 17%, type 8%, exec 1%.

### 2. Instrucoes

- Registros: `n` neutro 34%, `i` informal 24%, `t` tecnico 18%, `g` telegrafico 12% (ex.: `orders: status = "paid" e total >= 10; ordem total descendente; limite 5`), `q` pergunta indireta 12% (ex.: `Tem como me mostrar os registros de orders onde ...?`, `Could you list ...`). Sorteio por `chat.register`.
- Mais paraphrases por operacao (4-7 cabecas por registro/idioma/op), conectores de filtro, sinonimos do substantivo (documentos/registros/itens/entradas; documents/records/docs/entries), operadores de comparacao (+1 variante cada), 2-3 variantes por estagio, correcao/explain com listas por registro. pt-BR com e sem acento (30%), erro de digitacao (4%).
- Variedade medida: a mesma consulta (find com filtro de 2 campos, sort, limit) rende >= 35 instrucoes distintas por idioma com 12 sorteios x 5 registros (teste `test_unique_keys_and_instruction_variety`).
- Multi-turno `kind: chat_multiturn` (9% dos registros; conta como `chat` na mistura): 2 turnos, `history = [{role:user, content, editor}, {role:assistant, content: codigo}]`, `context.editor` = resposta do turno 1, `instruction` = refinamento, `output` = consulta final. Refinamentos: `tail` (+ordenacao/limite/skip/projecao ou $sort/$limit finais: "agora ordene por ...") e `filter` (condicao adicional: "agora so os onde ...", e "agora so os de 2025 (por createdAt)" quando o filtro e um ano fechado). Medido: tail 355, filter 185 em 540. Historico e saida validados (sintaxe, API, campos, execucao, plano igual); estilo de render fixado por seed para base e final diferirem so pelo refinamento (tambem vale para `modify`).

### 3. Negativos (11% dos registros de chat; 3% da mistura; limite 3-5%)

Tipos (6000 registros): out_of_scope 21%, write 20%, ambiguous_field 17%, no_collection 16%, ddl 14%, ambiguous 12%.
Novos: `ddl` (drop/rename/createIndex/dropIndex/dropDatabase/sharding -> recusa; 30% com `countDocuments({})` de leitura), `ambiguous_field` (pergunta curta com candidatos do schema: "Qual campo de data devo usar para ordenar por recencia: createdAt ou updatedAt?"; so quando ha >=2 campos de data/numericos), 24 pedidos fora de escopo por idioma, mais frases de escrita generica; recusa de escrita pode trazer `find` limit 20 **ou** `countDocuments(filtro)` de leitura equivalente (executado e validado). Saida de negativo: so comentarios `//` + no maximo uma leitura. `known_collections` continua campo estruturado (decisao de contrato pendente).

### 4. Split por schema

`schema_split_role` (`train|eval`, lido defensivamente de `Schema.split_role`; ausente => `None`) e gravado no registro. `split` do chat = `split` do snippet (que ja o herda do schema). Sem dependencia dura: sem `split_role` nada muda. Com papel conhecido, snippet em split incompativel (`eval` em `train`; `train` em `test/bench` fora de holdout) sai do pool (`pool.split_role_conflicts_skipped` no manifesto).
Estado atual: `snippets.jsonl` v1 e anterior aos papeis -> 3957 snippets (de 11 669 leituras validas) excluidos; pool de chat = 7974 leituras / 7951 planos distintos (era 10 901 / 10 877). Regerar snippets com o gen_snippets role-aware recupera o pool.

### 5. Teto de repeticao e capacidade (`--capacity-report`)

- Chave de unicidade: `(schema, colecao, kind, instrucao normalizada [sem acento/caixa/espaco], codigo normalizado [sem layout/aspas/;], + 1o turno)`.
- **16 000 registros gerados com 14 duplicatas descartadas** (`attempts` 16 512, `unique_keys` 16 000) - alvo >= 15 000 atingido com o pool reduzido de 7951 planos. 6000 registros: 6193 tentativas, 0 repeticao.
- Capacidade medida por familia/schema (6 planos/schema, 3 sorteios; limite inferior; `chat.capacity.json`): total estimado 471 mil unicos; por kind: chat 35 959 (necessario 7500 para 15k), chat_multiturn 19 819 (1350), correction 284 766 (3750), negative 108 197 (1650), explain 22 555 (750). Com 12 sorteios/plano os unicos por plano crescem ~linearmente (chat 142 mil; correction 1,14 M): a variedade de texto nao satura; **o limite real e o numero de planos distintos** (7951 -> 10 877 apos regerar snippets), que e o que decontaminacao e semantica enxergam.
- Gargalos por familia (planos elegiveis / schemas sem nenhum): lookup_field 343 planos (79 de 110 schemas com 0), group_no_id 1259 (33), projection_mix 1626 (27), negative.write 1733 planos de escrita, wrong_type 2087 (13), stage_order 2456 (11), multiturn.filter 2402 (11). Schemas mais fracos: commerce/prism_com (607 unicos), healthcare-synthetic/cinder_hlt (654), energy/kestrel_nrg (744). Gargalo para chegar a 15k: `chat` (cobertura 4,8x) e, dentro de correction, `lookup_field`/`group_no_id`; nenhuma familia impede 15k.

### 6. Decontaminacao (proxy interno)

Sobreposicao de 13-grama (regra literal, k=0) entre registros de treino e nao-treino **do proprio chat**: 61% no v1 antigo e 61% no ciclo 2 (6000 registros) - dominada pelo codigo (`output`/`editor` do mesmo schema: ~64% dos chat; instrucao < 1%). Ou seja, variedade de instrucao nao reduz a remocao; so muda com schemas disjuntos por split (split_role; nao medido aqui). Mitigacao feita: instrucoes de correcao (mensagem de erro e frases fixas) ficam com <= 12 tokens, pois a mensagem e boilerplate compartilhado; sem isso a sobreposicao de instrucao em `correction` subia de 4 para ~80 de ~1070 registros; com o ajuste e 0.

### 7. Pendencias

1. `validate_pipeline.py` (fora do meu escopo): linha ~643 mapeia `kind` por dicionario fechado (`chat`, `explain`, `correction`, `negative`) -> `chat_multiturn` gera KeyError; mapear para `chat`. O orcamento de tokens (etapa 7) deve incluir `history` (hoje ignorado).
2. `.hint()` tratado como fora do subconjunto do Console; confirmar se a IDE aceita hint (se aceitar, a familia vira `exec`: indice inexistente).
3. `known_collections` (negativos `no_collection`/`ambiguous`) segue campo estruturado ate a decisao de contrato.
4. Regerar `snippets.jsonl`/`validation.jsonl` com gen_snippets role-aware e re-medir decontaminacao (3957 snippets hoje fora do pool por conflito de papel).
5. Planos com `$lookup` simples sao raros (343 elegiveis): pedir ao gen_snippets mais templates `lookup` (localField/foreignField) para `lookup_field`.
6. Capacidade e de texto unico por plano; diversidade semantica = planos distintos. Mais planos (gen_snippets) e o proximo gargalo real.
