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
