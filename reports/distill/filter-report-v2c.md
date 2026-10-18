# Filtro da destilacao - `v2c`

Prompts 300000 - candidatas 300000 - **aceitos 267498 (89.2% dos prompts)** - greedy valida em 93.0%.

Projecao para o pool de 600k: ~267510 exemplos (alvo T1 230000, T0 160000).

## Concordancia professor x gold (greedy)

- Prompts com gold executavel: 221677; igual ao gold 38.2%, normalizado 38.2%, plano igual 39.0%, concorda 39.0%, valida 92.9%.
- Todos os prompts: igual 39.8%, concorda 40.6%, valida 93.0%.
- Aceitas que divergem do gold: 147588 (55.2%) - `agrees_with_gold:false`, peso 0,85.

## Aceitacao por context_kind

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| full | 213753 | 198937 | 93.1% |
| irrelevant | 45410 | 35865 | 79.0% |
| none | 40837 | 32696 | 80.1% |

## Aceitacao por cut_kind

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| key | 48406 | 43025 | 88.9% |
| member | 30253 | 27188 | 89.9% |
| multiline | 23740 | 17737 | 74.7% |
| operator | 40538 | 38487 | 94.9% |
| partial | 30176 | 27806 | 92.2% |
| rest | 35408 | 31720 | 89.6% |
| stage | 37390 | 30860 | 82.5% |
| stmt_start | 11738 | 9397 | 80.1% |
| value | 42351 | 41278 | 97.5% |

## Aceitacao por dominio

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| commerce | 20299 | 18076 | 89.0% |
| content | 19804 | 17693 | 89.3% |
| energy | 29002 | 25778 | 88.9% |
| finance | 25032 | 22477 | 89.8% |
| gaming | 32686 | 29251 | 89.5% |
| healthcare-synthetic | 25104 | 22419 | 89.3% |
| hr | 26320 | 23418 | 89.0% |
| iot | 26236 | 23490 | 89.5% |
| logistics | 29648 | 26177 | 88.3% |
| saas | 22027 | 19954 | 90.6% |
| travel | 22663 | 19958 | 88.1% |
| workspace | 21179 | 18807 | 88.8% |

## Aceitacao por categoria

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| aggregation | 83921 | 75655 | 90.1% |
| atlas_search | 38530 | 31291 | 81.2% |
| crud | 53811 | 48758 | 90.6% |
| dsl | 42842 | 38686 | 90.3% |
| indexes | 13857 | 12147 | 87.7% |
| json_ejson | 33479 | 30184 | 90.2% |
| operators | 33560 | 30777 | 91.7% |

## Aceitacao por origem do prompt

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| base | 200000 | 181403 | 90.7% |
| rerender | 100000 | 86095 | 86.1% |

## Aceitacao por contexto re-renderizado

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| None | 200000 | 181403 | 90.7% |
| full | 49885 | 46305 | 92.8% |
| irrelevant | 25270 | 19944 | 78.9% |
| none | 24845 | 19846 | 79.9% |

## Greedy por context_kind (campo inexistente = G2.2)

| context_kind | n | valida | campo inexistente |
|---|---:|---:|---:|
| irrelevant | 45410 | 82.4% | 12.2% |
| full | 213753 | 97.0% | 2.2% |
| none | 40837 | 83.5% | 11.4% |

## Principais motivos de rejeicao (greedy)

| etapa | motivo | n |
|---|---|---:|
| decontam | coincide com avaliacao: ngram# | 4732 |
| decontam | coincide com avaliacao: minhash | 4558 |
| semantic | resposta incompleta: # tokens sem EOS nem eco do sufixo | 763 |
| fields | campo/estrutura invalido: history.change.from | 474 |
| fields | campo/estrutura invalido: info.window | 283 |
| fields | campo/estrutura invalido: extra.window | 241 |
| fields | campo/estrutura invalido: attrs.window | 222 |
| syntax | SyntaxError: Unexpected token '}' | 179 |
| fields | campo/estrutura invalido: meta.window | 179 |
| fields | campo/estrutura invalido: meta.quota | 168 |
| fields | campo/estrutura invalido: meta.quota.window.at | 167 |
| fields | campo/estrutura invalido: meta.extra | 159 |
| fields | campo/estrutura invalido: info.quota | 155 |
| fields | campo/estrutura invalido: attrs.quota | 145 |
| decontam | coincide com avaliacao: approx_key | 142 |
| fields | campo/estrutura invalido: meta.quota.at | 140 |
| fields | campo/estrutura invalido: info.quota.at | 108 |
| fields | campo/estrutura invalido: Meta.Extra.Attrs | 106 |
| fields | campo/estrutura invalido: criado_em | 102 |
| fields | campo/estrutura invalido: Address.Street | 97 |
| fields | campo/estrutura invalido: endereco.rua | 88 |
| fields | campo/estrutura invalido: meta.quota.window | 88 |
| syntax | SyntaxError: missing ) after argument list | 87 |
| fields | campo/estrutura invalido: versionNo | 82 |
| api | ObjectId invalido: '#' | 78 |

Dedup {'exact': 725, 'near_dup': 1072}; decontaminacao (regra oficial k=2, blind not_sealed): removidos 9607, 2a passada 0.
Escritas vistas pelo guard do driver: **0**; operacoes enviadas ao driver: {'aggregate': 11939894, 'countDocuments': 1076776, 'find': 5974752, 'findOne': 307367, 'distinct': 357959}.
