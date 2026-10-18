# Filtro da destilacao - `v2c-final`

Prompts 300000 - candidatas 321098 - **aceitos 270993 (90.3% dos prompts)** - greedy valida em 93.0%.

Projecao para o pool de 600k: ~270990 exemplos (alvo T1 230000, T0 160000).

## Concordancia professor x gold (greedy)

- Prompts com gold executavel: 221677; igual ao gold 38.2%, normalizado 38.2%, plano igual 39.0%, concorda 39.0%, valida 92.9%.
- Todos os prompts: igual 39.8%, concorda 40.6%, valida 93.0%.
- Aceitas que divergem do gold: 150754 (55.6%) - `agrees_with_gold:false`, peso 0,85.

## Aceitacao por context_kind

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| full | 213753 | 200565 | 93.8% |
| irrelevant | 45410 | 36830 | 81.1% |
| none | 40837 | 33598 | 82.3% |

## Aceitacao por cut_kind

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| key | 48406 | 43563 | 90.0% |
| member | 30253 | 27303 | 90.2% |
| multiline | 23740 | 18573 | 78.2% |
| operator | 40538 | 38663 | 95.4% |
| partial | 30176 | 27963 | 92.7% |
| rest | 35408 | 32207 | 91.0% |
| stage | 37390 | 31756 | 84.9% |
| stmt_start | 11738 | 9555 | 81.4% |
| value | 42351 | 41410 | 97.8% |

## Aceitacao por dominio

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| commerce | 20299 | 18282 | 90.1% |
| content | 19804 | 17875 | 90.3% |
| energy | 29002 | 26114 | 90.0% |
| finance | 25032 | 22756 | 90.9% |
| gaming | 32686 | 29646 | 90.7% |
| healthcare-synthetic | 25104 | 22739 | 90.6% |
| hr | 26320 | 23743 | 90.2% |
| iot | 26236 | 23779 | 90.6% |
| logistics | 29648 | 26595 | 89.7% |
| saas | 22027 | 20195 | 91.7% |
| travel | 22663 | 20199 | 89.1% |
| workspace | 21179 | 19070 | 90.0% |

## Aceitacao por categoria

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| aggregation | 83921 | 76684 | 91.4% |
| atlas_search | 38530 | 31699 | 82.3% |
| crud | 53811 | 49355 | 91.7% |
| dsl | 42842 | 39247 | 91.6% |
| indexes | 13857 | 12291 | 88.7% |
| json_ejson | 33479 | 30621 | 91.5% |
| operators | 33560 | 31096 | 92.7% |

## Aceitacao por origem do prompt

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| base | 200000 | 183436 | 91.7% |
| rerender | 100000 | 87557 | 87.6% |

## Aceitacao por contexto re-renderizado

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| None | 200000 | 183436 | 91.7% |
| full | 49885 | 46685 | 93.6% |
| irrelevant | 25270 | 20482 | 81.0% |
| none | 24845 | 20390 | 82.1% |

## Greedy por context_kind (campo inexistente = G2.2)

| context_kind | n | valida | campo inexistente |
|---|---:|---:|---:|
| irrelevant | 45410 | 82.4% | 12.2% |
| full | 213753 | 97.0% | 2.2% |
| none | 40837 | 83.5% | 11.4% |

## Principais motivos de rejeicao (greedy)

| etapa | motivo | n |
|---|---|---:|
| decontam | coincide com avaliacao: ngram# | 4731 |
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

Dedup {'exact': 736, 'near_dup': 1085}; decontaminacao (regra oficial k=2, blind not_sealed): removidos 9902, 2a passada 0.
Escritas vistas pelo guard do driver: **0**; operacoes enviadas ao driver: {'aggregate': 12113960, 'countDocuments': 1087760, 'find': 6040353, 'findOne': 309900, 'distinct': 361464}.
