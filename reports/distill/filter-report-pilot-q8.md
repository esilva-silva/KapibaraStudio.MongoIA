# Filtro da destilacao - `pilot-q8`

Prompts 1000 - candidatas 2000 - **aceitos 921 (92.1% dos prompts)** - greedy valida em 94.8%.

Projecao para o pool de 600k: ~552600 exemplos (alvo T1 230000, T0 160000).

## Concordancia professor x gold (greedy)

- Prompts com gold executavel: 736; igual ao gold 39.3%, normalizado 39.3%, plano igual 40.6%, concorda 40.6%, valida 94.6%.
- Todos os prompts: igual 40.7%, concorda 41.9%, valida 94.8%.
- Aceitas que divergem do gold: 482 (52.3%) - `agrees_with_gold:false`, peso 0,85.

## Aceitacao por context_kind

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| full | 797 | 752 | 94.3% |
| irrelevant | 112 | 92 | 82.1% |
| none | 91 | 77 | 84.6% |

## Aceitacao por cut_kind

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| key | 169 | 155 | 91.7% |
| member | 83 | 78 | 94.0% |
| multiline | 79 | 64 | 81.0% |
| operator | 132 | 129 | 97.7% |
| partial | 118 | 114 | 96.6% |
| rest | 123 | 113 | 91.9% |
| stage | 122 | 98 | 80.3% |
| stmt_start | 38 | 35 | 92.1% |
| value | 136 | 135 | 99.3% |

## Aceitacao por dominio

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| commerce | 75 | 70 | 93.3% |
| content | 78 | 75 | 96.2% |
| energy | 106 | 96 | 90.6% |
| finance | 84 | 76 | 90.5% |
| gaming | 99 | 89 | 89.9% |
| healthcare-synthetic | 81 | 73 | 90.1% |
| hr | 87 | 80 | 92.0% |
| iot | 79 | 70 | 88.6% |
| logistics | 102 | 98 | 96.1% |
| saas | 70 | 66 | 94.3% |
| travel | 65 | 58 | 89.2% |
| workspace | 74 | 70 | 94.6% |

## Aceitacao por categoria

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| aggregation | 295 | 271 | 91.9% |
| atlas_search | 135 | 113 | 83.7% |
| crud | 168 | 162 | 96.4% |
| dsl | 143 | 134 | 93.7% |
| indexes | 42 | 40 | 95.2% |
| json_ejson | 110 | 99 | 90.0% |
| operators | 107 | 102 | 95.3% |

## Aceitacao por origem do prompt

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| base | 750 | 690 | 92.0% |
| rerender | 250 | 231 | 92.4% |

## Aceitacao por contexto re-renderizado

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| None | 750 | 690 | 92.0% |
| full | 179 | 172 | 96.1% |
| irrelevant | 40 | 33 | 82.5% |
| none | 31 | 26 | 83.9% |

## Greedy por context_kind (campo inexistente = G2.2)

| context_kind | n | valida | campo inexistente |
|---|---:|---:|---:|
| irrelevant | 112 | 79.5% | 14.3% |
| full | 797 | 97.6% | 1.6% |
| none | 91 | 89.0% | 6.6% |

## Principais motivos de rejeicao (greedy)

| etapa | motivo | n |
|---|---|---:|
| select | valida, mas outra candidata do mesmo prompt venceu o rank | 331 |
| decontam | coincide com avaliacao: ngram# | 14 |
| decontam | coincide com avaliacao: minhash | 11 |
| semantic | resposta incompleta: # tokens sem EOS nem eco do sufixo | 3 |
| fields | campo/estrutura invalido: history.change.from | 3 |
| fields | campo/estrutura invalido: meta.quota.at | 1 |
| fields | campo/estrutura invalido: info.extra | 1 |
| fields | tipo incompativel: _id e objectid mas o valor e date | 1 |
| fields | campo/estrutura invalido: banned.until | 1 |
| fields | campo/estrutura invalido: Meta.Quota | 1 |
| fields | campo/estrutura invalido: attrs.quota | 1 |
| fields | colecao 'invoices' nao e 'prescriptions' | 1 |
| fields | campo/estrutura invalido: weightKg | 1 |
| api | ObjectId invalido: '#' | 1 |
| fields | colecao 'Orders' nao e 'Carts' | 1 |
| fields | campo/estrutura invalido: info.stats.audit.size | 1 |
| fields | campo/estrutura invalido: resultado.em | 1 |
| fields | campo/estrutura invalido: uuid | 1 |
| fields | campo/estrutura invalido: class | 1 |
| fields | campo/estrutura invalido: attrs.quota.label | 1 |
| fields | campo/estrutura invalido: dispositivo.os | 1 |
| api | $oid invalido: {'$oid': '#'} | 1 |
| fields | campo/estrutura invalido: sku | 1 |
| fields | campo/estrutura invalido: reactions.emoji | 1 |
| fields | campo/estrutura invalido: attrs.enabled | 1 |

Dedup {}; decontaminacao (regra oficial k=2, blind not_sealed): removidos 39, 2a passada 0.
Escritas vistas pelo guard do driver: **0**; operacoes enviadas ao driver: {'aggregate': 611, 'countDocuments': 48, 'find': 278, 'findOne': 21, 'distinct': 21}.
