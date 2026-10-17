# Filtro da destilacao - `pilot-hf-nf4`

Prompts 1000 - candidatas 2000 - **aceitos 919 (91.9% dos prompts)** - greedy valida em 94.5%.

Projecao para o pool de 600k: ~551400 exemplos (alvo T1 230000, T0 160000).

## Concordancia professor x gold (greedy)

- Prompts com gold executavel: 736; igual ao gold 40.4%, normalizado 40.4%, plano igual 41.7%, concorda 41.7%, valida 94.7%.
- Todos os prompts: igual 41.5%, concorda 42.7%, valida 94.5%.
- Aceitas que divergem do gold: 480 (52.2%) - `agrees_with_gold:false`, peso 0,85.

## Aceitacao por context_kind

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| full | 797 | 751 | 94.2% |
| irrelevant | 112 | 92 | 82.1% |
| none | 91 | 76 | 83.5% |

## Aceitacao por cut_kind

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| key | 169 | 153 | 90.5% |
| member | 83 | 78 | 94.0% |
| multiline | 79 | 67 | 84.8% |
| operator | 132 | 129 | 97.7% |
| partial | 118 | 110 | 93.2% |
| rest | 123 | 115 | 93.5% |
| stage | 122 | 97 | 79.5% |
| stmt_start | 38 | 35 | 92.1% |
| value | 136 | 135 | 99.3% |

## Aceitacao por dominio

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| commerce | 75 | 67 | 89.3% |
| content | 78 | 75 | 96.2% |
| energy | 106 | 96 | 90.6% |
| finance | 84 | 79 | 94.0% |
| gaming | 99 | 91 | 91.9% |
| healthcare-synthetic | 81 | 73 | 90.1% |
| hr | 87 | 80 | 92.0% |
| iot | 79 | 72 | 91.1% |
| logistics | 102 | 95 | 93.1% |
| saas | 70 | 65 | 92.9% |
| travel | 65 | 58 | 89.2% |
| workspace | 74 | 68 | 91.9% |

## Aceitacao por categoria

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| aggregation | 295 | 273 | 92.5% |
| atlas_search | 135 | 110 | 81.5% |
| crud | 168 | 161 | 95.8% |
| dsl | 143 | 132 | 92.3% |
| indexes | 42 | 38 | 90.5% |
| json_ejson | 110 | 102 | 92.7% |
| operators | 107 | 103 | 96.3% |

## Aceitacao por origem do prompt

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| base | 750 | 691 | 92.1% |
| rerender | 250 | 228 | 91.2% |

## Aceitacao por contexto re-renderizado

| grupo | prompts | aceitos | taxa |
|---|---:|---:|---:|
| None | 750 | 691 | 92.1% |
| full | 179 | 170 | 95.0% |
| irrelevant | 40 | 33 | 82.5% |
| none | 31 | 25 | 80.7% |

## Greedy por context_kind (campo inexistente = G2.2)

| context_kind | n | valida | campo inexistente |
|---|---:|---:|---:|
| irrelevant | 112 | 82.1% | 11.6% |
| full | 797 | 97.1% | 2.0% |
| none | 91 | 86.8% | 7.7% |

## Principais motivos de rejeicao (greedy)

| etapa | motivo | n |
|---|---|---:|
| select | valida, mas outra candidata do mesmo prompt venceu o rank | 124 |
| decontam | coincide com avaliacao: minhash | 14 |
| decontam | coincide com avaliacao: ngram# | 11 |
| semantic | resposta incompleta: # tokens sem EOS nem eco do sufixo | 4 |
| fields | campo/estrutura invalido: meta.quota.size | 1 |
| fields | campo/estrutura invalido: meta.extra | 1 |
| fields | campo/estrutura invalido: Vitals.Bp.Sis | 1 |
| fields | tipo incompativel: _id e objectid mas o valor e date | 1 |
| api | ObjectId invalido: '#a' | 1 |
| fields | campo/estrutura invalido: Meta.Quota | 1 |
| fields | campo/estrutura invalido: info.window.count | 1 |
| fields | campo/estrutura invalido: attrs.attrs.audit.em | 1 |
| fields | campo/estrutura invalido: attrs.settings | 1 |
| fields | campo/estrutura invalido: history.change.to | 1 |
| fields | campo/estrutura invalido: extra.window | 1 |
| fields | colecao 'invoices' nao e 'prescriptions' | 1 |
| fields | colecao 'Orders' nao e 'Carts' | 1 |
| fields | campo/estrutura invalido: info.stats.audit.size | 1 |
| fields | campo/estrutura invalido: em | 1 |
| fields | campo/estrutura invalido: uid | 1 |
| fields | campo/estrutura invalido: attrs.label | 1 |
| fields | campo/estrutura invalido: sku | 1 |
| fields | campo/estrutura invalido: reactions.emoji | 1 |
| fields | campo/estrutura invalido: attrs.stats | 1 |
| fields | campo/estrutura invalido: stats.spd | 1 |

Dedup {}; decontaminacao (regra oficial k=2, blind not_sealed): removidos 31, 2a passada 0.
Escritas vistas pelo guard do driver: **0**; operacoes enviadas ao driver: {'aggregate': 602, 'countDocuments': 44, 'find': 281, 'findOne': 20, 'distinct': 23}.
