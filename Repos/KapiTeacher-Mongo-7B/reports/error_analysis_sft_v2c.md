# Análise de erros — KapiTeacher-Mongo-7B, SFT v2c (adapter `final`)

Gerada em 2026-10-15 a partir de `reports/eval/sft_v2c-final/predictions.jsonl` (val v2c, n=2.000, contrato `editor-context-v1`, greedy)
e do contexto de cada exemplo em `data/validated/v2c/fim-*.jsonl`. Objetivo: dizer onde a segunda iteração de SFT deve concentrar dados
(gates G2.2: `exec_valid_vs_ceiling` 0,9265 < 0,95 e campos inexistentes 4,05% > 2%).

## 1. Falhas causadas pelo modelo (e não pelo validador)

O validador só executa um subconjunto do Console: o próprio `middle` de referência passa em 1.251/2.000 (62,55%, o teto).
Falha causada pelo modelo = referência válida **e** predição inválida.

| medida | valor |
|---|---|
| referência válida | 1.251 (62,55%) |
| predição válida entre elas (adapter) | 1.139 de 1.251 (91,0%); o `exec_valid_vs_ceiling` é 0,9265 porque conta também acertos onde a referência falha |
| falhas causadas pelo modelo | **112** (5,6% do val) |
| por estágio | `fields` **88 (79%)**, `api` 18, `syntax` 3, `type` 2, `exec` 1 |
| paradas | 1.993 em EOS, 7 em `max_new_tokens`; só 2 falhas por comprimento |

Conclusão: sintaxe e comprimento **não** são o problema (o SFT já os resolveu: base tinha 519 falhas de sintaxe, adapter 246 contra a referência
também falhando). O problema restante é **inventar campo que não existe no schema**.

## 2. Onde as falhas se concentram

Entre os exemplos em que a referência é válida (n por linha):

| cut_kind | n | falha do modelo | campos inexistentes |
|---|---:|---:|---:|
| multiline | 101 | **25,7%** | 21,8% |
| stmt_start | 63 | 15,9% | 12,7% |
| key | 186 | 11,3% | 9,7% |
| stage | 145 | 9,7% | 8,3% |
| rest | 156 | 9,6% | 3,2% |
| partial | 128 | 8,6% | 8,6% |
| member | 137 | 6,6% | 5,1% |
| operator | 168 | 2,4% | 2,4% |
| value | 167 | 1,2% | 0,6% |

| domínio | falha | | domínio | falha |
|---|---:|---|---|---:|
| saas | **13,3%** | | content | 8,2% |
| iot | 11,6% | | hr | 7,5% |
| gaming | 10,9% | | energy | 7,2% |
| logistics | 10,1% | | travel | 3,8% |
| commerce | 9,3% | | finance | 8,3% |
| workspace | 9,1% | | healthcare | 8,7% |

Por papel do cursor: `ObjectBody` 25,7%, `StatementStart` 15,9%, `PropertyKey` 11,9%, `FieldReferenceString` 10,5%, `ArrayElement` 9,7%;
`PropertyValue` só 1,2%.

## 3. A causa dominante: contexto sem o schema

| `context_kind` | n | falha do modelo | campos inexistentes |
|---|---:|---:|---:|
| `full` | 1.038 | **5,3%** | **3,3%** |
| `irrelevant` | 123 | 29,3% | 26,8% |
| `none` | 90 | 23,3% | 23,3% |

- 17% dos exemplos (irrelevant + none) concentram **54 das 88** falhas de campo (61%).
- Com contexto completo o modelo erra campo em 3,3%, perto do alvo de 2%; sem schema visível ele inventa nomes plausíveis.
- Em 139 exemplos (referência válida) algum campo de `fields_used` não aparece nos fatos/`result_fields`; 26 das 88 falhas de campo estão nesse
  grupo (potencialmente inevitáveis sem a informação). As outras 62 têm o campo no contexto e são erro do modelo; **não foi verificado** se o
  campo também aparecia no prefixo/sufixo (verificação por texto simples sobre `facts` + `result_fields` apenas).

## 4. Recomendações para a segunda iteração (em ordem de retorno)

1. **Definir o gate de campos por tipo de contexto.** Medir "campos inexistentes" só em `context_kind=full` (onde o schema é conhecível) e
   reportar `none`/`irrelevant` à parte; hoje o gate mistura casos que nenhum modelo acertaria.
2. **Mais dados de `none`/`irrelevant` com alvo conservador:** o `middle` deve reutilizar nomes que **aparecem no prefixo/sufixo/editor** ou
   usar `_id`/campos genéricos, em vez de nomes inventados. Hoje o alvo é o campo verdadeiro (invisível), que ensina a adivinhar.
3. **Sobre-amostrar `ObjectBody`/`multiline`, `StatementStart` e `PropertyKey`** (os papéis com maior taxa), +2× no corpus; manter `value` e
   `operator` como estão (já ≤ 2,4%).
4. **Saas, iot, gaming, logistics:** auditar os schemas (nomes parecidos, aninhamento, `signature_salt`) e dar mais exemplos desses domínios.
5. **Não** investir em sintaxe ou comprimento: já resolvidos.
6. Reavaliar com o mesmo `ids-from` (mesma amostra) e comparar com este adapter pelo `compare_models.py`.

## 5. Limites desta análise

- Vale para o val v2c (n=2.000; subconjuntos de 60–190 exemplos têm IC95 largos, ±6–10 p.p.): tratar as taxas por domínio/cut como indício.
- O teto 0,6255 é do validador atual; ampliar o subconjunto do Console o move (e muda este relatório).
- `context_kind` vem da simulação do gerador, não de uso real da IDE (a proporção 82/10/8% é uma hipótese).
