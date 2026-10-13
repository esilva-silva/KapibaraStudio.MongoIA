# Dataset v1 - snippets e FIM (relatorio)

Gerado com `gen_snippets.py --ver v1 --seeded-only`, `validate_snippets.py --ver v1`, `gen_fim.py --ver v1 --validation ...` (seed 20261008, configs/dataset.yaml: 20000 snippets, 60000 FIM).

## Cobertura
- 110/110 schemas de data/schemas presentes em data/fixtures/manifest.json (domain, schema_id); 110 bancos kapi_fx_*, 854 colecoes, 940100 docs. `--seeded-only` mantem os 110 (nenhum nao semeado). O manifest.json de v0 tambem ja registrava 110 schemas.
- Colecoes sem fixture: 0.

## Totais
| conjunto | n | train | val | test | bench |
|---|---|---|---|---|---|
| snippets | 20000 | 13671 | 945 | 3843 | 1541 |
| fim | 60000 | 41115 | 2830 | 11433 | 4622 |

Split por grupo (`domain/schema::template`); templates reservados (10%) e dominios holdout (education, legal) vao para test/bench, por isso train < 88% do alvo nominal.

## Snippets por dominio e split
| dominio | n | train | val | test | bench |
|---|---|---|---|---|---|
| commerce | 1282 | 1051 | 55 | 140 | 36 |
| content | 1269 | 988 | 78 | 151 | 52 |
| education | 1074 | 0 | 0 | 656 | 418 |
| energy | 1678 | 1336 | 79 | 227 | 36 |
| finance | 1347 | 1061 | 68 | 198 | 20 |
| gaming | 1788 | 1409 | 99 | 225 | 55 |
| healthcare-synthetic | 1497 | 1154 | 85 | 209 | 49 |
| hr | 1421 | 1081 | 83 | 218 | 39 |
| iot | 1453 | 1124 | 83 | 193 | 53 |
| legal | 1538 | 0 | 0 | 917 | 621 |
| logistics | 1639 | 1303 | 100 | 199 | 37 |
| saas | 1266 | 998 | 72 | 151 | 45 |
| travel | 1428 | 1140 | 80 | 172 | 36 |
| workspace | 1320 | 1026 | 63 | 187 | 44 |

## FIM por dominio e split
| dominio | n | train | val | test | bench |
|---|---|---|---|---|---|
| commerce | 3890 | 3196 | 172 | 417 | 105 |
| content | 3770 | 2958 | 233 | 416 | 163 |
| education | 3191 | 0 | 0 | 1935 | 1256 |
| energy | 5077 | 4028 | 266 | 686 | 97 |
| finance | 4070 | 3210 | 221 | 578 | 61 |
| gaming | 5353 | 4244 | 307 | 659 | 143 |
| healthcare-synthetic | 4511 | 3467 | 247 | 651 | 146 |
| hr | 4206 | 3218 | 265 | 600 | 123 |
| iot | 4320 | 3360 | 225 | 584 | 151 |
| legal | 4646 | 0 | 0 | 2761 | 1885 |
| logistics | 4948 | 3978 | 247 | 598 | 125 |
| saas | 3842 | 3009 | 233 | 451 | 149 |
| travel | 4192 | 3364 | 221 | 530 | 77 |
| workspace | 3984 | 3083 | 193 | 567 | 141 |

## FIM por cut_kind e split (train = alvo +-2 p.p.)
| cut | n | train | val | test | bench | % train | alvo |
|---|---|---|---|---|---|---|---|
| key | 9618 | 6484 | 443 | 1906 | 785 | 0.158 | 0.16 |
| value | 8461 | 5805 | 377 | 1625 | 654 | 0.141 | 0.14 |
| operator | 8479 | 5819 | 381 | 1611 | 668 | 0.142 | 0.14 |
| stage | 7225 | 5025 | 353 | 1318 | 529 | 0.122 | 0.12 |
| member | 6049 | 4111 | 302 | 1136 | 500 | 0.100 | 0.10 |
| partial | 5963 | 4101 | 285 | 1134 | 443 | 0.100 | 0.10 |
| rest | 7076 | 4924 | 348 | 1305 | 499 | 0.120 | 0.12 |
| multiline | 4805 | 3295 | 230 | 917 | 363 | 0.080 | 0.08 |
| stmt_start | 2324 | 1551 | 111 | 481 | 181 | 0.038 | 0.04 |

## Exec-valid (leituras, MongoDB real local, somente leitura, maxTimeMS=2000)

- leituras executadas: 17182; ok 17182; erro 0; escritas puladas (skipped, nunca executadas): 2818
- exec-valid leituras: 1.0000; resultado nao vazio: 0.6791; comandos de escrita vistos no servidor: 0

### Por dominio
| chave | leituras | exec-valid | >=1 doc |
|---|---|---|---|
| commerce | 1102 | 1.0000 | 0.626 |
| content | 1101 | 1.0000 | 0.650 |
| education | 915 | 1.0000 | 0.697 |
| energy | 1417 | 1.0000 | 0.703 |
| finance | 1163 | 1.0000 | 0.683 |
| gaming | 1547 | 1.0000 | 0.713 |
| healthcare-synthetic | 1285 | 1.0000 | 0.702 |
| hr | 1226 | 1.0000 | 0.666 |
| iot | 1240 | 1.0000 | 0.672 |
| legal | 1314 | 1.0000 | 0.635 |
| logistics | 1416 | 1.0000 | 0.676 |
| saas | 1085 | 1.0000 | 0.706 |
| travel | 1231 | 1.0000 | 0.716 |
| workspace | 1140 | 1.0000 | 0.651 |

### Por operacao
| chave | leituras | exec-valid | >=1 doc |
|---|---|---|---|
| aggregate | 6795 | 1.0000 | 0.684 |
| countDocuments | 1053 | 1.0000 | 0.666 |
| distinct | 817 | 1.0000 | 0.898 |
| find | 7975 | 1.0000 | 0.654 |
| findOne | 542 | 1.0000 | 0.690 |

## Limites
- Validacao executa o plano estruturado (EJSON) gravado pelo gerador, nao parseia o JS do texto; consistencia texto<->plano vem do renderizador.
- Escritas (update/delete, 14% dos snippets) nunca executadas; so checagem estatica de campos (analyze_plan: 0 campos inexistentes).
- ~32% das leituras retornam 0 docs, concentrado em filtros `eq`/`in` com valores sorteados; aceitavel (meta >=50% nao vazio).
- FIM e filtrado por exec != error; multiplos cortes por snippet (60000 cortes sobre 20000 snippets).
- Guarda do blind (`blind_guard.check_record/check_text`) aplicada em snippets e FIM, mas data/eval/blind/sealed/manifest.json esta ausente/vazio hoje, logo o filtro e no-op ate o selamento; re-gerar apos selar.
- Contagem de nao vazios varia +-0.01 pp entre execucoes (estado do banco/ordenacao).

## Apendice: mudancas do ciclo 2 (codigo; volume final NAO regerado)

Mudancas em `scripts/data/gen_snippets.py`, `gen_fim.py`, `validate_snippets.py` (so o atalho `exec_mode`) e `configs/dataset.yaml`
(blocos `snippets`, `fim`, `split_policy`). Numeros abaixo vem de uma amostra de verificacao (3000 snippets, seed 20261008, 6000 cortes),
nao do volume v1.

- **Teto do middle com tokenizer Qwen**: `fim.max_tokens` (96) e medido com `tokenizer.json` do Qwen2.5-Coder (mesma contagem da
  etapa 7 do `validate_pipeline`); sem o pacote `tokenizers` o gerador tenta `KAPI_TOKENIZERS_SITE`/`fim.tokenizer_site`/venv do
  bake-off e, por fim, cai em chars/3 com `middle_tokens_method: approx` no registro. Amostra: max middle = 96 tokens, 0 acima do teto
  (v1 tinha 885 respostas > 96 por usar chars/3).
- **split_role**: schema "train" -> so train; "eval" -> so val/test/bench (`split_policy.eval_fractions` val 0.50/test 0.25/bench 0.25);
  holdout vence qualquer papel; ausente = comportamento v1. Efeito colateral documentado: com papel definido, templates reservados
  (10%) de schemas "train" nao vao mais para test (isolamento por schema > por template).
- **Categorias novas** (`category` por registro; fracao do total de snippets em `snippets.extra_categories`): `atlas_search`
  ($search/$searchMeta/$vectorSearch, `exec_mode: skipped_static`), `json_ejson` (dialect `json`, JSON puro com $oid/$date/
  $numberLong/$numberDecimal/$uuid; executavel), `indexes` (createIndex/dropIndex, `skipped_write`; getIndexes/explain implementados
  mas com peso 0 ate a API do Console ser confirmada) e `dsl` (getConnection().getDatabase().getCollection(), ConnectionPool.a.b.c
  e ConnectionPool["Prod Cluster"]["Db"], db.getSiblingDB, ENV.get, cursor em variavel, console.log; plano executavel).
- **Amostragem FIM**: distribuicao conjunta categoria x cut_kind por IPF (marginais = `fim.category_targets` com piso 4% e
  `fim.cut_targets` / `fim.decontam_survival`). Os pesos de reamostragem (alvo/sobrevivencia v1: stage 0.327, multiline 0.378 sao os que
  mais perdem) fazem a distribuicao ESPERADA pos-decontaminacao ficar em +-2 p.p. do alvo; a distribuicao bruta do fim.jsonl
  difere do alvo de proposito (p.ex. stage ~15.8% bruto -> ~12.1% esperado). Amostra: desvio maximo esperado 1.0 p.p.
- Distribuicao por categoria na amostra (cortes): aggregation 27.0%, crud 18.6%, operators 11.4%, atlas_search 13.3%, dsl 13.9%,
  json_ejson 10.9%, indexes 4.9% (piso 4% cumprido).
- Execucao na amostra: 2191 leituras executadas, 0 erros (exec-valid 100%); 428 escritas (createIndex/dropIndex/update/delete) e 378
  atlas_search nunca chegam ao driver (`skipped_write`/`skipped_static`); 0 comandos de escrita vistos pelo listener.

Lacunas e pendencias: (1) a spec nao detalha a API de `ENV` alem de `ENV.get(name)` (usada so como nome de conexao) nem
a gramatica do catalogo `mongodb-language.v1.json` (nao exportado; operadores Atlas Search seguem a documentacao publica);
(2) `$vectorSearch` usa o campo sintetico `embedding` quando o schema nao tem vetor; (3) o parser Console do `chat_validate`/
`validate_pipeline` precisa aceitar getConnection/getSiblingDB/ENV.get/console.log/createIndex/$search e o dialect `json` (hoje
rejeitaria na etapa API/sintaxe); os registros ja carregam `exec_mode` para a etapa 5 pular a execucao; (4) a taxa de sobrevivencia
usada na reamostragem e a do v1 e deve ser recalibrada apos a decontaminacao do ciclo 2.
