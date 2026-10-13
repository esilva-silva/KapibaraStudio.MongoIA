# Dataset v1 - relatorio de validacao (pipeline de 9 etapas)

Gerado por `scripts/data/validate_pipeline.py` (`validate-pipeline/1`), seed 20261008, commit `64deed17c2a8` (arvore com alteracoes). Execucao (etapa 5): `live`; orcamento (etapa 7): `qwen2.5-coder tokenizer.json`; sintaxe JS: `node vm.Script`.

## 1. Funil por etapa

| etapa | entrada | aprovados | rejeitados | fim (ent->saida) | chat (ent->saida) |
|---|---:|---:|---:|---|---|
| sanitize | 66000 | 65958 | 42 | 60000->59958 | 6000->6000 |
| syntax | 65958 | 65958 | 0 | 59958->59958 | 6000->6000 |
| api | 65958 | 65958 | 0 | 59958->59958 | 6000->6000 |
| fields | 65958 | 65958 | 0 | 59958->59958 | 6000->6000 |
| exec | 65958 | 65958 | 0 | 59958->59958 | 6000->6000 |
| semantic | 65958 | 65958 | 0 | 59958->59958 | 6000->6000 |
| budget | 65958 | 65073 | 885 | 59958->59080 | 6000->5993 |
| dedup | 65073 | 64949 | 124 | 59080->59041 | 5993->5908 |
| decontam | 64949 | 28964 | 35985 | 59041->25964 | 5908->3000 |

Entrada: 66000 registros (60000 fim_ctx + 6000 chat). Saida: 28964 (train 8548, val 3084, test 12337, bench 4995).

### Principais motivos de rejeicao

| etapa | motivo | n |
|---|---|---:|
| decontam | coincide com avaliacao: ngram# | 35749 |
| budget | resposta com # tokens > # | 885 |
| dedup | duplicata exata | 119 |
| decontam | coincide com avaliacao: approx_key | 110 |
| decontam | coincide com avaliacao: ngram#,approx_key | 58 |
| decontam | coincide com avaliacao: exact,minhash | 46 |
| sanitize | credencial/URI mongodb:// (padrao proibido) | 42 |
| decontam | coincide com avaliacao: exact,approx_key,minhash | 11 |
| dedup | near-dup MinHash J~# >= # | 5 |
| decontam | coincide com avaliacao: ngram#,minhash | 5 |
| decontam | coincide com avaliacao: approx_key,minhash | 3 |
| decontam | coincide com avaliacao: minhash | 1 |
| decontam | coincide com avaliacao: ngram#,approx_key,minhash | 1 |
| decontam | coincide com avaliacao: exact,ngram#,minhash | 1 |

## 2. Decontaminacao (etapa 9)

- Treino de entrada: 44533 registros; removidos: **35985** (80.8%); mantidos 8548.
- Documentos de avaliacao indexados: 19719 {'test': 4678, 'bench': 1880, 'val': 1161, 'legacy-v1': 12000}. Conjunto cego: **not_sealed**.

| regra | removidos (qualquer coincidencia) | removidos (1a regra) |
|---|---:|---:|
| blind | 0 | 0 |
| exact | 58 | 58 |
| ngram13 | 35814 | 35813 |
| approx_key | 183 | 113 |
| minhash | 68 | 1 |

Sensibilidade do 13-grama (registros de treino removidos se o 13-grama exigisse >= k tokens de conteudo): k=0: 35814, k=1: 25118, k=2: 9086, k=3: 3393, k=4: 2136 (aplicado: k=0, 0 = regra literal da spec).

Parametros: {'ngram': 13, 'minhash_perm': 128, 'shingle_k': 5, 'bands': 25, 'rows': 5, 'jaccard': 0.7, 'seed': 20261008, 'ngram_min_content': 0}. Relatorio completo: `reports/decontam/1671c7d8ce3e64bd.json`. Verificacao pos-remocao (2a passada treino x avaliacao): 0 registros ainda coincidentes.

## 3. Por split, tipo e domino

| split | tipo | entrada | saida | rejeitados |
|---|---|---:|---:|---:|
| bench | chat | 452 | 450 | 2 |
| bench | fim | 4622 | 4545 | 77 |
| test | chat | 1105 | 1097 | 8 |
| test | fim | 11433 | 11240 | 193 |
| train | chat | 4146 | 1156 | 2990 |
| train | fim | 41115 | 7392 | 33723 |
| val | chat | 297 | 297 | 0 |
| val | fim | 2830 | 2787 | 43 |

| dominio | holdout | entrada | saida | train | val | test | bench |
|---|---|---:|---:|---:|---:|---:|---:|
| commerce | nao | 4236 | 1445 | 710 | 176 | 446 | 113 |
| content | nao | 4136 | 1474 | 587 | 259 | 450 | 178 |
| education | sim | 3514 | 3462 | 0 | 0 | 2099 | 1363 |
| energy | nao | 5565 | 1948 | 827 | 283 | 733 | 105 |
| finance | nao | 4481 | 1553 | 611 | 236 | 637 | 69 |
| gaming | nao | 5940 | 2162 | 953 | 343 | 714 | 152 |
| healthcare-synthetic | nao | 4976 | 1895 | 760 | 273 | 701 | 161 |
| hr | nao | 4602 | 1687 | 615 | 285 | 654 | 133 |
| iot | nao | 4744 | 1775 | 736 | 248 | 628 | 163 |
| legal | sim | 5074 | 4988 | 0 | 0 | 2967 | 2021 |
| logistics | nao | 5480 | 1940 | 884 | 276 | 647 | 133 |
| saas | nao | 4244 | 1526 | 631 | 255 | 482 | 158 |
| travel | nao | 4654 | 1595 | 686 | 242 | 577 | 90 |
| workspace | nao | 4354 | 1514 | 548 | 208 | 602 | 156 |

| kind (fim: cut_kind; chat: kind) | tipo | entrada | saida |
|---|---|---:|---:|
| chat | chat | 3720 | 1787 |
| correction | chat | 1320 | 565 |
| explain | chat | 300 | 141 |
| negative | chat | 660 | 507 |
| key | fim | 9618 | 4561 |
| member | fim | 6049 | 2864 |
| multiline | fim | 4805 | 1817 |
| operator | fim | 8479 | 3705 |
| partial | fim | 5963 | 2844 |
| rest | fim | 7076 | 2920 |
| stage | fim | 7225 | 2364 |
| stmt_start | fim | 2324 | 1027 |
| value | fim | 8461 | 3862 |

## 4. Gates de dataset (spec §6) medidos sobre a saida

| gate | resultado | medicao |
|---|---|---|
| G3.1 paridade de contrato | NAO MEDIDO aqui | depende de ContractProbe/IDE (`data/contracts/parity_inputs.jsonl`) |
| G3.2 validade | PASSA | 28964/28964 com syntax/api/fields ok; exec ok 24215, skipped 4749 (escritas e respostas sem codigo); exec-valid das leituras na entrada do pipeline: 100.0%; chat/correcao com equivalencia ok |
| G3.3 volume e distribuicao | FALHA | train validado 8548 (alvo >= 1.000.000: FALHA); cut_kind +-2pp: FALHA; idiomas +-2pp: ok; dialetos: so console; categorias ausentes no gerador v1: atlas_search, json_ejson, indexes, console_dsl(ENV/cursores) |
| G3.4 vazamento | PASSA | grupos compartilhados entre splits: 0; treino x (val/test/bench/legacy-v1) restantes: 0; blind: not_sealed; legacy-v1 conferido: 12000 docs |
| G3.5 privacidade | PASSA | 0 ocorrencias de padroes proibidos na saida |
| G3.6 blind | PENDENTE | conjunto cego nao selado (`blind: not_sealed`); 400 itens escritos/revisados por pessoa |
| G3.7 revisao humana | PENDENTE | 300 exemplos aleatorios do train (amostrar de `data/splits/v1/train.ids.txt`) |
| G3.8 reproducibilidade | COBERTO POR TESTE (2 execucoes => mesmos sha256) | sha256 de saidas no `manifest.json`; mesmo seed/config/commit => mesmos hashes |
| G3.9 licencas | PENDENTE (licenses.md ausente) | fontes geradas (`source_license: generated`); `data/sources/licenses.md` |

### Gates adicionais da tarefa

- **Chat+correcao <= 25%** (pool de treino validado): 10.0%; na mistura-alvo da config: 21.0% -> PASSA. Pool por componente: {'autocomplete': 7392, 'chat': 666, 'negative': 300, 'correction': 190}; alvo: {'autocomplete': 0.76, 'chat': 0.15, 'correction': 0.06, 'negative': 0.03}; maximo sem repeticao: 3166 exemplos (limitante: `correction`).
- **Piso de grupos pequenos (4%)**: nao aplicavel no v1 (grupos Indexes, Scripts, EJSON, DSL nao sao gerados); informativo, share por cut_kind abaixo de 4%: nenhum.
- **Cobertura**: dominios no train 12; operadores distintos no train 51; operadores raros (<20 docs): ['$add', '$addFields', '$all', '$avg', '$concat', '$count', '$first', '$ifNull', '$map', '$mod', '$multiply', '$round', '$skip', '$subtract', '$sum', '$toUpper', '$year']. cut_kind (train, alvo +-2pp): key 19.4% (alvo 16%), value 16.4% (alvo 14%), operator 14.3% (alvo 14%), stage 2.8% (alvo 12%), member 12.7% (alvo 10%), partial 13.4% (alvo 10%), rest 11.0% (alvo 12%), multiline 6.2% (alvo 8%), stmt_start 3.9% (alvo 4%).
- **G2.7 (consulta ao manifesto)**: ids de val/test/bench no train = 0; holdout em train/val = 0 -> PASSA.
- **0 escritas executaveis**: outputs de chat com escrita executavel = 0; comandos de escrita vistos pelo servidor = 0; documentos FIM que sao escritas (sem execucao) = 4261 -> PASSA.

## 5. Falhas e ajustes propostos (sem alterar geradores)

- G3.3 volume: train validado 8548 << 1.000.000 (v1 e um piloto de 20 mil snippets/60 mil cortes). Ajuste: gen_snippets `snippets.count` ~ 400k+ e `fim.count` ~ 1,5 M, mantendo seed e splits por grupo; mais schemas/templates (a diversidade por grupo e que limita, nao o volume).
- G3.3 cut_kind fora de +-2pp apos o pipeline: ['key', 'value', 'stage', 'member', 'partial']. Ajuste: rebalancear `fim.cut_targets` por sobre-geracao dos cortes afetados (a decontaminacao remove desproporcionalmente cortes de documentos estruturalmente comuns).
- G3.3 categorias: atlas_search, json_ejson, indexes e console_dsl (ENV/cursores) nao sao geradas no v1 (0% do train); a tabela de categorias da spec §4.6 nao e atingivel. Ajuste: novos templates em gen_snippets (Atlas Search/EJSON/createIndex/DSL) ou revisar a tabela de alvos para o escopo do v1.
- G3.4: a decontaminacao removeu 35985 de 44533 itens do treino (81%): templates compartilham 13-gramas/chave aproximada entre grupos (mesmos nomes de campo/colecao em schemas do mesmo dominio). Ajuste de gerador: tornar nomes de colecao/campo disjuntos por split (schemas exclusivos por split, nao so grupos), ou reservar schemas inteiros para val/test/bench.
- Mistura: a componente `correction` limita a mistura-alvo a 3166 exemplos sem repeticao. Ajuste: aumentar `chat.count` (e kinds) proporcionalmente ao fim.count.
- G3.5/etapa 1: 42 cortes FIM rejeitados por padrao proibido: campos de schema chamados `secret`/`token`/`password` (colecao webhooks de saas) disparam o filtro de privacidade da IDE. Ajuste de gerador: renomear esses campos nos schemas (ex.: `signing_key_ref`) ou excluir a colecao; sem isso os cortes sao descartados (perda ~0,07%).
- Etapa 7: 885 registros com resposta acima do limite de tokens medido com o tokenizer Qwen; o gen_fim usa a aproximacao chars/3. Ajuste: gen_fim deve medir `max_tokens` com o tokenizer (ou reduzir `fim.max_tokens` efetivo ~ 80 em chars/3).
- G3.6: conjunto cego nao selado; a decontaminacao contra blind so passa a valer apos `blind_tool.py seal`.

## 6. Limites

- Orcamento de chat usa renderizacao aproximada do `local-agent-fim-v0` (spec §4.4), nao o ContractProbe.
- Etapa 5 em FIM executa o plano do snippet (o documento reconstruido e igual ao texto e ao plano); escritas nunca sao executadas.
- Dedup exato por split usa tokens com literais preservados; near-dup (Jaccard >= 0,85) so no treino, decidido por documento (cortes distintos do mesmo snippet nao sao duplicatas).
- `limite por grupo` (max_per_group) e nomes de pessoas (regras v1) nao estao na config atual e nao foram aplicados.
