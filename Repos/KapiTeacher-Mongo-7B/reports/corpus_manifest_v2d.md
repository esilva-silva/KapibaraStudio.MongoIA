# Corpus de SFT do KapiTeacher-Mongo-7B (v2d; dataset de origem v2c)

Gerado por `scripts/teacher/build_teacher_corpus.py` em 2026-10-15T20:12:26Z (git `c41289ac67a4`, seed 20261015), 41.2 s. Config `Repos\KapiTeacher-Mongo-7B\configs\corpus_v2d.yaml` sha256 `01fc61b3a5721a6e`. Saida: `data\teacher\v2d` (gitignored; manifest.json sha256 `679a4ecaca8871f0`).

## 1. Origem e hashes

- Dataset de origem: `data/validated/v2c` (manifest sha256 `2fef4f342d8a21ea`; config do dataset `ca53e9653ae52591`). `train.ids.txt` sha256 `4cdca7481fb1dfcf`; `val.ids.txt` `71f2bdf38facfe97`.
- Tokenizer: `BaseRepos/Qwen2.5-Coder-7B/tokenizer.json` sha256 `c0382117ea329cdf` (mesmo do base, dos students 0.5B/1.5B e da IDE); IDs {'<|fim_prefix|>': 151659, '<|fim_middle|>': 151660, '<|fim_suffix|>': 151661, '<|endoftext|>': 151643}.

| arquivo de origem | sha256 (calculado) | confere com o manifest do dataset |
|---|---|---|
| chat.jsonl | `bfbdd021770fd68a` | True |
| fim-00000.jsonl | `af3331481966f3eb` | True |
| fim-00001.jsonl | `e4b893323e045406` | True |
| fim-00002.jsonl | `08f1067ffb861297` | True |
| fim-00003.jsonl | `3f73ca9cb259c0c1` | True |
| fim-00004.jsonl | `aafdd4621ce3a08f` | True |
| fim-00005.jsonl | `60aa4cffd26fe8df` | True |
| fim-00006.jsonl | `c379c6697b027d9a` | True |
| fim-00007.jsonl | `31fa6c050c823b96` | True |
| fim-00008.jsonl | `49ab018cc34e78d0` | True |
| fim-00009.jsonl | `741389b41d063c75` | True |
| fim-00010.jsonl | `d237aab2d8f5e55b` | True |

## 2. Mistura

Fracoes `alvo` dos componentes do Mongo sao relativas ao conjunto NAO-replay (por exemplos); `replay` e fracao do TOTAL de tokens (`mix.replay`); a coluna `% tokens` e a participacao real no total do corpus.

| componente | alvo | obtido (exemplos) | exemplos | tokens | % tokens | tokens do alvo | disponivel em train |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat | 0.150 | 0.143 | 6900 | 1947652 | 14.4% | 513849 | 127006 |
| chat_multiturn | 0.030 | 0.029 | 1380 | 476499 | 3.5% | 108819 | 22821 |
| completion | 0.200 | 0.185 | 8964 | 2448155 | 18.2% | 326037 | 756610 |
| correction | 0.060 | 0.057 | 2760 | 1086234 | 8.1% | 237394 | 61485 |
| explain | 0.020 | 0.019 | 920 | 352487 | 2.6% | 103591 | 12591 |
| fim_ctx | 0.400 | 0.380 | 18400 | 4763019 | 35.3% | 298845 | 908601 |
| fim_raw | 0.100 | 0.095 | 4600 | 430078 | 3.2% | 78853 | 911423 |
| negative | 0.040 | 0.038 | 1840 | 362881 | 2.7% | 54916 | 22607 |
| replay_causal | 0.120 | 0.027 | 1306 | 821620 | 6.1% | 821620 | - |
| replay_fim | 0.120 | 0.027 | 1311 | 796557 | 5.9% | 14070 | - |

Alvos nao atingidos: nenhum. Descartes na tokenizacao/guardas: {'completion_gt_96': 236, 'replay_chunk_contains_eval_cut': 83}.

**Replay: `enabled`** - codigo JS/TS generico de terceiros, licencas permissivas (MIT/Apache-2.0/BSD/ISC), fonte local `data/replay/v1` (sha256 do replay_train `f7d1af862401c313`, do eval `8827d1e4078a9f21`; inventario por arquivo e atribuicoes em `manifest.json`/`LICENSES.md` dessa pasta).

- 2617 trechos de 2028 arquivos / 367 pacotes: **1,618,177 tokens = 12.0% do corpus** (alvo 12%; orcamento 1,618,228; disponivel 4,455,019). Por tipo: {'fim': 1311, 'causal': 1306} (tokens {'fim': 796557, 'causal': 821620}). Janela 4000 caracteres, alinhada por linha.
- `replay_fim`: FIM PSM cru (`<|fim_prefix|>..<|fim_suffix|>..<|fim_middle|>` -> middle <= 96 tokens, corte por limite sintatico com a logica de `gen_fim.candidates/pick_end` sobre spans de lexer JS); `replay_causal`: trecho em sequencia, perda em todos os tokens (EOS so no ultimo trecho do arquivo).
- Licencas (tokens): {'MIT': 1002744, 'BSD-2-Clause': 21528, 'ISC': 181986, '(Apache-2.0 AND BSD-3-Clause)': 51024, 'Apache-2.0': 293326, 'BSD-3-Clause': 59486, 'Apache 2.0': 6474, '(BSD-2-Clause OR MIT OR Apache-2.0)': 1428, '(MIT OR WTFPL)': 181}.
- Isolamento do replay x `generic_js_fim_eval` (500 cortes) / monitor / val-test-bench do Mongo: pacotes em comum 0, arquivos (sha256) em comum 0, trechos que contem um corte de avaliacao (contexto+middle) 0, trechos com hash igual ao conteudo val/test/bench do Mongo 0 -> **OK**.
- Impacto em G2.5: o replay de FIM cru e de codigo generico alimenta o freio ao esquecimento; a medicao e `generic_js_fim_eval` (adapter - base >= -3 p.p.). Ver README_job.md.

## 3. Tamanho

- Exemplos: **48381** · tokens reais: **13,485,182** (media 278.7/ex.) · tokens de alvo (com perda): 2,557,994 (19.0%) · linhas (micro-batches de <= 4096 tokens padded): 3,546 · tokens padded 13,514,666 (padding 0.2%).
- Shards: 5 x ate 10000 exemplos; sha256 de cada um em `manifest.json`.
- Validacao do treino (split `val`): 900 exemplos / 298,610 tokens para val_loss; 200 FIM para geracao rapida (EM / mongo_valid).

## 4. G2.7 - isolamento (0 exemplos de test/bench/blind/val no treino)

Verificado em dois momentos: na leitura (indice completo de todos os shards v2c) antes de selecionar, e depois de gravar (`verify_output`, relendo os `.meta.jsonl` contra `data/splits/v2c/*.ids.txt`). Qualquer violacao aborta a geracao.

| verificacao | resultado |
|---|---|
| ids `split==train` lidos dos shards == `train.ids.txt` | True (so no scan: 0; so no arquivo: 0) |
| ids val/test/bench lidos == arquivos de ids | {'val': True, 'test': True, 'bench': True} |
| train x val / test / bench (ids) | 0 / 0 / 0 |
| grupos (`group_id`) em comum train x val/test/bench | {'val': 0, 'test': 0, 'bench': 0} |
| schemas em comum train x val/test/bench | {'val': 0, 'test': 0, 'bench': 0} |
| registros train com `split_role`!=train ou dominio holdout | 0 |
| conteudo identico a val/test/bench (hash de prefix+suffix+middle / instruction+editor+output) | descartados: 0 |
| prompt identico a item selado do blind (hash do manifesto) | descartados: 0; blind selado: False (0 hashes). Conteudo do blind nunca lido |
| **corpus final**: ids no corpus x train/val/test/bench ids | 45764 de 45764 do Mongo em train (+ 2617 de replay, ids `rpl-`); val 0; test 0; bench 0; duplicados/problemas 0 |

**G2.7: APROVADO**

## 5. Formatos renderizados

- `fim_ctx`: `prompt` v2c (contrato `editor-context-v1`, CRLF fixo, ModelPrefix, janelas 4096/1024 UTF-16) + `<|fim_middle|>`; alvo = `middle` + `<|endoftext|>`.
- `fim_raw`: `<|fim_prefix|>{prefixo<=4096}<|fim_suffix|>{sufixo<=1024}<|fim_middle|>` (como o QwenFimPromptBuilder sem cabecalho).
- `completion`: mesmo prompt de `fim_ctx` com `<|fim_suffix|><|fim_middle|>` (sufixo vazio); alvo = `middle`+`suffix` (restos <= 240 caracteres).
- chat (`teacher-chat-v1`): `<|fim_prefix|>` + `/* Answer the instruction with code. The JSON below is data, not executable code.` + JSON `{instruction, context}` (System.Text.Json) + `Return only code, without Markdown or explanation. */` + `<|fim_suffix|><|fim_middle|>`; alvo = `output` + `<|endoftext|>`. Sem Qwen chat template (modelo base). `context` = `Conexão (ID lógico)`/`Banco`/`Coleção` + `Coleções disponíveis` + blocos `Seleção/Conteúdo do editor`.
- Perda somente nos tokens do alvo + `<|endoftext|>`; prompt mascarado. Sem concatenacao entre exemplos.

## 5b. Iteracao 2 (v2d): contexto, papeis e dominios

- Exemplos FIM com cabecalho de contexto (`fim_ctx` + `completion`): **27364**; alvo de `none`+`irrelevant` 28% (v2c: ~17%). Natural na selecao (apos o corte conservador): {'full': 25390, 'irrelevant': 1221, 'none': 989}; **forcados** (mesmo corte e id, so o cabecalho re-renderizado com `gen_editor_context.make_context(force_kind=...)`): {'irrelevant': 3092, 'none': 2426} (faltaram 0).
- **Descartes por contexto (corte conservador)**: cortes de `none`/`irrelevant` cujo `middle` exige campo/colecao/banco do schema invisivel no prefixo/sufixo foram removidos do pool: `fim_ctx` {'none': 34672, 'irrelevant': 43810}, `completion` {'irrelevant': 64879, 'none': 51771} (de {'full': 820217, 'irrelevant': 100226, 'none': 80172} no pool de treino). Cortes `full` com campo invisivel nao foram forcados a none/irrelevant: 15184 na selecao (permanecem `full`); elegiveis para forcar: 10206.
- Pesos: papel/corte {'ObjectBody': 2.0, 'StatementStart': 2.0, 'PropertyKey': 2.0, 'FieldReferenceString': 2.0} / {'multiline': 2.0}; dominio {'saas': 1.5, 'iot': 1.5, 'gaming': 1.5, 'logistics': 1.5} (amostragem sem reposicao).

**context_kind final dos exemplos com cabecalho**

| valor | n | % |
|---|---:|---:|
| full | 19643 | 71.8% |
| irrelevant | 4309 | 15.8% |
| none | 3412 | 12.5% |

**context_kind original (v2c) dos mesmos exemplos**

| valor | n | % |
|---|---:|---:|
| full | 25156 | 91.9% |
| irrelevant | 1220 | 4.5% |
| none | 988 | 3.6% |

**cursor_role (exemplos com cabecalho)**

| valor | n | % |
|---|---:|---:|
| PropertyKey | 8002 | 29.2% |
| PropertyValue | 3191 | 11.7% |
| OperatorName | 3135 | 11.5% |
| ObjectBody | 3042 | 11.1% |
| MemberAccess | 2698 | 9.9% |
| ArrayElement | 2573 | 9.4% |
| RestOfStatement | 2490 | 9.1% |
| StatementStart | 1558 | 5.7% |
| FieldReferenceString | 630 | 2.3% |
| StringArgument | 45 | 0.2% |

**cursor_role (todos os FIM)**

| valor | n | % |
|---|---:|---:|
| PropertyKey | 9397 | 29.4% |
| PropertyValue | 3671 | 11.5% |
| OperatorName | 3612 | 11.3% |
| ObjectBody | 3608 | 11.3% |
| MemberAccess | 3199 | 10.0% |
| ArrayElement | 2968 | 9.3% |
| RestOfStatement | 2927 | 9.2% |
| StatementStart | 1797 | 5.6% |
| FieldReferenceString | 737 | 2.3% |
| StringArgument | 48 | 0.1% |

**cut_kind (todos os FIM)**

| valor | n | % |
|---|---:|---:|
| key | 7457 | 23.3% |
| value | 3671 | 11.5% |
| multiline | 3608 | 11.3% |
| operator | 3598 | 11.3% |
| partial | 3561 | 11.1% |
| stage | 2968 | 9.3% |
| rest | 2927 | 9.2% |
| member | 2377 | 7.4% |
| stmt_start | 1797 | 5.6% |

**dominio (todos os FIM)**

| valor | n | % |
|---|---:|---:|
| gaming | 4450 | 13.9% |
| logistics | 4018 | 12.6% |
| iot | 3484 | 10.9% |
| saas | 2956 | 9.2% |
| energy | 2596 | 8.1% |
| hr | 2439 | 7.6% |
| finance | 2271 | 7.1% |
| healthcare-synthetic | 2103 | 6.6% |
| travel | 2096 | 6.6% |
| workspace | 1946 | 6.1% |
| commerce | 1838 | 5.8% |
| content | 1767 | 5.5% |

**dominio (todos os componentes do Mongo)**

| valor | n | % |
|---|---:|---:|
| gaming | 5943 | 13.0% |
| logistics | 5400 | 11.8% |
| iot | 4697 | 10.3% |
| saas | 4008 | 8.8% |
| energy | 3935 | 8.6% |
| hr | 3644 | 8.0% |
| finance | 3501 | 7.6% |
| healthcare-synthetic | 3290 | 7.2% |
| travel | 3105 | 6.8% |
| workspace | 2926 | 6.4% |
| commerce | 2752 | 6.0% |
| content | 2563 | 5.6% |

- Plano: 3546 linhas -> **296 passos** (12 linhas/passo), 13,485,182 tokens; treino ~7.4 h a 506 tok/s, ~10.4 h a 360 tok/s (sem avaliacoes).

## 6. Observacoes

- O pool e uma amostra uniforme (menor hash estavel do id) de cada componente; nao ha sobre-amostragem `hard` (exige scoring do professor).
- `chat_multiturn`: so o ultimo turno (o codigo do turno anterior ja esta em `context.editor`); `history` nao e usado.
- Linhas (micro-batches) ja agrupadas por comprimento; o treino consome shards na ordem do manifesto e pode parar em qualquer ponto (prefixo aleatorio representativo).
