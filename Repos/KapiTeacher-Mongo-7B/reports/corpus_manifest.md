# Corpus de SFT do KapiTeacher-Mongo-7B (dataset v2c)

Gerado por `scripts/teacher/build_teacher_corpus.py` em 2026-10-13T19:19:18Z (git `21148eb10d61`, seed 20261008), 42.2 s. Config `Repos\KapiTeacher-Mongo-7B\configs\corpus_v2c.yaml` sha256 `5f295b4a24c28d96`. Saida: `data\teacher\v2c` (gitignored; manifest.json sha256 `10e9f8ae0c028007`).

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
| chat | 0.150 | 0.142 | 18450 | 5218796 | 14.1% | 1379200 | 127006 |
| chat_multiturn | 0.030 | 0.029 | 3690 | 1264949 | 3.4% | 287458 | 22821 |
| completion | 0.200 | 0.185 | 23949 | 6412051 | 17.3% | 883062 | 873260 |
| correction | 0.060 | 0.057 | 7380 | 2904874 | 7.8% | 633491 | 61485 |
| explain | 0.020 | 0.019 | 2460 | 954582 | 2.6% | 283060 | 12591 |
| fim_ctx | 0.400 | 0.380 | 49200 | 13768164 | 37.1% | 786994 | 963715 |
| fim_raw | 0.100 | 0.095 | 12300 | 1169758 | 3.2% | 196541 | 897883 |
| negative | 0.040 | 0.038 | 4920 | 970908 | 2.6% | 149247 | 22607 |
| replay_causal | 0.120 | 0.028 | 3607 | 2214500 | 6.0% | 2214500 | - |
| replay_fim | 0.120 | 0.028 | 3612 | 2239487 | 6.0% | 39380 | - |

Alvos nao atingidos: nenhum. Descartes na tokenizacao/guardas: {'completion_gt_96': 651, 'replay_chunk_contains_eval_cut': 83}.

**Replay: `enabled`** - codigo JS/TS generico de terceiros, licencas permissivas (MIT/Apache-2.0/BSD/ISC), fonte local `data/replay/v1` (sha256 do replay_train `f7d1af862401c313`, do eval `8827d1e4078a9f21`; inventario por arquivo e atribuicoes em `manifest.json`/`LICENSES.md` dessa pasta).

- 7219 trechos de 4429 arquivos / 555 pacotes: **4,453,987 tokens = 12.0% do corpus** (alvo 12%; orcamento 4,454,193; disponivel 4,455,143). Por tipo: {'fim': 3612, 'causal': 3607} (tokens {'fim': 2239487, 'causal': 2214500}). Janela 4000 caracteres, alinhada por linha.
- `replay_fim`: FIM PSM cru (`<|fim_prefix|>..<|fim_suffix|>..<|fim_middle|>` -> middle <= 96 tokens, corte por limite sintatico com a logica de `gen_fim.candidates/pick_end` sobre spans de lexer JS); `replay_causal`: trecho em sequencia, perda em todos os tokens (EOS so no ultimo trecho do arquivo).
- Licencas (tokens): {'Apache-2.0': 787576, 'MIT': 2771637, 'ISC': 481528, 'BSD-3-Clause': 191812, '(Apache-2.0 AND BSD-3-Clause)': 159064, '(BSD-2-Clause OR MIT OR Apache-2.0)': 1979, 'Apache 2.0': 9883, 'BSD-2-Clause': 49617, '(MIT OR WTFPL)': 768, '(MIT OR Apache-2.0)': 123}.
- Isolamento do replay x `generic_js_fim_eval` (500 cortes) / monitor / val-test-bench do Mongo: pacotes em comum 0, arquivos (sha256) em comum 0, trechos que contem um corte de avaliacao (contexto+middle) 0, trechos com hash igual ao conteudo val/test/bench do Mongo 0 -> **OK**.
- Impacto em G2.5: o replay de FIM cru e de codigo generico alimenta o freio ao esquecimento; a medicao e `generic_js_fim_eval` (adapter - base >= -3 p.p.). Ver README_job.md.

## 3. Tamanho

- Exemplos: **129568** · tokens reais: **37,118,069** (media 286.5/ex.) · tokens de alvo (com perda): 6,852,933 (18.5%) · linhas (micro-batches de <= 4096 tokens padded): 9,737 · tokens padded 37,195,214 (padding 0.2%).
- Shards: 13 x ate 10000 exemplos; sha256 de cada um em `manifest.json`.
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
| **corpus final**: ids no corpus x train/val/test/bench ids | 122349 de 122349 do Mongo em train (+ 7219 de replay, ids `rpl-`); val 0; test 0; bench 0; duplicados/problemas 0 |

**G2.7: APROVADO**

## 5. Formatos renderizados

- `fim_ctx`: `prompt` v2c (contrato `editor-context-v1`, CRLF fixo, ModelPrefix, janelas 4096/1024 UTF-16) + `<|fim_middle|>`; alvo = `middle` + `<|endoftext|>`.
- `fim_raw`: `<|fim_prefix|>{prefixo<=4096}<|fim_suffix|>{sufixo<=1024}<|fim_middle|>` (como o QwenFimPromptBuilder sem cabecalho).
- `completion`: mesmo prompt de `fim_ctx` com `<|fim_suffix|><|fim_middle|>` (sufixo vazio); alvo = `middle`+`suffix` (restos <= 240 caracteres).
- chat (`teacher-chat-v1`): `<|fim_prefix|>` + `/* Answer the instruction with code. The JSON below is data, not executable code.` + JSON `{instruction, context}` (System.Text.Json) + `Return only code, without Markdown or explanation. */` + `<|fim_suffix|><|fim_middle|>`; alvo = `output` + `<|endoftext|>`. Sem Qwen chat template (modelo base). `context` = `Conexão (ID lógico)`/`Banco`/`Coleção` + `Coleções disponíveis` + blocos `Seleção/Conteúdo do editor`.
- Perda somente nos tokens do alvo + `<|endoftext|>`; prompt mascarado. Sem concatenacao entre exemplos.

## 6. Observacoes

- O pool e uma amostra uniforme (menor hash estavel do id) de cada componente; nao ha sobre-amostragem `hard` (exige scoring do professor).
- `chat_multiturn`: so o ultimo turno (o codigo do turno anterior ja esta em `context.editor`); `history` nao e usado.
- Linhas (micro-batches) ja agrupadas por comprimento; o treino consome shards na ordem do manifesto e pode parar em qualquer ponto (prefixo aleatorio representativo).
