# Corpus de SFT do KapiTeacher-Mongo-7B (pro_context_v2; dataset de origem pro-context-41k-chatfix-v2)

Gerado por `scripts/teacher/build_teacher_corpus.py` em 2026-10-29T00:12:34Z (git `d01e04b669be`, seed 20261015), 17.4 s. Config `Repos\KapiTeacher-Mongo-7B\configs\corpus_pro_context_v2.yaml` sha256 `0688f9276c5e92c1`. Saida: `data\teacher\pro_context_v2` (gitignored; manifest.json sha256 `940d2ada16e7261b`).

## 1. Origem e hashes

- Dataset de origem: `data/generated/pro-teacher-input-v5/source` (manifest sha256 `658e8f934f891dcd`; config do dataset ``). `train.ids.txt` sha256 `eedaafcb23816d2d`; `val.ids.txt` `dea69f30a5f37499`.
- Tokenizer: `C:/IA/models/KapibaraStudio.MongoIA/BaseRepos/Qwen2.5-Coder-7B/tokenizer.json` sha256 `c0382117ea329cdf` (mesmo do base, dos students 0.5B/1.5B e da IDE); IDs {'<|fim_prefix|>': 151659, '<|fim_middle|>': 151660, '<|fim_suffix|>': 151661, '<|endoftext|>': 151643}.

| arquivo de origem | sha256 (calculado) | confere com o manifest do dataset |
|---|---|---|
| chat.jsonl | `b6829fdda3f159bc` | True |
| fim-00000.jsonl | `b88e0b648e17722e` | True |
| fim-00001.jsonl | `7a6fed893944b099` | True |
| fim-00002.jsonl | `c8e25b17e202393b` | True |

## 2. Mistura

Fracoes `alvo` dos componentes do Mongo sao relativas ao conjunto NAO-replay (por exemplos); `replay` e fracao do TOTAL de tokens (`mix.replay`); a coluna `% tokens` e a participacao real no total do corpus.

| componente | alvo | obtido (exemplos) | exemplos | tokens | % tokens | tokens do alvo | disponivel em train |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat | 0.183 | 0.173 | 6000 | 1818328 | 18.5% | 330664 | 6000 |
| fim_ctx | 0.817 | 0.772 | 26800 | 6813623 | 69.5% | 365885 | 26800 |
| replay_causal | 0.120 | 0.027 | 943 | 598799 | 6.1% | 598799 | - |
| replay_fim | 0.120 | 0.028 | 964 | 578247 | 5.9% | 10320 | - |

Alvos nao atingidos: nenhum. Descartes na tokenizacao/guardas: {'replay_chunk_contains_eval_cut': 83}.

**Replay: `enabled`** - codigo JS/TS generico de terceiros, licencas permissivas (MIT/Apache-2.0/BSD/ISC), fonte local `C:/IA/models/KapibaraStudio.MongoIA/data/replay/v1` (sha256 do replay_train `f7d1af862401c313`, do eval `8827d1e4078a9f21`; inventario por arquivo e atribuicoes em `manifest.json`/`LICENSES.md` dessa pasta).

- 1907 trechos de 1552 arquivos / 311 pacotes: **1,177,046 tokens = 12.0% do corpus** (alvo 12%; orcamento 1,177,084; disponivel 4,455,019). Por tipo: {'fim': 964, 'causal': 943} (tokens {'fim': 578247, 'causal': 598799}). Janela 4000 caracteres, alinhada por linha.
- `replay_fim`: FIM PSM cru (`<|fim_prefix|>..<|fim_suffix|>..<|fim_middle|>` -> middle <= 96 tokens, corte por limite sintatico com a logica de `gen_fim.candidates/pick_end` sobre spans de lexer JS); `replay_causal`: trecho em sequencia, perda em todos os tokens (EOS so no ultimo trecho do arquivo).
- Licencas (tokens): {'MIT': 734380, 'BSD-2-Clause': 14620, 'ISC': 129989, '(Apache-2.0 AND BSD-3-Clause)': 36439, 'Apache-2.0': 212485, 'BSD-3-Clause': 41915, 'Apache 2.0': 6474, '(BSD-2-Clause OR MIT OR Apache-2.0)': 744}.
- Isolamento do replay x `generic_js_fim_eval` (500 cortes) / monitor / val-test-bench do Mongo: pacotes em comum 0, arquivos (sha256) em comum 0, trechos que contem um corte de avaliacao (contexto+middle) 0, trechos com hash igual ao conteudo val/test/bench do Mongo 0 -> **OK**.
- Impacto em G2.5: o replay de FIM cru e de codigo generico alimenta o freio ao esquecimento; a medicao e `generic_js_fim_eval` (adapter - base >= -3 p.p.). Ver README_job.md.

## 3. Tamanho

- Exemplos: **34707** · tokens reais: **9,808,997** (media 282.6/ex.) · tokens de alvo (com perda): 1,305,668 (13.3%) · linhas (micro-batches de <= 4096 tokens padded): 2,498 · tokens padded 9,827,798 (padding 0.2%).
- Shards: 4 x ate 10000 exemplos; sha256 de cada um em `manifest.json`.
- Validacao do treino (split `val`): 900 exemplos / 281,464 tokens para val_loss; 200 FIM para geracao rapida (EM / mongo_valid).

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
| **corpus final**: ids no corpus x train/val/test/bench ids | 32800 de 32800 do Mongo em train (+ 1907 de replay, ids `rpl-`); val 0; test 0; bench 0; duplicados/problemas 0 |

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
