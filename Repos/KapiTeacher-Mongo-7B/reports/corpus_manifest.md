# Corpus de SFT do KapiTeacher-Mongo-7B (dataset v2c)

Gerado por `scripts/teacher/build_teacher_corpus.py` em 2026-10-13T18:45:22Z (git `be8cac0c559c`, seed 20261008), 55.8 s. Config `Repos\KapiTeacher-Mongo-7B\configs\corpus_v2c.yaml` sha256 `1f048d8b28c38ca8`. Saida: `data\teacher\v2c` (gitignored; manifest.json sha256 `a6cb47bd38244cc0`).

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

## 2. Mistura (fracao do corpus)

| componente | alvo | obtido | exemplos | tokens | tokens do alvo | disponivel em train |
|---|---:|---:|---:|---:|---:|---:|
| chat | 0.150 | 0.151 | 21000 | 5929705 | 1566284 | 127006 |
| chat_multiturn | 0.030 | 0.030 | 4200 | 1441514 | 327699 | 22821 |
| completion | 0.200 | 0.196 | 27258 | 7295327 | 1002657 | 873260 |
| correction | 0.060 | 0.060 | 8400 | 3303388 | 720594 | 61485 |
| explain | 0.020 | 0.020 | 2800 | 1084042 | 321592 | 12591 |
| fim_ctx | 0.400 | 0.402 | 56000 | 15661997 | 896681 | 958615 |
| fim_raw | 0.100 | 0.101 | 14000 | 1327499 | 223342 | 894793 |
| negative | 0.040 | 0.040 | 5600 | 1103477 | 169289 | 22607 |

Alvos nao atingidos: nenhum. Descartes na tokenizacao/guardas: {'completion_gt_96': 742}.

**Replay: `none`**. sem fonte local de codigo generico com licenca verificada
Impacto em G2.5 (queda <= 3 p.p. no Generic-JS FIM): sem replay o unico freio ao esquecimento e o LoRA (r=64, lr 1e-4, 1 epoca, base congelada em NF4) e os 10% de FIM cru (+ ~3% de FIM sem cabecalho nos dados), que mantem o infilling nativo; o risco de G2.5 e **medio** e nao pode ser medido hoje (nao existe o conjunto Generic-JS FIM nem fonte de replay). Ver README_job.md.

## 3. Tamanho

- Exemplos: **139258** · tokens reais: **37,146,949** (media 266.7/ex.) · tokens de alvo (com perda): 5,228,138 (14.1%) · linhas (micro-batches de <= 4096 tokens padded): 9,681 · tokens padded 37,220,519 (padding 0.2%).
- Shards: 14 x ate 10000 exemplos; sha256 de cada um em `manifest.json`.
- Validacao do treino (split `val`): 800 exemplos / 235,323 tokens para val_loss; 200 FIM para geracao rapida (EM / mongo_valid).

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
| **corpus final**: ids no corpus x train/val/test/bench ids | 139258 de 139258 em train; val 0; test 0; bench 0; duplicados/problemas 0 |

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
