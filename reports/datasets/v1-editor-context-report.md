# Dataset v1 - contexto de editor (`editor-context-v1`)

Gerado por `scripts/data/gen_editor_context.py --ver v1` (seed 20261008), sobre `fim.jsonl` (SHA-256 `6b4065ce37dff97c...`). Saida: `data/generated/v1/fim_ctx.jsonl` (SHA-256 `5ccdeeab15ff5039...`).

## Contrato e renderer
- Renderer Python: `scripts/data/render_contract.py` (porta de `AutocompleteContextBuilder.Build/ModelPrefix` + `QwenFimPromptBuilder`).
- Cabecalho com CRLF fixo; embrulho do `ModelPrefix` com `\n` literal; `*/` -> `* /`; `AVAILABLE COMMANDS` por linguagem; janelas 4096/1024 em unidades UTF-16.
- Golden congelado da IDE copiado para `data/contracts/editor-context-v1.golden` (20 casos x context/modelPrefix.false/modelPrefix.true).

- Golden: 20 casos, 114 conferencias (texto, origem H/L dos terminadores, SHA-256 CRLF/LF, prefix/suffix/dictionary); falhas: 0.

## Paridade com a v1 (`bakeoff/data/v1_*.jsonl`, renderer `compact/contracts.py` intocado)

| conjunto | registros com contexto | texto identico (CRLF fixo) | identico se `newline` da v1 honrado | ids de token identicos |
|---|---:|---:|---:|---:|
| v1_benchmark | 1183 | 816 | 1183 | 1183 |
| v1_validation | 7332 | 5060 | 7332 | 7332 |

A v1 gravou `newline: "\n"` em parte dos registros (artefato do gerador antigo); o contrato congelado da IDE e CRLF fixo, entao esses registros divergem so nos separadores do cabecalho.

## Distribuicao

- Registros: 60000 · splits {'train': 41115, 'test': 11433, 'bench': 4622, 'val': 2830}
- `context_kind`: {'full': 49145, 'irrelevant': 6083, 'none': 4772}
- com result_fields: 46808 · com recent_commands: 25640 (por quantidade {0: 34360, 1: 12310, 2: 6990, 3: 6340}) · com input_panel: 1989
- fatos: {'Target': 55228, 'FieldSchema': 538381, 'Operator': 479794, 'Indexes': 55185, 'SimilarStatement': 7072, 'LocalVariable': 10516}

## Orcamento de tokens (tokenizer Qwen2.5-Coder, contexto 2048)

| medida | p50 | p95 | max |
|---|---:|---:|---:|
| tokens do prompt (pos-orcamento) | 245 | 434 | 713 |
| tokens antes do orcamento | 245 | 434 | 713 |

- truncados: 0 de 60000; por tipo de contexto: {'full': {'p50': 263, 'p95': 445, 'max': 713}, 'irrelevant': {'p50': 205, 'p95': 312, 'max': 479}, 'none': {'p50': 67, 'p95': 165, 'max': 295}}

## Limites
- `result_fields`/`recent_commands` sao simulados a partir do schema e dos snippets (nao ha execucao real do explorer da IDE).
- Os fixtures (dumps `mongodump`) nao sao lidos aqui: o resultado simulado usa os caminhos do schema (uniao dos campos opcionais).
- `facts` so alimentam contratos v2 (`kapi-context-v2`); o prompt v1 usa so `editor`.
- Os snippets sao curtos (prefixo <= ~950 caracteres): o orcamento de 2048 tokens nunca e atingido aqui; a IDE real pode enviar ate 4096 caracteres de prefixo. O truncamento esta coberto por testes sinteticos, mas a distribuicao de documentos longos nao existe no dataset.
- Paridade byte a byte contra a IDE real depende do `kapilab contract render/tokenize/diff` (PR 1): entradas em `data/contracts/parity_inputs.jsonl`.

## Baseline zero-shot com contexto (val, n=500 pareado com `fim-raw`)

`eval_teacher.py --contract editor-context-v1 --n 500 --ids-from <sample_ids do baseline> --name <modelo>-ctx` (greedy, 96 tokens, contexto 2048; 7B em NF4, 1.5B em bf16; GPU 553 s e 432 s). Saidas em `Repos/KapiTeacher-Mongo-7B/reports/baseline_base/<modelo>-ctx/`; tabela gerada por `scripts/teacher/compare_ctx.py`. Nao e medida de qualidade final (zero-shot, sem treino com o cabecalho).

### Qwen2.5-Coder-1.5B (n=500, ids pareados)

| metrica | fim-raw | editor-context-v1 | delta (p.p.) |
|---|---:|---:|---:|
| exact_match | 14.8% | 15.8% | +1.0 |
| normalized_match | 14.8% | 15.8% | +1.0 |
| edit_similarity | 35.9% | 34.0% | -1.9 |
| chars_accepted | 4.33 | 4.55 | +0.2 |
| accepted_prefix_tokens | 1.57 | 1.61 | +0.0 |
| syntax_valid | 78.6% | 73.0% | -5.6 |
| mongo_valid | 72.2% | 67.4% | -4.8 |
| exec_valid | 44.2% | 49.2% | +5.0 |
| nonexistent_fields_rate | 12.2% | 5.6% | -6.6 |
| empty_rate | 1.4% | 0.6% | -0.8 |

Tokens do prompt p50/p95/max: fim-raw 69/157/268 ; ctx 226/371/491

### Qwen2.5-Coder-7B (n=500, ids pareados)

| metrica | fim-raw | editor-context-v1 | delta (p.p.) |
|---|---:|---:|---:|
| exact_match | 19.6% | 21.0% | +1.4 |
| normalized_match | 19.6% | 21.0% | +1.4 |
| edit_similarity | 41.8% | 42.0% | +0.3 |
| chars_accepted | 4.64 | 5.06 | +0.4 |
| accepted_prefix_tokens | 1.66 | 1.74 | +0.1 |
| syntax_valid | 86.2% | 83.0% | -3.2 |
| mongo_valid | 80.6% | 80.4% | -0.2 |
| exec_valid | 51.8% | 59.0% | +7.2 |
| nonexistent_fields_rate | 12.2% | 6.0% | -6.2 |
| empty_rate | 1.0% | 0.6% | -0.4 |

Tokens do prompt p50/p95/max: fim-raw 69/157/268 ; ctx 226/371/491


Leitura: o contexto reduz campos inexistentes pela metade e eleva exec-valid (+5 a +7 p.p.), mas piora a sintaxe estatica (-3 a -6 p.p.) e o ganho de EM e pequeno (+1 p.p.). A causa da queda de sintaxe nao foi investigada (hipotese: o modelo base reage ao cabecalho em comentario); o SFT com este contexto deve ser o teste real.
