# v2 / v2c - status

Estado: ciclo concluido; nada rodando em background; sem commit. pytest scripts\data: 155 passed, 4 skipped.
Versao OFICIAL de treino: **v2c** (decontaminacao opcao B, `decontam.ngram_min_content: 2`). v2 (regra literal) fica como baseline.

## Entregas
- data\validated\v2c\{fim-00000..00014.jsonl, chat.jsonl, manifest.json, rejected.jsonl}; data\splits\v2c\*.ids.txt (69 MB, no .gitignore);
  reports\datasets\v2c-validation-report.md (funil, G3.x, literal x B, v1 x v2 x v2c); reports\decontam\8885372255e21e4b.json.
- data\validated\v2 + reports\datasets\v2-validation-report.md / v2-base-validation-report.md: baseline regra literal (G3.3 falha).
- data\contracts\v2\parity_*.jsonl (2.000 registros p/ a prova de paridade da IDE; gerados da v2, refazer para v2c com
  `export_parity_inputs.py --ver v2c --src data\generated\v2c\fim_ctx.jsonl --out-dir data\contracts\v2c`).

## Volume v2c
train 1.247.125 (fim 1.000.615 + chat 246.510), val 45.381, test 28.455, bench 16.938. G3.2/G3.3/G3.4/G3.5 passam;
G3.1 (IDE), G3.6 (blind nao selado), G3.7 (revisao humana), G3.9 (licenses.md) pendentes.
Mistura real (treino): autocomplete 80,2 / chat 13,0 / correcao 4,9 / negativo 1,8 (alvo 76/15/6/3); gargalo = negativos.

## Retomar (seed 20261008; MongoDB no WSL: `wsl -- true` sobe o servico; mantenha uma sessao wsl aberta)
```
set KAPI_TOKENIZERS_SITE=C:\SlopStudio.MongoAI-artifacts\.venv\Lib\site-packages
:: snippets/validation ja existem em data\generated\v2 (2.997.498; ver snippets.manifest.json); para regerar:
env\.venv\Scripts\python.exe scripts\data\gen_snippets.py --seed 20261008 --count 3000000 --ver v2
env\.venv\Scripts\python.exe scripts\data\validate_snippets.py --ver v2
:: v2c
set S=data\generated\v2\snippets.jsonl
env\.venv\Scripts\python.exe scripts\data\gen_fim.py --seed 20261008 --ver v2c --snippets %S% --validation data\generated\v2\validation.jsonl --shard-size 100000
env\.venv\Scripts\python.exe scripts\data\gen_editor_context.py --ver v2c --snippets %S% --shard-size 100000
env\.venv\Scripts\python.exe scripts\data\gen_chat.py --ver v2c --seed 20261008 --snippets %S% --validation data\generated\v2\validation.jsonl --survivors data\generated\v2c\fim.survivors.txt
env\.venv\Scripts\python.exe scripts\data\validate_pipeline.py --ver v2c --snippets %S% --shard-size 100000 --compare data\validated\v1
```
`--workers N` (gen_fim, gen_editor_context, validate_pipeline): saida identica para qualquer N. Passar `--snippets` ao pipeline e
obrigatorio (sem ele a etapa 5 reprova tudo por "banco nao semeado").

## Problemas encontrados
Regra literal de 13-grama removia 85% do treino piloto (estrutura da linguagem) e 100% de atlas_search; vocabulario de argumentos
MQL/Atlas contava como "conteudo" (corrigido em `_STRUCT`); MongoDB do WSL parou entre sessoes; pipeline sem `--snippets` falhou
na 1a tentativa (reexecutado); edicoes de scripts via heredoc interpretavam `\\`.

## Arquivos alterados
configs\dataset.yaml (bloco `decontam`, version v2c, eval_counts, chat.count 300000, survival), .gitignore (splits v2/v2c),
scripts\data\{chat_validate, validate_pipeline, decontaminate, gen_snippets, gen_fim, gen_editor_context, gen_chat, validate_snippets,
render_contract, export_parity_inputs, chat_describe}.py; novos jsonl_shards.py, par.py, tests\test_v2_constructs.py; testes ajustados
em test_chat/test_editor_context/test_generators.
Apagados: data\generated\v2b; data\generated\v2\{fim-*,fim_ctx-*,chat*}.jsonl e manifestos (superseded).
