# env: fixtures MongoDB (fase 03 §4.2)

Servidor: MongoDB 8.0.32 no WSL (Ubuntu), `mongodb://127.0.0.1:27017`, sem auth. Só bancos `kapi_fx_*` são criados/dropados
(`--drop` dropa apenas os bancos `kapi_fx_*` listados no manifest; nunca admin/config/local/sample_mflix/kapi_sim_*).

## Números finais (seed 20261008)
- 110 schemas em 14 domínios (6–10 por domínio; holdout education=6, legal=8), 854 coleções (4–12 por schema), 940.100 documentos.
- 15 schemas escritos à mão + 95 procedurais (biblioteca `ENT` em `build_schemas.py`); 30 schemas pt-BR sem acento, 80 EN;
  estilos camel/snake/pascal, 35 coleções com hífen/espaço, aninhamento até 4 níveis, UUID subtipo 3 e 4, Decimal128, Int64.
- `$jsonSchema` em 48% das coleções (43% com `enum`); índices: 423 únicos, 164 TTL de 10 anos (só em datas >= 2025), 44 parciais, 15 text, multikey.
- Atlas Search/Vector: ~23% das coleções têm `search_indexes` no JSON do schema (`static_only: true`; NÃO criados no servidor).
- seed completo ~118 s; check completo (com hash) ~8 s; dumps 110 arquivos ~76 MB (gitignored: `data/fixtures/dumps/`).
- Atenção: TTL de 10 anos com datas antigas faria o TTL monitor apagar docs; o gerador só usa TTL em campos de data >= 2025-01-01.

## split_role (train|eval) por schema
- Cada schema grava `split_role` no JSON (`data/schemas/*/*.json`) e em `data/fixtures/manifest.json`; regra e parametros em
  `configs/dataset.yaml` (bloco `schema_split`), implementada em `scripts/data/split_roles.py` (`assign_roles`, `role_for`).
- education/legal (holdout) = `eval` sempre; nos demais dominios ~25% dos schemas `eval` (ordem por sha256 de `seed:schema_id`,
  minimo 1 `eval` com >= 6 schemas, estilos de nome/idioma nos dois lados). Resumo: `python scripts/data/split_roles.py`.
- Webhooks (saas): campo `secret` renomeado para `signature_salt` (valores `SALT-000123`), p/ nao disparar o sanitizador
  (`secret ... [:=]` em render_contract/validate_pipeline).

```powershell
# venv (ignorado no git); recriar: py -3.12 -m venv env\.venv; env\.venv\Scripts\pip install -r env\requirements.txt
env\.venv\Scripts\Activate.ps1
python scripts\data\build_schemas.py                     # data\schemas\<domain>\<schema_id>.json (seed 20261008)
python scripts\data\seed_fixtures.py --seed 20261008 --drop [--domains commerce,iot] [--schemas shopfront] [--uri ...]
python scripts\data\check_fixtures.py [--no-hash]        # manifest x servidor; exit 1 se divergir
```

`data\fixtures\manifest.json`: por coleção `count`, `docs_sha256`, `schema_sha256`, índices, validator, seed, versão do servidor.

## Idempotência / determinismo
Cada documento vem de `random.Random(f"{seed}:{schema_id}:{coleção}:{i}")`, datas em faixas fixas (nunca "agora").
`docs_sha256` = sha256 do BSON de todos os docs ordenados por `_id`. Rodar `seed --drop` duas vezes e
`cmp manifest.json` deve ser idêntico; `check_fixtures.py` recalcula o hash lendo do servidor.

## Dump / restore
```powershell
powershell -File env\dump_fixtures.ps1        # data\fixtures\dumps\<db>.archive.gz (mongodump --gzip no WSL)
# restaurar um banco (ou use --nsFrom/--nsTo para outro nome kapi_fx_*):
wsl -d Ubuntu -e mongorestore --gzip --drop --archive=/mnt/c/IA/models/KapibaraStudio.MongoIA/data/fixtures/dumps/kapi_fx_iot_sensorgrid.archive.gz
```
No Git Bash use `MSYS_NO_PATHCONV=1` antes de chamar `wsl ... /mnt/c/...`.
