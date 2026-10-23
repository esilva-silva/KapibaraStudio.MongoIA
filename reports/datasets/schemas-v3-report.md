# Schemas v3 (realistas e dificeis) - relatorio

Gerado por `scripts/data/report_schemas_v3.py` a partir de `data/schemas` e `data/fixtures/manifest.json`. Seed do lote: 20261022 (`build_schemas_v3.py`); os 110 schemas antigos (seed 20261008) ficam intactos.

## 1. Contagens

- schemas v3: **104** (train 61, eval 15, hom 28); hom = 26.9% (minimo pedido: 25%). Total no repositorio: 214 schemas.
- colecoes: **1111** (5-14 por schema); documentos semeados: **1.204.450** em 104 bancos `kapi_fx_<dominio>_v3_*` (limite: 2.000.000).

| dominio | schemas | train | eval | hom | colecoes | docs |
|---|---:|---:|---:|---:|---:|---:|
| adtech (novo) | 14 | 0 | 0 | 14 | 150 | 150.750 |
| commerce | 4 | 3 | 1 | 0 | 44 | 48.900 |
| content | 4 | 4 | 0 | 0 | 44 | 40.500 |
| education (holdout) | 2 | 0 | 2 | 0 | 18 | 20.400 |
| energy | 4 | 3 | 1 | 0 | 52 | 63.100 |
| finance | 4 | 3 | 1 | 0 | 34 | 39.450 |
| gaming | 6 | 5 | 1 | 0 | 68 | 75.300 |
| healthcare-synthetic | 4 | 4 | 0 | 0 | 33 | 31.100 |
| hr | 4 | 4 | 0 | 0 | 35 | 34.050 |
| iot | 6 | 5 | 1 | 0 | 64 | 69.450 |
| legal (holdout) | 2 | 0 | 2 | 0 | 20 | 21.450 |
| logistics | 6 | 5 | 1 | 0 | 65 | 71.700 |
| manufacturing (novo) | 8 | 8 | 0 | 0 | 89 | 89.650 |
| retail-pos (novo) | 8 | 7 | 1 | 0 | 87 | 85.400 |
| saas | 6 | 4 | 2 | 0 | 72 | 94.250 |
| telecom (novo) | 14 | 0 | 0 | 14 | 149 | 169.900 |
| travel | 4 | 3 | 1 | 0 | 46 | 48.700 |
| workspace | 4 | 3 | 1 | 0 | 41 | 50.400 |
| **total** | 104 | 61 | 15 | 28 | 1111 | 1.204.450 |

Papeis: `hom` = homologacao (domínios inteiros `telecom` e `adtech`; nunca em train/val/test/bench dos conjuntos antigos); `eval` = education/legal (holdout, so test/bench) + ~15% dos demais por hash; `train` = o resto (cada dominio mantem >= 1 train).

## 2. Estatisticas de heterogeneidade e dificuldade

- campos (todos os niveis): 20.145; **1.021 com 2+ tipos no mesmo campo** (5.1%; `alt`); 644/1111 colecoes tem ao menos um campo heterogeneo.
- tipos de heterogeneidade: Date|string ISO: 409, ref ObjectId|string: 278, numero|string numerica: 101, int|double: 98, double|int: 62, bool|string Y/N: 54, array|escalar: 19.
- mapas dinamicos (chaves variaveis): 424; GeoJSON Point: 74; arrays de arrays: 79; campos raros (presentes em < 10% dos docs): 1834.
- profundidade (`max_depth`, doc=1): 1: 89, 2: 297, 3: 133, 4: 50, 5: 203, 6: 230, 7: 109; **542/1111 colecoes (49%) com profundidade >= 5**; 104/104 schemas tem alguma colecao com >= 5.
- `_id` por colecao: objectid: 882, uuid: 107, string: 89, int64: 33 (referencias entre colecoes seguem o tipo do `_id` alvo, e ~12% delas aparecem tambem como string: `$lookup` exige conversao).
- tipos de colecao: collection: 1030, timeseries: 48, capped: 33 (time-series e capped criados no servidor 8.0.32; `options` no JSON).
- validators `$jsonSchema`: 587/1111 colecoes (114 com `validationAction: warn`; unioes viram `bsonType: [..]`); Atlas Search/Vector (so definicao estatica): 240 colecoes.
- indices: 3650 no total; composto: 1065, sparse: 518, unico: 491, wildcard: 233, TTL: 135, collation (case-insensitive): 120, hashed: 107, 2dsphere: 47, parcial: 31, text: 11.
- nomes bagunçados: estilo dominante {'snake': 48, 'camel': 38, 'pascal': 18}; 24 schemas pt-BR sem acento; em cada colecao ~15% dos nomes de campo fogem do estilo (outro estilo, abreviacoes `cust`/`qty`/`addr`, prefixo de tipo `dt_`/`is_`/`fk_`, sufixos `_v2`/`_old`, `ID`, `_` inicial, MAIUSCULAS) e ha campos legados duplicados (`campo_v2`, `old_campo`, `legacy_campo`); nomes de colecao com `tbl_`/`col_`, `_v2`/`_old`/`_bkp`, hifen ou espaco.

Todo schema traz `realism_notes` (lista de strings: o que foi aplicado por colecao) e cada colecao `v3` (contadores: alt_fields, map_fields, geo_fields, rare_fields, array_of_arrays, id_type, kind).

## 3. O que foi aditivo

- schemas antigos: 110 arquivos e 854 entradas de manifest; **0 com `schema_sha256` divergente** do arquivo atual.
- manifest no `HEAD` x atual para os schemas antigos: 854/854 entradas **identicas** (incl. `docs_sha256`).
- novos arquivos: `scripts/data/{build_schemas_v3,schemas_v3_lib,report_schemas_v3}.py`, 104 `data/schemas/*/v3_*.json`, bancos `kapi_fx_*_v3_*`. Alterados de forma compativel: `seed_fixtures.py` (novos construtos so ativam com chaves ausentes nos schemas antigos; fluxo de RNG inalterado), `check_fixtures.py` (hash canonico p/ time-series, checagem de `options`), `split_roles.py` (papel `hom`, `assign_roles_v3`, `is_hom`, `is_trainable`), `configs/dataset.yaml` (`schema_split.v3`), `env/dump_fixtures.ps1` (`-SchemaPrefix`), testes de split.
- formato do schema: `format_version: 2` nos v3; construtos novos sempre mantem um `t` primario valido para leitores antigos (`alt` = variantes extras; `map` em objeto com `f: {}`; `geo` em objeto GeoJSON; `_id` por colecao; `options`).

## 4. Validacao

- seed: 104 bancos v3 / 1.111 colecoes / 1.204.450 docs em ~188 s (medido em 3 execucoes: 188 s, 208 s, 188 s; mongod no WSL, insercao em lotes de 500)
- idempotencia: 3 execucoes seguidas de `seed_fixtures.py --schema-prefix v3_ --drop`: `docs_sha256` das 1.111 colecoes identicos nas 3; manifest completo byte a byte identico (cmp) entre as 2 ultimas
- check_fixtures: `python scripts/data/check_fixtures.py` (com hash): OK, 1965 colecoes (854 antigas + 1.111 v3), 0 divergencias
- schemas antigos: `build_schemas.py` regenera os 110 arquivos byte a byte iguais; manifest antigo (854 entradas) identico ao do HEAD
- dumps: 104 arquivos `data/fixtures/dumps/kapi_fx_*_v3_*.archive.gz`, 118,1 MB (limite 300 MB; gitignored), ~97 s
- pytest: `pytest scripts/data` -> 227 passed (inclui 11 testes de split_roles com casos `hom`, e os geradores existentes rodando sobre os 214 schemas)
- calibracao: amostra de 139 enums: share do valor mais frequente mediano 0,47 (max 0,67), nenhum > 0,9; todos os 574 campos `alt` obrigatorios no nivel raiz realmente aparecem com 2 tipos no servidor
- ajuste durante a validacao: indices wildcard (`campo.$**`) movidos de `indexes` para `extra_indexes` porque `test_editor_context::test_facts_only_use_schema_fields` mapeia chaves de indice para campos; seed e manifest incluem `extra_indexes`

## 5. Papel `hom` x geradores existentes

Geradores que leem `split_role` e so conhecem `train|eval` (arquivos de outro agente; **nao editados** aqui) e precisam tratar `hom`:

- `scripts/data/gen_snippets.py`: `SPLIT_ROLES = ("train","eval")` -> `hom` vira `None` e cai no split v1 por hash (poderia ir para train!). Precisa filtrar `split_role == "hom"` (excluir de train/val/test/bench e emitir so no conjunto de homologacao).
- `scripts/data/gen_chat.py` (`schema_role`, pool de snippets), `scripts/data/gen_fim.py` / `gen_editor_context.py` (herdam do snippet): rejeitar registros cujo `split_role` seja `hom` fora do conjunto de homologacao.
- `scripts/data/validate_pipeline.py` e `scripts/data/decontaminate.py`: isolamento train x eval deve incluir `hom` como terceiro conjunto (decontaminar hom x train).
- `scripts/distill/build_prompt_pool.py` (`(rec.get('split_role') or 'train') != 'train'` -> ja exclui `hom`, seguro), `scripts/teacher/build_teacher_corpus.py` (usa `split_role`/`schema_split_role` p/ isolamento; conferir que `hom` nunca entra no treino), `scripts/teacher/audit_schemas.py` (so rotulo), `scripts/student/build_student_corpus.py` (propaga o campo).
- Use `split_roles.role_for(...)`/`is_trainable(...)`/`is_hom(...)`: `is_trainable` so e True para `train` (e `None` legado).
