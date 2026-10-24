# Gramatica dos schemas v3 (cortes FIM de treino e de avaliacao)

Gerado por `scripts/data/report_grammar_v3.py` a partir de `scripts/data/build_grammar_v3.py` (rodado via `scripts/data/llm/run_guarded.py`; sem `--include-hom`). Seeds fixas: treino 20261024, avaliacao 20261025. Saidas: `data/generated/v3/grammar/` e `data/validated/v3g[/eval]/` (gitignored); ids em `data/splits/v3g[_eval]/`.

Pipeline: schemas v3 por PAPEL (train: 61; eval: 15; hom: 28, nunca lidos) -> snippets (`gen_snippets`, `--seeded-only`) -> execucao somente-leitura dos snippets (`validate_snippets`) -> prefiltro de decontaminacao oficial (`ngram_min_content: 2`) contra val/test/bench v2c, v3-eval e a homologacao selada -> cortes FIM (`gen_fim`, 9 cut_kinds, ate 96 tokens Qwen, categorias com mais atlas_search/json_ejson/dsl/indexes) -> contexto `editor-context-v1` com `context_kind` CONSERVADOR (>= 25% none/irrelevant, so onde o rotulo usa apenas nomes visiveis) -> `validate_pipeline` (9 etapas, etapa 9 tambem contra as mesmas referencias).

## Treino (schemas `train`)

- Schemas: **61**; snippets brutos: **149,208**.
- **Taxa exec-valid dos snippets** (leituras executadas no MongoDB, somente-leitura): **99.10%** (111,836 ok / 1,015 erro); estaticos (atlas) 18,048; escritas nao executadas 18,309.
- Prefiltro: entram 149,208, ficam **107,686**; descartes {'decontam_hit': 40507, 'exec_error': 1015}; documentos de referencia 72,510 {'val': 31304, 'test': 23376, 'bench': 13660, 'homolog': 4170}.
- Cortes FIM pedidos/gerados: 152,000; contexto: {'full': 104880, 'none': 23504, 'irrelevant': 23616} (none+irrelevant 31.0%, alvo conservador).

Funil da pipeline de 9 etapas (registros rejeitados por etapa):

| etapa | rejeitados |
|---|---:|
| sanitize | 3 |
| syntax | 0 |
| api | 0 |
| fields | 0 |
| exec | 2 |
| semantic | 0 |
| budget | 0 |
| dedup | 119 |
| decontam | 699 |

Entrada 152,000 -> **validados 151,177** (99.5%); por split {train 151,177, val 0, test 0, bench 0}; modo de execucao da etapa 5: `live`.
- Decontaminacao etapa 9: removidos 699; por regra {'blind': 0, 'exact': 1, 'ngram13': 581, 'approx_key': 49, 'minhash': 71}.
- Gates (padrao da pipeline v2c; G3.3 de volume/pisos nao se aplica a um lote parcial): {'G3.2': True, 'G3.3': False, 'G3.4': True, 'G3.5': True, 'mixture': True, 'floors': True, 'coverage': None, 'G2.7': True, 'G2.7-hom': True, 'writes': True}.

Composicao validada (151,177 cortes): cut_kind key 16.2%, value 14.2%, operator 13.1%, stage 12.3%, rest 12.2%, partial 10.2%, member 10.1%, multiline 8.1%, stmt_start 3.5%.

categoria: atlas_search 20.1%, json_ejson 18.3%, aggregation 18.3%, dsl 16.3%, operators 10.2%, crud 10.2%, indexes 6.6%.

context_kind: full 69.0%, irrelevant 15.5%, none 15.5%; dominios: manufacturing 14.3%, retail-pos 12.1%, iot 8.2%, gaming 8.0%, logistics 7.6%, healthcare-synthetic 7.2%, content 6.4%, saas 6.2%, hr 6.1%, finance 5.1%, workspace 5.0%, energy 4.8%, travel 4.6%, commerce 4.5%.

Schemas distintos: 61; snippets distintos: 76,186; hom na saida: 0.

## Avaliacao (schemas `eval`: so val/test/bench, NUNCA treino)

- Schemas: **15**; snippets brutos: **23,871**.
- **Taxa exec-valid dos snippets** (leituras executadas no MongoDB, somente-leitura): **99.14%** (17,891 ok / 156 erro); estaticos (atlas) 2,936; escritas nao executadas 2,888.
- Prefiltro: entram 23,871, ficam **23,715**; descartes {'exec_error': 156}; documentos de referencia 57,471 {'val': 27194, 'test': 19663, 'bench': 10614}.
- Cortes FIM pedidos/gerados: 15,000; contexto: {'irrelevant': 2337, 'full': 10350, 'none': 2313} (none+irrelevant 31.0%, alvo conservador).

Funil da pipeline de 9 etapas (registros rejeitados por etapa):

| etapa | rejeitados |
|---|---:|
| sanitize | 0 |
| syntax | 0 |
| api | 0 |
| fields | 0 |
| exec | 1 |
| semantic | 0 |
| budget | 0 |
| dedup | 1 |
| decontam | 0 |

Entrada 15,000 -> **validados 14,998** (100.0%); por split {train 0, val 5,674, test 5,035, bench 4,289}; modo de execucao da etapa 5: `live`.
- Decontaminacao etapa 9: removidos 0; por regra {'blind': 0, 'exact': 0, 'ngram13': 0, 'approx_key': 0, 'minhash': 0}.
- Gates (padrao da pipeline v2c; G3.3 de volume/pisos nao se aplica a um lote parcial): {'G3.2': True, 'G3.3': False, 'G3.4': True, 'G3.5': True, 'mixture': True, 'floors': False, 'coverage': None, 'G2.7': True, 'G2.7-hom': True, 'writes': True}.

Composicao validada (14,998 cortes): cut_kind key 16.1%, value 14.2%, operator 13.1%, stage 12.3%, rest 12.2%, partial 10.1%, member 10.1%, multiline 8.1%, stmt_start 3.8%.

categoria: atlas_search 20.3%, aggregation 18.2%, json_ejson 18.2%, dsl 16.2%, crud 10.1%, operators 10.1%, indexes 6.9%.

context_kind: full 69.0%, irrelevant 15.6%, none 15.4%; dominios: saas 13.5%, education 12.7%, legal 12.4%, finance 7.0%, gaming 7.0%, energy 6.9%, travel 6.9%, iot 6.9%, retail-pos 6.8%, commerce 6.7%, logistics 6.6%, workspace 6.6%.

Schemas distintos: 15; snippets distintos: 10,869; hom na saida: 0.

## Isolamento e determinismo

- G2.7: treino so de schemas `train` (split `train`); avaliacao so de schemas `eval` (val/test/bench); nenhum schema nos dois lados. G2.7-hom: schemas `hom` nunca copiados ao stage (`assert` + `hom_guard` por registro) e `validate_pipeline` aborta se vir um.
- Decontaminacao oficial (`ngram_min_content: 2`) do treino contra val/test/bench v2c (cache `data/generated/v3/_eval_docs_v2c.jsonl`), v3-eval e a homologacao selada. A leitura da homologacao usa `scripts/data/homolog_refs.py` (API autorizada, log em `access.log.jsonl`); o `llm_to_fim.homolog_docs()` procura o arquivo no caminho errado (`data/homolog/v1/items` em vez de `data/homolog/v1/v1/items`) e devolve lista vazia sem erro - ver `reports/student/corpus_v3_report.md`.
- Reprodutibilidade: seeds fixas, `par.ordered_map` (saida identica para qualquer numero de workers); teste `scripts/data/tests/test_grammar_v3.py`.

