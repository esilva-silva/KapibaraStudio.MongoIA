# Corpus v3 dos alunos T0 Nano / T1 Small

Gerado 2026-10-24T18:30:34 por `scripts/student/build_student_corpus_v3.py` (config `configs/student_corpus_v3.yaml` sobre `configs/student_corpus.yaml`, seed 20261024, git 2271398ef8f6, 308 s). Dados em `data/student/v3/{t0,t1}/` (gitignored).

## Por que e o que muda

Homologacao v1 (`reports/datasets/homolog-v1-eval.md`): em schemas NOVOS os alunos inventam campos (contexto `full`: T0 15,35%, T1 9,19%, bases ~7%). O corpus v3 ataca o ponto com tres alavancas, todas medidas aqui: (1) **mais schemas novos**: cortes do LLM (Qwen3-14B, rotulo gold validado por execucao do texto real) e da gramatica dos 61 schemas v3 `train` (rotulo gold), somando 42.0% dos exemplos de cortes; (2) **mais none/irrelevant com alvo conservador** (o rotulo so usa nomes visiveis no texto): T1 34.8%, T0 39.1% dos cortes contra ~29% no v2c (teto = oferta de cortes conservadores de cada fonte); (3) **filtro de fidelidade ao schema**: exemplo `full` cujo rotulo usa campo/colecao ausente do prompt inteiro so entra com p=0.25 (o modelo aprenderia a adivinhar nome), e none/irrelevant exige `field_need` <= 1.

**Sem destilacao nova**: o rotulo dos dados novos e gold; o efeito da destilacao (J14 x J10b) foi pequeno e o gold e exato nesses dados (o professor so poderia piorar o rotulo). O destilado atual segue com `teacher_seq`. Cada exemplo grava `source` e `label_source` no `.meta.jsonl`.

## T1 KapiCoder-Mongo-Small (`BaseRepos/Qwen2.5-Coder-1.5B`)

- **309,571 exemplos** (292,812 cortes Mongo + 16,759 replay), **86,000,018 tokens/epoca** (media 277.8/ex.; alvo com perda 13.1%), 6,156 linhas <= 16384 tokens (pad 0.0%), **3,078 passos/epoca**.
- Treino: **1.5 epocas = 4,617 passos (`--total-steps 4617` se < 6,156), 129,000,027 tokens** -> **15.65 h** a 2290 tok/s medidos (17.92 h a 2000 tok/s).
- Replay: 16,759 trechos, 10,319,942 tokens = **12.0%** dos tokens.

| fonte | `label_source` | exemplos | tokens | % dos cortes Mongo |
|---|---|---:|---:|---:|
| accepted-v2c-teacher | teacher_seq (fim_raw: gold) | 169,831 | 44,382,514 | 58.0% |
| llm-qwen3-14b | gold | 76,248 | 19,182,252 | 26.0% |
| grammar-v3 | gold | 46,733 | 12,115,310 | 16.0% |
| replay-js | replay | 16,759 | 10,319,942 |  |

- **Smoke GPU** (20 passos + retomada do passo 12, RX 7800 XT confirmada por contadores do Windows; `reports/student/smoke_v3_metrics.json`): **2388 tok/s reais** (mediana 2425), 11.84 s/passo, pico de VRAM alocada **9.17 GB** (reservada 12.2 GB), 275 s de GPU no total -> treino completo a essa velocidade: **15.0 h**.

**Composicao dos cortes Mongo**: LLM + gramatica v3 = **42.0%** (alvo 42.0%); em schemas `v3_*` (qualquer fonte) 30.2%; **none+irrelevant = 34.8%** (full 65.2%); cortes cujo rotulo usa nome do schema ausente do prompt (`need_prompt`=2): 1.3%.

context_kind por fonte: {'accepted': {'full': 0.7196, 'irrelevant': 0.1489, 'none': 0.1316}, 'grammar': {'full': 0.5264, 'irrelevant': 0.2391, 'none': 0.2345}, 'llm': {'full': 0.5773, 'irrelevant': 0.217, 'none': 0.2057}} (teto = oferta de cortes conservadores de cada fonte).

| cut_kind | % | categoria | % | componente | % |
|---|---:|---|---:|---|---:|
| key | 16.2% | aggregation | 28.5% | completion | 26.3% |
| member | 10.2% | atlas_search | 12.6% | fim_ctx | 68.7% |
| multiline | 7.8% | crud | 18.0% | fim_raw | 5.1% |
| operator | 13.6% | dsl | 14.4% |  |  |
| partial | 10.1% | indexes | 4.5% |  |  |
| rest | 11.9% | json_ejson | 11.1% |  |  |
| stage | 12.2% | operators | 11.0% |  |  |
| stmt_start | 3.8% |  |  |  |  |
| value | 14.2% |  |  |  |  |

Funil das fontes (carga + G2.7 + fidelidade):

| fonte | lidos | mantidos | violacoes G2.7 | duplicados | descartes de fidelidade |
|---|---:|---:|---:|---:|---|
| accepted | 270,993 | 244,803 | 0 | 0 | {'unobservable_full': 7266, 'unobservable_blind': 18915} |
| llm | 192,275 | 185,045 | 0 | 0 | {'unobservable_full': 6833} |
| grammar | 151,177 | 139,084 | 0 | 3 | {'unobservable_full': 11502} |

Decontaminacao oficial (k=2) reverificada: {"train_in": 568932, "removed_cuts_direct": 53910, "removed_docs": 16169, "by_rule": {"blind": 0, "exact": 106, "ngram13": 46842, "approx_key": 297, "minhash": 7735}, "matched_against": {"test": 2851, "bench": 2921, "homolog": 44540, "val": 4668}, "eval_docs": 72510, "min_content": 2, "dropped_by_source": {"accepted": 10004, "llm": 43906, "grammar": 0}, "kept_by_source": {"accepted": 234799, "llm": 141139, "grammar": 139084}}

Verificacao independente: problemas **0**, fontes {'accepted-v2c-teacher': 169831, 'grammar-v3': 46733, 'llm-qwen3-14b': 76248, 'replay-js': 16759}, `ids_sha256` 22fb98ef1294f5f9...

## T0 KapiCoder-Mongo-Nano (`BaseRepos/Qwen2.5-Coder-0.5B`)

- **238,453 exemplos** (225,835 cortes Mongo + 12,618 replay), **64,999,846 tokens/epoca** (media 272.6/ex.; alvo com perda 13.3%), 4,704 linhas <= 16384 tokens (pad 0.0%), **2,352 passos/epoca**.
- Treino: **2.0 epocas = 4,704 passos (`--total-steps 4704` se < 4,704), 129,999,692 tokens** -> **4.46 h** a 8100 tok/s medidos (5.73 h a 6300 tok/s).
- Replay: 12,618 trechos, 7,799,590 tokens = **12.0%** dos tokens.

| fonte | `label_source` | exemplos | tokens | % dos cortes Mongo |
|---|---|---:|---:|---:|
| accepted-v2c-teacher | teacher_seq (fim_raw: gold) | 130,984 | 33,447,478 | 58.0% |
| llm-qwen3-14b | gold | 58,808 | 14,518,040 | 26.0% |
| grammar-v3 | gold | 36,043 | 9,234,738 | 16.0% |
| replay-js | replay | 12,618 | 7,799,590 |  |

- **Smoke GPU** (20 passos + retomada do passo 12, RX 7800 XT confirmada por contadores do Windows; `reports/student/smoke_v3_metrics.json`): **8484 tok/s reais** (mediana 9013), 3.1 s/passo, pico de VRAM alocada **11.58 GB** (reservada 12.97 GB), 201 s de GPU no total -> treino completo a essa velocidade: **4.3 h**.

**Composicao dos cortes Mongo**: LLM + gramatica v3 = **42.0%** (alvo 42.0%); em schemas `v3_*` (qualquer fonte) 30.5%; **none+irrelevant = 39.1%** (full 60.9%); cortes cujo rotulo usa nome do schema ausente do prompt (`need_prompt`=2): 1.3%.

context_kind por fonte: {'accepted': {'full': 0.6695, 'irrelevant': 0.1776, 'none': 0.1529}, 'grammar': {'full': 0.5048, 'irrelevant': 0.2508, 'none': 0.2443}, 'llm': {'full': 0.5391, 'irrelevant': 0.2365, 'none': 0.2244}} (teto = oferta de cortes conservadores de cada fonte).

| cut_kind | % | categoria | % | componente | % |
|---|---:|---|---:|---|---:|
| key | 16.2% | aggregation | 27.9% | completion | 26.3% |
| member | 10.1% | atlas_search | 13.0% | fim_ctx | 68.7% |
| multiline | 8.1% | crud | 17.7% | fim_raw | 5.1% |
| operator | 13.4% | dsl | 14.3% |  |  |
| partial | 10.0% | indexes | 4.6% |  |  |
| rest | 11.8% | json_ejson | 11.2% |  |  |
| stage | 12.5% | operators | 11.2% |  |  |
| stmt_start | 4.0% |  |  |  |  |
| value | 13.9% |  |  |  |  |

Funil das fontes (carga + G2.7 + fidelidade):

| fonte | lidos | mantidos | violacoes G2.7 | duplicados | descartes de fidelidade |
|---|---:|---:|---:|---:|---|
| accepted | 270,993 | 244,803 | 0 | 0 | {'unobservable_full': 7266, 'unobservable_blind': 18915} |
| llm | 192,275 | 185,045 | 0 | 0 | {'unobservable_full': 6833} |
| grammar | 151,177 | 139,084 | 0 | 3 | {'unobservable_full': 11502} |

Decontaminacao oficial (k=2) reverificada: {"train_in": 568932, "removed_cuts_direct": 53910, "removed_docs": 16169, "by_rule": {"blind": 0, "exact": 106, "ngram13": 46842, "approx_key": 297, "minhash": 7735}, "matched_against": {"test": 2851, "bench": 2921, "homolog": 44540, "val": 4668}, "eval_docs": 72510, "min_content": 2, "dropped_by_source": {"accepted": 10004, "llm": 43906, "grammar": 0}, "kept_by_source": {"accepted": 234799, "llm": 141139, "grammar": 139084}}

Verificacao independente: problemas **0**, fontes {'accepted-v2c-teacher': 130984, 'grammar-v3': 36043, 'llm-qwen3-14b': 58808, 'replay-js': 12618}, `ids_sha256` 434f2e2cbe01b80b...

## Achado: decontaminacao contra a homologacao (importante)

`llm_to_fim.homolog_docs()` (usado no J18) le `data/homolog/v1/items/items.jsonl`, mas o conjunto selado esta em `data/homolog/v1/v1/items/items.jsonl`; com o caminho errado devolve lista vazia SEM erro, e a decontaminacao n-grama do corpus LLM contra a homologacao nao ocorreu (so o hash exato do manifesto). Aqui a leitura vem de `scripts/data/homolog_refs.py` (`homolog_guard.set_dir`, falha alto se o selo existe e os itens nao). Efeito medido pela regra OFICIAL (ngram_min_content 2): {'accepted': 10004, 'llm': 43906, 'grammar': 0} cortes removidos por fonte; `matched_against` {'test': 2851, 'bench': 2921, 'homolog': 44540, 'val': 4668}. Ou seja, **~24% dos cortes do LLM e ~4% do corpus destilado compartilhavam 13-gramas com >= 2 tokens de conteudo com a homologacao** (nomes de campo genericos como status/createdAt). Os modelos atuais (sft_distill_v1) foram treinados com o destilado sem esse corte: a homologacao v1 deles pode estar levemente otimista; o v3 e o primeiro treino decontaminado contra a homologacao. O codigo de `scripts/data/llm` nao foi alterado (fora do escopo); correcao sugerida: `items_path` = `hg.set_dir() / 'items' / 'items.jsonl'`.

## Isolamento

G2.7 por registro (aborta em qualquer violacao): `Isolation` (split train, holdout, papel do schema, ids, grupos, schemas, snippets, prompt_sha256, prefixo+sufixo, cego), hash de conteudo fora de `eval_docs` (v2c) e do v3-eval, schema em papel `train`. G2.7-hom: `hom_guard.assert_no_hom` em cada registro, hashes do selo da homologacao e a decontaminacao oficial (ngram_min_content 2) reexecutada contra v2c + v3-eval + homologacao selada (leitura SO pela API autorizada `homolog_guard.authorized`, linha em `data/homolog/v1/access.log.jsonl`). Replay: pacotes/arquivos/cortes fora do holdout Generic-JS e do conteudo Mongo de avaliacao. Testes: `scripts/student/tests/test_student_corpus_v3.py`.

