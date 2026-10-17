# Pool de prompts da destilacao - 300000 prompts (autocomplete FIM, editor-context-v1)

Gerado 2026-10-17T00:24:40 (git caa947a0431c), seed 20261016, `configs/distill_v2c.yaml`. Chat fora de escopo (Q1 pendente).

## Composicao

- base (contexto natural do v2c): **200000**; re-renderizados (`make_context(force_kind)`): **100000**; treino visto: 1000615.
- none + irrelevant: **86247 (28.7%)** (none 40837, irrelevant 45410); piso exigido 0.25.
- origem: {'base': 200000, 'rerender': 100000}; re-render por tipo: {'irrelevant': 25270, 'full': 49885, 'none': 24845}.
- gold executavel: {'skipped_static': 38530, 'skipped_write': 39793, 'ok': 221677}.

## Isolamento G2.7

- Indice de avaliacao: {'eval_ids': 68998, 'groups': 41099, 'schemas': 38, 'snippets': 57471, 'prompts': 68987, 'prefix_suffix': 68146}.
- Apenas `train`; dominios holdout ['education', 'legal'] nunca; blind: **not_sealed** (via `blind_guard`, manifesto lido uma vez).
- Descartados na amostragem: {'eval_prefix_suffix': 1859, 'eval_prompt_sha': 23}; **violacoes no pool final: 0**.
- Regra de decontaminacao do filtro (mesma de `configs/dataset.yaml`): `ngram_min_content: 2`.

## Cobertura vs corpus de treino (tolerancia +-2.0 p.p. em cut_kind/dominio/cursor_role; pior desvio 0.27 p.p.; ok=True)

### cut_kind

| valor | treino % | pool % | delta pp |
|---|---:|---:|---:|
| key | 16.1 | 16.14 | +0.04 |
| value | 14.01 | 14.12 | +0.11 |
| operator | 13.41 | 13.51 | +0.1 |
| stage | 12.47 | 12.46 | -0.0 |
| rest | 12.07 | 11.8 | -0.27 |
| member | 10.02 | 10.08 | +0.06 |
| partial | 10.0 | 10.06 | +0.06 |
| multiline | 8.07 | 7.91 | -0.15 |
| stmt_start | 3.85 | 3.91 | +0.06 |

### domain

| valor | treino % | pool % | delta pp |
|---|---:|---:|---:|
| gaming | 10.92 | 10.9 | -0.02 |
| logistics | 9.89 | 9.88 | -0.0 |
| energy | 9.64 | 9.67 | +0.03 |
| hr | 8.77 | 8.77 | +0.01 |
| iot | 8.73 | 8.75 | +0.01 |
| finance | 8.4 | 8.34 | -0.06 |
| healthcare-synthetic | 8.37 | 8.37 | -0.0 |
| travel | 7.6 | 7.55 | -0.05 |
| saas | 7.3 | 7.34 | +0.05 |
| workspace | 7.08 | 7.06 | -0.02 |
| commerce | 6.76 | 6.77 | +0.01 |
| content | 6.56 | 6.6 | +0.04 |

### cursor_role

| valor | treino % | pool % | delta pp |
|---|---:|---:|---:|
| PropertyKey | 20.11 | 20.19 | +0.08 |
| OperatorName | 14.42 | 14.49 | +0.07 |
| PropertyValue | 14.01 | 14.12 | +0.11 |
| MemberAccess | 13.36 | 13.46 | +0.1 |
| ArrayElement | 12.47 | 12.46 | -0.0 |
| RestOfStatement | 12.07 | 11.8 | -0.27 |
| ObjectBody | 8.07 | 7.91 | -0.15 |
| StatementStart | 3.85 | 3.91 | +0.06 |
| FieldReferenceString | 1.5 | 1.51 | +0.01 |
| StringArgument | 0.15 | 0.15 | -0.0 |

### category

| valor | treino % | pool % | delta pp |
|---|---:|---:|---:|
| aggregation | 27.94 | 27.97 | +0.04 |
| crud | 17.99 | 17.94 | -0.05 |
| dsl | 14.11 | 14.28 | +0.18 |
| atlas_search | 12.89 | 12.84 | -0.05 |
| operators | 11.2 | 11.19 | -0.02 |
| json_ejson | 11.13 | 11.16 | +0.03 |
| indexes | 4.74 | 4.62 | -0.12 |

### context_kind

| valor | treino % | pool % | delta pp |
|---|---:|---:|---:|
| full | 81.97 | 71.25 | -10.72 |
| irrelevant | 10.02 | 15.14 | +5.12 |
| none | 8.01 | 13.61 | +5.6 |

## Hashes

- config_sha256: `c2ae1069a796540d3212f8085c3756ca2d9a14f22964a04b597fa81640e6219b`
- dataset_yaml_sha256: `ca53e9653ae52591f266e3dcc3d14a6816ebd25dca1a5af3c7b5cfabf90c7935`
- eval_index_sha256: `3c929fd02e1019d50b5f512dda64f411c04df7802fc543136f138e7645aa37af`
- eval_docs_sha256: `956b0c8019a5479b070a11e010ac42a650f8a2861e5388fa98003cb91b2066c4`
- pool_stream_sha256: `adf03161c561a8455cd6a38b67c331c491516c30ad4185940cb1149b19c987ea`
- source_manifest_sha256: `2fef4f342d8a21eaf6d7e9eb1f223b3c8094d56c8dcd00e64f04bc20b20cd3cb`
- pool-00000.jsonl: 50000 registros, `4893699fe9252a0b96b84ec88ff93a12ff6b745d22f95cce9893c28b6472ca86`
- pool-00001.jsonl: 50000 registros, `2c16dbd649b93d5c540b3314b2540e3761f1b69183dd527ad858a3d5dabe2648`
- pool-00002.jsonl: 50000 registros, `f5217d0c844923b05d04a07c765fb133fc54ef7d006b69ac7779adbeccc152ca`
- pool-00003.jsonl: 50000 registros, `bdac67f02aeea2892bfdfc853664a4b43ab031a4ced1203e175b2e0889db4422`
- pool-00004.jsonl: 50000 registros, `3505432c0ceb0a7e68c2cd6033d897094f27277f90b3e62c1ab9e787d73820cd`
- pool-00005.jsonl: 50000 registros, `ec9d96b95e0ce2a6b20a70aa2a4ab583596cecba6071e2d0cd1b54afcfce522e`

Tempo de construcao: 107.4 s.
