# Baseline zero-shot FIM `sft_v2d-merged-nf4`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=300 (de 2000), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiTeacher-Mongo-7B\merged\sft_v2d-bf16` · carga `nf4` · SHA-256 do modelo `1b2a6ee8f88f4d3c…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `797407a65e597a09…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 261.4 s totais (226.2 s de geracao) · VRAM pico 5.51 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 40.7% | 40.7% | — |
| Normalized match (FIM acc.) | 40.7% | 40.7% | — |
| Prefix match | 41.3% | 41.3% | — |
| Syntax valid | 99.0% | 84.0% | 100.0% |
| Mongo valid (estatico) | 98.7% | 83.7% | 100.0% |
| Exec-valid | 58.0% | 49.3% | 61.3% |
| Campos inexistentes | 2.0% | 1.7% | 0.0% |
| Chars aceitos (media) | 9.1933 | 9.1933 | — |
| APT (tokens) | 2.9667 | 2.9667 | — |

Distribuicao do estagio de validacao (pred): {'syntax': 38, 'ok': 174, 'api': 77, 'fields': 10, 'type': 1}
Paradas: {'eos': 296, 'max': 4}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 48 | 54.2% | 54.2% | 8.6458 | 100.0% | 100.0% | 50.0% | 4.2% |
| member | 32 | 71.9% | 71.9% | 7.6562 | 100.0% | 100.0% | 62.5% | 0.0% |
| multiline | 24 | 4.2% | 4.2% | 15.875 | 100.0% | 100.0% | 45.8% | 8.3% |
| operator | 39 | 56.4% | 56.4% | 7.641 | 100.0% | 100.0% | 76.9% | 0.0% |
| partial | 30 | 53.3% | 53.3% | 6.6333 | 100.0% | 96.7% | 56.7% | 3.3% |
| rest | 36 | 36.1% | 36.1% | 10.8889 | 100.0% | 100.0% | 63.9% | 0.0% |
| stage | 36 | 2.8% | 2.8% | 12.4722 | 94.4% | 94.4% | 41.7% | 2.8% |
| stmt_start | 12 | 41.7% | 41.7% | 20.9167 | 91.7% | 91.7% | 66.7% | 0.0% |
| value | 43 | 34.9% | 34.9% | 2.9767 | 100.0% | 100.0% | 60.5% | 0.0% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 25 | 32.0% | 32.0% | 10.36 | 100.0% | 100.0% | 56.0% | 4.0% |
| content | 23 | 30.4% | 30.4% | 8.4348 | 95.7% | 95.7% | 47.8% | 0.0% |
| energy | 26 | 50.0% | 50.0% | 8.0769 | 96.2% | 96.2% | 73.1% | 0.0% |
| finance | 27 | 40.7% | 40.7% | 9.7778 | 100.0% | 100.0% | 59.3% | 3.7% |
| gaming | 25 | 40.0% | 40.0% | 9.36 | 100.0% | 100.0% | 48.0% | 4.0% |
| healthcare-synthetic | 26 | 42.3% | 42.3% | 8.6538 | 100.0% | 100.0% | 53.8% | 0.0% |
| hr | 26 | 50.0% | 50.0% | 11.5 | 100.0% | 96.2% | 65.4% | 0.0% |
| iot | 23 | 43.5% | 43.5% | 8.0435 | 100.0% | 100.0% | 60.9% | 0.0% |
| logistics | 24 | 37.5% | 37.5% | 12.6667 | 100.0% | 100.0% | 54.2% | 8.3% |
| saas | 26 | 46.2% | 46.2% | 6.5385 | 100.0% | 100.0% | 50.0% | 0.0% |
| travel | 24 | 33.3% | 33.3% | 9.0 | 100.0% | 100.0% | 62.5% | 0.0% |
| workspace | 25 | 40.0% | 40.0% | 7.92 | 96.0% | 96.0% | 64.0% | 4.0% |

