# Baseline zero-shot FIM `int8`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 34499), estratificado por cut_kind x dominio, seed 20261020
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Small\exports\KapiCoder-Mongo-Small-v3-ONNX-int8` · carga `onnx:cpu` · SHA-256 do modelo `23e53b94514a68b1…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `573a4f438d630536…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: cpu · 636.1 s totais (636.1 s de geracao) · VRAM pico 0 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 40.0% | 40.0% | — |
| Normalized match (FIM acc.) | 40.0% | 40.0% | — |
| Prefix match | 40.8% | 40.8% | — |
| Syntax valid | 99.8% | 83.4% | 100.0% |
| Mongo valid (estatico) | 99.8% | 83.4% | 100.0% |
| Exec-valid | 58.6% | 49.8% | 62.4% |
| Campos inexistentes | 2.6% | 1.4% | 0.0% |
| Chars aceitos (media) | 8.17 | 8.17 | — |
| APT (tokens) | 2.634 | 2.634 | — |

Distribuicao do estagio de validacao (pred): {'ok': 293, 'api': 130, 'fields': 17, 'syntax': 59, 'type': 1}
Paradas: {'eos': 499, 'max': 1}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 80 | 43.8% | 43.8% | 4.85 | 100.0% | 100.0% | 47.5% | 6.2% |
| member | 52 | 84.6% | 84.6% | 8.9423 | 100.0% | 100.0% | 59.6% | 0.0% |
| multiline | 43 | 0.0% | 0.0% | 13.5814 | 100.0% | 100.0% | 65.1% | 4.7% |
| operator | 66 | 54.5% | 54.5% | 6.6667 | 100.0% | 100.0% | 62.1% | 0.0% |
| partial | 57 | 68.4% | 68.4% | 5.1404 | 100.0% | 100.0% | 52.6% | 5.3% |
| rest | 64 | 28.1% | 28.1% | 9.8125 | 100.0% | 100.0% | 62.5% | 4.7% |
| stage | 60 | 1.7% | 1.7% | 14.7667 | 100.0% | 100.0% | 60.0% | 0.0% |
| stmt_start | 12 | 50.0% | 50.0% | 15.6667 | 100.0% | 100.0% | 66.7% | 0.0% |
| value | 66 | 31.8% | 31.8% | 3.2273 | 98.5% | 98.5% | 62.1% | 0.0% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 35 | 42.9% | 42.9% | 8.7714 | 100.0% | 100.0% | 57.1% | 0.0% |
| content | 46 | 43.5% | 43.5% | 8.587 | 100.0% | 100.0% | 56.5% | 2.2% |
| energy | 41 | 43.9% | 43.9% | 9.0976 | 97.6% | 97.6% | 56.1% | 7.3% |
| finance | 42 | 23.8% | 23.8% | 5.4762 | 100.0% | 100.0% | 54.8% | 7.1% |
| gaming | 47 | 48.9% | 48.9% | 9.234 | 100.0% | 100.0% | 61.7% | 0.0% |
| healthcare-synthetic | 43 | 23.3% | 23.3% | 8.3488 | 100.0% | 100.0% | 53.5% | 0.0% |
| hr | 42 | 52.4% | 52.4% | 7.9762 | 100.0% | 100.0% | 61.9% | 0.0% |
| iot | 41 | 39.0% | 39.0% | 10.2439 | 100.0% | 100.0% | 61.0% | 4.9% |
| logistics | 33 | 36.4% | 36.4% | 12.9697 | 100.0% | 100.0% | 57.6% | 6.1% |
| saas | 48 | 41.7% | 41.7% | 5.9375 | 100.0% | 100.0% | 64.6% | 2.1% |
| travel | 37 | 48.6% | 48.6% | 6.1351 | 100.0% | 100.0% | 59.5% | 2.7% |
| workspace | 45 | 35.6% | 35.6% | 6.4889 | 100.0% | 100.0% | 57.8% | 0.0% |

