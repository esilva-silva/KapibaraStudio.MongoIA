# Baseline zero-shot FIM `int8`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 34499), estratificado por cut_kind x dominio, seed 20261020
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Small\exports\KapiCoder-Mongo-Small-ONNX-int8` · carga `onnx:cpu` · SHA-256 do modelo `f7f71e24b606e26d…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `573a4f438d630536…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: cpu · 607.7 s totais (607.7 s de geracao) · VRAM pico 0 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 40.8% | 40.8% | — |
| Normalized match (FIM acc.) | 40.8% | 40.8% | — |
| Prefix match | 41.8% | 41.8% | — |
| Syntax valid | 99.4% | 82.2% | 100.0% |
| Mongo valid (estatico) | 99.4% | 82.2% | 100.0% |
| Exec-valid | 58.2% | 49.2% | 62.4% |
| Campos inexistentes | 2.6% | 1.4% | 0.0% |
| Chars aceitos (media) | 8.436 | 8.436 | — |
| APT (tokens) | 2.708 | 2.708 | — |

Distribuicao do estagio de validacao (pred): {'ok': 291, 'api': 132, 'fields': 16, 'syntax': 61}
Paradas: {'eos': 497, 'max': 3}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 80 | 52.5% | 52.5% | 5.6 | 100.0% | 100.0% | 52.5% | 2.5% |
| member | 52 | 82.7% | 82.7% | 8.75 | 100.0% | 100.0% | 59.6% | 0.0% |
| multiline | 43 | 0.0% | 0.0% | 14.1163 | 100.0% | 100.0% | 60.5% | 7.0% |
| operator | 66 | 48.5% | 48.5% | 6.9242 | 100.0% | 100.0% | 60.6% | 1.5% |
| partial | 57 | 66.7% | 66.7% | 4.8421 | 98.2% | 98.2% | 50.9% | 5.3% |
| rest | 64 | 31.2% | 31.2% | 10.4375 | 98.4% | 98.4% | 62.5% | 3.1% |
| stage | 60 | 1.7% | 1.7% | 15.0333 | 100.0% | 100.0% | 56.7% | 3.3% |
| stmt_start | 12 | 50.0% | 50.0% | 15.6667 | 100.0% | 100.0% | 66.7% | 0.0% |
| value | 66 | 33.3% | 33.3% | 3.2879 | 98.5% | 98.5% | 62.1% | 0.0% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 35 | 42.9% | 42.9% | 8.7714 | 100.0% | 100.0% | 57.1% | 0.0% |
| content | 46 | 47.8% | 47.8% | 8.9783 | 100.0% | 100.0% | 54.4% | 2.2% |
| energy | 41 | 46.3% | 46.3% | 9.1951 | 97.6% | 97.6% | 58.5% | 4.9% |
| finance | 42 | 28.6% | 28.6% | 7.4048 | 97.6% | 97.6% | 57.1% | 2.4% |
| gaming | 47 | 53.2% | 53.2% | 9.2766 | 100.0% | 100.0% | 59.6% | 2.1% |
| healthcare-synthetic | 43 | 23.3% | 23.3% | 8.6047 | 100.0% | 100.0% | 53.5% | 0.0% |
| hr | 42 | 52.4% | 52.4% | 7.7143 | 100.0% | 100.0% | 61.9% | 2.4% |
| iot | 41 | 39.0% | 39.0% | 10.1707 | 100.0% | 100.0% | 61.0% | 4.9% |
| logistics | 33 | 33.3% | 33.3% | 13.4242 | 100.0% | 100.0% | 57.6% | 6.1% |
| saas | 48 | 37.5% | 37.5% | 6.6458 | 100.0% | 100.0% | 60.4% | 4.2% |
| travel | 37 | 51.3% | 51.3% | 5.7297 | 100.0% | 100.0% | 62.2% | 0.0% |
| workspace | 45 | 33.3% | 33.3% | 6.4222 | 97.8% | 97.8% | 55.6% | 2.2% |

