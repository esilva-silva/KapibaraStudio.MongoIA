# Baseline zero-shot FIM `int4-trunc-lean0`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 34499), estratificado por cut_kind x dominio, seed 20261020
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-ONNX-int4` · carga `onnx:cpu` · SHA-256 do modelo `4e3ec2d44a0423f9…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `573a4f438d630536…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: cpu · 182.3 s totais (182.3 s de geracao) · VRAM pico 0 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 37.4% | 37.4% | — |
| Normalized match (FIM acc.) | 37.4% | 37.4% | — |
| Prefix match | 38.4% | 38.4% | — |
| Syntax valid | 98.8% | 82.0% | 100.0% |
| Mongo valid (estatico) | 98.8% | 82.0% | 100.0% |
| Exec-valid | 51.4% | 45.4% | 62.4% |
| Campos inexistentes | 8.8% | 4.6% | 0.0% |
| Chars aceitos (media) | 7.654 | 7.654 | — |
| APT (tokens) | 2.492 | 2.492 | — |

Distribuicao do estagio de validacao (pred): {'ok': 257, 'api': 125, 'fields': 54, 'syntax': 64}
Paradas: {'eos': 498, 'max': 2}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 80 | 46.2% | 46.2% | 5.0 | 98.8% | 98.8% | 45.0% | 10.0% |
| member | 52 | 76.9% | 76.9% | 7.75 | 100.0% | 100.0% | 51.9% | 3.9% |
| multiline | 43 | 0.0% | 0.0% | 13.6047 | 100.0% | 100.0% | 37.2% | 32.6% |
| operator | 66 | 43.9% | 43.9% | 6.3333 | 97.0% | 97.0% | 56.1% | 6.1% |
| partial | 57 | 61.4% | 61.4% | 4.4386 | 100.0% | 100.0% | 45.6% | 8.8% |
| rest | 64 | 26.6% | 26.6% | 9.6094 | 100.0% | 100.0% | 59.4% | 7.8% |
| stage | 60 | 1.7% | 1.7% | 12.6833 | 98.3% | 98.3% | 51.7% | 8.3% |
| stmt_start | 12 | 50.0% | 50.0% | 15.6667 | 100.0% | 100.0% | 58.3% | 0.0% |
| value | 66 | 33.3% | 33.3% | 3.0909 | 97.0% | 97.0% | 59.1% | 1.5% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 35 | 40.0% | 40.0% | 7.0286 | 100.0% | 100.0% | 45.7% | 8.6% |
| content | 46 | 30.4% | 30.4% | 7.413 | 97.8% | 97.8% | 54.4% | 4.3% |
| energy | 41 | 41.5% | 41.5% | 7.0 | 95.1% | 95.1% | 56.1% | 7.3% |
| finance | 42 | 26.2% | 26.2% | 7.1429 | 97.6% | 97.6% | 50.0% | 7.1% |
| gaming | 47 | 53.2% | 53.2% | 8.7021 | 100.0% | 100.0% | 53.2% | 8.5% |
| healthcare-synthetic | 43 | 23.3% | 23.3% | 8.9302 | 100.0% | 100.0% | 41.9% | 9.3% |
| hr | 42 | 42.9% | 42.9% | 7.5238 | 100.0% | 100.0% | 52.4% | 9.5% |
| iot | 41 | 43.9% | 43.9% | 9.8537 | 100.0% | 100.0% | 53.7% | 12.2% |
| logistics | 33 | 33.3% | 33.3% | 12.9091 | 97.0% | 97.0% | 48.5% | 15.2% |
| saas | 48 | 35.4% | 35.4% | 5.2708 | 97.9% | 97.9% | 54.2% | 10.4% |
| travel | 37 | 51.3% | 51.3% | 5.6216 | 100.0% | 100.0% | 59.5% | 2.7% |
| workspace | 45 | 28.9% | 28.9% | 5.6222 | 100.0% | 100.0% | 46.7% | 11.1% |

