# Baseline zero-shot FIM `int4-trunc-fields0`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 34499), estratificado por cut_kind x dominio, seed 20261020
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-ONNX-int4` · carga `onnx:cpu` · SHA-256 do modelo `4e3ec2d44a0423f9…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `573a4f438d630536…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: cpu · 218.3 s totais (218.3 s de geracao) · VRAM pico 0 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 38.2% | 38.2% | — |
| Normalized match (FIM acc.) | 38.2% | 38.2% | — |
| Prefix match | 39.2% | 39.2% | — |
| Syntax valid | 99.0% | 81.8% | 100.0% |
| Mongo valid (estatico) | 99.0% | 81.8% | 100.0% |
| Exec-valid | 54.8% | 47.8% | 62.4% |
| Campos inexistentes | 5.8% | 2.6% | 0.0% |
| Chars aceitos (media) | 8.018 | 8.018 | — |
| APT (tokens) | 2.614 | 2.614 | — |

Distribuicao do estagio de validacao (pred): {'ok': 274, 'api': 127, 'fields': 36, 'syntax': 63}
Paradas: {'eos': 499, 'max': 1}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 80 | 46.2% | 46.2% | 5.65 | 98.8% | 98.8% | 47.5% | 7.5% |
| member | 52 | 80.8% | 80.8% | 8.4038 | 100.0% | 100.0% | 55.8% | 1.9% |
| multiline | 43 | 0.0% | 0.0% | 14.6279 | 100.0% | 100.0% | 51.2% | 18.6% |
| operator | 66 | 43.9% | 43.9% | 6.4242 | 97.0% | 97.0% | 60.6% | 1.5% |
| partial | 57 | 61.4% | 61.4% | 4.386 | 100.0% | 100.0% | 47.4% | 7.0% |
| rest | 64 | 28.1% | 28.1% | 10.2812 | 100.0% | 100.0% | 62.5% | 4.7% |
| stage | 60 | 1.7% | 1.7% | 12.7667 | 98.3% | 98.3% | 51.7% | 8.3% |
| stmt_start | 12 | 50.0% | 50.0% | 15.6667 | 100.0% | 100.0% | 66.7% | 0.0% |
| value | 66 | 34.8% | 34.8% | 3.1061 | 98.5% | 98.5% | 59.1% | 1.5% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 35 | 40.0% | 40.0% | 7.1714 | 100.0% | 100.0% | 51.4% | 2.9% |
| content | 46 | 32.6% | 32.6% | 7.6087 | 97.8% | 97.8% | 58.7% | 0.0% |
| energy | 41 | 41.5% | 41.5% | 7.0 | 97.6% | 97.6% | 61.0% | 2.4% |
| finance | 42 | 26.2% | 26.2% | 8.1905 | 97.6% | 97.6% | 54.8% | 4.8% |
| gaming | 47 | 53.2% | 53.2% | 9.617 | 100.0% | 100.0% | 55.3% | 6.4% |
| healthcare-synthetic | 43 | 23.3% | 23.3% | 9.0233 | 100.0% | 100.0% | 41.9% | 9.3% |
| hr | 42 | 47.6% | 47.6% | 8.1667 | 100.0% | 100.0% | 54.8% | 7.1% |
| iot | 41 | 41.5% | 41.5% | 10.0732 | 100.0% | 100.0% | 56.1% | 9.8% |
| logistics | 33 | 33.3% | 33.3% | 12.6667 | 97.0% | 97.0% | 54.5% | 9.1% |
| saas | 48 | 39.6% | 39.6% | 5.7917 | 97.9% | 97.9% | 56.2% | 10.4% |
| travel | 37 | 48.6% | 48.6% | 5.4595 | 100.0% | 100.0% | 59.5% | 2.7% |
| workspace | 45 | 31.1% | 31.1% | 6.2889 | 100.0% | 100.0% | 53.3% | 4.4% |

