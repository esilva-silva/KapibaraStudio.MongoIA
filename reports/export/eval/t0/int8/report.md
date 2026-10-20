# Baseline zero-shot FIM `int8`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 34499), estratificado por cut_kind x dominio, seed 20261020
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-ONNX-int8` · carga `onnx:cpu` · SHA-256 do modelo `4e3ec2d44a0423f9…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `573a4f438d630536…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: cpu · 224.5 s totais (224.5 s de geracao) · VRAM pico 0 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 38.4% | 38.4% | — |
| Normalized match (FIM acc.) | 38.4% | 38.4% | — |
| Prefix match | 39.4% | 39.4% | — |
| Syntax valid | 99.2% | 83.0% | 100.0% |
| Mongo valid (estatico) | 99.2% | 83.0% | 100.0% |
| Exec-valid | 56.8% | 48.6% | 62.4% |
| Campos inexistentes | 3.8% | 2.2% | 0.0% |
| Chars aceitos (media) | 8.078 | 8.078 | — |
| APT (tokens) | 2.632 | 2.632 | — |

Distribuicao do estagio de validacao (pred): {'ok': 284, 'api': 129, 'fields': 25, 'syntax': 62}
Paradas: {'eos': 498, 'max': 2}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 80 | 45.0% | 45.0% | 5.0 | 98.8% | 98.8% | 47.5% | 7.5% |
| member | 52 | 75.0% | 75.0% | 7.9615 | 100.0% | 100.0% | 55.8% | 0.0% |
| multiline | 43 | 0.0% | 0.0% | 14.8837 | 100.0% | 100.0% | 60.5% | 7.0% |
| operator | 66 | 48.5% | 48.5% | 6.6667 | 100.0% | 100.0% | 60.6% | 1.5% |
| partial | 57 | 66.7% | 66.7% | 4.8596 | 100.0% | 100.0% | 50.9% | 7.0% |
| rest | 64 | 28.1% | 28.1% | 10.5312 | 100.0% | 100.0% | 64.1% | 3.1% |
| stage | 60 | 1.7% | 1.7% | 13.3333 | 98.3% | 98.3% | 56.7% | 3.3% |
| stmt_start | 12 | 50.0% | 50.0% | 16.1667 | 100.0% | 100.0% | 66.7% | 0.0% |
| value | 66 | 33.3% | 33.3% | 3.0303 | 97.0% | 97.0% | 59.1% | 1.5% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 35 | 40.0% | 40.0% | 7.6286 | 100.0% | 100.0% | 54.3% | 0.0% |
| content | 46 | 39.1% | 39.1% | 8.3478 | 100.0% | 100.0% | 56.5% | 0.0% |
| energy | 41 | 43.9% | 43.9% | 7.5854 | 95.1% | 95.1% | 61.0% | 2.4% |
| finance | 42 | 26.2% | 26.2% | 8.119 | 97.6% | 97.6% | 52.4% | 7.1% |
| gaming | 47 | 51.1% | 51.1% | 8.7021 | 100.0% | 100.0% | 55.3% | 6.4% |
| healthcare-synthetic | 43 | 18.6% | 18.6% | 8.3953 | 100.0% | 100.0% | 48.8% | 2.3% |
| hr | 42 | 50.0% | 50.0% | 7.8095 | 100.0% | 100.0% | 61.9% | 2.4% |
| iot | 41 | 34.2% | 34.2% | 9.878 | 100.0% | 100.0% | 58.5% | 7.3% |
| logistics | 33 | 36.4% | 36.4% | 13.6061 | 100.0% | 100.0% | 57.6% | 6.1% |
| saas | 48 | 39.6% | 39.6% | 5.75 | 97.9% | 97.9% | 58.3% | 8.3% |
| travel | 37 | 51.3% | 51.3% | 6.2162 | 100.0% | 100.0% | 62.2% | 0.0% |
| workspace | 45 | 31.1% | 31.1% | 6.1778 | 100.0% | 100.0% | 55.6% | 2.2% |

