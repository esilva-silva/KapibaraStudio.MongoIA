# Baseline zero-shot FIM `int8-emb4`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 34499), estratificado por cut_kind x dominio, seed 20261020
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-v3-ONNX-int8-emb4` · carga `onnx:cpu` · SHA-256 do modelo `e30c625c12e24dfd…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `573a4f438d630536…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: cpu · 240.9 s totais (240.9 s de geracao) · VRAM pico 0 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 39.4% | 39.4% | — |
| Normalized match (FIM acc.) | 39.4% | 39.4% | — |
| Prefix match | 40.0% | 40.0% | — |
| Syntax valid | 99.2% | 83.6% | 100.0% |
| Mongo valid (estatico) | 99.2% | 83.6% | 100.0% |
| Exec-valid | 57.6% | 49.4% | 62.4% |
| Campos inexistentes | 3.2% | 1.6% | 0.0% |
| Chars aceitos (media) | 8.038 | 8.038 | — |
| APT (tokens) | 2.562 | 2.562 | — |

Distribuicao do estagio de validacao (pred): {'ok': 288, 'api': 128, 'fields': 21, 'syntax': 62, 'type': 1}
Paradas: {'eos': 498, 'max': 2}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 80 | 48.8% | 48.8% | 5.2375 | 98.8% | 98.8% | 51.2% | 2.5% |
| member | 52 | 78.8% | 78.8% | 8.2308 | 100.0% | 100.0% | 57.7% | 0.0% |
| multiline | 43 | 0.0% | 0.0% | 15.6279 | 97.7% | 97.7% | 58.1% | 7.0% |
| operator | 66 | 51.5% | 51.5% | 6.3788 | 100.0% | 100.0% | 62.1% | 3.0% |
| partial | 57 | 64.9% | 64.9% | 4.7368 | 100.0% | 100.0% | 50.9% | 7.0% |
| rest | 64 | 25.0% | 25.0% | 8.7344 | 100.0% | 100.0% | 62.5% | 4.7% |
| stage | 60 | 1.7% | 1.7% | 14.15 | 98.3% | 98.3% | 56.7% | 3.3% |
| stmt_start | 12 | 50.0% | 50.0% | 16.1667 | 100.0% | 100.0% | 66.7% | 0.0% |
| value | 66 | 34.8% | 34.8% | 3.1364 | 98.5% | 98.5% | 60.6% | 0.0% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 35 | 40.0% | 40.0% | 10.4571 | 100.0% | 100.0% | 54.3% | 0.0% |
| content | 46 | 43.5% | 43.5% | 8.4565 | 100.0% | 100.0% | 56.5% | 0.0% |
| energy | 41 | 43.9% | 43.9% | 7.6341 | 97.6% | 97.6% | 61.0% | 2.4% |
| finance | 42 | 23.8% | 23.8% | 5.9286 | 97.6% | 97.6% | 52.4% | 9.5% |
| gaming | 47 | 51.1% | 51.1% | 8.6596 | 97.9% | 97.9% | 61.7% | 0.0% |
| healthcare-synthetic | 43 | 25.6% | 25.6% | 8.6977 | 100.0% | 100.0% | 51.2% | 2.3% |
| hr | 42 | 54.8% | 54.8% | 7.619 | 97.6% | 97.6% | 57.1% | 2.4% |
| iot | 41 | 34.2% | 34.2% | 9.6585 | 100.0% | 100.0% | 56.1% | 9.8% |
| logistics | 33 | 36.4% | 36.4% | 14.4242 | 100.0% | 100.0% | 57.6% | 6.1% |
| saas | 48 | 39.6% | 39.6% | 5.3542 | 100.0% | 100.0% | 60.4% | 6.2% |
| travel | 37 | 48.6% | 48.6% | 5.4865 | 100.0% | 100.0% | 62.2% | 0.0% |
| workspace | 45 | 31.1% | 31.1% | 5.9778 | 100.0% | 100.0% | 60.0% | 0.0% |

