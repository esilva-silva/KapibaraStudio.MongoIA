# Baseline zero-shot FIM `dml-int4`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 34499), estratificado por cut_kind x dominio, seed 20261020
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Small\exports\KapiCoder-Mongo-Small-ONNX-dml-int4` · carga `onnx:dml` · SHA-256 do modelo `f7f71e24b606e26d…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `573a4f438d630536…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: dml · 166.5 s totais (166.5 s de geracao) · VRAM pico 0 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 39.4% | 39.4% | — |
| Normalized match (FIM acc.) | 39.4% | 39.4% | — |
| Prefix match | 40.6% | 40.6% | — |
| Syntax valid | 99.2% | 82.6% | 100.0% |
| Mongo valid (estatico) | 99.2% | 82.6% | 100.0% |
| Exec-valid | 57.4% | 49.4% | 62.4% |
| Campos inexistentes | 3.4% | 1.4% | 0.0% |
| Chars aceitos (media) | 7.944 | 7.944 | — |
| APT (tokens) | 2.528 | 2.528 | — |

Distribuicao do estagio de validacao (pred): {'ok': 287, 'api': 130, 'fields': 21, 'syntax': 61, 'type': 1}
Paradas: {'eos': 498, 'max': 2}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 80 | 47.5% | 47.5% | 5.2125 | 100.0% | 100.0% | 51.2% | 3.8% |
| member | 52 | 82.7% | 82.7% | 8.75 | 100.0% | 100.0% | 59.6% | 0.0% |
| multiline | 43 | 0.0% | 0.0% | 14.5814 | 100.0% | 100.0% | 55.8% | 11.6% |
| operator | 66 | 48.5% | 48.5% | 6.5303 | 100.0% | 100.0% | 62.1% | 0.0% |
| partial | 57 | 66.7% | 66.7% | 4.807 | 100.0% | 100.0% | 50.9% | 5.3% |
| rest | 64 | 28.1% | 28.1% | 9.2656 | 100.0% | 100.0% | 62.5% | 4.7% |
| stage | 60 | 1.7% | 1.7% | 13.15 | 98.3% | 98.3% | 55.0% | 5.0% |
| stmt_start | 12 | 41.7% | 41.7% | 15.6667 | 100.0% | 100.0% | 66.7% | 0.0% |
| value | 66 | 33.3% | 33.3% | 3.0 | 95.5% | 95.5% | 60.6% | 0.0% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 35 | 42.9% | 42.9% | 8.7714 | 100.0% | 100.0% | 57.1% | 0.0% |
| content | 46 | 45.6% | 45.6% | 8.1739 | 100.0% | 100.0% | 56.5% | 2.2% |
| energy | 41 | 46.3% | 46.3% | 7.4878 | 95.1% | 95.1% | 58.5% | 4.9% |
| finance | 42 | 28.6% | 28.6% | 6.0238 | 97.6% | 97.6% | 54.8% | 4.8% |
| gaming | 47 | 53.2% | 53.2% | 9.2766 | 100.0% | 100.0% | 57.5% | 4.3% |
| healthcare-synthetic | 43 | 23.3% | 23.3% | 8.4651 | 100.0% | 100.0% | 51.2% | 2.3% |
| hr | 42 | 45.2% | 45.2% | 6.8571 | 100.0% | 100.0% | 61.9% | 2.4% |
| iot | 41 | 31.7% | 31.7% | 9.6829 | 100.0% | 100.0% | 58.5% | 7.3% |
| logistics | 33 | 33.3% | 33.3% | 13.0303 | 97.0% | 97.0% | 54.5% | 9.1% |
| saas | 48 | 37.5% | 37.5% | 6.0417 | 100.0% | 100.0% | 60.4% | 4.2% |
| travel | 37 | 51.3% | 51.3% | 5.7297 | 100.0% | 100.0% | 62.2% | 0.0% |
| workspace | 45 | 33.3% | 33.3% | 6.9333 | 100.0% | 100.0% | 55.6% | 0.0% |

