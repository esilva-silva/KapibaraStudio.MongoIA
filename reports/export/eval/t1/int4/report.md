# Baseline zero-shot FIM `int4`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 34499), estratificado por cut_kind x dominio, seed 20261020
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Small\exports\KapiCoder-Mongo-Small-ONNX-int4` · carga `onnx:cpu` · SHA-256 do modelo `f7f71e24b606e26d…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `573a4f438d630536…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: cpu · 586.4 s totais (586.4 s de geracao) · VRAM pico 0 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 39.8% | 39.8% | — |
| Normalized match (FIM acc.) | 39.8% | 39.8% | — |
| Prefix match | 40.8% | 40.8% | — |
| Syntax valid | 99.4% | 82.4% | 100.0% |
| Mongo valid (estatico) | 99.4% | 82.4% | 100.0% |
| Exec-valid | 58.0% | 49.2% | 62.4% |
| Campos inexistentes | 3.2% | 1.6% | 0.0% |
| Chars aceitos (media) | 8.33 | 8.33 | — |
| APT (tokens) | 2.658 | 2.658 | — |

Distribuicao do estagio de validacao (pred): {'ok': 290, 'api': 130, 'fields': 19, 'syntax': 61}
Paradas: {'eos': 498, 'max': 2}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 80 | 50.0% | 50.0% | 5.425 | 100.0% | 100.0% | 51.2% | 3.8% |
| member | 52 | 82.7% | 82.7% | 8.75 | 100.0% | 100.0% | 59.6% | 0.0% |
| multiline | 43 | 0.0% | 0.0% | 14.6744 | 100.0% | 100.0% | 62.8% | 7.0% |
| operator | 66 | 47.0% | 47.0% | 6.7121 | 100.0% | 100.0% | 62.1% | 1.5% |
| partial | 57 | 66.7% | 66.7% | 4.8421 | 98.2% | 98.2% | 50.9% | 5.3% |
| rest | 64 | 29.7% | 29.7% | 10.3594 | 100.0% | 100.0% | 62.5% | 4.7% |
| stage | 60 | 1.7% | 1.7% | 14.3833 | 100.0% | 100.0% | 55.0% | 5.0% |
| stmt_start | 12 | 50.0% | 50.0% | 15.6667 | 100.0% | 100.0% | 66.7% | 0.0% |
| value | 66 | 31.8% | 31.8% | 3.2121 | 97.0% | 97.0% | 60.6% | 0.0% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 35 | 42.9% | 42.9% | 8.7714 | 100.0% | 100.0% | 57.1% | 0.0% |
| content | 46 | 45.6% | 45.6% | 8.5435 | 100.0% | 100.0% | 56.5% | 2.2% |
| energy | 41 | 43.9% | 43.9% | 9.0732 | 97.6% | 97.6% | 58.5% | 4.9% |
| finance | 42 | 26.2% | 26.2% | 7.119 | 97.6% | 97.6% | 52.4% | 7.1% |
| gaming | 47 | 53.2% | 53.2% | 9.2766 | 100.0% | 100.0% | 59.6% | 2.1% |
| healthcare-synthetic | 43 | 25.6% | 25.6% | 9.7209 | 100.0% | 100.0% | 53.5% | 0.0% |
| hr | 42 | 50.0% | 50.0% | 7.381 | 100.0% | 100.0% | 61.9% | 2.4% |
| iot | 41 | 34.2% | 34.2% | 9.6829 | 100.0% | 100.0% | 58.5% | 7.3% |
| logistics | 33 | 33.3% | 33.3% | 13.4242 | 100.0% | 100.0% | 54.5% | 9.1% |
| saas | 48 | 37.5% | 37.5% | 6.0833 | 100.0% | 100.0% | 60.4% | 4.2% |
| travel | 37 | 51.3% | 51.3% | 5.7297 | 100.0% | 100.0% | 62.2% | 0.0% |
| workspace | 45 | 33.3% | 33.3% | 6.3556 | 97.8% | 97.8% | 60.0% | 0.0% |

