# Baseline zero-shot FIM `dml-int4`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 34499), estratificado por cut_kind x dominio, seed 20261020
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-ONNX-dml-int4` · carga `onnx:dml` · SHA-256 do modelo `4e3ec2d44a0423f9…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `573a4f438d630536…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: dml · 139.5 s totais (139.5 s de geracao) · VRAM pico 0 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 38.2% | 38.2% | — |
| Normalized match (FIM acc.) | 38.2% | 38.2% | — |
| Prefix match | 39.0% | 39.0% | — |
| Syntax valid | 99.0% | 81.6% | 100.0% |
| Mongo valid (estatico) | 99.0% | 81.6% | 100.0% |
| Exec-valid | 56.0% | 48.0% | 62.4% |
| Campos inexistentes | 4.6% | 2.2% | 0.0% |
| Chars aceitos (media) | 7.988 | 7.988 | — |
| APT (tokens) | 2.604 | 2.604 | — |

Distribuicao do estagio de validacao (pred): {'ok': 280, 'api': 126, 'fields': 30, 'syntax': 63, 'type': 1}
Paradas: {'eos': 498, 'max': 2}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 80 | 43.8% | 43.8% | 5.3 | 97.5% | 97.5% | 48.8% | 5.0% |
| member | 52 | 78.8% | 78.8% | 8.3077 | 100.0% | 100.0% | 57.7% | 0.0% |
| multiline | 43 | 0.0% | 0.0% | 13.814 | 100.0% | 100.0% | 55.8% | 11.6% |
| operator | 66 | 48.5% | 48.5% | 6.6667 | 100.0% | 100.0% | 60.6% | 1.5% |
| partial | 57 | 63.2% | 63.2% | 4.6316 | 100.0% | 100.0% | 49.1% | 7.0% |
| rest | 64 | 28.1% | 28.1% | 10.5938 | 100.0% | 100.0% | 62.5% | 4.7% |
| stage | 60 | 1.7% | 1.7% | 12.8167 | 98.3% | 98.3% | 53.3% | 8.3% |
| stmt_start | 12 | 50.0% | 50.0% | 16.1667 | 100.0% | 100.0% | 66.7% | 0.0% |
| value | 66 | 33.3% | 33.3% | 3.0152 | 97.0% | 97.0% | 59.1% | 1.5% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 35 | 40.0% | 40.0% | 7.6 | 100.0% | 100.0% | 54.3% | 0.0% |
| content | 46 | 41.3% | 41.3% | 8.413 | 100.0% | 100.0% | 56.5% | 2.2% |
| energy | 41 | 41.5% | 41.5% | 7.3171 | 92.7% | 92.7% | 58.5% | 2.4% |
| finance | 42 | 26.2% | 26.2% | 8.381 | 97.6% | 97.6% | 52.4% | 7.1% |
| gaming | 47 | 48.9% | 48.9% | 7.6596 | 100.0% | 100.0% | 57.5% | 6.4% |
| healthcare-synthetic | 43 | 23.3% | 23.3% | 8.7674 | 100.0% | 100.0% | 46.5% | 4.7% |
| hr | 42 | 47.6% | 47.6% | 7.7143 | 100.0% | 100.0% | 59.5% | 4.8% |
| iot | 41 | 31.7% | 31.7% | 9.6341 | 100.0% | 100.0% | 56.1% | 9.8% |
| logistics | 33 | 36.4% | 36.4% | 13.6364 | 100.0% | 100.0% | 54.5% | 9.1% |
| saas | 48 | 41.7% | 41.7% | 5.9375 | 97.9% | 97.9% | 58.3% | 8.3% |
| travel | 37 | 48.6% | 48.6% | 5.9459 | 100.0% | 100.0% | 62.2% | 0.0% |
| workspace | 45 | 31.1% | 31.1% | 6.1778 | 100.0% | 100.0% | 55.6% | 0.0% |

