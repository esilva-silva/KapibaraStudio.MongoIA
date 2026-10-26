# Baseline zero-shot FIM `int4`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 34499), estratificado por cut_kind x dominio, seed 20261020
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-v3-ONNX-int4` · carga `onnx:cpu` · SHA-256 do modelo `e30c625c12e24dfd…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `573a4f438d630536…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: cpu · 207.9 s totais (207.9 s de geracao) · VRAM pico 0 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 39.0% | 39.0% | — |
| Normalized match (FIM acc.) | 39.0% | 39.0% | — |
| Prefix match | 39.8% | 39.8% | — |
| Syntax valid | 99.2% | 83.8% | 100.0% |
| Mongo valid (estatico) | 99.2% | 83.8% | 100.0% |
| Exec-valid | 57.8% | 49.6% | 62.4% |
| Campos inexistentes | 3.0% | 1.4% | 0.0% |
| Chars aceitos (media) | 8.242 | 8.242 | — |
| APT (tokens) | 2.66 | 2.66 | — |

Distribuicao do estagio de validacao (pred): {'ok': 289, 'api': 128, 'fields': 20, 'syntax': 62, 'type': 1}
Paradas: {'eos': 497, 'max': 3}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 80 | 46.2% | 46.2% | 5.175 | 100.0% | 100.0% | 51.2% | 2.5% |
| member | 52 | 78.8% | 78.8% | 8.1154 | 100.0% | 100.0% | 59.6% | 0.0% |
| multiline | 43 | 0.0% | 0.0% | 15.4419 | 97.7% | 97.7% | 55.8% | 9.3% |
| operator | 66 | 51.5% | 51.5% | 6.4091 | 100.0% | 100.0% | 62.1% | 3.0% |
| partial | 57 | 64.9% | 64.9% | 4.7368 | 100.0% | 100.0% | 50.9% | 7.0% |
| rest | 64 | 25.0% | 25.0% | 9.3281 | 100.0% | 100.0% | 64.1% | 3.1% |
| stage | 60 | 1.7% | 1.7% | 15.35 | 96.7% | 96.7% | 56.7% | 1.7% |
| stmt_start | 12 | 41.7% | 41.7% | 15.6667 | 100.0% | 100.0% | 66.7% | 0.0% |
| value | 66 | 36.4% | 36.4% | 3.3636 | 98.5% | 98.5% | 60.6% | 0.0% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 35 | 40.0% | 40.0% | 9.5714 | 100.0% | 100.0% | 54.3% | 0.0% |
| content | 46 | 43.5% | 43.5% | 8.1739 | 100.0% | 100.0% | 56.5% | 0.0% |
| energy | 41 | 43.9% | 43.9% | 8.6829 | 97.6% | 97.6% | 61.0% | 2.4% |
| finance | 42 | 26.2% | 26.2% | 6.2857 | 97.6% | 97.6% | 52.4% | 9.5% |
| gaming | 47 | 51.1% | 51.1% | 7.9149 | 97.9% | 97.9% | 61.7% | 0.0% |
| healthcare-synthetic | 43 | 25.6% | 25.6% | 9.9767 | 100.0% | 100.0% | 51.2% | 2.3% |
| hr | 42 | 52.4% | 52.4% | 7.7857 | 97.6% | 97.6% | 59.5% | 0.0% |
| iot | 41 | 29.3% | 29.3% | 9.5854 | 100.0% | 100.0% | 58.5% | 7.3% |
| logistics | 33 | 36.4% | 36.4% | 14.697 | 100.0% | 100.0% | 54.5% | 9.1% |
| saas | 48 | 39.6% | 39.6% | 6.1458 | 100.0% | 100.0% | 60.4% | 6.2% |
| travel | 37 | 48.6% | 48.6% | 5.9459 | 100.0% | 100.0% | 62.2% | 0.0% |
| workspace | 45 | 31.1% | 31.1% | 5.9778 | 100.0% | 100.0% | 60.0% | 0.0% |

