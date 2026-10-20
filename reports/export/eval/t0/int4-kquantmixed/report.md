# Baseline zero-shot FIM `int4-kquantmixed`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 34499), estratificado por cut_kind x dominio, seed 20261020
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-ONNX-int4-kquantmixed` · carga `onnx:cpu` · SHA-256 do modelo `4e3ec2d44a0423f9…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `573a4f438d630536…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: cpu · 220.3 s totais (220.3 s de geracao) · VRAM pico 0 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 39.2% | 39.2% | — |
| Normalized match (FIM acc.) | 39.2% | 39.2% | — |
| Prefix match | 40.2% | 40.2% | — |
| Syntax valid | 99.0% | 82.6% | 100.0% |
| Mongo valid (estatico) | 99.0% | 82.6% | 100.0% |
| Exec-valid | 56.8% | 48.8% | 62.4% |
| Campos inexistentes | 3.2% | 1.6% | 0.0% |
| Chars aceitos (media) | 8.264 | 8.264 | — |
| APT (tokens) | 2.692 | 2.692 | — |

Distribuicao do estagio de validacao (pred): {'ok': 284, 'api': 129, 'fields': 23, 'syntax': 63, 'type': 1}
Paradas: {'eos': 497, 'max': 3}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 80 | 46.2% | 46.2% | 5.7375 | 100.0% | 100.0% | 50.0% | 5.0% |
| member | 52 | 80.8% | 80.8% | 8.5 | 100.0% | 100.0% | 55.8% | 0.0% |
| multiline | 43 | 0.0% | 0.0% | 14.7674 | 100.0% | 100.0% | 55.8% | 9.3% |
| operator | 66 | 48.5% | 48.5% | 6.7879 | 98.5% | 98.5% | 60.6% | 1.5% |
| partial | 57 | 64.9% | 64.9% | 4.8246 | 98.2% | 98.2% | 49.1% | 5.3% |
| rest | 64 | 29.7% | 29.7% | 10.8594 | 100.0% | 100.0% | 65.6% | 1.6% |
| stage | 60 | 1.7% | 1.7% | 13.0667 | 98.3% | 98.3% | 56.7% | 3.3% |
| stmt_start | 12 | 50.0% | 50.0% | 16.1667 | 100.0% | 100.0% | 66.7% | 0.0% |
| value | 66 | 33.3% | 33.3% | 3.0303 | 97.0% | 97.0% | 59.1% | 1.5% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 35 | 40.0% | 40.0% | 7.6286 | 100.0% | 100.0% | 54.3% | 0.0% |
| content | 46 | 37.0% | 37.0% | 8.3696 | 97.8% | 97.8% | 56.5% | 0.0% |
| energy | 41 | 46.3% | 46.3% | 7.7805 | 95.1% | 95.1% | 61.0% | 2.4% |
| finance | 42 | 26.2% | 26.2% | 9.1905 | 95.2% | 95.2% | 54.8% | 2.4% |
| gaming | 47 | 53.2% | 53.2% | 8.9787 | 100.0% | 100.0% | 57.5% | 4.3% |
| healthcare-synthetic | 43 | 20.9% | 20.9% | 8.0 | 100.0% | 100.0% | 46.5% | 2.3% |
| hr | 42 | 50.0% | 50.0% | 8.0714 | 100.0% | 100.0% | 61.9% | 2.4% |
| iot | 41 | 34.2% | 34.2% | 10.1951 | 100.0% | 100.0% | 58.5% | 7.3% |
| logistics | 33 | 36.4% | 36.4% | 13.1212 | 100.0% | 100.0% | 57.6% | 6.1% |
| saas | 48 | 41.7% | 41.7% | 6.25 | 100.0% | 100.0% | 58.3% | 8.3% |
| travel | 37 | 51.3% | 51.3% | 6.2162 | 100.0% | 100.0% | 62.2% | 0.0% |
| workspace | 45 | 33.3% | 33.3% | 6.4222 | 100.0% | 100.0% | 53.3% | 2.2% |

