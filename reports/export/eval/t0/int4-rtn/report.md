# Baseline zero-shot FIM `int4-rtn`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 34499), estratificado por cut_kind x dominio, seed 20261020
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-ONNX-int4-rtn` · carga `onnx:cpu` · SHA-256 do modelo `4e3ec2d44a0423f9…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `573a4f438d630536…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: cpu · 213.8 s totais (213.8 s de geracao) · VRAM pico 0 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 38.8% | 38.8% | — |
| Normalized match (FIM acc.) | 38.8% | 38.8% | — |
| Prefix match | 39.6% | 39.6% | — |
| Syntax valid | 99.4% | 82.0% | 100.0% |
| Mongo valid (estatico) | 99.2% | 81.8% | 100.0% |
| Exec-valid | 56.2% | 47.6% | 62.4% |
| Campos inexistentes | 4.8% | 2.2% | 0.0% |
| Chars aceitos (media) | 8.05 | 8.05 | — |
| APT (tokens) | 2.616 | 2.616 | — |

Distribuicao do estagio de validacao (pred): {'ok': 281, 'api': 128, 'fields': 29, 'syntax': 61, 'type': 1}
Paradas: {'eos': 499, 'max': 1}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 80 | 43.8% | 43.8% | 4.8875 | 100.0% | 100.0% | 48.8% | 6.2% |
| member | 52 | 82.7% | 82.7% | 8.5577 | 100.0% | 100.0% | 59.6% | 0.0% |
| multiline | 43 | 0.0% | 0.0% | 14.5814 | 100.0% | 100.0% | 58.1% | 9.3% |
| operator | 66 | 47.0% | 47.0% | 6.6515 | 98.5% | 98.5% | 60.6% | 1.5% |
| partial | 57 | 66.7% | 66.7% | 4.9825 | 100.0% | 98.2% | 50.9% | 5.3% |
| rest | 64 | 25.0% | 25.0% | 10.0938 | 100.0% | 100.0% | 59.4% | 7.8% |
| stage | 60 | 3.3% | 3.3% | 13.2833 | 100.0% | 100.0% | 53.3% | 8.3% |
| stmt_start | 12 | 50.0% | 50.0% | 16.1667 | 100.0% | 100.0% | 66.7% | 0.0% |
| value | 66 | 34.8% | 34.8% | 3.0606 | 97.0% | 97.0% | 59.1% | 1.5% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 35 | 45.7% | 45.7% | 7.9143 | 100.0% | 100.0% | 57.1% | 0.0% |
| content | 46 | 37.0% | 37.0% | 8.1304 | 97.8% | 97.8% | 58.7% | 0.0% |
| energy | 41 | 41.5% | 41.5% | 7.2683 | 97.6% | 97.6% | 58.5% | 4.9% |
| finance | 42 | 26.2% | 26.2% | 7.7619 | 97.6% | 97.6% | 52.4% | 7.1% |
| gaming | 47 | 48.9% | 48.9% | 7.6596 | 100.0% | 100.0% | 57.5% | 6.4% |
| healthcare-synthetic | 43 | 23.3% | 23.3% | 8.5116 | 100.0% | 100.0% | 46.5% | 7.0% |
| hr | 42 | 47.6% | 47.6% | 7.7381 | 100.0% | 100.0% | 57.1% | 4.8% |
| iot | 41 | 34.2% | 34.2% | 9.878 | 100.0% | 100.0% | 58.5% | 7.3% |
| logistics | 33 | 39.4% | 39.4% | 14.0909 | 100.0% | 100.0% | 54.5% | 9.1% |
| saas | 48 | 39.6% | 39.6% | 5.9583 | 100.0% | 100.0% | 56.2% | 10.4% |
| travel | 37 | 51.3% | 51.3% | 6.2432 | 100.0% | 100.0% | 62.2% | 0.0% |
| workspace | 45 | 33.3% | 33.3% | 6.9333 | 100.0% | 97.8% | 55.6% | 0.0% |

