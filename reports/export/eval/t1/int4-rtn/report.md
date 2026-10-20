# Baseline zero-shot FIM `int4-rtn`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 34499), estratificado por cut_kind x dominio, seed 20261020
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Small\exports\KapiCoder-Mongo-Small-ONNX-int4-rtn` · carga `onnx:cpu` · SHA-256 do modelo `f7f71e24b606e26d…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `573a4f438d630536…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: cpu · 589.7 s totais (589.7 s de geracao) · VRAM pico 0 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 39.4% | 39.4% | — |
| Normalized match (FIM acc.) | 39.4% | 39.4% | — |
| Prefix match | 40.4% | 40.4% | — |
| Syntax valid | 99.0% | 82.0% | 100.0% |
| Mongo valid (estatico) | 99.0% | 82.0% | 100.0% |
| Exec-valid | 56.8% | 48.4% | 62.4% |
| Campos inexistentes | 3.8% | 1.6% | 0.0% |
| Chars aceitos (media) | 8.098 | 8.098 | — |
| APT (tokens) | 2.592 | 2.592 | — |

Distribuicao do estagio de validacao (pred): {'ok': 284, 'api': 130, 'fields': 23, 'syntax': 62, 'type': 1}
Paradas: {'eos': 497, 'max': 3}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 80 | 46.2% | 46.2% | 5.05 | 100.0% | 100.0% | 50.0% | 3.8% |
| member | 52 | 82.7% | 82.7% | 8.75 | 100.0% | 100.0% | 59.6% | 0.0% |
| multiline | 43 | 0.0% | 0.0% | 13.9535 | 97.7% | 97.7% | 55.8% | 11.6% |
| operator | 66 | 48.5% | 48.5% | 6.7727 | 100.0% | 100.0% | 60.6% | 1.5% |
| partial | 57 | 66.7% | 66.7% | 4.9474 | 100.0% | 100.0% | 50.9% | 5.3% |
| rest | 64 | 28.1% | 28.1% | 9.2969 | 98.4% | 98.4% | 62.5% | 4.7% |
| stage | 60 | 1.7% | 1.7% | 14.3833 | 98.3% | 98.3% | 53.3% | 6.7% |
| stmt_start | 12 | 50.0% | 50.0% | 16.1667 | 100.0% | 100.0% | 66.7% | 0.0% |
| value | 66 | 33.3% | 33.3% | 3.1667 | 97.0% | 97.0% | 60.6% | 0.0% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 35 | 42.9% | 42.9% | 9.0857 | 97.1% | 97.1% | 57.1% | 0.0% |
| content | 46 | 41.3% | 41.3% | 8.2174 | 100.0% | 100.0% | 54.4% | 4.3% |
| energy | 41 | 43.9% | 43.9% | 8.6098 | 97.6% | 97.6% | 58.5% | 4.9% |
| finance | 42 | 26.2% | 26.2% | 5.7381 | 97.6% | 97.6% | 54.8% | 4.8% |
| gaming | 47 | 53.2% | 53.2% | 9.3191 | 100.0% | 100.0% | 57.5% | 4.3% |
| healthcare-synthetic | 43 | 25.6% | 25.6% | 8.6047 | 100.0% | 100.0% | 48.8% | 4.7% |
| hr | 42 | 50.0% | 50.0% | 7.6667 | 97.6% | 97.6% | 59.5% | 0.0% |
| iot | 41 | 34.2% | 34.2% | 9.878 | 100.0% | 100.0% | 58.5% | 7.3% |
| logistics | 33 | 33.3% | 33.3% | 13.1212 | 97.0% | 97.0% | 54.5% | 9.1% |
| saas | 48 | 37.5% | 37.5% | 5.5 | 100.0% | 100.0% | 58.3% | 6.2% |
| travel | 37 | 51.3% | 51.3% | 5.7568 | 100.0% | 100.0% | 62.2% | 0.0% |
| workspace | 45 | 33.3% | 33.3% | 6.9778 | 100.0% | 100.0% | 57.8% | 0.0% |

