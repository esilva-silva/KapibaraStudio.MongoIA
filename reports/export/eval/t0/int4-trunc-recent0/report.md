# Baseline zero-shot FIM `int4-trunc-recent0`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 34499), estratificado por cut_kind x dominio, seed 20261020
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-ONNX-int4` · carga `onnx:cpu` · SHA-256 do modelo `4e3ec2d44a0423f9…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `573a4f438d630536…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: cpu · 183.6 s totais (183.6 s de geracao) · VRAM pico 0 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 38.2% | 38.2% | — |
| Normalized match (FIM acc.) | 38.2% | 38.2% | — |
| Prefix match | 39.2% | 39.2% | — |
| Syntax valid | 98.6% | 81.8% | 100.0% |
| Mongo valid (estatico) | 98.6% | 81.8% | 100.0% |
| Exec-valid | 55.4% | 48.0% | 62.4% |
| Campos inexistentes | 4.4% | 2.2% | 0.0% |
| Chars aceitos (media) | 8.128 | 8.128 | — |
| APT (tokens) | 2.636 | 2.636 | — |

Distribuicao do estagio de validacao (pred): {'ok': 277, 'api': 126, 'fields': 30, 'syntax': 65, 'type': 2}
Paradas: {'eos': 497, 'max': 3}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 80 | 46.2% | 46.2% | 5.7 | 98.8% | 98.8% | 47.5% | 6.2% |
| member | 52 | 78.8% | 78.8% | 8.2885 | 100.0% | 100.0% | 55.8% | 0.0% |
| multiline | 43 | 0.0% | 0.0% | 14.907 | 100.0% | 100.0% | 58.1% | 9.3% |
| operator | 66 | 47.0% | 47.0% | 6.7727 | 97.0% | 97.0% | 60.6% | 1.5% |
| partial | 57 | 63.2% | 63.2% | 4.7895 | 100.0% | 100.0% | 47.4% | 7.0% |
| rest | 64 | 28.1% | 28.1% | 10.0938 | 98.4% | 98.4% | 60.9% | 4.7% |
| stage | 60 | 1.7% | 1.7% | 12.95 | 98.3% | 98.3% | 53.3% | 6.7% |
| stmt_start | 12 | 50.0% | 50.0% | 16.1667 | 100.0% | 100.0% | 66.7% | 0.0% |
| value | 66 | 31.8% | 31.8% | 3.0152 | 97.0% | 97.0% | 59.1% | 1.5% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 35 | 40.0% | 40.0% | 7.7429 | 100.0% | 100.0% | 54.3% | 0.0% |
| content | 46 | 37.0% | 37.0% | 8.1304 | 97.8% | 97.8% | 58.7% | 0.0% |
| energy | 41 | 46.3% | 46.3% | 7.7805 | 95.1% | 95.1% | 58.5% | 4.9% |
| finance | 42 | 26.2% | 26.2% | 8.6905 | 95.2% | 95.2% | 52.4% | 4.8% |
| gaming | 47 | 53.2% | 53.2% | 8.7234 | 100.0% | 100.0% | 55.3% | 6.4% |
| healthcare-synthetic | 43 | 20.9% | 20.9% | 8.8837 | 100.0% | 100.0% | 44.2% | 4.7% |
| hr | 42 | 45.2% | 45.2% | 7.6429 | 100.0% | 100.0% | 57.1% | 2.4% |
| iot | 41 | 34.2% | 34.2% | 10.0976 | 100.0% | 100.0% | 53.7% | 12.2% |
| logistics | 33 | 33.3% | 33.3% | 13.1212 | 97.0% | 97.0% | 57.6% | 6.1% |
| saas | 48 | 39.6% | 39.6% | 5.9167 | 97.9% | 97.9% | 58.3% | 8.3% |
| travel | 37 | 51.3% | 51.3% | 5.7568 | 100.0% | 100.0% | 62.2% | 0.0% |
| workspace | 45 | 31.1% | 31.1% | 6.1778 | 100.0% | 100.0% | 53.3% | 2.2% |

