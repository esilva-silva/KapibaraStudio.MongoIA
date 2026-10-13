# Baseline zero-shot FIM `Qwen2.5-Coder-1.5B`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=1000 (de 2830), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-1.5B` · carga `bf16` · SHA-256 do modelo `8bcb242c05f27140…` (revisao `df3ce67c0e24480f20468b6ef2894622d69eb73b`)
- Dataset: `data\generated\v1\fim.jsonl` SHA-256 `6b4065ce37dff97c…` · amostra `9317e5b9c45c0006…`
- Contrato `fim-raw` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 740.9 s totais (730.8 s de geracao) · VRAM pico 3.01 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 13.0% | 12.4% | — |
| Normalized match (FIM acc.) | 13.0% | 12.4% | — |
| Prefix match | 21.2% | 21.2% | — |
| Syntax valid | 76.5% | 60.5% | 100.0% |
| Mongo valid (estatico) | 68.9% | 55.7% | 100.0% |
| Exec-valid | 40.2% | 35.5% | 89.6% |
| Campos inexistentes | 13.7% | 7.9% | 0.0% |
| Chars aceitos (media) | 3.952 | 3.952 | — |
| APT (tokens) | 1.454 | 1.454 | — |

Distribuicao do estagio de validacao (pred): {'ok': 416, 'syntax': 250, 'fields': 159, 'api': 156, 'exec': 7, 'type': 12}
Paradas: {'eos': 796, 'max': 167, 'suffix': 37}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 158 | 10.1% | 10.1% | 1.8418 | 77.2% | 74.1% | 42.4% | 20.2% |
| member | 105 | 22.9% | 22.9% | 2.581 | 82.9% | 46.7% | 31.4% | 2.9% |
| multiline | 82 | 0.0% | 0.0% | 5.3537 | 75.6% | 75.6% | 29.3% | 29.3% |
| operator | 134 | 20.9% | 20.9% | 4.4104 | 82.1% | 80.6% | 50.0% | 10.4% |
| partial | 100 | 29.0% | 29.0% | 3.69 | 79.0% | 72.0% | 43.0% | 11.0% |
| rest | 125 | 9.6% | 9.6% | 6.48 | 74.4% | 63.2% | 29.6% | 16.0% |
| stage | 125 | 0.0% | 0.0% | 7.192 | 85.6% | 85.6% | 62.4% | 20.0% |
| stmt_start | 39 | 0.0% | 0.0% | 0.0256 | 41.0% | 28.2% | 2.6% | 0.0% |
| value | 132 | 15.9% | 15.9% | 2.1288 | 67.4% | 63.6% | 39.4% | 6.1% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 61 | 16.4% | 16.4% | 4.3443 | 80.3% | 70.5% | 32.8% | 18.0% |
| content | 82 | 15.8% | 15.8% | 4.3537 | 80.5% | 78.0% | 50.0% | 11.0% |
| energy | 94 | 9.6% | 9.6% | 4.2021 | 70.2% | 64.9% | 41.5% | 14.9% |
| finance | 78 | 16.7% | 16.7% | 4.0513 | 71.8% | 66.7% | 38.5% | 15.4% |
| gaming | 109 | 11.0% | 11.0% | 4.1376 | 80.7% | 72.5% | 45.0% | 15.6% |
| healthcare-synthetic | 87 | 6.9% | 6.9% | 4.2644 | 77.0% | 67.8% | 34.5% | 16.1% |
| hr | 93 | 12.9% | 12.9% | 3.5054 | 68.8% | 59.1% | 37.6% | 5.4% |
| iot | 81 | 11.1% | 11.1% | 3.7284 | 65.4% | 59.3% | 30.9% | 9.9% |
| logistics | 89 | 13.5% | 13.5% | 3.1573 | 82.0% | 69.7% | 38.2% | 12.4% |
| saas | 80 | 15.0% | 15.0% | 4.7625 | 85.0% | 80.0% | 50.0% | 13.8% |
| travel | 78 | 15.4% | 15.4% | 2.8333 | 78.2% | 69.2% | 48.7% | 15.4% |
| workspace | 68 | 14.7% | 14.7% | 4.2059 | 79.4% | 70.6% | 30.9% | 19.1% |

