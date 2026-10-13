# Baseline zero-shot FIM `Qwen2.5-Coder-7B`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=1000 (de 2830), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-7B` · carga `nf4` · SHA-256 do modelo `74ac207b4deacd56…` (revisao `0396a76181e127dfc13e5c5ec48a8cee09938b02`)
- Dataset: `data\generated\v1\fim.jsonl` SHA-256 `6b4065ce37dff97c…` · amostra `9317e5b9c45c0006…`
- Contrato `fim-raw` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 952.0 s totais (926.7 s de geracao) · VRAM pico 5.46 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 16.9% | 16.7% | — |
| Normalized match (FIM acc.) | 16.9% | 16.7% | — |
| Prefix match | 23.9% | 23.8% | — |
| Syntax valid | 85.5% | 71.3% | 100.0% |
| Mongo valid (estatico) | 81.1% | 68.0% | 100.0% |
| Exec-valid | 50.4% | 44.3% | 89.6% |
| Campos inexistentes | 12.3% | 7.9% | 0.0% |
| Chars aceitos (media) | 4.193 | 4.193 | — |
| APT (tokens) | 1.496 | 1.496 | — |

Distribuicao do estagio de validacao (pred): {'ok': 514, 'api': 141, 'fields': 173, 'syntax': 152, 'type': 14, 'exec': 6}
Paradas: {'eos': 864, 'max': 119, 'suffix': 17}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 158 | 12.7% | 12.7% | 2.3038 | 85.4% | 84.8% | 44.9% | 25.9% |
| member | 105 | 24.8% | 24.8% | 3.0571 | 86.7% | 66.7% | 29.5% | 0.0% |
| multiline | 82 | 0.0% | 0.0% | 5.6463 | 82.9% | 81.7% | 40.2% | 28.1% |
| operator | 134 | 23.9% | 23.9% | 4.3358 | 89.5% | 88.1% | 62.7% | 9.0% |
| partial | 100 | 45.0% | 45.0% | 4.71 | 96.0% | 96.0% | 67.0% | 14.0% |
| rest | 125 | 16.8% | 16.8% | 5.984 | 81.6% | 72.0% | 48.0% | 8.0% |
| stage | 125 | 0.0% | 0.0% | 7.08 | 90.4% | 89.6% | 70.4% | 16.8% |
| stmt_start | 39 | 5.1% | 5.1% | 1.1282 | 82.0% | 71.8% | 10.3% | 0.0% |
| value | 132 | 17.4% | 17.4% | 2.3939 | 74.2% | 72.7% | 50.0% | 1.5% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 61 | 16.4% | 16.4% | 4.9508 | 93.4% | 86.9% | 37.7% | 18.0% |
| content | 82 | 23.2% | 23.2% | 4.7683 | 86.6% | 84.2% | 57.3% | 8.5% |
| energy | 94 | 18.1% | 18.1% | 4.6596 | 86.2% | 85.1% | 63.8% | 10.6% |
| finance | 78 | 20.5% | 20.5% | 4.7821 | 88.5% | 84.6% | 60.3% | 10.3% |
| gaming | 109 | 14.7% | 14.7% | 3.4495 | 80.7% | 77.1% | 45.9% | 14.7% |
| healthcare-synthetic | 87 | 9.2% | 9.2% | 4.5862 | 79.3% | 75.9% | 44.8% | 6.9% |
| hr | 93 | 17.2% | 17.2% | 4.2366 | 86.0% | 81.7% | 49.5% | 9.7% |
| iot | 81 | 16.1% | 16.1% | 3.3086 | 69.1% | 65.4% | 39.5% | 9.9% |
| logistics | 89 | 19.1% | 19.1% | 3.382 | 92.1% | 83.2% | 47.2% | 15.7% |
| saas | 80 | 15.0% | 15.0% | 4.9125 | 91.2% | 87.5% | 58.8% | 11.2% |
| travel | 78 | 16.7% | 16.7% | 3.2564 | 88.5% | 82.0% | 52.6% | 17.9% |
| workspace | 68 | 17.6% | 17.6% | 4.4706 | 88.2% | 82.3% | 44.1% | 16.2% |

