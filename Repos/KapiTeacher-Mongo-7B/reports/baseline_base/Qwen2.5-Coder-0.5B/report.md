# Baseline zero-shot FIM `Qwen2.5-Coder-0.5B`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=1000 (de 2830), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-0.5B` · carga `bf16` · SHA-256 do modelo `96217dc3a23c0290…` (revisao `8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301`)
- Dataset: `data\generated\v1\fim.jsonl` SHA-256 `6b4065ce37dff97c…` · amostra `9317e5b9c45c0006…`
- Contrato `fim-raw` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 691.5 s totais (683.9 s de geracao) · VRAM pico 1.04 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 9.7% | 9.0% | — |
| Normalized match (FIM acc.) | 9.7% | 9.0% | — |
| Prefix match | 17.7% | 17.7% | — |
| Syntax valid | 66.1% | 51.4% | 100.0% |
| Mongo valid (estatico) | 57.5% | 45.6% | 100.0% |
| Exec-valid | 31.4% | 25.8% | 89.6% |
| Campos inexistentes | 10.9% | 7.1% | 0.0% |
| Chars aceitos (media) | 3.546 | 3.546 | — |
| APT (tokens) | 1.318 | 1.318 | — |

Distribuicao do estagio de validacao (pred): {'ok': 325, 'syntax': 351, 'api': 150, 'fields': 136, 'type': 20, 'exec': 18}
Paradas: {'eos': 707, 'max': 259, 'suffix': 34}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 158 | 5.1% | 5.1% | 1.8354 | 65.2% | 60.1% | 25.9% | 20.2% |
| member | 105 | 17.1% | 17.1% | 2.2095 | 79.0% | 49.5% | 26.7% | 3.8% |
| multiline | 82 | 0.0% | 0.0% | 5.8659 | 75.6% | 59.8% | 29.3% | 14.6% |
| operator | 134 | 14.2% | 14.2% | 3.2836 | 62.7% | 59.0% | 35.8% | 10.4% |
| partial | 100 | 19.0% | 19.0% | 3.44 | 71.0% | 60.0% | 34.0% | 8.0% |
| rest | 125 | 11.2% | 11.2% | 4.864 | 51.2% | 44.0% | 25.6% | 4.8% |
| stage | 125 | 0.0% | 0.0% | 6.952 | 72.0% | 71.2% | 43.2% | 22.4% |
| stmt_start | 39 | 0.0% | 0.0% | 0.0 | 38.5% | 23.1% | 0.0% | 0.0% |
| value | 132 | 14.4% | 14.4% | 2.1364 | 67.4% | 65.9% | 40.2% | 3.8% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 61 | 9.8% | 9.8% | 3.6066 | 63.9% | 50.8% | 24.6% | 11.5% |
| content | 82 | 14.6% | 14.6% | 4.0244 | 69.5% | 58.5% | 35.4% | 8.5% |
| energy | 94 | 6.4% | 6.4% | 4.2872 | 68.1% | 60.6% | 39.4% | 11.7% |
| finance | 78 | 11.5% | 11.5% | 3.5897 | 71.8% | 62.8% | 39.7% | 11.5% |
| gaming | 109 | 11.9% | 11.9% | 3.4679 | 60.6% | 55.0% | 24.8% | 11.0% |
| healthcare-synthetic | 87 | 3.5% | 3.5% | 3.4598 | 60.9% | 52.9% | 24.1% | 10.3% |
| hr | 93 | 8.6% | 8.6% | 3.3978 | 66.7% | 55.9% | 31.2% | 10.8% |
| iot | 81 | 7.4% | 7.4% | 3.5556 | 64.2% | 56.8% | 24.7% | 14.8% |
| logistics | 89 | 13.5% | 13.5% | 2.4045 | 73.0% | 61.8% | 37.1% | 9.0% |
| saas | 80 | 12.5% | 12.5% | 4.3125 | 61.3% | 60.0% | 36.2% | 10.0% |
| travel | 78 | 10.3% | 10.3% | 3.1795 | 70.5% | 57.7% | 35.9% | 9.0% |
| workspace | 68 | 5.9% | 5.9% | 3.2794 | 63.2% | 55.9% | 22.1% | 13.2% |

