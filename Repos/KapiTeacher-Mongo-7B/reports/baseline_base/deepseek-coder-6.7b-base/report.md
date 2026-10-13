# Baseline zero-shot FIM `deepseek-coder-6.7b-base`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=1000 (de 2830), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\deepseek-coder-6.7b-base` · carga `nf4` · SHA-256 do modelo `5f55430e1e55055e…` (revisao `None`)
- Dataset: `data\generated\v1\fim.jsonl` SHA-256 `6b4065ce37dff97c…` · amostra `9317e5b9c45c0006…`
- Contrato `fim-raw` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 883.8 s totais (845.3 s de geracao) · VRAM pico 3.93 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 18.5% | 18.3% | — |
| Normalized match (FIM acc.) | 18.5% | 18.3% | — |
| Prefix match | 24.1% | 24.0% | — |
| Syntax valid | 91.2% | 77.2% | 100.0% |
| Mongo valid (estatico) | 86.2% | 72.6% | 100.0% |
| Exec-valid | 52.5% | 45.9% | 89.6% |
| Campos inexistentes | 14.5% | 9.9% | 0.0% |
| Chars aceitos (media) | 4.54 | 4.526 | — |
| APT (tokens) | 2.091 | 2.087 | — |

Distribuicao do estagio de validacao (pred): {'ok': 529, 'fields': 190, 'api': 158, 'syntax': 99, 'exec': 7, 'type': 17}
Paradas: {'eos': 924, 'max': 62, 'suffix': 14}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 158 | 13.3% | 13.3% | 2.2025 | 93.0% | 92.4% | 48.1% | 28.5% |
| member | 105 | 26.7% | 26.7% | 2.9048 | 96.2% | 68.6% | 39.1% | 1.9% |
| multiline | 82 | 0.0% | 0.0% | 6.4268 | 86.6% | 84.2% | 30.5% | 36.6% |
| operator | 134 | 26.1% | 26.1% | 4.6567 | 91.0% | 88.8% | 59.0% | 11.9% |
| partial | 100 | 41.0% | 41.0% | 4.44 | 94.0% | 90.0% | 61.0% | 13.0% |
| rest | 125 | 24.0% | 24.0% | 6.736 | 89.6% | 84.0% | 63.2% | 8.8% |
| stage | 125 | 0.0% | 0.0% | 8.8 | 92.8% | 92.8% | 68.0% | 16.8% |
| stmt_start | 39 | 10.3% | 10.3% | 0.3077 | 92.3% | 84.6% | 25.6% | 2.6% |
| value | 132 | 19.7% | 19.7% | 2.5606 | 85.6% | 84.9% | 52.3% | 4.5% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 61 | 24.6% | 24.6% | 5.4426 | 93.4% | 90.2% | 54.1% | 11.5% |
| content | 82 | 23.2% | 23.2% | 5.1341 | 92.7% | 90.2% | 59.8% | 13.4% |
| energy | 94 | 22.3% | 22.3% | 5.4681 | 92.5% | 87.2% | 64.9% | 13.8% |
| finance | 78 | 24.4% | 24.4% | 5.0641 | 87.2% | 83.3% | 52.6% | 16.7% |
| gaming | 109 | 17.4% | 17.4% | 4.2202 | 92.7% | 88.1% | 56.0% | 15.6% |
| healthcare-synthetic | 87 | 11.5% | 11.5% | 5.2069 | 93.1% | 86.2% | 52.9% | 12.6% |
| hr | 93 | 11.8% | 11.8% | 3.6667 | 89.2% | 82.8% | 45.2% | 15.0% |
| iot | 81 | 19.8% | 19.8% | 4.4198 | 84.0% | 79.0% | 44.4% | 12.3% |
| logistics | 89 | 15.7% | 15.7% | 2.7191 | 93.3% | 83.2% | 39.3% | 18.0% |
| saas | 80 | 16.2% | 16.2% | 5.7625 | 92.5% | 92.5% | 57.5% | 12.5% |
| travel | 78 | 17.9% | 17.9% | 3.5769 | 89.7% | 84.6% | 56.4% | 16.7% |
| workspace | 68 | 20.6% | 20.6% | 4.1765 | 94.1% | 88.2% | 45.6% | 14.7% |

