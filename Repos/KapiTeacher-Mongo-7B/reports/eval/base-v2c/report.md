# Baseline zero-shot FIM `base-v2c`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=2000 (de 34499), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-7B` · carga `nf4` · SHA-256 do modelo `74ac207b4deacd56…` (revisao `0396a76181e127dfc13e5c5ec48a8cee09938b02`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `28ba86be496839a3…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 2324.2 s totais (2296.6 s de geracao) · VRAM pico 5.51 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 21.0% | 20.6% | — |
| Normalized match (FIM acc.) | 21.1% | 20.8% | — |
| Prefix match | 28.8% | 28.8% | — |
| Syntax valid | 83.0% | 65.3% | 100.0% |
| Mongo valid (estatico) | 79.3% | 62.4% | 100.0% |
| Exec-valid | 43.6% | 35.2% | 62.5% |
| Campos inexistentes | 5.1% | 2.9% | 0.0% |
| Chars aceitos (media) | 4.6635 | 4.6635 | — |
| APT (tokens) | 1.606 | 1.606 | — |

Distribuicao do estagio de validacao (pred): {'ok': 890, 'api': 419, 'syntax': 519, 'fields': 138, 'exec': 13, 'type': 21}
Paradas: {'eos': 1669, 'max': 300, 'suffix': 31}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 319 | 20.4% | 20.4% | 3.7837 | 82.8% | 79.9% | 36.7% | 6.6% |
| member | 201 | 42.8% | 43.3% | 4.9204 | 93.0% | 79.6% | 47.3% | 1.0% |
| multiline | 161 | 0.0% | 0.0% | 5.4907 | 79.5% | 78.9% | 40.4% | 9.9% |
| operator | 268 | 32.1% | 32.1% | 4.7239 | 85.1% | 80.6% | 49.6% | 6.0% |
| partial | 200 | 53.5% | 53.5% | 5.505 | 89.0% | 85.5% | 45.5% | 5.5% |
| rest | 241 | 10.8% | 11.2% | 6.2531 | 76.8% | 73.4% | 45.2% | 3.3% |
| stage | 251 | 0.4% | 0.8% | 6.1713 | 78.9% | 78.1% | 41.8% | 8.4% |
| stmt_start | 79 | 10.1% | 10.1% | 3.7468 | 79.8% | 74.7% | 41.8% | 1.3% |
| value | 280 | 14.6% | 14.6% | 1.8857 | 81.8% | 80.4% | 44.6% | 1.8% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 168 | 25.0% | 25.0% | 5.1726 | 91.1% | 86.3% | 51.2% | 5.4% |
| content | 161 | 18.0% | 18.6% | 4.2547 | 80.8% | 77.0% | 42.2% | 1.9% |
| energy | 171 | 21.1% | 21.1% | 4.0351 | 86.6% | 81.9% | 46.8% | 4.7% |
| finance | 175 | 16.6% | 16.6% | 4.6 | 79.4% | 76.6% | 41.7% | 6.9% |
| gaming | 163 | 16.6% | 17.8% | 4.5276 | 79.8% | 77.3% | 40.5% | 5.5% |
| healthcare-synthetic | 173 | 20.8% | 20.8% | 4.6358 | 84.4% | 79.2% | 39.9% | 4.0% |
| hr | 168 | 29.2% | 29.2% | 6.0595 | 86.3% | 83.3% | 45.2% | 7.1% |
| iot | 168 | 22.6% | 22.6% | 4.2619 | 79.8% | 78.6% | 45.2% | 8.9% |
| logistics | 157 | 19.1% | 19.1% | 4.9809 | 81.5% | 77.1% | 46.5% | 7.6% |
| saas | 165 | 26.1% | 26.1% | 4.2182 | 81.8% | 78.2% | 38.8% | 0.6% |
| travel | 162 | 19.1% | 19.1% | 4.1728 | 82.1% | 79.0% | 46.3% | 6.2% |
| workspace | 169 | 17.8% | 17.8% | 5.0296 | 82.2% | 76.9% | 39.6% | 1.8% |

