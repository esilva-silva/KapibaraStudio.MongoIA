# Baseline zero-shot FIM `sft_distill_v1`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=2000 (de 2000), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Small\merged\sft_distill_v1` · carga `bf16` · SHA-256 do modelo `f7f71e24b606e26d…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `28ba86be496839a3…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 555.8 s totais (543.4 s de geracao) · VRAM pico 3.04 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 41.0% | 40.9% | — |
| Normalized match (FIM acc.) | 41.0% | 40.9% | — |
| Prefix match | 41.8% | 41.8% | — |
| Syntax valid | 99.6% | 82.5% | 100.0% |
| Mongo valid (estatico) | 99.5% | 82.3% | 100.0% |
| Exec-valid | 59.5% | 50.3% | 62.5% |
| Campos inexistentes | 2.9% | 1.6% | 0.0% |
| Chars aceitos (media) | 8.1585 | 8.146 | — |
| APT (tokens) | 2.6665 | 2.662 | — |

Distribuicao do estagio de validacao (pred): {'ok': 1190, 'api': 486, 'syntax': 248, 'type': 2, 'fields': 72, 'exec': 2}
Paradas: {'eos': 1992, 'max': 8}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 319 | 47.0% | 47.0% | 6.3009 | 100.0% | 100.0% | 52.7% | 5.0% |
| member | 201 | 80.6% | 80.6% | 8.5423 | 100.0% | 99.5% | 65.7% | 0.0% |
| multiline | 161 | 0.0% | 0.0% | 11.1863 | 99.4% | 99.4% | 59.0% | 10.6% |
| operator | 268 | 58.6% | 58.6% | 7.6828 | 99.6% | 99.6% | 62.3% | 0.8% |
| partial | 200 | 72.0% | 72.0% | 6.28 | 99.5% | 99.5% | 59.0% | 4.0% |
| rest | 241 | 34.8% | 34.8% | 9.9461 | 98.8% | 98.3% | 58.5% | 0.8% |
| stage | 251 | 1.6% | 1.6% | 11.9522 | 99.6% | 99.6% | 57.4% | 3.2% |
| stmt_start | 79 | 38.0% | 38.0% | 16.0633 | 98.7% | 98.7% | 73.4% | 6.3% |
| value | 280 | 32.1% | 32.1% | 2.8857 | 99.6% | 99.6% | 59.6% | 0.0% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 168 | 38.1% | 38.1% | 7.0298 | 100.0% | 100.0% | 66.1% | 3.0% |
| content | 161 | 40.4% | 40.4% | 8.3789 | 100.0% | 100.0% | 57.8% | 2.5% |
| energy | 171 | 42.1% | 42.1% | 8.1754 | 99.4% | 99.4% | 61.4% | 3.5% |
| finance | 175 | 36.6% | 36.6% | 8.3029 | 98.9% | 98.9% | 58.3% | 4.0% |
| gaming | 163 | 38.6% | 38.6% | 7.9141 | 100.0% | 98.8% | 60.7% | 1.8% |
| healthcare-synthetic | 173 | 42.2% | 42.2% | 8.711 | 99.4% | 99.4% | 57.8% | 1.2% |
| hr | 168 | 46.4% | 46.4% | 8.869 | 100.0% | 100.0% | 59.5% | 2.4% |
| iot | 168 | 42.3% | 42.3% | 8.4167 | 100.0% | 100.0% | 63.7% | 3.6% |
| logistics | 157 | 43.3% | 43.3% | 9.1465 | 100.0% | 100.0% | 58.6% | 4.5% |
| saas | 165 | 46.7% | 46.7% | 6.9152 | 99.4% | 99.4% | 49.1% | 3.6% |
| travel | 162 | 40.1% | 40.1% | 8.2037 | 100.0% | 100.0% | 65.4% | 1.8% |
| workspace | 169 | 36.1% | 36.1% | 7.8639 | 97.6% | 97.6% | 55.6% | 3.0% |

