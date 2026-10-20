# Baseline zero-shot FIM `base-qwen2.5-coder-1.5b`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=2000 (de 2000), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-1.5B` · carga `bf16` · SHA-256 do modelo `8bcb242c05f27140…` (revisao `df3ce67c0e24480f20468b6ef2894622d69eb73b`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `28ba86be496839a3…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 1502.6 s totais (1488.1 s de geracao) · VRAM pico 3.04 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 14.2% | 13.4% | — |
| Normalized match (FIM acc.) | 14.2% | 13.4% | — |
| Prefix match | 25.0% | 25.0% | — |
| Syntax valid | 66.7% | 52.2% | 100.0% |
| Mongo valid (estatico) | 59.5% | 46.5% | 100.0% |
| Exec-valid | 31.6% | 25.4% | 62.5% |
| Campos inexistentes | 5.1% | 2.9% | 0.0% |
| Chars aceitos (media) | 4.0695 | 4.0695 | — |
| APT (tokens) | 1.439 | 1.439 | — |

Distribuicao do estagio de validacao (pred): {'ok': 650, 'api': 397, 'syntax': 787, 'fields': 132, 'type': 23, 'exec': 11}
Paradas: {'eos': 1269, 'suffix': 95, 'max': 636}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 319 | 13.5% | 13.5% | 2.7868 | 63.3% | 59.9% | 28.5% | 6.0% |
| member | 201 | 34.8% | 34.8% | 4.1144 | 77.1% | 57.2% | 33.8% | 3.5% |
| multiline | 161 | 0.0% | 0.0% | 4.9938 | 70.8% | 62.1% | 31.1% | 9.9% |
| operator | 268 | 19.8% | 19.8% | 4.209 | 61.2% | 57.8% | 32.1% | 4.5% |
| partial | 200 | 38.5% | 38.5% | 4.105 | 74.0% | 62.5% | 32.5% | 6.5% |
| rest | 241 | 6.6% | 6.6% | 5.7801 | 68.5% | 59.3% | 31.9% | 5.0% |
| stage | 251 | 0.4% | 0.4% | 5.9163 | 61.4% | 59.0% | 34.7% | 7.6% |
| stmt_start | 79 | 3.8% | 3.8% | 4.3165 | 70.9% | 57.0% | 32.9% | 0.0% |
| value | 280 | 7.5% | 7.5% | 1.6107 | 62.9% | 60.0% | 29.3% | 1.4% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 168 | 10.7% | 10.7% | 3.7976 | 72.6% | 64.9% | 41.7% | 4.8% |
| content | 161 | 13.7% | 13.7% | 4.2422 | 69.6% | 62.7% | 32.9% | 3.1% |
| energy | 171 | 15.2% | 15.2% | 3.4269 | 70.2% | 62.6% | 33.9% | 6.4% |
| finance | 175 | 11.4% | 11.4% | 3.92 | 62.9% | 56.0% | 33.1% | 5.1% |
| gaming | 163 | 12.3% | 12.3% | 3.9693 | 67.5% | 60.7% | 31.3% | 5.5% |
| healthcare-synthetic | 173 | 16.2% | 16.2% | 4.2428 | 69.4% | 63.0% | 32.4% | 2.9% |
| hr | 168 | 16.1% | 16.1% | 4.625 | 66.1% | 58.9% | 27.4% | 7.1% |
| iot | 168 | 14.3% | 14.3% | 4.2083 | 64.3% | 59.5% | 33.9% | 6.6% |
| logistics | 157 | 10.8% | 10.8% | 4.3694 | 63.1% | 52.9% | 28.7% | 4.5% |
| saas | 165 | 18.8% | 18.8% | 3.7697 | 64.2% | 55.8% | 27.3% | 3.6% |
| travel | 162 | 16.1% | 16.1% | 4.0988 | 64.8% | 59.3% | 32.1% | 5.6% |
| workspace | 169 | 14.8% | 14.8% | 4.1953 | 65.7% | 57.4% | 24.3% | 5.9% |

