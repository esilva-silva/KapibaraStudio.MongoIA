# Baseline zero-shot FIM `sft_v2d-final`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=2000 (de 2000), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-7B` · carga `nf4` · SHA-256 do modelo `74ac207b4deacd56…` (revisao `0396a76181e127dfc13e5c5ec48a8cee09938b02`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `28ba86be496839a3…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 3509.3 s totais (3475.9 s de geracao) · VRAM pico 6.41 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 40.9% | 40.8% | — |
| Normalized match (FIM acc.) | 41.0% | 40.9% | — |
| Prefix match | 41.5% | 41.6% | — |
| Syntax valid | 99.6% | 84.7% | 100.0% |
| Mongo valid (estatico) | 99.5% | 84.5% | 100.0% |
| Exec-valid | 57.6% | 49.6% | 62.5% |
| Campos inexistentes | 3.9% | 2.2% | 0.0% |
| Chars aceitos (media) | 8.3635 | 8.3445 | — |
| APT (tokens) | 2.736 | 2.7295 | — |

Distribuicao do estagio de validacao (pred): {'ok': 1152, 'api': 504, 'syntax': 247, 'fields': 93, 'type': 2, 'exec': 2}
Paradas: {'eos': 1991, 'max': 9}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 319 | 45.5% | 45.8% | 5.9592 | 99.4% | 99.4% | 50.5% | 6.6% |
| member | 201 | 79.6% | 79.6% | 8.5274 | 100.0% | 99.5% | 65.2% | 0.0% |
| multiline | 161 | 0.6% | 0.6% | 11.8447 | 99.4% | 99.4% | 54.0% | 13.7% |
| operator | 268 | 59.3% | 59.3% | 8.2463 | 99.6% | 99.6% | 61.6% | 1.1% |
| partial | 200 | 70.5% | 70.5% | 6.165 | 100.0% | 99.5% | 58.5% | 5.0% |
| rest | 241 | 35.3% | 35.3% | 10.0207 | 100.0% | 99.6% | 58.1% | 2.9% |
| stage | 251 | 2.0% | 2.0% | 12.1793 | 99.2% | 99.2% | 52.6% | 4.4% |
| stmt_start | 79 | 38.0% | 38.0% | 17.8101 | 97.5% | 97.5% | 65.8% | 3.8% |
| value | 280 | 33.2% | 33.2% | 3.1536 | 100.0% | 100.0% | 59.6% | 0.0% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 168 | 36.9% | 36.9% | 7.1726 | 100.0% | 100.0% | 63.7% | 4.2% |
| content | 161 | 40.4% | 40.4% | 8.3602 | 100.0% | 100.0% | 57.8% | 1.9% |
| energy | 171 | 40.4% | 40.4% | 8.4327 | 99.4% | 99.4% | 60.2% | 2.9% |
| finance | 175 | 36.6% | 36.6% | 8.2457 | 99.4% | 99.4% | 56.6% | 4.6% |
| gaming | 163 | 39.9% | 39.9% | 7.9755 | 99.4% | 98.2% | 57.1% | 3.7% |
| healthcare-synthetic | 173 | 39.9% | 40.5% | 9.1561 | 99.4% | 99.4% | 54.9% | 3.5% |
| hr | 168 | 47.0% | 47.0% | 8.6667 | 100.0% | 99.4% | 58.9% | 3.0% |
| iot | 168 | 42.9% | 42.9% | 8.2024 | 99.4% | 99.4% | 59.5% | 6.6% |
| logistics | 157 | 42.0% | 42.0% | 9.293 | 100.0% | 100.0% | 56.7% | 5.7% |
| saas | 165 | 49.7% | 49.7% | 8.3091 | 100.0% | 100.0% | 50.3% | 2.4% |
| travel | 162 | 39.5% | 39.5% | 8.3642 | 99.4% | 99.4% | 61.7% | 3.7% |
| workspace | 169 | 36.7% | 36.7% | 8.213 | 98.8% | 98.8% | 53.8% | 4.1% |

