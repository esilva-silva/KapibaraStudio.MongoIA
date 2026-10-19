# Baseline zero-shot FIM `base-qwen2.5-coder-0.5b`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=2000 (de 2000), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-0.5B` · carga `bf16` · SHA-256 do modelo `96217dc3a23c0290…` (revisao `8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `28ba86be496839a3…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 1866.4 s totais (1853.4 s de geracao) · VRAM pico 1.05 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 7.6% | 7.2% | — |
| Normalized match (FIM acc.) | 7.6% | 7.2% | — |
| Prefix match | 19.6% | 19.6% | — |
| Syntax valid | 49.8% | 38.5% | 100.0% |
| Mongo valid (estatico) | 42.8% | 33.5% | 100.0% |
| Exec-valid | 21.3% | 16.9% | 62.5% |
| Campos inexistentes | 4.9% | 3.5% | 0.0% |
| Chars aceitos (media) | 3.387 | 3.387 | — |
| APT (tokens) | 1.2045 | 1.2045 | — |

Distribuicao do estagio de validacao (pred): {'api': 296, 'syntax': 1101, 'ok': 437, 'fields': 115, 'type': 32, 'exec': 19}
Paradas: {'eos': 983, 'max': 907, 'suffix': 110}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 319 | 4.1% | 4.1% | 2.4828 | 46.4% | 42.0% | 17.2% | 6.0% |
| member | 201 | 18.9% | 18.9% | 3.2139 | 58.7% | 37.3% | 26.4% | 3.5% |
| multiline | 161 | 0.0% | 0.0% | 4.4596 | 57.8% | 49.7% | 26.1% | 6.8% |
| operator | 268 | 10.1% | 10.1% | 3.0299 | 44.0% | 40.3% | 19.0% | 9.0% |
| partial | 200 | 20.5% | 20.5% | 3.78 | 54.5% | 41.0% | 17.0% | 3.5% |
| rest | 241 | 5.4% | 5.4% | 4.0332 | 41.9% | 35.7% | 19.9% | 2.1% |
| stage | 251 | 0.4% | 0.4% | 6.0757 | 51.0% | 49.0% | 27.9% | 8.0% |
| stmt_start | 79 | 1.3% | 1.3% | 2.2278 | 41.8% | 29.1% | 12.7% | 0.0% |
| value | 280 | 6.4% | 6.4% | 1.3464 | 52.9% | 51.4% | 22.5% | 1.4% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 168 | 8.3% | 8.3% | 3.3452 | 51.8% | 48.2% | 27.4% | 4.2% |
| content | 161 | 5.6% | 5.6% | 3.2422 | 54.0% | 47.8% | 23.6% | 3.1% |
| energy | 171 | 6.4% | 6.4% | 3.2398 | 50.3% | 42.1% | 22.8% | 7.6% |
| finance | 175 | 5.7% | 5.7% | 3.3943 | 53.7% | 43.4% | 22.9% | 5.7% |
| gaming | 163 | 8.0% | 8.0% | 3.3374 | 52.1% | 43.6% | 19.6% | 6.1% |
| healthcare-synthetic | 173 | 7.5% | 7.5% | 3.9133 | 42.8% | 38.7% | 19.1% | 2.9% |
| hr | 168 | 10.1% | 10.1% | 3.756 | 48.8% | 40.5% | 21.4% | 4.8% |
| iot | 168 | 7.1% | 7.1% | 3.1429 | 50.0% | 44.0% | 22.0% | 6.6% |
| logistics | 157 | 8.3% | 8.3% | 3.4268 | 46.5% | 40.8% | 22.3% | 3.8% |
| saas | 165 | 6.7% | 6.7% | 2.8121 | 43.0% | 32.7% | 13.3% | 3.6% |
| travel | 162 | 12.3% | 12.3% | 3.463 | 56.2% | 50.0% | 24.7% | 4.3% |
| workspace | 169 | 5.3% | 5.3% | 3.5444 | 48.5% | 41.4% | 16.6% | 5.3% |

