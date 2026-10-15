# Baseline zero-shot FIM `sft_v2c-final`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=2000 (de 2000), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-7B` · carga `nf4` · SHA-256 do modelo `74ac207b4deacd56…` (revisao `0396a76181e127dfc13e5c5ec48a8cee09938b02`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `28ba86be496839a3…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 3542.9 s totais (3503.8 s de geracao) · VRAM pico 6.41 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 41.2% | 41.1% | — |
| Normalized match (FIM acc.) | 41.2% | 41.1% | — |
| Prefix match | 41.7% | 41.8% | — |
| Syntax valid | 99.7% | 85.7% | 100.0% |
| Mongo valid (estatico) | 99.5% | 85.5% | 100.0% |
| Exec-valid | 58.0% | 51.1% | 62.5% |
| Campos inexistentes | 4.0% | 2.1% | 0.0% |
| Chars aceitos (media) | 8.18 | 8.161 | — |
| APT (tokens) | 2.694 | 2.6875 | — |

Distribuicao do estagio de validacao (pred): {'ok': 1159, 'api': 495, 'syntax': 246, 'fields': 97, 'type': 2, 'exec': 1}
Paradas: {'eos': 1993, 'max': 7}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 319 | 45.5% | 45.5% | 5.8621 | 99.7% | 99.4% | 51.7% | 5.6% |
| member | 201 | 81.1% | 81.1% | 8.6119 | 100.0% | 100.0% | 66.7% | 0.0% |
| multiline | 161 | 0.6% | 0.6% | 12.0683 | 99.4% | 99.4% | 50.9% | 16.8% |
| operator | 268 | 60.8% | 60.8% | 7.9254 | 99.6% | 99.2% | 61.6% | 1.9% |
| partial | 200 | 72.0% | 72.0% | 6.305 | 100.0% | 100.0% | 58.5% | 4.5% |
| rest | 241 | 34.8% | 34.8% | 9.5187 | 100.0% | 99.6% | 59.3% | 2.1% |
| stage | 251 | 1.2% | 1.2% | 11.7809 | 98.8% | 98.8% | 53.8% | 4.8% |
| stmt_start | 79 | 35.4% | 35.4% | 16.9747 | 98.7% | 98.7% | 67.1% | 5.1% |
| value | 280 | 33.2% | 33.2% | 2.9964 | 100.0% | 99.6% | 58.9% | 0.4% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 168 | 37.5% | 37.5% | 6.5536 | 100.0% | 99.4% | 64.9% | 3.0% |
| content | 161 | 39.1% | 39.1% | 8.0248 | 100.0% | 100.0% | 55.9% | 3.1% |
| energy | 171 | 41.5% | 41.5% | 8.0877 | 98.8% | 98.8% | 60.8% | 2.9% |
| finance | 175 | 38.3% | 38.3% | 8.0229 | 99.4% | 99.4% | 57.1% | 4.0% |
| gaming | 163 | 41.1% | 41.1% | 7.7914 | 100.0% | 99.4% | 57.1% | 4.9% |
| healthcare-synthetic | 173 | 42.2% | 42.2% | 8.7341 | 99.4% | 99.4% | 56.1% | 2.9% |
| hr | 168 | 47.0% | 47.0% | 8.6488 | 100.0% | 100.0% | 58.9% | 3.0% |
| iot | 168 | 42.3% | 42.3% | 8.4286 | 99.4% | 99.4% | 60.7% | 7.1% |
| logistics | 157 | 40.1% | 40.1% | 8.8726 | 100.0% | 99.4% | 56.7% | 5.7% |
| saas | 165 | 48.5% | 48.5% | 8.6121 | 100.0% | 100.0% | 48.5% | 4.9% |
| travel | 162 | 40.7% | 40.7% | 8.5432 | 100.0% | 100.0% | 63.6% | 3.1% |
| workspace | 169 | 36.1% | 36.1% | 7.8817 | 98.8% | 98.2% | 55.0% | 4.1% |

