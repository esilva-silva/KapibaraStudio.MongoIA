# Baseline zero-shot FIM `sft_gold_v1`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=2000 (de 2000), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\merged\sft_gold_v1` · carga `bf16` · SHA-256 do modelo `ab0be65211592471…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `28ba86be496839a3…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 337.6 s totais (324.4 s de geracao) · VRAM pico 1.05 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 41.1% | 41.0% | — |
| Normalized match (FIM acc.) | 41.1% | 41.0% | — |
| Prefix match | 41.9% | 42.0% | — |
| Syntax valid | 99.7% | 85.2% | 100.0% |
| Mongo valid (estatico) | 99.6% | 85.1% | 100.0% |
| Exec-valid | 56.5% | 50.3% | 62.5% |
| Campos inexistentes | 4.7% | 2.5% | 0.0% |
| Chars aceitos (media) | 8.2985 | 8.2375 | — |
| APT (tokens) | 2.744 | 2.721 | — |

Distribuicao do estagio de validacao (pred): {'ok': 1130, 'api': 507, 'syntax': 244, 'fields': 117, 'type': 2}
Paradas: {'eos': 1996, 'max': 4}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 319 | 44.5% | 44.5% | 5.9122 | 99.7% | 99.7% | 50.5% | 7.2% |
| member | 201 | 79.6% | 79.6% | 8.1791 | 100.0% | 99.5% | 63.2% | 0.0% |
| multiline | 161 | 0.6% | 0.6% | 11.6398 | 100.0% | 100.0% | 51.5% | 16.8% |
| operator | 268 | 61.6% | 61.6% | 7.9664 | 99.6% | 99.6% | 60.1% | 2.6% |
| partial | 200 | 71.5% | 71.5% | 7.08 | 100.0% | 100.0% | 56.5% | 5.5% |
| rest | 241 | 33.6% | 33.6% | 9.6763 | 100.0% | 99.6% | 58.5% | 2.5% |
| stage | 251 | 2.8% | 2.8% | 12.6534 | 98.4% | 98.4% | 49.8% | 6.4% |
| stmt_start | 79 | 35.4% | 35.4% | 16.4557 | 100.0% | 100.0% | 68.3% | 2.5% |
| value | 280 | 34.3% | 34.3% | 2.9786 | 100.0% | 100.0% | 58.9% | 0.7% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 168 | 38.7% | 38.7% | 7.119 | 100.0% | 100.0% | 63.7% | 4.2% |
| content | 161 | 43.5% | 43.5% | 8.3975 | 99.4% | 99.4% | 56.5% | 2.5% |
| energy | 171 | 43.3% | 43.3% | 8.2164 | 99.4% | 99.4% | 56.7% | 6.4% |
| finance | 175 | 33.1% | 33.1% | 7.6629 | 100.0% | 100.0% | 55.4% | 6.9% |
| gaming | 163 | 38.6% | 38.6% | 7.9939 | 100.0% | 98.8% | 57.7% | 3.1% |
| healthcare-synthetic | 173 | 40.5% | 40.5% | 8.6012 | 100.0% | 100.0% | 53.2% | 4.6% |
| hr | 168 | 47.6% | 47.6% | 8.3214 | 100.0% | 100.0% | 60.1% | 0.6% |
| iot | 168 | 42.3% | 42.3% | 8.6369 | 100.0% | 100.0% | 59.5% | 5.9% |
| logistics | 157 | 41.4% | 41.4% | 10.3567 | 99.4% | 99.4% | 53.5% | 8.3% |
| saas | 165 | 49.7% | 49.7% | 7.8788 | 99.4% | 99.4% | 47.3% | 4.2% |
| travel | 162 | 42.0% | 42.0% | 8.9136 | 100.0% | 100.0% | 63.0% | 3.1% |
| workspace | 169 | 33.7% | 33.7% | 7.6509 | 98.8% | 98.8% | 51.5% | 6.5% |

