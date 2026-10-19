# Baseline zero-shot FIM `sft_distill_v1`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=2000 (de 2000), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\merged\sft_distill_v1` · carga `bf16` · SHA-256 do modelo `4e3ec2d44a0423f9…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `28ba86be496839a3…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 538.7 s totais (528.5 s de geracao) · VRAM pico 1.05 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 40.3% | 40.2% | — |
| Normalized match (FIM acc.) | 40.3% | 40.2% | — |
| Prefix match | 41.0% | 41.0% | — |
| Syntax valid | 99.5% | 82.7% | 100.0% |
| Mongo valid (estatico) | 99.4% | 82.5% | 100.0% |
| Exec-valid | 58.2% | 49.8% | 62.5% |
| Campos inexistentes | 3.8% | 2.3% | 0.0% |
| Chars aceitos (media) | 8.061 | 8.0485 | — |
| APT (tokens) | 2.6605 | 2.656 | — |

Distribuicao do estagio de validacao (pred): {'ok': 1164, 'api': 478, 'syntax': 249, 'fields': 103, 'type': 4, 'exec': 2}
Paradas: {'eos': 1993, 'max': 7}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 319 | 45.5% | 45.5% | 6.2884 | 99.7% | 99.7% | 50.2% | 7.8% |
| member | 201 | 76.6% | 76.6% | 7.8657 | 99.5% | 99.0% | 62.7% | 0.0% |
| multiline | 161 | 0.0% | 0.0% | 11.5217 | 100.0% | 100.0% | 56.5% | 13.0% |
| operator | 268 | 60.1% | 60.1% | 7.847 | 99.6% | 99.6% | 62.3% | 1.5% |
| partial | 200 | 70.5% | 70.5% | 6.32 | 100.0% | 100.0% | 56.0% | 5.0% |
| rest | 241 | 35.3% | 35.3% | 10.0664 | 99.6% | 99.2% | 59.8% | 0.8% |
| stage | 251 | 1.6% | 1.6% | 11.4661 | 99.2% | 99.2% | 57.0% | 3.6% |
| stmt_start | 79 | 36.7% | 36.7% | 15.9747 | 98.7% | 98.7% | 70.9% | 5.1% |
| value | 280 | 31.1% | 31.1% | 2.6679 | 98.9% | 98.9% | 58.9% | 0.4% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 168 | 38.7% | 38.7% | 6.8214 | 100.0% | 100.0% | 64.9% | 2.4% |
| content | 161 | 38.5% | 38.5% | 8.1615 | 99.4% | 99.4% | 57.8% | 1.9% |
| energy | 171 | 40.9% | 40.9% | 7.7778 | 98.8% | 98.8% | 60.8% | 4.7% |
| finance | 175 | 33.1% | 33.1% | 8.6171 | 98.9% | 98.9% | 54.3% | 6.3% |
| gaming | 163 | 38.0% | 38.0% | 7.865 | 99.4% | 98.2% | 57.1% | 4.3% |
| healthcare-synthetic | 173 | 39.9% | 39.9% | 8.4855 | 99.4% | 99.4% | 54.3% | 3.5% |
| hr | 168 | 48.8% | 48.8% | 8.6429 | 100.0% | 100.0% | 60.1% | 2.4% |
| iot | 168 | 40.5% | 40.5% | 8.3214 | 100.0% | 100.0% | 61.3% | 5.4% |
| logistics | 157 | 41.4% | 41.4% | 8.8217 | 100.0% | 100.0% | 58.6% | 3.8% |
| saas | 165 | 47.9% | 47.9% | 6.9939 | 99.4% | 99.4% | 48.5% | 4.9% |
| travel | 162 | 42.0% | 42.0% | 8.6975 | 100.0% | 100.0% | 64.8% | 2.5% |
| workspace | 169 | 34.3% | 34.3% | 7.5503 | 98.8% | 98.8% | 56.2% | 3.5% |

