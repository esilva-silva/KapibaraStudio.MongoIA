# Baseline zero-shot FIM `Qwen2.5-Coder-1.5B-ctx`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 1000), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-1.5B` · carga `bf16` · SHA-256 do modelo `8bcb242c05f27140…` (revisao `df3ce67c0e24480f20468b6ef2894622d69eb73b`)
- Dataset: `data\generated\v1\fim_ctx.jsonl` SHA-256 `49a48066ed551bc7…` · amostra `d3debd9d53110e94…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 432.0 s totais (421.2 s de geracao) · VRAM pico 3.02 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 15.8% | 15.4% | — |
| Normalized match (FIM acc.) | 15.8% | 15.4% | — |
| Prefix match | 26.4% | 26.4% | — |
| Syntax valid | 73.0% | 58.0% | 100.0% |
| Mongo valid (estatico) | 67.4% | 53.2% | 100.0% |
| Exec-valid | 49.2% | 40.0% | 89.8% |
| Campos inexistentes | 5.6% | 2.8% | 0.0% |
| Chars aceitos (media) | 4.546 | 4.546 | — |
| APT (tokens) | 1.614 | 1.614 | — |

Distribuicao do estagio de validacao (pred): {'api': 58, 'syntax': 140, 'ok': 249, 'fields': 36, 'type': 10, 'exec': 7}
Paradas: {'eos': 358, 'max': 117, 'suffix': 25}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 81 | 16.1% | 16.1% | 3.5679 | 74.1% | 71.6% | 48.1% | 8.6% |
| member | 55 | 40.0% | 40.0% | 4.2909 | 90.9% | 76.4% | 52.7% | 1.8% |
| multiline | 44 | 0.0% | 0.0% | 5.1591 | 79.5% | 75.0% | 54.5% | 13.6% |
| operator | 69 | 14.5% | 14.5% | 4.1739 | 72.5% | 65.2% | 47.8% | 5.8% |
| partial | 52 | 40.4% | 40.4% | 4.0577 | 76.9% | 67.3% | 51.9% | 5.8% |
| rest | 61 | 8.2% | 8.2% | 7.459 | 62.3% | 55.7% | 42.6% | 3.3% |
| stage | 59 | 0.0% | 0.0% | 6.5593 | 79.7% | 79.7% | 67.8% | 5.1% |
| stmt_start | 17 | 11.8% | 11.8% | 2.7059 | 82.3% | 76.5% | 41.2% | 0.0% |
| value | 62 | 9.7% | 9.7% | 2.1613 | 50.0% | 48.4% | 33.9% | 3.2% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 31 | 12.9% | 12.9% | 6.6129 | 74.2% | 61.3% | 38.7% | 3.2% |
| content | 42 | 19.1% | 19.1% | 5.6667 | 83.3% | 81.0% | 57.1% | 7.1% |
| energy | 47 | 17.0% | 17.0% | 5.0213 | 72.3% | 72.3% | 55.3% | 8.5% |
| finance | 39 | 20.5% | 20.5% | 5.4103 | 76.9% | 66.7% | 61.5% | 7.7% |
| gaming | 54 | 16.7% | 16.7% | 4.3889 | 68.5% | 63.0% | 46.3% | 3.7% |
| healthcare-synthetic | 43 | 14.0% | 14.0% | 4.5581 | 67.4% | 62.8% | 51.2% | 2.3% |
| hr | 46 | 15.2% | 15.2% | 4.5652 | 71.7% | 65.2% | 39.1% | 4.3% |
| iot | 41 | 19.5% | 19.5% | 3.0488 | 63.4% | 61.0% | 48.8% | 4.9% |
| logistics | 44 | 15.9% | 15.9% | 3.6818 | 79.5% | 77.3% | 52.3% | 2.3% |
| saas | 40 | 12.5% | 12.5% | 5.125 | 75.0% | 72.5% | 52.5% | 12.5% |
| travel | 39 | 10.3% | 10.3% | 2.8974 | 74.4% | 64.1% | 51.3% | 5.1% |
| workspace | 34 | 14.7% | 14.7% | 3.9706 | 70.6% | 58.8% | 32.4% | 5.9% |

