# Baseline zero-shot FIM `Qwen2.5-Coder-7B-ctx`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=500 (de 1000), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-7B` · carga `nf4` · SHA-256 do modelo `74ac207b4deacd56…` (revisao `0396a76181e127dfc13e5c5ec48a8cee09938b02`)
- Dataset: `data\generated\v1\fim_ctx.jsonl` SHA-256 `49a48066ed551bc7…` · amostra `d3debd9d53110e94…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 553.1 s totais (524.7 s de geracao) · VRAM pico 5.49 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 21.0% | 20.8% | — |
| Normalized match (FIM acc.) | 21.0% | 20.8% | — |
| Prefix match | 30.2% | 30.2% | — |
| Syntax valid | 83.0% | 67.2% | 100.0% |
| Mongo valid (estatico) | 80.4% | 65.2% | 100.0% |
| Exec-valid | 59.0% | 49.2% | 89.8% |
| Campos inexistentes | 6.0% | 3.0% | 0.0% |
| Chars aceitos (media) | 5.062 | 5.062 | — |
| APT (tokens) | 1.744 | 1.744 | — |

Distribuicao do estagio de validacao (pred): {'ok': 298, 'exec': 7, 'syntax': 92, 'api': 56, 'fields': 35, 'type': 12}
Paradas: {'eos': 420, 'max': 74, 'suffix': 6}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 81 | 18.5% | 18.5% | 4.0617 | 81.5% | 80.2% | 54.3% | 6.2% |
| member | 55 | 52.7% | 52.7% | 5.3273 | 98.2% | 87.3% | 65.5% | 0.0% |
| multiline | 44 | 0.0% | 0.0% | 5.0 | 84.1% | 84.1% | 56.8% | 20.4% |
| operator | 69 | 24.6% | 24.6% | 4.2464 | 81.2% | 79.7% | 63.8% | 2.9% |
| partial | 52 | 50.0% | 50.0% | 5.2308 | 94.2% | 92.3% | 71.2% | 5.8% |
| rest | 61 | 14.8% | 14.8% | 7.5082 | 72.1% | 67.2% | 54.1% | 1.6% |
| stage | 59 | 0.0% | 0.0% | 7.8305 | 84.8% | 83.0% | 61.0% | 15.2% |
| stmt_start | 17 | 0.0% | 0.0% | 3.0588 | 88.2% | 88.2% | 52.9% | 0.0% |
| value | 62 | 14.5% | 14.5% | 2.4516 | 71.0% | 71.0% | 50.0% | 1.6% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 31 | 25.8% | 25.8% | 8.0323 | 74.2% | 74.2% | 48.4% | 0.0% |
| content | 42 | 33.3% | 33.3% | 6.3095 | 83.3% | 83.3% | 54.8% | 4.8% |
| energy | 47 | 19.1% | 19.1% | 5.7234 | 89.4% | 87.2% | 76.6% | 6.4% |
| finance | 39 | 28.2% | 28.2% | 5.7949 | 84.6% | 79.5% | 69.2% | 2.6% |
| gaming | 54 | 24.1% | 24.1% | 4.6111 | 85.2% | 79.6% | 57.4% | 7.4% |
| healthcare-synthetic | 43 | 7.0% | 7.0% | 4.093 | 83.7% | 83.7% | 65.1% | 9.3% |
| hr | 46 | 17.4% | 17.4% | 4.6304 | 87.0% | 82.6% | 54.4% | 6.5% |
| iot | 41 | 12.2% | 12.2% | 3.6098 | 75.6% | 75.6% | 48.8% | 7.3% |
| logistics | 44 | 27.3% | 27.3% | 4.5 | 90.9% | 88.6% | 65.9% | 4.5% |
| saas | 40 | 27.5% | 27.5% | 5.775 | 82.5% | 82.5% | 55.0% | 7.5% |
| travel | 39 | 10.3% | 10.3% | 4.1282 | 84.6% | 79.5% | 61.5% | 10.3% |
| workspace | 34 | 20.6% | 20.6% | 4.2941 | 67.7% | 61.8% | 44.1% | 2.9% |

