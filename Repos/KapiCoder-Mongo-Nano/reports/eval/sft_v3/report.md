# Baseline zero-shot FIM `sft_v3`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=2000 (de 2000), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\merged\sft_v3` · carga `bf16` · SHA-256 do modelo `e30c625c12e24dfd…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `28ba86be496839a3…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 349.6 s totais (337.7 s de geracao) · VRAM pico 1.05 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 40.5% | 40.4% | — |
| Normalized match (FIM acc.) | 40.5% | 40.4% | — |
| Prefix match | 41.0% | 41.0% | — |
| Syntax valid | 99.6% | 83.6% | 100.0% |
| Mongo valid (estatico) | 99.2% | 83.3% | 100.0% |
| Exec-valid | 59.5% | 50.5% | 62.5% |
| Campos inexistentes | 2.6% | 1.7% | 0.0% |
| Chars aceitos (media) | 7.9775 | 7.9575 | — |
| APT (tokens) | 2.62 | 2.6125 | — |

Distribuicao do estagio de validacao (pred): {'ok': 1189, 'api': 483, 'syntax': 248, 'fields': 75, 'exec': 2, 'type': 3}
Paradas: {'eos': 1994, 'max': 6}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 319 | 46.4% | 46.4% | 6.2351 | 99.7% | 99.4% | 53.9% | 3.8% |
| member | 201 | 77.1% | 77.1% | 7.9254 | 100.0% | 99.5% | 63.7% | 0.0% |
| multiline | 161 | 0.0% | 0.0% | 11.5155 | 99.4% | 99.4% | 59.0% | 8.7% |
| operator | 268 | 60.5% | 60.5% | 7.6194 | 99.6% | 99.6% | 63.1% | 1.9% |
| partial | 200 | 69.5% | 69.5% | 6.34 | 99.5% | 98.0% | 57.0% | 5.0% |
| rest | 241 | 34.4% | 34.4% | 9.4855 | 99.6% | 98.8% | 58.1% | 1.2% |
| stage | 251 | 1.2% | 1.2% | 11.4861 | 99.2% | 99.2% | 58.2% | 2.4% |
| stmt_start | 79 | 36.7% | 36.7% | 16.0633 | 98.7% | 98.7% | 74.7% | 3.8% |
| value | 280 | 32.5% | 32.5% | 2.7536 | 99.6% | 99.6% | 59.3% | 0.0% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 168 | 37.5% | 37.5% | 6.9881 | 100.0% | 99.4% | 65.5% | 3.0% |
| content | 161 | 39.1% | 39.1% | 8.0497 | 100.0% | 100.0% | 59.0% | 0.6% |
| energy | 171 | 42.7% | 42.7% | 7.7953 | 99.4% | 99.4% | 61.4% | 3.5% |
| finance | 175 | 34.9% | 34.9% | 8.1257 | 99.4% | 99.4% | 59.4% | 4.0% |
| gaming | 163 | 36.8% | 36.8% | 7.6196 | 98.8% | 97.5% | 59.5% | 2.5% |
| healthcare-synthetic | 173 | 41.0% | 41.0% | 8.3815 | 99.4% | 98.8% | 56.1% | 2.3% |
| hr | 168 | 46.4% | 46.4% | 7.8155 | 99.4% | 99.4% | 58.3% | 1.2% |
| iot | 168 | 41.1% | 41.1% | 8.4821 | 100.0% | 100.0% | 63.7% | 5.4% |
| logistics | 157 | 40.8% | 40.8% | 8.9554 | 100.0% | 99.4% | 59.9% | 2.5% |
| saas | 165 | 47.9% | 47.9% | 7.0242 | 99.4% | 99.4% | 48.5% | 3.6% |
| travel | 162 | 41.4% | 41.4% | 8.4506 | 100.0% | 100.0% | 65.4% | 1.2% |
| workspace | 169 | 36.7% | 36.7% | 8.0828 | 98.8% | 97.6% | 56.8% | 1.8% |

