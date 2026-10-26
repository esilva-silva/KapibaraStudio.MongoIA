# Baseline zero-shot FIM `sft_v3`

- Spec: `docs/fase-02-professor.md` §3.2 · split `val` · n=2000 (de 2000), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Small\merged\sft_v3` · carga `bf16` · SHA-256 do modelo `23e53b94514a68b1…` (revisao `None`)
- Dataset: `data\validated\v2c\fim-00000.jsonl` SHA-256 `af3331481966f3eb…` · amostra `28ba86be496839a3…`
- Contrato `editor-context-v1` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 674.1 s totais (656.7 s de geracao) · VRAM pico 3.04 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo 4.18.3

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 41.0% | 40.9% | — |
| Normalized match (FIM acc.) | 41.0% | 40.9% | — |
| Prefix match | 41.6% | 41.6% | — |
| Syntax valid | 99.8% | 83.6% | 100.0% |
| Mongo valid (estatico) | 99.8% | 83.5% | 100.0% |
| Exec-valid | 60.2% | 51.1% | 62.5% |
| Campos inexistentes | 2.1% | 1.2% | 0.0% |
| Chars aceitos (media) | 8.214 | 8.2 | — |
| APT (tokens) | 2.698 | 2.693 | — |

Distribuicao do estagio de validacao (pred): {'ok': 1203, 'api': 490, 'syntax': 243, 'type': 3, 'fields': 59, 'exec': 2}
Paradas: {'eos': 1995, 'max': 5}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 319 | 45.1% | 45.1% | 6.4639 | 100.0% | 100.0% | 53.6% | 4.1% |
| member | 201 | 80.1% | 80.1% | 8.6965 | 100.0% | 100.0% | 65.2% | 0.0% |
| multiline | 161 | 0.0% | 0.0% | 11.5652 | 98.8% | 98.8% | 61.5% | 8.1% |
| operator | 268 | 61.2% | 61.2% | 7.7052 | 100.0% | 100.0% | 63.1% | 0.4% |
| partial | 200 | 72.0% | 72.0% | 6.265 | 100.0% | 100.0% | 59.0% | 3.5% |
| rest | 241 | 34.4% | 34.4% | 9.8714 | 100.0% | 99.6% | 58.9% | 1.2% |
| stage | 251 | 2.4% | 2.4% | 11.8367 | 100.0% | 100.0% | 59.8% | 1.2% |
| stmt_start | 79 | 36.7% | 36.7% | 16.2785 | 98.7% | 98.7% | 70.9% | 2.5% |
| value | 280 | 31.8% | 31.8% | 2.8643 | 99.6% | 99.6% | 59.6% | 0.0% |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commerce | 168 | 38.1% | 38.1% | 7.1667 | 100.0% | 100.0% | 66.1% | 1.2% |
| content | 161 | 38.5% | 38.5% | 8.1366 | 100.0% | 100.0% | 60.9% | 0.6% |
| energy | 171 | 43.9% | 43.9% | 8.1637 | 99.4% | 99.4% | 60.2% | 4.7% |
| finance | 175 | 34.3% | 34.3% | 8.2457 | 99.4% | 99.4% | 59.4% | 2.9% |
| gaming | 163 | 37.4% | 37.4% | 7.9264 | 100.0% | 99.4% | 60.1% | 1.8% |
| healthcare-synthetic | 173 | 40.5% | 40.5% | 8.5896 | 99.4% | 99.4% | 58.4% | 1.2% |
| hr | 168 | 47.6% | 47.6% | 8.8333 | 100.0% | 100.0% | 60.1% | 1.2% |
| iot | 168 | 43.5% | 43.5% | 8.8393 | 100.0% | 100.0% | 64.3% | 3.6% |
| logistics | 157 | 43.3% | 43.3% | 8.9172 | 99.4% | 99.4% | 58.6% | 3.2% |
| saas | 165 | 47.9% | 47.9% | 7.2424 | 100.0% | 100.0% | 50.9% | 3.0% |
| travel | 162 | 40.7% | 40.7% | 8.8086 | 100.0% | 100.0% | 66.0% | 0.6% |
| workspace | 169 | 36.7% | 36.7% | 7.7278 | 100.0% | 100.0% | 56.8% | 1.2% |

