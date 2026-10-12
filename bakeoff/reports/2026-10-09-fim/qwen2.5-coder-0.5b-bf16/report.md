# Benchmark `qwen2.5-coder-0.5b-bf16`

- Backend: `{"backend": "hf", "load": "bf16", "model": "C:\\IA\\models\\KapibaraStudio.MongoIA\\BaseRepos\\Qwen2.5-Coder-0.5B", "adapter": null, "device": "AMD Radeon RX 7800 XT", "load_seconds": 0.7}` · contrato `qwen2.5-coder-fim`
- Exemplos: 1000 · 509.5 s · pico RAM 0.88 GB · memória {'vram_max_allocated_gb': 1.05, 'vram_max_reserved_gb': 1.07}
- Modelo: 942.3 MB em disco · parâmetros 494032768

## Principais

| métrica | valor | @32 tokens (IDE) |
|---|---:|---:|
| Mean Accepted Prefix Tokens | 1.4570 | 1.4570 |
| Exact Match | 0.0530 | 0.0460 |
| Exact Match (autocomplete) | 0.0530 | 0.0460 |
| Prefix Match | 0.2120 | 0.2120 |
| Edit Distance | 114.9 | 73.1 |
| FIM Accuracy | 0.0742 | 0.0640 |
| Syntax Valid | 0.5386 | 0.4340 |
| MongoDB Valid | 0.4043 | 0.3532 |
| Chat Intent Accuracy | — | — |
| PT-BR Understanding | — | — |
| English Understanding | — | — |
| Correction Accuracy | — | — |
| Conversacional/Markdown | 0.0000 | 0.0000 |
| TTFT médio (ms) | 14.5 | 14.5 |
| TTFT p95 (ms) | 21.0 | 21.0 |
| Tokens/s | 85.4 | 85.4 |

## Por grupo do benchmark

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregation | 257 | 1.7315 | 0.0389 | 0.0562 | 0.4686 | 0.3575 | — | 0.0000 | 15.1 |
| Atlas Search | 157 | 1.1401 | 0.0191 | 0.0273 | 0.5303 | 0.3409 | — | 0.0000 | 14.6 |
| CRUD | 157 | 1.2739 | 0.0446 | 0.0660 | 0.5308 | 0.4077 | — | 0.0000 | 14.3 |
| DSL | 107 | 1.8505 | 0.1121 | 0.1714 | 0.6818 | 0.4886 | — | 0.0000 | 14.4 |
| Extended JSON | 60 | 1.5000 | 0.0667 | 0.0870 | 0.5893 | 0.4821 | — | 0.0000 | 14.0 |
| Indexes | 54 | 0.9259 | 0.0185 | 0.0286 | 0.4510 | 0.2549 | — | 0.0000 | 13.6 |
| JSON | 48 | 0.7917 | 0.0208 | 0.0323 | 0.5128 | 0.3846 | — | 0.0000 | 13.2 |
| Operators | 108 | 1.3889 | 0.0370 | 0.0649 | 0.5638 | 0.4787 | — | 0.0000 | 14.5 |
| Scripts | 52 | 2.0577 | 0.2115 | 0.2353 | 0.6364 | 0.5682 | — | 0.0000 | 13.9 |

## Por tipo

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| completion | 313 | 1.4473 | 0.0096 | — | 0.6039 | 0.3442 | — | 0.0000 | 14.2 |
| fim | 687 | 1.4614 | 0.0728 | 0.0742 | 0.5240 | 0.4178 | — | 0.0000 | 14.6 |

## Por categoria

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aggregation | 257 | 1.7315 | 0.0389 | 0.0562 | 0.4686 | 0.3575 | — | 0.0000 | 15.1 |
| atlas_search | 157 | 1.1401 | 0.0191 | 0.0273 | 0.5303 | 0.3409 | — | 0.0000 | 14.6 |
| console_scripts | 52 | 2.0577 | 0.2115 | 0.2353 | 0.6364 | 0.5682 | — | 0.0000 | 13.9 |
| crud | 157 | 1.2739 | 0.0446 | 0.0660 | 0.5308 | 0.4077 | — | 0.0000 | 14.3 |
| indexes | 54 | 0.9259 | 0.0185 | 0.0286 | 0.4510 | 0.2549 | — | 0.0000 | 13.6 |
| json_ejson | 108 | 1.1852 | 0.0463 | 0.0649 | 0.5579 | 0.4421 | — | 0.0000 | 13.6 |
| operators | 108 | 1.3889 | 0.0370 | 0.0649 | 0.5638 | 0.4787 | — | 0.0000 | 14.5 |
| slop_dsl | 107 | 1.8505 | 0.1121 | 0.1714 | 0.6818 | 0.4886 | — | 0.0000 | 14.4 |

## Chat por idioma

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Por dialeto

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| console | 534 | 1.4981 | 0.0562 | 0.0822 | 0.5673 | 0.3969 | — | 0.0000 | 14.3 |
| json | 127 | 1.7402 | 0.0157 | 0.0211 | 0.3909 | 0.3727 | — | 0.0000 | 14.1 |
| mongosh | 339 | 1.2861 | 0.0619 | 0.0837 | 0.5509 | 0.4281 | — | 0.0000 | 14.8 |

## Domínio

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| in_domain | 597 | 1.4288 | 0.0653 | 0.0922 | 0.5533 | 0.4266 | — | 0.0000 | 14.5 |
| holdout | 403 | 1.4988 | 0.0347 | 0.0473 | 0.5174 | 0.3721 | — | 0.0000 | 14.4 |
