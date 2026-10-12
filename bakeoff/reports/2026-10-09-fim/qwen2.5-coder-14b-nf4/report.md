# Benchmark `qwen2.5-coder-14b-nf4`

- Backend: `{"backend": "hf", "load": "nf4", "model": "C:\\IA\\models\\KapibaraStudio.MongoIA\\BaseRepos\\Qwen2.5-Coder-14B", "adapter": null, "device": "AMD Radeon RX 7800 XT", "load_seconds": 39.2}` · contrato `qwen2.5-coder-fim`
- Exemplos: 1000 · 1633.6 s · pico RAM 1.13 GB · memória {'vram_max_allocated_gb': 9.69, 'vram_max_reserved_gb': 9.88}
- Modelo: 27.5 GB em disco · parâmetros 14770033664

## Principais

| métrica | valor | @32 tokens (IDE) |
|---|---:|---:|
| Mean Accepted Prefix Tokens | 2.2180 | 2.2160 |
| Exact Match | 0.1660 | 0.1620 |
| Exact Match (autocomplete) | 0.1660 | 0.1620 |
| Prefix Match | 0.2850 | 0.2850 |
| Edit Distance | 72.3 | 54.3 |
| FIM Accuracy | 0.2329 | 0.2271 |
| Syntax Valid | 0.8038 | 0.6730 |
| MongoDB Valid | 0.7717 | 0.6516 |
| Chat Intent Accuracy | — | — |
| PT-BR Understanding | — | — |
| English Understanding | — | — |
| Correction Accuracy | — | — |
| Conversacional/Markdown | 0.0000 | 0.0000 |
| TTFT médio (ms) | 576.1 | 576.1 |
| TTFT p95 (ms) | 709.2 | 709.2 |
| Tokens/s | 25.1 | 25.1 |

## Por grupo do benchmark

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregation | 257 | 2.8833 | 0.1829 | 0.2528 | 0.7729 | 0.7391 | — | 0.0000 | 610.9 |
| Atlas Search | 157 | 2.4586 | 0.1274 | 0.1909 | 0.7500 | 0.7045 | — | 0.0000 | 579.7 |
| CRUD | 157 | 1.6178 | 0.0955 | 0.1415 | 0.8385 | 0.8077 | — | 0.0000 | 550.6 |
| DSL | 107 | 2.3084 | 0.2336 | 0.3429 | 0.8636 | 0.8409 | — | 0.0000 | 566.9 |
| Extended JSON | 60 | 1.8167 | 0.2167 | 0.2391 | 0.8393 | 0.7857 | — | 0.0000 | 551.1 |
| Indexes | 54 | 1.4815 | 0.1111 | 0.1714 | 0.8235 | 0.8039 | — | 0.0000 | 534.0 |
| JSON | 48 | 1.6042 | 0.1667 | 0.2258 | 0.7692 | 0.7179 | — | 0.0000 | 529.1 |
| Operators | 108 | 1.7500 | 0.1019 | 0.1688 | 0.7553 | 0.7447 | — | 0.0000 | 591.9 |
| Scripts | 52 | 2.5962 | 0.4038 | 0.5294 | 0.9545 | 0.9318 | — | 0.0000 | 573.2 |

## Por tipo

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| completion | 313 | 2.3482 | 0.0288 | — | 0.8117 | 0.7468 | — | 0.0000 | 553.1 |
| fim | 687 | 2.1587 | 0.2285 | 0.2329 | 0.8020 | 0.7773 | — | 0.0000 | 586.6 |

## Por categoria

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aggregation | 257 | 2.8833 | 0.1829 | 0.2528 | 0.7729 | 0.7391 | — | 0.0000 | 610.9 |
| atlas_search | 157 | 2.4586 | 0.1274 | 0.1909 | 0.7500 | 0.7045 | — | 0.0000 | 579.7 |
| console_scripts | 52 | 2.5962 | 0.4038 | 0.5294 | 0.9545 | 0.9318 | — | 0.0000 | 573.2 |
| crud | 157 | 1.6178 | 0.0955 | 0.1415 | 0.8385 | 0.8077 | — | 0.0000 | 550.6 |
| indexes | 54 | 1.4815 | 0.1111 | 0.1714 | 0.8235 | 0.8039 | — | 0.0000 | 534.0 |
| json_ejson | 108 | 1.7222 | 0.1944 | 0.2338 | 0.8105 | 0.7579 | — | 0.0000 | 541.3 |
| operators | 108 | 1.7500 | 0.1019 | 0.1688 | 0.7553 | 0.7447 | — | 0.0000 | 591.9 |
| slop_dsl | 107 | 2.3084 | 0.2336 | 0.3429 | 0.8636 | 0.8409 | — | 0.0000 | 566.9 |

## Chat por idioma

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Por dialeto

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| console | 534 | 2.1554 | 0.1723 | 0.2438 | 0.8139 | 0.7713 | — | 0.0000 | 577.4 |
| json | 127 | 2.5669 | 0.2047 | 0.2737 | 0.7091 | 0.7091 | — | 0.0000 | 569.0 |
| mongosh | 339 | 2.1858 | 0.1416 | 0.1982 | 0.8246 | 0.7965 | — | 0.0000 | 576.8 |

## Domínio

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| in_domain | 597 | 2.1307 | 0.1709 | 0.2427 | 0.7807 | 0.7485 | — | 0.0000 | 579.0 |
| holdout | 403 | 2.3474 | 0.1588 | 0.2182 | 0.8372 | 0.8052 | — | 0.0000 | 571.9 |
