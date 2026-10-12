# Benchmark `deepseek-6.7b-nf4`

- Backend: `{"backend": "hf", "load": "nf4", "model": "C:\\IA\\models\\KapibaraStudio.MongoIA\\BaseRepos\\deepseek-coder-6.7b-base", "adapter": null, "device": "AMD Radeon RX 7800 XT", "load_seconds": 11.7}` · contrato `deepseek-coder-fim`
- Exemplos: 1000 · 1056.1 s · pico RAM 1.02 GB · memória {'vram_max_allocated_gb': 4.14, 'vram_max_reserved_gb': 6.37}
- Modelo: 25.1 GB em disco · parâmetros 6740512768

## Principais

| métrica | valor | @32 tokens (IDE) |
|---|---:|---:|
| Mean Accepted Prefix Tokens | 2.7190 | 2.7110 |
| Exact Match | 0.1730 | 0.1710 |
| Exact Match (autocomplete) | 0.1730 | 0.1710 |
| Prefix Match | 0.2830 | 0.2830 |
| Edit Distance | 70.3 | 50.9 |
| FIM Accuracy | 0.2460 | 0.2431 |
| Syntax Valid | 0.6790 | 0.6100 |
| MongoDB Valid | 0.6457 | 0.5791 |
| Chat Intent Accuracy | — | — |
| PT-BR Understanding | — | — |
| English Understanding | — | — |
| Correction Accuracy | — | — |
| Conversacional/Markdown | 0.0070 | 0.0030 |
| TTFT médio (ms) | 362.9 | 362.9 |
| TTFT p95 (ms) | 417.1 | 417.1 |
| Tokens/s | 46.2 | 46.2 |

## Por grupo do benchmark

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregation | 257 | 3.5720 | 0.1984 | 0.2809 | 0.6473 | 0.6280 | — | 0.0039 | 371.3 |
| Atlas Search | 157 | 2.6497 | 0.1210 | 0.1818 | 0.5985 | 0.5455 | — | 0.0191 | 365.3 |
| CRUD | 157 | 1.8089 | 0.1146 | 0.1698 | 0.7923 | 0.7615 | — | 0.0000 | 359.2 |
| DSL | 107 | 2.8411 | 0.2336 | 0.3429 | 0.7159 | 0.6591 | — | 0.0093 | 361.0 |
| Extended JSON | 60 | 2.3500 | 0.1667 | 0.1957 | 0.6964 | 0.6786 | — | 0.0167 | 353.6 |
| Indexes | 54 | 2.0926 | 0.0741 | 0.1143 | 0.6471 | 0.5490 | — | 0.0000 | 342.6 |
| JSON | 48 | 2.0208 | 0.1458 | 0.2258 | 0.5897 | 0.5897 | — | 0.0000 | 341.3 |
| Operators | 108 | 2.5741 | 0.1667 | 0.2468 | 0.6489 | 0.6277 | — | 0.0093 | 368.5 |
| Scripts | 52 | 3.2308 | 0.4038 | 0.5294 | 0.8182 | 0.8182 | — | 0.0000 | 370.1 |

## Por tipo

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| completion | 313 | 2.7252 | 0.0192 | — | 0.5779 | 0.5130 | — | 0.0192 | 349.4 |
| fim | 687 | 2.7162 | 0.2431 | 0.2460 | 0.7016 | 0.6754 | — | 0.0015 | 369.1 |

## Por categoria

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aggregation | 257 | 3.5720 | 0.1984 | 0.2809 | 0.6473 | 0.6280 | — | 0.0039 | 371.3 |
| atlas_search | 157 | 2.6497 | 0.1210 | 0.1818 | 0.5985 | 0.5455 | — | 0.0191 | 365.3 |
| console_scripts | 52 | 3.2308 | 0.4038 | 0.5294 | 0.8182 | 0.8182 | — | 0.0000 | 370.1 |
| crud | 157 | 1.8089 | 0.1146 | 0.1698 | 0.7923 | 0.7615 | — | 0.0000 | 359.2 |
| indexes | 54 | 2.0926 | 0.0741 | 0.1143 | 0.6471 | 0.5490 | — | 0.0000 | 342.6 |
| json_ejson | 108 | 2.2037 | 0.1574 | 0.2078 | 0.6526 | 0.6421 | — | 0.0093 | 348.2 |
| operators | 108 | 2.5741 | 0.1667 | 0.2468 | 0.6489 | 0.6277 | — | 0.0093 | 368.5 |
| slop_dsl | 107 | 2.8411 | 0.2336 | 0.3429 | 0.7159 | 0.6591 | — | 0.0093 | 361.0 |

## Chat por idioma

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Por dialeto

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| console | 534 | 2.5449 | 0.1704 | 0.2438 | 0.6928 | 0.6614 | — | 0.0019 | 364.0 |
| json | 127 | 3.2913 | 0.2047 | 0.2842 | 0.6636 | 0.6364 | — | 0.0394 | 350.6 |
| mongosh | 339 | 2.7788 | 0.1652 | 0.2335 | 0.6632 | 0.6246 | — | 0.0029 | 365.8 |

## Domínio

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| in_domain | 597 | 2.5511 | 0.1843 | 0.2646 | 0.6680 | 0.6358 | — | 0.0067 | 361.3 |
| holdout | 403 | 2.9677 | 0.1563 | 0.2182 | 0.6948 | 0.6599 | — | 0.0074 | 365.3 |
