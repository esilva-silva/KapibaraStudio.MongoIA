# Benchmark `qwen2.5-coder-7b-nf4`

- Backend: `{"backend": "hf", "load": "nf4", "model": "C:\\IA\\models\\KapibaraStudio.MongoIA\\BaseRepos\\Qwen2.5-Coder-7B", "adapter": null, "device": "AMD Radeon RX 7800 XT", "load_seconds": 17.9}` · contrato `qwen2.5-coder-fim`
- Exemplos: 1000 · 982.1 s · pico RAM 1.09 GB · memória {'vram_max_allocated_gb': 5.51, 'vram_max_reserved_gb': 6.73}
- Modelo: 14.2 GB em disco · parâmetros 7615616512

## Principais

| métrica | valor | @32 tokens (IDE) |
|---|---:|---:|
| Mean Accepted Prefix Tokens | 2.1130 | 2.1110 |
| Exact Match | 0.1620 | 0.1610 |
| Exact Match (autocomplete) | 0.1620 | 0.1610 |
| Prefix Match | 0.2740 | 0.2740 |
| Edit Distance | 73.4 | 55.3 |
| FIM Accuracy | 0.2329 | 0.2314 |
| Syntax Valid | 0.7824 | 0.6576 |
| MongoDB Valid | 0.7348 | 0.6219 |
| Chat Intent Accuracy | — | — |
| PT-BR Understanding | — | — |
| English Understanding | — | — |
| Correction Accuracy | — | — |
| Conversacional/Markdown | 0.0010 | 0.0000 |
| TTFT médio (ms) | 313.1 | 313.1 |
| TTFT p95 (ms) | 398.0 | 398.0 |
| Tokens/s | 41.1 | 41.1 |

## Por grupo do benchmark

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregation | 257 | 2.6109 | 0.1751 | 0.2416 | 0.7440 | 0.7101 | — | 0.0039 | 333.3 |
| Atlas Search | 157 | 2.1847 | 0.0892 | 0.1364 | 0.6818 | 0.5758 | — | 0.0000 | 312.5 |
| CRUD | 157 | 1.6688 | 0.1210 | 0.1792 | 0.8308 | 0.7846 | — | 0.0000 | 303.2 |
| DSL | 107 | 2.3178 | 0.2336 | 0.3571 | 0.8409 | 0.8068 | — | 0.0000 | 306.3 |
| Extended JSON | 60 | 1.5833 | 0.1500 | 0.1957 | 0.8571 | 0.7857 | — | 0.0000 | 303.8 |
| Indexes | 54 | 1.4630 | 0.1852 | 0.2857 | 0.8039 | 0.7255 | — | 0.0000 | 293.1 |
| JSON | 48 | 1.5208 | 0.1250 | 0.1935 | 0.7692 | 0.7692 | — | 0.0000 | 288.4 |
| Operators | 108 | 2.0463 | 0.1204 | 0.1948 | 0.8085 | 0.7872 | — | 0.0000 | 316.4 |
| Scripts | 52 | 2.3269 | 0.4038 | 0.5294 | 0.8409 | 0.8409 | — | 0.0000 | 306.8 |

## Por tipo

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| completion | 313 | 2.1597 | 0.0160 | — | 0.7727 | 0.7013 | — | 0.0032 | 299.9 |
| fim | 687 | 2.0917 | 0.2285 | 0.2329 | 0.7846 | 0.7424 | — | 0.0000 | 319.2 |

## Por categoria

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aggregation | 257 | 2.6109 | 0.1751 | 0.2416 | 0.7440 | 0.7101 | — | 0.0039 | 333.3 |
| atlas_search | 157 | 2.1847 | 0.0892 | 0.1364 | 0.6818 | 0.5758 | — | 0.0000 | 312.5 |
| console_scripts | 52 | 2.3269 | 0.4038 | 0.5294 | 0.8409 | 0.8409 | — | 0.0000 | 306.8 |
| crud | 157 | 1.6688 | 0.1210 | 0.1792 | 0.8308 | 0.7846 | — | 0.0000 | 303.2 |
| indexes | 54 | 1.4630 | 0.1852 | 0.2857 | 0.8039 | 0.7255 | — | 0.0000 | 293.1 |
| json_ejson | 108 | 1.5556 | 0.1389 | 0.1948 | 0.8211 | 0.7789 | — | 0.0000 | 296.9 |
| operators | 108 | 2.0463 | 0.1204 | 0.1948 | 0.8085 | 0.7872 | — | 0.0000 | 316.4 |
| slop_dsl | 107 | 2.3178 | 0.2336 | 0.3571 | 0.8409 | 0.8068 | — | 0.0000 | 306.3 |

## Chat por idioma

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Por dialeto

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| console | 534 | 2.0768 | 0.1610 | 0.2356 | 0.7691 | 0.7242 | — | 0.0019 | 313.3 |
| json | 127 | 2.5591 | 0.1732 | 0.2421 | 0.7182 | 0.7091 | — | 0.0000 | 307.5 |
| mongosh | 339 | 2.0029 | 0.1593 | 0.2247 | 0.8281 | 0.7614 | — | 0.0000 | 315.0 |

## Domínio

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| in_domain | 597 | 1.9246 | 0.1658 | 0.2403 | 0.7827 | 0.7384 | — | 0.0017 | 314.5 |
| holdout | 403 | 2.3921 | 0.1563 | 0.2218 | 0.7820 | 0.7297 | — | 0.0000 | 311.0 |
