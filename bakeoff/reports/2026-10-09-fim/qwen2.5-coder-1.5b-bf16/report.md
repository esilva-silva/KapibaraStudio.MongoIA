# Benchmark `qwen2.5-coder-1.5b-bf16`

- Backend: `{"backend": "hf", "load": "bf16", "model": "C:\\IA\\models\\KapibaraStudio.MongoIA\\BaseRepos\\Qwen2.5-Coder-1.5B", "adapter": null, "device": "AMD Radeon RX 7800 XT", "load_seconds": 2.2}` · contrato `qwen2.5-coder-fim`
- Exemplos: 1000 · 564.8 s · pico RAM 0.88 GB · memória {'vram_max_allocated_gb': 3.03, 'vram_max_reserved_gb': 3.05}
- Modelo: 2.9 GB em disco · parâmetros 1543714304

## Principais

| métrica | valor | @32 tokens (IDE) |
|---|---:|---:|
| Mean Accepted Prefix Tokens | 1.9410 | 1.9390 |
| Exact Match | 0.1010 | 0.0980 |
| Exact Match (autocomplete) | 0.1010 | 0.0980 |
| Prefix Match | 0.2480 | 0.2480 |
| Edit Distance | 91.8 | 63.3 |
| FIM Accuracy | 0.1441 | 0.1397 |
| Syntax Valid | 0.6290 | 0.5339 |
| MongoDB Valid | 0.5470 | 0.4542 |
| Chat Intent Accuracy | — | — |
| PT-BR Understanding | — | — |
| English Understanding | — | — |
| Correction Accuracy | — | — |
| Conversacional/Markdown | 0.0000 | 0.0000 |
| TTFT médio (ms) | 30.6 | 30.6 |
| TTFT p95 (ms) | 44.6 | 44.6 |
| Tokens/s | 61.6 | 61.6 |

## Por grupo do benchmark

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregation | 257 | 2.4475 | 0.0934 | 0.1348 | 0.5459 | 0.5024 | — | 0.0000 | 34.1 |
| Atlas Search | 157 | 1.8089 | 0.0446 | 0.0636 | 0.5682 | 0.4924 | — | 0.0000 | 30.7 |
| CRUD | 157 | 1.5032 | 0.0637 | 0.0943 | 0.6923 | 0.5615 | — | 0.0000 | 28.6 |
| DSL | 107 | 2.2336 | 0.1495 | 0.2286 | 0.6591 | 0.5455 | — | 0.0000 | 29.8 |
| Extended JSON | 60 | 1.6000 | 0.1500 | 0.1957 | 0.7321 | 0.5536 | — | 0.0000 | 28.7 |
| Indexes | 54 | 1.2593 | 0.1296 | 0.2000 | 0.7255 | 0.6275 | — | 0.0000 | 26.7 |
| JSON | 48 | 1.3542 | 0.0625 | 0.0968 | 0.5897 | 0.5641 | — | 0.0000 | 26.4 |
| Operators | 108 | 1.8426 | 0.0648 | 0.0909 | 0.6383 | 0.5745 | — | 0.0000 | 30.9 |
| Scripts | 52 | 2.4038 | 0.3462 | 0.4706 | 0.7273 | 0.7045 | — | 0.0000 | 29.6 |

## Por tipo

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| completion | 313 | 1.9585 | 0.0064 | — | 0.7403 | 0.5584 | — | 0.0000 | 28.5 |
| fim | 687 | 1.9330 | 0.1441 | 0.1441 | 0.6041 | 0.5444 | — | 0.0000 | 31.5 |

## Por categoria

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aggregation | 257 | 2.4475 | 0.0934 | 0.1348 | 0.5459 | 0.5024 | — | 0.0000 | 34.1 |
| atlas_search | 157 | 1.8089 | 0.0446 | 0.0636 | 0.5682 | 0.4924 | — | 0.0000 | 30.7 |
| console_scripts | 52 | 2.4038 | 0.3462 | 0.4706 | 0.7273 | 0.7045 | — | 0.0000 | 29.6 |
| crud | 157 | 1.5032 | 0.0637 | 0.0943 | 0.6923 | 0.5615 | — | 0.0000 | 28.6 |
| indexes | 54 | 1.2593 | 0.1296 | 0.2000 | 0.7255 | 0.6275 | — | 0.0000 | 26.7 |
| json_ejson | 108 | 1.4907 | 0.1111 | 0.1558 | 0.6737 | 0.5579 | — | 0.0000 | 27.7 |
| operators | 108 | 1.8426 | 0.0648 | 0.0909 | 0.6383 | 0.5745 | — | 0.0000 | 30.9 |
| slop_dsl | 107 | 2.2336 | 0.1495 | 0.2286 | 0.6591 | 0.5455 | — | 0.0000 | 29.8 |

## Chat por idioma

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Por dialeto

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| console | 534 | 1.9139 | 0.0899 | 0.1315 | 0.6435 | 0.5404 | — | 0.0000 | 30.6 |
| json | 127 | 2.2835 | 0.1260 | 0.1684 | 0.5545 | 0.5273 | — | 0.0000 | 30.0 |
| mongosh | 339 | 1.8555 | 0.1091 | 0.1542 | 0.6351 | 0.5649 | — | 0.0000 | 30.7 |

## Domínio

| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| in_domain | 597 | 1.9079 | 0.1139 | 0.1626 | 0.6358 | 0.5634 | — | 0.0000 | 30.8 |
| holdout | 403 | 1.9901 | 0.0819 | 0.1164 | 0.6192 | 0.5233 | — | 0.0000 | 30.3 |
