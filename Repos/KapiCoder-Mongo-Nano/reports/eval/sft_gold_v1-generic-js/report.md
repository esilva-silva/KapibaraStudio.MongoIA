# Baseline zero-shot FIM `sft_gold_v1-generic-js`

- Spec: `docs/fase-02-professor.md` §3.2 · split `eval` · n=500 (de 500), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\merged\sft_gold_v1` · carga `bf16` · SHA-256 do modelo `ab0be65211592471…` (revisao `None`)
- Dataset: `data\replay\v1\generic_js_fim_eval.jsonl` SHA-256 `8827d1e4078a9f21…` · amostra `fbbfc5471d4a0a7b…`
- Contrato `fim-raw` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 71.8 s totais (63.5 s de geracao) · VRAM pico 1.1 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo —

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 49.6% | 49.6% | — |
| Normalized match (FIM acc.) | 50.4% | 50.4% | — |
| Prefix match | 52.4% | 52.4% | — |
| Syntax valid | 91.4% | 88.0% | 100.0% |
| Mongo valid (estatico) | — | — | — |
| Exec-valid | — | — | — |
| Campos inexistentes | — | — | — |
| Chars aceitos (media) | 9.876 | 9.876 | — |
| APT (tokens) | 2.602 | 2.602 | — |

Distribuicao do estagio de validacao (pred): {}
Paradas: {'eos': 487, 'max': 13}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 39 | 64.1% | 64.1% | 8.8205 | 97.4% | — | — | — |
| member | 101 | 52.5% | 53.5% | 6.5743 | 98.0% | — | — | — |
| multiline | 43 | 14.0% | 16.3% | 16.907 | 88.4% | — | — | — |
| partial | 55 | 65.5% | 65.5% | 5.8545 | 94.5% | — | — | — |
| rest | 103 | 45.6% | 47.6% | 11.0485 | 84.5% | — | — | — |
| stmt_start | 55 | 34.5% | 34.5% | 17.8727 | 83.6% | — | — | — |
| value | 104 | 59.6% | 59.6% | 7.3077 | 93.3% | — | — | — |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| @azure/core-client | 9 | 66.7% | 66.7% | 7.1111 | 100.0% | — | — | — |
| @azure/core-util | 6 | 50.0% | 50.0% | 7.1667 | 83.3% | — | — | — |
| @babel/core | 30 | 50.0% | 50.0% | 15.4667 | 100.0% | — | — | — |
| @babel/helper-builder-binary-assignment-operator-visitor | 2 | 50.0% | 50.0% | 8.0 | 50.0% | — | — | — |
| @babel/helper-compilation-targets | 4 | 25.0% | 25.0% | 31.5 | 75.0% | — | — | — |
| @babel/helper-module-imports | 3 | 0.0% | 0.0% | 3.0 | 100.0% | — | — | — |
| @babel/helper-module-transforms | 4 | 50.0% | 75.0% | 13.75 | 75.0% | — | — | — |
| @babel/plugin-transform-block-scoping | 3 | 0.0% | 0.0% | 1.3333 | 66.7% | — | — | — |
| @babel/template | 4 | 50.0% | 50.0% | 1.75 | 100.0% | — | — | — |
| @babel/types | 30 | 56.7% | 60.0% | 9.4 | 90.0% | — | — | — |
| @grpc/proto-loader | 2 | 50.0% | 50.0% | 28.5 | 100.0% | — | — | — |
| @hapi/hoek | 12 | 41.7% | 41.7% | 6.4167 | 100.0% | — | — | — |
| @npmcli/metavuln-calculator | 2 | 50.0% | 50.0% | 7.0 | 100.0% | — | — | — |
| @npmcli/promise-spawn | 2 | 100.0% | 100.0% | 3.5 | 100.0% | — | — | — |
| @pkgjs/parseargs | 4 | 25.0% | 25.0% | 10.5 | 100.0% | — | — | — |
| @protobufjs/float | 2 | 0.0% | 0.0% | 16.5 | 50.0% | — | — | — |
| @sigstore/bundle | 4 | 50.0% | 50.0% | 21.75 | 75.0% | — | — | — |
| @sigstore/tuf | 5 | 80.0% | 80.0% | 5.0 | 100.0% | — | — | — |
| @tufjs/models | 14 | 71.4% | 71.4% | 14.2143 | 92.9% | — | — | — |
| adm-zip | 7 | 42.9% | 42.9% | 4.4286 | 85.7% | — | — | — |
| babel-plugin-polyfill-corejs2 | 3 | 66.7% | 66.7% | 21.6667 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs3 | 4 | 25.0% | 25.0% | 5.5 | 75.0% | — | — | — |
| browserslist | 4 | 75.0% | 75.0% | 6.0 | 75.0% | — | — | — |
| cacache | 12 | 41.7% | 41.7% | 6.3333 | 91.7% | — | — | — |
| cli-progress | 8 | 62.5% | 62.5% | 8.0 | 100.0% | — | — | — |
| cmd-shim | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| commander | 4 | 75.0% | 75.0% | 7.5 | 100.0% | — | — | — |
| core-js-compat | 3 | 33.3% | 33.3% | 4.3333 | 100.0% | — | — | — |
| d | 2 | 50.0% | 50.0% | 2.5 | 100.0% | — | — | — |
| date-fns | 30 | 53.3% | 60.0% | 12.8667 | 96.7% | — | — | — |
| debug | 2 | 0.0% | 0.0% | 4.0 | 50.0% | — | — | — |
| diff | 12 | 50.0% | 50.0% | 4.5 | 91.7% | — | — | — |
| ecdsa-sig-formatter | 1 | 0.0% | 0.0% | 2.0 | 100.0% | — | — | — |
| es5-ext | 30 | 40.0% | 40.0% | 6.3 | 100.0% | — | — | — |
| exponential-backoff | 6 | 83.3% | 83.3% | 10.6667 | 83.3% | — | — | — |
| http-proxy | 6 | 33.3% | 33.3% | 6.5 | 83.3% | — | — | — |
| https-proxy-agent | 2 | 50.0% | 50.0% | 3.5 | 100.0% | — | — | — |
| json-source-map | 2 | 50.0% | 50.0% | 17.5 | 100.0% | — | — | — |
| jsonwebtoken | 5 | 0.0% | 0.0% | 6.8 | 80.0% | — | — | — |
| libnpmexec | 3 | 33.3% | 33.3% | 3.3333 | 100.0% | — | — | — |
| minimatch | 3 | 33.3% | 33.3% | 4.0 | 100.0% | — | — | — |
| minimist | 11 | 72.7% | 72.7% | 8.4545 | 90.9% | — | — | — |
| minipass-fetch | 6 | 50.0% | 50.0% | 4.3333 | 83.3% | — | — | — |
| nearley | 8 | 37.5% | 37.5% | 3.375 | 87.5% | — | — | — |
| node-gyp-build | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| postcss-selector-parser | 17 | 41.2% | 41.2% | 5.9412 | 88.2% | — | — | — |
| pump | 2 | 50.0% | 50.0% | 18.5 | 100.0% | — | — | — |
| readable-stream | 12 | 41.7% | 41.7% | 13.8333 | 75.0% | — | — | — |
| requires-port | 2 | 50.0% | 50.0% | 4.0 | 100.0% | — | — | — |
| rxjs | 30 | 66.7% | 66.7% | 9.9 | 86.7% | — | — | — |
| selfsigned | 2 | 50.0% | 50.0% | 7.5 | 100.0% | — | — | — |
| shell-quote | 1 | 100.0% | 100.0% | 12.0 | 100.0% | — | — | — |
| tar-stream | 2 | 50.0% | 50.0% | 2.5 | 100.0% | — | — | — |
| type | 30 | 60.0% | 60.0% | 9.8333 | 100.0% | — | — | — |
| uglify-js | 4 | 50.0% | 50.0% | 9.25 | 100.0% | — | — | — |
| undici | 30 | 43.3% | 43.3% | 13.3 | 80.0% | — | — | — |
| valid-url | 3 | 66.7% | 66.7% | 7.0 | 100.0% | — | — | — |
| yaml | 30 | 36.7% | 36.7% | 13.2667 | 90.0% | — | — | — |
| yargs | 2 | 100.0% | 100.0% | 14.5 | 100.0% | — | — | — |
| zod-to-json-schema | 15 | 46.7% | 46.7% | 12.7333 | 93.3% | — | — | — |

