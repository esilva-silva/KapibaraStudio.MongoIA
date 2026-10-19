# Baseline zero-shot FIM `sft_distill_v1-generic-js`

- Spec: `docs/fase-02-professor.md` §3.2 · split `eval` · n=500 (de 500), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\merged\sft_distill_v1` · carga `bf16` · SHA-256 do modelo `4e3ec2d44a0423f9…` (revisao `None`)
- Dataset: `data\replay\v1\generic_js_fim_eval.jsonl` SHA-256 `8827d1e4078a9f21…` · amostra `fbbfc5471d4a0a7b…`
- Contrato `fim-raw` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 102.0 s totais (94.1 s de geracao) · VRAM pico 1.1 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo —

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 51.0% | 51.0% | — |
| Normalized match (FIM acc.) | 51.8% | 51.8% | — |
| Prefix match | 53.4% | 53.6% | — |
| Syntax valid | 93.2% | 88.2% | 100.0% |
| Mongo valid (estatico) | — | — | — |
| Exec-valid | — | — | — |
| Campos inexistentes | — | — | — |
| Chars aceitos (media) | 10.808 | 10.786 | — |
| APT (tokens) | 2.814 | 2.81 | — |

Distribuicao do estagio de validacao (pred): {}
Paradas: {'eos': 489, 'max': 11}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 39 | 59.0% | 59.0% | 9.2051 | 89.7% | — | — | — |
| member | 101 | 52.5% | 53.5% | 6.703 | 97.0% | — | — | — |
| multiline | 43 | 16.3% | 18.6% | 18.3023 | 95.3% | — | — | — |
| partial | 55 | 72.7% | 72.7% | 5.8 | 98.2% | — | — | — |
| rest | 103 | 48.5% | 50.5% | 14.5049 | 88.3% | — | — | — |
| stmt_start | 55 | 36.4% | 36.4% | 18.0182 | 89.1% | — | — | — |
| value | 104 | 59.6% | 59.6% | 7.4712 | 94.2% | — | — | — |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| @azure/core-client | 9 | 66.7% | 66.7% | 16.6667 | 100.0% | — | — | — |
| @azure/core-util | 6 | 33.3% | 33.3% | 5.0 | 66.7% | — | — | — |
| @babel/core | 30 | 56.7% | 56.7% | 16.6667 | 100.0% | — | — | — |
| @babel/helper-builder-binary-assignment-operator-visitor | 2 | 50.0% | 50.0% | 8.0 | 50.0% | — | — | — |
| @babel/helper-compilation-targets | 4 | 50.0% | 50.0% | 39.25 | 75.0% | — | — | — |
| @babel/helper-module-imports | 3 | 0.0% | 0.0% | 2.6667 | 100.0% | — | — | — |
| @babel/helper-module-transforms | 4 | 25.0% | 50.0% | 11.5 | 100.0% | — | — | — |
| @babel/plugin-transform-block-scoping | 3 | 33.3% | 33.3% | 5.0 | 100.0% | — | — | — |
| @babel/template | 4 | 25.0% | 25.0% | 0.75 | 100.0% | — | — | — |
| @babel/types | 30 | 46.7% | 50.0% | 9.6667 | 83.3% | — | — | — |
| @grpc/proto-loader | 2 | 50.0% | 50.0% | 28.5 | 100.0% | — | — | — |
| @hapi/hoek | 12 | 50.0% | 50.0% | 9.0 | 100.0% | — | — | — |
| @npmcli/metavuln-calculator | 2 | 50.0% | 50.0% | 5.5 | 100.0% | — | — | — |
| @npmcli/promise-spawn | 2 | 100.0% | 100.0% | 3.5 | 100.0% | — | — | — |
| @pkgjs/parseargs | 4 | 50.0% | 50.0% | 11.25 | 75.0% | — | — | — |
| @protobufjs/float | 2 | 0.0% | 0.0% | 20.5 | 50.0% | — | — | — |
| @sigstore/bundle | 4 | 75.0% | 75.0% | 27.25 | 75.0% | — | — | — |
| @sigstore/tuf | 5 | 80.0% | 80.0% | 5.0 | 100.0% | — | — | — |
| @tufjs/models | 14 | 64.3% | 64.3% | 10.4286 | 92.9% | — | — | — |
| adm-zip | 7 | 42.9% | 42.9% | 4.4286 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs2 | 3 | 100.0% | 100.0% | 29.6667 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs3 | 4 | 25.0% | 25.0% | 6.25 | 75.0% | — | — | — |
| browserslist | 4 | 75.0% | 75.0% | 6.0 | 100.0% | — | — | — |
| cacache | 12 | 41.7% | 41.7% | 6.6667 | 100.0% | — | — | — |
| cli-progress | 8 | 62.5% | 62.5% | 8.0 | 100.0% | — | — | — |
| cmd-shim | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| commander | 4 | 75.0% | 75.0% | 7.5 | 100.0% | — | — | — |
| core-js-compat | 3 | 66.7% | 66.7% | 5.6667 | 100.0% | — | — | — |
| d | 2 | 100.0% | 100.0% | 3.0 | 100.0% | — | — | — |
| date-fns | 30 | 56.7% | 63.3% | 13.3 | 100.0% | — | — | — |
| debug | 2 | 0.0% | 0.0% | 4.0 | 50.0% | — | — | — |
| diff | 12 | 58.3% | 58.3% | 6.0 | 83.3% | — | — | — |
| ecdsa-sig-formatter | 1 | 0.0% | 0.0% | 20.0 | 100.0% | — | — | — |
| es5-ext | 30 | 43.3% | 43.3% | 6.4667 | 100.0% | — | — | — |
| exponential-backoff | 6 | 50.0% | 50.0% | 8.6667 | 83.3% | — | — | — |
| http-proxy | 6 | 33.3% | 33.3% | 9.6667 | 83.3% | — | — | — |
| https-proxy-agent | 2 | 50.0% | 50.0% | 3.5 | 100.0% | — | — | — |
| json-source-map | 2 | 50.0% | 50.0% | 16.5 | 100.0% | — | — | — |
| jsonwebtoken | 5 | 0.0% | 0.0% | 2.6 | 80.0% | — | — | — |
| libnpmexec | 3 | 33.3% | 33.3% | 1.6667 | 100.0% | — | — | — |
| minimatch | 3 | 33.3% | 33.3% | 4.0 | 100.0% | — | — | — |
| minimist | 11 | 54.5% | 54.5% | 7.1818 | 100.0% | — | — | — |
| minipass-fetch | 6 | 33.3% | 33.3% | 4.3333 | 66.7% | — | — | — |
| nearley | 8 | 37.5% | 37.5% | 2.5 | 87.5% | — | — | — |
| node-gyp-build | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| postcss-selector-parser | 17 | 52.9% | 52.9% | 6.7647 | 94.1% | — | — | — |
| pump | 2 | 50.0% | 50.0% | 18.5 | 100.0% | — | — | — |
| readable-stream | 12 | 50.0% | 50.0% | 17.6667 | 83.3% | — | — | — |
| requires-port | 2 | 50.0% | 50.0% | 4.0 | 100.0% | — | — | — |
| rxjs | 30 | 66.7% | 66.7% | 9.8333 | 90.0% | — | — | — |
| selfsigned | 2 | 50.0% | 50.0% | 13.0 | 100.0% | — | — | — |
| shell-quote | 1 | 100.0% | 100.0% | 12.0 | 100.0% | — | — | — |
| tar-stream | 2 | 50.0% | 50.0% | 2.5 | 100.0% | — | — | — |
| type | 30 | 66.7% | 66.7% | 13.7667 | 96.7% | — | — | — |
| uglify-js | 4 | 50.0% | 50.0% | 11.0 | 100.0% | — | — | — |
| undici | 30 | 43.3% | 43.3% | 16.8333 | 93.3% | — | — | — |
| valid-url | 3 | 66.7% | 66.7% | 7.0 | 100.0% | — | — | — |
| yaml | 30 | 40.0% | 40.0% | 10.6 | 96.7% | — | — | — |
| yargs | 2 | 100.0% | 100.0% | 14.5 | 100.0% | — | — | — |
| zod-to-json-schema | 15 | 46.7% | 46.7% | 15.7333 | 93.3% | — | — | — |

