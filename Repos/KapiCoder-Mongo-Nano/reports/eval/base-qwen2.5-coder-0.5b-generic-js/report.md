# Baseline zero-shot FIM `base-qwen2.5-coder-0.5b-generic-js`

- Spec: `docs/fase-02-professor.md` §3.2 · split `eval` · n=500 (de 500), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-0.5B` · carga `bf16` · SHA-256 do modelo `96217dc3a23c0290…` (revisao `8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301`)
- Dataset: `data\replay\v1\generic_js_fim_eval.jsonl` SHA-256 `8827d1e4078a9f21…` · amostra `fbbfc5471d4a0a7b…`
- Contrato `fim-raw` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 306.7 s totais (299.1 s de geracao) · VRAM pico 1.1 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo —

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 33.2% | 33.2% | — |
| Normalized match (FIM acc.) | 34.0% | 34.0% | — |
| Prefix match | 55.6% | 55.8% | — |
| Syntax valid | 70.8% | 66.6% | 100.0% |
| Mongo valid (estatico) | — | — | — |
| Exec-valid | — | — | — |
| Campos inexistentes | — | — | — |
| Chars aceitos (media) | 11.63 | 11.544 | — |
| APT (tokens) | 3.05 | 3.022 | — |

Distribuicao do estagio de validacao (pred): {}
Paradas: {'max': 132, 'eos': 366, 'suffix': 2}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 39 | 35.9% | 35.9% | 9.0256 | 59.0% | — | — | — |
| member | 101 | 31.7% | 32.7% | 6.6436 | 73.3% | — | — | — |
| multiline | 43 | 9.3% | 14.0% | 20.3488 | 67.4% | — | — | — |
| partial | 55 | 49.1% | 49.1% | 7.2 | 74.6% | — | — | — |
| rest | 103 | 26.2% | 27.2% | 15.3204 | 63.1% | — | — | — |
| stmt_start | 55 | 34.5% | 34.5% | 17.7636 | 81.8% | — | — | — |
| value | 104 | 41.3% | 41.3% | 9.2885 | 74.0% | — | — | — |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| @azure/core-client | 9 | 55.6% | 55.6% | 15.1111 | 77.8% | — | — | — |
| @azure/core-util | 6 | 16.7% | 16.7% | 6.3333 | 83.3% | — | — | — |
| @babel/core | 30 | 23.3% | 23.3% | 13.3667 | 76.7% | — | — | — |
| @babel/helper-builder-binary-assignment-operator-visitor | 2 | 50.0% | 50.0% | 8.0 | 50.0% | — | — | — |
| @babel/helper-compilation-targets | 4 | 25.0% | 25.0% | 33.25 | 75.0% | — | — | — |
| @babel/helper-module-imports | 3 | 33.3% | 33.3% | 7.3333 | 100.0% | — | — | — |
| @babel/helper-module-transforms | 4 | 50.0% | 50.0% | 19.75 | 100.0% | — | — | — |
| @babel/plugin-transform-block-scoping | 3 | 33.3% | 33.3% | 5.0 | 66.7% | — | — | — |
| @babel/template | 4 | 25.0% | 25.0% | 1.75 | 25.0% | — | — | — |
| @babel/types | 30 | 36.7% | 36.7% | 11.2333 | 60.0% | — | — | — |
| @grpc/proto-loader | 2 | 50.0% | 50.0% | 28.5 | 100.0% | — | — | — |
| @hapi/hoek | 12 | 33.3% | 33.3% | 8.5833 | 75.0% | — | — | — |
| @npmcli/metavuln-calculator | 2 | 0.0% | 0.0% | 7.0 | 50.0% | — | — | — |
| @npmcli/promise-spawn | 2 | 0.0% | 0.0% | 3.5 | 0.0% | — | — | — |
| @pkgjs/parseargs | 4 | 25.0% | 25.0% | 11.25 | 100.0% | — | — | — |
| @protobufjs/float | 2 | 0.0% | 0.0% | 13.0 | 100.0% | — | — | — |
| @sigstore/bundle | 4 | 50.0% | 50.0% | 36.25 | 75.0% | — | — | — |
| @sigstore/tuf | 5 | 60.0% | 60.0% | 3.0 | 100.0% | — | — | — |
| @tufjs/models | 14 | 50.0% | 50.0% | 11.7143 | 71.4% | — | — | — |
| adm-zip | 7 | 42.9% | 42.9% | 6.0 | 71.4% | — | — | — |
| babel-plugin-polyfill-corejs2 | 3 | 66.7% | 66.7% | 24.0 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs3 | 4 | 0.0% | 0.0% | 6.0 | 25.0% | — | — | — |
| browserslist | 4 | 0.0% | 0.0% | 3.25 | 100.0% | — | — | — |
| cacache | 12 | 16.7% | 16.7% | 4.5 | 83.3% | — | — | — |
| cli-progress | 8 | 62.5% | 62.5% | 9.75 | 87.5% | — | — | — |
| cmd-shim | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| commander | 4 | 50.0% | 50.0% | 14.25 | 100.0% | — | — | — |
| core-js-compat | 3 | 33.3% | 33.3% | 4.3333 | 66.7% | — | — | — |
| d | 2 | 100.0% | 100.0% | 3.0 | 100.0% | — | — | — |
| date-fns | 30 | 43.3% | 56.7% | 15.9333 | 86.7% | — | — | — |
| debug | 2 | 0.0% | 0.0% | 4.0 | 50.0% | — | — | — |
| diff | 12 | 25.0% | 25.0% | 5.5 | 58.3% | — | — | — |
| ecdsa-sig-formatter | 1 | 0.0% | 0.0% | 0.0 | 0.0% | — | — | — |
| es5-ext | 30 | 13.3% | 13.3% | 8.1333 | 60.0% | — | — | — |
| exponential-backoff | 6 | 33.3% | 33.3% | 11.0 | 66.7% | — | — | — |
| http-proxy | 6 | 50.0% | 50.0% | 8.8333 | 83.3% | — | — | — |
| https-proxy-agent | 2 | 50.0% | 50.0% | 3.5 | 50.0% | — | — | — |
| json-source-map | 2 | 50.0% | 50.0% | 18.5 | 50.0% | — | — | — |
| jsonwebtoken | 5 | 20.0% | 20.0% | 7.2 | 80.0% | — | — | — |
| libnpmexec | 3 | 33.3% | 33.3% | 2.6667 | 66.7% | — | — | — |
| minimatch | 3 | 0.0% | 0.0% | 8.3333 | 100.0% | — | — | — |
| minimist | 11 | 27.3% | 27.3% | 7.8182 | 36.4% | — | — | — |
| minipass-fetch | 6 | 16.7% | 16.7% | 5.1667 | 50.0% | — | — | — |
| nearley | 8 | 25.0% | 25.0% | 2.25 | 87.5% | — | — | — |
| node-gyp-build | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| postcss-selector-parser | 17 | 41.2% | 41.2% | 10.2941 | 88.2% | — | — | — |
| pump | 2 | 50.0% | 50.0% | 18.5 | 50.0% | — | — | — |
| readable-stream | 12 | 25.0% | 25.0% | 26.25 | 50.0% | — | — | — |
| requires-port | 2 | 50.0% | 50.0% | 4.0 | 50.0% | — | — | — |
| rxjs | 30 | 60.0% | 60.0% | 12.2 | 80.0% | — | — | — |
| selfsigned | 2 | 50.0% | 50.0% | 7.0 | 100.0% | — | — | — |
| shell-quote | 1 | 0.0% | 0.0% | 0.0 | 0.0% | — | — | — |
| tar-stream | 2 | 50.0% | 50.0% | 2.5 | 50.0% | — | — | — |
| type | 30 | 30.0% | 30.0% | 8.7667 | 70.0% | — | — | — |
| uglify-js | 4 | 25.0% | 25.0% | 11.0 | 50.0% | — | — | — |
| undici | 30 | 33.3% | 33.3% | 18.6667 | 63.3% | — | — | — |
| valid-url | 3 | 0.0% | 0.0% | 9.3333 | 0.0% | — | — | — |
| yaml | 30 | 16.7% | 16.7% | 14.4 | 56.7% | — | — | — |
| yargs | 2 | 100.0% | 100.0% | 14.5 | 100.0% | — | — | — |
| zod-to-json-schema | 15 | 40.0% | 40.0% | 17.1333 | 93.3% | — | — | — |

