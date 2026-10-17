# Baseline zero-shot FIM `sft_v2d-final-generic-js`

- Spec: `docs/fase-02-professor.md` §3.2 · split `eval` · n=500 (de 500), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-7B` · carga `nf4` · SHA-256 do modelo `74ac207b4deacd56…` (revisao `0396a76181e127dfc13e5c5ec48a8cee09938b02`)
- Dataset: `data\replay\v1\generic_js_fim_eval.jsonl` SHA-256 `8827d1e4078a9f21…` · amostra `fbbfc5471d4a0a7b…`
- Contrato `fim-raw` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 823.5 s totais (796.5 s de geracao) · VRAM pico 6.41 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo —

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 70.6% | 69.8% | — |
| Normalized match (FIM acc.) | 71.0% | 70.0% | — |
| Prefix match | 71.6% | 72.2% | — |
| Syntax valid | 97.2% | 91.4% | 100.0% |
| Mongo valid (estatico) | — | — | — |
| Exec-valid | — | — | — |
| Campos inexistentes | — | — | — |
| Chars aceitos (media) | 17.868 | 16.776 | — |
| APT (tokens) | 4.626 | 4.378 | — |

Distribuicao do estagio de validacao (pred): {}
Paradas: {'eos': 491, 'max': 9}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 39 | 87.2% | 87.2% | 15.8974 | 97.4% | — | — | — |
| member | 101 | 70.3% | 70.3% | 10.0198 | 98.0% | — | — | — |
| multiline | 43 | 30.2% | 32.6% | 28.7907 | 95.3% | — | — | — |
| partial | 55 | 87.3% | 87.3% | 8.4545 | 100.0% | — | — | — |
| rest | 103 | 63.1% | 64.1% | 27.8252 | 94.2% | — | — | — |
| stmt_start | 55 | 67.3% | 67.3% | 29.6364 | 96.4% | — | — | — |
| value | 104 | 81.7% | 81.7% | 10.6058 | 99.0% | — | — | — |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| @azure/core-client | 9 | 77.8% | 77.8% | 19.4444 | 100.0% | — | — | — |
| @azure/core-util | 6 | 66.7% | 66.7% | 15.5 | 83.3% | — | — | — |
| @babel/core | 30 | 70.0% | 70.0% | 21.9 | 100.0% | — | — | — |
| @babel/helper-builder-binary-assignment-operator-visitor | 2 | 50.0% | 50.0% | 8.0 | 50.0% | — | — | — |
| @babel/helper-compilation-targets | 4 | 75.0% | 75.0% | 39.75 | 75.0% | — | — | — |
| @babel/helper-module-imports | 3 | 66.7% | 66.7% | 11.3333 | 100.0% | — | — | — |
| @babel/helper-module-transforms | 4 | 100.0% | 100.0% | 20.75 | 100.0% | — | — | — |
| @babel/plugin-transform-block-scoping | 3 | 33.3% | 33.3% | 5.6667 | 100.0% | — | — | — |
| @babel/template | 4 | 25.0% | 25.0% | 0.75 | 100.0% | — | — | — |
| @babel/types | 30 | 70.0% | 70.0% | 14.2 | 96.7% | — | — | — |
| @grpc/proto-loader | 2 | 100.0% | 100.0% | 28.5 | 100.0% | — | — | — |
| @hapi/hoek | 12 | 50.0% | 50.0% | 8.4167 | 100.0% | — | — | — |
| @npmcli/metavuln-calculator | 2 | 50.0% | 50.0% | 14.5 | 100.0% | — | — | — |
| @npmcli/promise-spawn | 2 | 100.0% | 100.0% | 3.5 | 100.0% | — | — | — |
| @pkgjs/parseargs | 4 | 100.0% | 100.0% | 34.25 | 100.0% | — | — | — |
| @protobufjs/float | 2 | 0.0% | 50.0% | 24.5 | 100.0% | — | — | — |
| @sigstore/bundle | 4 | 100.0% | 100.0% | 38.5 | 100.0% | — | — | — |
| @sigstore/tuf | 5 | 80.0% | 80.0% | 5.0 | 100.0% | — | — | — |
| @tufjs/models | 14 | 78.6% | 78.6% | 14.7857 | 92.9% | — | — | — |
| adm-zip | 7 | 71.4% | 71.4% | 8.0 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs2 | 3 | 100.0% | 100.0% | 29.6667 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs3 | 4 | 75.0% | 75.0% | 11.0 | 100.0% | — | — | — |
| browserslist | 4 | 100.0% | 100.0% | 7.5 | 100.0% | — | — | — |
| cacache | 12 | 66.7% | 66.7% | 13.8333 | 100.0% | — | — | — |
| cli-progress | 8 | 87.5% | 87.5% | 10.5 | 100.0% | — | — | — |
| cmd-shim | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| commander | 4 | 100.0% | 100.0% | 14.25 | 100.0% | — | — | — |
| core-js-compat | 3 | 100.0% | 100.0% | 16.0 | 100.0% | — | — | — |
| d | 2 | 100.0% | 100.0% | 3.0 | 100.0% | — | — | — |
| date-fns | 30 | 90.0% | 93.3% | 25.7667 | 100.0% | — | — | — |
| debug | 2 | 100.0% | 100.0% | 19.5 | 100.0% | — | — | — |
| diff | 12 | 66.7% | 66.7% | 10.6667 | 91.7% | — | — | — |
| ecdsa-sig-formatter | 1 | 100.0% | 100.0% | 25.0 | 100.0% | — | — | — |
| es5-ext | 30 | 60.0% | 60.0% | 10.8333 | 100.0% | — | — | — |
| exponential-backoff | 6 | 100.0% | 100.0% | 28.6667 | 100.0% | — | — | — |
| http-proxy | 6 | 66.7% | 66.7% | 15.8333 | 100.0% | — | — | — |
| https-proxy-agent | 2 | 50.0% | 50.0% | 3.5 | 100.0% | — | — | — |
| json-source-map | 2 | 50.0% | 50.0% | 16.5 | 100.0% | — | — | — |
| jsonwebtoken | 5 | 80.0% | 80.0% | 14.4 | 100.0% | — | — | — |
| libnpmexec | 3 | 66.7% | 66.7% | 5.6667 | 100.0% | — | — | — |
| minimatch | 3 | 33.3% | 33.3% | 4.3333 | 100.0% | — | — | — |
| minimist | 11 | 81.8% | 81.8% | 8.6364 | 100.0% | — | — | — |
| minipass-fetch | 6 | 83.3% | 83.3% | 9.0 | 83.3% | — | — | — |
| nearley | 8 | 62.5% | 62.5% | 8.5 | 100.0% | — | — | — |
| node-gyp-build | 1 | 0.0% | 0.0% | 54.0 | 100.0% | — | — | — |
| postcss-selector-parser | 17 | 64.7% | 64.7% | 21.5882 | 100.0% | — | — | — |
| pump | 2 | 50.0% | 50.0% | 18.5 | 100.0% | — | — | — |
| readable-stream | 12 | 75.0% | 75.0% | 64.75 | 83.3% | — | — | — |
| requires-port | 2 | 100.0% | 100.0% | 16.0 | 100.0% | — | — | — |
| rxjs | 30 | 80.0% | 80.0% | 28.2 | 93.3% | — | — | — |
| selfsigned | 2 | 100.0% | 100.0% | 15.0 | 100.0% | — | — | — |
| shell-quote | 1 | 100.0% | 100.0% | 12.0 | 100.0% | — | — | — |
| tar-stream | 2 | 100.0% | 100.0% | 12.0 | 100.0% | — | — | — |
| type | 30 | 66.7% | 66.7% | 11.9667 | 96.7% | — | — | — |
| uglify-js | 4 | 75.0% | 75.0% | 14.25 | 100.0% | — | — | — |
| undici | 30 | 50.0% | 50.0% | 20.1333 | 96.7% | — | — | — |
| valid-url | 3 | 66.7% | 66.7% | 7.0 | 100.0% | — | — | — |
| yaml | 30 | 63.3% | 63.3% | 18.2 | 96.7% | — | — | — |
| yargs | 2 | 100.0% | 100.0% | 14.5 | 100.0% | — | — | — |
| zod-to-json-schema | 15 | 53.3% | 53.3% | 14.2667 | 100.0% | — | — | — |

