# Baseline zero-shot FIM `Qwen2.5-Coder-7B-generic-js`

- Spec: `docs/fase-02-professor.md` §3.2 · split `eval` · n=500 (de 500), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-7B` · carga `nf4` · SHA-256 do modelo `74ac207b4deacd56…` (revisao `0396a76181e127dfc13e5c5ec48a8cee09938b02`)
- Dataset: `data\replay\v1\generic_js_fim_eval.jsonl` SHA-256 `8827d1e4078a9f21…` · amostra `fbbfc5471d4a0a7b…`
- Contrato `fim-raw` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 446.8 s totais (425.9 s de geracao) · VRAM pico 5.68 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo —

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 55.6% | 55.2% | — |
| Normalized match (FIM acc.) | 55.8% | 55.2% | — |
| Prefix match | 70.8% | 71.2% | — |
| Syntax valid | 80.4% | 74.2% | 100.0% |
| Mongo valid (estatico) | — | — | — |
| Exec-valid | — | — | — |
| Campos inexistentes | — | — | — |
| Chars aceitos (media) | 17.61 | 16.626 | — |
| APT (tokens) | 4.492 | 4.282 | — |

Distribuicao do estagio de validacao (pred): {}
Paradas: {'max': 81, 'eos': 418, 'suffix': 1}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 39 | 69.2% | 69.2% | 15.8974 | 87.2% | — | — | — |
| member | 101 | 53.5% | 53.5% | 9.703 | 78.2% | — | — | — |
| multiline | 43 | 20.9% | 23.3% | 31.3721 | 74.4% | — | — | — |
| partial | 55 | 60.0% | 60.0% | 8.2727 | 80.0% | — | — | — |
| rest | 103 | 47.6% | 47.6% | 26.5243 | 76.7% | — | — | — |
| stmt_start | 55 | 60.0% | 60.0% | 28.0727 | 80.0% | — | — | — |
| value | 104 | 70.2% | 70.2% | 10.8173 | 86.5% | — | — | — |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| @azure/core-client | 9 | 77.8% | 77.8% | 18.6667 | 88.9% | — | — | — |
| @azure/core-util | 6 | 33.3% | 33.3% | 14.0 | 66.7% | — | — | — |
| @babel/core | 30 | 50.0% | 50.0% | 20.7 | 86.7% | — | — | — |
| @babel/helper-builder-binary-assignment-operator-visitor | 2 | 0.0% | 0.0% | 8.0 | 50.0% | — | — | — |
| @babel/helper-compilation-targets | 4 | 50.0% | 50.0% | 39.75 | 75.0% | — | — | — |
| @babel/helper-module-imports | 3 | 33.3% | 33.3% | 7.6667 | 100.0% | — | — | — |
| @babel/helper-module-transforms | 4 | 75.0% | 75.0% | 20.75 | 100.0% | — | — | — |
| @babel/plugin-transform-block-scoping | 3 | 0.0% | 0.0% | 5.6667 | 33.3% | — | — | — |
| @babel/template | 4 | 25.0% | 25.0% | 0.75 | 75.0% | — | — | — |
| @babel/types | 30 | 66.7% | 66.7% | 11.6333 | 90.0% | — | — | — |
| @grpc/proto-loader | 2 | 50.0% | 50.0% | 28.5 | 50.0% | — | — | — |
| @hapi/hoek | 12 | 41.7% | 41.7% | 9.8333 | 83.3% | — | — | — |
| @npmcli/metavuln-calculator | 2 | 50.0% | 50.0% | 14.5 | 100.0% | — | — | — |
| @npmcli/promise-spawn | 2 | 100.0% | 100.0% | 3.5 | 100.0% | — | — | — |
| @pkgjs/parseargs | 4 | 25.0% | 25.0% | 25.75 | 75.0% | — | — | — |
| @protobufjs/float | 2 | 0.0% | 50.0% | 25.0 | 100.0% | — | — | — |
| @sigstore/bundle | 4 | 75.0% | 75.0% | 38.25 | 100.0% | — | — | — |
| @sigstore/tuf | 5 | 60.0% | 60.0% | 5.0 | 60.0% | — | — | — |
| @tufjs/models | 14 | 71.4% | 71.4% | 13.7857 | 85.7% | — | — | — |
| adm-zip | 7 | 57.1% | 57.1% | 7.8571 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs2 | 3 | 100.0% | 100.0% | 29.6667 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs3 | 4 | 25.0% | 25.0% | 11.0 | 75.0% | — | — | — |
| browserslist | 4 | 50.0% | 50.0% | 7.25 | 75.0% | — | — | — |
| cacache | 12 | 58.3% | 58.3% | 14.1667 | 83.3% | — | — | — |
| cli-progress | 8 | 75.0% | 75.0% | 10.5 | 87.5% | — | — | — |
| cmd-shim | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| commander | 4 | 100.0% | 100.0% | 14.25 | 100.0% | — | — | — |
| core-js-compat | 3 | 66.7% | 66.7% | 16.0 | 66.7% | — | — | — |
| d | 2 | 100.0% | 100.0% | 3.0 | 100.0% | — | — | — |
| date-fns | 30 | 93.3% | 93.3% | 26.6 | 100.0% | — | — | — |
| debug | 2 | 50.0% | 50.0% | 19.5 | 50.0% | — | — | — |
| diff | 12 | 50.0% | 50.0% | 10.9167 | 83.3% | — | — | — |
| ecdsa-sig-formatter | 1 | 100.0% | 100.0% | 25.0 | 100.0% | — | — | — |
| es5-ext | 30 | 33.3% | 33.3% | 10.5 | 66.7% | — | — | — |
| exponential-backoff | 6 | 66.7% | 66.7% | 28.6667 | 83.3% | — | — | — |
| http-proxy | 6 | 83.3% | 83.3% | 23.6667 | 100.0% | — | — | — |
| https-proxy-agent | 2 | 0.0% | 0.0% | 3.5 | 100.0% | — | — | — |
| json-source-map | 2 | 50.0% | 50.0% | 16.5 | 50.0% | — | — | — |
| jsonwebtoken | 5 | 40.0% | 40.0% | 14.4 | 40.0% | — | — | — |
| libnpmexec | 3 | 33.3% | 33.3% | 2.6667 | 66.7% | — | — | — |
| minimatch | 3 | 33.3% | 33.3% | 4.3333 | 66.7% | — | — | — |
| minimist | 11 | 63.6% | 63.6% | 8.0 | 72.7% | — | — | — |
| minipass-fetch | 6 | 33.3% | 33.3% | 5.5 | 66.7% | — | — | — |
| nearley | 8 | 50.0% | 50.0% | 8.0 | 75.0% | — | — | — |
| node-gyp-build | 1 | 0.0% | 0.0% | 46.0 | 0.0% | — | — | — |
| postcss-selector-parser | 17 | 58.8% | 58.8% | 13.2353 | 82.3% | — | — | — |
| pump | 2 | 50.0% | 50.0% | 18.5 | 100.0% | — | — | — |
| readable-stream | 12 | 58.3% | 58.3% | 64.75 | 66.7% | — | — | — |
| requires-port | 2 | 100.0% | 100.0% | 16.0 | 100.0% | — | — | — |
| rxjs | 30 | 73.3% | 73.3% | 29.0 | 83.3% | — | — | — |
| selfsigned | 2 | 50.0% | 50.0% | 15.0 | 50.0% | — | — | — |
| shell-quote | 1 | 100.0% | 100.0% | 12.0 | 100.0% | — | — | — |
| tar-stream | 2 | 100.0% | 100.0% | 12.0 | 100.0% | — | — | — |
| type | 30 | 60.0% | 60.0% | 9.8333 | 86.7% | — | — | — |
| uglify-js | 4 | 25.0% | 25.0% | 14.25 | 50.0% | — | — | — |
| undici | 30 | 30.0% | 30.0% | 17.9667 | 73.3% | — | — | — |
| valid-url | 3 | 0.0% | 0.0% | 9.3333 | 0.0% | — | — | — |
| yaml | 30 | 43.3% | 43.3% | 25.7333 | 73.3% | — | — | — |
| yargs | 2 | 50.0% | 50.0% | 6.0 | 50.0% | — | — | — |
| zod-to-json-schema | 15 | 60.0% | 60.0% | 17.9333 | 86.7% | — | — | — |

