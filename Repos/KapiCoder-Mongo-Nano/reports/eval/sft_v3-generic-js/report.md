# Baseline zero-shot FIM `sft_v3-generic-js`

- Spec: `docs/fase-02-professor.md` §3.2 · split `eval` · n=500 (de 500), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Nano\merged\sft_v3` · carga `bf16` · SHA-256 do modelo `e30c625c12e24dfd…` (revisao `None`)
- Dataset: `data\replay\v1\generic_js_fim_eval.jsonl` SHA-256 `8827d1e4078a9f21…` · amostra `fbbfc5471d4a0a7b…`
- Contrato `fim-raw` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 73.6 s totais (65.5 s de geracao) · VRAM pico 1.1 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo —

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 52.6% | 52.6% | — |
| Normalized match (FIM acc.) | 53.0% | 53.0% | — |
| Prefix match | 55.4% | 55.6% | — |
| Syntax valid | 92.2% | 87.2% | 100.0% |
| Mongo valid (estatico) | — | — | — |
| Exec-valid | — | — | — |
| Campos inexistentes | — | — | — |
| Chars aceitos (media) | 10.61 | 10.596 | — |
| APT (tokens) | 2.79 | 2.786 | — |

Distribuicao do estagio de validacao (pred): {}
Paradas: {'eos': 487, 'max': 13}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 39 | 61.5% | 61.5% | 9.1795 | 97.4% | — | — | — |
| member | 101 | 54.5% | 55.5% | 6.396 | 95.0% | — | — | — |
| multiline | 43 | 18.6% | 18.6% | 20.0465 | 93.0% | — | — | — |
| partial | 55 | 70.9% | 70.9% | 5.6545 | 94.5% | — | — | — |
| rest | 103 | 50.5% | 51.5% | 13.0097 | 86.4% | — | — | — |
| stmt_start | 55 | 38.2% | 38.2% | 18.9818 | 89.1% | — | — | — |
| value | 104 | 61.5% | 61.5% | 7.1538 | 93.3% | — | — | — |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| @azure/core-client | 9 | 66.7% | 66.7% | 16.6667 | 100.0% | — | — | — |
| @azure/core-util | 6 | 33.3% | 33.3% | 5.3333 | 83.3% | — | — | — |
| @babel/core | 30 | 50.0% | 50.0% | 13.6667 | 100.0% | — | — | — |
| @babel/helper-builder-binary-assignment-operator-visitor | 2 | 50.0% | 50.0% | 8.0 | 50.0% | — | — | — |
| @babel/helper-compilation-targets | 4 | 50.0% | 50.0% | 39.25 | 75.0% | — | — | — |
| @babel/helper-module-imports | 3 | 0.0% | 0.0% | 2.6667 | 100.0% | — | — | — |
| @babel/helper-module-transforms | 4 | 75.0% | 75.0% | 16.25 | 100.0% | — | — | — |
| @babel/plugin-transform-block-scoping | 3 | 33.3% | 33.3% | 5.6667 | 100.0% | — | — | — |
| @babel/template | 4 | 25.0% | 25.0% | 0.75 | 100.0% | — | — | — |
| @babel/types | 30 | 66.7% | 66.7% | 11.2 | 93.3% | — | — | — |
| @grpc/proto-loader | 2 | 50.0% | 50.0% | 2.0 | 50.0% | — | — | — |
| @hapi/hoek | 12 | 50.0% | 50.0% | 7.5 | 100.0% | — | — | — |
| @npmcli/metavuln-calculator | 2 | 50.0% | 50.0% | 6.5 | 100.0% | — | — | — |
| @npmcli/promise-spawn | 2 | 100.0% | 100.0% | 3.5 | 100.0% | — | — | — |
| @pkgjs/parseargs | 4 | 50.0% | 50.0% | 11.25 | 75.0% | — | — | — |
| @protobufjs/float | 2 | 0.0% | 0.0% | 15.0 | 50.0% | — | — | — |
| @sigstore/bundle | 4 | 75.0% | 75.0% | 27.25 | 75.0% | — | — | — |
| @sigstore/tuf | 5 | 80.0% | 80.0% | 5.0 | 100.0% | — | — | — |
| @tufjs/models | 14 | 64.3% | 64.3% | 10.5714 | 92.9% | — | — | — |
| adm-zip | 7 | 42.9% | 42.9% | 4.8571 | 85.7% | — | — | — |
| babel-plugin-polyfill-corejs2 | 3 | 100.0% | 100.0% | 29.6667 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs3 | 4 | 25.0% | 25.0% | 5.5 | 100.0% | — | — | — |
| browserslist | 4 | 50.0% | 50.0% | 4.5 | 100.0% | — | — | — |
| cacache | 12 | 50.0% | 50.0% | 6.9167 | 91.7% | — | — | — |
| cli-progress | 8 | 50.0% | 50.0% | 7.375 | 100.0% | — | — | — |
| cmd-shim | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| commander | 4 | 75.0% | 75.0% | 7.5 | 100.0% | — | — | — |
| core-js-compat | 3 | 66.7% | 66.7% | 5.6667 | 100.0% | — | — | — |
| d | 2 | 100.0% | 100.0% | 3.0 | 100.0% | — | — | — |
| date-fns | 30 | 63.3% | 70.0% | 16.2333 | 96.7% | — | — | — |
| debug | 2 | 0.0% | 0.0% | 4.0 | 100.0% | — | — | — |
| diff | 12 | 66.7% | 66.7% | 8.0 | 91.7% | — | — | — |
| ecdsa-sig-formatter | 1 | 0.0% | 0.0% | 20.0 | 100.0% | — | — | — |
| es5-ext | 30 | 43.3% | 43.3% | 5.8333 | 96.7% | — | — | — |
| exponential-backoff | 6 | 33.3% | 33.3% | 13.6667 | 100.0% | — | — | — |
| http-proxy | 6 | 50.0% | 50.0% | 8.6667 | 83.3% | — | — | — |
| https-proxy-agent | 2 | 50.0% | 50.0% | 3.5 | 50.0% | — | — | — |
| json-source-map | 2 | 50.0% | 50.0% | 16.5 | 100.0% | — | — | — |
| jsonwebtoken | 5 | 20.0% | 20.0% | 7.2 | 80.0% | — | — | — |
| libnpmexec | 3 | 66.7% | 66.7% | 2.3333 | 100.0% | — | — | — |
| minimatch | 3 | 33.3% | 33.3% | 4.0 | 100.0% | — | — | — |
| minimist | 11 | 72.7% | 72.7% | 7.4545 | 100.0% | — | — | — |
| minipass-fetch | 6 | 33.3% | 33.3% | 5.1667 | 83.3% | — | — | — |
| nearley | 8 | 50.0% | 50.0% | 4.625 | 87.5% | — | — | — |
| node-gyp-build | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| postcss-selector-parser | 17 | 52.9% | 52.9% | 8.4118 | 94.1% | — | — | — |
| pump | 2 | 50.0% | 50.0% | 18.5 | 100.0% | — | — | — |
| readable-stream | 12 | 50.0% | 50.0% | 13.5 | 83.3% | — | — | — |
| requires-port | 2 | 50.0% | 50.0% | 4.0 | 100.0% | — | — | — |
| rxjs | 30 | 66.7% | 66.7% | 10.0667 | 86.7% | — | — | — |
| selfsigned | 2 | 50.0% | 50.0% | 7.0 | 100.0% | — | — | — |
| shell-quote | 1 | 100.0% | 100.0% | 12.0 | 100.0% | — | — | — |
| tar-stream | 2 | 50.0% | 50.0% | 2.5 | 100.0% | — | — | — |
| type | 30 | 60.0% | 60.0% | 10.1667 | 86.7% | — | — | — |
| uglify-js | 4 | 50.0% | 50.0% | 11.0 | 100.0% | — | — | — |
| undici | 30 | 43.3% | 43.3% | 13.5 | 83.3% | — | — | — |
| valid-url | 3 | 66.7% | 66.7% | 7.0 | 100.0% | — | — | — |
| yaml | 30 | 30.0% | 30.0% | 13.6 | 93.3% | — | — | — |
| yargs | 2 | 50.0% | 50.0% | 6.0 | 100.0% | — | — | — |
| zod-to-json-schema | 15 | 53.3% | 53.3% | 15.9333 | 93.3% | — | — | — |

