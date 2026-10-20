# Baseline zero-shot FIM `sft_distill_v1-generic-js`

- Spec: `docs/fase-02-professor.md` §3.2 · split `eval` · n=500 (de 500), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Small\merged\sft_distill_v1` · carga `bf16` · SHA-256 do modelo `f7f71e24b606e26d…` (revisao `None`)
- Dataset: `data\replay\v1\generic_js_fim_eval.jsonl` SHA-256 `8827d1e4078a9f21…` · amostra `fbbfc5471d4a0a7b…`
- Contrato `fim-raw` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 117.2 s totais (106.6 s de geracao) · VRAM pico 3.13 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo —

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 63.2% | 63.0% | — |
| Normalized match (FIM acc.) | 63.2% | 63.0% | — |
| Prefix match | 66.2% | 66.2% | — |
| Syntax valid | 96.2% | 91.8% | 100.0% |
| Mongo valid (estatico) | — | — | — |
| Exec-valid | — | — | — |
| Campos inexistentes | — | — | — |
| Chars aceitos (media) | 13.602 | 13.574 | — |
| APT (tokens) | 3.576 | 3.566 | — |

Distribuicao do estagio de validacao (pred): {}
Paradas: {'eos': 494, 'max': 6}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 39 | 74.4% | 74.4% | 12.7949 | 100.0% | — | — | — |
| member | 101 | 62.4% | 62.4% | 8.8317 | 97.0% | — | — | — |
| multiline | 43 | 20.9% | 20.9% | 20.2093 | 95.3% | — | — | — |
| partial | 55 | 81.8% | 81.8% | 7.9636 | 100.0% | — | — | — |
| rest | 103 | 57.3% | 57.3% | 15.8641 | 91.3% | — | — | — |
| stmt_start | 55 | 58.2% | 58.2% | 26.9091 | 92.7% | — | — | — |
| value | 104 | 76.0% | 76.0% | 9.5096 | 99.0% | — | — | — |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| @azure/core-client | 9 | 77.8% | 77.8% | 19.4444 | 100.0% | — | — | — |
| @azure/core-util | 6 | 50.0% | 50.0% | 8.0 | 83.3% | — | — | — |
| @babel/core | 30 | 70.0% | 70.0% | 16.1333 | 96.7% | — | — | — |
| @babel/helper-builder-binary-assignment-operator-visitor | 2 | 50.0% | 50.0% | 8.0 | 100.0% | — | — | — |
| @babel/helper-compilation-targets | 4 | 75.0% | 75.0% | 39.75 | 75.0% | — | — | — |
| @babel/helper-module-imports | 3 | 66.7% | 66.7% | 11.3333 | 100.0% | — | — | — |
| @babel/helper-module-transforms | 4 | 100.0% | 100.0% | 20.75 | 100.0% | — | — | — |
| @babel/plugin-transform-block-scoping | 3 | 33.3% | 33.3% | 5.6667 | 100.0% | — | — | — |
| @babel/template | 4 | 25.0% | 25.0% | 0.75 | 100.0% | — | — | — |
| @babel/types | 30 | 70.0% | 70.0% | 12.1333 | 100.0% | — | — | — |
| @grpc/proto-loader | 2 | 100.0% | 100.0% | 28.5 | 100.0% | — | — | — |
| @hapi/hoek | 12 | 50.0% | 50.0% | 8.25 | 100.0% | — | — | — |
| @npmcli/metavuln-calculator | 2 | 0.0% | 0.0% | 6.5 | 50.0% | — | — | — |
| @npmcli/promise-spawn | 2 | 100.0% | 100.0% | 3.5 | 100.0% | — | — | — |
| @pkgjs/parseargs | 4 | 75.0% | 75.0% | 25.75 | 100.0% | — | — | — |
| @protobufjs/float | 2 | 0.0% | 0.0% | 15.0 | 50.0% | — | — | — |
| @sigstore/bundle | 4 | 100.0% | 100.0% | 38.5 | 100.0% | — | — | — |
| @sigstore/tuf | 5 | 80.0% | 80.0% | 5.8 | 100.0% | — | — | — |
| @tufjs/models | 14 | 71.4% | 71.4% | 11.7143 | 92.9% | — | — | — |
| adm-zip | 7 | 71.4% | 71.4% | 7.8571 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs2 | 3 | 66.7% | 66.7% | 21.6667 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs3 | 4 | 25.0% | 25.0% | 9.25 | 50.0% | — | — | — |
| browserslist | 4 | 50.0% | 50.0% | 6.0 | 100.0% | — | — | — |
| cacache | 12 | 66.7% | 66.7% | 12.75 | 100.0% | — | — | — |
| cli-progress | 8 | 87.5% | 87.5% | 10.5 | 100.0% | — | — | — |
| cmd-shim | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| commander | 4 | 100.0% | 100.0% | 14.25 | 100.0% | — | — | — |
| core-js-compat | 3 | 100.0% | 100.0% | 16.0 | 100.0% | — | — | — |
| d | 2 | 100.0% | 100.0% | 3.0 | 100.0% | — | — | — |
| date-fns | 30 | 83.3% | 83.3% | 19.7 | 100.0% | — | — | — |
| debug | 2 | 50.0% | 50.0% | 17.0 | 100.0% | — | — | — |
| diff | 12 | 66.7% | 66.7% | 8.5 | 91.7% | — | — | — |
| ecdsa-sig-formatter | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| es5-ext | 30 | 53.3% | 53.3% | 10.1667 | 96.7% | — | — | — |
| exponential-backoff | 6 | 83.3% | 83.3% | 27.6667 | 100.0% | — | — | — |
| http-proxy | 6 | 66.7% | 66.7% | 16.0 | 100.0% | — | — | — |
| https-proxy-agent | 2 | 50.0% | 50.0% | 3.5 | 50.0% | — | — | — |
| json-source-map | 2 | 50.0% | 50.0% | 16.5 | 100.0% | — | — | — |
| jsonwebtoken | 5 | 60.0% | 60.0% | 11.8 | 80.0% | — | — | — |
| libnpmexec | 3 | 66.7% | 66.7% | 2.6667 | 100.0% | — | — | — |
| minimatch | 3 | 33.3% | 33.3% | 2.6667 | 100.0% | — | — | — |
| minimist | 11 | 72.7% | 72.7% | 14.0909 | 100.0% | — | — | — |
| minipass-fetch | 6 | 50.0% | 50.0% | 6.3333 | 83.3% | — | — | — |
| nearley | 8 | 50.0% | 50.0% | 7.25 | 100.0% | — | — | — |
| node-gyp-build | 1 | 0.0% | 0.0% | 10.0 | 100.0% | — | — | — |
| postcss-selector-parser | 17 | 58.8% | 58.8% | 12.7059 | 100.0% | — | — | — |
| pump | 2 | 50.0% | 50.0% | 18.5 | 100.0% | — | — | — |
| readable-stream | 12 | 58.3% | 58.3% | 23.6667 | 100.0% | — | — | — |
| requires-port | 2 | 50.0% | 50.0% | 4.0 | 100.0% | — | — | — |
| rxjs | 30 | 66.7% | 66.7% | 10.9667 | 93.3% | — | — | — |
| selfsigned | 2 | 50.0% | 50.0% | 13.0 | 100.0% | — | — | — |
| shell-quote | 1 | 100.0% | 100.0% | 12.0 | 100.0% | — | — | — |
| tar-stream | 2 | 100.0% | 100.0% | 12.0 | 100.0% | — | — | — |
| type | 30 | 70.0% | 70.0% | 11.9 | 100.0% | — | — | — |
| uglify-js | 4 | 75.0% | 75.0% | 12.75 | 100.0% | — | — | — |
| undici | 30 | 53.3% | 53.3% | 16.9333 | 93.3% | — | — | — |
| valid-url | 3 | 66.7% | 66.7% | 9.3333 | 100.0% | — | — | — |
| yaml | 30 | 36.7% | 36.7% | 14.3667 | 96.7% | — | — | — |
| yargs | 2 | 100.0% | 100.0% | 14.5 | 100.0% | — | — | — |
| zod-to-json-schema | 15 | 46.7% | 46.7% | 14.8667 | 93.3% | — | — | — |

