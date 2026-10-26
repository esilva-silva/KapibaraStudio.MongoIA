# Baseline zero-shot FIM `sft_v3-generic-js`

- Spec: `docs/fase-02-professor.md` §3.2 · split `eval` · n=500 (de 500), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\Repos\KapiCoder-Mongo-Small\merged\sft_v3` · carga `bf16` · SHA-256 do modelo `23e53b94514a68b1…` (revisao `None`)
- Dataset: `data\replay\v1\generic_js_fim_eval.jsonl` SHA-256 `8827d1e4078a9f21…` · amostra `fbbfc5471d4a0a7b…`
- Contrato `fim-raw` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 141.3 s totais (131.1 s de geracao) · VRAM pico 3.13 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo —

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 60.4% | 60.4% | — |
| Normalized match (FIM acc.) | 60.6% | 60.6% | — |
| Prefix match | 62.8% | 62.8% | — |
| Syntax valid | 95.8% | 91.0% | 100.0% |
| Mongo valid (estatico) | — | — | — |
| Exec-valid | — | — | — |
| Campos inexistentes | — | — | — |
| Chars aceitos (media) | 12.846 | 12.846 | — |
| APT (tokens) | 3.38 | 3.38 | — |

Distribuicao do estagio de validacao (pred): {}
Paradas: {'eos': 490, 'max': 10}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 39 | 74.4% | 74.4% | 12.3846 | 100.0% | — | — | — |
| member | 101 | 61.4% | 61.4% | 7.9406 | 95.0% | — | — | — |
| multiline | 43 | 18.6% | 18.6% | 20.6512 | 93.0% | — | — | — |
| partial | 55 | 80.0% | 80.0% | 7.6364 | 100.0% | — | — | — |
| rest | 103 | 54.4% | 55.3% | 15.7087 | 93.2% | — | — | — |
| stmt_start | 55 | 50.9% | 50.9% | 22.4545 | 94.5% | — | — | — |
| value | 104 | 72.1% | 72.1% | 9.3942 | 97.1% | — | — | — |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| @azure/core-client | 9 | 77.8% | 77.8% | 19.4444 | 100.0% | — | — | — |
| @azure/core-util | 6 | 50.0% | 50.0% | 7.1667 | 100.0% | — | — | — |
| @babel/core | 30 | 63.3% | 63.3% | 19.3333 | 100.0% | — | — | — |
| @babel/helper-builder-binary-assignment-operator-visitor | 2 | 50.0% | 50.0% | 8.0 | 100.0% | — | — | — |
| @babel/helper-compilation-targets | 4 | 75.0% | 75.0% | 39.75 | 75.0% | — | — | — |
| @babel/helper-module-imports | 3 | 66.7% | 66.7% | 11.3333 | 100.0% | — | — | — |
| @babel/helper-module-transforms | 4 | 100.0% | 100.0% | 20.75 | 100.0% | — | — | — |
| @babel/plugin-transform-block-scoping | 3 | 33.3% | 33.3% | 5.6667 | 100.0% | — | — | — |
| @babel/template | 4 | 25.0% | 25.0% | 0.75 | 100.0% | — | — | — |
| @babel/types | 30 | 63.3% | 63.3% | 10.9 | 93.3% | — | — | — |
| @grpc/proto-loader | 2 | 100.0% | 100.0% | 28.5 | 100.0% | — | — | — |
| @hapi/hoek | 12 | 41.7% | 41.7% | 7.4167 | 100.0% | — | — | — |
| @npmcli/metavuln-calculator | 2 | 0.0% | 0.0% | 14.0 | 50.0% | — | — | — |
| @npmcli/promise-spawn | 2 | 100.0% | 100.0% | 3.5 | 100.0% | — | — | — |
| @pkgjs/parseargs | 4 | 75.0% | 75.0% | 25.75 | 100.0% | — | — | — |
| @protobufjs/float | 2 | 0.0% | 0.0% | 16.5 | 50.0% | — | — | — |
| @sigstore/bundle | 4 | 75.0% | 75.0% | 33.0 | 100.0% | — | — | — |
| @sigstore/tuf | 5 | 80.0% | 80.0% | 5.8 | 100.0% | — | — | — |
| @tufjs/models | 14 | 71.4% | 71.4% | 12.7143 | 92.9% | — | — | — |
| adm-zip | 7 | 71.4% | 71.4% | 7.8571 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs2 | 3 | 66.7% | 66.7% | 14.3333 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs3 | 4 | 25.0% | 25.0% | 9.25 | 75.0% | — | — | — |
| browserslist | 4 | 50.0% | 50.0% | 6.0 | 100.0% | — | — | — |
| cacache | 12 | 58.3% | 58.3% | 12.0 | 100.0% | — | — | — |
| cli-progress | 8 | 87.5% | 87.5% | 10.5 | 100.0% | — | — | — |
| cmd-shim | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| commander | 4 | 100.0% | 100.0% | 14.25 | 100.0% | — | — | — |
| core-js-compat | 3 | 100.0% | 100.0% | 16.0 | 100.0% | — | — | — |
| d | 2 | 100.0% | 100.0% | 3.0 | 100.0% | — | — | — |
| date-fns | 30 | 73.3% | 76.7% | 16.3 | 100.0% | — | — | — |
| debug | 2 | 0.0% | 0.0% | 11.0 | 100.0% | — | — | — |
| diff | 12 | 66.7% | 66.7% | 8.5833 | 91.7% | — | — | — |
| ecdsa-sig-formatter | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| es5-ext | 30 | 46.7% | 46.7% | 8.5667 | 93.3% | — | — | — |
| exponential-backoff | 6 | 66.7% | 66.7% | 9.6667 | 100.0% | — | — | — |
| http-proxy | 6 | 66.7% | 66.7% | 15.8333 | 100.0% | — | — | — |
| https-proxy-agent | 2 | 50.0% | 50.0% | 3.5 | 50.0% | — | — | — |
| json-source-map | 2 | 50.0% | 50.0% | 16.5 | 100.0% | — | — | — |
| jsonwebtoken | 5 | 60.0% | 60.0% | 10.2 | 100.0% | — | — | — |
| libnpmexec | 3 | 66.7% | 66.7% | 2.6667 | 100.0% | — | — | — |
| minimatch | 3 | 33.3% | 33.3% | 2.6667 | 100.0% | — | — | — |
| minimist | 11 | 72.7% | 72.7% | 7.6364 | 100.0% | — | — | — |
| minipass-fetch | 6 | 50.0% | 50.0% | 6.3333 | 100.0% | — | — | — |
| nearley | 8 | 37.5% | 37.5% | 4.125 | 100.0% | — | — | — |
| node-gyp-build | 1 | 0.0% | 0.0% | 13.0 | 100.0% | — | — | — |
| postcss-selector-parser | 17 | 58.8% | 58.8% | 12.6471 | 100.0% | — | — | — |
| pump | 2 | 50.0% | 50.0% | 18.5 | 100.0% | — | — | — |
| readable-stream | 12 | 58.3% | 58.3% | 22.5 | 100.0% | — | — | — |
| requires-port | 2 | 50.0% | 50.0% | 4.0 | 100.0% | — | — | — |
| rxjs | 30 | 66.7% | 66.7% | 11.2333 | 90.0% | — | — | — |
| selfsigned | 2 | 50.0% | 50.0% | 7.0 | 100.0% | — | — | — |
| shell-quote | 1 | 100.0% | 100.0% | 12.0 | 100.0% | — | — | — |
| tar-stream | 2 | 100.0% | 100.0% | 12.0 | 100.0% | — | — | — |
| type | 30 | 70.0% | 70.0% | 12.3 | 96.7% | — | — | — |
| uglify-js | 4 | 50.0% | 50.0% | 10.25 | 100.0% | — | — | — |
| undici | 30 | 46.7% | 46.7% | 15.4 | 86.7% | — | — | — |
| valid-url | 3 | 66.7% | 66.7% | 9.3333 | 100.0% | — | — | — |
| yaml | 30 | 40.0% | 40.0% | 14.0 | 93.3% | — | — | — |
| yargs | 2 | 100.0% | 100.0% | 14.5 | 100.0% | — | — | — |
| zod-to-json-schema | 15 | 66.7% | 66.7% | 17.8 | 100.0% | — | — | — |

