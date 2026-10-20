# Baseline zero-shot FIM `base-qwen2.5-coder-1.5b-generic-js`

- Spec: `docs/fase-02-professor.md` §3.2 · split `eval` · n=500 (de 500), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-1.5B` · carga `bf16` · SHA-256 do modelo `8bcb242c05f27140…` (revisao `df3ce67c0e24480f20468b6ef2894622d69eb73b`)
- Dataset: `data\replay\v1\generic_js_fim_eval.jsonl` SHA-256 `8827d1e4078a9f21…` · amostra `fbbfc5471d4a0a7b…`
- Contrato `fim-raw` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 296.6 s totais (287.6 s de geracao) · VRAM pico 3.13 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo —

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 40.6% | 40.6% | — |
| Normalized match (FIM acc.) | 40.6% | 40.6% | — |
| Prefix match | 63.6% | 64.0% | — |
| Syntax valid | 72.0% | 65.6% | 100.0% |
| Mongo valid (estatico) | — | — | — |
| Exec-valid | — | — | — |
| Campos inexistentes | — | — | — |
| Chars aceitos (media) | 13.55 | 13.456 | — |
| APT (tokens) | 3.592 | 3.564 | — |

Distribuicao do estagio de validacao (pred): {}
Paradas: {'max': 109, 'eos': 390, 'suffix': 1}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 39 | 35.9% | 35.9% | 13.0 | 71.8% | — | — | — |
| member | 101 | 37.6% | 37.6% | 7.4851 | 74.3% | — | — | — |
| multiline | 43 | 16.3% | 16.3% | 22.8372 | 58.1% | — | — | — |
| partial | 55 | 49.1% | 49.1% | 7.3636 | 69.1% | — | — | — |
| rest | 103 | 35.9% | 35.9% | 17.6602 | 70.9% | — | — | — |
| stmt_start | 55 | 49.1% | 49.1% | 23.5091 | 87.3% | — | — | — |
| value | 104 | 51.0% | 51.0% | 9.7404 | 70.2% | — | — | — |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| @azure/core-client | 9 | 66.7% | 66.7% | 19.4444 | 77.8% | — | — | — |
| @azure/core-util | 6 | 16.7% | 16.7% | 5.6667 | 83.3% | — | — | — |
| @babel/core | 30 | 50.0% | 50.0% | 20.4667 | 80.0% | — | — | — |
| @babel/helper-builder-binary-assignment-operator-visitor | 2 | 50.0% | 50.0% | 8.0 | 50.0% | — | — | — |
| @babel/helper-compilation-targets | 4 | 25.0% | 25.0% | 39.75 | 50.0% | — | — | — |
| @babel/helper-module-imports | 3 | 33.3% | 33.3% | 7.6667 | 33.3% | — | — | — |
| @babel/helper-module-transforms | 4 | 50.0% | 50.0% | 20.75 | 75.0% | — | — | — |
| @babel/plugin-transform-block-scoping | 3 | 33.3% | 33.3% | 5.6667 | 66.7% | — | — | — |
| @babel/template | 4 | 25.0% | 25.0% | 1.75 | 100.0% | — | — | — |
| @babel/types | 30 | 36.7% | 36.7% | 10.1333 | 70.0% | — | — | — |
| @grpc/proto-loader | 2 | 50.0% | 50.0% | 28.5 | 50.0% | — | — | — |
| @hapi/hoek | 12 | 16.7% | 16.7% | 7.4167 | 58.3% | — | — | — |
| @npmcli/metavuln-calculator | 2 | 50.0% | 50.0% | 7.0 | 100.0% | — | — | — |
| @npmcli/promise-spawn | 2 | 50.0% | 50.0% | 3.5 | 50.0% | — | — | — |
| @pkgjs/parseargs | 4 | 25.0% | 25.0% | 25.75 | 75.0% | — | — | — |
| @protobufjs/float | 2 | 0.0% | 0.0% | 16.5 | 50.0% | — | — | — |
| @sigstore/bundle | 4 | 100.0% | 100.0% | 38.5 | 100.0% | — | — | — |
| @sigstore/tuf | 5 | 60.0% | 60.0% | 5.0 | 80.0% | — | — | — |
| @tufjs/models | 14 | 64.3% | 64.3% | 13.7857 | 85.7% | — | — | — |
| adm-zip | 7 | 42.9% | 42.9% | 7.8571 | 42.9% | — | — | — |
| babel-plugin-polyfill-corejs2 | 3 | 66.7% | 66.7% | 21.6667 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs3 | 4 | 25.0% | 25.0% | 9.0 | 75.0% | — | — | — |
| browserslist | 4 | 0.0% | 0.0% | 6.5 | 50.0% | — | — | — |
| cacache | 12 | 41.7% | 41.7% | 10.1667 | 83.3% | — | — | — |
| cli-progress | 8 | 50.0% | 50.0% | 10.5 | 87.5% | — | — | — |
| cmd-shim | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| commander | 4 | 50.0% | 50.0% | 8.0 | 75.0% | — | — | — |
| core-js-compat | 3 | 66.7% | 66.7% | 5.6667 | 100.0% | — | — | — |
| d | 2 | 100.0% | 100.0% | 3.0 | 100.0% | — | — | — |
| date-fns | 30 | 56.7% | 56.7% | 19.3333 | 86.7% | — | — | — |
| debug | 2 | 50.0% | 50.0% | 17.0 | 50.0% | — | — | — |
| diff | 12 | 33.3% | 33.3% | 10.8333 | 58.3% | — | — | — |
| ecdsa-sig-formatter | 1 | 0.0% | 0.0% | 25.0 | 0.0% | — | — | — |
| es5-ext | 30 | 23.3% | 23.3% | 8.6667 | 63.3% | — | — | — |
| exponential-backoff | 6 | 33.3% | 33.3% | 9.6667 | 83.3% | — | — | — |
| http-proxy | 6 | 50.0% | 50.0% | 14.3333 | 83.3% | — | — | — |
| https-proxy-agent | 2 | 50.0% | 50.0% | 3.5 | 100.0% | — | — | — |
| json-source-map | 2 | 50.0% | 50.0% | 16.5 | 100.0% | — | — | — |
| jsonwebtoken | 5 | 60.0% | 60.0% | 12.6 | 100.0% | — | — | — |
| libnpmexec | 3 | 33.3% | 33.3% | 2.6667 | 66.7% | — | — | — |
| minimatch | 3 | 33.3% | 33.3% | 4.3333 | 66.7% | — | — | — |
| minimist | 11 | 36.4% | 36.4% | 7.6364 | 72.7% | — | — | — |
| minipass-fetch | 6 | 16.7% | 16.7% | 6.3333 | 66.7% | — | — | — |
| nearley | 8 | 50.0% | 50.0% | 5.75 | 100.0% | — | — | — |
| node-gyp-build | 1 | 0.0% | 0.0% | 10.0 | 0.0% | — | — | — |
| postcss-selector-parser | 17 | 58.8% | 58.8% | 12.7647 | 82.3% | — | — | — |
| pump | 2 | 50.0% | 50.0% | 18.5 | 100.0% | — | — | — |
| readable-stream | 12 | 16.7% | 16.7% | 27.5833 | 41.7% | — | — | — |
| requires-port | 2 | 0.0% | 0.0% | 15.0 | 50.0% | — | — | — |
| rxjs | 30 | 70.0% | 70.0% | 14.3 | 76.7% | — | — | — |
| selfsigned | 2 | 0.0% | 0.0% | 7.0 | 0.0% | — | — | — |
| shell-quote | 1 | 0.0% | 0.0% | 12.0 | 0.0% | — | — | — |
| tar-stream | 2 | 50.0% | 50.0% | 2.5 | 50.0% | — | — | — |
| type | 30 | 46.7% | 46.7% | 10.3667 | 63.3% | — | — | — |
| uglify-js | 4 | 0.0% | 0.0% | 12.75 | 0.0% | — | — | — |
| undici | 30 | 23.3% | 23.3% | 18.3667 | 66.7% | — | — | — |
| valid-url | 3 | 0.0% | 0.0% | 9.3333 | 33.3% | — | — | — |
| yaml | 30 | 26.7% | 26.7% | 14.4667 | 76.7% | — | — | — |
| yargs | 2 | 0.0% | 0.0% | 6.0 | 50.0% | — | — | — |
| zod-to-json-schema | 15 | 40.0% | 40.0% | 19.2 | 80.0% | — | — | — |

