# Baseline zero-shot FIM `sft_v2c-final-generic-js`

- Spec: `docs/fase-02-professor.md` §3.2 · split `eval` · n=500 (de 500), estratificado por cut_kind x dominio, seed 20261008
- Modelo: `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-7B` · carga `nf4` · SHA-256 do modelo `74ac207b4deacd56…` (revisao `0396a76181e127dfc13e5c5ec48a8cee09938b02`)
- Dataset: `data\replay\v1\generic_js_fim_eval.jsonl` SHA-256 `8827d1e4078a9f21…` · amostra `fbbfc5471d4a0a7b…`
- Contrato `fim-raw` · greedy · max_new_tokens 96 · contexto 2048 tokens
- GPU: AMD Radeon RX 7800 XT · 842.1 s totais (812.3 s de geracao) · VRAM pico 6.41 GB
- Libs: torch 2.10.0+rocm7.14.0a20260611 · transformers 4.57.6 · bitsandbytes 0.50.2 · pymongo —

## Principais

| metrica | 96 tokens | @32 (IDE) | teto da referencia |
|---|---:|---:|---:|
| Exact match | 70.6% | 70.0% | — |
| Normalized match (FIM acc.) | 70.8% | 70.0% | — |
| Prefix match | 72.0% | 72.8% | — |
| Syntax valid | 97.0% | 91.6% | 100.0% |
| Mongo valid (estatico) | — | — | — |
| Exec-valid | — | — | — |
| Campos inexistentes | — | — | — |
| Chars aceitos (media) | 17.578 | 16.604 | — |
| APT (tokens) | 4.542 | 4.322 | — |

Distribuicao do estagio de validacao (pred): {}
Paradas: {'eos': 491, 'max': 9}

## Por cut_kind

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 39 | 84.6% | 84.6% | 14.7179 | 97.4% | — | — | — |
| member | 101 | 70.3% | 70.3% | 9.9208 | 97.0% | — | — | — |
| multiline | 43 | 27.9% | 30.2% | 26.0465 | 95.3% | — | — | — |
| partial | 55 | 85.5% | 85.5% | 8.4182 | 100.0% | — | — | — |
| rest | 103 | 67.0% | 67.0% | 28.4272 | 96.1% | — | — | — |
| stmt_start | 55 | 63.6% | 63.6% | 28.7636 | 94.5% | — | — | — |
| value | 104 | 82.7% | 82.7% | 10.7692 | 98.1% | — | — | — |

## Por dominio

| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| @azure/core-client | 9 | 77.8% | 77.8% | 19.4444 | 88.9% | — | — | — |
| @azure/core-util | 6 | 66.7% | 66.7% | 15.5 | 83.3% | — | — | — |
| @babel/core | 30 | 73.3% | 73.3% | 21.0333 | 96.7% | — | — | — |
| @babel/helper-builder-binary-assignment-operator-visitor | 2 | 50.0% | 50.0% | 8.0 | 50.0% | — | — | — |
| @babel/helper-compilation-targets | 4 | 75.0% | 75.0% | 39.75 | 75.0% | — | — | — |
| @babel/helper-module-imports | 3 | 66.7% | 66.7% | 11.3333 | 100.0% | — | — | — |
| @babel/helper-module-transforms | 4 | 100.0% | 100.0% | 20.75 | 100.0% | — | — | — |
| @babel/plugin-transform-block-scoping | 3 | 33.3% | 33.3% | 5.6667 | 100.0% | — | — | — |
| @babel/template | 4 | 25.0% | 25.0% | 0.75 | 100.0% | — | — | — |
| @babel/types | 30 | 66.7% | 66.7% | 14.1 | 100.0% | — | — | — |
| @grpc/proto-loader | 2 | 100.0% | 100.0% | 28.5 | 100.0% | — | — | — |
| @hapi/hoek | 12 | 50.0% | 50.0% | 9.25 | 100.0% | — | — | — |
| @npmcli/metavuln-calculator | 2 | 50.0% | 50.0% | 14.5 | 100.0% | — | — | — |
| @npmcli/promise-spawn | 2 | 100.0% | 100.0% | 3.5 | 100.0% | — | — | — |
| @pkgjs/parseargs | 4 | 100.0% | 100.0% | 34.25 | 100.0% | — | — | — |
| @protobufjs/float | 2 | 0.0% | 50.0% | 25.0 | 100.0% | — | — | — |
| @sigstore/bundle | 4 | 100.0% | 100.0% | 38.5 | 100.0% | — | — | — |
| @sigstore/tuf | 5 | 80.0% | 80.0% | 5.0 | 100.0% | — | — | — |
| @tufjs/models | 14 | 78.6% | 78.6% | 14.7857 | 92.9% | — | — | — |
| adm-zip | 7 | 71.4% | 71.4% | 8.0 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs2 | 3 | 100.0% | 100.0% | 29.6667 | 100.0% | — | — | — |
| babel-plugin-polyfill-corejs3 | 4 | 50.0% | 50.0% | 11.0 | 75.0% | — | — | — |
| browserslist | 4 | 100.0% | 100.0% | 7.5 | 100.0% | — | — | — |
| cacache | 12 | 75.0% | 75.0% | 14.0833 | 100.0% | — | — | — |
| cli-progress | 8 | 87.5% | 87.5% | 10.5 | 100.0% | — | — | — |
| cmd-shim | 1 | 0.0% | 0.0% | 0.0 | 100.0% | — | — | — |
| commander | 4 | 100.0% | 100.0% | 14.25 | 100.0% | — | — | — |
| core-js-compat | 3 | 100.0% | 100.0% | 16.0 | 100.0% | — | — | — |
| d | 2 | 100.0% | 100.0% | 3.0 | 100.0% | — | — | — |
| date-fns | 30 | 93.3% | 93.3% | 26.3667 | 100.0% | — | — | — |
| debug | 2 | 100.0% | 100.0% | 19.5 | 100.0% | — | — | — |
| diff | 12 | 75.0% | 75.0% | 11.75 | 100.0% | — | — | — |
| ecdsa-sig-formatter | 1 | 100.0% | 100.0% | 25.0 | 100.0% | — | — | — |
| es5-ext | 30 | 60.0% | 60.0% | 10.8333 | 100.0% | — | — | — |
| exponential-backoff | 6 | 66.7% | 66.7% | 28.6667 | 83.3% | — | — | — |
| http-proxy | 6 | 66.7% | 66.7% | 15.8333 | 100.0% | — | — | — |
| https-proxy-agent | 2 | 50.0% | 50.0% | 3.5 | 50.0% | — | — | — |
| json-source-map | 2 | 50.0% | 50.0% | 16.5 | 100.0% | — | — | — |
| jsonwebtoken | 5 | 80.0% | 80.0% | 14.4 | 100.0% | — | — | — |
| libnpmexec | 3 | 66.7% | 66.7% | 5.6667 | 100.0% | — | — | — |
| minimatch | 3 | 33.3% | 33.3% | 4.3333 | 100.0% | — | — | — |
| minimist | 11 | 90.9% | 90.9% | 9.0909 | 100.0% | — | — | — |
| minipass-fetch | 6 | 83.3% | 83.3% | 9.0 | 83.3% | — | — | — |
| nearley | 8 | 62.5% | 62.5% | 8.0 | 100.0% | — | — | — |
| node-gyp-build | 1 | 0.0% | 0.0% | 54.0 | 100.0% | — | — | — |
| postcss-selector-parser | 17 | 64.7% | 64.7% | 22.0588 | 100.0% | — | — | — |
| pump | 2 | 50.0% | 50.0% | 18.5 | 100.0% | — | — | — |
| readable-stream | 12 | 75.0% | 75.0% | 61.0833 | 83.3% | — | — | — |
| requires-port | 2 | 100.0% | 100.0% | 16.0 | 100.0% | — | — | — |
| rxjs | 30 | 76.7% | 76.7% | 25.0667 | 93.3% | — | — | — |
| selfsigned | 2 | 100.0% | 100.0% | 15.0 | 100.0% | — | — | — |
| shell-quote | 1 | 100.0% | 100.0% | 12.0 | 100.0% | — | — | — |
| tar-stream | 2 | 100.0% | 100.0% | 12.0 | 100.0% | — | — | — |
| type | 30 | 70.0% | 70.0% | 10.9333 | 100.0% | — | — | — |
| uglify-js | 4 | 75.0% | 75.0% | 14.25 | 100.0% | — | — | — |
| undici | 30 | 53.3% | 53.3% | 20.3667 | 96.7% | — | — | — |
| valid-url | 3 | 66.7% | 66.7% | 7.0 | 100.0% | — | — | — |
| yaml | 30 | 56.7% | 56.7% | 17.9333 | 100.0% | — | — | — |
| yargs | 2 | 100.0% | 100.0% | 14.5 | 100.0% | — | — | — |
| zod-to-json-schema | 15 | 53.3% | 53.3% | 14.2667 | 100.0% | — | — | — |

