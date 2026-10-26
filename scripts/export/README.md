# Exportacao ONNX dos alunos T0 Nano / T1 Small (Fase 07)

Pacotes: `Repos/KapiCoder-Mongo-<Nano|Small>/exports/KapiCoder-Mongo-<Nano|Small>-ONNX-<variante>/` (gitignored). Builder: `onnxruntime-genai` **0.15.2**
(mesma versao da IDE). Modelos de origem (bf16, pesos completos): `Repos/KapiCoder-Mongo-Nano/merged/sft_distill_v1` (T0, 0.5B) e
`Repos/KapiCoder-Mongo-Small/merged/sft_distill_v1` (T1, 1.5B, LoRA ja fundido). Relatorio: `reports/export/onnx_export_t0_t1.md`.

Venvs: `PY=C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe` (ROCm, onnxruntime-genai CPU 0.15.2) e
`W=C:/IA/models/SlopStudio.MongoAI-Compact/.venv-winml/Scripts/python.exe` (onnxruntime-genai-winml 0.15.2; DML; **sem `-I`**, o `sitecustomize` carrega o
`onnxruntime.dll` do pacote WinML; nao importar `onnxruntime`). Mongo no WSL (`wsl -d Ubuntu -u root -e systemctl start mongod`) so para o exec-valid.

## Geracao v3 (retreino J20-J23; `KAPI_EXPORT_GEN=v3`)

Todos os scripts respeitam a variavel `KAPI_EXPORT_GEN` (defina ANTES de iniciar; os subprocessos herdam): vazio = legado (`merged/sft_distill_v1`, dataset v2c, pacotes
`<familia>-ONNX-<variante>`); `v3` = `merged/sft_v3`, dataset v3, pacotes **`<familia>-v3-ONNX-<variante>`**, e diretorios separados (`reports/export/work_v3|eval_v3|gates_v3`,
`results_v3_<tier>.json`). O legado nao e sobrescrito. Receita completa (PowerShell, `$env:KAPI_EXPORT_GEN='v3'`):

```powershell
$env:KAPI_EXPORT_GEN='v3'; $PY = "C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe"
& $PY -I scripts/export/build_variants.py --tier t0 --variant int4 --variant int8 --variant dml-fp16 --variant int8-emb4
& $PY -I scripts/export/build_variants.py --tier t1 --variant int8 --variant dml-fp16
& $PY -I scripts/export/package_variant.py --tier t0 --all ; & $PY -I scripts/export/package_variant.py --tier t1 --all   # metadata kapi v3 (run_manifest do treino)
& $PY -I scripts/export/prep_prompts.py --tier t0         # GPU 7800 XT (HF bf16 do sft_v3); idem t1
# LUIDs mudam por boot: calibre (torch-ROCm aloca 3 GB; gpu_probe.ps1) e exporte antes das medidas DML
$env:KAPI_RTX_LUID='0x00000000_0x00012602'; $env:KAPI_IGPU_LUID='0x00000000_0x00014830'
& $PY -I scripts/export/measure_all.py --tier t0 --phase quality ; & $PY -I scripts/export/measure_all.py --tier t0 --phase latency   # idem t1
& $PY -I scripts/export/compare_onnx.py --tier t0 ; & $PY -I scripts/export/gates_proxy.py --tier t0
# homologacao v1 (n=500, SO pela API autorizada de homolog_guard; agregados em reports/export/homolog_onnx_v3_<tier>.json)
& $PY -I scripts/export/homolog_onnx.py prep --tier t0
& $PY -I scripts/export/homolog_onnx.py run  --tier t0 --variant int4      # DML: venv WinML sem -I
& $PY -I scripts/export/homolog_onnx.py post --tier t0 --variant int4
& env/.venv/Scripts/python.exe -I scripts/eval/eval_homolog.py score --name onnx_v3_t0_int4
& $PY -I scripts/export/homolog_onnx.py compare --tier t0
```

**Contorno do EOS da IDE (decisao do usuario PENDENTE; os pacotes padrao NAO o aplicam):** gere a variante com contorno, sem alterar o pacote original (copia por hardlinks
`<pacote>-eosfix`, `genai_config.model.eos_token_id = 151935`, manifestos refeitos, metadata `eosWorkaround`):
`& $PY -I scripts/export/package_variant.py --tier t0 --variant int4 --eos-workaround`.

## Variantes (`build_variants.py`)

| variante | precisao / EP | opcoes do builder | nota |
|---|---|---|---|
| `fp32-cpu` | fp32 / cpu | - | referencia (grafo correto); nao publicar |
| `int4` | int4 / cpu | `algo_config=k_quant matmul_mixed_precision=last_matmul:int8 block_size=32 accuracy_level=4` | recomendada pela Fase 07 §5.2 |
| `int4-rtn` | int4 / cpu | `algo_config=rtn block_size=32 accuracy_level=4` | controle |
| `int4-kquantmixed` | int4 / cpu | `algo_config=k_quant_mixed` | melhor INT4 da sonda |
| `int8` | int8 / cpu | `op_types_to_quantize=MatMul shared_embeddings=false block_size=32 accuracy_level=4` | contorno do `-p int8` (grafo invalido no 0.15.2); embedding fp32 |
| `int8-emb4` | int8 / cpu | idem + `quantize_gather.py` (embedding em GatherBlockQuantized INT4) | cabe nos 700 MB do T0 (Gather so aceita 4 bits no ORT 1.30) |
| `dml-fp16` | fp16 / dml | - | |
| `dml-int4` | int4 / dml | `op_types_to_quantize=MatMul shared_embeddings=false block_size=32` | `GatherBlockQuantized` nao roda no DML |

`model.type` permanece `qwen2` (adapter Qwen da IDE).

## Pipeline de reproducao

```powershell
$PY = "C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe"
# 1) exportar (um builder por vez; cache proprio por build)
& $PY -I scripts/export/build_variants.py --tier t0 --variant int4 --variant int8 --variant dml-fp16     # t0 | t1
# 2) completar o pacote: tokenizer byte a byte, kapibarastudio-model.json (schema kapilab-kapi-model-v1), LICENSE/NOTICE/README, manifestos
& $PY -I scripts/export/package_variant.py --tier t0 --all
# 3) prompts editor-context-v1 (500 ids do val v2c, seed 20261020) + referencia HF bf16 token a token (200 primeiros) na RX 7800 XT
& $PY -I scripts/export/prep_prompts.py --tier t0
# 4) qualidade/paridade (tokenizer nativo x HF, 500 prompts greedy, scorer do eval_teacher) e latencia PROXY, um processo por pacote
& $PY -I scripts/export/measure_all.py --tier t0 --phase quality
& $PY -I scripts/export/measure_all.py --tier t0 --phase latency        # maquina ociosa; DML so e aceito se os contadores confirmarem a RX 7800 XT
# 5) agregacao, gates proxy (check_gates) e testes
& $PY -I scripts/export/compare_onnx.py --tier t0
& $PY -I scripts/export/gates_proxy.py --tier t0
& env/.venv/Scripts/python.exe -m pytest scripts/export/tests -q -p no:cacheprovider   # KAPI_EXPORT_FULL_HASH=1 re-hasheia os arquivos grandes (o venv ROCm nao tem pytest)
& $PY -I -X utf8 scripts/export/make_tables.py      # tabelas markdown do relatorio
# 6) KapiLab (apphost publicado, somente leitura; ver reports/export/kapilab_commands.md)
bash scripts/export/kapilab_bench_cpu.sh            # bench autocomplete CPU, 203 registros
& $PY -I scripts/export/kapilab_bench_stats.py
```

Pecas: `onnx_run.py` (runtime GenAI: modos `tokenizer|verify|quality|latency`), `score_onnx.py` (converte para o formato do `eval_teacher` e roda o exec-valid em
`kapi_fx_*`), `gpu_probe.ps1` (contadores `GPU Process Memory` / `GPU Engine` por pid e LUID), `kapilab_records.py` + `kapilab_compare_tokens.py` (ponte com o KapiLab).

## Regra da GPU (duas GPUs: RX 7800 XT + iGPU)

O GenAI-WinML escolhe o adaptador sozinho; `device_id` nao e confiavel. Todo pacote DML passa por `onnx_run.py --mode verify` (e a medida de latencia reconfere no fim):
o processo precisa ter memoria dedicada > 100 MB e utilizacao de engine no LUID da 7800 XT (`0x00000000_0x00012951` nesta sessao, identificado alocando 3 GB com
torch-ROCm em `cuda:0` = "AMD Radeon RX 7800 XT") e **zero** no LUID da iGPU (`0x00000000_0x00014bfc`). Se falhar, o script sai com codigo 3 e a medida e descartada.
LUIDs mudam por boot: reconfirme com a calibracao descrita acima e passe `--luid`.

## KapiLab (apphost publicado, somente leitura)

`F:\source\esilva-silva\EsilvaSoft.KapibaraStudio\artifacts\f7b\publish\{Cpu,WinML}\win-x64\EsilvaSoft.KapibaraStudio.KapiLab.exe` (SHA-256 confere com
`tools/kapilab.lock.json`). Comandos prontos: `reports/export/kapilab_commands.md`. Use sempre `--workspace` em `C:` (o KapiLab so grava em `data/lab`, `reports/lab`, `tmp`).

## Achado importante: EOS duplica o ultimo token na IDE (onnxruntime-genai 0.15.2)

No GenAI 0.15.2 o passo que gera `<|endoftext|>` marca `is_done` mas **nao** o acrescenta a sequencia; `OnnxLocalModelRuntime.Generate` le
`generator.GetSequence(0)[^1]` e, nesse passo, recebe de novo o token anterior (`'1'` vira `'11'`, `"` vira `""`, `({` vira `({({`). Reproduzido em Python
(`get_sequence(0)[-1]` x `get_next_tokens()[0]`) e no KapiLab (`model run autocomplete`): concordancia texto IDE x Python 12-13% (T0 int4 / T1 int8). Com
`model.eos_token_id` de `genai_config.json` apontando para um id inalcancavel (151935; a IDE ja para pelos seus proprios tokens de parada) a concordancia sobe para
93% (95,8% nos registros com cabecalho de contexto; teste em copia, os pacotes entregues NAO foram alterados). Correcao propria: ler `GetNextTokens()` ou
tratar `IsDone()` antes de usar `GetSequence()[^1]`. Ver `reports/export/kapilab_agreement_*.json`. O proxy deste diretorio (onnx_run.py) le `get_next_tokens()` e nao e afetado.

## Limites conhecidos

- Latencia e PROXY: GenAI Python, ids pre-tokenizados, sem o caminho `LocalAiModelService`; a IDE chama `ClearProviders()/AppendProvider()` e o proxy faz o mesmo.
- Paridade top-1 *teacher-forced* (Fase 07 §7.1) nao e medida: o GenAI 0.15 so expoe os logits da ultima posicao; usa-se concordancia greedy token a token.
- O `genai_config.json` recebe `search.max_length = 2048` (Fase 07 §6.5); `context_length` continua o do builder (32768).
