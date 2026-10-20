# KapiLab: comandos prontos para os pacotes T0 Nano / T1 Small

**Status (2026-10-20):** o binario publicado EXISTE (nao foi preciso compilar nada, nada foi escrito em `F:\`):

- `F:\source\esilva-silva\EsilvaSoft.KapibaraStudio\artifacts\f7b\publish\Cpu\win-x64\EsilvaSoft.KapibaraStudio.KapiLab.exe` (backend CPU)
- `F:\source\esilva-silva\EsilvaSoft.KapibaraStudio\artifacts\f7b\publish\WinML\win-x64\EsilvaSoft.KapibaraStudio.KapiLab.exe` (backend WinML/DML)
- SHA-256 do apphost = `a8da67cf...735e` (igual ao `tools/kapilab.lock.json`); `kapilab env`: GenAI 0.15.2, runtime .NET 10.0.13, `ideCommit` do checkout atual
  `7844bd71...` (o lock registra `b1e10b09...`, `ideDirty: true`: o checkout andou desde o publish; os hashes do apphost batem, mas o gate oficial deve
  usar um publish do commit da IDE sob avaliacao).
- `dotnet` SDK 10.0.402 presente (nao usado).
- O KapiLab so grava em `data/lab`, `reports/lab`, `tmp` do **workspace**: usar `--workspace C:\IA\models\KapibaraStudio.MongoIA\tmp\kapilab_ws` (nunca `F:\`).

**O que NAO foi possivel:** `bench`/`model run` com `--device gpu` exigem `--via-queue` + `KAPILAB_GPU_TOKEN` emitido pela fila local de GPU (coordenacao entre jobs);
nao ha token nesta sessao e ele nao foi contornado. Logo o gate oficial DML (T1 `dml_fp16`) fica pendente do KapiLab com a fila (ver abaixo). QNN/NPU: sem
maquina Snapdragon; nada exportado para `qnn-*`.

## Comandos (PowerShell)

```powershell
$K  = 'F:\source\esilva-silva\EsilvaSoft.KapibaraStudio\artifacts\f7b\publish\Cpu\win-x64\EsilvaSoft.KapibaraStudio.KapiLab.exe'
$KW = 'F:\source\esilva-silva\EsilvaSoft.KapibaraStudio\artifacts\f7b\publish\WinML\win-x64\EsilvaSoft.KapibaraStudio.KapiLab.exe'
$WS = 'C:\IA\models\KapibaraStudio.MongoIA\tmp\kapilab_ws'
$EX = 'C:\IA\models\KapibaraStudio.MongoIA\Repos'
Set-Location $WS
& $K env --expect-genai 0.15.2

# 1) validacao do pacote (IDE + schema kapi + hashes do kapicoder_manifest.json)
& $K model validate --package "$EX\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-ONNX-int4" --strict-kapi --verify-hashes
& $K model validate --models-dir "$EX\KapiCoder-Mongo-Small\exports" --strict-kapi --verify-hashes      # todas as variantes do tier

# 2) registros -> render -> tokenizacao pelo adapter real (ids comparaveis com o Python)
C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe -I scripts/export/kapilab_records.py --tier t0 --n 100 --out tmp/kapilab_ws/tmp/records_t0.jsonl   # na raiz do repo
& $K --workspace $WS contract render   --in tmp/records_t0.jsonl --out tmp/render_t0.jsonl
& $K --workspace $WS contract tokenize --package "$EX\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-ONNX-int4" --in tmp/render_t0.jsonl --out tmp/tokens_t0.jsonl
C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe -I scripts/export/kapilab_compare_tokens.py --tier t0 --tokens tmp/kapilab_ws/tmp/tokens_t0.jsonl     # raiz do repo

# 3) bench CPU pelo caminho real da IDE (203 registros, 1 aquecimento) - rodar com a maquina ociosa
& $K --workspace $WS bench autocomplete --package "$EX\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-ONNX-int4" --in tmp/records_t0_203.jsonl --device cpu --warmup 1 --iterations 1 --out tmp/bench_t0_int4_cpu.jsonl
bash scripts/export/kapilab_bench_cpu.sh        # todos os pacotes CPU (raiz do repo)

# 4) GPU/DML (PENDENTE: precisa da fila de GPU e do token)
$env:KAPILAB_GPU_TOKEN = '<token emitido por scripts/gpu_queue.ps1>'
& $KW --workspace $WS bench autocomplete --package "$EX\KapiCoder-Mongo-Small\exports\KapiCoder-Mongo-Small-ONNX-dml-fp16" --in tmp/records_t1_203.jsonl --device gpu --via-queue --warmup 1 --iterations 1 --out tmp/bench_t1_dml-fp16.jsonl
```

Antes de qualquer rodada GPU, confirmar que o DML esta na RX 7800 XT (nao na iGPU): `scripts/export/gpu_probe.ps1 -ProcId <pid>` durante a execucao
(LUID `0x00000000_0x00012951` nesta sessao = 7800 XT; iGPU `0x00000000_0x00014bfc`).

## Resultados desta sessao

Ver `reports/export/onnx_export_t0_t1.md` (secao KapiLab). Arquivos brutos: `tmp/kapilab_ws/tmp/*.jsonl` (bench), `reports/export/kapilab_validate_*.json`,
`reports/export/kapilab_contract_tokens_t0.json`.
