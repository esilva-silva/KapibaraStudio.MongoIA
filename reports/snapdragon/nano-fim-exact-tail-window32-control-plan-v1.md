# Nano FIM exact tail: inspeção e smoke CPU da janela 32

Preparação de comandos, sem execução sobre o artefato em construção. O plano está em `nano-fim-exact-tail-window32-control-plan-v1.json`. Nenhum comando abaixo compila HTP, executa NPU, mede qualidade ou publica arquivos.

Receipts mínimos antes de copiar/inspecionar o pacote:

| Evidência | Gate mínimo |
| --- | --- |
| GPTQ job v3 | GptqModel concluído; 30 blocos; captura real `quantize_entry` e `after_prepare_dataset`, 61.273 tokens e hash `07f7dec2ffed64e72f7d015bfd86bbce5d16f0297538c6fffd079fe9183da362`; hashes de cada bloco/máscara conferidos |
| Ledger runtime GPTQ | Mesmos 256 IDs cobertos, 29×2048+1881, máscaras 1, sem padding/shuffle, helper igual ao pin do preflight |
| Trace CPU real | `status=completed`, `calibration_status=input_complete`, `calibration_inputs_seen=256`, `fim_preflight.matched=256`, `all_match=true`, `fim_complete=true`; ordem/IDs/máscaras vinculados ao preflight; CPUExecutionProvider, QUInt16, sem stride que descarte entradas |
| Preflights reais | Nano GPTQ workflow v3 e Reader/guard CPU v3 aprovados; a equivalência da ativação v3 permanece registrada no preflight GPTQ |
| Checkpoint | Inspeção corrigida aprovada: finitude e paridade de IDs, sem declarar qualidade |
| Pipeline CPU | Workflow QNN retornou e ModelBuilder, MatMulNBitsToQDQ, GraphSurgeries, DynamicToFixedShape, OnnxStaticQuantization, SplitModel e StaticLLM concluíram; receita janela 32 |
| Pacote | Cópia nova/relocável, tokenizer original byte a byte, inspeção ONNX/escala/finitude/456 prompts aprovada e `transfer_ready=true` |

O verificador novo lê somente JSONs e arquivos para hashes. Não carrega modelos nem executa inferência. Os gates recusam receipts ausentes/incompletos; uma falha deve ser preservada com novo nome de relatório para a repetição. Os caminhos do plano são concretos para Nano v3; se o job for relançado, atualizar o plano explicitamente com o novo trace/receipts, sem misturar tentativas.

Executar manualmente somente depois do término bem-sucedido do job CPU atual, no worktree `C:/IA/models/KapibaraStudio.MongoIA-snapdragon-rx7800xt`:

```powershell
$ErrorActionPreference = 'Stop'
$taskRoot = 'C:/IA/models/KapibaraStudio.MongoIA-snapdragon-rx7800xt'
$taskArtifacts = 'C:/IA/models/KapiSnapdragonArtifacts/nano-fim-exact-tail-v1-rx7800xt'
$taskSnapshot = 'C:/IA/models/KapiPublishedSnapshots/KapiCoder-Mongo-Nano/c03bfe15ba1ff4e5b58a4b0953349eca82bb43f1'
$taskPackage = 'C:/IA/models/KapiSnapdragonArtifacts/nano-fim-exact-tail-window32-transfer-v1'
$taskPlan = "$taskRoot/reports/snapdragon/nano-fim-exact-tail-window32-control-plan-v1.json"
$taskProbePython = "$taskRoot/env/.venv-snapdragon-rx-probe/Scripts/python.exe"
$taskCpuPython = "$taskRoot/env/.venv-snapdragon-qnn-cpu/Scripts/python.exe"
$taskReceiptReport = "$taskRoot/reports/snapdragon/nano-fim-exact-tail-window32-receipts-ready-v1.json"
$env:HIP_VISIBLE_DEVICES = '-1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
Remove-Item Env:KAPI_QNN_PREP_TRACE_DIR -ErrorAction SilentlyContinue
Remove-Item Env:KAPI_QNN_FIM_PREFLIGHT -ErrorAction SilentlyContinue
Set-Location -LiteralPath $taskRoot

& $taskProbePython -B scripts/snapdragon/verify_prepared_qnn_receipts.py --manifest $taskPlan --phase receipts --out $taskReceiptReport
if ($LASTEXITCODE -ne 0) { throw 'Receipts incompletos; não copiar nem executar smoke.' }

& $taskCpuPython -B scripts/snapdragon/prepare_qnn_transfer.py --source "$taskArtifacts/qnn-output-v3" --destination $taskPackage --provenance "$taskArtifacts/base/prepare-manifest.json" --source-snapshot $taskSnapshot --preserve-tokenizer-source --report $taskReceiptReport --report "$taskArtifacts/gptq-job-status-v3.json" --report "$taskArtifacts/tail-v3-projected/runtime-token-ledger.json" --report "$taskArtifacts/qnn-trace-v3/qnn-qdq-62164-eda69972b1e34314844c6d57d60d1340.json" --report "$taskRoot/reports/snapdragon/nano-fim-exact-tail-gptq-real-preflight-rx7800xt-v3.json" --report "$taskRoot/reports/snapdragon/nano-fim-exact-tail-activation-reader-real-preflight-rx7800xt-v3.json" --report "$taskArtifacts/gptq-checkpoint-inspection-v3-corrected.json"
if ($LASTEXITCODE -ne 0) { throw 'Preparação da cópia falhou.' }

& $taskCpuPython -B scripts/snapdragon/inspect_qnn_prepared.py --artifact-dir $taskPackage --source $taskSnapshot --samples "$taskRoot/data/calib/snapdragon-20261026" --expected-window 32 --expected-head-size 64 --require-original-tokenizer --out "$taskRoot/reports/snapdragon/nano-fim-exact-tail-window32-static-validation-v1.json"
if ($LASTEXITCODE -ne 0) { throw 'Inspeção estática falhou.' }

& $taskProbePython -B scripts/snapdragon/verify_prepared_qnn_receipts.py --manifest $taskPlan --phase package --out "$taskRoot/reports/snapdragon/nano-fim-exact-tail-window32-package-ready-v1.json"
if ($LASTEXITCODE -ne 0) { throw 'Pacote mudou, não é relocável ou contém provider diferente de CPU.' }

& $taskCpuPython -B scripts/snapdragon/probe_qnn_static_cpu.py --model $taskPackage --corpus "$taskRoot/data/calib/snapdragon-20261026/evaluation.jsonl" --sample-id 'fim-legal.cobalt_leg.billing-000714-stage-000279' --out "$taskRoot/reports/snapdragon/nano-fim-exact-tail-window32-cpu-short-control-v1.json"
if ($LASTEXITCODE -ne 0) { throw 'Smoke CPU curto falhou.' }

& $taskCpuPython -B scripts/snapdragon/probe_qnn_static_cpu.py --model $taskPackage --corpus "$taskRoot/tmp/qnn-window32-boundary-2016.jsonl" --sample-id 'boundary-static-window-2016' --out "$taskRoot/reports/snapdragon/nano-fim-exact-tail-window32-cpu-2016-control-v1.json"
if ($LASTEXITCODE -ne 0) { throw 'Smoke CPU 2016 falhou.' }

& $taskProbePython -B scripts/snapdragon/verify_prepared_qnn_receipts.py --manifest $taskPlan --phase smoke --out "$taskRoot/reports/snapdragon/nano-fim-exact-tail-window32-cpu-controls-summary-v1.json"
if ($LASTEXITCODE -ne 0) { throw 'Controle geracional ou orçamento/padding não conferiu.' }
```

Essas variáveis afetam somente o processo manual; não alteram o tracing do job que já deve ter terminado. Inspeção e smoke não são calibração. O script legado de smoke é reutilizado sem modificação: solicita greedy, usa `generation_budget` e confere `prompt_ids` contra o tokenizer do pacote.

O controle curto usa 361 tokens, prefill efetivo 384 e máximo 416. A fronteira usa 2016 tokens, prefill 2016 e máximo 2048. O resumo exige 1–32 tokens gerados; menos de 32 só passa com token final EOS/FIM explícito. Não exige primeiro token ou texto igual ao artefato legado. O corpus de fronteira é separado dos 200 exemplos de avaliação. A inspeção verifica os 456 prompts e os bytes originais do tokenizer; os smoke são dois controles de geração, não avaliação completa.

O verificador também rejeita configurações de provider não CPU, verifica se os arquivos inspecionados mudaram e mantém `qualified=false`/`htp_validated=false`. Perfis de execução HTP/NPU, compilação no Snapdragon, ausência de fallback e avaliação comparativa de qualidade continuam como evidências independentes. Nenhum upload é recomendado nesta sequência.

Validação desta preparação: sintaxe Python e três fixtures sintéticas do orçamento (32 tokens, rejeição do corte prematuro em 9 sem EOS/FIM e aceitação de EOS explícito). Nenhuma verificação foi executada sobre o pacote incompleto, e nenhuma inferência foi iniciada nesta etapa.
