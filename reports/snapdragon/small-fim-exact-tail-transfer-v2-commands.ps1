$ErrorActionPreference='Stop'
$taskRoot='C:/IA/models/KapibaraStudio.MongoIA-snapdragon-rx7800xt'
$taskBase='C:/IA/models/KapiSnapdragonArtifacts/small-fim-exact-tail-v1-rx7800xt'
$taskSnapshot='C:/IA/models/KapiPublishedSnapshots/KapiCoder-Mongo-Small/4ba57bc1f56a5b3cbe486dd314321ce4cdb2cae6'
$taskPackage='C:/IA/models/KapiSnapdragonArtifacts/small-fim-exact-tail-transfer-v2'
$taskCpuPython="$taskRoot/env/.venv-snapdragon-qnn-cpu/Scripts/python.exe"
$taskReceiptPath="$taskRoot/reports/snapdragon/small-fim-exact-tail-transfer-v2-receipts.json"
foreach ($taskReportName in @('small-fim-exact-tail-transfer-v2-receipts.json','small-fim-exact-tail-transfer-v2-static-validation.json','small-fim-exact-tail-transfer-v2-cpu-short-control.json','small-fim-exact-tail-transfer-v2-cpu-2016-control.json')) { if (Test-Path -LiteralPath "$taskRoot/reports/snapdragon/$taskReportName") { throw "Relatório já existe: $taskReportName" } }
$taskTrace="$taskBase/qnn-trace-v2/qnn-qdq-65752-cd497ba454274e5d9efcf2c4e7382e5b.json"
Set-Location -LiteralPath $taskRoot
$taskJob=Get-Content "$taskBase/qnn-job-status-v2.json" -Raw|ConvertFrom-Json
$taskCpu=Get-Content $taskTrace -Raw|ConvertFrom-Json
$taskGptq=Get-Content "$taskBase/gptq-job-status-v2.json" -Raw|ConvertFrom-Json
$taskLedger=Get-Content "$taskBase/tail-v1-projected/runtime-token-ledger-v2.json" -Raw|ConvertFrom-Json
$taskPreflight=Get-Content "$taskBase/tail-v1-projected/preflight-token-ledger.json" -Raw|ConvertFrom-Json
$taskCheckpoint=Get-Content "$taskBase/gptq-inspection-v2.json" -Raw|ConvertFrom-Json
if ($taskJob.status -ne 'workflow_returned_artifact_validation_pending') { throw 'Job CPU ainda não terminou.' }
foreach ($taskPass in @('ModelBuilder','MatMulNBitsToQDQ','GraphSurgeries','DynamicToFixedShape','OnnxStaticQuantization','SplitModel','StaticLLM')) {
 if (-not ($taskJob.stages|Where-Object { $_.pass -eq $taskPass -and $_.status -eq 'completed' })) { throw "Pass incompleto: $taskPass" }
}
if ($taskCpu.status -ne 'completed' -or $taskCpu.calibration_status -ne 'input_complete' -or $taskCpu.calibration_inputs_seen -ne 256 -or $taskCpu.fim_preflight.matched -ne 256 -or $taskCpu.fim_preflight.all_match -ne $true -or $taskCpu.fim_preflight.fim_complete -ne $true) { throw 'Trace CPU/guard incompleto.' }
if ($taskCpu.calibration_providers.Count -ne 1 -or $taskCpu.calibration_providers[0] -ne 'CPUExecutionProvider' -or $taskCpu.activation_type -ne 'QUInt16') { throw 'Provider/tipo CPU incorreto.' }
if ($taskCpu.fim_preflight.preflight_sha256 -ne (Get-FileHash "$taskBase/activation-v1-projected/preflight.json" -Algorithm SHA256).Hash.ToLowerInvariant()) { throw 'Trace aponta a outro preflight.' }
$taskKv=@($taskCpu.first_input_shapes.PSObject.Properties|Where-Object { $_.Name.StartsWith('past_key_values.') })
if ($taskKv.Count -ne 56 -or @($taskKv|Where-Object { $_.Value.shape[-1] -ne 128 }).Count -gt 0) { throw 'KV Small precisa de 28 camadas e head128.' }
if ($taskCheckpoint.passed -ne $true -or $taskGptq.status -ne 'workflow_returned_artifact_validation_pending') { throw 'GPTQ/checkpoint incompleto.' }
if ($taskGptq.runtime_calibration_audit.preflight_sha256 -ne (Get-FileHash "$taskBase/tail-v1-projected/preflight-token-ledger.json" -Algorithm SHA256).Hash.ToLowerInvariant()) { throw 'GPTQ aponta a outro preflight.' }
foreach ($taskScope in @('quantize_entry','after_prepare_dataset')) {
 $taskHook=$taskGptq.runtime_calibration_audit.$taskScope
 if ($taskHook.matched -ne $true -or $taskHook.batch_count -ne 30 -or $taskHook.token_count -ne 61273 -or $taskHook.stream_token_ids_sha256 -ne $taskLedger.stream_token_ids_sha256) { throw "Hook incompleto: $taskScope" }
 for ($taskIndex=0;$taskIndex -lt 30;$taskIndex++) {
  $taskExpectedLength=if($taskIndex -eq 29){1881}else{2048}
  $taskExpectedMask=if($taskIndex -eq 29){'3d2f57fd7812a60ee2db5046baae3fc6b6efb817b2a3456e2c672a4aaed66374'}else{'3ff6c238a98ca6a7da11b2e65354523857170384301c95ae445e24514d8fc904'}
  if ($taskHook.batches[$taskIndex].tokens -ne $taskExpectedLength -or $taskHook.batches[$taskIndex].token_ids_sha256 -ne $taskLedger.batches[$taskIndex].token_ids_sha256 -or $taskHook.batches[$taskIndex].attention_mask_sha256 -ne $taskExpectedMask) { throw "Bloco/máscara divergente: $taskScope/$taskIndex" }
 }
}
if ($taskLedger.stream_token_ids_sha256 -ne '07f7dec2ffed64e72f7d015bfd86bbce5d16f0297538c6fffd079fe9183da362' -or $taskLedger.helper_sha256 -ne $taskPreflight.helper_sha256) { throw 'Pin runtime GPTQ divergente.' }
if (Test-Path -LiteralPath $taskPackage) { throw 'Destino já existe; escolher novo nome.' }
$taskReceiptSources=[ordered]@{gptq_status="$taskBase/gptq-job-status-v2.json";gptq_runtime="$taskBase/tail-v1-projected/runtime-token-ledger-v2.json";gptq_preflight="$taskBase/tail-v1-projected/preflight-token-ledger.json";activation_trace=$taskTrace;activation_preflight="$taskBase/activation-v1-projected/preflight.json";qnn_status="$taskBase/qnn-job-status-v2.json";checkpoint="$taskBase/gptq-inspection-v2.json";workflow_preflight="$taskRoot/reports/snapdragon/small-fim-exact-tail-gptq-real-preflight-rx7800xt-v1.json"}
$taskReceiptEntries=[ordered]@{}
foreach ($taskEntry in $taskReceiptSources.GetEnumerator()) { $taskReceiptEntries[$taskEntry.Key]=[ordered]@{path=$taskEntry.Value;sha256=(Get-FileHash -LiteralPath $taskEntry.Value -Algorithm SHA256).Hash.ToLowerInvariant()} }
$taskReceipt=[ordered]@{schema='kapi-small-fim-exact-tail-receipts-v2';qualified=$false;htp_validated=$false;source_revision='4ba57bc1f56a5b3cbe486dd314321ce4cdb2cae6';gptq_blocks=30;gptq_tokens=61273;gptq_stream_sha256=$taskLedger.stream_token_ids_sha256;activation_matched=256;activation_input_complete=$true;KV_head_size=128;window=32;artifact_inspection='pending';receipts=$taskReceiptEntries}
$taskReceiptJson=($taskReceipt|ConvertTo-Json -Depth 12).Replace("`r`n","`n")+"`n"
[System.IO.File]::WriteAllText($taskReceiptPath,$taskReceiptJson,[System.Text.UTF8Encoding]::new($false))
$env:HIP_VISIBLE_DEVICES='-1'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
Remove-Item Env:KAPI_QNN_PREP_TRACE_DIR -ErrorAction SilentlyContinue
Remove-Item Env:KAPI_QNN_FIM_PREFLIGHT -ErrorAction SilentlyContinue
& $taskCpuPython -B scripts/snapdragon/prepare_qnn_transfer.py --source "$taskBase/qnn-output-v2" --destination $taskPackage --provenance "$taskBase/base/prepare-manifest.json" --source-snapshot $taskSnapshot --preserve-tokenizer-source --report $taskReceiptPath --report "$taskBase/gptq-job-status-v2.json" --report "$taskBase/tail-v1-projected/runtime-token-ledger-v2.json" --report "$taskBase/tail-v1-projected/preflight-token-ledger.json" --report $taskTrace --report "$taskBase/qnn-job-status-v2.json" --report "$taskBase/gptq-inspection-v2.json" --report "$taskRoot/reports/snapdragon/small-fim-exact-tail-gptq-real-preflight-rx7800xt-v1.json"
if ($LASTEXITCODE -ne 0) { throw 'Empacotamento falhou.' }
& $taskCpuPython -B scripts/snapdragon/inspect_qnn_prepared.py --artifact-dir $taskPackage --source $taskSnapshot --samples "$taskRoot/data/calib/snapdragon-20261026" --expected-window 32 --expected-head-size 128 --require-original-tokenizer --out "$taskRoot/reports/snapdragon/small-fim-exact-tail-transfer-v2-static-validation.json"
if ($LASTEXITCODE -ne 0) { throw 'Inspeção Small falhou.' }
$taskInspection=Get-Content "$taskRoot/reports/snapdragon/small-fim-exact-tail-transfer-v2-static-validation.json" -Raw|ConvertFrom-Json
if ($taskInspection.passed -ne $true -or $taskInspection.transfer_ready -ne $true) { throw 'Pacote não está relocável/pronto para controle CPU.' }
$taskGenai=Get-Content "$taskPackage/genai_config.json" -Raw|ConvertFrom-Json
if ($taskGenai.model.context_length -ne 2048 -or $taskGenai.model.decoder.sliding_window.window_size -ne 32 -or $taskGenai.model.decoder.head_size -ne 128) { throw 'Metadados de contexto/janela/head incorretos.' }
function AssertTaskCpuProviderOptions($taskOptions) {
 foreach ($taskProviderEntry in $taskOptions.provider_options) {
  foreach ($taskProviderName in $taskProviderEntry.PSObject.Properties.Name) { if ($taskProviderName.ToLowerInvariant() -ne 'cpu') { throw "Provider não CPU: $taskProviderName" } }
 }
}
AssertTaskCpuProviderOptions $taskGenai.model.decoder.session_options
foreach ($taskGroup in $taskGenai.model.decoder.pipeline) { foreach ($taskStage in $taskGroup.PSObject.Properties) { AssertTaskCpuProviderOptions $taskStage.Value.session_options } }
& $taskCpuPython -B scripts/snapdragon/probe_qnn_static_cpu.py --model $taskPackage --corpus "$taskRoot/data/calib/snapdragon-20261026/evaluation.jsonl" --sample-id 'fim-legal.cobalt_leg.billing-000714-stage-000279' --out "$taskRoot/reports/snapdragon/small-fim-exact-tail-transfer-v2-cpu-short-control.json"
if ($LASTEXITCODE -ne 0) { throw 'Controle CPU curto falhou.' }
& $taskCpuPython -B scripts/snapdragon/probe_qnn_static_cpu.py --model $taskPackage --corpus "$taskRoot/tmp/qnn-window32-boundary-2016.jsonl" --sample-id 'boundary-static-window-2016' --out "$taskRoot/reports/snapdragon/small-fim-exact-tail-transfer-v2-cpu-2016-control.json"
if ($LASTEXITCODE -ne 0) { throw 'Controle CPU 2016 falhou.' }
