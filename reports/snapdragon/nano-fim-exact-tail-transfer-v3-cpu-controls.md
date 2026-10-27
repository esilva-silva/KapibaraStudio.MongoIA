# Nano FIM exact tail, transfer v3: controles CPU concluídos

Pacote: `C:/IA/models/KapiSnapdragonArtifacts/nano-fim-exact-tail-transfer-v3`. Janela 32, contexto 2048, ORT GenAI 0.18.0. A inspeção estática do parent já estava aprovada; o verificador adicional de receipts, hashes do pacote e contratos geracionais também passou. Nenhum arquivo do pacote ou seu manifesto foi alterado nesta etapa.

| Controle | ID | Prompt | Após append/padding | Máximo | Novos tokens | Comprimento final |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Curto | fim-legal.cobalt_leg.billing-000714-stage-000279 | 361 | 384 | 416 | 32 | 416 |
| Fronteira | boundary-static-window-2016 | 2016 | 2016 | 2048 | 32 | 2048 |

Ambos geraram o primeiro token 1666. Não houve corte prematuro por padding; os 32 tokens novos foram obtidos. Não se exigiu saída igual ao modelo legado. A evidência é de controle CPU, sem qualificar qualidade, desempenho ou execução HTP/NPU.

Comandos usados, no worktree, com HIP/CUDA ocultos e UTF-8 no stdout. O script legado foi reutilizado sem alterações:

```powershell
$env:HIP_VISIBLE_DEVICES='-1'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
Remove-Item Env:KAPI_QNN_PREP_TRACE_DIR -ErrorAction SilentlyContinue
Remove-Item Env:KAPI_QNN_FIM_PREFLIGHT -ErrorAction SilentlyContinue
& env/.venv-snapdragon-qnn-cpu/Scripts/python.exe -B scripts/snapdragon/probe_qnn_static_cpu.py --model C:/IA/models/KapiSnapdragonArtifacts/nano-fim-exact-tail-transfer-v3 --corpus data/calib/snapdragon-20261026/evaluation.jsonl --sample-id fim-legal.cobalt_leg.billing-000714-stage-000279 --out reports/snapdragon/nano-fim-exact-tail-transfer-v3-cpu-short-control.json
& env/.venv-snapdragon-qnn-cpu/Scripts/python.exe -B scripts/snapdragon/probe_qnn_static_cpu.py --model C:/IA/models/KapiSnapdragonArtifacts/nano-fim-exact-tail-transfer-v3 --corpus tmp/qnn-window32-boundary-2016.jsonl --sample-id boundary-static-window-2016 --out reports/snapdragon/nano-fim-exact-tail-transfer-v3-cpu-2016-control.json
& env/.venv-snapdragon-rx-probe/Scripts/python.exe -B scripts/snapdragon/verify_prepared_qnn_receipts.py --manifest reports/snapdragon/nano-fim-exact-tail-window32-control-plan-v2.json --phase smoke --out reports/snapdragon/nano-fim-exact-tail-transfer-v3-cpu-controls-gate.json
```

Os destinos acima agora são evidências existentes; qualquer repetição deve usar nomes novos e atualizar o plano correspondente, sem sobrescrevê-los. O resumo `nano-fim-exact-tail-transfer-v3-cpu-controls-summary.json` registra hashes dos relatórios, script, corpora, manifesto do pacote e IDs gerados. O plano v1 foi somente preparação; o plano v2 aponta ao pacote final efetivamente controlado.
