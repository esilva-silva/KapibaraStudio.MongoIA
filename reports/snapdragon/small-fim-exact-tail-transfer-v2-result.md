# Small transfer v2: execução concluída em CPU

Pacote criado em `C:/IA/models/KapiSnapdragonArtifacts/small-fim-exact-tail-transfer-v2`, sem alterações após empacotar. Receipts GPTQ/guard CPU passaram, a inspeção estática passou com `transfer_ready=true`, janela 32/head128, quatro componentes e 456 prompts verificados. Os cinco TOKEN_FILES presentes no snapshot, incluindo `generation_config.json`, permanecem byte a byte originais.

| Controle | ID | Prompt | Após append | Máximo | Novos tokens | Final |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Curto | fim-legal.cobalt_leg.billing-000714-stage-000279 | 361 | 384 | 416 | 32 | 416 |
| Fronteira | boundary-static-window-2016 | 2016 | 2016 | 2048 | 32 | 2048 |

Ambos os controles ORT GenAI 0.18.0 geraram primeiro token1666. Não se exigiu igualdade de saída com Nano ou com o modelo legado. O manifesto foi verificado novamente após as inferências, sem arquivo alterado, faltante ou adicional.

O log da execução está em `small-fim-exact-tail-transfer-v2-execution.log`; o receipt da execução em `small-fim-exact-tail-transfer-v2-execution.json`; o resultado e hashes em `small-fim-exact-tail-transfer-v2-result.json`. Os comandos executados são os do plano revisado `small-fim-exact-tail-transfer-v2-commands.ps1`, com SHA registrado no receipt.

Foram apenas dois controles CPU greedy. Nenhuma GPU, execução HTP/NPU, upload, commit ou push foi realizado. Qualidade comparativa e homologação no Snapdragon permanecem fora desta evidência.
