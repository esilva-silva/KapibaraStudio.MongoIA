# Publicação ONNX e download pela IDE — 26/10/2026

Publicação pública autorizada explicitamente pelo usuário. O SDK oficial usou a sessão autenticada esilva; nenhum token foi exibido ou lido manualmente. O publicador existente conferiu o staging antes de enviar e verificou os arquivos na tag depois do upload.

| Repositório | Revisão da tag v0.1.0-preview | Arquivos conferidos |
|---|---|---:|
| https://huggingface.co/esilva/KapiCoder-Mongo-Nano-ONNX | b34d90bb97067ae888d51c391898a27c1707fafe | 51 |
| https://huggingface.co/esilva/KapiCoder-Mongo-Small-ONNX | 913ef5fc78e70000e118d4384cadb401fcac4f4f | 35 |

Consulta independente sem autenticação confirmou private=false e tag=main nas duas revisões. Evidências: tmp/onnx-publish-nano.log, tmp/onnx-publish-small.log e reports/export/onnx-published-revisions-20261026.json. Os manifests de upload permanecem em Repos/KapiCoder-Mongo-{Nano,Small}/hf/upload_manifest_onnx.json.

## Download e uso pela IDE

O teste explícito KapiCoderPublicDownloadManualTests.PublicCatalogDownloadsNanoAndSmallThroughPreferences passou com rede pública real, sem token, mock de transporte ou pesos do staging. Usa a janela Headless e o comando DownloadModelCommand de Preferências, o downloader e armazenamento reais. Afirma Nano cpu-int4 como seleção inicial e baixa Nano CPU INT4 (410666723 bytes) e Small CPU INT8 (2703298435 bytes), conferindo hashes, tamanhos, manifesto e validade no catálogo. Evidência IDE: tests/EsilvaSoft.KapibaraStudio.IntegrationTests/TestResults/onnx-public-download.trx. Não é inspeção de janela desktop nativa ou leitor de tela.

Os arquivos foram mantidos na IDE em tmp/public-onnx-models/KapiCoder-Mongo-Nano-ONNX-cpu-int4 e tmp/public-onnx-models/KapiCoder-Mongo-Small-ONNX-cpu-int8; nenhuma preferência real do usuário foi gravada. As duas capturas claro/escuro kapicoder-public-installed-*.png em ui-evidence/render-0e21642de0ad486b96cca3fdfcd1bb30 foram inspecionadas: dois modelos locais, Small selecionado após o segundo download e download desabilitado para item instalado.

Geração nativa foi repetida usando as próprias pastas baixadas: wrapper CPU com Nano, 714 aprovados (IDE tmp/onnx-public-native-nano.log); Small, quatro testes nativos aprovados de paridade/EOS/parada/cancelamento (tools/KapiLab.Tests/TestResults/onnx-public-native-small.trx). Referências sintéticas de oito prompts por tier, mesma DLL ORT por hash. A prova anterior de 500 prompts por tier continua documentada; não há nova medição de TTFT/RAM ou qualidade real.

Nano INT4 continua o padrão do catálogo; Small INT8 é alternativa CPU. Prefix cache segue desligado e DirectML experimental. Publicar não equivale a homologar GPU ou resolver pendências Atlas, simulador e locks de scripts/treino. Não houve treino, conversão, início da Fase 04 ou sufixo eosfix.
