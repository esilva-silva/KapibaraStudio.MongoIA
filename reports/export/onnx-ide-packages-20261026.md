# Pacotes ONNX GenAI preparados para a IDE — 26/10/2026

Staging local verificado, sem upload. Nano cpu-int4 é o padrão; Small cpu-int8 é a alternativa CPU. DirectML permanece experimental, sem homologação do adaptador efetivo.

| Repositório proposto | Variante | Bytes da variante |
|---|---|---:|
| esilva/KapiCoder-Mongo-Nano-ONNX | cpu-int4 | 410666723 |
| esilva/KapiCoder-Mongo-Nano-ONNX | cpu-int8 | 1123062556 |
| esilva/KapiCoder-Mongo-Nano-ONNX | dml-fp16 | 1005607530 |
| esilva/KapiCoder-Mongo-Small-ONNX | cpu-int8 | 2703298435 |
| esilva/KapiCoder-Mongo-Small-ONNX | dml-fp16 | 3107611211 |

Origens: exports v3 existentes; sem novos pesos, treino ou conversão. GenAI 0.15.2; contrato editor-context-v1; contexto recomendado 2048 e autocomplete greedy de 32 tokens. Cache de prefixo desabilitado. Não há sufixo eosfix. A IDE precisa da correção de leitura por crescimento efetivo da sequência.

Manifestos completos: Repos/KapiCoder-Mongo-{Nano,Small}/hf/upload_manifest_onnx.json. Pastas de staging: cache/hf_stage/{nano,small}-onnx. Hashes de todas as cópias e manifestos conferidos. Cards mantêm licença Apache-2.0, origem e limitações.

Comandos locais executados:

```powershell
env/.venv/Scripts/python.exe scripts/publish/stage_hf_package.py --tier both --target onnx --write-manifest --stage
env/.venv/Scripts/python.exe scripts/publish/stage_hf_package.py --tier both --target onnx --check-manifest
env/.venv/Scripts/python.exe scripts/publish/upload_to_hf.py --tier nano --target onnx --public
env/.venv/Scripts/python.exe scripts/publish/upload_to_hf.py --tier small --target onnx --public
```

Os dois últimos comandos são somente dry-run: não leem credenciais e não enviam arquivos. Seus logs tmp/onnx-publication-plan-{nano,small}.log contêm o plano concreto de publicação pública. Nenhuma aprovação de upload foi inferida do pedido de preparar pacotes. Não se declara disponibilidade pública. Após publicação autorizada, verificar o Hub por revisão e hashes e executar a listagem real pela IDE.

Testes do publicador: 82 passaram, 2 ignorados (tmp/onnx-publish-tests.log). Paridade anterior: 500 prompts sintéticos por tier, Python e IDE sem cache e duas repetições com cache, com IDs exatos; não substitui qualidade real nem TTFT/RAM. Fonte IDE: docs/architecture/local-ai-eos-prefix-cache.md.

Locks de treinamento/scripts seguem pendentes de árvore limpa; os manifestos de upload desta preparação não pretendem resolver essa pendência.

## Provas com o staging real

O downloader real da IDE instalou Nano cpu-int4 e Small cpu-int8 usando transporte Hub simulado com FileStream dos arquivos reais. Ambos passaram nas checagens SHA-256, manifesto e LocalModelCatalog (IDE tests/EsilvaSoft.KapibaraStudio.IntegrationTests/TestResults/onnx-staged-roundtrip.trx, dois testes). Não é prova de rede pública.

O wrapper KapiLab Cpu passou 714 testes com Nano a partir de cache/hf_stage/nano-onnx/cpu-int4; os quatro testes nativos de paridade/EOS/cancelamento também passaram com cache/hf_stage/small-onnx/cpu-int8. Referências sintéticas de oito prompts de cada tier, mesma DLL ORT verificada por hash, sem novas medições de latência. Evidências IDE: tmp/onnx-staged-native-nano.log e tools/KapiLab.Tests/TestResults/onnx-staged-native-small.trx.

## Atualização posterior

O usuário autorizou publicar. Os dois pacotes foram publicados e o download real pelo fluxo da IDE passou. Revisões, hashes e provas de geração: reports/export/onnx-publication-20261026.md. As observações de ausência de upload acima descrevem a fase inicial de preparação.
