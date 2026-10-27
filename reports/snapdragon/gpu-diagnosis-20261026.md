# Diagnóstico GPU Snapdragon — 26/10/2026

Os snapshots Safetensors e os exports públicos FP16 de Nano e Small foram baixados por revisão fixa; a identidade dos pesos, configurações e tokenizer foi conferida por SHA-256. A branch de preparação parte de `master` em `d01e04b`.

## Evidências atuais

- As sondas pequenas de HTP/QNN e DirectML passaram com fallback CPU desabilitado. Isso confirma acesso aos aceleradores, não qualifica um LLM.
- Adaptador DXGI 0: Qualcomm Adreno X1-45, driver 31.0.137.0; índice 1 é Microsoft Basic Render Driver. O detector existente classifica o identificador Qualcomm observado como desconhecido; sua recomendação automática não é homologação deste experimento.
- Nano FP16 original: 200 pedidos completados, sem exceções; mediana TTFT 1.235,19 ms, p95 1.819,89 ms, mediana decode 2,98 tokens/s, pico RSS 6,25 GiB. A convenção desse primeiro teste exclui criação do Generator e houve trabalho de preparação no notebook: não é um benchmark de release.
- Esse teste teve prefixo aceito e match textual iguais a zero. Portanto o pacote original não está aprovado para a Adreno, apesar da execução DirectML comprovada.
- O perfil inclui DML Shape e DmlFusedNode; CPU Gather/Cast pertencem à preparação da máscara de atenção. Eventos capturados não representam percentuais de trabalho, especialmente com graph capture.
- No mesmo primeiro pedido e nos mesmos arquivos FP16, Python CPU e .NET CPU geraram os mesmos 32 tokens começando por `{\r\n`. DML começou por `ando`. Desativar graph capture e usar GenAI 0.15.2 não corrigiram a divergência.
- Small FP16 original: um pedido terminou sem acrescentar token; o runtime não expôs um ID EOS. Não foi ampliado para 200 pedidos antes de investigar a qualidade.

## Experimento RMSNorm

Uma [investigação no repositório ONNX Runtime](https://github.com/microsoft/onnxruntime/issues/29739) descreve divergência de RMSNorm sob GenAI/DirectML em outro equipamento. Isso motivou uma hipótese, sem presumir causa comum.

O experimento decompõe SimplifiedLayerNormalization e SkipSimplifiedLayerNormalization em operações primitivas com acumulação FP32, mantendo pesos FP16 e arquivos originais intactos. O residual de Skip é arredondado para FP16 somente na saída; a soma FP32 alimenta a normalização. Os testes numéricos CPU cobrem também a saída residual omitida da última camada.

Na cópia Nano com 49 operadores decompostos e `ORT_DISABLE_ALL`, o pedido de controle na Adreno produziu exatamente os mesmos 32 tokens do controle CPU. A configuração persistida evita refusão pelo otimizador. A origem e a transformação possuem manifestos próprios; esse teste curto não substitui a avaliação de 200 pedidos nem demonstra desempenho ou qualidade geral.

Relatórios locais: `nano-gpu.json`, `nano-gpu-summary.json`, `nano-dotnet-cpu-control.json`, `nano-fp16-cpu-sample-control.json`, `nano-gpu-no-capture-control.json`, `nano-gpu-oga015-control.json`, `nano-gpu-rmsnorm-control.json` e `small-gpu-smoke.json`. Os perfis brutos ficam no diretório local de relatórios e não entram automaticamente no Git.

## Preparação remota

O Esilva-pc confirmou RX 7800 XT com 16 GB de VRAM no índice ROCm 0 e iGPU no índice 1. O seletor exige a identidade da RX antes de carregar pesos. As referências BF16 concluíram 200 pedidos por modelo sem erros; a comparação local aguarda recebimento dos JSONs da execução. A preparação W4A16/QNN depende da apuração do erro no microteste GPTQ e da execução/validação dos artefatos reais. Não há pacote NPU de LLM aprovado até aqui.
