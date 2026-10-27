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

Na cópia Nano com 49 operadores decompostos e `ORT_DISABLE_ALL`, o pedido de controle na Adreno produziu exatamente os mesmos 32 tokens do controle CPU. A configuração persistida evita refusão pelo otimizador. A origem e a transformação possuem manifestos próprios; o controle foi seguido por avaliação completa de 200 pedidos, sem exceções. O match textual foi 39,5% (referência BF16 na RX: 40%), com prefixo aceito médio de 2,58 versus 2,585 tokens. TTFT mediano 1.254,31 ms, p95 1.994,11 ms e decode mediano 3,45 tokens/s. O pico RSS foi 7,24 GiB; o crescimento de memória durante a sequência ainda requer investigação. A evidência de profiling inclui DirectML e operações CPU na preparação da máscara. A amostra é sintética e não houve execução de JavaScript nem qualificação na IDE.

Relatórios locais: `nano-gpu.json`, `nano-gpu-summary.json`, `nano-dotnet-cpu-control.json`, `nano-fp16-cpu-sample-control.json`, `nano-gpu-no-capture-control.json`, `nano-gpu-oga015-control.json`, `nano-gpu-rmsnorm-control.json` e `small-gpu-smoke.json`. Os perfis brutos ficam no diretório local de relatórios e não entram automaticamente no Git.

## Preparação remota

O Esilva-pc confirmou RX 7800 XT com 16 GB de VRAM no índice ROCm 0 e iGPU no índice 1. O seletor exige a identidade da RX antes de carregar pesos. As referências BF16 concluíram 200 pedidos por modelo sem erros; os JSONs foram recebidos e a comparação Nano está em `nano-rmsnorm-vs-bf16.json`. A preparação W4A16/QNN depende da apuração do erro no microteste GPTQ e da execução/validação dos artefatos reais. Não há pacote NPU de LLM aprovado até aqui.

## Small: controle CPU adicional

O primeiro pedido do Small gera 32 tokens de ID 0 (`!`) tanto no export FP16 original quanto na cópia com RMSNorm decomposto, usando .NET/WinML ORT 1.24. O controle Python/ORT 1.31 também reproduz o problema. A cópia decomposta na Adreno gera repetição incorreta. Assim, a falha não está isolada ao provider DirectML e nenhum desses pacotes Small foi aprovado. A varredura completa dos pesos FP16 não encontrou NaN/Inf nem intervalos externos inválidos; amostras comparadas aos tensores BF16 coincidem com o cast esperado. O próximo diagnóstico busca valores não finitos nas ativações. Relatórios: `small-fp16-cpu-control.json`, `small-rmsnorm-cpu-control.json`, `small-python-cpu-control.json`, `small-gpu-rmsnorm-control.json`, `small-weight-audit.json`.

## Entrega experimental Nano

O pacote foi publicado e todos os arquivos foram baixados e verificados por SHA-256 em [commit HF b173484e97e24978d521a2723fa300f3f1d26224](https://huggingface.co/esilva/KapiCoder-Mongo-Nano-ONNX/tree/b173484e97e24978d521a2723fa300f3f1d26224/dml-fp16-rmsnorm). Um controle real na Adreno usando a cópia baixada produziu os mesmos 32 tokens do primeiro caso da avaliação completa. A branch principal permanece preservada.

Na referência Nano BF16, 163 casos terminaram por EOS e 37 pelo limite. Na Adreno, 164 terminaram por `runtime_done_without_token` e 36 pelo limite. O runtime GenAI não acrescentou o token de parada nesses 164 casos; portanto o relatório não inventa um ID EOS observado. Nenhuma das 200 gerações apresentou exceção. A referência Small BF16 terminou 160 casos por EOS e 40 pelo limite.

## Atualização 27/10: atenção CPU e embedding Adreno

A sonda ONNX CPU do Small localizou a primeira não-finitude no GQA da camada 0: 21.248 valores inválidos; todos os logits ficaram inválidos. Promover apenas o cálculo da atenção a FP32 em uma cópia restaurou os 32 tokens de código plausível na CPU. Isso não resolveu a Adreno.

A comparação dos intermediários DirectML localizou uma divergência anterior à atenção, na leitura do embedding transposto. Dos 554.496 elementos do prefill, 542.420 diferiram dos pesos esperados. No primeiro token, os 1.536 elementos coincidem exatamente com índices reduzidos módulo 2^27 elementos FP16 (256 MiB); os demais tokens apresentam discrepâncias adicionais. Essa observação não identifica uma causa geral no driver. As operações primitivas de quadrado e redução foram consistentes com o embedding já divergente.

Dividir o Gather em duas views externas de 768 dimensões ocultas e concatenar suas saídas corrigiu o controle Small na Adreno. Os 32 IDs coincidiram com o controle CPU com atenção FP32. A matriz original do lm_head e os valores FP16 permanecem intactos; a mudança só reorganiza o lookup. TTFT do controle: 3,53 s; decode: 1,49 tokens/s. Esse controle ainda não constitui validação completa do Small.

O embedding Nano também supera por pouco o tamanho observado; será reavaliado com chunks de 448 dimensões. O pacote Nano publicado anteriormente permanece experimental e seus números de 200 exemplos não se transferem automaticamente ao novo grafo. Pacotes novos em `models/.../experimental/rmsnorm-split-fp16-v1` têm manifestos próprios e aguardam avaliação completa.

## Nano com embedding particionado: 200 casos concluídos

O novo grafo completou 200 pedidos na Adreno: match textual 39,5% contra 40% BF16; prefixo aceito 2,57 contra 2,585. TTFT mediano 1.233,47 ms, p95 1.982,48 ms, decode mediano 3,69 tokens/s e pico RSS 8,21 GiB. As 200 observações têm TTFT; 180 têm taxa de decode válida, pois o runtime pode encerrar sem acrescentar EOS e pedidos curtos não têm duas observações de token. Houve 164 encerramentos `runtime_done_without_token` e 36 pelo limite, sem exceções. Não se interpreta esse encerramento como um ID EOS observado.

O pacote foi publicado e seus 19 arquivos foram verificados no [commit HF ce0a7bd6c431bc6737e0b90a5a06dbba1ba8b2ea](https://huggingface.co/esilva/KapiCoder-Mongo-Nano-ONNX/tree/ce0a7bd6c431bc6737e0b90a5a06dbba1ba8b2ea/dml-fp16-rmsnorm-split). A comparação detalhada está em `nano-rmsnorm-split-vs-bf16-detailed.json`. RSS mede o processo, não VRAM ou memória exclusiva do acelerador; não é comparação de memória total entre a RX discreta e a Adreno UMA. Os tempos são observações no ambiente de trabalho, sem controle de benchmark de release.

O compilador QNN ARM64 gerou um EPContext de teste e binário de 49.152 bytes, recarregou o contexto e executou a saída esperada: um kernel QNN, zero kernels CPU, fallback desabilitado. O perfil HTP registra quatro threads HVX e execução no acelerador. Esse grafo Relu QDQ pequeno não qualifica Nano ou Small NPU. Evidências: `qnn-compiled-context-smoke.json` e `qnn-compiled-context-htp-profile.csv`.
