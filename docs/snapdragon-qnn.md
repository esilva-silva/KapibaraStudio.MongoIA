# Diagnóstico QNN no Snapdragon

O diagnóstico não baixa modelos nem altera a instalação. Execute no Python ARM64 destinado à inferência:

```powershell
python scripts/snapdragon/qnn_probe.py --out reports/snapdragon/qnn-inventory.json
python scripts/snapdragon/qnn_probe.py --out reports/snapdragon/qnn-smoke.json --smoke
python -m unittest discover -s scripts/snapdragon/tests
```

`--config` aceita JSON com `qnn_probe.backend_path` e `qnn_probe.provider_options`, opcionais. O padrão é `QnnHtp.dll`; passe o caminho real da DLL se necessário. O relatório separa fatos, impedimentos e resultado do smoke. Esse teste cria um grafo QDQ pequeno, executa o HTP com `session.disable_cpu_ep_fallback=1` e verifica a saída. `inconclusive` significa que o teste não pôde ser executado; `failed` é falha da tentativa, não prova de incompatibilidade de todo o hardware. A ausência de QNN na lista de providers pode indicar necessidade de registrar o plugin em uma versão mais nova do runtime. A instalação de `onnx`, `numpy`, `onnxruntime` e `onnxruntime-qnn` compatíveis com o Python ARM64 é necessária para o smoke; a disponibilidade de wheels deve ser verificada. Sucesso não prova suporte ao LLM nem cobertura completa de seu grafo.

## Preparação e critérios

A [receita oficial Qwen2.5 1.5B QNN](https://github.com/microsoft/olive-recipes/blob/main/Qwen-Qwen2.5-1.5B-Instruct/QNN/config.json) é um ponto de partida, não uma receita validada para estes pesos: fixa GPTQ em `cuda`, calibração no CUDA EP, contexto de 64 e `soc_model` 60. Em uma cópia da receita, configure o caminho do modelo local, ambiente QNN e opções apropriadas ao dispositivo, sem adivinhar SoC. A execução de referência é `olive run --config <copia-adaptada.json>`; fixe e registre versões, revisões, dataset de calibração e logs antes de executá-la. Não execute essa receita sem adaptação no notebook.

Uma máquina AMD ROCm pode expor o dispositivo por `torch.cuda`; isso não garante que GPTQModel, seus kernels ou ONNX Runtime CUDA EP sejam compatíveis com ROCm. Antes da quantização completa, registre `torch.version.hip`, `torch.cuda.is_available()`, dispositivo, versões dos pacotes e rode uma quantização GPTQ pequena com o backend exato escolhido; confirme também que a calibração usa um provider efetivamente disponível (CPU se necessário). Sem esses testes, W4A16 permanece hipótese. Não substitua silenciosamente um estágio nem declare o resultado equivalente à receita original. Calibre com exemplos representativos de tarefas MongoDB e valide perda de qualidade.

O [QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) documenta quantização em Python x64, inclusive em Windows ARM64, e inferência ARM64. O modelo exige operadores e shapes suportados e quantização adequada ao HTP. A compilação do contexto no dispositivo e o uso do [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai/blob/main/docs/qnn.md) dependem de artefatos e versões compatíveis. Comece pelo Nano, valide contexto, tokenizer, geração, ausência de fallback, memória e qualidade; depois repita com Small. Smoke de grafo pequeno não substitui essas provas.

## Geração da receita em duas fases

`make_qnn_recipe.py` verifica o manifesto SHA-256 do snapshot e o hash da receita vendorizada, cria um JSONL com a coluna `text` e gera configurações experimentais sem executar Olive. Use um diretório de saída separado por modelo e fase:

```powershell
python scripts/snapdragon/make_qnn_recipe.py --source <snapshot> --calibration <calibracao.jsonl> --qnn-python <diretorio-python-qnn> --out-dir <saida-prepare> --phase prepare --gptq-device cuda
olive run --config <saida-prepare>/prepare.json
python scripts/snapdragon/make_qnn_recipe.py --source <snapshot> --calibration <calibracao.jsonl> --qnn-python <diretorio-python-qnn> --out-dir <saida-compile> --phase compile --input-model-config <artifact-root>/model_config.json --artifact-root <artifact-root> --original-artifact-root <diretorio-absoluto-original>
olive run --config <saida-compile>/compile.json
```

`--qnn-python` é o diretório do interpretador esperado pelo sistema `PythonEnvironment` do Olive, não o executável. A fase de compilação recebe o objeto serializado Olive com `type` e `config` da fase anterior. O gerador rebasa os caminhos de modelo e additional_files para o diretório transferido, rejeitando recursos ausentes e caminhos fora dele; use o caminho absoluto original Linux ou Windows em --original-artifact-root. Valide também os arquivos externos referenciados dentro dos ONNX após a transferência. `--soc-model` é opcional e só deve ser fornecido com identificação confiável. O gerador substitui a calibração CUDA por CPU e atualiza nomes de opções do ModelBuilder para Olive 0.13, configura search.max_length como 2048, sem comprovar o limite efetivo do modelo e preserva `StaticLLM` com contexto 64 da receita. Não assume que 64 é o limite de contexto de geração. Ajustes de schema, formato intermediário, pesos W4A16, compatibilidade de kernels, e compilação ainda requerem execução e avaliação real; os JSONs gerados não constituem um modelo funcional.

Antes da quantização completa, execute no ambiente de preparação com PyTorch ROCm e GPTQModel já instalados:

```powershell
python scripts/snapdragon/probe_gptq.py --source <snapshot> --calibration <calibracao.jsonl> --out <relatorio-microgptq.json> --device cuda
```

A sonda usa um Qwen2 aleatório de uma camada e até quatro prompts truncados em 64 tokens, usando o tokenizer local. Ela testa a chamada GPTQ da receita e registra versões, HIP, GPU, tempo, pico de memória e erro. O resultado não mede qualidade nem prevê a duração ou memória para o modelo completo. `cuda` é o nome da API PyTorch também em instalações ROCm; os kernels necessários ainda precisam estar disponíveis. A sonda não instala dependências nem carrega os pesos do modelo.

Por padrão, a sonda exige que o dispositivo CUDA/ROCm de índice 0 seja a **RX 7800 XT**. O relatório enumera os dispositivos visíveis e sua VRAM; se o índice 0 for a GPU integrada, a sonda para antes da quantização. Selecione a GPU discreta usando `HIP_VISIBLE_DEVICES` antes de iniciar um novo processo e confira a enumeração. `--expected-gpu` permite configurar outro identificador explícito quando apropriado.

A preparação mantém o target `QNNExecutionProvider` para selecionar `get_qnn_qdq_config` no Olive 0.13, enquanto a execução da calibração de ativações usa `CPUExecutionProvider`. Informe `--prepare-python <diretorio-do-python-qnn-x64>` se o ambiente QNN da máquina de preparação diferir do ambiente indicado por `--qnn-python`. O ambiente e o provider precisam ser reconhecidos pelo Olive; a geração da configuração não comprova que o fluxo executou. O GPTQ usa estratégia `join`, até 128 blocos de 2048 tokens, que não equivalem a 128 prompts; a calibração de ativações usa até 256 exemplos linha a linha. A capacidade real de contexto de 2048 ainda exige inspeção dos shapes de KV e teste de geração.


## Compilação direta no notebook

O helper `scripts/snapdragon/compile_qnn_context.py` usa ORT/QNN ARM64 diretamente, sem Torch ou Olive completos. Ele seleciona apenas dispositivos NPU do plugin, usa HTP e desativa fallback CPU. A lógica de contexto deriva do Olive 0.13; a composição do pipeline GenAI continua dependendo dos artefatos estáticos reais.

```powershell
env/.venv-snapdragon-arm64/Scripts/python.exe scripts/snapdragon/compile_qnn_context.py --smoke --out-dir tmp/qnn-context-smoke-20261027
env/.venv-snapdragon-arm64/Scripts/python.exe scripts/snapdragon/compile_qnn_context.py --model <context.onnx> --model <iterator.onnx> --manifest <manifesto-transferencia.json> --affinity-htp --free-dim past_sequence_length=2048 --weight-sharing --out-dir <nova-pasta-contextos>
env/.venv-snapdragon-arm64/Scripts/python.exe scripts/snapdragon/assemble_qnn_package.py --prepared <pacote-preparado> --compiled <nova-pasta-contextos> --source-tokenizer C:/dev/models/KapiCoder-Mongo-Nano/c03bfe15ba1ff4e5b58a4b0953349eca82bb43f1 --out-dir <novo-pacote-misto>
env/.venv-snapdragon-arm64/Scripts/python.exe scripts/snapdragon/run_sample.py --model <novo-pacote-misto> --corpus data/calib/snapdragon-20261026/evaluation.jsonl --backend qnn --config configs/snapdragon.json --out reports/snapdragon/nano-npu-repeat.json
```

O manifesto deve cobrir ONNX e dados externos com `path`, `bytes` e `sha256`. O diretório de saída precisa ser novo. O helper verifica os binários referenciados pelo EPContext; compilação de LLM é registrada como `compiled_not_inference_validated`. Apenas o smoke recarrega o contexto e verifica a saída fixa, com fallback desabilitado. Nem o código nem um smoke aprovado substituem geração e comparação dos LLMs completos no HTP.

`--affinity-htp` fixa GroupQueryAttention em HTP; `--free-dim past_sequence_length=2048` fixa a dimensão livre do cache, e `--weight-sharing` compartilha os pesos entre context/iterator. Informe os dois grafos nessa ordem. O assembler verifica manifesto, hashes dos artefatos compilados e identidade do snapshot do tokenizer; restaura seus arquivos originais exatos e mantém embeddings/lm_head em CPU. Context/iterator usam backend HTP e `disable_cpu_ep_fallback='1'` como string no GenAI; na sessão ORT de compilação, a entrada equivalente é `session.disable_cpu_ep_fallback` com valor string `'1'`.

O QNN compilado exige `max_length=2048` fixo para o cache; o runner limita manualmente a geração a 32 tokens, greedy, preservando FIM. O orçamento também exige prompt com padding mais 32 dentro de 2048, sem truncamento implícito. Para o pacote preparado em CPU, `max_length` é padding + 32: 361 tokens com janela 64 viram 384 e pedem 416. Não aplicar esse 416 ao cache QNN fixo.

## Evidência Nano NPU e limite pendente

O pacote Nano de janela 64 passou os 200 casos em processo único (`reports/snapdragon/nano-npu-compiled-window64-200.json`). A comparação `nano-npu-window64-vs-bf16.json` registra match textual de 39,5%, prefixo aceito médio de 2,495 tokens (BF16: 40% e 2,585), TTFT mediano de 156,76 ms/p95 243,38 ms e decode mediano de 56,63 tokens/s. Pico RSS: 958795776 bytes (0,893 GiB); tempo total: 76,355 s; carga: 1,559 s. Permanece `proxy: true`, `qualified: false`, sem executar JavaScript ou homologar a IDE.

`nano-npu-window64-200-stage-summary.json` separa a atribuição: context tem 881 kernels QNN/0 CPU, iterator 2333 QNN/0 CPU, embeddings 6428 CPU e lm_head 5066 CPU. É um pipeline misto. Os perfis ORT e a configuração real HTP sustentam a execução transformer na NPU. O CSV QNN solicitado não foi produzido para esse LLM; não há timings HTP em CSV do modelo completo. O microteste GQA tem CSV disponível, com escopo restrito ao microteste.

A janela 64 não aceita 2016 tokens de entrada mais 32 de saída: o prompt é arredondado para 2048 antes da geração. A variante janela 32 v2 passou o controle CPU 2016+32 e preserva os arquivos originais do tokenizer, mas seu upload segue bloqueado aguardando aprovação específica do usuário após rejeição automática. Ainda não há validação desse limite na NPU. Small NPU está em preparação remota; a meta NPU continua ativa até a validação da variante janela 32 e do limite, sem alterar Nano CPU int4 como padrão da IDE.

A recarga direta dos mesmos EPContexts também gerou CSV físico HTP em ambos os estágios: `reports/snapdragon/nano-qnn-compiled-context-htp.csv` e `nano-qnn-compiled-iterator-htp.csv`. As sondas `probe_compiled_qnn.py` usaram entradas sintéticas, fallback CPU desabilitado, um evento QNN e nenhum CPU por estágio, saídas finitas e encerramento com código zero. Isso comprova o backend físico; os tempos dessas sondas não são métricas de geração do corpus. Os relatórios `nano-qnn-compiled-*-direct-profile.json` fixam os hashes das wrappers e do binário compartilhado usados também no teste de 200 casos. O CSV da execução GenAI completa continua ausente; os perfis ORT dessa execução são a evidência de atribuição por estágio.
