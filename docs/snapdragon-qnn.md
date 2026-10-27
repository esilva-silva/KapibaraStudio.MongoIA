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

Uma máquina AMD ROCm pode expor o dispositivo por `torch.cuda`; isso não garante que GPTQModel, seus kernels ou ONNX Runtime CUDA EP sejam compatíveis com ROCm. Antes da quantização completa, registre `torch.version.hip`, `torch.cuda.is_available()`, dispositivo, versões dos pacotes e rode uma quantização GPTQ pequena com o backend exato escolhido; confirme também que a calibração usa um provider efetivamente disponível (CPU se necessário). O Nano foi posteriormente compilado e avaliado no HTP conforme a evidência abaixo; o Small corrigido também foi preparado, compilado e avaliado no HTP. Não substitua silenciosamente um estágio nem declare o resultado equivalente à receita original. Calibre com exemplos representativos de tarefas MongoDB e valide perda de qualidade.

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

A preparação mantém o target `QNNExecutionProvider` para selecionar `get_qnn_qdq_config` no Olive 0.13, enquanto a execução da calibração de ativações usa `CPUExecutionProvider`. Informe `--prepare-python <diretorio-do-python-qnn-x64>` se o ambiente QNN da máquina de preparação diferir do ambiente indicado por `--qnn-python`. O ambiente e o provider precisam ser reconhecidos pelo Olive; a geração da configuração não comprova que o fluxo executou. O GPTQ usa estratégia `join`, até 128 blocos de 2048 tokens, que não equivalem a 128 prompts; a calibração de ativações usa até 256 exemplos linha a linha. A validação local do Nano em 2016 tokens de entrada mais 32 de saída é descrita abaixo; o Small corrigido também passou seu próprio boundary, descrito abaixo.


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

## Nano NPU atual: calibração FIM exata e cauda completa

A preparação corrigida do Nano foi transferida pelo repositório privado `esilva/KapiCoder-Mongo-Snapdragon-Transfer`, commit `58717bfab63ffc6b241e0713dc065e246c22896b`, prefixo `nano/qnn-fim-exact-v5-prepared`. Seu manifesto SHA-256 é `79f6567de10e2e4a6dfd4786888a155d264089d65c841447d4e7203f09ef35cb`. Preserva tokenizer e `generation_config.json` originais. A calibração real comprovou 256 IDs/máscaras FIM exatos, sem `None`/EOS extra nem padding artificial; o GPTQ consumiu 61273 tokens em 29 blocos de 2048 mais a cauda de 1881. Ambos os hooks, `quantize_entry` e `after_prepare_dataset`, registram 30 blocos e o mesmo SHA do stream `07f7dec2ffed64e72f7d015bfd86bbce5d16f0297538c6fffd079fe9183da362`.

Os contextos estão em `C:/dev/models/KapiCoder-Mongo-Nano-QNN-Compiled/fim-exact-v1-contexts`; o pacote avaliado é `mixed-htp-fim-exact-v1` e o promovido é `validated-fim-exact-v1` sob a mesma raiz. Este Nano passou 200 casos e o boundary de 2016+32, com janela 32/cache 2048/greedy até 32. Qualidade: 39% de match e 2,46 tokens de prefixo aceito (BF16: 40% e 2,585). TTFT mediano 190,184 ms/p95 303,207 ms, decode 57,613 tokens/s e pico RSS 946950144 bytes. Evidências em `reports/snapdragon/`: `nano-npu-fim-exact-200.json`, `nano-npu-fim-exact-vs-bf16.json`, `nano-npu-fim-exact-stage-summary.json`, `nano-npu-fim-exact-boundary-2048.json` e `nano-qnn-fim-exact-compile.json`.

Os perfis registram QNN nos estágios context/iterator com zero kernels CPU; embeddings/head ficam na CPU. As sondas físicas `nano-fim-exact-htp-context-probe/report.json` e `nano-fim-exact-htp-iterator-probe/report.json`, sob reports/snapdragon, verificam as wrappers HTP. Seus timings sintéticos têm escopo próprio. A comparação permanece `proxy:true`, `qualified:false`, sem executar o JavaScript nem homologar a IDE.

```powershell
# Com conta HF autorizada ao repositório privado, reproduzir a preparação fixada.
env/.venv-snapdragon-tools/Scripts/python.exe scripts/snapdragon/download_qnn_prepared.py --repo-id esilva/KapiCoder-Mongo-Snapdragon-Transfer --revision 58717bfab63ffc6b241e0713dc065e246c22896b --subfolder nano/qnn-fim-exact-v5-prepared --source C:/dev/models/KapiCoder-Mongo-Nano/c03bfe15ba1ff4e5b58a4b0953349eca82bb43f1 --manifest-sha256 79f6567de10e2e4a6dfd4786888a155d264089d65c841447d4e7203f09ef35cb --require-original-tokenizer

# Para recompilar, escolher diretórios novos e usar o pacote preparado baixado.
env/.venv-snapdragon-arm64/Scripts/python.exe scripts/snapdragon/compile_qnn_context.py --model <preparado>/context.onnx --model <preparado>/iterator.onnx --manifest <preparado>/export_manifest.json --affinity-htp --free-dim past_sequence_length=2048 --weight-sharing --out-dir <novos-contextos>
env/.venv-snapdragon-arm64/Scripts/python.exe scripts/snapdragon/assemble_qnn_package.py --prepared <preparado> --compiled <novos-contextos> --source-tokenizer C:/dev/models/KapiCoder-Mongo-Nano/c03bfe15ba1ff4e5b58a4b0953349eca82bb43f1 --out-dir <novo-pacote-misto>
env/.venv-snapdragon-arm64/Scripts/python.exe scripts/snapdragon/run_sample.py --model C:/dev/models/KapiCoder-Mongo-Nano-QNN-Compiled/validated-fim-exact-v1 --corpus data/calib/snapdragon-20261026/evaluation.jsonl --backend qnn --out reports/snapdragon/nano-npu-fim-exact-repeat.json
```

O fallback permanece desabilitado como string `disable_cpu_ep_fallback='1'` em context/iterator. Cache QNN fixa `max_length=2048`; o loop limita 32 novos tokens manualmente. O pacote preparado CPU usa orçamento padding+32. A transferência privada final passou download, verificação de origem original e smoke NPU: commit `a42805ec7740e84dd392e316eab3486dd4e130d7`, prefixo `nano/htp-fim-exact-v1`, manifesto SHA-256 `0720f8859521628fc43aea38d73c5cc6d7e32156e1bcb98c9ecff3199ecf4abe`. `configs/snapdragon.json` já registra esses pins em `npu_exports.nano` (repo_id/revision/subfolder/manifest_sha256). Leia esses campos ao baixar a release final; os pins de preparação acima e os legados abaixo têm escopos diferentes.

Small NPU corrigido também passou calibração, 200 casos, boundary e transferência final privada com download e geração NPU da cópia baixada. As GPUs Nano/Small FP16 permanecem validadas e inalteradas. Nano CPU int4 segue padrão da IDE.

## Comparação atual das quatro variantes

Todos os resultados usam os mesmos 200 casos e permanecem `proxy:true`, `qualified:false`. GPU usa FP16 DirectML; NPU usa a calibração corrigida `fim-exact-tail-v1` com transformer HTP e embeddings/head CPU. Cada variante passou o boundary de 2016+32.

| Variante | Protocolo | Match / BF16 | Prefixo aceito / BF16 | TTFT p50 | TTFT p95 | Decode mediano | Pico RSS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Nano GPU | 1 processo, 200 | 39,5% / 40% | 2,57 / 2,585 | 1233,471 ms | 1982,475 ms | 3,691 tokens/s | 8813236224 bytes |
| Nano NPU corrigido | 1 processo, 200 | 39% / 40% | 2,46 / 2,585 | 190,184 ms | 303,207 ms | 57,613 tokens/s | 946950144 bytes |
| Small GPU | 8 processos × 25 | 42% / 42% | 2,57 / 2,55 | 2202,518 ms | 3134,338 ms | 2,822 tokens/s | 6238363648 bytes por filho |
| Small NPU corrigido | 1 processo, 200 | 41% / 42% | 2,585 / 2,55 | 372,960 ms | 579,178 ms | 25,833 tokens/s | 1971896320 bytes |

RSS é memória do processo, não VRAM. Small GPU não comprovou estabilidade contínua: a tentativa em processo único falhou após 175 casos. As medianas de decode GPU usam observações não nulas; o GenAI pode omitir EOS. Os resultados NPU corrigidos incluem os eventos EOS registrados pelo executor. As releases privadas Nano e Small passaram na conferência de todos os arquivos por download e na geração NPU das cópias baixadas.

## Small NPU corrigido: 200 casos e boundary validados

O pacote avaliado é `C:/dev/models/KapiCoder-Mongo-Small-QNN-Compiled/mixed-htp-fim-exact-v1`; sua cópia entregue é `validated-fim-exact-v1` na mesma raiz, com os mesmos bytes de runtime avaliados. Passou 200 casos em processo único e o boundary de 2016+32. Os perfis registram 1667 kernels QNN no context e 2363 no iterator, sem kernels CPU nesses estágios; embeddings/head usam CPU.

Evidências sob `reports/snapdragon/`: `small-npu-fim-exact-200.json`, `small-npu-fim-exact-vs-bf16.json`, `small-npu-fim-exact-stage-summary.json`, `small-npu-fim-exact-boundary-2048.json` e `small-qnn-fim-exact-compile.json`. As sondas físicas `small-fim-exact-htp-context-probe/report.json` e `small-fim-exact-htp-iterator-probe/report.json` produziram CSV HTP; seus timings sintéticos não são métricas de geração.

A preparação privada está em `esilva/KapiCoder-Mongo-Snapdragon-Transfer`, commit `2c6f3ce72c4ffdf7ff1f4eb449efd1f565b9fd95`, prefixo `small/qnn-fim-exact-v3-prepared`, manifesto SHA-256 `db2be4555d5b501afa8782f8d270bcb304ffb2fadc747f6e26da7a270653f450`. Sua auditoria comprovou 256 IDs/máscaras FIM exatos, 61273 tokens GPTQ, 30 blocos incluindo a cauda de 1881 e hashes idênticos nos dois hooks downstream; ativações QUInt16 e KV head 128. Os arquivos originais do tokenizer e de geração foram preservados.

```powershell
# Reproduzir inferência do pacote Small avaliado, sem sobrescrever evidências.
env/.venv-snapdragon-arm64/Scripts/python.exe scripts/snapdragon/run_sample.py --model C:/dev/models/KapiCoder-Mongo-Small-QNN-Compiled/mixed-htp-fim-exact-v1 --corpus data/calib/snapdragon-20261026/evaluation.jsonl --backend qnn --out reports/snapdragon/small-npu-fim-exact-repeat.json

# Obter os pins finais verificados do Small no config.
$smallConfig = Get-Content configs/snapdragon.json -Raw | ConvertFrom-Json
$smallExport = $smallConfig.npu_exports.small
$smallRawRevision = $smallConfig.models.small.revision
env/.venv-snapdragon-tools/Scripts/python.exe scripts/snapdragon/download_qnn_prepared.py --repo-id $smallExport.repo_id --revision $smallExport.revision --subfolder $smallExport.subfolder --manifest-sha256 $smallExport.manifest_sha256 --source "C:/dev/models/KapiCoder-Mongo-Small/$smallRawRevision" --require-original-tokenizer
```

O comando de download final exige que `npu_exports.small` já contenha os pins verificados da release compilada; os pins da preparação acima são de outro artefato. A release final privada é o commit `54e169e170f6f3ba51eb0ebc3a37f8f83268a08a`, prefixo `small/htp-fim-exact-v1`, manifesto SHA-256 `e82e3cc5f6bcd0ffc3b31d5a4ed76cf3fb122726537d68b48742cde8dcc15ebb`. Passou download completo, identidade da origem/tokenizer e geração real NPU da cópia baixada. Não há homologação na IDE nem execução do JavaScript gerado. Nano CPU int4 continua padrão da IDE.

## Histórico legado: Nano NPU janela 32

O pacote legado é `C:/dev/models/KapiCoder-Mongo-Nano-QNN-Compiled/mixed-htp-window32-v1`. Seus números permanecem históricos e não atendem ao aceite final `fim-exact-tail-v1`. Ele passou a avaliação proxy de 200 prompts e o teste de limite com 2016 tokens de entrada e 32 de saída, sem truncamento, no HTP. A comparação `reports/snapdragon/nano-npu-window32-vs-bf16.json` registra match textual de 39,5%, prefixo aceito médio de 2,495 tokens, TTFT mediano de 188,52 ms (p95 294,36 ms) e decode mediano de 57,62 tokens/s. Foram 163 paradas EOS e 37 no limite de tokens; o pico RSS foi 946941952 bytes, com 82,587 s totais e 1,476 s de carga. A comparação continua `proxy: true`, `qualified: false`; não executa o JavaScript gerado nem homologa a IDE.

O perfil por estágio em `reports/snapdragon/nano-npu-window32-200-stage-summary.json` registra 1667 eventos QNN no context e 2333 no iterator, sem eventos CPU nesses dois estágios. Embeddings e lm_head permanecem em CPU, com 8000 e 5066 eventos respectivamente. Sondas diretas das duas wrappers também produziram CSV HTP com o mesmo binário compartilhado; seus tempos com entradas sintéticas não são métricas de geração.

A derivação local para janela 32 comparou os grafos de context e iterator originais: após normalizar as 388 diferenças de dimensões de sequência de 64 para 1, os protobufs completos são idênticos. A transformação muda apenas essas posições de metadata no context para 32; nós, inicializadores e dados externos permanecem iguais. Os relatórios copiados do pacote pai são evidência da janela 64, não validação da janela 32. A publicação externa da derivação preparada continua opcional e pendente de aprovação; não é necessária para usar e validar o pacote local.

Exemplo de reprodução, com diretórios de saída novos:

```powershell
$prepared64 = 'C:/dev/models/KapiCoder-Mongo-Nano-QNN-Prepared/39a55f1c17411d8d2cfa0167095bff5250f7c98e/qnn-prepared'
$prepared32 = 'C:/dev/models/KapiCoder-Mongo-Nano-QNN-Prepared/local-window32-shapes-v1'
$compiled32 = '<novo-diretorio-contextos-window32>'
$env32 = 'env/.venv-snapdragon-arm64/Scripts/python.exe'
& $env32 scripts/snapdragon/retarget_prepared_window.py --prepared $prepared64 --out-dir $prepared32 --window 32
& $env32 scripts/snapdragon/compile_qnn_context.py --model "$prepared32/context.onnx" --model "$prepared32/iterator.onnx" --manifest "$prepared32/export_manifest.json" --affinity-htp --free-dim past_sequence_length=2048 --weight-sharing --out-dir $compiled32
& $env32 scripts/snapdragon/assemble_qnn_package.py --prepared $prepared32 --compiled $compiled32 --source-tokenizer C:/dev/models/KapiCoder-Mongo-Nano/c03bfe15ba1ff4e5b58a4b0953349eca82bb43f1 --out-dir <novo-pacote-misto>
& $env32 scripts/snapdragon/run_sample.py --model C:/dev/models/KapiCoder-Mongo-Nano-QNN-Compiled/mixed-htp-window32-v1 --corpus data/calib/snapdragon-20261026/evaluation.jsonl --backend qnn --config configs/snapdragon.json --out <novo-relatorio.json>
```

O diretório `local-window32-shapes-v1` já existe nesta máquina; para reproduzir a derivação, escolha outro diretório novo e ajuste `$prepared32`. A avaliação registra a execução QNN dos estágios de transformer e as partes CPU separadamente.

A janela 64 permanece como histórico: passou 200 casos com 56,63 tokens/s e TTFT mediano de 156,76 ms, mas não atende ao limite 2016+32 porque o padding arredonda o prompt para 2048 antes da geração. O relatório é `reports/snapdragon/nano-npu-compiled-window64-200.json`. A calibração legada do Small foi concluída e arquivada; a revisão corrigida do Small também já tem 200 casos e boundary validados, descritos acima.

## Histórico legado: conjunto selecionado e consumo por fase

A amostra comum contém 256 exemplos de calibração e 200 de avaliação, com grupos separados. A contagem selecionada não representa automaticamente a cobertura efetiva de cada pré-processador.

A reconstrução de dados em `reports/snapdragon/gptq-calibration-token-summary-rx7800xt.json`, validada contra as funções instaladas do Olive/GPTQ, encontrou o mesmo comportamento para Nano e Small: 29 blocos de 2048 tokens, 59.392 tokens consumidos de 61.273 concatenados. Isso cobre 248 prompts integralmente, um parcialmente e omite sete na cauda descartada. Vinte e oito dos prompts cobertos atravessam fronteiras de bloco. É uma reconstrução determinística dos dados, versões e receitas fixados; os jobs históricos não capturaram hashes dos tokens de entrada em tempo de execução.

A calibração de ativações QNN é uma fase distinta. Nos jobs legados arquivados, Nano e Small concluíram a calibração de ativações; isso não comprova a política FIM corrigida, comprovada separadamente nos traces corrigidos dos dois modelos. O campo genérico `status=completed` do helper QNN, sozinho, só indica que a configuração foi criada e não comprova que as 256 entradas terminaram. A auditoria final exige a contagem e o estado de consumo dessa fase, além dos testes do modelo compilado.

A quantização legada não foi refeita pela auditoria inicial. A revisão corrigida abaixo executou novos GPTQ e calibração de ativações, preservando a cauda do JOIN e repetindo a avaliação do Nano. O relatório detalhado por ID e bloco está em `gptq-calibration-token-attribution-rx7800xt.json`.

## Revisão FIM exata: Nano e Small executados

A auditoria adicional encontrou uma falha real na configuração legada de calibração de ativações: `add_special_tokens=true` com `bos_token=None` acrescenta o texto literal `None` e um EOS ao prompt. Os testes de geração dos artefatos legados continuam registrados, mas suas variantes NPU aguardam uma revisão de calibração antes de serem consideradas a entrega final. Os exports GPU FP16 permanecem os mesmos.

A revisão `fim-exact-tail-v1` foi executada para Nano e Small em diretórios novos, preservando o histórico. `gptq_join_tail_v1.py` retém o bloco final de 1881 tokens e cobre os 256 prompts nos 30 blocos, sem adicionar dados ou padding. `prepare_fim_calibration.py` desativa special tokens e padding artificial na calibração de ativações, com batch1 e teto2048. O preflight local dos dois modelos confirmou os IDs e as máscaras originais de todos os 256 prompts, com comprimentos reais de8 a549 tokens; os traces reais posteriores dos dois modelos também comprovaram essa execução e a calibração completa.

O template isolado `qnn_cpu_sitecustomize.py` aceita `KAPI_QNN_FIM_PREFLIGHT=<preflight.json>` junto com `KAPI_QNN_PREP_TRACE_DIR` para conferir os IDs e as máscaras reais de cada lote antes da calibração. A política exata rejeita stride incompatível e exige256 correspondências completas. Nenhum arquivo do ambiente em uso deve ser trocado durante o job legado. Nano e Small já executaram novos GPTQ, ranges, compilação e avaliações. Não há novo treinamento.
