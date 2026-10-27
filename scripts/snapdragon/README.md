# Nano/Small no Snapdragon

Este fluxo usa os pesos publicados, sem novo treino. A preparação pesada é destinada ao Esilva-pc; inferência e medições finais são feitas no ESILVA-ARM. Configuração: `configs/snapdragon.json`. Nunca reutilize um export de outra geração apenas pelo nome do diretório.

As quatro variantes passaram nos mesmos 200 exemplos e no limite de 2016+32 tokens. O aceite está em `reports/snapdragon/fim-exact-final-acceptance-20261027.json`; o comparativo completo está em `reports/snapdragon/delivery-summary-fim-exact-final-v3-20261027.json`. GPU usa os pins de `experimental_exports`; NPU corrigida usa `npu_exports`, com pacotes locais `C:/dev/models/KapiCoder-Mongo-{Nano,Small}-QNN-Compiled/validated-fim-exact-v1`. As duas releases NPU privadas passaram por download, hashes, identidade da origem e geração real da cópia baixada. Small GPU usa oito processos de 25 casos; isso não demonstra estabilidade contínua em uma sessão. Não há integração na IDE nesta etapa.

## Ambiente e snapshots

Os ambientes isolados ficam em `env/.venv-snapdragon-tools` (preparação de amostras/testes) e `env/.venv-snapdragon-arm64` (ORT/QNN/GenAI). As dependências instaladas estão fixadas em `env/snapdragon-tools.lock.txt` e `env/snapdragon-arm64.lock.txt`. Use o Python da arquitetura correspondente. Não instale os dois locks no mesmo ambiente.

```powershell
python -m venv env/.venv-snapdragon-tools
env/.venv-snapdragon-tools/Scripts/python.exe -m pip install -r env/snapdragon-tools.lock.txt
env/.venv-snapdragon-tools/Scripts/python.exe scripts/snapdragon/download_models.py --config configs/snapdragon.json
```

`--root` muda o destino no Esilva-pc. Os snapshots ficam em `<root>/<nome>/<commit-HF>/`, acompanhados de `source_manifest.json`. O download é retomável. Para conferir um snapshot ou pacote transferido:

```powershell
python scripts/snapdragon/verify_manifest.py <diretorio> --out reports/snapdragon/verification.json
```

## Amostra comum

```powershell
env/.venv-snapdragon-tools/Scripts/python.exe scripts/snapdragon/prepare_samples.py --config configs/snapdragon.json --tokenizer C:/dev/models/KapiCoder-Mongo-Nano/c03bfe15ba1ff4e5b58a4b0953349eca82bb43f1 --out data/calib/snapdragon-20261026
```

São 256 exemplos de calibração e 200 de avaliação, com schemas e grupos separados. O manifesto registra geradores, configuração, tokenizer e hashes dos arquivos. Os dois snapshots publicados possuem `tokenizer.json` idêntico; confirme os IDs também no runtime. Esta amostra sintética não substitui a homologação completa. Nenhuma consulta MongoDB é executada pela preparação ou pelo executor.

## Export e execução

No Esilva-pc, use um Python que já tenha o builder GenAI e suas dependências, com o snapshot fixado disponível:

```powershell
python scripts/snapdragon/export_gpu.py --source <snapshot> --out <pacote-fp16-dml>
python scripts/snapdragon/run_sample.py --model <snapshot> --corpus <evaluation.jsonl> --backend hf --device cuda:0 --out <baseline.json>
```

`cuda:0` é também a interface PyTorch do ROCm; confirme `torch.version.hip` e a GPU real. Exports usam saída vazia e não sobrescrevem pacotes existentes. Transfira pacote, avaliação, manifesto e baseline juntos, depois confira os hashes no notebook. O pacote de exportação registra a revisão de origem.

O wheel Python `onnxruntime-genai==0.18.0` instalado para Windows ARM64 oferece QNN, mas reporta `is_dml_available() == False`; não serve como executor DirectML. A distribuição DirectML/WinML deve ser validada separadamente. O executor Python falha explicitamente se o provider solicitado estiver ausente.

O executor .NET nativo ARM64 usa GenAI.WinML 0.18.0 e dependências fixadas em `tools/SnapdragonRunner/packages.lock.json`. Execute no notebook:

```powershell
dotnet restore tools/SnapdragonRunner/SnapdragonRunner.csproj --locked-mode
dotnet build tools/SnapdragonRunner/SnapdragonRunner.csproj -c Release --no-restore
tools/SnapdragonRunner/bin/Release/net10.0/win-arm64/SnapdragonRunner.exe --backend dml --model <pacote-fp16-dml> --corpus data/calib/snapdragon-20261026/evaluation.jsonl --out reports/snapdragon/nano-gpu.json
```

O runner registra profiling por operador, incluindo eventos CPU, e só marca evidência de GPU quando observa kernels DirectML. Para a sonda independente de hardware:

```powershell
env/.venv-snapdragon-arm64/Scripts/python.exe scripts/snapdragon/make_gpu_smoke.py --out reports/snapdragon/gpu-smoke.onnx
tools/SnapdragonRunner/bin/Release/net10.0/win-arm64/SnapdragonRunner.exe --backend dml-smoke --model reports/snapdragon/gpu-smoke.onnx --out reports/snapdragon/gpu-probe.json
```

As sondas de hardware GPU e NPU passaram em 26/10/2026 neste notebook (`reports/snapdragon/gpu-probe.json` e `qnn-probe.json`), com fallback CPU desabilitado. Esses resultados não são testes dos modelos completos.

```powershell
env/.venv-snapdragon-arm64/Scripts/python.exe scripts/snapdragon/qnn_probe.py --config configs/snapdragon.json --smoke --out reports/snapdragon/qnn-probe.json
python scripts/snapdragon/run_sample.py --model <pacote> --corpus <evaluation.jsonl> --backend dml --out <gpu.json>
python scripts/snapdragon/run_sample.py --model <pacote-qnn> --corpus <evaluation.jsonl> --backend qnn --out <npu.json>
```

Receita, limitações de ROCm e diagnóstico HTP: `docs/snapdragon-qnn.md`. Um grafo pequeno aprovado no HTP não aprova o LLM. Não trocar apenas o provider de um ONNX FP16 e tratá-lo como pacote QNN W4A16.

## Comparação e aceite

```powershell
env/.venv-snapdragon-tools/Scripts/python.exe scripts/snapdragon/compare_results.py --corpus <evaluation.jsonl> --baseline <baseline.json> --candidate <gpu-ou-npu.json> --tokenizer <snapshot>/tokenizer.json --out reports/snapdragon/comparison.json
env/.venv-snapdragon-tools/Scripts/python.exe -m pytest scripts/snapdragon/tests -q
```

O comparador exige a execução completa da mesma amostra e recusa falhas, IDs faltantes e duplicados. Registra prefixo aceito, match textual, TTFT p50/p95 e velocidade. Resultados fora da IDE continuam `proxy: true`, `qualified: false`. Disponibilidade/configuração de um provider não prova atribuição dos nós: guarde evidências de profiling/HTP e registre fallback. Primeiro Nano GPU, depois Nano NPU, repetindo as provas para Small. Gates de integração na IDE não são avaliados nesta etapa.

## Preparação remota isolada

```powershell
python scripts/snapdragon/make_transfer_bundle.py --out tmp/snapdragon-transfer/snapdragon-inputs.zip
```

O bundle contém apenas scripts/configurações e a amostra sintética, com manifesto SHA-256. Extraia em um diretório próprio no Esilva-pc e confira `bundle_manifest.json` com `verify_manifest.py --manifest bundle_manifest.json`. O driver usa o lock de GPU do projeto existente:

```powershell
<python-rocm> scripts/snapdragon/remote_prepare.py --config configs/snapdragon.json --project-root C:/IA/models/KapibaraStudio.MongoIA --models-root C:/IA/models/KapiPublishedSnapshots --out-root C:/IA/models/KapiSnapdragonArtifacts --corpus data/calib/snapdragon-20261026/evaluation.jsonl --models nano small
```

As exportações e referências são sequenciais. A receita QNN separa preparação e compilação; antes de executar o processo completo, rode `probe_gptq.py` no ROCm e consulte `docs/snapdragon-qnn.md`. Os scripts não configuram compartilhamento de rede nem publicam pesos. A transferência dos pacotes requer um canal de arquivos acessível entre os hosts.


## GPU experimental preferida: RMSNorm com embeddings separados (27/10/2026)

Para Nano e Small na Adreno X1-45, use o subdiretório `dml-fp16-rmsnorm-split` fixado em `experimental_exports` de `configs/snapdragon.json`: [Nano, commit ce0a7bd6c431bc6737e0b90a5a06dbba1ba8b2ea](https://huggingface.co/esilva/KapiCoder-Mongo-Nano-ONNX/tree/ce0a7bd6c431bc6737e0b90a5a06dbba1ba8b2ea/dml-fp16-rmsnorm-split) e [Small, commit ed1af97eb31b4ff504f9e09398ffe2c024a44380](https://huggingface.co/esilva/KapiCoder-Mongo-Small-ONNX/tree/ed1af97eb31b4ff504f9e09398ffe2c024a44380/dml-fp16-rmsnorm-split). Esses pacotes substituem o RMSNorm sem split como opção GPU preferida. Preservam pesos FP16, FIM, contexto de 2048 tokens e geração greedy de até 32 novos tokens; `ORT_DISABLE_ALL` está persistido na configuração.

| Modelo / protocolo | Casos | Match textual | Prefixo aceito médio | TTFT mediano | Decode mediano | Pico RSS |
| --- | --- | --- | --- | --- | --- | --- |
| Nano, processo único | 200 | 39,5% | 2,57 tokens | 1233 ms | 3,69 tokens/s | 8,21 GiB |
| Small, 8 processos novos de 25 casos | 200 | 42% | 2,57 tokens | 2203 ms | 2,82 tokens/s | 5,81 GiB por processo filho |

Evidências: `reports/snapdragon/nano-gpu-rmsnorm-split-200.json`, `nano-rmsnorm-split-vs-bf16-detailed.json`, `small-gpu-rmsnorm-split-batched-exe-200.json` e `small-rmsnorm-split-batched-vs-bf16.json`. O Nano completou os 200 casos em um processo. O Small em processo único falhou após 175 casos com `DeviceRemoved` (`0x887A0005`); o protocolo aprovado para estes 200 casos recarrega o modelo a cada 25 requisições e não demonstra estabilidade contínua em uma sessão. RSS é memória do processo, não VRAM. As medianas de decode usam somente observações não nulas; o GenAI pode omitir o EOS.

O profiling confirmou kernels DirectML e registrou operações auxiliares em CPU nos dois modelos. A comparação permanece `proxy: true`, `qualified: false`, sem execução do JavaScript gerado ou homologação na IDE. Nano CPU int4 continua como padrão da IDE. Os dois LLMs NPU/QNN corrigidos também passaram na avaliação e na transferência final descritas abaixo.

```powershell
# Baixa os snapshots de origem fixados e verifica identidade e manifesto dos exports.
env/.venv-snapdragon-tools/Scripts/python.exe scripts/snapdragon/download_models.py --config configs/snapdragon.json
env/.venv-snapdragon-tools/Scripts/python.exe scripts/snapdragon/download_exports.py --config configs/snapdragon.json --catalog experimental_exports --models nano small

# Nano: execução contínua dos 200 casos.
tools/SnapdragonRunner/bin/Release/net10.0/win-arm64/SnapdragonRunner.exe --backend dml --model C:/dev/models/KapiCoder-Mongo-Nano-ONNX/ce0a7bd6c431bc6737e0b90a5a06dbba1ba8b2ea/dml-fp16-rmsnorm-split --corpus data/calib/snapdragon-20261026/evaluation.jsonl --out reports/snapdragon/nano-gpu-rmsnorm-split-repeat.json

# Small: processos sequenciais novos, 25 casos por processo; saída e work-dir devem ser novos.
env/.venv-snapdragon-tools/Scripts/python.exe scripts/snapdragon/run_batched_gpu.py --runner tools/SnapdragonRunner/bin/Release/net10.0/win-arm64/SnapdragonRunner.exe --model C:/dev/models/KapiCoder-Mongo-Small-ONNX/ed1af97eb31b4ff504f9e09398ffe2c024a44380/dml-fp16-rmsnorm-split --corpus data/calib/snapdragon-20261026/evaluation.jsonl --batch-size 25 --work-dir reports/snapdragon/small-split-batches-repeat --out reports/snapdragon/small-gpu-rmsnorm-split-batched-repeat.json
```

Use o apphost `.exe` ARM64 com seus arquivos de runtime ao lado. Iniciar com `dotnet SnapdragonRunner.dll` selecionou ORT 1.17.1 incorreto neste ambiente e não reproduz o executor validado. O agregador verifica manifesto, IDs/ordem, hashes do executor e evidência DirectML de cada filho; registra oito cargas do modelo e o maior pico RSS dos filhos.

## Preparação RX / QNN CPU isolada (27/10/2026)

Resultados GPTQ completos e limitações: `reports/snapdragon/rx7800xt-gptq-qnn-checkpoint-20261027.md`. A referência GPTQ dos200 casos é somente qualidade; não comparar latência nula com baseline v1. NanoCPUint4 permanece padrão.

`run_olive_fwht.py` força backend Torch e registra a adaptação FWHT somente no processo GPU. `run_gptq_reference.py --quality-only` recarrega o artefato sem Triton, verifica RX e anula métricas de latência quando há preparação CPU concorrente. A seleção real exige mascarar a GPU antes de iniciar Python e confirmar o nome.

A fase posterior usa um ambiente CPU separado: ORT1.31.1 + plugin oficial onnxruntime-qnn2.7.0, GenAI0.18.0, Olive0.13.0, ONNX1.23.2, Transformers5.3.0, Optimum2.1.0. Nenhum desses pins altera o runtime da IDE. A ponte `olive_genai18_compat.py` é estrita para0.18.0 e só adapta a chamada do ModelBuilder; a equivalência final ainda depende de geração.

O template `qnn_cpu_sitecustomize.py` foi usado SOMENTE como sitecustomize.py no site-packages do venvCPU isolado. Ele registra a biblioteca oficial QNN e, quando KAPI_QNN_PREP_TRACE_DIR está definido, audita o helper QNN e os lotes efetivamente lidos. Não instalar em ambientes originais. Status completed do helper significa criação da configuração, não fim da calibração.

Para a fase CPU, iniciar processo com HIP_VISIBLE_DEVICES=-1, HF_HUB_OFFLINE=1 e caches HF próprios; não reutilizar cache compartilhado sujeito à limpeza dos intermediários Olive. A receita precisa manter target=qnn_system e QNNExecutionProvider; sq.calibration_providers deve ser CPUExecutionProvider. `run_olive_qnn_prepare.py --config <receita> --report <status>` valida essas condições e exige configuração de saída real, sem aceitar apenas retorno não vazio do Olive.

`make_qnn_recipe.py` fixa somente kv_cache_dim, obtido de head_dim ou hidden_size/num_attention_heads verificados, antes da calibração. Nunca congelar past_sequence_length para contornar o leitor. Modelos GPTQ serializados com load_kwargs:null precisam de `{}` na configuração de retomada. A v6 e a alteração derivada de capacidade2048 estão documentadas nos relatórios, sem modificar o snapshot ou os pesos originais.

Transferir todos os dados externos referenciados, inclusive transformer.onnx.data compartilhado entre context/iterator, Composite, genai_config e tokenizer. Compilar HTP apenas context/iterator; embeddings/head separados podem usar CPU. Fallback desabilitado nessas sessões não prova pipeline100%NPU. Validar shapes, IDs, geração atravessando blocos64 e capacidade2048, além de profiling real no Snapdragon.

## Nano NPU preferido: FIM exato e cauda GPTQ preservada

A revisão `fim-exact-tail-v1` é a referência atual do Nano. Pacote promovido local: `C:/dev/models/KapiCoder-Mongo-Nano-QNN-Compiled/validated-fim-exact-v1`; pacote avaliado: `mixed-htp-fim-exact-v1`; contextos compilados: `fim-exact-v1-contexts`, todos sob `C:/dev/models/KapiCoder-Mongo-Nano-QNN-Compiled/`.

A nova calibração comprovou os 256 prompts FIM originais, sem special tokens ou padding artificial. GPTQ consumiu os 61273 tokens em 30 blocos (29×2048 + cauda de 1881), com hashes coincidentes em `quantize_entry` e `after_prepare_dataset`. Os arquivos originais do tokenizer e `generation_config.json` foram preservados. A preparação está no repositório privado `esilva/KapiCoder-Mongo-Snapdragon-Transfer`, commit `58717bfab63ffc6b241e0713dc065e246c22896b`, prefixo `nano/qnn-fim-exact-v5-prepared`, manifesto SHA-256 `79f6567de10e2e4a6dfd4786888a155d264089d65c841447d4e7203f09ef35cb`.

O Nano corrigido passou 200/200 casos e o boundary de 2016+32 na NPU, mantendo janela 32, contexto 2048 e greedy até 32 tokens. Match textual: 39% contra 40% BF16; prefixo aceito médio: 2,46 contra 2,585. TTFT mediano: 190,184 ms; p95: 303,207 ms; decode mediano: 57,613 tokens/s; pico RSS: 946950144 bytes (aproximadamente 0,882 GiB). Evidências em `reports/snapdragon/`: `nano-npu-fim-exact-200.json`, `nano-npu-fim-exact-vs-bf16.json`, `nano-npu-fim-exact-stage-summary.json`, `nano-npu-fim-exact-boundary-2048.json` e `nano-qnn-fim-exact-compile.json`.

Context/iterator executam QNN HTP com zero kernels CPU; embeddings/lm_head permanecem CPU. Sondas físicas próprias estão em `reports/snapdragon/nano-fim-exact-htp-context-probe/report.json` e `nano-fim-exact-htp-iterator-probe/report.json`; seus tempos sintéticos não são latências de geração. O resultado continua proxy, `qualified:false`, sem execução do JavaScript ou homologação da IDE. Nano CPU int4 continua padrão.

```powershell
env/.venv-snapdragon-arm64/Scripts/python.exe scripts/snapdragon/run_sample.py --model C:/dev/models/KapiCoder-Mongo-Nano-QNN-Compiled/validated-fim-exact-v1 --corpus data/calib/snapdragon-20261026/evaluation.jsonl --backend qnn --out reports/snapdragon/nano-npu-fim-exact-repeat.json
```

A transferência privada final foi verificada no commit `a42805ec7740e84dd392e316eab3486dd4e130d7`, prefixo `nano/htp-fim-exact-v1`, manifesto SHA-256 `0720f8859521628fc43aea38d73c5cc6d7e32156e1bcb98c9ecff3199ecf4abe`; download, origem original e smoke NPU passaram. `configs/snapdragon.json` já contém os pins verificados: use `npu_exports.nano.repo_id`, `revision`, `subfolder` e `manifest_sha256` para o download final, com `download_qnn_prepared.py --source <snapshot-original> --require-original-tokenizer`. Não reutilize os pins do pacote legado abaixo como se fossem da revisão corrigida. A receita completa está em `docs/snapdragon-qnn.md`.

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

## Histórico legado: Nano QNN preparado

A preparação completa está no commit imutável Hugging Face `39a55f1c17411d8d2cfa0167095bff5250f7c98e`, prefixo `qnn-prepared`, branch `experimental/snapdragon-qnn-rx7800xt-20261027`. É uma publicação intermediária para compilação HTP no Snapdragon, com qualified=false. Nano CPU int4 continua como padrão da IDE. O download público dos 31 arquivos foi verificado por tamanho e SHA-256.

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('esilva/KapiCoder-Mongo-Nano-ONNX', revision='39a55f1c17411d8d2cfa0167095bff5250f7c98e', allow_patterns=['qnn-prepared/**'], local_dir='<diretorio-novo>', token=False)"
python scripts/snapdragon/verify_manifest.py <diretorio-novo>/qnn-prepared
```

`prepare_qnn_transfer.py` preserva a fonte, copia todos os dados externos, normaliza o Composite relativo e verifica atribuições contra o snapshot fixado. `inspect_qnn_prepared.py` audita grafos, valores numéricos, interfaces e IDs sem criar sessões. A cópia passou os quatro checkers e os 456 prompts. Erros de hardlinks e caminhos do source estão preservados em relatório separado.

O runner reserva geração após o padding estático sem truncar: 361 tokens com janela 64 viram 384 e exigem orçamento mínimo de 416. No pacote preparado executado em CPU, `max_length` é esse mínimo (padding + 32). No QNN compilado, o cache mantém `max_length=2048` fixo e o loop limita manualmente a geração a 32 tokens. O controle `probe_qnn_static_cpu.py --model <pacote> --out <relatorio>` confirmou 32 tokens na CPU. A janela 64 não aceita o limite de 2016 tokens de entrada mais 32 de saída: o padding já ocupa 2048. A revisão corrigida de janela 32 descrita acima já passou esse limite na NPU.

## Histórico legado: Nano NPU janela 32

O pacote legado é `C:/dev/models/KapiCoder-Mongo-Nano-QNN-Compiled/validated-window32-v2`; sua calibração não atende ao aceite final `fim-exact-tail-v1`. Completou 200/200 casos em processo único e o boundary de 2016 tokens de entrada mais 32 de saída na NPU, com janela 32 e contexto 2048. Match textual: 39,5%; prefixo aceito médio: 2,495 tokens (referência BF16: 40% e 2,585). TTFT mediano: 188,522 ms; p95: 294,36 ms; decode mediano: 57,615 tokens/s; pico RSS: 946941952 bytes (aproximadamente 0,882 GiB).

Evidências: `reports/snapdragon/nano-npu-compiled-window32-200.json`, `nano-npu-window32-vs-bf16.json`, `nano-npu-window32-200-stage-summary.json` e `nano-npu-window32-boundary-2048.json`. A cópia validada inclui relatórios, corpus, perfis, relatório de compilação e lock ARM64 em `validation/`, com manifesto completo e `VALIDATION.md`. O relatório de compilação registra versões do compilador; o lock registra dependências pretendidas de inferência, sem afirmar as versões efetivamente carregadas pelo processo.

```powershell
# Reproduzir a promoção local com evidências existentes; escolher sempre uma saída nova.
env/.venv-snapdragon-tools/Scripts/python.exe scripts/snapdragon/prepare_qnn_release.py --package C:/dev/models/KapiCoder-Mongo-Nano-QNN-Compiled/mixed-htp-window32-v1 --run reports/snapdragon/nano-npu-compiled-window32-200.json --comparison reports/snapdragon/nano-npu-window32-vs-bf16.json --stage-summary reports/snapdragon/nano-npu-window32-200-stage-summary.json --boundary reports/snapdragon/nano-npu-window32-boundary-2048.json --corpus data/calib/snapdragon-20261026/evaluation.jsonl --compile-report reports/snapdragon/nano-qnn-htp-compile-window32.json --runtime-lock env/snapdragon-arm64.lock.txt --boundary-corpus tmp/snapdragon-boundary-2016.jsonl --out-dir C:/dev/models/KapiCoder-Mongo-Nano-QNN-Compiled/validated-window32-repeat

# Executar a cópia local validada; usar novo nome para o relatório.
env/.venv-snapdragon-arm64/Scripts/python.exe scripts/snapdragon/run_sample.py --model C:/dev/models/KapiCoder-Mongo-Nano-QNN-Compiled/validated-window32-v2 --corpus data/calib/snapdragon-20261026/evaluation.jsonl --backend qnn --out reports/snapdragon/nano-npu-window32-repeat.json
```

O pipeline é misto: context e iterator usam QNN HTP com zero kernels CPU nas etapas transformer; embeddings e lm_head usam CPU. Os eventos ORT e a configuração real HTP comprovam atribuição das etapas. O CSV QNN solicitado não apareceu no teste completo; não há timings HTP em CSV do LLM. O teste micro GQA dispõe de CSV próprio, limitado ao microteste.

A receita de compilação e montagem está em `docs/snapdragon-qnn.md`: `--affinity-htp --free-dim past_sequence_length=2048 --weight-sharing`. O assembler restaura os arquivos originais exatos do tokenizer e usa `disable_cpu_ep_fallback='1'` como string nas sessões context/iterator. Permanecem FIM, greedy, até 32 novos tokens, cache QNN com `max_length=2048` fixo e limite manual de geração. São resultados sintéticos `proxy: true`, `qualified: false`; o JavaScript não foi executado e a IDE não foi homologada. Nano CPU int4 continua padrão da IDE. Small NPU corrigido passou 200 casos, boundary e verificação da transferência final privada.

A janela 64 é histórica: passou 200 casos com 39,5% de match, 2,495 tokens de prefixo aceito, TTFT mediano 156,76 ms e decode mediano 56,63 tokens/s, mas não aceita 2016+32 devido ao padding. Seus relatórios `nano-npu-compiled-window64-200.json`, `nano-npu-window64-vs-bf16.json` e `nano-npu-window64-200-stage-summary.json` permanecem preservados.

## Derivação e transferência da janela 32

`derive_static_window.py` usa StaticLLM sobre uma cópia do SplitModel, sem repetir GPTQ ou calibração. `prepare_qnn_transfer.py --preserve-tokenizer-source` preserva os arquivos originais do tokenizer, com proveniência conferida. Audite a derivação com `inspect_qnn_prepared.py --expected-window 32 --require-original-tokenizer`. A cópia preparada v2 passou 456 IDs e o controle CPU 2016+32; a cópia compilada legada acima passou também a amostra e o boundary na NPU.

Foi criado o repositório **privado** [esilva/KapiCoder-Mongo-Snapdragon-Transfer](https://huggingface.co/esilva/KapiCoder-Mongo-Snapdragon-Transfer) como alternativa segura para transferência. O pacote Nano HTP janela 32 foi transferido e baixado novamente com todos os 33 arquivos conferidos, no commit imutável `b5ce1fa141278362abbd9178113be4b8c1ecacb3`, prefixo `nano/htp-window32`; manifesto SHA-256 `c66043115b5f319d88e7bb1bfa6d35651cca1b58caf5238e614912e690a6d325`. O repositório permaneceu privado durante a operação. O resultado está em `reports/snapdragon/nano-npu-private-transfer-result.json`. A publicação em Hugging Face público continua opcional e pendente de aprovação específica após a rejeição automática; isso não bloqueia o Nano janela 32 local validado. Os pacotes anteriores permanecem preservados.

Para baixar um próximo pacote QNN preparado com verificação da revisão imutável, do manifesto e de todos os arquivos de origem, use:

```powershell
env/.venv-snapdragon-tools/Scripts/python.exe scripts/snapdragon/download_qnn_prepared.py --repo-id esilva/KapiCoder-Mongo-Nano-ONNX --revision 39a55f1c17411d8d2cfa0167095bff5250f7c98e --subfolder qnn-prepared --source C:/dev/models/KapiCoder-Mongo-Nano/c03bfe15ba1ff4e5b58a4b0953349eca82bb43f1 --manifest-sha256 7c757d18321779c2b8b29c0d8a366aee30dff38fe7a3ec023742bb90ad0a39f6
```

O comando acima reproduz o pacote intermediário de janela 64. O destino padrão é `C:/dev/models/<repo>/<commit>/qnn-prepared`. Para outra revisão, use seus pins efetivos e acrescente `--require-original-tokenizer`. O verificador não alega ter validado IDs de prompts; essa verificação ocorre na execução dos exemplos.

A comparação consolidada das quatro variantes está em `reports/snapdragon/delivery-summary-fim-exact-final-v3-20261027.json`. Os resumos anteriores registram suas respectivas revisões históricas. Para regenerá-la após conferir as evidências, escolhendo uma saída nova:

```powershell
env/.venv-snapdragon-tools/Scripts/python.exe scripts/snapdragon/summarize_delivery.py --root C:/dev/KapibaraStudio.MongoIA --out reports/snapdragon/delivery-summary-repeat.json
```

O resumo verifica os arquivos reais de calibração e avaliação, separação de grupos, ordem dos IDs, protocolos, perfis e revisões das publicações. Ele mantém a conclusão geral falsa enquanto as variantes NPU tiverem requisitos pendentes.


Download do Nano NPU legado, com a conta Hugging Face autenticada que tem acesso ao repositório privado:

```powershell
env/.venv-snapdragon-tools/Scripts/python.exe scripts/snapdragon/download_qnn_prepared.py --repo-id esilva/KapiCoder-Mongo-Snapdragon-Transfer --revision b5ce1fa141278362abbd9178113be4b8c1ecacb3 --subfolder nano/htp-window32 --source C:/dev/models/KapiCoder-Mongo-Nano/c03bfe15ba1ff4e5b58a4b0953349eca82bb43f1 --manifest-sha256 c66043115b5f319d88e7bb1bfa6d35651cca1b58caf5238e614912e690a6d325 --require-original-tokenizer
```

## Estado da revisão de calibração NPU

Os resultados NPU legados foram preservados, mas não servem ao aceite final: a auditoria encontrou `None`/EOS na calibração de ativações e perda da cauda JOIN GPTQ. O Nano corrigido acima já tem calibração e inferência reais; o Small corrigido também passou calibração, 200 casos e boundary. As quatro variantes passaram nos testes locais; a integração na IDE continua fora desta etapa. As variantes GPU FP16 não mudaram.


## Publicação pública dos NPUs corrigidos

Os dois pacotes finais também estão públicos, com os mesmos bytes e manifestos das releases privadas validadas. Foram conferidos por download (33 arquivos por modelo) e consulta sem autenticação. Nenhum peso ou teste de geração mudou nesta publicação.

| Modelo | Revisão pública imutável | Pasta | Manifesto SHA-256 |
| --- | --- | --- | --- |
| [Nano](https://huggingface.co/esilva/KapiCoder-Mongo-Nano-ONNX/tree/a85d11c9f8db2f4388836c991b292b1cdb160d39/htp-fim-exact-v1) | `a85d11c9f8db2f4388836c991b292b1cdb160d39` | `htp-fim-exact-v1` | `0720f8859521628fc43aea38d73c5cc6d7e32156e1bcb98c9ecff3199ecf4abe` |
| [Small](https://huggingface.co/esilva/KapiCoder-Mongo-Small-ONNX/tree/c03ba2339ba263a3881134d0acd3ee24e1a80717/htp-fim-exact-v1) | `c03ba2339ba263a3881134d0acd3ee24e1a80717` | `htp-fim-exact-v1` | `e82e3cc5f6bcd0ffc3b31d5a4ed76cf3fb122726537d68b48742cde8dcc15ebb` |

Branch pública nos dois repositórios: `experimental/snapdragon-htp-fim-exact-20261027`. Os pins privados em `configs/snapdragon.json` continuam válidos. Os recibos públicos estão em `reports/snapdragon/{nano,small}-npu-fim-exact-publication-result.json`; o inventário público foi confirmado em `fim-exact-publication-public-access-verification.json`.

```powershell
# Download público do Nano; o verificador confere arquivos, origem e tokenizer.
env/.venv-snapdragon-tools/Scripts/python.exe scripts/snapdragon/download_qnn_prepared.py --repo-id esilva/KapiCoder-Mongo-Nano-ONNX --revision a85d11c9f8db2f4388836c991b292b1cdb160d39 --subfolder htp-fim-exact-v1 --source C:/dev/models/KapiCoder-Mongo-Nano/c03bfe15ba1ff4e5b58a4b0953349eca82bb43f1 --manifest-sha256 0720f8859521628fc43aea38d73c5cc6d7e32156e1bcb98c9ecff3199ecf4abe --require-original-tokenizer

# Download público do Small.
env/.venv-snapdragon-tools/Scripts/python.exe scripts/snapdragon/download_qnn_prepared.py --repo-id esilva/KapiCoder-Mongo-Small-ONNX --revision c03ba2339ba263a3881134d0acd3ee24e1a80717 --subfolder htp-fim-exact-v1 --source C:/dev/models/KapiCoder-Mongo-Small/4ba57bc1f56a5b3cbe486dd314321ce4cdb2cae6 --manifest-sha256 e82e3cc5f6bcd0ffc3b31d5a4ed76cf3fb122726537d68b48742cde8dcc15ebb --require-original-tokenizer
```
