# Nano/Small no Snapdragon

Este fluxo usa os pesos publicados, sem novo treino. A preparação pesada é destinada ao Esilva-pc; inferência e medições finais são feitas no ESILVA-ARM. Configuração: `configs/snapdragon.json`. Nunca reutilize um export de outra geração apenas pelo nome do diretório.

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


## Pacote experimental Nano validado neste notebook

O pacote corrigido está em [Hugging Face, commit b173484e97e24978d521a2723fa300f3f1d26224](https://huggingface.co/esilva/KapiCoder-Mongo-Nano-ONNX/tree/b173484e97e24978d521a2723fa300f3f1d26224/dml-fp16-rmsnorm). A transformação de RMSNorm preserva pesos FP16 e exige `ORT_DISABLE_ALL`, persistido na configuração. O pacote inclui manifesto e comparação; `nano-publication-result.json` registra a verificação de todos os arquivos no commit imutável.

Na Adreno X1-45, 200 exemplos completaram sem exceções, com match textual de 39,5% contra 40% da referência BF16; TTFT mediano 1,254 s e decode mediano 3,45 tokens/s. Pico RSS 7,24 GiB. Operações auxiliares em CPU foram observadas. Resultados são proxy sintético, sem execução de JavaScript ou qualificação na IDE. O export original não produz resultados corretos neste notebook.

```powershell
# Download do artefato exato; autenticação pública não é necessária.
env/.venv-snapdragon-tools/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('esilva/KapiCoder-Mongo-Nano-ONNX',revision='b173484e97e24978d521a2723fa300f3f1d26224',allow_patterns=['dml-fp16-rmsnorm/*'],local_dir='C:/dev/models/Nano-Adreno-tested')"
tools/SnapdragonRunner/bin/Release/net10.0/win-arm64/SnapdragonRunner.exe --backend dml --model C:/dev/models/Nano-Adreno-tested/dml-fp16-rmsnorm --corpus data/calib/snapdragon-20261026/evaluation.jsonl --out reports/snapdragon/nano-gpu-repeat.json
```

Small FP16 e os dois LLMs QNN ainda não foram aprovados. Consulte `reports/snapdragon/gpu-diagnosis-20261026.md` antes de reutilizar outros pacotes.

## Preparação RX / QNN CPU isolada (27/10/2026)

Resultados GPTQ completos e limitações: `reports/snapdragon/rx7800xt-gptq-qnn-checkpoint-20261027.md`. A referência GPTQ dos200 casos é somente qualidade; não comparar latência nula com baseline v1. NanoCPUint4 permanece padrão.

`run_olive_fwht.py` força backend Torch e registra a adaptação FWHT somente no processo GPU. `run_gptq_reference.py --quality-only` recarrega o artefato sem Triton, verifica RX e anula métricas de latência quando há preparação CPU concorrente. A seleção real exige mascarar a GPU antes de iniciar Python e confirmar o nome.

A fase posterior usa um ambiente CPU separado: ORT1.31.1 + plugin oficial onnxruntime-qnn2.7.0, GenAI0.18.0, Olive0.13.0, ONNX1.23.2, Transformers5.3.0, Optimum2.1.0. Nenhum desses pins altera o runtime da IDE. A ponte `olive_genai18_compat.py` é estrita para0.18.0 e só adapta a chamada do ModelBuilder; a equivalência final ainda depende de geração.

O template `qnn_cpu_sitecustomize.py` foi usado SOMENTE como sitecustomize.py no site-packages do venvCPU isolado. Ele registra a biblioteca oficial QNN e, quando KAPI_QNN_PREP_TRACE_DIR está definido, audita o helper QNN e os lotes efetivamente lidos. Não instalar em ambientes originais. Status completed do helper significa criação da configuração, não fim da calibração.

Para a fase CPU, iniciar processo com HIP_VISIBLE_DEVICES=-1, HF_HUB_OFFLINE=1 e caches HF próprios; não reutilizar cache compartilhado sujeito à limpeza dos intermediários Olive. A receita precisa manter target=qnn_system e QNNExecutionProvider; sq.calibration_providers deve ser CPUExecutionProvider. `run_olive_qnn_prepare.py --config <receita> --report <status>` valida essas condições e exige configuração de saída real, sem aceitar apenas retorno não vazio do Olive.

`make_qnn_recipe.py` fixa somente kv_cache_dim, obtido de head_dim ou hidden_size/num_attention_heads verificados, antes da calibração. Nunca congelar past_sequence_length para contornar o leitor. Modelos GPTQ serializados com load_kwargs:null precisam de `{}` na configuração de retomada. A v6 e a alteração derivada de capacidade2048 estão documentadas nos relatórios, sem modificar o snapshot ou os pesos originais.

Transferir todos os dados externos referenciados, inclusive transformer.onnx.data compartilhado entre context/iterator, Composite, genai_config e tokenizer. Compilar HTP apenas context/iterator; embeddings/head separados podem usar CPU. Fallback desabilitado nessas sessões não prova pipeline100%NPU. Validar shapes, IDs, geração atravessando blocos64 e capacidade2048, além de profiling real no Snapdragon.
