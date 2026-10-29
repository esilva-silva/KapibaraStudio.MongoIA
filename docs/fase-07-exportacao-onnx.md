# Fase 07 · Exportação e quantização ONNX (CPU, DirectML, CUDA, NPU/QNN)

> **Atualização de andamento — 28/10/2026:** o usuário informou que a validação de hardware dos modelos Nano e Small foi concluída com sucesso nos notebooks e ambientes ARM, x86, GPU e CPU adequados. Este registro não inclui a matriz detalhada nem medições; as descrições e pendências históricas abaixo devem ser lidas nesse contexto. Isso não estende o aceite aos demais modelos ou a recursos específicos não mencionados.


**Revisão 2026-10-08b — troca do professor:** o T2 exportado é o **KapiTeacher-Mongo-14B** (DML INT4 ~8,5–9 GB + KV ~192 KiB/token) ou **7B** (~5 GB), ambos `qwen2` (Qwen2.5-Coder, Apache-2.0) pelo builder do ORT GenAI e servidos pelo `QwenCoderModelAdapter`; removidos o caminho DeepSeek/`LlamaForCausalLM` e o `slopcoder_manifest.json` para o T2; `kapibarastudio-model.json` do T2 com `architecture: qwen2` e `parameters: 14B` (ou 7B); pacotes Apache-2.0 sem Anexo A da licença DeepSeek (atribuição e aviso de modificação mantidos).

> Plano KapiCoder-Mongo. Entradas vindas de [Fase 05](fase-05-treinamento.md) (checkpoints mesclados) e [Fase 06](fase-06-avaliacao-benchmarks.md) (gates de qualidade); saída consumida pela [Fase 08](fase-08-integracao-kapibara.md).
> Legenda: **[fato]** = verificado em arquivo/URL citado; **[proposta]** = decisão deste plano; **[hipótese]** = precisa de validação antes de virar regra.

## 1. Objetivo

Transformar cada checkpoint aprovado na Fase 06 (T0…T4) em **pacotes ONNX Runtime GenAI** que a IDE carrega sem mudança de código quando possível, com:

- uma variante por alvo de hardware (CPU, GPU DirectML, CUDA opcional, NPU Snapdragon via QNN, OpenVINO só como investigação);
- paridade numérica e de geração medida contra o PyTorch;
- `kapibarastudio-model.json`, `genai_config.json`, tokenizer byte a byte do modelo base e manifesto com SHA-256;
- publicação no Hugging Face (`esilva/...`) no formato que o **Baixar modelo** da IDE já entende ([ADR-039](F:/source/esilva-silva/EsilvaSoft.KapibaraStudio/docs/10-decisoes-arquiteturais.md)).

Não é objetivo desta fase alterar a IDE: as mudanças necessárias estão em [Fase 08](fase-08-integracao-kapibara.md).

## 2. Estado de partida (fatos)

### 2.1 IDE

| Item | Estado | Fonte |
|---|---|---|
| ONNX Runtime GenAI | **0.15.2** (`Microsoft.ML.OnnxRuntimeGenAI`, `.WinML`, `.Cuda`); `Microsoft.ML.OnnxRuntime.Managed` 1.28.0 | `Directory.Packages.props` |
| Builds | `WinML` (padrão Windows, inclusive **win-arm64**), `Cpu` (Linux), `Cuda` (opcional); o `.csproj` rejeita qualquer outro `SlopOnnxBackend` | `Directory.Build.props`, `Infrastructure.LocalAi.OnnxAdapter.csproj`, `.github/workflows/release.yml` (linhas 59–62) |
| WinML | dependência transitiva `Microsoft.Windows.AI.MachineLearning` **2.1.1** | `packages.WinML.lock.json` |
| Providers conhecidos | CPU, DirectML (`dml`), CUDA, QNN (`qnn`), OpenVINO, VitisAI — NPU só "se o runtime expuser" | `OnnxExecutionProviders.cs`, `docs/26-ia-local-multimodelo.md` |
| Registro de EPs do Windows ML | **inexistente**: nenhuma chamada a `ExecutionProviderCatalog` no código | busca por `ExecutionProviderCatalog`/`EnsureReady` em `src/` (0 ocorrências) |
| Sonda de hardware | `OnnxHardwareProbe` filtra por `GetAvailableProviders()` e depois por `GetEpDevices()` | `OnnxHardwareProbe.cs` |
| Adapters | `QwenCoderModelAdapter` (`model.type = qwen2`, tokens FIM do Qwen) e `DeepSeekCoderModelAdapter` (`llama` + `slopcoder_manifest.json`) | `ModelAdapters.cs` |
| Catálogo | exige `genai_config.json` com `model.decoder.filename` dentro da pasta, `tokenizer.json`, `tokenizer_config.json` | `LocalModelCatalog.cs` |
| Nomes | `<família>-ONNX-<variante>`; `ModelDisplayNames.ParseVariant` só entende `int4`/`int8`/`fp16`/`fp32`/`bf16` com prefixo opcional `dml`/`cuda` | `ModelDisplayNames.cs` |
| Metadata | `kapibarastudio-model.json`: capacidades `autocomplete`/`chat`/`fim`/`embeddings` (nomes desconhecidos ignorados), `hardware` `cpu`/`gpu`/`npu`, `contextContract` (**contrato desconhecido torna a pasta `Invalid`**) | `LocalModelMetadataReader.cs`, `LocalModelContextContracts.cs` |
| Download | `HuggingFaceModelSource.DefaultRepositories` fixo no código: só `esilva/SlopCoder-Mongo-0.5B-ONNX` e `-1.5B-full-ONNX` | `HuggingFaceModelSource.cs` |

### 2.2 Lições da v1 que viram regra

| Lição v1 | Regra nesta fase | Fonte |
|---|---|---|
| Exportação CPU (k-quant misto) falha em DirectML (`DmlFusedNode_0_0`, `80070057`) | Variante DirectML **sempre** exportada com `execution_provider=dml` no builder | `SlopStudio.MongoAI-Compact/docs/07-onnx-selecao.md`; `docs/23-onnx-slopcoder.md` |
| INT4 DML falha com embeddings compartilhadas (`GatherBlockQuantized` não roda sob captura de grafo do DML) | INT4 DML com `op_types_to_quantize=MatMul`, `shared_embeddings=false` (embedding fp16) e sem precisão mista | `configs/export.yaml` v1 |
| Builder 0.15.2 gera INT8 inválido para Qwen com `lm_head` compartilhado (`weight_Q4` × `weight_Q8`) | INT8 por `MatMulNBitsQuantizer` pós-exportação sobre o fp32 aprovado | `scripts/export_onnx.py` (`posthoc_quantize`) |
| `save_pretrained` reserializa o tokenizer | Tokenizer dos pacotes = cópia byte a byte do modelo base; paridade de IDs do tokenizer nativo do GenAI = 100% | `docs/07-onnx-selecao.md` |
| Carregar um 2º modelo DirectML no mesmo processo derrubou o processo | Benchmarks DML um processo por pacote | `docs/INTEGRATION.md` v1 |
| Medição com CPU disputada inflou latências | Benchmarks de latência com máquina ociosa, registrados no relatório | `docs/07-onnx-selecao.md` |
| NPU nunca exportado nem homologado | Toda variante NPU nasce **experimental** até passar no hardware real (seção 7) | `docs/07-onnx-selecao.md#npu` |

### 2.3 Ecossistema em 2026-10-08 (pesquisa web)

| Tema | Fato | URL |
|---|---|---|
| ORT GenAI após 0.15.2 | NuGet `Microsoft.ML.OnnxRuntimeGenAI.WinML`: 0.16.0 (13/09/2026), 0.17.0 (25/09/2026), **0.17.1 (29/09/2026)**, dependente de `Microsoft.Windows.AI.MachineLearning` ≥ 2.1.1 | https://www.nuget.org/packages/Microsoft.ML.OnnxRuntimeGenAI.WinML |
| 0.16.0 | builder com Qwen2.5-VL, Qwen3-VL, Qwen3.5 (denso e MoE), Granite; saída estruturada por guidance e tool calling; EP AMDGPU; decodificação especulativa | https://github.com/microsoft/onnxruntime-genai/releases |
| 0.17.0 | LFM2/LFM2.5; **prefix caching**; perfis de runtime por memória de GPU; pacotes Windows ML x64 e arm64 | https://github.com/microsoft/onnxruntime-genai/releases |
| Constrained decoding | LLGuidance; tipos `lark_grammar` (recomendado, permite texto + tool call), `json_schema`, `regex`; C# `GeneratorParams.SetGuidance(type, data, enableFFTokens)`; tokens de tool call precisam ser `special` no `tokenizer.json` | https://www.mintlify.com/microsoft/onnxruntime-genai/guides/constrained-decoding (espelho; confirmar na fonte oficial) |
| Chat template e raciocínio no C# **0.15.2** | `Tokenizer.ApplyChatTemplate(template_str, messages, tools, add_generation_prompt)` — **sem parâmetro `enable_thinking`** nem kwargs de template (só `UpdateOptions(Dictionary)` genérico); `GetBotTokenId`/`GetEotTokenId` (tool call) **e `GetBorTokenId`/`GetEorTokenId` (início/fim de raciocínio)** já existem na 0.15.2 | https://github.com/microsoft/onnxruntime-genai/blob/v0.15.2/src/csharp/Tokenizer.cs |
| Generator API 0.15.2 | `AppendTokens` pode ser chamado no meio da geração, `RewindTo(newLength)` e `TokenCount()` existem → injetar `</think>` e reaproveitar KV entre passos do agente **sem** a Engine nova | https://github.com/microsoft/onnxruntime-genai/blob/v0.15.2/src/csharp/Generator.cs |
| Exemplo oficial de chat | carrega `chat_template.jinja` da pasta do modelo e passa como `template_str`; **não** trata `<think>`, raciocínio nem `enable_thinking`; tokens de tool call vêm por parâmetro e "precisam ser `special`" | https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/common.py |
| Engine 0.16 / 0.17 | 0.16: a Engine gera tool calls e continua após o resultado sem reenviar o prompt, guidance trocável por turno, KV mantido entre turnos, KV **quantizado INT4/INT8/FP8** (paged); 0.17: **prefix caching automático só para layouts de KV paginado**, rewind para turno anterior. Nenhuma nota cita parsing de raciocínio, `enable_thinking`, C# ou DirectML para esses recursos | https://github.com/microsoft/onnxruntime-genai/releases/tag/v0.16.0, https://github.com/microsoft/onnxruntime-genai/releases/tag/v0.17.0 |
| LLGuidance com raciocínio | Lark aceita referência a token por id (`<[151667]>`) ou nome, e o exemplo da doc é `start: <think> "\n" /(.\|\n)*/ </think> address` com `address: %json {…}` — bloco livre seguido de JSON restrito; `max_tokens` só como opção de lexema terminal e **desliga rollback** | https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md |
| Qwen3-8B (template e amostragem) | 36 camadas, 8 KV heads × 128, `tie_word_embeddings=false`, `max_position_embeddings` 40.960; `generation_config.json` = `do_sample` T=0,6/top-p 0,95/top-k 20; thinking: "DO NOT use greedy decoding"; non-thinking T=0,7/top-p 0,8/top-k 20; histórico multi-turno só com a resposta final; com `enable_thinking=false` o template emite `<|im_start|>assistant\n<think>\n\n</think>\n\n`; `<think>`=151667, `</think>`=151668, `<tool_call>`=151657 — **nenhum marcado `special`** | https://huggingface.co/Qwen/Qwen3-8B, https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json, https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json |
| Windows ML: EPs incluídos | **CPU e DirectML (marcado "legacy")** | https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers |
| Windows ML: EPs sob demanda | MIGraphX, NvTensorRtRtx, OpenVINO, **QNN**, VitisAI, WebGPU (experimental), via `ExecutionProviderCatalog`, Windows 11 24H2 (build 26100)+ | mesma URL |
| QNN no Windows ML | requisito: Snapdragon X Elite **X1E** / X Plus **X1P**, driver Hexagon ≥ 30.0.140.0; **X2 e o Snapdragon X "X1-26-100" (chip do nosso notebook) não aparecem na lista**; nenhuma exigência de edição (Home serve), só Windows 11 24H2 (26100)+ (página revisada em 05/10/2026) | mesma URL |
| QNN EP como atualização do Windows | KB5077526 — QNN EP 1.8.30.0 (jan/2026, QAIRT 2.40); trilha Windows ML 2.x: QNN EP 2.2480.49.0 com **QAIRT 2.48.40** (ago/2026), 2.2500.54.0 previsto para out/2026 | https://support.microsoft.com/help/5077526, https://github.com/microsoft/WindowsML/wiki/Windows-ML-Execution-Provider-Releases |
| Hardware de homologação NPU **[fato: usuário]** | **ASUS Vivobook 16 X1607QA**: Snapdragon X **X1-26-100**, 16 GB LPDDR5X soldados (compartilhados por SO, IDE e NPU; máx. 16 GB), SSD 512 GB, **Windows 11 Home** | https://www.asus.com/br/laptops/for-home/vivobook/asus-vivobook-16-x1607q/techspec/ |
| X1-26-100 × X1E/X1P | 8 núcleos Oryon até 2,97 GHz (X1E: 12 núcleos, 3,4–4,2 GHz); GPU Adreno **X1-45** ≈ 1,7 TFLOPS (X1E: X1-85 3,8–4,6 TFLOPS); **NPU Hexagon 45 TOPS — igual à X1E/X1P**; LPDDR5x-8448, 128 bits, **≈ 135 GB/s — igual à X1E** **[F: terceiros, sem ficha oficial Qualcomm indexada]**; die de 8 núcleos "Purwa", o mesmo do X1P-42-100 **[H forte: fórum/imprensa]**; HTP **v73** relatado no X1P-42-100 (`QnnHtpV73StubDrv.dll`) **[F: relato]** | https://www.notebookcheck.net/Qualcomm-Snapdragon-X1P-26-100-Processor-Benchmarks-and-Specs.942259.0.html, https://www.geektopia.es/es/product/qualcomm/snapdragon-x-x1-26-100/, https://heise.de/-11129306, https://mysupport.qualcomm.com/supportforums/s/topic/0TOdK0000000zUwWAI/windows-on-snapdragon |
| AI Hub e Copilot+ | AI Hub tem o alvo **"Snapdragon X Plus 8-Core CRD"** (Purwa) desde out/2024, além de "X Elite CRD"; *context binaries* "são compilados específicos para a arquitetura de hardware do dispositivo". O Vivobook é vendido como **Copilot+ PC** (NPU ≥ 40 TOPS) → Phi Silica/Windows AI APIs devem rodar nele | https://workbench.aihub.qualcomm.com/docs/hub/release_notes.html, https://aihub.qualcomm.com/models/qwen3_5_0_8b, https://support.microsoft.com/help/5039678 |
| LLM 1B na NPU (referência) | Llama-3.2-1B w4a16, contexto 4096, **X Elite**: 34–43 tokens/s, TTFT 0,06–1,9 s (GENIE/GENIEX); nenhum número publicado para X1-26-100/X Plus 8-core | https://huggingface.co/qualcomm/Llama-v3.2-1B-Instruct |
| `onnxruntime-qnn` | agora em repositório próprio (`onnxruntime/onnxruntime-qnn`), 2.6.0 (10/09/2026), wheels **win_amd64 e win_arm64** | https://pypi.org/project/onnxruntime-qnn/ |
| Registro de EPs | `ExecutionProviderCatalog.GetDefault().EnsureAndRegisterCertifiedAsync()` ou `FindAllProviders()` → `EnsureReadyAsync()` → `TryRegister()`; antes do registro `GetEpDevices()` só mostra CPU e DML | https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/register-execution-providers |
| DirectML | "sustained engineering": suportado, sem novas funcionalidades; recomendação de Windows ML para projetos novos | https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html |
| QNN EP (requisitos) | HTP só executa modelos **quantizados QDQ** (ativações uint8/uint16, pesos uint8; fp16 opcional); **sem shapes dinâmicos**; sem `Loop`/`If`; quantização em Python **x64**, inferência ARM64; `ep.context_enable`/`ep.context_embed_mode` para EPContext; `session.disable_cpu_ep_fallback=1` para provar cobertura total; `htp_performance_mode`, `soc_model`, `htp_arch`; weight sharing só em x86_64 | https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html |
| Receita Olive QNN (Qwen2.5-1.5B-Instruct) | passes: `CaptureSplitInfo` → `GptqModel` (4 bits, simétrico, per-channel, rotação Hadamard, `lm_head` 8 bits g32, calibração 128 amostras) → `ModelBuilder` (int4, só `Gather`) → `MatMulNBitsToQDQ` → `GraphSurgeries` → `OnnxStaticQuantization` (ativação **uint16**, 256 amostras) → `SplitModel` → `StaticLLM` (batch 1, `context_length` 64) → `EPContextBinaryGenerator` (`soc_model` 60, `burst`, weight sharing) → `ComposeOnnxModels` | https://raw.githubusercontent.com/microsoft/olive-recipes/main/Qwen-Qwen2.5-1.5B-Instruct/QNN/config.json |
| Outras receitas QNN | Qwen2.5-7B, Qwen3-0.6B (aitk), Phi-3/3.5/4-mini, Llama 3.1 8B com `x_elite_config.json` **e** `x2_elite_config.json`; nenhuma para Qwen2.5-Coder | https://github.com/microsoft/olive-recipes |
| Caminho AI Hub | ORT GenAI documenta Llama 3.2 3B / Phi-3.5 mini via Qualcomm AI Hub (`qai_hub_models`, horas de compilação) + wrappers ONNX (`gen_qnn_ctx_onnx_model.py`); pasta com `genai_config.json`, tokenizer, `quantizer.onnx`, `dequantizer.onnx`, `position-processor.onnx`, `*.bin` + `*.json` + wrappers `*.onnx` | https://onnxruntime.ai/docs/genai/howto/build-models-for-snapdragon.html, https://onnxruntime.ai/docs/genai/tutorials/snapdragon.html |
| Como a Microsoft publica para NPU | Foundry Local detecta NPU/GPU/CPU e baixa EPs; catálogo com variantes `*-qnn-npu` (ex.: `qwen2.5-7b-instruct-qnn-npu`) e `*-vitis-npu` (ex.: Qwen2.5-Coder-7B para AMD) | https://devblogs.microsoft.com/foundry/foundry-local-meets-more-silicon/, https://ai.azure.com/catalog/models/qwen2.5-7b-instruct-qnn-npu |
| Qualcomm | `qualcomm/Qwen3-8B` com ativos w4a16 para X Elite (QAIRT 2.45, runtimes GENIE/GENIEX; Genie "será descontinuado") | https://huggingface.co/qualcomm/Qwen3-8B |
| Phi Silica | modelo do sistema nos Copilot+ PCs, acessível só pela Windows App SDK; aceita adaptador **LoRA** (experimental), não pesos próprios | https://learn.microsoft.com/windows/ai/apis/phi-silica-lora |

**Consequências para este plano:**

1. QNN para LLM **não** sai do builder do GenAI como CPU/DML: exige Olive (ou AI Hub), quantização estática QDQ com calibração, shapes estáticos e contexto binário por SoC. **[fato]**
2. A IDE já publica `win-arm64` com o build **WinML**: o QNN EP pode chegar ao usuário pelo catálogo de EPs do Windows ML, sem um build `Qnn` novo — desde que a IDE registre os EPs (hoje não registra). **[fato + hipótese a validar no spike S-NPU-2]**
3. Pacotes QNN quase certamente exigem GenAI **≥ 0.17.x** e um `genai_config.json` em formato de pipeline, que o `LocalModelCatalog` atual provavelmente rejeita (procura `model.decoder.filename`). **[hipótese forte]**
4. DirectML continua válido (incluído no Windows ML, mas "legacy"/sustained engineering): mantém-se como alvo GPU da v2, sem investir em novas otimizações exclusivas de DML. **[fato + proposta]**
5. O raciocínio do T4 **não depende** do upgrade para 0.17: template, ids de raciocínio, injeção de tokens (`AppendTokens`) e reuso de KV por `RewindTo` já estão no C# 0.15.2. Ficam como **[hipótese]**: renderização fiel do template Jinja do Qwen3 pelo `ApplyChatTemplate` nativo, guidance compilada nos NuGet da IDE, e KV quantizado/prefix caching da Engine nova funcionando em DirectML. O pacote, portanto, **não** pode depender de nenhum deles (seção 5.7). **[fato + proposta]**
6. **Chip de homologação = Snapdragon X X1-26-100 (entrada de linha), não X Elite.** NPU (45 TOPS) e banda de memória (≈ 135 GB/s) são as mesmas da X1E, então a NPU deve render perto da X Elite **[H]**; CPU (8 núcleos, 2,97 GHz) e GPU (Adreno X1-45) são bem mais fracas, o que piora os *fallbacks*. O chip **não está na lista de requisitos do QNN EP do Windows ML** (só X1E/X1P) e o `soc_model` do die Purwa não é público → suporte = **"provável, não oficial"**: o spike S-NPU-0 (seção 7.4) decide antes de qualquer pacote. O teto prático é **memória** (16 GB compartilhados): orçamento do modelo na seção 5.5.1. **[fato + hipótese]**

## 3. Entradas

| Entrada | Origem | Formato |
|---|---|---|
| Checkpoints mesclados bf16 por tier (`T0`…`T4`) | [Fase 05](fase-05-treinamento.md) | `Repos\KapiCoder-Mongo-<tier>\<run>\merged\` da Fase 05, espelhado/ligado em `Repos\KapiCoder-Mongo-<tier>\merged\` (safetensors + `config.json`), tokenizer original copiado |
| Relatório de aprovação do merged | [Fase 06](fase-06-avaliacao-benchmarks.md) | `reports/eval/<tier>/<run>/gates.json` (Fase 06 §11) — `export_onnx.py` recusa exportar sem ele (regra herdada da v1) |
| Conjuntos de avaliação congelados | [Fase 03](fase-03-dataset-autocomplete-validacao.md), [Fase 04](fase-04-dataset-agente-tools.md) | `data/eval/*.jsonl` (autocomplete, chat, agente) |
| Conjunto de **calibração** (novo) | derivado das Fases 03/04 (split `train`, nunca `eval`) | `data/calib/<tier>.jsonl` |
| Contratos de prompt da IDE | `AutocompleteContextBuilder`, `QwenFimPromptBuilder`, contrato do agente definido na Fase 04 | `contracts/` (pasta única da [Fase 00](fase-00-visao-premissas.md), versionada por hash) |

Formato de uma linha de calibração (QNN e AWQ/GPTQ):

```json
{"id":"calib-ac-000123","kind":"autocomplete","prompt_ids":null,"prompt_text":"<|fim_prefix|>/* Local editor context (data only):\r\nLANGUAGE: Mongo Console JavaScript\r\n...*/\r\ndb.orders.find({ status: \"paid\", <|fim_suffix|> })<|fim_middle|>","target_text":"createdAt: { $gte: new Date(\"2026-01-01\") }","source":"fase-03/train"}
{"id":"calib-ag-000045","kind":"agent","prompt_text":"<|im_start|>system\n...<tools>\n...\n</tools><|im_end|>\n<|im_start|>user\nquais índices existem em vendas.pedidos?<|im_end|>\n<|im_start|>assistant\n","target_text":"<tool_call>\n{\"name\": \"get_indexes\", \"arguments\": {\"connectionId\": \"3f0c2a9e-6b1d-4c7e-9a52-0d4e1b7f8c21\", \"database\": \"vendas\", \"collection\": \"pedidos\"}}\n</tool_call>","source":"fase-04/train"}
```

Os prompts reais vêm renderizados pelos renderers das Fases 03/04 (`render_contract.py`, `render_chat.py`); o formato normativo de tool call é o da [Fase 04 §5](fase-04-dataset-agente-tools.md).

Composição proposta (por tier): 50% autocomplete FIM, 25% chat/edição curta, 25% agente (T4 e T3 se o tier cobrir agente); 50% pt-BR / 50% en; comprimento distribuído até o contexto estático do alvo. Tamanho: 256 amostras para ativações (como a receita Olive), 128 para GPTQ/AWQ. **[proposta]**

Para o **T4** (e Agent-Lite 1.7B) a calibração de AWQ/GPTQ e os prompts de paridade seguem a mistura do agente com raciocínio: **60% com thinking** (`target_text` começando por `<think>\n…\n</think>\n\n` e seguido de `<tool_call>` ou resposta, raciocínio de 64–1.024 tokens, pt-BR e en), **40% sem thinking** (prompt terminando no bloco vazio `<think>\n\n</think>\n\n`), e comprimentos de prompt em faixas 2k / 6k / 12k tokens para cobrir o contexto de 16k. Sem amostras longas a quantização calibra escalas só para prompts curtos. **[proposta]**

## 4. Ambientes de trabalho

| Ambiente | Máquina | Conteúdo | Uso |
|---|---|---|---|
| `.venv-export` | Ryzen 9 7900 + RX 7800 XT (Windows x64) | `torch` (ROCm já usado no treino), `transformers`, `onnxruntime-genai` **na mesma versão do runtime da IDE** (0.15.2 hoje; 0.17.1 se a ADR da Fase 08 for aceita), `onnxruntime` CPU | builder, paridade fp32, INT4/INT8 CPU |
| `.venv-winml` | idem | `onnxruntime-genai-winml` (mesma versão) + `sitecustomize.py` que pré-carrega as DLLs do pacote WinML (lição v1) | benchmark DirectML |
| `.venv-olive-qnn` | x64 (mesma máquina ou WSL2/Linux para geração de contexto) | `olive-ai`, `onnxruntime-qnn` x64, QAIRT/QNN SDK (versão fixada no manifesto), `gptqmodel` | quantização QDQ, `EPContextBinaryGenerator` (requer x64; weight sharing só x86_64) |
| Calibração/GPTQ (**local, sem nuvem** — revisão 2026-10-08) | RX 7800 XT (ROCm) ou CPU Ryzen 9 7900 (64 GB) | `.venv-olive-qnn` com o PyTorch ROCm do treino; alternativa WSL2 + ROCm oficial ([Fase 05 §4.2](fase-05-treinamento.md), ambiente B) | a receita Olive roda GPTQ e calibração em `cuda`; no ROCm o dispositivo também se chama `cuda` no PyTorch, mas o caminho Olive/GPTQModel no Windows é **[hipótese]** não testada → sonda no S-NPU-1; se falhar, GPTQ na CPU para 0.5B/1.5B (128 amostras, ~1–3 h por modelo **[hipótese]**) ou no WSL2. Horas de GPU no calendário único da Fase 05 §9 |
| `device-arm64` | **ASUS Vivobook 16 X1607QA** (Snapdragon X X1-26-100, 16 GB, Windows 11 Home) — disponível; X1E/X2 só se surgirem | build `win-arm64` WinML da IDE, `tools/ide-harness` ARM64, Python ARM64 + `onnxruntime-qnn` win_arm64 (diagnóstico e geração de contexto no próprio dispositivo), driver Hexagon e QNN EP do Windows Update | inventário do SoC (S-NPU-0), paridade e benchmark NPU (seção 7.4) |
| `.venv-cuda` (opcional) | GPU NVIDIA **física emprestada** (não há nuvem no plano) | `onnxruntime-genai-cuda` | só se CUDA entrar no release; sem máquina NVIDIA, variantes `cuda-*` não são publicadas (§5.4) |

Pastas (raiz `C:\IA\models\KapibaraStudio.MongoIA\`):

```text
configs/
  export.yaml                 # variantes por tier × alvo (seção 5)
  olive/qnn-<tier>-x1.json    # receitas Olive QNN derivadas da receita Qwen2.5-1.5B; soc_model/htp_arch da família X1 (v73) definidos no S-NPU-0
  olive/qnn-<tier>-x2e.json   # só se houver máquina X2
  # contratos de prompt: usar a pasta raiz contracts/ (Fase 00), não uma cópia em configs/
scripts/
  export_onnx.py              # herdado da v1 (builder + paridade); ganha --tier e --target
  quantize_onnx.py            # variantes CPU/DML/CUDA e relatório comparativo
  build_calibration.py        # data/calib/<tier>.jsonl a partir das Fases 03/04
  export_qnn.py               # orquestra Olive (x64) e empacota o pipeline QNN
  parity_check.py             # logits/top-1/sequência PyTorch × ONNX, qualquer provider
  package_exports.py          # v2: kapibarastudio-model.json, kapicoder_manifest.json, SHA-256
  publish_hf.py               # upload por variante, model card, verificação pós-upload
  ide_harness.py              # herdado: roda o pacote pelas DLLs da IDE
Repos\<modelo>\                 # um workspace por modelo (T2 = Repos\KapiTeacher-Mongo-14B\)
  merged\
  exports\onnx\<variante>\      # saídas brutas do builder/Olive
  exports\<Família>-ONNX-<variante>\   # pacotes finais (seção 6)
reports/export/<tier>/<variante>/    # paridade, benchmark, logs de build
```

## 5. Pipeline por alvo

Todos os alvos partem do mesmo `Repos\<modelo>\merged\` e passam primeiro pela **referência fp32 CPU** (paridade estrita). Uma variante só é empacotada se a referência do mesmo tier foi aprovada.

### 5.1 Referência fp32 (CPU)

```powershell
python scripts/export_onnx.py --tier T1 --variant-name fp32-cpu --precision fp32 --execution-provider cpu
```

- Builder do GenAI (`onnxruntime_genai.models.builder`, `hf_remote=false`), KV cache `past_key_values.*` → `present.*`.
- Gate (herdado da v1): 1º token greedy 100% e ≥ 75% de concordância de sequência em 24 tokens, 12 prompts (metade autocomplete, metade chat) **mais** 6 prompts de agente (T4); cosseno dos logits do primeiro passo ≥ 0,999; tokenizer nativo do GenAI = HF em 400 prompts (100% dos IDs, marcadores FIM e de tool call como tokens únicos).

### 5.2 CPU — INT4 e INT8 (MatMulNBits)

| Variante | Método | Parâmetros iniciais | Observação |
|---|---|---|---|
| `int4` (distribuição) | builder, `algo_config=k_quant`, `matmul_mixed_precision=last_matmul:int8`, `block_size=32`, `accuracy_level=4` | herdado da v1 | venceu RTN na v1 (0.5B: ΔAPT −0,07 × −0,12) |
| `int4-rtn` (controle) | builder RTN | `block_size=32` | só relatório |
| `int4-awq` / `int4-gptq` (candidatos) | checkpoint pré-quantizado (AutoAWQ/GPTQModel com calibração do domínio) passado ao builder como entrada quantizada | `w_bit=4`, `group_size=32/128`, 128 amostras de `data/calib` | **[hipótese]** ganho sobre k-quant no T2 7B; adotar só se ΔAPT melhorar ≥ 0,05 nos mesmos exemplos |
| `int8` | `MatMulNBitsQuantizer` pós-exportação sobre `fp32-cpu` (bug do builder 0.15.2) | `bits=8`, `block_size=32`, `accuracy_level=4`, simétrico | reavaliar o builder direto se o runtime subir para 0.17.1 |

### 5.3 GPU DirectML (RX 7800 XT; backend WinML)

| Variante | Parâmetros | Regra |
|---|---|---|
| `dml-fp16` | `--precision fp16 --execution-provider dml` | padrão GPU para T0/T1 (v1: mesma qualidade do merged) |
| `dml-int4` | `--precision int4 --execution-provider dml`, `op_types_to_quantize=MatMul`, `shared_embeddings=false`, `block_size=32`, sem precisão mista | padrão GPU para T2 (14B/7B) e T4 (8B) |

- T2 14B (~29 GB), T2 7B / T4 8B (Qwen3-8B) em FP16 ocupam ≈ 15–16 GB (14B: ~29 GB) só de pesos: **não cabem com folga em 16 GB** junto do KV cache → `dml-fp16` não é gerado para 7B+. **[proposta]** Orçamento de memória do T4 com raciocínio e 16k de contexto: seção 5.7.
- **T2 padrão = KapiTeacher-Mongo-14B** (Qwen2.5-Coder-14B, `Qwen2ForCausalLM`/`qwen2`, 48 camadas, hidden 5120, 8 KV heads GQA, vocab 152.064, embeddings **não** atrelados **[F a confirmar no `config.json`]**; [Fase 05 §7](fase-05-treinamento.md)) **ou 7B** se o alternativo vencer o bake-off (Fase 01): exportado com `-e dml` INT4 pelo builder com arquitetura `qwen2` e servido pelo `QwenCoderModelAdapter` existente (FIM Qwen; sem `DeepSeekCoderModelAdapter`, sem manifesto DeepSeek). 14B: pesos DML INT4 ≈ 8,5–9 GB + KV-cache GQA ≈ 192 KiB/token (≈ 0,4 GB a 2k, ≈ 0,8 GB a 4k) → cabe em 16 GB, mas só com "um modelo por vez"; TTFT maior que o 7B, a medir **[H]**; 7B: ≈ 5 GB (KV 4 heads ≈ 112 KiB/token **[H]**). O KV e o gate de VRAM do T2 (Fase 06) valem por tamanho escolhido.
- Qwen2.5-Coder-7B e 14B têm `tie_word_embeddings=false` (0.5B/1.5B têm `true`) **[hipótese — conferir no `config.json` do base escolhido na [Fase 01](fase-01-selecao-modelos-base.md)]**; se confirmado, o problema de `GatherBlockQuantized` não ocorre, mas mantém-se `shared_embeddings=false` por segurança.
- Benchmark sempre no `.venv-winml`, um processo por pacote.

### 5.4 CUDA (opcional)

`cuda-int4` (builder `--execution-provider cuda`, INT4 RTN/AWQ) e `cuda-fp16` só para T2/T4, gerados se houver GPU NVIDIA para validar; caso contrário, **não publicados** (regra da v1: sem evidência, sem pacote). O build `Cuda` da IDE já existe.

### 5.5 NPU Snapdragon — QNN (experimental)

Pipeline proposto, adaptado da receita oficial Olive para Qwen2.5-1.5B (seção 2.3):

| # | Passo (Olive) | Parâmetros iniciais propostos | Racional |
|---|---|---|---|
| 1 | `CaptureSplitInfo` | `num_splits` por tamanho (0.5B/1.5B: 1–2; 3B: 2–4), `unique_embeds_lm_head_splits=true` | embedding/lm_head separados; partes menores para o HTP |
| 2 | `GptqModel` | 4 bits, simétrico, per-channel (`group_size=-1`), rotação Hadamard, `lm_head` 8 bits g32; **calibração `data/calib/<tier>.jsonl` (128 amostras)** em vez de wikitext2 | domínio MongoDB/mongosh/JSON; wikitext distorce escalas de tokens raros como `$gte`, `ObjectId` |
| 3 | `ModelBuilder` | int4, `block_size=32`, quantiza só `Gather` | igual à receita |
| 4 | `MatMulNBitsToQDQ` | zero point, exclui `/lm_head/MatMulNBits` | pesos 4 bits em QDQ ⇒ **w4a16** |
| 5 | `GraphSurgeries` | `RemoveRopeMultiCache`, `AttentionMaskToSequenceLengths`, `RemoveGidxFromMatMulNBits`, `SimplifiedLayerNormToL2Norm` | igual à receita |
| 6 | `OnnxStaticQuantization` | ativações **uint16**, 256 amostras do domínio, exclui `GatherBlockQuantized`/`GroupQueryAttention`/`MatMulNBits` | a16; uint8 nas ativações só como experimento (`w4a8`) |
| 7 | `SplitModel` | — | |
| 8 | `StaticLLM` | `batch_size=1`, `context_length=64` (bloco de prefill) | shapes estáticos exigidos pelo QNN EP |
| 9 | `EPContextBinaryGenerator` | `htp_arch=73`; `soc_model` **do X1-26-100 (die Purwa), lido no S-NPU-0** — a receita usa 60, que é o da X Elite (SC8380XP, v73) **[F: cards QNN publicados]**; o id do Purwa não é público **[H]**. Ordem de tentativa: (a) id lido do dispositivo; (b) contexto `soc_model=60` testado no Vivobook (mesma v73; pode falhar por VTCM/SoC); (c) contexto gerado **no próprio Vivobook** com `onnxruntime-qnn` win_arm64 (`ep.context_enable=1`, sem weight sharing). `htp_performance_mode=burst`, `htp_graph_finalization_optimization_mode=3`, weight sharing (só na geração x64) | contexto pré-compilado: carga rápida, sem compilar no PC do usuário |
| 10 | `ComposeOnnxModels` + `genai_config.json` | janela total do KV **2048** (T0/T1/T3) e **4096** só se couber no orçamento | ver limites abaixo |

Variantes NPU:

| Variante | Conteúdo | Situação |
|---|---|---|
| `qnn-w4a16` | contexto binário da família Snapdragon X de 1ª geração (HTP v73), **compilado e homologado para o X1-26-100**; X1E/X1P declarados só se um contexto rodar nos dois (S-NPU-0) ou com pacote por `soc_model` | alvo principal de T3 |
| `qnn-w4a16-x2` | contexto binário para X2 Elite (receita Olive Llama 3.1 já tem `x2_elite_config.json`) | só com máquina X2 para validar |
| `qnn-w8a16` | pesos 8 bits | fallback de qualidade se w4a16 perder > gate |

Limites e decisões **[hipóteses a medir no spike S-NPU-1]**:

- **Tamanho de modelo:** começar por T0 (0,5B) e T1/T3 (1,5B); 3B só se a latência permitir; 7B está fora do escopo do T3 (Microsoft publica Qwen2.5-7B-qnn-npu, então é possível, mas com TTFT incompatível com autocomplete).
- **Contexto:** janela estática; prompts acima dela são erro do pedido (`LocalModelContextException` já existe na IDE). Autocomplete usa ≤ 2048 hoje (v1: p95 do chat 274 tokens).
- **Prefill em blocos de 64**: prompt de 1.500 tokens = 24 execuções do grafo de prefill; medir TTFT real antes de escolher contexto.
- **Particionamento:** criar a sessão com `session.disable_cpu_ep_fallback=1` no teste de cobertura: qualquer nó fora do HTP reprova a variante (exceto quantizer/dequantizer/position-processor previstos em CPU).
- **EPContext:** `ep.context_embed_mode=0` (`.bin` separado) para reduzir duplicação; o `.bin` é **específico do SoC/versão do QNN** → um pacote por família de SoC e versão de QAIRT registrada no manifesto.
- **Caminho alternativo:** Qualcomm AI Hub (`qai_hub_models` + wrappers ONNX), documentado pelo ORT GenAI para Llama 3.2/Phi-3.5, com alvo `"Snapdragon X Plus 8-Core CRD"` (mesmo die do X1-26-100 **[H forte]**). É **serviço de compilação em nuvem** da Qualcomm → **fora do plano** (sem nuvem); fica só como referência de que o die Purwa é alvo suportado pela Qualcomm. Se o Olive local falhar, o substituto local é gerar o contexto no próprio Vivobook (passo 9, opção c).

#### 5.5.1 Orçamento de memória no notebook de homologação (16 GB compartilhados)

O X1-26-100 não tem memória dedicada: SO, IDE, MongoDB e os buffers da NPU disputam os mesmos 16 GB LPDDR5X (não expansíveis). Orçamento proposto **[H: medir no S-NPU-6]**:

| Consumidor | Estimativa | Observação |
|---|---|---|
| Windows 11 Home + serviços (Defender, Search, Copilot/Widgets, driver NPU) | 3,5–4,5 GB | medir em repouso após boot |
| Kapibara Studio (.NET/Avalonia, editor, abas, cache de schema) sem modelo | 0,6–1,2 GB | medir com 3 abas e 1 conexão |
| MongoDB | **0** se remoto; 0,8–1,2 GB se `mongod` local com `storage.wiredTiger.engineConfig.cacheSizeGB: 0.5` | o padrão do WiredTiger é 50% de (RAM − 1 GB) ≈ **7,5 GB** **[F: https://www.mongodb.com/docs/manual/core/wiredtiger/]** → limitar é obrigatório; Docker/WSL2 no notebook não (custa +1–2 GB) |
| Navegador/outros apps do usuário | 1,5–2,5 GB | |
| Folga anti-paginação | 1,5–2 GB | abaixo disso o Windows comprime/pagina e a latência explode |
| **Modelo (processo + buffers NPU)** | **alvo ≤ 3 GB; teto 4 GB** | um modelo carregado por vez ([Fase 08](fase-08-integracao-kapibara.md) S2) |

Cabimento por modelo (pesos w4a16 = MatMuls 4 bits + `lm_head` 8 bits g32 + embedding `Gather` int4; KV fp16 alocado para a janela estática inteira; "processo" inclui ORT/GenAI, ativações uint16, logits do bloco de 64 e tokenizer) **[cálculo a partir dos `config.json`; tamanhos reais = hipótese]**:

| Modelo / uso | Geometria | Pesos | KV/token → KV na janela | Processo estimado | Cabe? |
|---|---|---|---|---|---|
| T3 Qwen2.5-Coder-0.5B, 2k | 24 camadas, 2 KV × 64 | ≈ 0,4 GB | 12 KiB → 24 MiB | 0,6–0,9 GB | ✔ folgado |
| T3 Qwen2.5-Coder-1.5B, 2k (4k) | 28 camadas, 2 KV × 128 | ≈ 1,1 GB | 28 KiB → 56 MiB (112 MiB) | 1,4–1,9 GB | ✔ |
| Agent-Lite Qwen3-0.6B, 4k | 28 camadas, 8 KV × 128 | ≈ 0,5 GB | 112 KiB → 0,46 GB | 1,1–1,5 GB | ✔ |
| Agent-Lite Qwen3-1.7B, 4k | 28 camadas, 8 KV × 128 | ≈ 1,3 GB | 112 KiB → 0,46 GB | 2,0–2,5 GB | ✔ (experimental) |
| Agent-Lite Qwen3-1.7B, 8k | idem | ≈ 1,3 GB | 112 KiB → 0,94 GB | 2,5–3,1 GB | ◐ no limite do alvo; grafo estático de 8k no HTP e prefill de 8k ⇒ só CPU |
| T4 Qwen3-8B | 36 camadas, 8 KV × 128 | ≈ 5,6 GB | 144 KiB → 2,4 GB (16k) | > 8 GB | ✘ não é alvo do notebook |

*Fallbacks* no mesmo notebook **[H: medir no S-NPU-6]**: **CPU Oryon ARM64** (`int4`, MatMulNBits ARM64) é o caminho garantido — 0.5B para autocomplete e Agent-Lite 1.7B com 8k (≈ 1,2–1,4 GB de pesos + 0,94 GB de KV ≈ 2,7–3,0 GB), com 8 núcleos a 2,97 GHz o prefill será ~1,5–2× mais lento que na X Elite; **GPU Adreno X1-45 (DirectML)** só `dml-fp16` do 0.5B (≈ 1 GB) para relatório (○) — 1,7 TFLOPS e memória compartilhada não justificam pacote; o `dml-fp16` do 1.5B (≈ 2,9 GB) consome o orçamento inteiro.

### 5.6 OpenVINO (investigação, sem pacote)

O EP OpenVINO está no catálogo do Windows ML (https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers) e o descritor já existe na IDE, mas não há máquina Intel Core Ultra no plano. Registrar apenas um spike de 1 dia (Olive + `OpenVINOExecutionProvider`, NPU Intel) se surgir hardware; **nenhum pacote OpenVINO é publicado nesta versão**. VitisAI (AMD Ryzen AI) idem.

### 5.7 T4 com raciocínio (thinking) e contexto de 16k

Decisão fixada para o T4 ([Fase 01](fase-01-selecao-modelos-base.md)): Qwen3-8B híbrido, **thinking ligado por padrão** no painel "Agente IA", desligável; orçamento de raciocínio padrão **1.024** tokens, máximo **4.096**, com `</think>` forçado pelo runtime ao estourar; raciocínio de turnos anteriores fora do histórico; contexto alvo **16k**. Agent-Lite Qwen3-1.7B com thinking curto opcional; Qwen3-0.6B/NPU **sem** thinking. Contrato de formato: [Fase 04 §5.6](fase-04-dataset-agente-tools.md); runtime/UI: [Fase 08 §3.3](fase-08-integracao-kapibara.md).

#### 5.7.1 O que muda no `genai_config.json`

| Campo | Valor do builder | Valor no pacote T4 | Motivo |
|---|---|---|---|
| `model.context_length` | 40.960 (`max_position_embeddings` do Qwen3-8B **[fato: `config.json`]**) | **16.384** | a IDE usa `context_length` como janela e recusa prompt + `MaximumTokens` acima dela (`LocalModelContextException`) **[fato: `OnnxLocalModelRuntime.cs`]**; 16k é o alvo e o que cabe no orçamento de VRAM abaixo. Reduzir só o campo não altera o grafo (cache RoPE continua até 40.960) **[hipótese: validar com o teste de 14k em 7.1]** |
| `search.max_length` | = contexto | **16.384** | teto; a IDE sobrescreve por pedido com `prompt + MaximumTokens` **[fato]** |
| `search.past_present_share_buffer` | `true` em DML | manter | com buffer compartilhado o KV é alocado para `max_length` **na criação do `Generator`** → um `Generator` vivo por turno do agente (reuso de KV, Fase 08 A6) aloca os 16k de uma vez **[hipótese forte: medir pico de VRAM]** |
| `search.do_sample` / `temperature` / `top_p` / `top_k` | do `generation_config.json` do base (T=0,6, top-p 0,95, top-k 20) | manter como padrão do modo thinking | o card do Qwen3 desaconselha greedy no thinking **[fato]**; a IDE hoje só envia `max_length`, `do_sample` e `temperature` e **não** envia `top_p`/`top_k` **[fato: `OnnxLocalModelRuntime.Generate`]** → os valores do pacote valem enquanto a Fase 08 não os sobrescrever |
| `model.decoder.filename`, `model.type` | `model.onnx`, `qwen3` | inalterado | `qwen3` depende do adapter generalizado (Fase 08 C2) |

`provider_options` continua fora do pacote (regra 6.5).

#### 5.7.2 Memória na RX 7800 XT (16 GB) — Qwen3-8B `dml-int4`

Cálculo a partir do `config.json` (36 camadas, 8 KV heads, `head_dim` 128, hidden 4.096, intermediate 12.288, vocabulário 151.936, `tie_word_embeddings=false`) **[fato]**; tamanhos efetivos **[hipótese: medir no benchmark]**:

| Componente | Conta | Memória |
|---|---|---|
| MatMuls das 36 camadas em INT4 (`block_size=32`, escala fp16 + zero point) | 6,945 B parâmetros × ~4,625 bits | **≈ 4,0 GB** |
| `lm_head` INT4 (é MatMul; não compartilhado) | 151.936 × 4.096 × 4,625 bits | ≈ 0,36 GB (≈ 1,25 GB se mantido fp16 por qualidade) |
| Embedding fp16 (`shared_embeddings=false`, regra DML) | 151.936 × 4.096 × 2 B | ≈ 1,25 GB |
| **Pesos** | | **≈ 5,6 GB** (≈ 6,5 GB com `lm_head` fp16) |
| KV cache **fp16** | 36 × 2 (K,V) × 8 × 128 × 2 B = **144 KiB/token** | 8k: 1,21 GB · **16k: 2,42 GB** (2,25 GiB) |
| KV cache INT8 (Engine 0.16+, paginado) | 72 KiB/token + escalas | 16k: ≈ 1,2–1,3 GB |
| KV cache INT4 | 36 KiB/token + escalas | 16k: ≈ 0,6–0,7 GB |
| Logits do prefill (se o grafo emite logits de todas as posições) | 151.936 × 2–4 B por posição | bloco de 2.048: 0,6–1,2 GB; prompt inteiro de 14k: **4–8,5 GB** ⇒ prefill em blocos obrigatório **[hipótese: conferir a forma de saída do grafo do builder]** |
| Ativações/scratch de atenção + overhead DML/driver | depende do bloco de prefill | 0,5–1,5 GB **[hipótese]** |
| **Total com KV fp16 a 16k** | | **≈ 9–11 GB** |

Conclusões **[proposta]**:

- **KV em fp16** na v2. O KV quantizado existe só na Engine nova (0.16+) com layout paginado e sem evidência em DirectML; a IDE usa a Generator API. Vira spike (`kv-int8`) depois do upgrade da Fase 08 R1, não requisito.
- **Prefill em blocos** de ≤ 2.048 tokens (`AppendTokens` sucessivos) quando o prompt passar de 4k, até medir se o grafo já corta os logits na última posição.
- Sobram ≈ 5–7 GB: suficiente para o desktop, mas **não** para o T2 carregado ao mesmo tempo (7B ≈ 5 GB + KV; 14B ≈ 8,5–9 GB + KV) → reforça "um modelo por vez" ([Fase 08](fase-08-integracao-kapibara.md) S2). Gate de VRAM em 7.3.
- 16k não é negociável para baixo sem reabrir a Fase 04: raciocínio de 1.024 tokens por passo, mantido nos passos do **turno corrente** (regra do template), mais ≤ 6 KiB por resultado de tool, esgota 8k em 2–3 passos.

#### 5.7.3 Template de chat embutido e tokens de raciocínio

- `chat_template.jinja` **[proposta: novo arquivo no pacote]** = cópia byte a byte do `chat_template` do `tokenizer_config.json` do Qwen3-8B base (também mantido dentro do `tokenizer_config.json`). A IDE passa esse texto como `template_str` (padrão do exemplo oficial). SHA-256 no `kapicoder_manifest.json` e no `kapibarastudio-model.json` (`reasoning.templateSha256`), para a IDE recusar template trocado.
- **Desligar o thinking sem kwargs:** como o `ApplyChatTemplate` não recebe `enable_thinking` **[fato]**, o pacote declara a **forma canônica** do modo desligado: renderizar com `add_generation_prompt=true` e acrescentar `<think>\n\n</think>\n\n` — exatamente o que o template do Qwen3 emite com `enable_thinking=false` **[fato]**. A chave branda `/no_think` não é usada pela IDE (não determinística). **[proposta]**
- **Tokens:** `<think>`/`</think>` (151667/151668) e `<tool_call>`/`</tool_call>` (151657/151658) **não** são `special` no tokenizer do Qwen3 **[fato]**. Regra: tokenizer continua byte a byte; a gramática da guidance referencia os tokens por id (`<[151667]>`) **[hipótese: confirmar que a llguidance do GenAI aceita id de token não `special`]**. Só se isso falhar, gerar um `tokenizer.json` com `special: true` nesses quatro tokens, versionado como variante e com a paridade de tokenizer (400 prompts, 100%) refeita — e conferir que `Decode` do GenAI não passa a omiti-los do texto.
- `GetBorTokenId`/`GetEorTokenId` do GenAI devem devolver 151667/151668 com o pacote; se devolverem −1 (o runtime não reconhece o par no tokenizer), a IDE usa os ids do metadata. Registrado no relatório de paridade. **[hipótese]**

#### 5.7.4 Agent-Lite

| Variante | Base | Thinking | Contexto do pacote | KV/token (fp16) | KV no contexto | Pesos |
|---|---|---|---|---|---|---|
| `int4` (CPU) / `dml-fp16` | Qwen3-1.7B (28 camadas, 8 KV heads × 128, embeddings compartilhados **[fato: `config.json`]**) | **opcional, curto**: padrão desligado, orçamento 256–512 (máx. 1.024) | **8.192** (16k só se a Fase 04 medir p95 > 7k no Lite) | 112 KiB | 8k: 0,94 GB · 16k: 1,88 GB | INT4 ≈ 1,1–1,4 GB; fp16 ≈ 3,4 GB |
| `qnn-w4a16` (NPU, ◐) | Qwen3-0.6B (28 camadas, 8 KV heads × 128, embeddings compartilhados **[fato: https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json]**) | **nenhum**: template sempre com o bloco vazio; `reasoning.supported=false` | 2.048–4.096 estático | 112 KiB | 4k: 0,46 GB | ≈ 0,5 GB |
| `qnn-w4a16` 1.7B (NPU, ○ relatório) | Qwen3-1.7B, sem thinking | nenhum | 4.096 estático | 112 KiB | 4k: 0,46 GB | ≈ 1,3 GB (processo 2,0–2,5 GB, cabe no orçamento do notebook de 16 GB, seção 5.5.1) — vira ◐ só se TTFT do prompt do agente (sistema + tools) no X1-26-100 ficar ≤ 3 s e ganhar ≥ 10 pp de sucesso sobre o 0.6B |

No notebook de homologação (X1-26-100, 16 GB) o Agent-Lite **padrão é o 1.7B `int4` na CPU** (8k, ≈ 2,7–3,0 GB); a NPU recebe o 0.6B sem thinking. **[proposta]**

O pacote Lite com `reasoning.defaultEnabled=false` e orçamento menor usa o mesmo template e as mesmas regras de 5.7.3. **[proposta]**

#### 5.7.5 Testes de exportação específicos do raciocínio

Detalhados em 7.1 (paridade) e 7.3 (latência/VRAM); resumo do que **não** existia antes: paridade de renderização do template (ORT × HF), paridade em contexto longo (até 14k), paridade por *teacher forcing* da sequência com `<think>`, fechamento forçado de `</think>` via `AppendTokens`, modo desligado sem `<think>` gerado, e pico de VRAM com o `Generator` alocado em 16k.

## 6. Matriz tier × alvo e convenção de pacotes

### 6.1 Matriz (o que gerar)

Legenda: ● gerar e publicar · ○ gerar só para relatório · ◐ experimental (publicar só se passar na seção 7.4) · — não gerar.

| Tier | `int4` | `int8` | `dml-fp16` | `dml-int4` | `cuda-int4` | `qnn-w4a16` | `qnn-w4a16-x2` |
|---|---|---|---|---|---|---|---|
| T0 Nano (~0,5B) | ● | ● | ● | ○ (v1: perde APT sem ganhar TTFT) | — | ◐ (homologado no X1-26-100) | — (sem máquina X2) |
| T1 Small (~1,5B) | ● | ● | ● | ● | — | ○ (base do T3) | — |
| T2 Pro (KapiTeacher 14B / 7B) | ○ (7B ~4,5 GB; 14B ~8–9 GB; CPU lenta) | — | — (não cabe) | ● | ◐ (se houver NVIDIA) | — | — |
| T3 NPU (modelo da [Fase 01](fase-01-selecao-modelos-base.md): 1,5B se passar nos gates do X1-26-100, senão 0,5B) | ● (fallback CPU no mesmo repo) | — | — | — | — | ◐ (homologado no X1-26-100) | — (sem máquina X2) |
| T4 Agent (Qwen3-8B híbrido, thinking; Agent-Lite Qwen3-1.7B thinking curto opcional / 0.6B sem thinking; fallback sem raciocínio Qwen2.5-Coder-7B-Instruct) | ● (Lite 1.7B) | — | ● (Lite 1.7B) | ● (8B, contexto 16k; 7B só como fallback sem raciocínio) | ◐ | ◐ (Lite 0.6B **sem thinking**, se o agente couber em 2048–4096 tokens) | — |

Estimativas de tamanho em disco **[hipótese, extrapoladas da v1]**: 0,5B `int4` ≈ 0,4 GB, `dml-fp16` ≈ 1 GB; 1,5B `int4` ≈ 1,1 GB, `int8` ≈ 2,5 GB, `dml-fp16` ≈ 2,9 GB; 7B `dml-int4` ≈ 5–5,5 GB (embedding fp16 ≈ 1,1 GB); 1,5B `qnn-w4a16` ≈ 1,2–1,6 GB (pesos 4 bits + `lm_head` 8 bits + contexto binário).

### 6.2 Nomes

| Item | Convenção | Exemplo |
|---|---|---|
| Família | `KapiCoder-Mongo-<Tier>` | `KapiCoder-Mongo-Small` |
| Repositório HF ONNX | `esilva/<Família>-ONNX` | `esilva/KapiCoder-Mongo-Small-ONNX` |
| Repositório HF safetensors | `esilva/<Família>` (só referência; o GenAI não carrega) | `esilva/KapiCoder-Mongo-Small` |
| Pasta de variante no repo | `<variante>` com `genai_config.json` no 1º nível (regra da ADR-039) | `dml-fp16/` |
| Pasta instalada pela IDE | `<repositório>-<variante>` | `KapiCoder-Mongo-Small-ONNX-dml-fp16` |
| Variantes NPU | `qnn-w4a16`, `qnn-w4a16-x2`, `qnn-w8a16` | `KapiCoder-Mongo-NPU-ONNX-qnn-w4a16` |

`ModelDisplayNames.ParseVariant` atual mostra `KapiCoder-Mongo-Small — GPU DirectML FP16` para `dml-fp16`; para `qnn-w4a16` retorna `null` e a IDE exibe o nome da pasta ou o `name` do metadata — funcional, mas sem o título amigável (correção proposta na [Fase 08](fase-08-integracao-kapibara.md)). **[fato: `ModelDisplayNames.cs`]**

### 6.3 Conteúdo de cada pacote

```text
KapiCoder-Mongo-Small-ONNX-dml-fp16/
├── genai_config.json
├── model.onnx
├── model.onnx.data
├── tokenizer.json              # cópia byte a byte do base
├── tokenizer_config.json
├── vocab.json / merges.txt / special_tokens_map.json / added_tokens.json (se existirem no base)
├── generation_config.json
├── chat_template.jinja         # só pacotes de agente (T4/Lite): template nativo byte a byte, SHA-256 no metadata (5.7.3)
├── kapibarastudio-model.json   # lido pela IDE
├── kapicoder_manifest.json     # proveniência, contratos, paridade, SHA-256 de todos os arquivos
├── reports/                    # paridade, benchmark, harness da IDE (resumos .md/.json)
├── LICENSE                     # Apache-2.0 (Qwen2.5-Coder 0.5B–7B/14B; conferir 3B), com atribuição e aviso de modificação
├── NOTICE.md                   # base (Apache-2.0), professor KapiTeacher (Qwen2.5-Coder, Apache-2.0; pesos do professor só distribuídos se for o T2), dados, avisos de modificação
└── README.md                   # model card (também no repo HF)

KapiCoder-Mongo-NPU-ONNX-qnn-w4a16/      # [hipótese: estrutura exata vem da saída do Olive]
├── genai_config.json                    # pipeline (embeddings → ctx/iter → lm_head), provider qnn
├── *.onnx (wrappers EPContext) + *.bin (contextos QNN) + *.json (metadados do contexto)
├── quantizer/dequantizer/position-processor *.onnx (se o pipeline os gerar)
└── tokenizer + kapibarastudio-model.json + kapicoder_manifest.json + reports/ + LICENSE/NOTICE/README
```

Nome do manifesto: `kapicoder_manifest.json` (não `slopcoder_manifest.json`) — o `DeepSeekCoderModelAdapter` ativa o caminho DeepSeek quando encontra `slopcoder_manifest.json` numa exportação `llama`; nenhum tier atual (T2 incluso, agora `qwen2`) usa esse caminho, e o nome novo evita ativação acidental. **[fato: `DeepSeekCoderModelAdapter.CanHandle`; proposta: nome novo]**

### 6.4 `kapibarastudio-model.json`

Regras de compatibilidade **[fato: `LocalModelMetadataReader.cs`]**: propriedades desconhecidas e nomes de capacidade/hardware desconhecidos são ignorados; **`contextContract` desconhecido invalida a pasta**. Logo:

- capacidades novas (`agent`, `tools`, `reasoning`) e blocos novos (`agent`, `reasoning`, `generation.agent`) podem ser publicados já — IDEs antigas os ignoram (o T4 nelas fica indisponível como agente, porque o `LocalAgentProvider` atual exige `fim`+`chat`);
- `contextContract` só recebe valor novo quando a IDE suportar o contrato (Fase 08); até lá, omitir (= `editor-context-v1`) se o modelo foi treinado nesse contrato.

Exemplo (T1, DirectML):

```json
{
  "name": "KapiCoder-Mongo-Small ONNX DML-FP16",
  "version": "2.0.0",
  "architecture": "qwen2",
  "parameters": "1.5B",
  "domain": ["mongodb", "mongosh", "json", "kapibara-dsl"],
  "capabilities": ["autocomplete", "chat", "fim"],
  "hardware": ["gpu"],
  "recommendedContextTokens": 2048,
  "recommendedCompletionTokens": 32,
  "generation": {
    "autocomplete": { "maxTokens": 32, "temperature": 0 },
    "chat": { "maxTokens": 256, "temperature": 0 }
  }
}
```

Exemplo (T4, agente; campos novos marcados como proposta — ignorados pela IDE atual):

```json
{
  "name": "KapiCoder-Mongo-Agent ONNX DML-INT4",
  "version": "2.0.0",
  "architecture": "qwen3",
  "parameters": "8B",
  "capabilities": ["chat", "agent", "tools", "reasoning"],
  "hardware": ["gpu"],
  "recommendedContextTokens": 16384,
  "generation": {
    "chat": { "maxTokens": 1024, "temperature": 0.7 },
    "agent": {
      "maxAnswerTokensPerStep": 1024,
      "sampling": {
        "thinking":    { "temperature": 0.6, "topP": 0.95, "topK": 20 },
        "nonThinking": { "temperature": 0.7, "topP": 0.8,  "topK": 20 }
      }
    }
  },
  "agent": {
    "promptFormat": "qwen3-chatml-tools-v1",
    "chatTemplateFile": "chat_template.jinja",
    "toolCallFormat": "qwen-hermes-json",
    "toolSchemaVersion": "kapibara-tools-2026-10",
    "maxToolCallsPerTurn": 12,
    "maxToolResultBytes": 6144,
    "constrainedDecoding": "lark_grammar"
  },
  "reasoning": {
    "supported": true,
    "format": "qwen3-think",
    "startTokenId": 151667,
    "endTokenId": 151668,
    "defaultEnabled": true,
    "toggle": "hard",
    "disabledGenerationSuffix": "<think>\n\n</think>\n\n",
    "budgetTokens": { "default": 1024, "minimum": 128, "maximum": 4096 },
    "turnBudgetTokens": 4096,
    "onBudgetExceeded": "force-close",
    "forceCloseText": "\n\n</think>\n\n",
    "history": "strip-previous-turns",
    "templateSha256": "<sha256 do chat_template.jinja>"
  },
  "runtime": { "minimumGenAi": "0.15.2" }
}
```

Bloco `reasoning` **[proposta; nomes a fixar na ADR da Fase 08]**:

| Campo | Significado | T4 | Lite 1.7B | Lite 0.6B / NPU |
|---|---|---|---|---|
| `capabilities` ∋ `"reasoning"` | o pacote foi treinado e validado (7.1) com thinking; IDE atual ignora o nome **[fato: nomes desconhecidos ignorados]** | sim | sim | **não** |
| `supported` | IDE pode oferecer o controle "Raciocínio" | `true` | `true` | `false` |
| `defaultEnabled` | estado inicial do controle no painel | `true` | `false` | — |
| `toggle` | `hard` = bloco vazio no prompt (5.7.3); `none` = sempre ligado (modelos thinking-only, ex. 4B-Thinking-2507) | `hard` | `hard` | — |
| `budgetTokens` | orçamento **por geração** (por passo do agente) escolhido pelo usuário dentro de `[minimum, maximum]` | 1.024 / 128–4.096 | 384 / 128–1.024 | — |
| `turnBudgetTokens` | soma de raciocínio mantida no contexto num turno; atingida, os passos seguintes saem com o bloco vazio (Fase 08 A11) | 4.096 | 1.024 | — |
| `onBudgetExceeded` + `forceCloseText` | o runtime injeta `forceCloseText` (que termina em `</think>`) e continua a geração; a frase de transição curta usada no treino (Fase 05) pode ser prefixada aqui **[hipótese: texto exato vem do treino]** | `force-close` | `force-close` | — |
| `history` | raciocínio de turnos anteriores não volta ao prompt (regra do template) | `strip-previous-turns` | idem | — |
| `startTokenId`/`endTokenId` | fallback quando `GetBorTokenId`/`GetEorTokenId` devolvem −1 | 151667/151668 | idem | — |

`generation.agent` separa o orçamento da resposta/tool call (`maxAnswerTokensPerStep`) do orçamento de raciocínio: o `MaximumTokens` de um passo é `budget + maxAnswerTokensPerStep + |forceCloseText|` (até ≈ 5,2k com orçamento 4.096), acima do teto atual de 1.024 do `generation.chat.maxTokens` **[fato: `LocalModelMetadataReader`]** — por isso campo novo, não reaproveitar `chat.maxTokens` (Fase 08 C5/C7). `temperature` > 0 no T4: o modo thinking não deve ser greedy (5.7.1); os testes de paridade usam greedy só como medida numérica (7.1).

Exemplo (T3 NPU):

```json
{
  "name": "KapiCoder-Mongo-NPU ONNX QNN-W4A16",
  "version": "2.0.0-preview.1",
  "architecture": "qwen2",
  "parameters": "1.5B",
  "capabilities": ["autocomplete", "chat", "fim"],
  "hardware": ["npu"],
  "recommendedContextTokens": 2048,
  "generation": { "autocomplete": { "maxTokens": 32, "temperature": 0 }, "chat": { "maxTokens": 256, "temperature": 0 } },
  "npu": { "provider": "qnn", "htpArch": 73, "socModel": "<id lido no S-NPU-0>", "homologatedOn": ["X1-26-100"], "soc": ["X1-26-100"], "staticContextTokens": 2048, "minMemoryGb": 16, "qairt": "<versão fixada>", "epContext": true },
  "runtime": { "minimumGenAi": "0.17.1" }
}
```

`hardware: ["npu"]` sozinho faz o `AiProviderSelector` recusar CPU/GPU para o pacote (sem fallback silencioso) — comportamento correto, porque o contexto binário não roda em outro provider. **[fato: `AiProviderSelector.Plan`]**

`generation.chat.maxTokens` aceita até 1024 e `recommendedContextTokens` até `AutocompleteSettings.AbsoluteContextMaximum` **[fato]**; com raciocínio o passo do agente passa de 1.024 tokens de saída — resolvido pelo bloco `generation.agent` + `reasoning` acima (Fase 08 C5).

### 6.5 `genai_config.json`

- Gerado pelo builder/Olive; o pipeline só altera `search` (greedy: `do_sample=false`, `max_length` = contexto recomendado) e confere `context_length`. **Exceção T4/Lite:** `context_length`/`max_length` = contexto do pacote e amostragem do modo thinking (seção 5.7.1).
- Para CPU/DML/CUDA, `model.type` deve permanecer o da arquitetura (`qwen2`) para o `QwenCoderModelAdapter` aceitar. **[fato]** O T4 Qwen3-8B/Agent-Lite exporta `qwen3`, que o adapter atual rejeita → depende da generalização do adapter ([Fase 08](fase-08-integracao-kapibara.md) C2).
- `provider_options` **não** é fixado no pacote para CPU/DML: a IDE chama `ClearProviders()`/`AppendProvider()` em `OnnxLocalModelRuntime.Load`. Para QNN o config do Olive traz opções do EP (por exemplo `backend_path`, `htp_performance_mode`); manter e documentar. **[fato: `OnnxLocalModelRuntime.cs`; hipótese: interação com `ClearProviders()` no pipeline QNN — validar no S-NPU-2]**

## 7. Validação

### 7.1 Paridade numérica (por variante)

`scripts/parity_check.py --package Repos\<modelo>\exports\<pasta> --reference Repos\<modelo>\merged --provider <cpu|dml|qnn>`

| Métrica | Como | Gate fp32 | Gate quantizado CPU/DML | Gate QNN |
|---|---|---|---|---|
| Cosseno dos logits (1º passo) | 64 prompts (autocomplete/chat/agente) | ≥ 0,999 | informativo | informativo |
| Top-1 do 1º token | idem | 100% | ≥ 95% | ≥ 90% |
| Top-5 contém o top-1 do PyTorch | idem | — | ≥ 99% | ≥ 97% |
| Concordância de sequência (24 tokens greedy) | 12 + 6 prompts (herdado) | ≥ 0,75 | informativo (v1: 0,58–0,83) | informativo |
| Tokenizer nativo GenAI × HF | 400 prompts | 100% | 100% | 100% |
| Tool call bem formado (T4) | 100 prompts de agente, parse JSON + schema | 100% dos casos em que o PyTorch acerta | ≥ 98% | ≥ 95% |

**Paridade com raciocínio (T4 e Lite 1.7B)** — `parity_check.py --reasoning` **[proposta]**. O thinking roda com amostragem no produto, mas a paridade continua numérica (greedy/teacher forcing), porque sequências amostradas não são comparáveis entre frameworks:

| Métrica | Como | Gate fp32 | Gate `dml-int4` / `int4` |
|---|---|---|---|
| Renderização do template | 200 conversas (tools, multi-turno com `reasoning_content`, `<tool_response>`, pt-BR/en, ligado e desligado): `Tokenizer.ApplyChatTemplate(chat_template.jinja, …)` do GenAI (+ sufixo de 5.7.3 no modo desligado) × `transformers.apply_chat_template(enable_thinking=…)` | 100% byte a byte | 100% |
| Remoção do raciocínio de turnos anteriores | subconjunto multi-turno da linha acima: nenhum `<think>` de turno ≤ última pergunta do usuário no texto renderizado | 100% | 100% |
| *Teacher forcing* com `<think>` | 64 sequências geradas pelo PyTorch (raciocínio 64–1.024 tokens + `<tool_call>`): top-1 do ONNX em cada posição | ≥ 99,5% | ≥ 97% |
| Contexto longo | 3 prompts de 4k / 8k / 14k tokens (prefill em blocos de 2.048): cosseno dos logits nas 32 posições seguintes; top-1 | ≥ 0,999 / 100% | informativo / ≥ 95% |
| Fechamento forçado | 100 prompts de agente com orçamento artificial de 64 tokens: injetar `forceCloseText` via `Generator.AppendTokens` e continuar; saída seguinte = `<tool_call>` parseável ou texto final | Δ ≤ 3 pp vs PyTorch com a mesma injeção | Δ ≤ 5 pp |
| Modo desligado | 100 prompts com o bloco vazio: ocorrências de 151667 geradas | 0 | 0 |
| Separação raciocínio → tool call | 100 prompts com thinking (greedy, orçamento 1.024): `<think>…</think>` fechado + `<tool_call>` parseável e do plano | 100% dos casos em que o PyTorch acerta | ≥ 98% |
| Ids de raciocínio | `GetBorTokenId`/`GetEorTokenId` | 151667/151668 (ou −1 registrado, ver 5.7.3) | idem |

A qualidade **com amostragem** (sucesso de tarefa, estouro de orçamento, coerência pensamento→ação; 3 seeds) é medida pelo harness da [Fase 06](fase-06-avaliacao-benchmarks.md) sobre o pacote exportado, nos mesmos limiares de paridade de 7.2.

### 7.2 Qualidade (benchmark da [Fase 06](fase-06-avaliacao-benchmarks.md))

Mesmo conjunto e mesmos exemplos do merged (regra da v1). **Limiares oficiais = paridade da [Fase 06 §6](fase-06-avaliacao-benchmarks.md)** (int8 ΔAPT ≥ −0,10; int4 ≥ −0,15; w4a16 ≥ −0,25; perdas líquidas no chat 2/3/5 de 120). Os valores mais frouxos originais desta fase — **ΔAPT ≥ −0,30**, **ΔIntent ≥ −2 pp**, **Δ taxa de sucesso do agente ≥ −3 pp** (T4) e, para QNN, ΔAPT ≥ −0,40 — servem só como piso técnico para descartar cedo uma variante quebrada. A variante que falhar não é publicada. Conjuntos escritos à mão (120 + cego 40 da v1, mais os novos da Fase 06) com teste do sinal pareado contra o merged.

### 7.3 Geração e desempenho

> Os orçamentos abaixo são o teto técnico da exportação. **Para adoção/release valem os gates de latência por tier da [Fase 06 §8](fase-06-avaliacao-benchmarks.md)** (mais estritos, medidos pelo harness da IDE), que prevalecem em caso de divergência.

| Alvo | Métricas | Orçamento (gate) |
|---|---|---|
| CPU (Ryzen 9 7900, `intra_op_num_threads=0` como a IDE) | TTFT médio/p95, tokens/s, pico de RAM, carga | T0 autocomplete: TTFT ≤ 300/500 ms, RAM ≤ 2 GB; T1: ≤ 800/1.500 ms, ≤ 4 GB (orçamento v1) |
| DirectML (RX 7800 XT) | idem + RSS | T0/T1 TTFT ≤ 200/300 ms; T2 7B/T4 8B TTFT ≤ 600/1.000 ms e T2 14B TTFT ≤ 900/1.500 ms no autocomplete com prompt de 1.500 tokens **[proposta]** |
| QNN (Vivobook X1-26-100, na tomada e na bateria, perfil "Melhor desempenho") | TTFT com prompts de 256/1.024/2.000 tokens, tokens/s, energia aproximada (bateria, `powercfg /srumutil`), **memória**: working set + commit privado do processo (`GetProcessMemoryInfo`), "Memória disponível" e crescimento do pagefile durante o benchmark, com a IDE aberta e `mongod` local limitado a 0,5 GB de cache (seção 5.5.1) | TTFT ≤ 400/800 ms (T0) e ≤ 800/1.500 ms (T3 1.5B) com 1.024 tokens; tokens/s ≥ 35 (0.5B) e ≥ 20 (1.5B) — referência: Llama-3.2-1B w4a16 faz 34–43 tok/s na X Elite, mesma NPU de 45 TOPS e mesma banda; processo ≤ 1,0 GB (0.5B), ≤ 2,0 GB (1.5B), ≤ 1,5 GB (Lite 0.6B 4k); sistema com ≥ 2 GB disponíveis e pagefile sem crescer **[proposta]** |
| CPU ARM64 (Oryon 8 núcleos, mesmo notebook) — fallback de T0/T3 e Agent-Lite | idem | T0 `int4`: TTFT ≤ 400/700 ms, ≥ 25 tok/s; T1/T3 `int4`: ≤ 1.200/2.000 ms; Agent-Lite 1.7B `int4` 8k: RAM do processo ≤ 3,0 GB **[proposta: ~1,5–2× mais lento que o Ryzen 9 7900]** |
| DirectML — T4 com raciocínio (Qwen3-8B `dml-int4`, 16k) | TTFT com prompt de 4k / 8k / 14k; tokens/s de decodificação com 1k e 14k de contexto; **TTFA** (tempo até a 1ª ação = `</think>` + `<tool_call>` completo) com thinking ligado (orçamento 1.024) e desligado; pico de VRAM com `Generator` em 16k (`GetProcessMemoryInfo`/contador de GPU dedicada); tempo de injeção do `</think>` forçado | TTFT ≤ 1.000 ms com 4k e ≤ 3.000 ms com 14k; decodificação ≥ 30 tok/s com 1k e ≥ 20 tok/s com 14k; TTFA p50 ≤ 30 s / p95 ≤ 45 s ligado (≈ 1.024 ÷ 30–40 tok/s) e ≤ 4 s desligado; **VRAM ≤ 11 GB** **[proposta; gates de adoção na Fase 06 §8]** |
| CPU — Agent-Lite 1.7B `int4` | idem com prompt de 2k / 6k e orçamento 384 | TTFA p50 ≤ 20 s ligado / ≤ 6 s desligado; RAM ≤ 4 GB **[proposta]** |

Testes de geração adicionais: parada imediata (1º token de parada → texto vazio, sem exceção), contexto acima da janela (erro do pedido, modelo continua carregado), cancelamento, 200 gerações seguidas sem crescimento de memória, segundo modelo no mesmo processo (DML: registrar se ainda derruba).

### 7.4 Homologação NPU (ASUS Vivobook 16 X1607QA — Snapdragon X X1-26-100, 16 GB, Windows 11 Home)

| ID | Teste | Critério |
|---|---|---|
| S-NPU-0 | **Inventário e controle do dispositivo (1 dia, antes de exportar)**: Windows Update em dia (≥ 24H2/26100), driver Hexagon, versão do QNN EP do Windows ML; `GetEpDevices()` num harness `win-arm64` após `EnsureAndRegisterCertifiedAsync()`; Python ARM64 + `onnxruntime-qnn` win_arm64 com a ferramenta de plataforma do QAIRT (`qnn-platform-validator`) para ler **SoC/`soc_model` e `htp_arch`**; rodar um LLM QNN pronto (Foundry Local `foundry model ls --device npu`, execução local; ou um pacote QNN pré-compilado baixado do Hugging Face, ex. `qualcomm/Llama-v3.2-1B-Instruct`) e conferir que a variante resolvida é NPU e não `-cpu`; testar se um contexto `soc_model=60` (X Elite) carrega | QNN EP registrado e listado como NPU apesar de o X1-26-100 não constar dos requisitos do Windows ML; `soc_model`/`htp_arch` registrados em `reports/npu/device.json`; LLM de controle gera texto na NPU. **Se o QNN EP não registrar ou o controle cair para CPU: T3 = "não homologado" e só os fallbacks CPU seguem** |
| S-NPU-1 | Olive x64 gera `qnn-w4a16` do T0 com calibração do domínio e o `soc_model` do S-NPU-0 | build sem erro; tamanho registrado |
| S-NPU-2 | No Vivobook, o build `win-arm64` WinML da IDE (ou harness) registra o QNN EP via `ExecutionProviderCatalog` e o GenAI carrega o pacote com provider `qnn` | `GetEpDevices()` lista `QNNExecutionProvider (NPU)`; sessão criada |
| S-NPU-3 | Cobertura | `session.disable_cpu_ep_fallback=1` (exceto pré/pós-processamento previstos) sem erro |
| S-NPU-4 | Paridade | gates da coluna QNN em 7.1 |
| S-NPU-5 | Qualidade | gates de 7.2 |
| S-NPU-6 | Latência/energia/**memória** | orçamentos de 7.3 (linha QNN e CPU ARM64), com a IDE aberta; 1.5B e 0.5B medidos lado a lado para a decisão do T3 ([Fase 01](fase-01-selecao-modelos-base.md)) |
| S-NPU-7 | Driver/EP | registrar driver Hexagon (≥ 30.0.140.0), versão do QNN EP do Windows ML (1.8.x ou 2.x; hoje QAIRT 2.40 / 2.48.40), Windows build, QAIRT do contexto; contexto gerado com QAIRT ≤ ao do EP instalado e retestado a cada atualização do QNN EP |
| S-NPU-8 | X1E/X1P ou X2 Elite (só se surgir máquina) | repetir S-NPU-0, 2..6 com o mesmo pacote (X1E/X1P) ou `qnn-w4a16-x2` |

Sem S-NPU-0 e S-NPU-2..6 aprovados no Vivobook, o pacote NPU **não** é publicado no repositório principal; no máximo em `esilva/<Família>-ONNX-preview` com aviso no card.

### 7.5 Harness da IDE

`scripts/ide_harness.py` (herdado) compila `tools/ide-harness` contra as DLLs da build da IDE (sem alterar o repositório) e executa 120 exemplos por pacote pelo caminho real (`LocalModelCatalog → LocalAiModelService → OnnxLocalModelRuntime`). Como o `LocalModelAiChatService` foi removido (ADR-055), o chat passa a ser exercitado pelo `LocalAgentProvider` (contrato atual de proposta de código) e, quando a Fase 08 implementar tool calling, pelo novo contrato. Gates: 0 nulos inesperados, 0 falhas do runtime, provider efetivo igual ao pedido, concordância com o Python ≥ 85% (autocomplete) e ≥ 90% (chat) — v1: 83–100% — como piso de exportação; o gate de release é o da [Fase 06](fase-06-avaliacao-benchmarks.md) (concordância ≥ 99,5%), divergência registrada no [README](README.md).

### 7.6 Empacotamento e publicação

| Passo | Script | Verificação |
|---|---|---|
| Montar pacote | `package_exports.py --tier T1 --variant dml-fp16` | metadata validado por um leitor equivalente ao `LocalModelMetadataReader` (tipos, limites) |
| Hashes | idem | SHA-256 de cada arquivo em `kapicoder_manifest.json`; releitura após cópia |
| Card | `publish_hf.py --render-card` | base, licença, professor (sem pesos), dados, contratos, tabela de variantes por hardware, limitações, resultados da Fase 06 |
| Upload | `publish_hf.py --repo esilva/KapiCoder-Mongo-Small-ONNX --variant dml-fp16` | `huggingface_hub` com LFS; depois baixar pela API `resolve/<sha>` e comparar SHA-256 (o download da IDE confere SHA-256 do LFS ou SHA-1 do blob) |
| Tag | `v2.0.0-<variante>` no repo HF | commit `sha` registrado no changelog (Fase 08) |
| Licenças | `LICENSE` + `NOTICE.md` + `license` no card | Qwen2.5-Coder 0.5B/1.5B/7B/14B Apache-2.0 **[verificar por modelo escolhido; a família 3B usa licença própria]**; sem obrigações da licença DeepSeek (Anexo A removido; DeepSeek só como referência histórica no relatório); QNN EP e contextos binários: termos do Windows ML/Qualcomm (https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers → third-party notices) |

## 8. Atividades (passo a passo)

1. [ ] Congelar versões: ORT GenAI do builder = versão do runtime da IDE (0.15.2 ou 0.17.1 conforme ADR da Fase 08); registrar em `configs/export.yaml`.
2. [ ] `build_calibration.py` para cada tier (Fases 03/04, split `train`).
3. [ ] Referência `fp32-cpu` + paridade estrita por tier.
4. [ ] Variantes CPU (`int4` k-quant misto, `int8` pós-quantizado; AWQ/GPTQ como candidatos no T2).
5. [ ] Variantes DML (`dml-fp16` T0/T1, `dml-int4` T1/T2/T4) + benchmark no `.venv-winml`.
6. [ ] Relatório comparativo por tier (`reports/export/<tier>/comparison.md`) e escolha de distribuição (regra v1: menor variante dentro do orçamento e da perda máxima).
6b. [ ] T4/Lite com raciocínio (5.7): `genai_config` com 16.384 (Lite 8.192), `chat_template.jinja` + SHA-256, bloco `reasoning` no metadata, paridade com raciocínio (7.1), VRAM/TTFA (7.3) e smoke de guidance com bloco `<think>` livre + `<tool_call>` restrito (`tools/GuidanceSmoke --reasoning`) no NuGet da IDE.
7. [ ] Spike S-NPU-1 (Olive x64) com T0; ajustar receita.
7a. [ ] S-NPU-0 no Vivobook X1-26-100 (pode rodar já na semana 1, independe do treino): registro do QNN EP, `soc_model`/`htp_arch`, LLM de controle na NPU, orçamento de memória real (5.5.1).
8. [ ] Spikes S-NPU-2..7 no Vivobook X1-26-100 (depende do protótipo de registro de EPs da Fase 08, em branch local fora do repositório principal ou harness próprio).
9. [ ] Variantes QNN de T3 (e T0/T4 pequeno se o T0 passou).
10. [ ] Harness da IDE em todas as variantes candidatas.
11. [ ] Empacotar, gerar cards, publicar em repos de pré-release (`-preview`) e só promover após a homologação da Fase 08.

## 9. Artefatos e entregáveis

- [ ] `configs/export.yaml` v2 e `configs/olive/qnn-*.json`
- [ ] `data/calib/<tier>.jsonl` + manifesto de origem
- [ ] `Repos\KapiCoder-Mongo-<Tier>\exports\KapiCoder-Mongo-<Tier>-ONNX-<variante>\` conforme matriz 6.1
- [ ] `reports/export/<tier>/<variante>/{parity.json,benchmark.json,harness.json,build.log}`
- [ ] `reports/export/<tier>/comparison.md` e `reports/export/npu-homologacao.md`
- [ ] Repositórios HF `esilva/KapiCoder-Mongo-{Nano,Small,Pro,NPU,Agent}-ONNX` (pré-release) com cards e hashes conferidos

## 10. Critérios de saída (gates)

| Gate | Critério |
|---|---|
| G7.1 | Referência fp32 de cada tier aprovada (seção 5.1) |
| G7.2 | Toda variante publicada passa 7.1 e 7.2 nos mesmos exemplos do merged |
| G7.3 | Orçamentos de 7.3 cumpridos para a variante de distribuição de cada tier × hardware |
| G7.4 | Harness da IDE: 0 falhas do runtime, provider efetivo correto, concordância ≥ 85/90% |
| G7.5 | Pacotes com `kapibarastudio-model.json` válido, SHA-256 conferido após download pelo hub |
| G7.6 (NPU) | S-NPU-0 e S-NPU-2..6 aprovados no Vivobook X1-26-100 (16 GB), incluindo os tetos de memória de 7.3; o card declara "homologado em Snapdragon X X1-26-100" e X1E/X1P como "não testado" até S-NPU-8; senão T3 sai como "não homologado" e não entra no **Baixar modelo** |
| G7.7 (raciocínio) | T4 e Lite 1.7B: tabela "Paridade com raciocínio" de 7.1 aprovada, VRAM ≤ 11 GB com 16k e TTFA dentro de 7.3; pacote sem `"reasoning"` em `capabilities` se qualquer item falhar (o agente continua utilizável sem thinking) |

## 11. Riscos e mitigação

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| X1-26-100 fora da lista oficial do QNN EP do Windows ML (só X1E/X1P) e `soc_model` do die Purwa não público | média | QNN EP não registra, ou contexto compilado com `soc_model` errado não carrega | S-NPU-0 na semana 1; contexto gerado no próprio dispositivo (`onnxruntime-qnn` win_arm64), sem AI Hub (nuvem); se o EP não registrar, T3 = "não homologado" e o notebook homologa só os fallbacks CPU ARM64 |
| 16 GB compartilhados (SO + IDE + MongoDB + NPU) | alta | paginação, TTFT instável, OOM do driver | orçamento de 5.5.1 (modelo ≤ 3 GB), `mongod` remoto ou com cache de 0,5 GB, um modelo por vez, janela estática 2k (T3) / 4k (Lite 0.6B), gate de memória em 7.3 |
| CPU/GPU de entrada (8 núcleos 2,97 GHz; Adreno X1-45) | média | fallback CPU lento, DML sem ganho | metas CPU ARM64 próprias em 7.3; DML do Adreno só relatório |
| A IDE não registra EPs do Windows ML; `GetAvailableProviders()` pode não listar EPs de plugin | alta | QNN invisível mesmo com pacote correto | Protótipo de registro (Fase 08, ADR) validado no S-NPU-2; harness próprio enquanto a IDE não muda |
| QNN para LLM exige GenAI mais novo que 0.15.2 e formato de pipeline que o `LocalModelCatalog` rejeita | alta | pacote `Invalid` na IDE | Atualizar GenAI (ADR) e aceitar `decoder.pipeline` no catálogo (Fase 08); `runtime.minimumGenAi` no metadata |
| Receita Olive não cobre Qwen2.5-Coder (só Qwen2.5-Instruct/Qwen3-0.6B) | média | falhas de grafo | Mesma arquitetura `qwen2`; se falhar, geração do contexto no próprio Vivobook (AI Hub é nuvem, fora do plano); em último caso, T3 com o modelo da receita oficial ajustado por LoRA |
| Calibração/GPTQ dependem de CUDA (receita) | média | sem GPU NVIDIA local e sem nuvem | ROCm local (PyTorch expõe o dispositivo como `cuda`) **[hipótese]**; WSL2 + ROCm oficial; GPTQ na CPU para 0.5B/1.5B; em último caso RTN/k-quant sem GPTQ e w8a16 |
| Contexto binário específico de SoC/QAIRT; X2 fora da lista de requisitos do Windows ML | alta | pacote do X1-26-100 não roda em X1E/X1P/X2 ou quebra após atualização do QNN EP | Um pacote por `soc_model`/`htp_arch`; S-NPU-8 se surgir outra máquina; manifestar QAIRT; reteste a cada KB do QNN EP; preferir contexto gerado com weight sharing e `embed_mode=0` |
| w4a16 perde qualidade no domínio (tokens raros) | média | ΔAPT > gate | Calibração do domínio; `lm_head` 8 bits; fallback `qnn-w8a16` |
| Janela estática curta (2048) incompatível com o agente | alta | T4 não roda em NPU | T4 NPU fica fora da v2 se o prompt do agente (sistema + tools + histórico) não couber; documentar |
| DirectML sem evolução ("legacy") | média | regressões futuras sem correção | Manter DML (funciona e está incluído no Windows ML); acompanhar EP de GPU do Windows ML (WebGPU experimental, MIGraphX "não suportado para GenAI") |
| Builder novo gera config que o runtime antigo não lê | média | pacote quebrado na IDE em produção | Exportar sempre com a versão exata do runtime da IDE; matriz de versão no manifesto |
| 14B/7B `dml-int4` excede VRAM com contexto grande (14B: ~9 GB + KV) | média | falha de carga | `recommendedContextTokens` conservador (4096–8192) e teste com o tier de memória da ADR-043; T4 8B: orçamento de 5.7.2 (≈ 9–11 GB com KV fp16 a 16k) medido no gate G7.7 |
| Logits do prefill de todas as posições em prompt de 14k (até ≈ 8 GB) | média | OOM no DML só com prompts longos | prefill em blocos de ≤ 2.048 tokens; conferir a saída do grafo do builder; teste de 14k em 7.1 |
| `ApplyChatTemplate` nativo não renderiza o template Jinja do Qwen3 igual ao HF (filtros/slices/`namespace`) | média | prompt fora do treino; histórico com raciocínio antigo | teste de renderização 100% (7.1); se falhar, a Fase 08 renderiza o formato fixo em C# validado pelos mesmos 200 goldens |
| Tokens `<think>`/`<tool_call>` não `special` impedem a gramática por token | média | guidance indisponível com thinking | gramática por id (`<[151667]>`); variante de tokenizer com `special:true` só com paridade refeita; sem guidance vale o parser estrito (Fase 04 §5.3) |
| Greedy no thinking repete até o orçamento (aviso do card) | alta se greedy | estouro sistemático, TTFA alto | pacote com amostragem do `generation_config`; IDE precisa enviar `top_p`/`top_k` (Fase 08); estouro medido em 7.1/Fase 06 |
| KV quantizado/prefix caching só na Engine 0.16+/paginada | alta | sem ganho de memória/latência no DML da v2 | KV fp16 e reuso por `RewindTo` na Generator API; `kv-int8` como spike pós-upgrade |
| Licenças (base 3B não Apache; termos do QNN EP) | baixa | publicação bloqueada | Checklist de licença por pacote no G7.5 |

## 12. Dependências

- [Fase 01](fase-01-selecao-modelos-base.md): arquitetura de cada tier (se não for `qwen2`, a exportação depende de suporte no builder e de novo adapter — Fase 08).
- [Fase 05](fase-05-treinamento.md)/[Fase 06](fase-06-avaliacao-benchmarks.md): merged aprovado e benchmark.
- [Fase 08](fase-08-integracao-kapibara.md): versão do GenAI na IDE, registro de EPs do Windows ML, catálogo aceitando pipeline QNN, contrato do agente, leitura dos blocos `reasoning`/`generation.agent` e envio de `top_p`/`top_k`.
- [Fase 04 §5.6](fase-04-dataset-agente-tools.md) / [Fase 05](fase-05-treinamento.md): formato do bloco `<think>`, texto de fechamento forçado usado no treino e contexto de 16k das trajetórias.
- Hardware: ASUS Vivobook 16 X1607QA (Snapdragon X X1-26-100, 16 GB, Windows 11 Home) — **disponível** para NPU e fallback CPU ARM64; GPU NVIDIA (opcional, só física; sem aluguel/nuvem). Horas de GPU da RX 7800 XT para export, calibração e benchmark no calendário único da [Fase 05 §9](fase-05-treinamento.md) (jobs J23, J27, J30: ~32 h).

## 13. Estimativa

| Bloco | Esforço | Máquina |
|---|---|---|
| Ambientes e versões | 1 dia | x64 |
| Calibração + fp32 + CPU/DML de T0/T1 | 3–4 dias (inclui benchmarks completos) | Ryzen + RX 7800 XT |
| T2 14B/7B e T4 8B (`dml-int4`, `int4` CPU, AWQ/GPTQ candidatos) | 3–5 dias (exportação 7B ≈ 1–2 h por variante, 14B ≈ 2–3 h **[H]**; benchmark ≈ 6–10 h) | idem |
| T4/Lite com raciocínio e 16k (5.7: config, template, metadata, paridade com raciocínio, contexto longo, VRAM/TTFA, smoke de guidance) | +3–4 dias (benchmark com thinking ligado/desligado ≈ 2× o tempo do T4 sem thinking) | Ryzen + RX 7800 XT |
| Spike S-NPU-1 (Olive) | 3–5 dias | x64 local: RX 7800 XT (ROCm) ou CPU para GPTQ/calibração; WSL2 se preciso |
| S-NPU-0 (inventário, controle, memória) | 1–2 dias | Vivobook X1-26-100 |
| S-NPU-2..7 + variantes QNN finais | 5–10 dias | Vivobook X1-26-100 |
| Empacotamento, cards, publicação | 2 dias | — |
| **Total** | **≈ 3,5–5,5 semanas**, das quais NPU ≈ 1,5–3 semanas no Vivobook X1-26-100 (hardware disponível; o risco restante é o suporte não oficial do QNN EP ao chip, decidido no S-NPU-0) | |
