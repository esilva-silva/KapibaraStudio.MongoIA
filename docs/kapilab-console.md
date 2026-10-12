# KapiLab — console .NET para gerar vetores de teste e testar os modelos pelo caminho da IDE

> Especificação (proposta) do executável `kapilab.exe` (nome proposto: **KapiLab**). Elaborada em 08/10/2026 a partir das Fases 00, 03, 04, 06, 07 e 08 deste plano, do harness da v1 (`C:\IA\models\SlopStudio.MongoAI-Compact\scripts\ide_harness.py`, `tools\ide-harness`, `docs\INTEGRATION.md`) e do código da IDE em **somente leitura** (`F:\source\esilva-silva\EsilvaSoft.KapibaraStudio`, HEAD `dba8546`).
>
> Convenção: **[F]** = fato verificado (arquivo/classe citados) · **[H]** = hipótese/proposta a validar. Identificadores em inglês; texto em pt-BR.

## 0. Resumo executivo

| Item | Decisão |
|---|---|
| Onde vive | **[H] Rota A estendida:** `tools\KapiLab\` **dentro do repositório da IDE**, com `ProjectReference` aos projetos de `src\` (precedente [F]: `tools\BrandAssets` já referencia o Desktop). Enquanto o repositório da IDE não aceitar a mudança (a Fase 00 diz que este plano não altera a IDE), **Rota B provisória**: a mesma árvore de código compilada a partir do workspace MongoIA com `HintPath` para as DLLs de uma build da IDE (como o harness v1). O código é organizado para a migração ser uma cópia de pasta. |
| Por que não reflexão | **[F]** quase tudo que o KapiLab precisa é `public` (catálogo, serviço de modelo, runtime ONNX, adapters, builder FIM, contrato `editor-context-v1`, sonda de hardware, `AgentToolRegistry.GetSessionChannel*`). Reflexão só para 2 textos privados (descrições das tools) — ver §2.3. |
| MVP (Fase K1) | `contract render\|tokenize\|diff` · `catalog export\|check` · `model inspect\|validate\|run` · `bench autocomplete` · `report` mínimo (+ `env`). |
| Depois | K2: `samples generate`, `bench chat`, `parity`, `guidance smoke`, `npu inventory\|smoke`. K3: `agent run\|bench` (laço de referência próprio até a Fase 08), `catalog invoke\|plans`. K4: `report` completo com gates, integração `gpu_queue`, win-arm64/QNN. |
| Exec-valid no MongoDB | **Fica no Python** (`eval_exec.py`/`exec_validate.py`). O console só fala com MongoDB no modo `agent --tools ide-registry` (opcional, K3) e nunca recebe connection string por argumento. |
| Principais bloqueios | Descrições das tools são `internal`/privadas; sem `InternalsVisibleTo` para fora dos assemblies de teste; tokenizer da IDE exige carregar o `Model` inteiro do GenAI; o agente local ainda não faz tool calling; EPs do Windows ML não são registrados pela IDE; `System.CommandLine` não está no `Directory.Packages.props`; lockfiles por backend. Detalhes na §2.3. |
| Estimativa | MVP ≈ 9–11 dias úteis; completo (K1–K4) ≈ 30–38 pessoa-dias (§10). |

## 1. Objetivo, usuários e não-objetivos

### 1.1 Objetivo

Um único executável de linha de comando que:

1. **Gera amostras e vetores de teste exatamente no contrato da IDE** — renderiza registros estruturados (`kapi-ds-record-v2`, Fase 03) no prompt **byte a byte** que a IDE monta (FIM Qwen + `editor-context-v1`/candidato `kapi-context-v2` + chat local), tokeniza com o tokenizer **da IDE**, e compara com o renderer Python usado no treino (gate G3.1 da Fase 03).
2. **Testa os modelos ONNX treinados pelo mesmo caminho de código do Kapibara Studio** — `LocalModelCatalog` → `LocalAiModelService` → `OnnxLocalModelRuntime` (autocomplete: `AiAutocompleteProvider`; chat: `LocalAgentProvider`), medindo TTFT, tok/s, RAM/VRAM, nulos e falhas (gates de latência da Fase 06 §5.3 passo 2, "o gate de latência usa o harness da IDE").
3. **Extrai da IDE os fatos que o dataset não pode inventar**: catálogo de tools do agente (schemas, categorias, limites, prompts), operadores/estágios reais (`LanguageDefinition`), papéis de cursor (`CompletionContextEngine`).
4. Cobre os spikes que só fazem sentido com as DLLs reais: `guidance smoke` (Fase 04 §5.4) e `npu smoke` (S-NPU-0..3, Fase 07 §7.4).

Substitui/absorve as ferramentas citadas nas fases 03–06: `ContractProbe`, `KapiToolCatalogExport`, `KapiRegistryHarness` (parte), `GuidanceSmoke`, `ide-harness`.

### 1.2 Usuários

| Usuário | Uso típico |
|---|---|
| Você, treinando | iterar rápido: renderizar 50 registros e conferir o prompt; carregar o T0 recém-exportado e ver o ghost text; medir TTFT antes de gastar uma noite de avaliação |
| CI local (`run_eval.ps1`, `gpu_queue.ps1`, `check_env.py`) | chamadas não interativas, saída JSON/JSONL, códigos de saída estáveis (§4.4) — não há CI na nuvem neste plano |

### 1.3 Não-objetivos

- **Não substitui** a suíte da IDE (`dotnet test` da solução, testes `Explicit` da Fase 08 §4.2) nem a homologação por hardware/uso interativo; é um instrumento do pipeline de treino.
- **Não tem UI** e não toca Avalonia, LiteDB do usuário, perfis de conexão reais, cofre de segredos nem rede (exceto MongoDB local de fixture, opcional).
- **Não treina, não exporta, não quantiza** (Python), **não executa queries de validação do dataset** (Python), **não publica** no Hugging Face.
- **Não altera** `Repos\*\exports`/`BaseRepos` (somente leitura); só escreve em `data\lab\`, `reports\lab\`, `Repos\<modelo>\reports\ide\` e `tmp\`.
- Não é fonte de verdade de gates: lê `gates.yaml` convertido (§5.11) e emite fatos/medidas.

### 1.4 Princípio "mesmo caminho de código" — mapa

| O que a IDE faz | Tipo público usado [F] | Comando do KapiLab |
|---|---|---|
| Descobrir/validar pasta de modelo | `LocalModelCatalog` (`OnnxAdapter`, `public sealed`), `LocalModelFileAccess` (`Infrastructure.System`) | `model inspect\|validate` |
| Escolher provider (NPU→GPU→CPU, explícito sem fallback) | `AiProviderSelector.Plan`, `OnnxHardwareProbe.Detect`, `OnnxExecutionProviders.Known` | `model inspect --hardware`, `model run --hardware` |
| Carregar e gerar | `LocalAiModelService` (`Application`) + `OnnxLocalModelRuntime` (`public sealed`) | `model run`, `bench *` |
| Montar contexto de editor | `AutocompleteContextBuilder.Build/ModelPrefix`, `EditorContextV1Contract.Instance`, `ExperimentalContextContracts.All` | `contract render` |
| Prompt FIM | `QwenFimPromptBuilder.Build(prefix, suffix, contextTokens, tokenizer)` | `contract tokenize` |
| Autocomplete (texto final do ghost) | `AiAutocompleteProvider.GetCompletionAsync` (+ `CompletionOutputProcessor`, `public static`) | `bench autocomplete`, `model run --role autocomplete` |
| Chat local atual | `LocalAgentProvider.CreateSessionAsync` → `IAgentSession.RunTurnAsync(AgentTurnRequest)` | `bench chat`, `model run --role chat` |
| Catálogo de tools | `AgentToolRegistry` (+ `GetSessionChannelDescriptors/InputSchemaJson/OutputSchemaJson`), `AgentProductToolNames.CategoryOf`, `AgentToolOutputScopes.For`, `AgentToolExposure.StageOf`, `AgentModePolicy.Plan`, `AgentSystemPromptBuilder.Build` | `catalog export\|plans` |

## 2. Fatos verificados na IDE (HEAD `dba8546`, SDK `10.0.400`)

### 2.1 Build, pacotes e lockfiles

| Fato | Fonte |
|---|---|
| SDK **10.0.400**, `rollForward: latestFeature`; TFM `net10.0`; `LangVersion=preview`; `Nullable`/`ImplicitUsings`; **`TreatWarningsAsErrors=true`**, `AnalysisLevel=latest-recommended`, `GenerateDocumentationFile=true` (`NoWarn 1591`) — o `Directory.Build.props` da raiz vale para qualquer projeto em `tools\` | `global.json`, `Directory.Build.props` [F] |
| `RestorePackagesWithLockFile=true`; **backend ONNX** `SlopOnnxBackend` ∈ `Cpu`/`WinML`/`Cuda` (qualquer outro valor quebra o build do `OnnxAdapter`); padrão `WinML` em Windows (`win-x64`, `win-arm64` ou sem RID) e `Cpu` no resto; lockfile `packages.lock.json` (Cpu) ou `packages.<Backend>.lock.json` | `Directory.Build.props`, `OnnxAdapter.csproj` [F] |
| Central Package Management + pinagem transitiva. Versões: `Microsoft.ML.OnnxRuntimeGenAI` / `.WinML` / `.Cuda` **0.15.2**, `Microsoft.ML.OnnxRuntime.Managed` **1.28.0** (só WinML), `MongoDB.Driver` 3.11.1, `ModelContextProtocol.Core` 2.2.0, `NUnit` 4.6.1, `BenchmarkDotNet` 0.15.8. **`System.CommandLine` não consta** | `Directory.Packages.props` [F] |
| WinML traz `Microsoft.Windows.AI.MachineLearning` **2.1.1** transitivo; a IDE tem **0 chamadas** a `ExecutionProviderCatalog`/`EnsureReady` (registro de EPs do Windows ML inexistente) | `packages.WinML.lock.json`; busca em `src\` [F] |
| CI: `dotnet restore EsilvaSoft.KapibaraStudio.slnx --locked-mode`; release: `restore <Desktop> --locked-mode -p:SlopOnnxBackend=<WinML\|Cpu>` para `win-x64`, `win-arm64` (cross-compile), `linux-*`. `tools\BrandAssets` **não** está na `.slnx` e tem o próprio `packages.lock.json` | `.github\workflows\*.yml`, `EsilvaSoft.KapibaraStudio.slnx` [F] |
| Projetos com RID declaram `<RuntimeIdentifiers>win-x64;win-arm64;linux-x64;linux-arm64</RuntimeIdentifiers>` para o lockfile carregar as seções de runtime (comentário "P7-CLP-3" no `McpServer.csproj`) | [F] |
| Existem pastas `src\EsilvaSoft.SlopStudio.*`/`tests\EsilvaSoft.SlopStudio.*` com só `bin\`/`obj\` (resíduo do renome), sem `.csproj` | [F] — ignorar |

### 2.2 Superfície de API e visibilidade

`InternalsVisibleTo` **existe só para assemblies de teste e vizinhos internos**: `Application` → `UnitTests`, `IntegrationTests`, `TestSupport`; `Core` → `Application`, `Infrastructure`, `Infrastructure.System`, `UnitTests`, `TestSupport`, `IntegrationTests`; `Infrastructure` → `UnitTests`, `IntegrationTests`; `Infrastructure.System` → `Infrastructure`, `Infrastructure.Agents`, `UnitTests`, `IntegrationTests`, `Infrastructure.Agents.Tests`; `McpServer` → `UnitTests`; `Desktop` → `UnitTests`, `IntegrationTests`. **`LocalAi.Core`, `Autocomplete.Core` e `OnnxAdapter` não declaram nenhum.** [F]

| Tipo | Projeto | Visibilidade [F] | Uso |
|---|---|---|---|
| `LocalModelCatalog` (ctor exige `ILocalModelFileAccess`; `workspacePaths` ou `defaultDirectory`) | `Infrastructure.LocalAi.OnnxAdapter` | `public sealed` | `Discover/ValidateAsync` |
| `LocalModelFileAccess`, `LocalProcessMemorySnapshot`, `LocalWorkspacePathResolver` | `Infrastructure.System` | `public sealed` | dependências do catálogo/runtime |
| `ILocalAiModelService` / `LocalAiModelService` | `LocalAi.Core` / `Application` | `public` | `GenerateAsync`, `StreamAsync`, `TestModelAsync`, `Status` (traz `FirstToken`, `TokensPerSecond`, `Backend`, `UsedFallback`, `ProcessMemoryBytes`) |
| `OnnxLocalModelRuntime` (ctor opcional: diagnostics, hardware, fileAccess, adapters, processMemory) | `OnnxAdapter` | `public sealed`; `RuntimeInfo` (`Backend`, `Provider`, `Device`, `LoadTime`, `ProcessMemoryBytes`, `UsedFallback`) | provider efetivo e fallback |
| `ModelGenerationResult` (`TimeToFirstToken`, `GeneratedTokens`, `Elapsed`, `Provider`, `IsComplete`, `UsedCpuFallback`), `GeneratedChunk` | `LocalAi.Core` | `public` | métricas |
| `IModelAdapter`, `QwenCoderModelAdapter`, `DeepSeekCoderModelAdapter`, `ModelAdapters.CreateDefault/For` | `OnnxAdapter` | `public` | tokenizer (`CreateTokenizer(Model, root)`), stop tokens, `Validate` |
| `QwenFimPromptBuilder` (`SpecialTokens`) | `Application` | `public sealed` | prompt FIM |
| `AutocompleteContextBuilder` (`Build`, `ModelPrefix`, `Commands`), `EditorContextV1Contract.Instance`, `AiContextContractResolver`, `ExperimentalContextContracts.All` (`experimental-{fewshot,json,minimal,sections}-v1`), `AutocompleteContextSnapshot` | `Application` | `public` | contratos de contexto |
| `AiAutocompleteProvider`, `AutocompleteService`, `CompletionOutputProcessor` (`Clean`, `ContainsReservedOrSensitiveText`, `public static`) | `Application` | `public` (mas `AiAutocompleteProvider.ContainsReservedOrSensitiveText/CleanGeneratedText` são `internal static`) | autocomplete |
| `LocalAgentProvider` (`Id="local"`), `AgentTurnRequest(TurnId, UserMessage, TabId, DocumentVersion, AuthorizedContext)`, `AgentContextProvider.CaptureAsync` | `Application` / `Core` | `public` (a sessão é classe aninhada privada) | chat local |
| `OnnxHardwareProbe` (`Detect()`, `Describe`), `AiProviderSelector.Plan`, `OnnxExecutionProviders.Known`, `AiExecutionPlan`, `OnnxEpDevice` | `OnnxAdapter` | `public` | inventário e plano de EPs |
| `AgentToolRegistry` (ctor público; `GetSessionChannelDescriptors`, `GetSessionChannelInputSchemaJson`, `GetSessionChannelOutputSchemaJson`, `IsSessionTool` `static`), `AgentToolExposure.Through/StageOf`, `AgentSessionToolPorts`, `AgentProductToolNames`, `AgentToolOutputScopes`, `AgentModePolicy.Plan`, `AgentSystemPromptBuilder.Build` | `Application` | `public` | catálogo/planos. Todas as portas (`IConnectionProfileRepository`, `IAgentAuthorizationPolicyProvider`, `IAgentPermissionEvaluator`, `IAgentAuditRepository`, `IMongoMetadataSource`, `IAgentMongoIndexSource`, `IAgentPrincipalAuthority`, `IAgentMcpSessionScopes`, `IAgentWorkspaceContextSource`, `IAgentEditProposalSink`, `IAgentWorkspaceFileCreator`, `IMetadataCache`) são `public interface` |
| `LanguageDefinition.Default` (símbolos/shapes reais de `mongodb-language.v1.json`), `CompletionContextEngine.Analyze` + `CompletionCursorRole`, `CompletionPrivacy`, `MongoLexer`/`TolerantParser` | `Autocomplete.Core` | `public` | `samples generate` (papel de cursor, operadores reais) |
| `MongoCodeValidator` (Acornima) | `Infrastructure` | `public sealed` | sintaxe das amostras |
| `AiRuntimeHarness`, `AiRuntimeScenario`, `AiRuntimeMeasurement`, `AiEvaluationDataset` (determinístico por índice), `AiRuntimeReportWriter` | `tests\EsilvaSoft.KapibaraStudio.Benchmarks` | `public` em projeto **Exe** de teste/benchmark (com BenchmarkDotNet/NUnit) | **referência de métricas/relatório**; [H] não referenciar no MVP (§8.2) |
| `LocalModelMetadataReader` | `OnnxAdapter` | **`internal static`** | só via `LocalModelCatalog` (`LocalModelDefinition.Metadata`) |
| `OnnxModelTokenizer`, `AdapterTokenizerFactory`, `AdapterPromptBuilder` | `OnnxAdapter` | **`internal`** | alcançáveis só por `IModelAdapter.CreateTokenizer` |
| `McpToolAdapter.Descriptions` | `McpServer` (Exe) | `internal` classe, campo `private static` | descrições das tools (§2.3 B1) |
| `CopilotSubscriptionAgentSession.ToolDescription` | `Infrastructure.Agents` | `internal sealed`, método `private static` | idem |

Capacidades/estados do catálogo [F]: `LocalModelState` = `NotInstalled, Available, Loading, Ready, Invalid, Unsupported, Failed, MissingFiles, NotLoaded` (a validação "ok" é `Available`); `LocalModelCapabilities` = `Autocomplete, Chat, Fim, Embeddings` (**não existem** `Agent`/`Tools`/`Reasoning` ainda — Fase 08 C4); `LocalModelContextContracts.Supported = ["editor-context-v1"]` e contrato desconhecido torna a pasta `Invalid`; `ModelGenerationRequest` só tem `Temperature`, `PromptTokens`, `StopSequences`, `PrefixCache` (inerte) — sem `TopP/TopK/Seed` (Fase 08 C8).

Do GenAI 0.15.2 (DLL gerenciada do build da IDE) confirmei por inspeção binária que existem `SetGuidance`, `Tokenizer.ApplyChatTemplate`, `GetBotTokenId`/`GetEotTokenId`/`GetBorTokenId`, `Generator.AppendTokens`, `RewindTo`, `GetOutput`, `GetSequence`, `Config.ClearProviders/AppendProvider`; `OrtEnv.GetEpDevices` está em `Microsoft.ML.OnnxRuntime.dll`. **[F]** (presença do símbolo; comportamento em DML/QNN é **[H]** até o smoke).

### 2.3 Bloqueios encontrados (e contorno)

| # | Bloqueio | Impacto | Contorno proposto |
|---|---|---|---|
| B1 | As **descrições** das tools mostradas aos modelos estão em `McpToolAdapter.Descriptions` (classe `internal` de um Exe) e `CopilotSubscriptionAgentSession.ToolDescription` (`internal`/`private static`); as de schemas/descriptors são públicas | `kapibara-tool-catalog.json` (Fase 04 §4.1) precisa de `description.mcp/copilot` | (a) **reflexão** `BindingFlags.NonPublic` só nesses dois membros, falhando com código 7 se sumirem (mantém a Rota B da Fase 04 reduzida a 2 membros — **os schemas não precisam mais de reflexão**: a Fase 04 §4.2 supunha `Descriptors`/`CatalogInputSchemaJson` privados, mas `GetSessionChannel*SchemaJson` são públicos [F]); (b) melhor: o Fase 08 criar uma fonte pública de descrições para o provider local (ele precisará delas no `chat_template`) e o KapiLab passar a lê-la |
| B2 | Sem `InternalsVisibleTo` para fora de testes; `LocalModelMetadataReader`, `OnnxModelTokenizer`, `AdapterTokenizerFactory` são `internal` | não dá para chamar o leitor de metadata nem a fábrica de tokenizer diretamente | metadata via `LocalModelCatalog` (público); tokenizer via `new QwenCoderModelAdapter().CreateTokenizer(model, root)` (público, devolve `ITokenizer`). **Pedir** (opcional) `InternalsVisibleTo Include="EsilvaSoft.KapibaraStudio.KapiLab"` no ADR; **não** nomear o assembly como `...IntegrationTests` para usar o IVT existente |
| B3 | O tokenizer da IDE só existe preso a um `Microsoft.ML.OnnxRuntimeGenAI.Model` (a `AdapterTokenizerFactory` abre uma sessão CPU "só para obtê-lo") — carregar o decoder inteiro para tokenizar | `contract tokenize` com o 14B custaria ~8 GB de RAM e segundos | **[H]** usar o menor pacote Qwen2.5 (T0 `int4`, ~0,4 GB) como *tokenizer host*; o `tokenizer.json` é cópia byte a byte do base e igual em todos os tiers Qwen2.5-Coder (Fase 07 §2.2, regra do tokenizer); `contract tokenize` grava o SHA-256 do `tokenizer.json` e recusa se divergir de `contracts\tokenizers.lock` |
| B4 | O prefixo do chat (`LocalAgentProvider.Session.BuildPrefix`) é `private` | `contract render --contract local-agent-fim-v0` não chama o método | **[H]** montar um `ILocalAiModelService` *gravador* e executar `RunTurnAsync`: o `Func<LocalModelDefinition, ModelGenerationRequest>` recebido por `StreamAsync` é invocado com um `LocalModelDefinition` falso e o `ModelGenerationRequest.Prefix/Suffix/ContextTokens/MaximumTokens/RequireFullContext` capturado é o prompt **real**, sem modelo e sem reimplementar o código. O `AuthorizedContext` real sai de `AgentContextProvider.CaptureAsync` com um `IConnectionProfileRepository` falso |
| B5 | `AiAutocompleteProvider.GetCompletionAsync` devolve só `AutocompleteResult` (sem TTFT/tokens) | `bench autocomplete` precisa de métricas **e** do texto final do ghost | **[H]** `MeasuringModelService`: decorador de `ILocalAiModelService` que repassa tudo e captura o `LocalModelGeneration` (TTFT, tokens, provider) de cada `GenerateAsync`; o `AiAutocompleteProvider` roda sem alteração por cima do decorador. Teste de equivalência: texto do decorador × provider direto = 100% |
| B6 | O agente local **não faz tool calling** (`SubmitToolResultAsync` lança `NotSupportedException`); `Agent/Tools/Reasoning` inexistentes; sem `top_p/top_k/seed` no runtime | `agent run\|bench` não tem o que chamar até a Fase 08 | **laço de referência próprio** (`--loop reference`) direto no GenAI (`ApplyChatTemplate`, parser §5.3 da Fase 04, orçamento de raciocínio com `AppendTokens`) e, quando a Fase 08 entregar, `--loop ide` sobre o `LocalAgentProvider`; um teste de paridade entre os dois laços vira gate (K3) |
| B7 | A IDE não registra EPs do Windows ML; `OnnxHardwareProbe` só vê `GetAvailableProviders()` + `GetEpDevices()` | `npu smoke` não pode "usar o que a IDE faz" | KapiLab implementa o registro (`ExecutionProviderCatalog.GetDefault().FindAllProviders()` → `EnsureReadyAsync` → `TryRegister`) como spike, **[H]** com o risco de o Windows ML exigir identidade de pacote/bootstrap do Windows App SDK em app não empacotado — verificar no S-NPU-0 |
| B8 | `System.CommandLine` fora do CPM; `TreatWarningsAsErrors`; lockfiles por backend | novo projeto precisa de `PackageVersion`, lockfiles `Cpu`/`WinML` e código sem warnings | `tools\KapiLab\Directory.Packages.props` que importa o da raiz e acrescenta `System.CommandLine` (**[H]**, evita editar o arquivo central); gerar os dois lockfiles (§3.3) |
| B9 | `ModelGenerationRequest.PrefixCache` inerte; `Generator` novo por pedido (sem KV reaproveitado) | latências do agente com raciocínio refletem esse custo | medir como está; registrar no relatório `runtime_features` (Fase 08 R/A6) |
| B10 | `AiRuntimeHarness` da IDE já mede matriz modelo × hardware × cenário (TTFT, tok/s, working set) mas vive em projeto Exe de benchmark com BenchmarkDotNet/NUnit | duplicar a medição é risco de divergência | KapiLab usa **os mesmos nomes de campo** (§6.4) e `LocalProcessMemorySnapshot`; consolidar (mover o harness para uma biblioteca) é item do ADR, não do MVP |

## 3. Decisão de arquitetura

### 3.1 Onde o projeto vive

| Opção | Prós | Contras | Veredito |
|---|---|---|---|
| **A. `tools\KapiLab\` no repositório da IDE**, `ProjectReference` a `src\*` | compila contra o código-fonte do commit (nunca diverge); Directory.Build.props/CPM/lockfiles herdados; testes de contrato podem usar os goldens (`EditorContextV1Corpus`); precedente `tools\BrandAssets`; ADR-xxx documenta | altera o repositório da IDE (Fase 00: "o repositório da IDE não é alterado por este plano" → vira PR/ADR, numeração ≥ ADR-070 [F: última é ADR-069]); `TreatWarningsAsErrors` vale para o código do laboratório; lockfiles novos | **Recomendada** (alvo) |
| B. no workspace MongoIA, `HintPath` para DLLs compiladas (como o `ide-harness` v1) | não toca a IDE; trabalha hoje | exige build prévio da IDE e cópia de nativos (`runtimes\win-x64\native`); fácil usar DLL velha; sem lockfile do GenAI controlando a versão; recompila quando a IDE muda | **Provisória** (bootstrap K1, até o aceite do PR) |
| C. no workspace MongoIA com `ProjectReference` por caminho absoluto para `F:\source\...` | código-fonte atual | o MSBuild escreve `obj\`/`bin\` e pode regravar lockfiles **dentro da IDE** (viola "somente leitura"); não herda `Directory.Build.props` da IDE no projeto do laboratório | **Rejeitada** |

Regras para a migração B → A ser trivial **[H]**: (1) todo acesso a tipos da IDE fica em `KapiLab.Ide\` (namespace único); (2) `KapiLab.Core` (JSONL, schemas, estatística, formatos, geradores) não referencia a IDE; (3) o `.csproj` usa `ProjectReference` com caminhos relativos a `$(KapibaraRoot)` e, na Rota B, `HintPath` com `$(IdeBin)` — escolhido por `-p:IdeMode=Source|Binary`.

### 3.2 Builds por backend (mesmo `-p:SlopOnnxBackend` da IDE)

O KapiLab **deve usar o mesmo backend e a mesma versão do GenAI que o app do usuário**, senão a medição não vale (Fase 06 §5.3: o gate é do harness da IDE; Fase 07 §4: `.venv-export` com `onnxruntime-genai` na versão da IDE).

| Build | Comando (a partir da raiz da IDE) | Serve para |
|---|---|---|
| `Cpu` | `dotnet publish tools/KapiLab -c Release -r win-x64 --self-contained false -p:SlopOnnxBackend=Cpu -o <out>` | smoke sem GPU, testes do próprio console, Linux (não homologado); **não** substitui a medição da CPU do usuário (o app Windows usa o pacote WinML) |
| `WinML` x64 (**padrão**) | idem com `-p:SlopOnnxBackend=WinML` | harness oficial: CPU EP e DirectML (RX 7800 XT) com a mesma DLL do app; `npu inventory` (sem QNN) |
| `WinML` **win-arm64** | `-r win-arm64 -p:SlopOnnxBackend=WinML` (cross-compile, como `release.yml`) | Vivobook X1-26-100: CPU Oryon ARM64, Adreno (DML), QNN após registro (S-NPU-2) |
| `Cuda` | `-p:SlopOnnxBackend=Cuda` | **não buildar** (sem GPU NVIDIA no plano — Fase 07 §5.4); fica documentado |

Detalhes **[H]**: publicar **dependente de framework** (sem `PublishSingleFile`) para manter as DLLs nativas do GenAI/DirectML ao lado do exe (lição v1 de pré-carga de DLLs, Fase 07 §2.2); artefatos copiados para `C:\IA\models\KapibaraStudio.MongoIA\tools\bin\kapilab\<backend>-<rid>\kapilab.exe`; `tools\kapilab.lock.json` registra `ide_commit`, `backend`, `rid`, `genai_version`, `ort_version`, SHA-256 do exe. Script `scripts\lab\build_kapilab.ps1 -Backend WinML -Rid win-x64`. `kapilab env` imprime tudo isso e `--expect-genai 0.15.2` retorna exit 7 se divergir (usado por `check_env.py`).

**Um processo, um modelo DML** (lição v1: carregar um 2º modelo DirectML no mesmo processo derrubou o processo — `docs\INTEGRATION.md` v1 [F]): `bench`/`model run`/`parity` carregam **um pacote por processo**; matrizes (`bench --matrix`) disparam processos-filho sequenciais.

### 3.3 Lockfiles

- Projeto com `RestorePackagesWithLockFile` herdado → gerar `tools/KapiLab/packages.lock.json` (Cpu) e `packages.WinML.lock.json` (WinML, com seções `win-x64` e `win-arm64`) declarando `<RuntimeIdentifiers>win-x64;win-arm64</RuntimeIdentifiers>` **[H, seguindo o comentário do McpServer]**. Restore sempre `--locked-mode -p:SlopOnnxBackend=<X>`.
- Como a `.slnx` não inclui `tools\` (precedente BrandAssets), o CI existente não passa a restaurar o KapiLab; um passo local `scripts\lab\build_kapilab.ps1` faz o restore travado. `NuGetLockFilePath` já é derivado do backend pelo `Directory.Build.props` [F].
- Upgrade do GenAI (Fase 08 R1: 0.15.2 → 0.17.1) regenera os lockfiles do KapiLab junto com os dos demais projetos e **exige reinstalar `onnxruntime-genai` nos `.venv-*`** do Python para a mesma versão (§11, R6).

## 4. Convenções da CLI

### 4.1 Forma

`kapilab <grupo> <ação> [opções]`, implementada com `System.CommandLine` (`--help` em pt-BR; nomes de opção em inglês). Opções **globais** (valem em todos os comandos):

| Opção | Padrão | Descrição |
|---|---|---|
| `--workspace <dir>` | env `KAPILAB_WORKSPACE`, senão `C:\IA\models\KapibaraStudio.MongoIA` | raiz do workspace (`BaseRepos`, `Repos`, `data`, `contracts`, `configs`, `reports`, `tmp`) |
| `--run-id <id>` | `yyyyMMdd-HHmmss-<cmd>` | nome do diretório de saída em `reports\lab\<run-id>\` |
| `--out <path>` | `-` (stdout) para dados; `reports\lab\<run-id>\` para artefatos | `-` = só stdout; logs **sempre** em stderr |
| `--format json\|jsonl\|md` | `jsonl` para listas, `json` para objetos | |
| `--seed <int>` | `20261008` (o seed do dataset, Fase 03) | geradores e amostragens |
| `--log-level error\|warn\|info\|debug` | `info` | `debug` **não** loga texto de prompt/predição (§7) |
| `--gpu-lock <path>` | `<workspace>\tmp\gpu.lock` | ver §4.5 |
| `--no-color` | — | |

### 4.2 Como um pacote é apontado

`--package <dir>` (pasta de um pacote ONNX) **ou** `--repo <Nome> --variant <dml-fp16\|int4\|...>` que resolve para `Repos\<Nome>\exports\<Nome>-ONNX-<variante>` (convenção da Fase 07 §6.2/§6.3 **[F-doc]**). Atalho de compatibilidade: se `--package` faltar, lê `KAPILAB_PACKAGE`, depois `SLOP_QWEN_MODEL` (mesma variável dos testes `Explicit` da IDE [F]). `--models-dir <dir>` + `--name <pasta>` usa `LocalModelCatalog.DiscoverAsync` como a IDE.

### 4.3 Hardware

`--hardware auto\|cpu\|gpu\|npu` → `AutocompleteSettings.Acceleration` (`AiAccelerationMode`); `--ep auto\|cpu\|dml\|cuda\|qnn\|openvino` → `AutocompleteSettings.ExecutionProvider` (`AiExecutionProvider`). **Explícito não cai para CPU** (regra da IDE, Fase 06 §5.3 passo 3): falha de provider = exit 5. `auto` registra a cadeia tentada e o fallback (`used_fallback`) — `bench`/`report` marcam como **não conformante** qualquer rodada de gate com `used_fallback=true`.

### 4.4 Códigos de saída

| Código | Significado |
|---:|---|
| 0 | sucesso |
| 1 | erro genérico/inesperado (exceção; stack em stderr, sem texto de prompt) |
| 2 | uso inválido (opção/argumento) |
| 3 | entrada inválida: JSONL fora do schema, campo obrigatório ausente, arquivo ilegível |
| 4 | pacote de modelo `Invalid`, `Unsupported`, `MissingFiles` ou `NotInstalled` |
| 5 | provider/hardware indisponível ou falha de carga **em modo explícito** |
| 6 | gate reprovado (`report`, `bench --gates`, `parity --gates`) |
| 7 | **deriva do contrato da IDE**: membro refletido sumiu, golden divergiu, `genai_version` ≠ esperado, hash do catálogo mudou (`catalog check`) |
| 8 | violação de privacidade: texto sensível/reservado na entrada ou saída (`CompletionPrivacy`, `<\|`, connection string) com `--on-sensitive fail` (padrão em `bench`/`samples`) |
| 9 | GPU em uso (`tmp\gpu.lock` retido por outro processo) |
| 10 | falha nativa/crash do runtime (processo-filho do `--matrix` terminou anormalmente) |
| 11 | divergência de paridade (`contract diff`, `parity`) acima da tolerância |
| 12 | cancelado/timeout |

Os códigos 5–11 são contratos com os scripts Python (§9): estáveis e documentados em `--help`.

### 4.5 Fila de GPU

Comandos que usam GPU (`--hardware gpu`/`--ep dml`, ou `auto` com GPU detectada, em `model run`, `bench`, `parity`, `agent run\|bench`, `npu smoke`) fazem, **antes de carregar o modelo**: `tmp\gpu.lock` retido por outro PID → exit 9 (mensagem com o PID/job); sem lock → **recusa** salvo `--via-queue` (o processo foi lançado por `gpu_queue.ps1`, que passa `KAPILAB_GPU_TOKEN`) ou `--standalone-gpu` (uso interativo na janela livre, cria e libera o lock sozinho, respeitando `tmp\gpu.pause`). `--allow-gpu-parallel` existe só para depuração e grava aviso no `env.json` e **desqualifica** a rodada para gates. O KapiLab **nunca** roda em paralelo com treino/llama.cpp (Fase 05 §9.6). Em CPU/NPU não há lock.

### 4.6 Saídas e logs

- Dados em JSON/JSONL com `schema` e `schema_version` em cada arquivo/linha de cabeçalho; UTF-8 **sem BOM**; `\n` entre linhas JSONL; *strings* de prompt preservadas exatamente (`\r\n` embutido vira `\r\n` escapado — nunca normalizar).
- `reports\lab\<run-id>\{env.json, args.json, metrics.json, per_example.jsonl, report.md}`; `env.json` registra: `ide_commit`, `backend`, `rid`, `genai_version`, `ort_version`, SO/driver GPU/NPU, CPU, RAM, provider efetivo, pacote (nome, SHA-256 do `kapicoder_manifest.json` e do `genai_config.json`), `used_fallback`, flags desqualificantes. **Nunca** grava variáveis de ambiente com segredos nem connection strings.
- stderr: linhas `LEVEL ts msg` e, ao final, um resumo de uma linha JSON (`{"cmd":..., "exit":..., "run_id":..., "out":...}`) que o Python lê.

## 5. Subcomandos

Notação: `<obrigatório>`, `[opcional]`. Todos aceitam as opções globais.

### 5.1 `env`

`kapilab env [--expect-genai <v>] [--expect-backend <b>] [--json]` — imprime build/ambiente (§4.6). Exit 7 se `--expect-*` divergir. **MVP.**

### 5.2 `contract render|tokenize|diff`

Objetivo: o prompt que o modelo vê, produzido pelo **código da IDE**, para os registros do dataset.

**Entrada:** JSONL `kapi-ds-record-v2` (Fase 03 §4.4; tipos `fim`, `completion`, `chat`, `correction`). **Contratos** (`--contract`):

| ID | Tipo de registro | Caminho de código [F] |
|---|---|---|
| `editor-context-v1` | `fim`, `completion` | `AutocompleteContextBuilder.Build(snapshot, settings)` → `ModelPrefix(request)`; `suffix = request.Suffix` |
| `experimental-{fewshot,json,minimal,sections}-v1` (candidatos a `kapi-context-v2`) | `fim` com `facts` | `ExperimentalContextContracts.All[..].Build(snapshot, settings)` **[H]**: precisa mapear `facts[]` → `AiFact`/`ExperimentalContextInput` |
| `local-agent-fim-v0` (chat local atual) | `chat`, `correction` | `LocalAgentProvider.RunTurnAsync` com *model service* gravador (B4); contexto via `AgentContextProvider.CaptureAsync` |
| `agent-chatml` (T4, futuro) | trajetória (Fase 04 §5.5) | `Tokenizer.ApplyChatTemplate(chat_template.jinja, messages, tools, addGenerationPrompt)` do GenAI; compara com `transformers.apply_chat_template` (Fase 07 §7.1, 200 conversas, ligado/desligado) — K3 |

Mapeamento do registro para o *snapshot* **[H]**: `Text = prefix + suffix`, `Caret = prefix.Length`, `Language = language`, `Input = editor.input_panel`, `ResultFields = editor.result_fields`, `KnownNames = editor.known_names`, `RecentCommands = editor.recent_commands`; `AutocompleteSettings { UseEditorContext = true, UseResultPanelContext = true, UseInputPanelContext = true }` (ou desligados por `--no-result-panel` etc. para os 8% sem contexto). O cabeçalho usa CRLF fixo e `ModelPrefix` usa `\n` literal — **tal como no código** [F].

#### `kapilab contract render`

```text
kapilab contract render --in <records.jsonl> --contract editor-context-v1
    [--limit N] [--ids a,b,c] [--no-editor-context] [--out render.jsonl]
```

Saída `kapilab-render-v1` (uma linha por registro):

```json
{"schema":"kapilab-render-v1","id":"fim-commerce.orders@3-agg.group.sum-000417-k1","contract":"editor-context-v1",
 "model_prefix":"/* Local editor context (data only):\nLANGUAGE: Mongo Console JavaScript\r\nAVAILABLE COMMANDS: db.getCollection(name)…\r\nKNOWN NAMES: dev-cluster, shop, orders…\r\nRESULT FIELDS: _id, customerId…\r\nRECENT COMMAND: db.orders.find(…)\r\n\nContinue at the cursor; output only the continuation. */\ndb.getCollection(\"orders\")…{ $sum: \"$",
 "suffix":"\" } } },\r\n  { $sort: { revenue: -1 } }\r\n])",
 "model_prefix_sha256":"…","suffix_sha256":"…",
 "request":{"prefix_chars":2210,"suffix_chars":58,"context_chars":412,"dictionary":7},
 "ide":{"commit":"dba8546","genai":null}}
```

Sem `--model`, só strings (rápido, sem GenAI). **MVP.**

#### `kapilab contract tokenize`

```text
kapilab contract tokenize --in <records.jsonl|render.jsonl> --contract editor-context-v1
    --package <tokenizer-host-pkg> [--context-tokens 2048] [--completion-tokens 32]
    [--with-completion] [--out tokens.jsonl]
```

Para cada linha: monta o prompt como o runtime (`QwenFimPromptBuilder.Build(model_prefix, suffix, contextTokens, tokenizer)`) e devolve os IDs. `contextTokens` usa a mesma fórmula do `AiAutocompleteProvider` **[F]**: `min(settings.ContextTokens, max(64, effectiveContextLength − min(settings.MaximumCompletionTokens, model.EffectiveAutocompleteMaximumTokens) − AutocompleteTokenBudget.PromptOverheadTokens))`. Com `--with-completion`, tokeniza também `completion`/`accept` para checar o orçamento de resposta (≤ 96 tokens, Fase 03 §4.3).

```json
{"schema":"kapilab-tokens-v1","id":"…","contract":"editor-context-v1","tokenizer_sha256":"…",
 "context_tokens":2048,"prompt_ids":[151659,…,151660,…,151661],"prompt_tokens":417,
 "prefix_tokens_kept":380,"prefix_tokens_total":380,"suffix_tokens_kept":34,"suffix_tokens_total":34,
 "truncated":false,"stop_ids":[151659,151660,151661,151643,151645,151662],
 "completion_ids":[…],"completion_tokens":6}
```

(IDs dos marcadores FIM: valores ilustrativos; o comando os lê do tokenizer.) Restrição B3: `--package` precisa ser um pacote Qwen2.5 (`qwen2`); o comando grava `tokenizer_sha256` e confere com `contracts\tokenizers.lock`. **MVP.**

#### `kapilab contract diff`

```text
kapilab contract diff --records <records.jsonl> --contract editor-context-v1
    --py <python_render.jsonl> [--package <tokenizer-host>] [--tolerance 0]
    [--sample 2000] [--out diff.jsonl]
```

`python_render.jsonl` é gerado por `scripts/data/render_contract.py --emit-jsonl` com os mesmos campos `id`, `model_prefix`, `suffix` (+ `prompt_ids` se houver tokenizer HF). O KapiLab renderiza pela IDE e compara **byte a byte** (UTF-8) e por IDs. Saída por registro `{"id","text_equal","ids_equal","first_diff":{"field","char_index","ide":"…","py":"…"}}` (trechos escapados, só de dados sintéticos) e resumo em `metrics.json` (`records`, `text_mismatch`, `ids_mismatch`, por campo/estratégia de corte). **Exit 11** se qualquer divergência (o gate G3.1 da Fase 03 é 100%, 2.000 registros por contrato). Armadilhas a cobrir em `first_diff.hint`: CRLF do cabeçalho × `\n` de `ModelPrefix`; troca `*/` → `* /`; escape do `JsonSerializer` no chat (`\u00E7`, `\u0022`…); janela 4.096/1.024 e `Take(128)` de nomes. **MVP.**

### 5.3 `catalog export|check|plans|invoke`

Substitui `KapiToolCatalogExport` (Fase 04 §4–§6) e a parte "golden de planos" do `KapiRegistryHarness`.

#### `kapilab catalog export`

```text
kapilab catalog export [--out data/agent/catalog/kapibara-tool-catalog.json]
    [--provider-id local|claude-code|github-copilot] [--descriptions reflect|none]
```

Como (**[F]** APIs públicas; **[H]** composição): constrói um `AgentToolRegistry` com portas falsas (todas as interfaces são públicas — §2.2), `AgentToolExposure.Through(AgentToolExposureStage.Metadata)`, `IAgentPrincipalAuthority` e `AgentSessionToolPorts` com `WorkspaceContext`, `MetadataCache`, `FileCreator`, `ProposalSink` não nulos — **sem isso `IsAvailable` esconde as tools de sessão** (`AgentToolRegistry.IsAvailable`). Depois: `GetSessionChannelDescriptors(providerId)`, `GetSessionChannelInputSchemaJson/OutputSchemaJson` (já com `$id = urn:esilvasoft:kapibarastudio:agent-tool:<nome>:v<versão>:<input|output>`), `AgentProductToolNames.CategoryOf`, `AgentToolOutputScopes.For`, `AgentToolExposure.StageOf`, `AgentToolRegistry.IsSessionTool`, `AgentSystemPromptBuilder.Build(AgentModePolicy.Plan(...))` sobre a matriz modo × permissões. Os nomes removidos (ADR-066) vêm das constantes públicas `AgentToolRegistry.*ToolName` (`MongoFindToolName`… no arquivo principal e as 5 de escrita em `AgentToolRegistry.LegacyNames.cs` [F]) — **o extrator parte dos descriptors, nunca das constantes** (armadilha da Fase 04 §3.4); `removedTools[]` é gravado só como lista negativa. `exposeToLocalAgent = (name != "approve")`.

Saída: o JSON da Fase 04 §4.1 (`catalogFormat: 1`), mais `catalog_sha256` = SHA-256 do JSON canônico (chaves ordenadas, sem `source.extractedAt`) e `source.{commit, dirty}` lido de `git rev-parse HEAD`/`git status --porcelain` do repositório da IDE. Descrições: B1 (reflexão em 2 membros, exit 7 se faltarem; `--descriptions none` as omite). **MVP.**

#### `kapilab catalog check`

`kapilab catalog check --against data/agent/catalog/kapibara-tool-catalog.<sha8>.json` → recalcula e compara o hash; exit 7 em divergência com *diff* estruturado (tools adicionadas/removidas, schemas alterados). Substitui o núcleo de `check_catalog_drift.py` (que passa a só chamar este comando). **MVP.**

#### `kapilab catalog plans` (K3)

Gera `plans-golden.json`: `AgentModePolicy.Plan` × `AgentOperationMode` × `AgentProviderPermissions` × `AgentPlatformFacts` (≥ 200 combinações, Fase 04 §6.1) + texto exato do `AgentSystemPromptBuilder` (≤ 2.048 bytes) — golden para a porta Python de `AgentModePolicy`.

#### `kapilab catalog invoke` (K3, **[H]**)

`--calls calls.jsonl` (`{"id","tool","arguments","state":{…}}`) → executa o `AgentToolRegistry.InvokeAsync` real com *doubles* de estado e devolve `{"status","errorCode","data","truncated"}` por chamada, para o teste de paridade espelho Python × registry (500 chamadas, ≥ 99%, Fase 04 §6.1). Risco: exige ~12 dublês (o `AgentSessionToolsTestRig` da IDE é `internal` no projeto de testes e não é reutilizável) — ver §11 R4.

### 5.4 `samples generate`

Gera **amostras/vetores de teste** (não dataset de treino) a partir de schemas/fixtures e do conhecimento real da IDE, determinísticas por seed, no formato JSONL das Fases 03/04.

```text
kapilab samples generate --kind fim|editor-context|chat|agent-case|all
    [--schemas data/schemas] [--n 500] [--seed 20261008]
    [--cut operator-value,property-key,stage,member-access,mid-token,eol,multiline,stmt-start]
    [--dialect console|json|mongosh] [--lang pt-BR|en|mixed]
    [--with-context all|none|mixed] [--out data/lab/<run-id>/samples.jsonl]
    [--validate syntax,api,cursor-role]
```

**Fontes (nada inventado):**

| Insumo | De onde | Para quê |
|---|---|---|
| Operadores, estágios, acumuladores, shapes | `LanguageDefinition.Default` (47 grupos, 13 snippets, 42 shapes [F, Fase 03 §2]) e `ShapeDefinition` | só emite operadores compatíveis com o *shape* e o tipo do campo |
| API do Console (métodos de banco/coleção/cursor, globais) | lista do `snapshot_ide.py` (`ConsoleBootstrap.cs`) lida de `data/sources/ide-snapshot/<commit>/console-api.json` | recusa método fora da lista (`validate api`) |
| Schemas e fixtures | `data/schemas/<domain>/<schema_id>.json` (Fase 03 §4.2) ou, sem eles, 3 schemas mínimos embutidos no KapiLab (`lab/orders`, `lab/customers`, `lab/events`) | campos e tipos realmente existentes |
| Papéis de cursor | `CompletionContextEngine.Analyze(ContextRequest)` → `CompletionCursorRole` | rotula `cursor_role` com o **mesmo motor da IDE** e verifica que o corte pedido cai no papel esperado |
| Sintaxe | `MongoCodeValidator` (Acornima) / `TolerantParser` | rejeita statements que a IDE não parseia |
| Contexto de editor | `AutocompleteContextSnapshot` sintético: conexões/bancos/coleções do schema, `known_names` ≤ 128, 3 `recent_commands` | `editor` do registro |
| Privacidade | `CompletionPrivacy` / `CompletionOutputProcessor.ContainsReservedOrSensitiveText` | descarta o que a IDE descartaria |

**Cortes FIM** seguem a tabela de alvos da Fase 03 §4.3 (chave 16%, valor tipado 14%, dentro de operador 14%, estágio 12%, membro 10%, token parcial 10%, resto de linha 12%, multilinha 8%, início de statement 4%); `--cut` seleciona subconjuntos para depurar uma classe de erro. Determinismo: PRNG próprio (SplitMix64 por índice, **[H]**, no estilo do `AiEvaluationDataset` da IDE — "gerar o índice 7 isolado dá o mesmo caso que gerar 10.000" [F]); nada de `System.Random` compartilhado nem relógio.

**Saída:** `kapi-ds-record-v2` com `origin: "kapilab:samples"`, `split: "lab"`, `label_source: "kapilab"`, `validation.exec: "skipped"`, `completion` = continuação do statement gerado (a referência é o próprio texto cortado). Para `agent-case`: linha `kapilab-agent-case-v1` (§6.5) com `plan`, `fixture`, `verify`, `accept_sequences` coerentes com o catálogo. **Guardas:** (1) recusa gravar sob `data\generated`, `data\validated`, `data\splits`, `data\eval` e `data\tiers` — só `data\lab\`; (2) recusa ler `data\eval\blind\` (G3.6); (3) nada de `ENV`/connection string/valores reais; só `example.*` e valores Faker-like embutidos.

**Uso:** entradas para `contract render`, `bench`, `parity` e para as 20 provas de `KAPI_AGENT_MODEL` (Fase 08 §4.2). Não entram em treino (o manifesto da Fase 03 rejeita `split:"lab"`). K2.

### 5.5 `model inspect|validate|run`

#### `kapilab model inspect`

```text
kapilab model inspect (--package <dir> | --models-dir <dir> | --repo <R> [--variant <v>])
    [--hardware] [--json]
```

Faz o que **Preferências → Autocomplete → Testar modelo** faz sem gerar texto: `LocalModelCatalog.ValidateAsync/DiscoverAsync` → estado (`Available`, `MissingFiles`, `Unsupported`, `Invalid`, `NotInstalled`), mensagem, arquitetura, `PromptFormat`, capacidades, `ContextLength`/fonte, `AutocompleteMaximumTokens`, `ModelSizeBytes`, bloco `Metadata` (hardware declarado, contrato de contexto, `recommendedContextTokens`, `generation.*`); com `--hardware`: `OnnxHardwareProbe.Detect()` (CPU/GPU/NPU: provider, dispositivo, memória, razão da indisponibilidade) e o `AiExecutionPlan` que `AiProviderSelector.Plan` produziria para cada `--hardware` (candidatos, `AllowFallback`, ou a exceção de incompatibilidade com o `hardware` do metadata). Saída `kapilab-model-inspect-v1`. Exit 4 se o estado ≠ `Available`. **MVP.**

#### `kapilab model validate`

`kapilab model validate --package <dir> [--strict-kapi] [--verify-hashes] [--load]`

1. Validação da IDE (acima). 2. `--strict-kapi`: valida `kapibarastudio-model.json` também contra os **blocos novos** que a IDE atual ignora (`agent`, `reasoning`, `generation.agent`, `npu`, `runtime.minimumGenAi` — Fase 07 §6.4/Fase 08 C5) com JSON Schema embutido, e confere `contextContract` ∈ {`editor-context-v1`, candidato}, `templateSha256` = SHA-256 de `chat_template.jinja`, limites (orçamento ≤ 8.192, `default ∈ [minimum, maximum]`). 3. `--verify-hashes`: `kapicoder_manifest.json` × SHA-256 real de cada arquivo (Fase 07 §7.6). 4. `--load`: carrega no provider pedido e gera 8 tokens (equivalente a `TestModelAsync`: pasta → tokenizer → sessão → provider → geração; devolve `LocalModelTestReport` com `Steps`). É o "leitor equivalente ao `LocalModelMetadataReader`" que o `package_exports.py` deve chamar (Fase 07 §7.6). Exit 4 em qualquer falha. **MVP** (`--strict-kapi` e `--verify-hashes` em K2).

#### `kapilab model run`

```text
kapilab model run --package <dir> --hardware auto|cpu|gpu|npu [--ep dml|qnn|…]
    --role autocomplete|chat
    ( --prefix <t> [--suffix <t>] | --file <f> --cursor <offset|line:col|marker> | --in <records.jsonl> | --interactive )
    [--contract editor-context-v1] [--max-tokens 32] [--temperature 0] [--stream] [--raw] [--json]
```

- `autocomplete`: monta `AutocompleteContextSnapshot` (contexto opcional: `--known-names`, `--result-fields`, `--recent-command`) → `AutocompleteContextBuilder.Build` → `AiAutocompleteProvider.GetCompletionAsync` sobre o `MeasuringModelService`; mostra o **texto do ghost** (pós-`CompletionOutputProcessor`) e, com `--raw`, o texto bruto do modelo.
- `chat`: `LocalAgentProvider` → `RunTurnAsync` com `AgentTurnRequest(…, AuthorizedContext)`; imprime os `MessageDelta` em streaming e o `AgentError` (código) se houver.
- Sempre imprime: **provider efetivo** (`RuntimeInfo.Backend/Provider/Device`), `LoadTime`, `UsedFallback`/`UsedCpuFallback`, TTFT, tok/s, tokens do prompt/gerados, RAM do processo. `--interactive` = REPL de uma sessão (modelo carregado uma vez; `:ctx`, `:hw`, `:raw`, `:quit`). Exit 5 se o provider explícito falhar. **MVP** (autocomplete) · chat em K2.

### 5.6 `bench autocomplete|chat`

Caminho real, métricas para gates e dados para os *scorers* Python.

```text
kapilab bench autocomplete --package <dir> --hardware <h> [--ep <e>]
    --in <records.jsonl|render.jsonl> [--limit 200] [--warmup 3] [--repeat 1]
    [--contract editor-context-v1] [--context-tokens 2048] [--completion-tokens 32]
    [--compare-python predictions.jsonl] [--gates configs/gates.ide.json --tier T1 --scenario cpu-int8]
    [--out reports/lab/<run-id>]
kapilab bench chat --package <dir> --hardware <h> --in <chat records.jsonl> [mesmas opções]
kapilab bench ... --matrix matrix.json     # vários pacotes×hardware, um processo-filho por célula
```

Protocolo (Fase 06 §5.3): 3 aquecimentos descartados; 200 pedidos (100 autocomplete com prompts de 512/1.024/2.048 tokens + 100 chat); **um processo por pacote**; máquina ociosa e sem treino (§4.5); autocomplete com 32 tokens (`MaximumCompletionTokens`), chat com teto 256 e `RequireFullContext` (como o `LocalAgentProvider`).

Por pedido, grava `per_example.jsonl` (`kapilab-bench-v1`, §6.4), com campos compatíveis com `AiRuntimeMeasurement` da IDE (`prompt_tokens`, `generated_tokens`, `ttft_ms_runtime`, `ttft_ms_observed`, `total_ms`, `decode_tok_s`, `working_set_bytes`, `provider`, `outcome`) mais: `prediction` (texto final do ghost/resposta), `raw` (opcional, `--keep-raw`), `is_complete`, `null_reason`, `error_type` (**só o tipo da exceção**, nunca a mensagem se puder conter texto), `record_sha256`.

Métricas agregadas (`metrics.json`): n, **nulos** (autocomplete que devolve `null`), **falhas do runtime** (exceção ≠ `LocalModelUnavailableException`), recusas por orçamento/cooldown, TTFT média/p50/p95 (runtime e observado), tok/s mediana/p05, RAM de pico (`LocalProcessMemorySnapshot`), VRAM dedicada/compartilhada de pico (contadores de desempenho do Windows `GPU Adapter Memory`/`GPU Process Memory` **[H]**), tempo de carga, `used_fallback`, concordância com `--compare-python` (normalização `" ".join(text.split())`, como o harness v1 [F]; gate ≥ 99,5%). Percentis: *nearest-rank* documentado. `--gates` avalia só os gates de dono `kapilab` do tier/cenário (§5.11) e dá exit 6 se algum falhar.

**Texto × métricas (B5):** o texto vem do `AiAutocompleteProvider` real; as métricas vêm do `MeasuringModelService` que o envolve. Uma varredura de equivalência (`--self-check 20`) roda os mesmos pedidos pelo runtime direto e confere texto bruto idêntico.

**Decisão sobre exec-valid no MongoDB: fica no Python.** Motivos: (1) as fixtures, dumps `mongodump`, clones por worker, `maxTimeMS` e o `mongosh` já são Python/Docker (Fase 03 §4.5, Fase 06 §4.3); (2) duplicar o executor em C# criaria duas verdades para "Exec Match"; (3) o console precisa só entregar `predictions` por `id`. O fluxo é `kapilab bench … --out R` → `eval_exec.py --predictions R/per_example.jsonl` (lê `id`, `prediction`). Exceção opcional (K3): `agent --tools ide-registry` usa `MongoDB.Driver` (já no CPM, 3.11.1) contra o Docker local para exercitar o registry real.

`bench chat` roda o `LocalAgentProvider` (contrato `local-agent-fim-v0`); quando a Fase 08 trouxer o contrato novo, o comando ganha `--contract agent-chatml`. K1: `bench autocomplete`; K2: `bench chat`.

### 5.7 `agent run|bench` (K3; depende da Fase 08 para o laço "ide")

```text
kapilab agent run   --package <T4 dir> --hardware gpu --case <case.jsonl|--prompt "…">
    [--loop reference|ide] [--tools sim|ide-registry|replay] [--thinking on|off] [--think-budget 1024]
    [--max-steps 12] [--seeds 3] [--mongo-uri-env KAPILAB_MONGO_URI] [--sim "python sim/mcp_server.py"]
    [--verify-cmd "python scripts/agent/verify.py --case {case} --run {run}"] [--trace out.jsonl]
kapilab agent bench --package … --in data/agent/eval/handwritten-v1.jsonl --thinking on,off --seeds 3
kapilab agent run   --replay trajectories.jsonl [--tools sim|ide-registry]   # sem modelo
```

- **Laço de referência (`--loop reference`)** — **[H]** direto no GenAI, seguindo o contrato da Fase 04: `Tokenizer.ApplyChatTemplate` (com o `chat_template.jinja` do pacote), `<think>` com orçamento (injeção do `forceCloseText` via `Generator.AppendTokens`), parser §5.3 (detecção `<tool_call>`, JSON estrito `MaxDepth=8`, allowlist do plano, um *retry* por mensagem), resultado em `<tool_response>` truncado a ≤ 6 KiB, amostragem não-greedy (T=0,6/top-p 0,95/top-k 20 no thinking; 0,7/0,8 sem) — que o runtime da IDE **ainda não suporta** (B6). É um *protótipo executável* do que a Fase 08 implementa e, por isso, o seu comportamento é o *oráculo* contra o `LocalAgentProvider` v2 (teste de paridade de tokens/rendering).
- **Laço `ide`**: `LocalAgentProvider` com tool calling quando existir; o KapiLab fornece o `IAgentToolRegistry` e a UI "ausente" (confirmações automáticas por política do caso).
- **Backends de ferramentas:** `sim` = processo `python sim/mcp_server.py` (stdio) acessado com `ModelContextProtocol.Core` (já no CPM, 2.2.0 [F]) — o mesmo servidor MCP que a Fase 06 §4.6b usa para o Claude CLI; `ide-registry` = `AgentToolRegistry` real com *doubles* de estado + `MongoAgentIndexSource`/`MongoMetadataSource` sobre o MongoDB Docker (**[H]**; connection string só por variável de ambiente); `replay` = resultados gravados.
- **Métricas** (`metrics.json`, nomes da Fase 06 §4.6/§4.6b): **tool selection** (cada passo ∈ conjunto aceitável do caso), **argument validity** (JSON Schema do catálogo), **tool hallucination** (nome fora do plano, inclui removidas), passos/ótimo, loops (≥ 3 chamadas idênticas), não-parada (teto 12), **violações de permissão** (chamada fora do plano/sem confirmação), **TTFA** (do envio ao 1º `<tool_call>` parseado ou 1º token visível da resposta final), tempo total (modelo + ferramentas, **excluindo** a execução da consulta pelo usuário), tokens de raciocínio (média/p95 por geração e por tarefa), **estouro de orçamento** (% de gerações com fechamento forçado), obediência à chave `enable_thinking`, idioma/CJK do raciocínio, vazamento de canários (`CANARY-…`) na resposta e no `<think>`. **Task success** vem do `--verify-cmd` (Python executa a consulta final no MongoDB — mesma regra do exec-valid) e é gravado de volta em `per_case.jsonl`; sem `--verify-cmd` o campo fica `null` e o comando não reprova gate de task success.
- **`--replay`**: (a) `--replay file` sem modelo: alimenta as mensagens `assistant` gravadas como se fossem saída do modelo, executa parser + ferramentas e compara resultados gravados × reexecução → valida parser/simulador/registry; (b) `--replay-tools`: modelo real, ferramentas servidas do gravado (isola o modelo).
- 3 sementes por caso; resultado do caso = maioria (Fase 06 §4.6). GPU: lock obrigatório (§4.5); T4 é DML INT4 16k (VRAM ≤ 10 GB).

### 5.8 `parity`

```text
kapilab parity --package <dir> --hardware <h> --ref data/lab/<run>/parity_ref.jsonl
    [--levels tokenizer,prompt,greedy,teacher] [--tolerance-file configs/parity.json] [--out …]
```

`parity_ref.jsonl` é gravado pelo `parity_check.py` (referência PyTorch/HF): `{"id","prompt":{"model_prefix","suffix"} ou "messages","prompt_ids","greedy_ids","greedy_text","teacher":{"ids":[…],"top1":[…],"top5":[[…]]}}`.

| Nível | O que compara | Caminho |
|---|---|---|
| `tokenizer` | IDs do tokenizer da IDE × HF (400 prompts, gate 100%) | `IModelAdapter.CreateTokenizer` |
| `prompt` | IDs do prompt FIM final (marcadores, truncagem por orçamento) | `QwenFimPromptBuilder` |
| `greedy` | texto/IDs gerados por `OnnxLocalModelRuntime` com `Temperature=0` × `greedy_ids` (Fase 06 §6: concordância ≥ 99% fp32, 97% fp16 DML; ONNX × harness ≥ 99,5%) | caminho real da IDE |
| `teacher` | top-1/top-5 por posição, *teacher forcing* | **GenAI direto** (`Generator.AppendTokens` + `GetOutput("logits")` **[H]**) — **não** é o caminho da IDE (o runtime da IDE só devolve texto); marcado `path:"genai-direct"` no relatório |

Saída por exemplo e agregada; tolerâncias vêm de um JSON derivado de `configs/gates.yaml`. Exit 11 se reprovar. K2.

### 5.9 `guidance smoke`

`kapilab guidance smoke --package <dir> [--grammar <file.lark>] [--hardware cpu|gpu]` — substitui `GuidanceSmoke` (Fase 04 §5.4): chama `GeneratorParams.SetGuidance("lark_grammar", …)` (símbolo presente em 0.15.2 [F]; funcionamento nos pacotes `WinML`/`Cuda` **[H]** — o build precisa de `--use_guidance`), gera com o prompt do T4, valida que não há "guidance not enabled" e que a saída respeita a gramática (`<tool_call>` + JSON do plano); testa também o lexema "preguiçoso" do `<think>` (Fase 04) e mede o overhead (µs/token). Saída `guidance-smoke.json` (`supported`, `error`, `grammar_ok`, `overhead_us_per_token`). Exit 1 se não suportado (a gramática é "defesa extra, não requisito"). K2.

### 5.10 `npu inventory|smoke`

Cobre o S-NPU-0..S-NPU-3 (Fase 07 §7.4). Requer o build **win-arm64 WinML** no Vivobook para a parte QNN.

```text
kapilab npu inventory [--out Repos/<modelo>/reports/npu/device.json]
kapilab npu smoke --package <qnn-pkg|cpu-arm64-pkg> [--register-ep] [--no-cpu-fallback] [--requests 200] [--control]
```

- `inventory` (**dia 1 da semana 1, K0/K1** — o README exige S-NPU-0 já na semana 1): SO (build, edição), CPU, RAM total/disponível, `OrtEnv.GetAvailableProviders()`, `OnnxHardwareProbe.Detect()`, `OrtEnv.GetEpDevices()` (hardware: tipo, vendor, descrição, metadados), driver da NPU (versão do pacote do driver Hexagon por PnP/WMI **[H]**), versão do Windows ML e QNN EP registrados, `soc_model`/`htp_arch` quando a ferramenta de plataforma da Qualcomm estiver presente (lida de arquivo, **[H]**: `qnn-platform-validator` roda fora do KapiLab e seu resultado é anexado). Grava `device.json`.
- `smoke --register-ep`: `ExecutionProviderCatalog.GetDefault()` → `FindAllProviders()` → para QNN `EnsureReadyAsync()` → `TryRegister()` (B7), repete o inventário e confirma `QNNExecutionProvider (NPU)`.
- `smoke --package`: carrega com provider `qnn` **explícito** e `session.disable_cpu_ep_fallback=1` (via overlay de `genai_config`/`Config.SetRuntimeOption` **[H]**), gera N pedidos do `samples`, mede TTFT, tok/s, **memória** (working set + commit privado + memória disponível do sistema + crescimento do pagefile durante a rodada — Fase 07 §7.3, orçamento ≤ 1,0/2,0 GB) e compara com o **mesmo modelo em CPU ARM64** no mesmo notebook (critério de valor do T3). `--control` roda o LLM de controle (pacote QNN pré-compilado) e confere que a variante resolvida é NPU. Saída `npu-smoke.json` com `verdict: viable|not_viable|inconclusive` e `reasons[]`. K2 (inventário em K1).

### 5.11 `report`

```text
kapilab report --repo <Modelo> [--tier T0] [--runs reports/lab/<id>,…] [--gates configs/gates.ide.json]
    [--merge-python Repos/<Modelo>/reports/eval/<run>/gates.json] [--out Repos/<Modelo>/reports/ide/<package>-<provider>/]
```

Consolida `bench`, `parity`, `npu`, `guidance`, `agent` em `Repos\<modelo>\reports\ide\<pacote>-<provider>\{metrics.json, per_example.jsonl, report.md, gates.json, env.json}` e **copia** o resumo para `reports\ide_harness\<pacote>-<provider>.json` (nome exigido pela Fase 06 §11 **[F-doc]**). **`gates.json` compatível com a Fase 06 §11** (cada gate: valor, limiar, aprovado, evidência):

```json
{"schema":"kapi-gates-v1","tier":"T1","package":"KapiCoder-Mongo-Small-ONNX-dml-fp16","provider":"DirectML",
 "owner":"kapilab","run_id":"20261120-031500-bench","ide_commit":"dba8546","genai":"0.15.2","qualified":true,
 "gates":[
  {"id":"T1.dml_fp16.ttft_mean_ms","value":151.3,"op":"<=","threshold":170,"passed":true,"evidence":"reports/lab/…/metrics.json#ttft.runtime.mean","owner":"kapilab"},
  {"id":"T1.dml_fp16.ttft_p95_ms","value":188.0,"op":"<=","threshold":200,"passed":true,"owner":"kapilab"},
  {"id":"T1.dml_fp16.decode_tok_s","value":96.4,"op":">=","threshold":90,"passed":true,"owner":"kapilab"},
  {"id":"common.harness.nulls","value":0,"op":"==","threshold":0,"passed":true,"owner":"kapilab"},
  {"id":"common.harness.runtime_failures","value":0,"op":"==","threshold":0,"passed":true,"owner":"kapilab"},
  {"id":"common.harness.python_agreement","value":0.9975,"op":">=","threshold":0.995,"passed":true,"owner":"kapilab"},
  {"id":"common.harness.effective_provider","value":"DirectML","op":"==","threshold":"requested","passed":true,"owner":"kapilab"}],
 "summary":{"passed":7,"failed":0,"skipped":0}}
```

**Donos dos gates:** o KapiLab avalia só os gates que dependem do código da IDE (latência TTFT/tok/s, RAM/VRAM, nulos, falhas, provider efetivo, concordância texto IDE × Python, TTFA/estouro/permissão no agente, memória e valor da NPU, paridade `tokenizer/prompt/greedy`, tamanho em disco). Os demais (APT, Exec Match, chat v2, McNemar…) são do Python (`check_gates.py`). `--gates configs/gates.ide.json` é gerado por `scripts/lab/gates_to_ide_json.py` a partir de `configs/gates.yaml` (**evita YAML no console**) e contém, por tier/cenário, `{id, metric, op, threshold}` — valores dos gates da Fase 06 §8 (T0: TTFT média ≤ 150 ms/p95 ≤ 250, RAM ≤ 1,8 GB, disco ≤ 700 MB; T1: CPU INT8 chat 500/650 ms, RAM ≤ 3,8 GB, DML FP16 170/200 ms e ≥ 90 tok/s; T2 DML INT4 por tamanho; T3 NPU p95 300/500 ms, ≥ 35/20 tok/s, memória; T4 TTFT 4k p95 ≤ 1,5 s, ≥ 30 tok/s, VRAM ≤ 10 GB, TTFA p50 ≤ 8 s/p95 ≤ 20 s). Quando as Fases 06 e 07 divergirem, **a Fase 06 prevalece** (README). `check_gates.py` funde `owner:"python"` e `owner:"kapilab"` num único `reports\eval\<tier>\<run>\gates.json`. `qualified:false` se a rodada teve `used_fallback`, `--allow-gpu-parallel`, máquina não ociosa (carga de CPU/GPU medida antes da rodada > 5%) ou `genai` ≠ o do pacote (`runtime.minimumGenAi`). Exit 6 se algum gate falhar. K1 (mínimo: bench→gates), K4 (completo).

## 6. Formatos de arquivo

Todos com `schema` (nome) e `schema_version` implícito no sufixo (`-v1`). Schemas JSON Schema 2020-12 embutidos no assembly (`KapiLab\Schemas\*.json`) e usados por `--validate-input`.

### 6.1 Entrada de registros (Fase 03)

`kapi-ds-record-v2` exatamente como na Fase 03 §4.4 (campos `type`, `prefix`, `suffix`, `completion`, `accept`, `language`, `editor{target, known_names, result_fields, recent_commands, input_panel, newline}`, `facts[]`, `agent_context`, `instruction`, `output`…). O KapiLab é **tolerante a campos extras** e **estrito nos que usa**; registro sem `id` ou com `type` desconhecido → exit 3 (ou `--skip-invalid`, que conta e relata).

### 6.2 `kapilab-render-v1` / `kapilab-tokens-v1` / `kapilab-contract-diff-v1`

Exemplos em §5.2. Campos estáveis: `id`, `contract`, `model_prefix`, `suffix`, `*_sha256` (SHA-256 dos bytes UTF-8), `prompt_ids`, `context_tokens`, `tokenizer_sha256`, `ide.commit`, `ide.genai`.

### 6.3 `kapibara-tool-catalog.json`

Formato `catalogFormat: 1` da Fase 04 §4.1 (`source`, `tools[]` com `name`, `version`, `risk`, `requiredPermissions`, `stage`, `isSessionTool`, `confirmationCategory`, `outputScope`, `exposeToLocalAgent`, `description{mcp,copilot}`, `inputSchema`, `outputSchema`; `removedTools[]`; `errorCodes{}`; `limits{}`; `systemPrompts[]`) + `catalog_sha256` + `kapilab{version, backend}`. Arquivo versionado com o hash no nome: `data\agent\catalog\kapibara-tool-catalog.<sha8>.json`.

### 6.4 `kapilab-bench-v1` (por exemplo) e `metrics.json`

```json
{"schema":"kapilab-bench-v1","id":"fim-commerce.orders@3-…-k1","kind":"autocomplete","package":"KapiCoder-Mongo-Small-ONNX-dml-fp16",
 "requested":"gpu/dml","effective":"DirectML","backend":"Gpu","device":"AMD Radeon RX 7800 XT","used_fallback":false,
 "outcome":"Generated","prompt_tokens":417,"generated_tokens":6,
 "ttft_ms_runtime":148.2,"ttft_ms_observed":153.9,"total_ms":201.4,"decode_tok_s":96.1,
 "working_set_bytes":2412339200,"vram_dedicated_bytes":3221225472,
 "prediction":"total","is_complete":true,"null_reason":null,"error_type":null,"record_sha256":"…"}
```

`outcome` ∈ `Generated|BudgetExhausted|Refused|Empty|Null|RuntimeFailure` (os quatro primeiros espelham `AiRuntimeOutcome` da IDE [F]). `metrics.json`: `{ "n", "nulls", "runtime_failures", "ttft_ms":{"mean","p50","p95","p99"}, "tok_s":{"median","p05"}, "ram_peak_bytes", "vram_peak_bytes", "load_ms", "agreement_python", "by_prompt_bucket":{…}, "by_cut_strategy":{…} }`.

### 6.5 Agente: `kapilab-agent-case-v1`, `kapilab-agent-run-v1`

Caso (uma linha; derivado do JSONL canônico da Fase 04 §5.5 — `id`, `catalog_sha256`, `lang`, `scenario`, `mode`, `plan`, `fixture`, `thinking{enabled,budget,soft_switch}`, `messages` só até a 1ª mensagem `user`, `verify`, `accept_sequences`):

```json
{"schema":"kapilab-agent-case-v1","id":"agt-s3-000104","catalog_sha256":"…","lang":"pt-BR","scenario":"S3","mode":"Agent",
 "plan":{"productTools":["get_workspace_context","get_indexes","get_query_results","propose_file_edit"],"proposalHandling":"ReviewRequired"},
 "fixture":{"db_seed":"commerce@3#seed20261008","workspace":"w1","connections":["dev-local"]},
 "thinking":{"enabled":true,"budget":1024,"soft_switch":null},
 "messages":[{"role":"user","content":"por que a consulta de pedidos está lenta?"}],
 "accept_sequences":[["get_workspace_context","get_indexes","propose_file_edit"]],
 "verify":{"min_calls":2,"max_calls":6,"required_tools":["get_indexes"],"forbidden_tools":["mongo_find"],"final_checks":["no_collscan_after_apply"]}}
```

Execução (`--trace`): uma linha por evento: `{"schema":"kapilab-agent-run-v1","case":"…","seed":1,"step":3,"t_ms":4210,"event":"reasoning_delta|reasoning_closed|tool_call|tool_result|final|budget_forced_close|protocol_violation","tool":"get_indexes","args_sha256":"…","valid_schema":true,"in_plan":true,"tokens":212}`. Texto de raciocínio/valores de documento **não** são gravados no trace (apenas contagens e hashes), salvo `--keep-text` (padrão desligado; só para fixtures sintéticas). `per_case.jsonl`: métricas do §5.7.

### 6.6 `parity_ref.jsonl`, `device.json`, `npu-smoke.json`, `guidance-smoke.json`

`parity_ref.jsonl` em §5.8. `device.json`/`npu-smoke.json` conforme §5.10 (`verdict`, `reasons[]`, `ep_devices[]`, `windows_ml`, `driver`, `soc_model`, `htp_arch`, `memory{ws,commit,available_start,available_end,pagefile_growth}`).

### 6.7 Matriz de `bench --matrix`

```json
{"cells":[{"package":"Repos/KapiCoder-Mongo-Small/exports/KapiCoder-Mongo-Small-ONNX-int8","hardware":"cpu","scenario":"cpu-int8"},
          {"package":"Repos/KapiCoder-Mongo-Small/exports/KapiCoder-Mongo-Small-ONNX-dml-fp16","hardware":"gpu","ep":"dml","scenario":"dml-fp16"}]}
```

## 7. Privacidade e segurança

- **Sem valores de documentos reais.** Entradas aceitas: `origin ∈ {gen:*, fixture:*, kapilab:*, v1:*}`; outras origens → exit 3. O trace do agente e os logs gravam hashes/contagens, não texto. `--log-level debug` continua sem texto de prompt/predição.
- **Filtros da IDE aplicados a toda entrada e saída**: `CompletionPrivacy.ContainsSensitiveText`, `CompletionOutputProcessor.ContainsReservedOrSensitiveText` (marcadores `<|`/`<｜`, segredos). Padrão `--on-sensitive fail` em `bench`/`samples` (exit 8); `skip` conta e relata. Replica o que o `AiAutocompleteProvider` já recusa.
- **Sem credenciais e sem rede.** Nada de HF, atualização, telemetria: `OrtEnv.DisableTelemetryEvents()` como a IDE [F]; **não instancia** `HuggingFaceModelSource`/`IRemoteModelSource`. O único destino de rede é o MongoDB **local de fixture** (`127.0.0.1`/`localhost` de `data\fixtures\docker-compose.yml`), lido da variável `KAPILAB_MONGO_URI`; o valor **nunca** vai para argv, logs, `env.json` nem relatórios (aparece como `"<redacted:local-fixture>"`); URI com host fora de loopback é recusada.
- **Isolamento do conjunto cego.** O KapiLab recusa ler `data\eval\blind\**` (por caminho e por hash do manifesto) a menos que `--allow-blind --reason "<texto>"`; cada leitura registra `reports\dataset\blind_access.log` (quem, quando, modelo) — Fase 03 §4.6, G3.6. `samples generate` nunca escreve em `data\eval`.
- **Somente leitura sobre `BaseRepos\`, `Repos\*\exports`, IDE e `contracts\`**; escritas só nas pastas da §1.3. Download/publicação: inexistentes.
- **Reflexão (B1) limitada** a 2 membros, sempre com falha ruidosa (exit 7); nenhum outro acesso a membros privados.
- **Licenças:** o exe não embute pesos nem tokenizers; os artefatos de dados (`samples`) usam só a gramática da IDE (MIT), nunca texto da documentação do MongoDB (CC BY-NC-SA — Fase 03 §3).

## 8. Estrutura do projeto

### 8.1 Árvore (Rota A, no repositório da IDE) **[H]**

```text
tools\KapiLab\
  KapiLab.csproj                    # net10.0, Exe, AssemblyName=kapilab; ProjectReference a src\*; PackageReference System.CommandLine
  Directory.Packages.props          # <Import> do raiz + PackageVersion System.CommandLine
  packages.lock.json  packages.WinML.lock.json
  Program.cs                        # RootCommand, opções globais, mapeamento de exit codes
  Commands\                         # um arquivo por grupo: ContractCommands, CatalogCommands, ModelCommands, BenchCommands,
                                    #   SamplesCommands, AgentCommands, ParityCommands, GuidanceCommands, NpuCommands, ReportCommands, EnvCommand
  Ide\                              # TODO acesso a tipos da IDE (único ponto a migrar entre Rota A/B)
    IdeComposition.cs               # LocalModelCatalog + LocalAiModelService + OnnxLocalModelRuntime + AutocompleteService
    MeasuringModelService.cs        # decorador de ILocalAiModelService (B5)
    RecordingModelService.cs        # captura ModelGenerationRequest (B4)
    ContractRenderers.cs            # editor-context-v1, experimental-*, local-agent-fim-v0
    CatalogExtractor.cs             # AgentToolRegistry + doubles (B1/B8)
    Hardware.cs  PrivacyGuards.cs
  Core\                             # sem dependência da IDE
    Jsonl\  Schemas\  Stats\ (percentis, bootstrap leve)  Samples\ (SplitMix64, cortes, schemas embutidos)
    Gates\ (avaliador de gates.ide.json)  Reports\  Workspace\ (resolução BaseRepos/Repos, locks, blind guard)
    Agent\ (laço de referência, parser, orçamento, métricas)  Npu\
  Schemas\*.json                    # JSON Schemas embutidos (§6)
tools\KapiLab.Tests\                # NUnit 4.6.1; fora da .slnx no início [H]
```

Rota B (provisória): mesma árvore em `C:\IA\models\KapibaraStudio.MongoIA\tools\KapiLab\` com `-p:IdeMode=Binary -p:IdeBin=F:\source\...\Desktop\bin\Debug\net10.0` (referências `HintPath` às DLLs `Core`, `Autocomplete.Core`, `LocalAi.Core`, `Application`, `Infrastructure.System`, `Infrastructure.LocalAi.OnnxAdapter`, `Infrastructure`, `Microsoft.ML.OnnxRuntimeGenAI` e cópia de `runtimes\<rid>\native\*.dll`, como o `IdeHarness.csproj` v1 [F]).

### 8.2 Dependências

| Dependência | Motivo | Situação |
|---|---|---|
| `System.CommandLine` | CLI | **[H]** novo `PackageVersion` local (não está no CPM da IDE) |
| Projetos da IDE: `Core`, `Autocomplete.Core`, `LocalAi.Core`, `Application`, `Infrastructure.System`, `Infrastructure.LocalAi` (→ `OnnxAdapter`), `Infrastructure` (só `MongoCodeValidator` e, em K3, `Mongo*Source`) | caminho real | [F] públicos |
| `Microsoft.ML.OnnxRuntimeGenAI` / `.WinML` + `Microsoft.ML.OnnxRuntime.Managed` | vêm **transitivos** do `OnnxAdapter` conforme `SlopOnnxBackend`; o KapiLab usa as APIs diretas (`Model`, `Tokenizer`, `Generator`, `GeneratorParams`, `OrtEnv`) para parity/guidance/agent/npu | [F] 0.15.2 / 1.28.0 |
| `ModelContextProtocol.Core` | cliente MCP para `kapisim` (K3) | [F] 2.2.0 no CPM |
| `MongoDB.Driver` | só `--tools ide-registry` (K3) — **opcional**, via `Infrastructure` | [F] 3.11.1 |
| `Microsoft.Windows.AI.MachineLearning` | `npu smoke --register-ep` | [F] transitivo no WinML; uso de API **[H]** |
| **Não** referenciar `tests\Benchmarks` | evitar BenchmarkDotNet/NUnit/`IsTestProject` no exe; campos de medição copiados (B10) | decisão do ADR consolidar depois |
| Nada de YAML | `gates.yaml` → JSON por script Python | **[H]** |

### 8.3 Testes do próprio console (`tools\KapiLab.Tests`)

1. **Goldens do contrato:** `contract render` sobre `EditorContextV1Corpus` (vetores golden da IDE, `tests\…\Ai\EditorContextV1Corpus.cs` [F]) = bytes idênticos aos `EditorContextV1GoldenTests` — prova que o console não reimplementa o contrato.
2. **Gravador de chat:** `RecordingModelService` reproduz o prefixo documentado da Fase 03 (`/* Answer the instruction with code…`) para os exemplos conhecidos (`LocalAgentProviderTests` da IDE).
3. **Catálogo:** `catalog export` tem 12 descriptors/11 expostos, `approve` fora, 0 nomes removidos; mutação (remover um membro refletido por *fake*) ⇒ exit 7.
4. **Determinismo:** `samples generate --seed X` duas vezes ⇒ SHA-256 idêntico; índice isolado ⇒ mesma linha.
5. **Privacidade:** entrada com `mongodb://…`, `<|fim_pad|>`, `.env` ⇒ exit 8; caminho `data\eval\blind` ⇒ recusa.
6. **Matemática:** percentis/tok/s com vetores pequenos; `gates` com `op` e `threshold` de todos os tipos.
7. **Sem pesos:** modelo falso (`ILocalModelRuntime` determinístico) para `bench`/`model run`; o teste com pesos reais é `Explicit` (`KAPILAB_PACKAGE`), como na IDE.
8. **Integração Python:** `pytest` em `scripts/lab/tests/` chama o exe com 20 registros e valida o schema/exit codes.

## 9. Integração com o pipeline Python, `gpu_queue.ps1` e `BaseRepos`/`Repos`

### 9.1 Wrappers

- `scripts\lab\kapilab.py`: `run(*args, expect=0) -> (returncode, stdout_jsonl, summary)` — localiza o exe por `tools\kapilab.lock.json`, passa `--workspace`, lê o resumo JSON do stderr, converte exit codes em exceções tipadas (`ContractDrift`, `PrivacyViolation`, `GateFailed`, `GpuBusy`).
- `scripts\lab\build_kapilab.ps1 -Backend WinML -Rid win-x64|win-arm64`: restore `--locked-mode`, publish, grava `kapilab.lock.json`.
- `scripts\lab\gates_to_ide_json.py`: `configs\gates.yaml` → `configs\gates.ide.json`.

### 9.2 Quem chama o quê

| Script / etapa | Chamada ao KapiLab | Gate/fase |
|---|---|---|
| `snapshot_ide.py` (Fase 03 §4.1) | `kapilab env` + `catalog export --out data/agent/catalog/…` | catálogo versionado por commit |
| `render_contract.py` + teste de paridade | `kapilab contract render/tokenize` → `diff` (2.000 registros por contrato) | **G3.1** (exit 11 bloqueia) |
| `check_catalog_drift.py` (Fase 04 §4.2) | `kapilab catalog check` | exit 7 bloqueia o treino |
| `sim/parity/` (espelho × registry, 500 chamadas) | `kapilab catalog invoke` | Fase 04 §6.1 (≥ 99%) |
| `package_exports.py` (Fase 07 §7.6) | `kapilab model validate --strict-kapi --verify-hashes --load` | leitor equivalente ao da IDE |
| `parity_check.py` (Fase 07 §7.1) | grava `parity_ref.jsonl` → `kapilab parity` | paridade CPU/DML/QNN |
| `run_eval.ps1` (Fase 06 §11) | `bench autocomplete/chat` → `eval_exec.py --predictions` → `report` → `check_gates.py` | gates de latência/harness |
| `ide_harness.py` (v1) | **substituído** por `kapilab bench`; o `.py` vira wrapper fino | — |
| `check_env.py` | `kapilab env --expect-genai <v> --expect-backend WinML` | R6 (versões) |
| Fase 08 §4.2 | `kapilab npu smoke`, `agent bench` como pré-triagem antes dos `Explicit` | — |

### 9.3 `gpu_queue.ps1`

Jobs de GPU do KapiLab entram em `configs\gpu_queue.yaml` como qualquer outro (job, comando, dependências, janela): ex.: `kapilab bench autocomplete --package … --hardware gpu --via-queue --gates …` com `max_restarts` 1 (watchdog da Fase 05 §9.6 mata o processo se o `metrics.jsonl`/`per_example.jsonl` não avançar em 15 min — o KapiLab escreve um *heartbeat* (`tmp\kapilab.heartbeat`) a cada 30 s). `gpu_queue.ps1 -Pause` (`tmp\gpu.pause`) deve ser respeitado pelo modo `--standalone-gpu`. Medições de latência só rodam com a fila **vazia de treino** (Fase 06 §5.3: máquina sem outras cargas); o `report` registra a carga medida (CPU/GPU) antes da rodada e marca `qualified:false` se alta. Horas de GPU entram em `reports\gpu_queue\ledger.csv` (a fila lê o `elapsed` do resumo).

### 9.4 Pastas

- Lê `Repos\<Modelo>\exports\<Família>-ONNX-<variante>` e `merged\` (só para metadados); `BaseRepos\<Base>\tokenizer.json` (para `tokenizers.lock`); `data\schemas`, `data\fixtures` (compose), `data\agent\eval`.
- Escreve `data\lab\<run-id>\` (amostras), `reports\lab\<run-id>\` (artefatos brutos), `Repos\<Modelo>\reports\ide\<pacote>-<provider>\` (consolidado), `reports\ide_harness\…` (compat Fase 06), `Repos\<Modelo>\reports\npu\device.json`, `tools\bin\kapilab\…`, `tmp\`.
- `model.lock.json` de cada repo (Fase 00) ganha, por `report`, o par `{kapilab_version, ide_commit, genai}` usado — rastreabilidade da medição.

## 10. Plano de implementação

| Fase | Entregas | Esforço (pessoa-dia) | Critério de saída |
|---|---|---:|---|
| **K0** Bootstrap (semana 1) | ADR proposto + PR de `tools\KapiLab` (ou Rota B), `KapiLab.csproj`, `Directory.Packages.props`, lockfiles Cpu/WinML (x64 e arm64), `env`, `build_kapilab.ps1`, **`npu inventory`** (S-NPU-0 no Vivobook) | 2–3 | `kapilab env` e `npu inventory` rodam nos dois hardwares; `restore --locked-mode` verde |
| **K1 MVP** | `contract render\|tokenize\|diff` (incl. wrapper Python de paridade), `catalog export\|check`, `model inspect\|validate\|run` (autocomplete), `bench autocomplete` (+ `MeasuringModelService`), `report` mínimo/gates de latência, testes 1–5 e 7 da §8.3 | 7–8 | G3.1 verde em 2.000 registros (T0/T1, `editor-context-v1`); catálogo gerado e `check` verde; `bench` reproduz o harness v1 no pacote SlopCoder 0.5B INT4 (0 nulos, 0 falhas, concordância Python ≥ 99,5%, latências coerentes com o v1: ~381 ms autocomplete [F, INTEGRATION v1]) |
| **K2** | `samples generate`, `bench chat` (via `LocalAgentProvider`), `model validate --strict-kapi/--verify-hashes`, `parity` (4 níveis), `guidance smoke`, `npu smoke`, `contract` para candidato v2 (`experimental-*`) | 9–11 | amostras determinísticas e aceitas pelo `contract`+`bench`; parity T0 fp16 DML ≥ gates; veredito S-NPU-0..3 gravado em `device.json`/`npu-smoke.json` |
| **K3** | `agent run\|bench` com laço de referência + `sim` (MCP) + `--replay`, `catalog plans\|invoke`, `--loop ide` quando a Fase 08 entregar, `contract agent-chatml` (200 goldens) | 10–12 | 20 cenários do `KAPI_AGENT_MODEL` rodam; paridade laço referência × `LocalAgentProvider` v2 (tokens/rendering) |
| **K4** | `report` completo (merge com `check_gates.py`), integração `gpu_queue` (heartbeat, ledger), VRAM por contadores, `--matrix`, runbook, build win-arm64 em rotina | 3–4 | `run_eval.ps1` fecha T0–T4 com `gates.json` único |
| **Total** | | **≈ 31–38** | MVP (K0+K1) ≈ 9–11 dias úteis |

Ordem sugerida em relação ao cronograma do README: K0–K1 nas semanas 1–2 (a paridade de contrato e o catálogo são pré-requisito das Fases 03/04 — o `ContractProbe` e o extrator já estavam planejados nesse intervalo); K2 antes do 1º export ONNX (≈ semana 9–10); K3 em paralelo à Fase 08 (semanas 7–13); K4 antes do primeiro bloco de homologação (semana 14+). O `npu inventory` precisa estar pronto na semana 1 (S-NPU-0).

## 11. Riscos e mitigação

| # | Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|---|
| R1 | **APIs internas/privadas mudam** (B1: descrições; construtor do `AgentToolRegistry`; `ILocalAiModelService` ganhar membros) | média | médio | reflexão só em 2 membros com exit 7; `Ide\` isolado; CI local roda `catalog check` e os goldens a cada mudança de commit da IDE; pedir à IDE fonte pública de descrições (Fase 08) e `InternalsVisibleTo` opcional |
| R2 | **Mudança do contrato de prompt** (v2 declarado na IDE, `LocalModelContextContracts.Supported` cresce; formato do chat da Fase 08) | alta | alto | renderização sempre pelo código da IDE (nunca cópia); `--contract` explícito; `diff` + G3.1 reprovam divergência; registros guardam fatos, não texto |
| R3 | Tokenizer da IDE exige carregar `Model` inteiro (B3) | alta | baixo | *tokenizer host* pequeno + `tokenizers.lock`; se a IDE expuser tokenizer sem sessão, trocar |
| R4 | `catalog invoke`/ide-registry exige ~12 dublês e `AgentSessionToolsTestRig` é `internal` de teste | média | médio | começar por `export/check/plans` (só descriptors); `invoke` na K3 com dublês mínimos; se inviável, a paridade espelho × registry roda como teste na IDE (Rota A da Fase 04 §4.2) |
| R5 | **Medição não é a do app**: backend diferente (Cpu × WinML), build Debug, DLLs de DML em pasta errada, 2 modelos DML no mesmo processo | média | alto | `env.json` obrigatório; gate só com `WinML` Release e mesma versão do GenAI; um pacote por processo; `qualified` |
| R6 | **Versão do ORT GenAI diferente** entre IDE (0.15.2; alvo 0.17.1 na Fase 08 R1), `.venv-export`/`.venv-winml` e o KapiLab | alta no upgrade | alto | `kapilab env --expect-genai` em `check_env.py`; um único `genai_version` por rodada gravado no `kapilab.lock.json` e nos pacotes (`runtime.minimumGenAi`); rebuild + reinstalação dos venvs no mesmo PR de upgrade; teste de paridade antes/depois |
| R7 | Windows ML/QNN exige identidade de pacote ou bootstrap em app não empacotado; X1-26-100 fora da lista oficial | média | alto (T3) | `npu inventory` cedo; fallback do S-NPU-0 (Python ARM64 + `onnxruntime-qnn`); KapiLab reporta `inconclusive` em vez de forçar |
| R8 | Laço de referência do agente diverge do que a Fase 08 implementar | média | médio | tratá-lo como protótipo/oráculo; teste de paridade (tokens renderizados, parser) com `LocalAgentProvider` v2; `--loop` explícito no relatório |
| R9 | Texto do ghost medido pelo provider × métricas pelo decorador divergem | baixa | médio | `--self-check` e teste de equivalência 100% |
| R10 | Vazamento de dados (trace com valores, `KAPILAB_MONGO_URI` em log) | baixa | alto | trace só com hashes; redaction no logger; testes da §8.3(5); proibição de argv para URI |
| R11 | Concorrência com treino/`llama.cpp` na GPU inflando latência ou travando o driver | média | alto | lock obrigatório (§4.5), `qualified`, jobs via `gpu_queue` |
| R12 | `TreatWarningsAsErrors` e analisadores da IDE atrasam o laboratório; ou `tools\` fora da `.slnx` apodrece | média | baixo | código limpo desde o início; testes do console rodando no `run_eval.ps1`; incluir `KapiLab.Tests` na `.slnx` só quando estável |
| R13 | Esforço do ADR/PR na IDE (Fase 00: IDE não é alterada por este plano) | média | médio | K0 em Rota B (sem tocar a IDE); migrar para A quando o PR for aceito |

## 12. Decisões em aberto

| # | Decisão | Opções | Recomendação [H] |
|---|---|---|---|
| D1 | Aceitar `tools\KapiLab` no repositório da IDE (ADR ≥ 070) | A (PR) · B (HintPath) | A, iniciando em B |
| D2 | `InternalsVisibleTo ...KapiLab` em `OnnxAdapter`/`Application` | pedir · não pedir | não pedir no MVP; reavaliar se B2/B3 doerem |
| D3 | Consolidar `AiRuntimeHarness` da IDE numa biblioteca compartilhada | sim · não | sim, depois do MVP (remove B10) |
| D4 | `kapicoder_manifest.json`/`kapibarastudio-model.json`: validar blocos novos no console | `--strict-kapi` · só IDE | `--strict-kapi` (a IDE ignora blocos desconhecidos [F]) |
| D5 | Tokenizer sem carregar pesos | host pequeno · pedir API à IDE | host pequeno |
| D6 | YAML de gates no console | YamlDotNet · JSON gerado | JSON gerado |
| D7 | Task success do agente no console | executar Mongo em C# · gancho Python | gancho Python (§5.6) |

### Apêndice — comando × fase/gate

| Comando | Fase do plano | Gate/uso |
|---|---|---|
| `contract diff` | 03 | G3.1 |
| `catalog export/check` | 04 | catálogo e `check_catalog_drift` |
| `catalog plans/invoke` | 04 | golden de planos; paridade espelho × registry |
| `model validate` | 07, 08 | §7.6 (leitor equivalente), catálogo da IDE |
| `bench autocomplete/chat` | 06 | §5.3 passo 2, "Gates comuns" |
| `parity` | 06 §6, 07 §7.1 | paridade |
| `guidance smoke` | 04 §5.4, 07 | `SetGuidance` |
| `npu inventory/smoke` | 07 §7.4 | S-NPU-0..3 |
| `agent run/bench` | 04 §10.2, 06 §4.6/§8 T4 | task success, TTFA, estouro |
| `report` | 06 §11 | `gates.json` |
