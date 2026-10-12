# Fase 08 · Integração no Kapibara Studio, homologação e release

**Revisão 2026-10-08b — troca do professor:** o T2 (KapiTeacher-Mongo-14B, ou 7B) é `qwen2` e usa o `QwenCoderModelAdapter` existente — nenhum trabalho de `DeepSeekCoderModelAdapter`/`LlamaKapiCoderModelAdapter` é necessário para ele; o teste Explicit do T2 passa a usar `SLOP_QWEN_MODEL` (`SLOP_DEEPSEEK_MODEL` fica só como regressão do caminho DeepSeek já existente na IDE); pacotes Apache-2.0, sem obrigações da licença DeepSeek (Anexo A); estimativa de esforço inalterada (≈ 8–11 semanas), com um risco novo de VRAM/TTFT do T2 14B.

> Plano KapiCoder-Mongo. Consome os pacotes da [Fase 07](fase-07-exportacao-onnx.md) e os gates da [Fase 06](fase-06-avaliacao-benchmarks.md); contratos de prompt e de ferramentas vêm das Fases [03](fase-03-dataset-autocomplete-validacao.md) e [04](fase-04-dataset-agente-tools.md).
> **Somente proposta:** nada é editado no repositório `F:\source\esilva-silva\EsilvaSoft.KapibaraStudio`. Cada mudança abaixo vira tarefa/ADR do próprio repositório da IDE.
> Legenda: **[fato]** verificado em arquivo/URL · **[proposta]** · **[hipótese]**.

## 1. Objetivo

1. Fazer a IDE descobrir, baixar, selecionar e executar os pacotes KapiCoder-Mongo T0…T4 em CPU, GPU (DirectML) e — se a Fase 07 homologar — NPU Snapdragon (QNN).
2. Dar ao **agente local** (`LocalAgentProvider`) um contrato de conversa com **tool calling** e **modo de raciocínio (thinking)** sobre o `IAgentToolRegistry` existente, para o T4, exposto como **opção do painel "Agente IA"** ao lado dos providers externos (Claude CLI/Claude Code, Copilot), com a mesma experiência de turno — streaming, cartões de tool, aprovações, propostas por hunk — e não só como autocomplete/proposta FIM.
3. Homologar com testes `Explicit` com modelos reais, matriz de hardware e uso interativo; versionar, publicar, permitir rollback.

## 2. Estado atual da IDE relevante (fatos)

| Área | Estado | Fonte |
|---|---|---|
| Runtime | ORT GenAI 0.15.2; builds `WinML` (Windows x64 **e arm64**), `Cpu`, `Cuda` | `Directory.Packages.props`, `release.yml` |
| EPs de NPU | descritores QNN/OpenVINO/VitisAI existem; nenhum build os registra; `docs/26` diz "NPU preparado na seleção e na sonda, mas … nada foi executado em NPU" | `OnnxExecutionProviders.cs`, `docs/26-ia-local-multimodelo.md` |
| Registro de EPs Windows ML | ausente (0 ocorrências de `ExecutionProviderCatalog`) | busca em `src/` |
| Sonda | `OnnxHardwareProbe.Describe` só aceita providers presentes em `GetAvailableProviders()` | `OnnxHardwareProbe.cs` |
| Seleção | Automático NPU → GPU → CPU filtrado pelo `hardware` do metadata; explícito sem fallback | `AiProviderSelector.cs`, ADR-037 |
| Adapters | `qwen2` (Qwen FIM; usado por T0–T3, inclusive o T2 KapiTeacher) e `llama`+`slopcoder_manifest.json` (DeepSeek FIM; legado, não usado pelos pacotes do plano) | `ModelAdapters.cs` |
| Catálogo | exige `model.decoder.filename` | `LocalModelCatalog.cs` |
| Metadata | capacidades `autocomplete`/`chat`/`fim`/`embeddings`; `contextContract` só `editor-context-v1` | `LocalModelMetadataReader.cs`, `LocalModelContextContracts.cs` |
| Requisição | `ModelGenerationRequest` já tem `PromptTokens`, `StopSequences` e `PrefixCache` (inerte) | `ModelGenerationRequest.cs` |
| Agente local | anuncia só `CodeProposals`; prompt `/* Answer the instruction with code … */` em FIM; `SubmitToolResultAsync` lança `NotSupportedException`; máx. 8.192 caracteres de contexto | `LocalAgentProvider.cs` |
| Agente no Release | **ADR-068**: o Desktop Release registra só o GitHub Copilot como `IAgentProvider`; o local é removido da exposição de agentes no Release (continua no DI de Infrastructure e em Debug) | `docs/10-decisoes-arquiteturais.md` (ADR-068), `Infrastructure/ServiceCollectionExtensions.cs` |
| Chat por aba | removido (ADR-055); agente integrado (ADR-056) com `AgentModePolicy`, registry único, propostas revisadas | ADR-055/056 |
| Tools do produto | leitura de metadados, `get_cached_schema`, `get_workspace_context`, `get_query_results`/`get_query_diagnostics`, `propose_file_edit`, `create_workspace_file`; sem escrita/consulta direta ao MongoDB (ADR-066) | `AgentProductToolNames.cs`, ADR-066/067 |
| Download | `HuggingFaceModelSource.DefaultRepositories` fixo: SlopCoder 0.5B e 1.5B-full | `HuggingFaceModelSource.cs`, ADR-039 |
| Testes reais | `Explicit` com `SLOP_QWEN_MODEL`, `SLOP_DEEPSEEK_MODEL`, `SLOP_TEST_GPU=1` | `tests/EsilvaSoft.KapibaraStudio.IntegrationTests/*` |
| Última ADR | ADR-069 (08/10/2026) | `docs/10-decisoes-arquiteturais.md` |
| Providers do painel "Agente IA" | composição em `App.AddDesktopAgentServices`: **Debug** = `local` (vindo da Infrastructure) + OpenAI API + **Claude Code** (`claude-code`, só com `ENABLE_CLAUDE_CODE_PANEL`) + Codex + Copilot; **Release** = `RemoveAll<IAgentProvider>()` e só Copilot. O `.csproj` do Desktop **quebra o build** se `EnableClaudeCodeIntegration=true` em Release ("até aprovação comercial"); ADR-065 moveu Claude/Codex/OpenAI para o backlog. A UI escolhe pelo catálogo (`DesktopAgentProviderCatalog`), sem branch por marca; o mapa de família não tem entrada para `local` (exibe o `DisplayName` "IA local") | `Desktop/App.axaml.cs` (linhas 85–131), `Desktop/EsilvaSoft.KapibaraStudio.Desktop.csproj` (linhas 7–15), ADR-065/068 |
| Contrato `IAgentProvider` | `ProviderId`, `IsLocal`, `Describe()`, `GetStatusAsync`, `CreateSessionAsync`; sessão: `RunTurnAsync` (stream de `AgentProviderEvent`), `SubmitToolResultAsync`, `SubmitApprovalAsync`, `CancelTurnAsync`, `ObservableNativeTools`, `ProviderErrorCodes`, `GetCancellationReport` | `Application/Agents/IAgentProvider.cs` |
| Capacidades | `AgentProviderCapabilities`: `Chat`, `Streaming`, `ToolCalling`, `NativeTools`, `Sessions`, `ModelSelection`, `CodeProposals`, `Mcp`, **`ThinkingSummary`** ("Officially published reasoning summary; never private chain-of-thought"), `UsesNetwork`, **`TurnPlan`** (sem ele o painel **não envia** plano, prompt de sistema nem chips); `FileEditing`/`CommandExecution`/`SubAgents` fixos em `false`; `Evidence` `None`/`AutomatedContract`/`Homologated`; `Normalize()` exige `Chat` para `ToolCalling`/`Sessions`/`TurnPlan`. **Nenhum provider declara `ThinkingSummary = true`** | `Core/Agents/AgentProviderCapabilities.cs`, `AgentCapabilityEvidence.cs`; busca por `ThinkingSummary` em `src/` |
| Raciocínio no runtime e na UI | **não existe**: `AgentEventKind` não tem evento de raciocínio e qualquer `Kind` não tratado encerra o turno com `ProviderProtocolViolation` (`default:` do switch). O Claude Code **descarta** blocos de thinking (texto vazio + assinatura) e o status `thinking_tokens`; o adapter Claude API guarda os blocos só para reenviá-los ao mesmo modelo no ciclo de tools ("nunca vão à UI"); OpenAI/Codex declaram que não oferecem conteúdo de raciocínio. O Desktop tem **0 ocorrências** de thinking/reasoning: a mensagem do agente é `AgentChatMessageItem` (blocos texto/código, `IsStreaming`), a apresentação expõe só `SupportsStreaming`/`SupportsToolCalling`/`SupportsTurnPlan`/`SupportsNativeTools` | `Core/Agents/AgentEvent.cs`, `AgentRuntime.Turn.cs` (switch ~l. 415–452), `ClaudeCodeStreamTranslator.cs` (l. 51, 189, 358), `Anthropic/ClaudeAgentSession.Stream.cs` (l. 242–244), `OpenAiAgentAvailability.cs`, `Desktop/ViewModels/AgentChatItems.cs`, `Desktop/Agents/AgentChatPorts.cs` |
| Persistência da conversa | `AgentConversationEntryKind`: `UserMessage`, `AssistantMessage`, `ToolCall` (nome e desfecho), `Notice`, `EditProposal`; métricas (`AgentUsageMetrics`: input/output/cache/custo) só em memória (ADR-064) | `Core/Agents/AgentConversation.cs`, `AgentUsageMetrics.cs` |
| Grants de turno | `NativeChatTurnPolicyProvider` só gera grants em memória para `ProviderId == GitHubCopilotSubscription`, exige consentimento de **destino externo** e usa `AgentOutputDestination.ProviderExternal` | `Application/Agents/NativeChatTurnPolicyProvider.cs` (l. 52–72) |
| Disponibilidade do local | exige modo ≠ Básico, `ChatEnabled`, modelo do papel `Chat` e capacidades `fim` **e** `chat` — um Qwen3-8B de agente (sem FIM) fica indisponível | `LocalAgentProvider.GetAvailabilityAsync` |
| Geração ONNX | um `Generator` novo por pedido; envia só `max_length`, `do_sample`, `temperature` (sem `top_p`/`top_k`/seed); para em ids e textos de parada; cancela por `terminate_session` | `OnnxLocalModelRuntime.Generate` (l. 271–329) |

Ecossistema (detalhes e URLs na [Fase 07 §2.3](fase-07-exportacao-onnx.md#23-ecossistema-em-2026-10-08-pesquisa-web)): ORT GenAI WinML 0.17.1 (29/09/2026); Windows ML traz CPU + DirectML ("legacy") e baixa QNN sob demanda via `ExecutionProviderCatalog` (Windows 11 24H2+, qualquer edição, X1E/X1P, driver Hexagon ≥ 30.0.140.0 — o X1-26-100 do notebook de homologação não consta da lista; ver §4.3); guidance/tool calling no GenAI (`SetGuidance`). Para o raciocínio: o C# 0.15.2 já tem `ApplyChatTemplate` (sem `enable_thinking`), `GetBorTokenId`/`GetEorTokenId`, `AppendTokens` durante a geração e `RewindTo`; KV quantizado e prefix caching existem só na Engine 0.16/0.17 com KV paginado, sem evidência em DirectML; nenhuma versão traz parser de raciocínio ou de tool call em C# (Fase 07 §2.3 e §5.7).

## 3. Mudanças propostas na IDE

### 3.1 Runtime e backend de NPU

**Decisão recomendada:** não criar um build `Qnn`; usar o build **WinML** existente (já publicado para `win-arm64`) e registrar os EPs do Windows ML em tempo de execução.

| # | Mudança | Arquivos | Detalhe |
|---|---|---|---|
| R1 | Atualizar ORT GenAI 0.15.2 → **0.17.1** (os três pacotes) e `Microsoft.ML.OnnxRuntime.Managed` compatível; regenerar os três lockfiles | `Directory.Packages.props`, `packages*.lock.json` | pré-requisito provável do QNN e do guidance moderno; reexecutar toda a homologação CPU/DML (pacotes v1 e v2) |
| R2 | Serviço `WindowsMlExecutionProviderRegistrar` (Infrastructure.LocalAi.OnnxAdapter, só `WinML`): `ExecutionProviderCatalog.GetDefault().FindAllProviders()` → para QNN: `EnsureReadyAsync()` → `TryRegister()` | novo arquivo + `ServiceCollectionExtensions.cs` | **por ação do usuário** (botão "Instalar aceleração de NPU" em Preferências → IA), nunca no startup; mostrar tamanho/licença do EP; publicar progresso na barra de operações (ADR-036); falha de rede = estado visível |
| R3 | Sonda considera EPs registrados como plugin | `OnnxHardwareProbe.cs` | aceitar descritor presente em `GetEpDevices()` mesmo que ausente de `GetAvailableProviders()` **[hipótese: plugin EPs não aparecem em `GetAvailableProviders` — validar no S-NPU-2]**; recalcular a sonda após registro (hoje é `Lazy` por processo) |
| R4 | Carga com QNN | `OnnxLocalModelRuntime.Load` | `config.AppendProvider("qnn")` (nome GenAI já em `OnnxExecutionProviders`) e opções do pacote (`htp_performance_mode`); confirmar se o GenAI enxerga EPs registrados no `OrtEnv` do processo **[hipótese]** |
| R5 | Opcional: build `Qnn` dedicado (ORT GenAI + `onnxruntime-qnn` empacotado) | `Directory.Build.props`, csproj | **só** se R2/R4 falharem; custo: novo lockfile, pipeline arm64 e redistribuição das DLLs QNN |
| R6 | `ModelDisplayNames.ParseVariant`: aceitar prefixos NPU (`qnn`, `openvino`) e precisões `W4A16`/`W8A16` | `ModelDisplayNames.cs` | título "KapiCoder-Mongo-NPU — NPU QNN W4A16" |

### 3.2 Catálogo e adapters

| # | Mudança | Arquivos | Detalhe |
|---|---|---|---|
| C1 | Aceitar `genai_config.json` com decoder em **pipeline** (vários `filename` em `decoder.pipeline[*]`), validando contenção de cada arquivo e presença dos `.bin` de contexto | `LocalModelCatalog.cs` | hoje lê só `model.decoder.filename` **[fato]**; a forma exata vem da saída do Olive **[hipótese]** |
| C2 | **Obrigatório para o T4:** generalizar `QwenCoderModelAdapter` para `qwen2\|qwen3` (ou `Qwen3ModelAdapter`), pois a [Fase 01](fase-01-selecao-modelos-base.md) escolheu Qwen3-8B (Agent-Lite Qwen3-1.7B/0.6B); sem `LlamaKapiCoderModelAdapter`/DeepSeek adapter: o `T1-ds` saiu do plano e o T2 KapiTeacher é `qwen2` | `IModelAdapter` + `ModelAdapters.CreateDefault` | Qwen2.5-Coder (`qwen2`, T0–T3 **e T2 KapiTeacher 14B/7B**) reutiliza `QwenCoderModelAdapter` sem mudança **[fato]** |
| C3 | Adapter declara formato de **chat/agente** além do FIM: `IChatPromptFormatter` (ChatML do Qwen: `<|im_start|>…<|im_end|>`), tokens de parada do chat e **ids de raciocínio**; para `qwen3` o formatter usa o `chat_template.jinja` do pacote via `Tokenizer.ApplyChatTemplate` (fallback: renderer C# do formato fixo, validado pelos 200 goldens da Fase 07 §7.1) | `IModelAdapter` (membro com implementação padrão nula) | mantém adapters atuais compilando; `<think>`/`</think>` continuam rejeitados no **autocomplete** e passam a ser esperados no agente |
| C4 | Novas capacidades: `LocalModelCapabilities.Agent` (= 16), `Tools` (= 32) e **`Reasoning` (= 64)**; leitor mapeia `"agent"`, `"tools"`, `"reasoning"` | `LocalModelCapabilities.cs`, `LocalModelMetadataReader.cs` | pacotes da Fase 07 já podem declarar (IDEs antigas ignoram) |
| C5 | Novo bloco opcional `agent` (`promptFormat`, `chatTemplateFile`, `toolCallFormat`, `toolSchemaVersion`, `maxToolCallsPerTurn`, `maxToolResultBytes`, `constrainedDecoding`), **`reasoning`** (`supported`, `defaultEnabled`, `toggle`, `disabledGenerationSuffix`, `budgetTokens{default,minimum,maximum}`, `turnBudgetTokens`, `onBudgetExceeded`, `forceCloseText`, `history`, `startTokenId`/`endTokenId`, `templateSha256`), **`generation.agent`** (`maxAnswerTokensPerStep`, `sampling.thinking`/`nonThinking` com `temperature`/`topP`/`topK`) e `runtime.minimumGenAi`; `npu` (`soc`, `staticContextTokens`, `qairt`) — formato em [Fase 07 §6.4](fase-07-exportacao-onnx.md) | `LocalModelMetadata` + leitor | limites validados (orçamento máx. ≤ 8.192, `budget.default` dentro de `[minimum, maximum]`, `templateSha256` confere com o arquivo, senão `Invalid`); `minimumGenAi` maior que o do build ⇒ `Unsupported` com mensagem clara; `npu.soc` incompatível com o dispositivo ⇒ aviso no download e falha clara na carga |
| C6 | Contrato `editor-context-v2` em `LocalModelContextContracts.Supported` **somente** se a Fase 03 mudar o cabeçalho do autocomplete | `LocalModelContextContracts.cs`, `AutocompleteContextBuilder` | lembrar: contrato desconhecido ⇒ pasta `Invalid` nas IDEs antigas **[fato]** |
| C7 | `generation.chat.maxTokens` continua 1–1024; o passo do agente usa `generation.agent.maxAnswerTokensPerStep` (≤ 2.048) **+** orçamento de raciocínio (≤ 4.096) — até ≈ 5,2k tokens por geração | `LocalModelMetadataReader.cs` | não reaproveitar `chat.maxTokens` (mudaria o significado para pacotes v1) |
| C8 | `ModelGenerationRequest` ganha `TopP`, `TopK`, `Seed` e um modo **sessão de geração** (ver A6/A11) que mantém o `Generator`, aceita `AppendTokens` no meio da geração e `RewindTo`; `OnnxLocalModelRuntime` envia `top_p`/`top_k`/`random_seed` | `LocalAi.Core/ModelGenerationRequest.cs`, `OnnxLocalModelRuntime.cs` | hoje só `max_length`/`do_sample`/`temperature` **[fato]**; greedy é desaconselhado no thinking **[fato: card Qwen3-8B]** |

### 3.3 Agente local com tool calling (`LocalAgentProvider`)

Contrato proposto (esboço ilustrativo; o formato **normativo** — uma assinatura JSON por linha em `<tools>`, `\n` ao redor do JSON da chamada, argumentos pelo schema do registry, envelope de erro — é o da [Fase 04 §5.2–5.3](fase-04-dataset-agente-tools.md); aqui o que a IDE precisa implementar):

```text
<|im_start|>system
Você é o agente local do Kapibara Studio … modo: {Agent|Planning|Automatic|AskConfirmations}; permissões: …
# Tools
<tools>
[{"type":"function","function":{"name":"list_collections","description":"…","parameters":{…JSON Schema do registry…}}}, …]
</tools>
Para chamar uma tool responda somente: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
<|im_end|>
<|im_start|>user
{mensagem}{chips autorizados: arquivo ativo, metadados da aba}<|im_end|>
<|im_start|>assistant
<think>
A aba aponta para vendas.pedidos; preciso dos índices antes de sugerir o filtro…
</think>

<tool_call>
{"name": "get_indexes", "arguments": {"connectionId": "3f0c2a9e-6b1d-4c7e-9a52-0d4e1b7f8c21", "database": "vendas", "collection": "pedidos"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"indexes":[…limitado…],"truncated":false}
</tool_response><|im_end|>
<|im_start|>assistant
<think>
…raciocínio do passo seguinte (mantido no turno corrente; removido quando o usuário enviar a próxima mensagem)…
</think>

…resposta final ou propose_file_edit…
```

Com o raciocínio **desligado**, cada passo começa com o bloco vazio `<think>\n\n</think>\n\n` acrescentado pela IDE após a geração do prompt (forma canônica do `enable_thinking=false`, [Fase 07 §5.7.3](fase-07-exportacao-onnx.md)).

| # | Mudança | Detalhe |
|---|---|---|
| A1 | `LocalAgentPromptBuilder` (Application/Agents) | reaproveita `AgentSystemPromptBuilder` (texto < 2 KB, ADR-056) e a lista de tools do `AgentTurnPlan` (`AgentModePolicy`), serializando os JSON Schemas do registry; renderiza com o `chat_template.jinja` do pacote (C3), acrescenta o bloco vazio quando o raciocínio está desligado; histórico de turnos anteriores **sem** `<think>` (só a resposta final, regra do template); tokeniza com o adapter e envia `PromptTokens` (campo já existente) |
| A2 | `LocalReasoningSplitter` + `LocalToolCallParser` | **1º estágio** (máquina de estados por id de token): `InThink` desde o início do passo (o modelo emite 151667) até 151668 → tudo é raciocínio; texto de raciocínio **nunca** é analisado como tool call (um `<tool_call>` dentro do `<think>` é só texto exibido, jamais executado); `</think>` ausente no fim do passo ⇒ raciocínio truncado. **2º estágio** sobre o texto após `</think>`: detecção incremental de `<tool_call>`/`</tool_call>`; JSON estrito (`System.Text.Json`, profundidade e tamanho limitados); nome deve existir no plano do turno; argumentos validados pelo schema do registry; inválido ⇒ 1 retry devolvendo `{"status":"Failed","errorCode":"InvalidToolCallFormat"}` ao modelo (código da Fase 04 §5.3), depois `AgentError` |
| A3 | Eventos | `ToolRequested` → registry (permissões, grants, aprovação, auditoria — **mesmo caminho** dos providers externos, ADR-048/050) → `SubmitToolResultAsync` deixa de lançar e anexa `<tool_response>` com saída já filtrada pela política (ADR-051/066); raciocínio sai por eventos próprios (A10) |
| A4 | Decodificação restrita | quando o metadata declara `constrainedDecoding: lark_grammar`, `GeneratorParams.SetGuidance("lark_grammar", grammar)` com gramática `start: think? (TEXT \| toolcall)`, `think: <[151667]> /(.\|\n)*/ <[151668]>` (livre) e `toolcall` restrito ao JSON de uma das tools do plano — mesmo padrão "bloco livre → JSON" do exemplo da llguidance (https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md); orçamento contado no runtime (A11), **não** por `max_tokens` da gramática (desliga rollback); com raciocínio desligado a gramática começa após o bloco vazio. `<think>`/`<tool_call>` **não** são `special` no Qwen3 **[fato]** ⇒ referência por id **[hipótese: aceita pela llguidance do GenAI; smoke `GuidanceSmoke --reasoning`]**; sem guidance vale o parser estrito |
| A5 | Limites | `maxToolCallsPerTurn` (metadata, padrão 12, orçamento local da Fase 04 §3.5; saída por tool ≤ 6 KiB no T4 / ≤ 2 KiB no Agent-Lite) + `AgentToolInvocationQuota` existente; janela de **16.384** (T4) / 8.192 (Lite): antes de cada passo, `prompt + budget + maxAnswerTokensPerStep ≤ context_length`, senão aplica A11 (raciocínio reduzido/desligado) e, se ainda não couber, encerra com código `LocalContextExhausted` visível (nunca trunca o meio do histórico em silêncio) |
| A6 | Sessão multi-turno e KV | `Sessions = true` em memória. **Dentro do turno:** um `Generator` vivo do primeiro passo à resposta final; cada `<tool_response>` entra por `AppendTokens` — sem re-prefill do sistema/tools/raciocínio (C# 0.15.2 já tem `AppendTokens`/`RewindTo` **[fato]**). **Entre turnos:** `RewindTo` até o fim da última mensagem do usuário anterior e reanexa a resposta final sem `<think>` + a nova mensagem (o template remove o raciocínio ⇒ o prefixo muda a partir dali). O prefix caching da Engine 0.17 (só KV paginado **[fato]**) fica como otimização pós-R1, com teste de equivalência greedy (regra de `onnx-strategy.md`); `PrefixCache` deixa de ser inerte só quando provado |
| A7 | Capacidades anunciadas | `Chat`, `Streaming`, `ToolCalling`, `Sessions`, **`TurnPlan`** (sem ele o painel não envia plano/prompt/chips **[fato]**) e **`Reasoning`** (nova, A10) = true **somente** quando o modelo declara `agent`+`tools` (+`reasoning` para a última) **e** passou nos testes da seção 4, com `Evidence = AutomatedContract` (e `Homologated` após 4.2–4.4); disponibilidade deixa de exigir `fim` quando o pacote é de agente; caso contrário mantém o comportamento atual (`CodeProposals`) |
| A8 | Exposição no Release | hoje o Release registra só Copilot (ADR-068) e o Claude Code é vetado no build Release ⇒ ADR-073 para expor "IA local (KapiCoder)" no painel do agente em Release (seção 3.3.5) |
| A9 | Fronteira | sem rede; `IsLocal = true`; tools nativas (shell/arquivo) continuam ausentes; edição só por `propose_file_edit`/`create_workspace_file` com as regras da ADR-056/067 |
| A10 | Raciocínio como evento neutro | novos `AgentEventKind.ReasoningStarted` / `ReasoningDelta` / `ReasoningCompleted` (este com `ReasoningTokens`, `DurationMs`, `TruncatedByBudget`), correlacionados ao `MessageId` do passo; o runtime os trata como **dados de exibição** (como `MessageDelta`): nunca autorizam tool, nunca entram em auditoria, fila com o mesmo controle de fluxo e `MaxDeltaChars`; provider sem `Reasoning` que emita esses eventos ⇒ `ProviderProtocolViolation` (comportamento atual para `Kind` desconhecido **[fato]**). Nova capacidade `AgentProviderCapabilities.Reasoning` (raciocínio **bruto e local**), distinta de `ThinkingSummary` (resumo oficial publicado), para não violar a definição atual "never private chain-of-thought" **[proposta; ver decisão D1]**. `AgentUsageMetrics` ganha `ReasoningTokens` opcional |
| A11 | Orçamento de raciocínio | contagem em tokens desde 151667: ao atingir o orçamento do passo, o runtime injeta `forceCloseText` (termina em `</think>`) por `AppendTokens` e continua para a resposta/tool call; `ReasoningCompleted.TruncatedByBudget = true`. Orçamento acumulado do turno (`turnBudgetTokens`, padrão 4.096): esgotado, os passos seguintes saem com o bloco vazio e o painel avisa "raciocínio pausado neste turno". `MaximumTokens` do passo = orçamento + `maxAnswerTokensPerStep` + `forceCloseText` |
| A12 | Amostragem | thinking: `generation.agent.sampling.thinking` (T=0,6, top-p 0,95, top-k 20); desligado: `nonThinking`; testes usam `Seed` fixo; greedy só em testes de paridade (C8) |
| A13 | Grants e permissões do provider local | rota de grants de turno para destino **Local** (`AgentOutputDestination` local, sem exigir `HasExternalDestinationConsent`), espelhando as regras de leitura do Copilot no `NativeChatTurnPolicyProvider` (hoje só Copilot **[fato]**); `AgentModePolicy` libera `get_query_results`/`get_query_diagnostics` também para `local` com o opt-in de documentos; tela Permissões do provider "IA local" sem a seção de destino externo |
| A14 | Cancelamento e preempção | "Parar" cancela o passo corrente (`terminate_session`, já usado **[fato]**), fecha `Reasoning*` com `Outcome = Cancelled` e não despacha tool call parcial; autocomplete continua preemptado pela prioridade `Interactive` do turno (ADR-037), e o modelo T4 só carrega quando o painel abre um turno local (carga sob demanda) |

#### 3.3.1 Loop do agente local (um turno)

```mermaid
sequenceDiagram
    participant U as Usuário (painel)
    participant RT as AgentRuntime
    participant LP as LocalAgentProvider
    participant G as Generator ONNX (T4)
    participant RG as Registry/Política
    U->>RT: mensagem + chips + modo (+ raciocínio on/off, orçamento)
    RT->>LP: RunTurnAsync(plano, prompt de sistema)
    LP->>G: prompt (template; bloco vazio se off) — AppendTokens
    loop passo (≤ maxToolCallsPerTurn)
        G-->>LP: tokens <think>…  → ReasoningDelta (UI recolhível, ao vivo)
        alt orçamento do passo atingido
            LP->>G: AppendTokens(forceCloseText …</think>)
        end
        G-->>LP: </think> → ReasoningCompleted
        alt <tool_call>
            LP->>RT: ToolRequested(nome, args)
            RT->>RG: grants/permissões/confirmação (modo) → aprovação humana se exigida
            RG-->>RT: AgentToolResult (Succeeded/Denied/…)
            RT->>LP: SubmitToolResultAsync
            LP->>G: AppendTokens(<tool_response> truncado ≤ 6 KiB)
        else resposta final
            G-->>LP: texto → MessageDelta/MessageCompleted
        end
    end
    LP->>RT: fim do turno; raciocínio descartado do histórico do próximo turno
```

Regras: a mensagem final e os cartões de tool são os **mesmos** componentes usados pelos providers externos (sem branch por marca); propostas `propose_file_edit` seguem ADR-056 (revisão por hunk, Automático só no buffer); `create_workspace_file` sempre com confirmação pontual (ADR-067).

#### 3.3.2 Exibição do raciocínio no painel

**[fato]** não há componente de raciocínio para reutilizar: o painel nunca mostrou thinking de nenhum provider (o Claude Code o descarta). **[proposta]** novo `AgentReasoningItem` (Desktop/ViewModels), reutilizável por qualquer provider que declare `Reasoning` (ou, futuramente, `ThinkingSummary`):

- linha própria **acima** da mensagem/tool call do mesmo passo, criada no `ReasoningStarted`; cabeçalho "Raciocinando…" com contador de tokens e tempo, depois "Raciocínio · 742 tokens · 21 s" (+ "orçamento atingido" quando `TruncatedByBudget`);
- **recolhível**: aberto durante o streaming do passo corrente (ou recolhido, conforme preferência), recolhe sozinho ao chegar a ação/resposta; passos anteriores ficam recolhidos;
- texto em fonte secundária, renderizado como **dados inertes** (mesmo `AgentChatMessageBlock` sem links nem comandos), rótulo "Gerado localmente; pode conter erros; não é a resposta";
- acessibilidade: cabeçalho é botão com estado expandido; leitores de tela anunciam só o resumo, não cada delta; PNGs nos dois temas.

#### 3.3.3 Controles do usuário

| Controle | Onde | Comportamento |
|---|---|---|
| Raciocínio ligado/desligado | chip/alternador no compositor, visível só para provider com `Reasoning`; padrão = `reasoning.defaultEnabled` do pacote (T4 ligado, Lite 1.7B desligado) | vale a partir do próximo passo; persistido em `WorkspacePreferences.AgentPanel` por provider |
| Orçamento | Preferências → IA local → Agente: seletor 256 / 512 / **1.024** / 2.048 / 4.096 (limitado a `budgetTokens.minimum..maximum` do pacote) | padrão 1.024; mostra estimativa de espera (orçamento ÷ tok/s medido) |
| Mostrar raciocínio | Preferências: "expandido durante a geração" / "sempre recolhido" | só apresentação |
| Parar | botão existente do turno | A14 |

#### 3.3.4 Privacidade do raciocínio

- O raciocínio **fica na máquina**: provider `IsLocal = true`, sem rede (A9).
- **Não é persistido** na conversa (`AgentConversationEntryKind` não ganha tipo de raciocínio; só a resposta final, cartões e propostas são salvos, como hoje); após reiniciar, o painel mostra "raciocínio não guardado". Opção **[decisão D3]**: "Guardar raciocínio no histórico" desligada por padrão, sujeita à mesma redação e ao **Apagar histórico**.
- **Não é reenviado**: raciocínio de turnos anteriores sai do prompt (template); trocar de provider na mesma conversa nunca transmite raciocínio local a um provider externo.
- Não entra em auditoria, diagnóstico (`docs/26`), métricas (só `ReasoningTokens`) nem em snapshots/relatórios exportados; testes de canário garantem que segredos citados no raciocínio não aparecem em LiteDB/logs.

#### 3.3.5 Exposição em Release (revisão da ADR-068)

| Opção | Efeito | Observação |
|---|---|---|
| **R-a (recomendada)** | Release registra Copilot **+ `local`** quando houver pacote de agente válido instalado; `local` só aparece no seletor do painel se `Reasoning`/`ToolCalling` estiverem comprovados (A7) | ADR-073 revisa ADR-068; nenhuma conta, rede ou custo; o `RemoveAll<IAgentProvider>()` passa a re-registrar o local |
| R-b | `local` só em Debug até a homologação 4.2–4.4 | sem mudança de ADR agora; usuário final não vê o T4 |
| R-c | `local` + Claude Code em Release | **bloqueado**: o build Release proíbe o Claude Code até aprovação comercial **[fato: `.csproj`]**; "ao lado do Claude CLI" só existe em Debug |

### 3.4 Seleção automática de modelo por hardware

| # | Mudança | Regra proposta |
|---|---|---|
| S1 | `LocalModelRecommender` (Application) | a partir da sonda + tier de memória (ADR-043) sugere, **sem trocar sozinho** a seleção salva: NPU QNN homologado → `KapiCoder-Mongo-NPU-ONNX-qnn-w4a16`; GPU ≥ 12 GB → `Pro-ONNX-dml-int4` (autocomplete) e `Agent-ONNX-dml-int4` (agente); GPU 4–12 GB → `Small-ONNX-dml-fp16`; só CPU ≥ 16 GB RAM → `Small-ONNX-int4`; CPU modesta → `Nano-ONNX-int4` |
| S2 | Papéis separados | autocomplete (`SelectedModel`) e agente (`ChatModel`, que já existe mas sem UI) com modelos distintos; lembrar a regra "um modelo por vez" (ADR-037): trocar de papel descarrega/carrega — medir custo; manter autocomplete pequeno em CPU e agente na GPU exigiria **dois modelos carregados** ⇒ ADR própria |
| S3 | Baixar modelo | `DefaultRepositories` passa a listar `esilva/KapiCoder-Mongo-{Nano,Small,Pro,NPU,Agent}-ONNX` com dicas por variante; variantes `qnn-*` só aparecem quando a sonda vê NPU Qualcomm ou com aviso "NPU não detectada"; manter SlopCoder como legado até a remoção |

### 3.5 ADRs a abrir (numeração a partir de ADR-070)

| ADR | Título | Decide |
|---|---|---|
| ADR-070 | Família KapiCoder-Mongo como modelos locais padrão | repositórios, nomes, convivência/remoção do SlopCoder, `kapicoder_manifest.json` |
| ADR-071 | ORT GenAI 0.17.x e EPs do Windows ML por ação explícita | R1–R5, consentimento do download do EP, comportamento offline |
| ADR-072 | Agente local com tool calling | contrato ChatML/`<tool_call>`, parser, guidance, limites, mesmo registry/política, rota de grants para destino Local (revisa ADR-046/055 quanto ao agente local) |
| ADR-073 | Exposição do agente local no Release | revisa ADR-068 (opções R-a/R-b/R-c de 3.3.5) |
| ADR-076 | Raciocínio do agente no painel | capacidade `Reasoning` × `ThinkingSummary`, eventos `Reasoning*`, orçamento/fechamento forçado, exibição recolhível, não persistência/não reenvio, controles; aplicável a qualquer provider que venha a expor raciocínio |

**Decisões pendentes do usuário** (entram nas ADRs acima):

| # | Decisão | Recomendação |
|---|---|---|
| D1 | Exibir o raciocínio **bruto** do modelo local (contraria a nota atual de `ThinkingSummary`: "never private chain-of-thought") ou só um resumo | exibir bruto, recolhível, com capacidade nova `Reasoning` restrita a `IsLocal = true` |
| D2 | Release: R-a, R-b ou R-c (3.3.5) | R-a após G8.3; R-b até lá |
| D3 | Persistir raciocínio no histórico | não persistir; opção desligada por padrão |
| D4 | Raciocínio dentro do turno (passos de tool) | manter no contexto (regra do template e reuso de KV) com teto acumulado de 4.096 tokens por turno |
| D5 | Modelo único × dois carregados (autocomplete T0–T2 + agente T4) | T4 carregado sob demanda do painel, descarregando o modelo de autocomplete na GPU (ADR-074) |
| ADR-074 | Recomendação de modelo por hardware e papéis autocomplete/agente | S1–S3, eventual segundo modelo carregado (revisa ADR-037) |
| ADR-075 | Métricas locais de IA com consentimento | seção 6.4 |

## 4. Homologação

### 4.1 Testes automatizados regulares (sem pesos)

- Leitor de metadata: `agent`, `tools`, bloco `agent`, `runtime.minimumGenAi`, `npu`, limites e tipos inválidos.
- Catálogo: pipeline QNN válido, `.bin` ausente ⇒ `MissingFiles`, arquivo fora da pasta ⇒ `Invalid`.
- `ModelDisplayNames`: `qnn-w4a16` ⇒ NPU QNN W4A16.
- `LocalToolCallParser`: JSON truncado, tool fora do plano, argumentos fora do schema, `<tool_call>` partido entre chunks, injeção de `<|im_start|>` em saída de tool (rejeitada como hoje `ContainsReservedOrSensitiveText`).
- `LocalAgentProvider` com runtime falso: ciclo tool → resultado → resposta; quota; cancelamento por turno; aprovação negada.
- `LocalReasoningSplitter`: `<think>` partido entre chunks/tokens, `</think>` ausente no fim do passo, `<tool_call>` **dentro** do raciocínio (exibido, nunca executado), texto antes de `<think>`, `<think>` gerado com o raciocínio desligado (descartado + aviso), bloco vazio.
- Orçamento (runtime falso que nunca fecha `</think>`): injeção do `forceCloseText` exatamente no token N, `TruncatedByBudget`, teto acumulado do turno desligando os passos seguintes, `LocalContextExhausted` quando nem sem raciocínio cabe.
- Runtime: `Reasoning*` aceitos só de provider com `Reasoning`; de outro provider ⇒ `ProviderProtocolViolation`; deltas de raciocínio não chegam a auditoria nem ao `AgentConversationEntry`; cancelamento fecha o raciocínio aberto.
- Template: render do prompt do turno N+1 sem raciocínio do turno N (goldens da Fase 07 §7.1); `RewindTo` + reanexo produz os mesmos tokens que um render do zero.
- Desktop: `AgentReasoningItem` (expande/recolhe, contador, "orçamento atingido", restauração após reiniciar sem raciocínio), alternador e orçamento visíveis só com `Reasoning`; PNGs nos dois temas.
- Privacidade: canário escrito no raciocínio não aparece em LiteDB, logs de diagnóstico nem exportações.
- Sonda com EPs de plugin (dispositivos falsos).

### 4.2 Testes `Explicit` com modelos reais

Variáveis (as existentes continuam):

| Variável | Uso |
|---|---|
| `SLOP_QWEN_MODEL` | pacote `qwen2` de autocomplete (T0/T1/T3 **e pacote T2 padrão** KapiTeacher-Mongo-14B/7B `dml-int4`, revisão 2026-10-08b — [Fase 05 §7](fase-05-treinamento.md); rodar uma vez por pacote) — testes existentes `RealModelTestRunsOnTheRequestedHardware(Cpu/Gpu/Auto)`, `RealQwenGeneratesWithCpuAndReusesNativeSession` |
| `SLOP_DEEPSEEK_MODEL` | apenas regressão do caminho DeepSeek já existente na IDE (não faz parte do plano de pacotes; T2 deixou de usá-lo na revisão 2026-10-08b) |
| `SLOP_TEST_GPU=1` | exige provider GPU (não aceita fallback) |
| `KAPI_AGENT_MODEL` **[nova]** | pacote T4: `LocalAgentToolCallingRealModelTests` (20 cenários fixos: listar coleções, índices, schema em cache, propor edição, recusar escrita), cada um com raciocínio **ligado e desligado** e `Seed` fixo |
| `KAPI_AGENT_REASONING=1` **[nova]** | `LocalAgentReasoningRealModelTests`: (1) orçamento 128 com tarefa difícil ⇒ `</think>` forçado e ação válida; (2) desligado ⇒ nenhum 151667 gerado; (3) turno de 3 passos ⇒ KV reaproveitado (sem re-prefill: tempo do passo 2 ≪ TTFT do passo 1) e prompt do turno seguinte sem raciocínio; (4) contexto perto de 16k ⇒ `LocalContextExhausted` ou raciocínio pausado, nunca exceção do runtime; (5) cancelar no meio do raciocínio ⇒ turno `Cancelled`, modelo continua carregado; (6) TTFA e VRAM registrados no TRX |
| `KAPI_AGENT_LITE_MODEL` **[nova]** | mesmos testes no Agent-Lite 1.7B em CPU (raciocínio padrão desligado, orçamento 384) |
| `KAPI_TEST_NPU=1` **[nova]** | exige provider QNN; inclui registro do EP pelo catálogo do Windows ML |

Gate por pacote × hardware: teste de modelo (pasta → tokenizer → sessão → geração) aprovado, provider efetivo igual ao pedido, sem fallback em modo explícito, TRX salvo em `TestResults` com nome `<pacote>-<hw>-<modo>.trx` (convenção do `docs/23`).

### 4.3 Matriz de hardware

| Pacote | Ryzen 9 7900 (CPU) | RX 7800 XT (DirectML) | Vivobook X1-26-100 — CPU Oryon ARM64 | — Adreno X1-45 (DirectML) | — Hexagon (QNN) | NVIDIA (CUDA) |
|---|---|---|---|---|---|---|
| Nano `int4`/`int8` | obrigatório | — | obrigatório | — | — | — |
| Nano `dml-fp16` | — | obrigatório | — | desejável (só relatório) | — | — |
| Small `int4`/`int8` | obrigatório | — | obrigatório | — | — | — |
| Small `dml-fp16`/`dml-int4` | — | obrigatório | — | — (≈ 2,9 GB de fp16 estoura o orçamento de 16 GB compartilhados) | — | — |
| Pro `dml-int4` | — | obrigatório | — | — | — | — |
| NPU `qnn-w4a16` (0.5B e/ou 1.5B) | — | — | — | — | **obrigatório para publicar** | — |
| Agent `dml-int4` (Qwen3-8B; alt. 7B) | — | obrigatório | — (não cabe em 16 GB compartilhados) | — | — | — |
| Agent-Lite 1.7B `int4` / `dml-fp16` | obrigatório | obrigatório | **obrigatório** (`int4`, 8k, processo ≤ 3 GB) | — | — | — |
| Agent-Lite 0.6B `qnn-w4a16` | — | — | — | — | obrigatório se publicado | — |
| `cuda-*` | — | — | — | — | — | obrigatório se publicado |

**Máquina ARM64 de homologação [fato: usuário]:** ASUS Vivobook 16 X1607QA — Snapdragon X **X1-26-100** (8 núcleos Oryon até 2,97 GHz, Adreno X1-45, NPU Hexagon 45 TOPS), **16 GB LPDDR5X compartilhados** entre SO, IDE e NPU, SSD 512 GB, **Windows 11 Home**. Consequências:

- Build da IDE: o `win-arm64` **WinML** existente (sem build `Qnn`); `tools/ide-harness` e `KapiRegistryHarness` também publicados em `win-arm64`. Testes com `KAPI_TEST_NPU=1` rodam nessa máquina.
- Windows ML/QNN EP: exige Windows 11 **24H2 (build 26100)+** e driver Hexagon ≥ 30.0.140.0; **não há exigência de edição** — Home serve **[F: https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers]**. Porém o QNN EP lista só X1E/X1P como requisito e o X1-26-100 **não aparece** → suporte "provável, não oficial", provado no S-NPU-0 da [Fase 07 §7.4](fase-07-exportacao-onnx.md). O EP é baixado do catálogo (Windows Update/Store); no Home não há como adiar updates por política de grupo → reteste a cada atualização do QNN EP (contextos dependem do QAIRT).
- Memória: modelo ≤ 3 GB (teto 4 GB), MongoDB de teste **remoto** ou `mongod` local com `cacheSizeGB: 0.5` (o padrão reservaria ≈ 7,5 GB) — orçamento na [Fase 07 §5.5.1](fase-07-exportacao-onnx.md). T2 (14B/7B)/T4 8B não são homologados nessa máquina.
- X1E/X1P/X2 não estão disponíveis: o card do pacote QNN diz "homologado em Snapdragon X X1-26-100"; outros SoCs ficam "não testado" (S-NPU-8 se surgirem).

Se o S-NPU-0 reprovar (QNN EP não registra no X1-26-100), a coluna QNN fica "não homologado", os pacotes `qnn-*` não entram no **Baixar modelo** (seção 5) e o notebook segue homologando só as colunas CPU ARM64.

### 4.4 Uso interativo (checklist manual)

- [ ] Instalar cada variante pelo **Baixar modelo** (download completo real — a ADR-039 registra que nunca foi homologado), conferir hash, seleção e **Testar modelo**.
- [ ] 30 min de edição real em mongosh/JSON com autocomplete ligado: sem travamento de UI, ghost text coerente, Tab/Escape.
- [ ] Agente local: 20 tarefas do roteiro da Fase 06 em conexão MongoDB real de teste, em cada modo (`Agent`, `Planning`, `Automatic`, `AskConfirmations`), conferindo aprovações e propostas por hunk.
- [ ] Mesmas tarefas lado a lado com o Claude Code (Debug) e o Copilot: mesma sequência de cartões (raciocínio → tool → aprovação → resposta), streaming fluido, painel responsivo durante raciocínio longo, alternar raciocínio/orçamento no meio da conversa, "Parar" durante o `<think>`, reiniciar a IDE e conferir que o raciocínio não foi guardado.
- [ ] Troca de modelo (autocomplete ↔ agente; DML ↔ CPU) sem crash; se o 2º modelo DML derrubar o processo (v1), registrar e bloquear a troca a quente.
- [ ] Vivobook X1-26-100 (Windows 11 Home): bateria vs tomada, primeira carga (instalação do QNN EP pelo botão de R2), offline após instalado, 30 min de autocomplete com a IDE + navegador abertos sem paginação (Gerenciador de Tarefas: memória disponível ≥ 2 GB), troca autocomplete (NPU) ↔ Agent-Lite (CPU) sem dois modelos residentes.
- [ ] Temas claro/escuro, pt-BR/en, PNGs para a documentação.

### 4.5 Critérios de release

| Critério | Valor |
|---|---|
| Qualidade | gates da [Fase 06](fase-06-avaliacao-benchmarks.md) para a variante distribuída; harness da IDE com concordância ≥ 99,5% (Fase 06; os 85/90% da Fase 07 §7.5 são só piso de exportação) |
| Estabilidade | 0 falhas do runtime em 120 exemplos por pacote; 0 crash na sessão interativa |
| Latência | gates por tier da [Fase 06 §8](fase-06-avaliacao-benchmarks.md) medidos pela IDE (os orçamentos da [Fase 07 §7.3](fase-07-exportacao-onnx.md#73-geração-e-desempenho) são o teto técnico da exportação) |
| Agente (T4) | ≥ meta da Fase 06 de tarefas concluídas; 0 chamadas de tool fora do plano executadas; 100% das escritas passaram por aprovação/proposta |
| Raciocínio (T4) | com orçamento 1.024 na RX 7800 XT: **TTFA** (tempo até a 1ª ação: tool call completa ou 1º token da resposta) p50 ≤ 30 s / p95 ≤ 45 s ligado e ≤ 4 s desligado; 1º token de raciocínio visível ≤ 1,5 s com prompt de 4k; **estouro do orçamento ≤ 15%** dos passos e, quando forçado, ação válida após `</think>` ≥ 95%; ganho de sucesso com raciocínio ≥ meta da Fase 06 §8 (senão o padrão vira desligado); 0 vazamentos de raciocínio para LiteDB/logs/provider externo; 0 `<tool_call>` executado a partir do bloco `<think>` |
| Agent-Lite (CPU) | TTFA p50 ≤ 20 s com orçamento 384 no Ryzen; no Vivobook X1-26-100 (CPU ARM64, 16 GB) raciocínio **desligado por padrão**, TTFA p50 ≤ 8 s desligado e processo ≤ 3 GB **[proposta]**; mesmos critérios de estouro e privacidade |
| Segurança | testes adversariais do parser e da saída de tools aprovados; nenhuma rede no provider local |
| Documentação | `docs/23`/`26` (ou sucessores) atualizados com evidência por pacote × hardware; ADRs 070+ aceitas |

## 5. Distribuição, versionamento, rollback e telemetria

### 5.1 Versionamento

- SemVer por **pacote**: `MAJOR` = contrato de prompt/tool ou arquitetura muda (exige IDE nova); `MINOR` = novo treino com mesmo contrato; `PATCH` = reexportação/quantização.
- `version` no `kapibarastudio-model.json` e no `kapicoder_manifest.json`; tag no repo HF `v<versão>`; o download da IDE fixa o `sha` do commit (ADR-039) **[fato]**.
- `runtime.minimumGenAi` e `contextContract` impedem carregar pacote incompatível com mensagem clara.

### 5.2 Changelog

`CHANGELOG.md` em cada repo HF e em `C:\IA\models\KapibaraStudio.MongoIA\reports\release\CHANGELOG.md`: versão, data, dataset, checkpoint, variantes, deltas da Fase 06, hardware homologado, problemas conhecidos.

### 5.3 Rollback

- A IDE nunca sobrescreve pasta instalada (ADR-039) **[fato]** ⇒ versões coexistem lado a lado; rollback = selecionar a pasta anterior.
- Proposta: pasta instalada com sufixo de versão quando o repo publicar mais de uma (`KapiCoder-Mongo-Small-ONNX-dml-fp16-2.1.0`), e o **Baixar modelo** oferecer "versão anterior" a partir das tags.
- No HF, nunca apagar tags; variante retirada = removida da lista de pastas configuradas na IDE (atualização da IDE) e marcada como "retirada" no card.
- SlopCoder v1 permanece disponível até a v2 passar nos critérios de release em todos os hardwares obrigatórios.

### 5.4 Telemetria local só com consentimento

- Hoje: diagnóstico local (`provider.fallback`, `model.ready`), sem envio **[fato: `docs/26`]**; `OnnxHardwareProbe` desliga a telemetria do ORT (`DisableTelemetryEvents`) **[fato]**.
- Proposta (ADR-075): métricas **agregadas e locais** (TTFT p50/p95, tokens/s, taxa de aceitação do ghost text por modelo/provider, falhas por tipo) gravadas no LiteDB, desligadas por padrão, com tela para ver/apagar e exportar um JSON que o usuário pode anexar manualmente a um relato. Nunca texto do editor, prompts, documentos ou nomes de coleções. Nenhum envio automático.

## 6. Atividades (passo a passo)

1. [ ] Abrir ADR-070…076 no repositório da IDE (rascunho a partir deste documento).
2. [ ] Spike R1+R2+R4 numa branch da IDE: GenAI 0.17.1, registro do QNN EP, carregar um pacote QNN público (por exemplo o Qwen2.5-1.5B da receita Olive, ou um modelo QNN do Foundry Local/AI Hub) no Vivobook X1-26-100 (build `win-arm64`, Windows 11 Home) — **valida o caminho antes de investir nos pacotes KapiCoder NPU** (alimenta S-NPU-2 da Fase 07).
3. [ ] Regressão completa CPU/DML com 0.17.1 (pacotes SlopCoder v1 e KapiCoder v2).
4. [ ] C1–C7 (catálogo, metadata, nomes).
5. [ ] A1–A14 (agente local, raciocínio, grants Local) e C8 com testes regulares; UI de raciocínio (3.3.2–3.3.3); depois `Explicit` com `KAPI_AGENT_MODEL`, `KAPI_AGENT_REASONING=1` e `KAPI_AGENT_LITE_MODEL`.
6. [ ] S1–S3 (recomendação e download).
7. [ ] Homologação 4.2–4.4 por pacote × hardware; evidências em `docs/` da IDE.
8. [ ] Promover repos HF de `-preview` para definitivos; publicar changelog; release da IDE com a lista nova.

## 7. Artefatos

- [ ] ADRs 070–076 aceitas (incluindo decisões D1–D5)
- [ ] PR(s) da IDE com R1–R6, C1–C8, A1–A14, S1–S3, UI de raciocínio e testes
- [ ] TRX `Explicit` por pacote × hardware + relatório `reports/release/homologacao.md`
- [ ] Checklist interativo preenchido com PNGs
- [ ] `CHANGELOG.md` e cards finais nos repos `esilva/KapiCoder-Mongo-*-ONNX`

## 8. Critérios de saída (gates)

| Gate | Critério |
|---|---|
| G8.1 | GenAI atualizado sem regressão: todos os testes regulares + `Explicit` CPU/DML dos pacotes v1 e v2 aprovados |
| G8.2 | T0/T1/T2 homologados em CPU e RX 7800 XT e listados no **Baixar modelo** |
| G8.3 | T4: agente local com tool calling **e raciocínio** cumpre 4.5 (linhas Agente, Raciocínio e Agent-Lite) em `dml-int4` (Qwen3-8B, 16k) e na variante Agent-Lite (Qwen3-1.7B) em CPU; o fallback 7B-Instruct só entra **sem** `Reasoning` anunciado |
| G8.7 | Painel "Agente IA": provider `local` selecionável ao lado dos demais na composição decidida em D2, com `Chat`/`Streaming`/`ToolCalling`/`TurnPlan`/`Reasoning` comprovados (A7) e ADR-072/073/076 aceitas |
| G8.4 | NPU: S-NPU-0, S-NPU-2..6 + `KAPI_TEST_NPU=1` aprovados no Vivobook X1-26-100 (16 GB, Windows 11 Home), dentro do orçamento de memória da Fase 07 §5.5.1 **ou** decisão registrada de "NPU fora da v2" com pacotes não listados |
| G8.5 | Download completo real homologado pela janela (pendência antiga da ADR-039) |
| G8.6 | Rollback testado (instalar 2 versões, alternar, remover) |

## 9. Riscos e mitigação

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Upgrade 0.15.2 → 0.17.1 quebra pacotes v1 ou DML | média | regressão do que já funciona | regressão completa (passo 3); manter 0.15.2 se só o NPU precisar e adiar NPU |
| GenAI não enxerga EPs registrados pelo Windows ML no mesmo processo | média | NPU inviável via WinML | R5 (build `Qnn` dedicado) ou adiar NPU |
| EP QNN baixado pela Store/Windows Update muda de versão e invalida contextos binários | média | NPU para de carregar após update | metadata com `qairt`; falha clara + fallback por **Automático** para o pacote CPU do mesmo tier; reteste a cada KB do QNN EP |
| X2 Elite não listado nos requisitos do QNN EP do Windows ML | alta | usuários X2 sem NPU | pacote `qnn-w4a16-x2` só com evidência; senão GPU Adreno/CPU |
| T2 14B `dml-int4` (~9 GB + KV) não convive com o autocomplete/agente carregados, ou TTFT acima do gate da Fase 06 | média | T2 14B indisponível ou lento em 16 GB | "um modelo por vez" (ADR-074); `recommendedContextTokens` 4096; cair para o KapiTeacher 7B (~5 GB) sem mudar adapter nem contrato |
| Agente local com 7B erra tool calls ou ignora permissões | média | experiência ruim / risco | guidance (A4), parser estrito, mesma política/registry dos providers externos, aprovação humana, anúncio de capacidade só com evidência (A7) |
| ADR-068 mantém Release só com Copilot | média | agente local invisível ao usuário final | ADR-073 explícita; até lá, agente local apenas em Debug/preview |
| "Um modelo por vez" × autocomplete e agente com modelos diferentes | alta | recarga a cada troca (segundos) | medir; ADR-074 decide entre modelo único T4 para ambos ou dois modelos carregados com limite de memória |
| Janela de contexto do agente (sistema + tools + histórico + raciocínio dos passos do turno) estoura 16k | média | erros de contexto | teto acumulado de raciocínio por turno (A11), checagem antes de cada passo (A5), tools só as do plano, saída de tool ≤ 6 KiB |
| Raciocínio deixa o agente local lento demais frente ao Claude CLI/Copilot | média | usuário desliga ou abandona o provider | orçamento padrão 1.024 com fechamento forçado, raciocínio conciso no treino, contador/estimativa visíveis, desligar com 1 clique; gate de TTFA em 4.5 |
| Exibir raciocínio bruto conflita com a política atual (`ThinkingSummary` "never private chain-of-thought") | média | ADR rejeitada, UI refeita | decisão D1 antes da UI; capacidade separada `Reasoning` só para `IsLocal` |
| Raciocínio cita segredos/valores lidos por tools e acaba persistido ou reenviado | baixa | vazamento local/externo | não persistir (D3), não reenviar, canários (4.1), troca de provider não transmite raciocínio |
| `ApplyChatTemplate` nativo diverge do template Jinja do Qwen3 | média | prompt fora do treino | goldens de renderização (Fase 07 §7.1); renderer C# do formato fixo como fallback |
| `Generator` vivo por turno segura 16k de KV e bloqueia o autocomplete | média | autocomplete parado durante o turno | prioridade `Interactive` já preempta o autocomplete (ADR-037); liberar o `Generator` no fim do turno e após inatividade; medir em 4.4 |
| X1-26-100 fora da lista oficial do QNN EP (só X1E/X1P) | média | EP não registra/NPU invisível na única máquina ARM64 | S-NPU-0 logo na semana 1 (antes de R1–R4 na IDE); se reprovar, G8.4 pela via "fora da v2" e o notebook homologa só CPU ARM64 |
| 16 GB compartilhados no notebook de homologação | alta | paginação/TTFT instável com IDE + navegador + modelo | um modelo por vez (ADR-074), MongoDB remoto ou `cacheSizeGB: 0.5`, `LocalModelRecommender` (S1) só recomenda NPU/Lite dentro do orçamento da Fase 07 §5.5.1 |

## 10. Dependências

- [Fase 07](fase-07-exportacao-onnx.md): pacotes e relatórios; S-NPU-1..8.
- [Fase 04](fase-04-dataset-agente-tools.md): formato exato de tool call e lista/versão das tools (deve espelhar `AgentProductToolNames` e os schemas do registry na data do treino).
- [Fase 06](fase-06-avaliacao-benchmarks.md): metas de qualidade e roteiro do agente.
- Repositório da IDE: decisões do usuário nas ADRs 070–076 (especialmente ADR-073, que revisa a ADR-068, e ADR-076, raciocínio) e D1–D5 (seção 3.5).

## 11. Estimativa

| Bloco | Esforço |
|---|---|
| ADRs e spike NPU na IDE (R1, R2, R4) | 4–6 dias (no Vivobook X1-26-100, já disponível) |
| Regressão pós-upgrade | 2–3 dias |
| Catálogo/metadata/nomes (C1–C8, R6) | 4–5 dias |
| Agente local com tool calling (A1–A9, A13) + testes | 8–12 dias |
| Raciocínio: eventos/capacidade no runtime (A10), splitter, orçamento e fechamento forçado (A11), amostragem (A12), `Generator` por turno com `RewindTo` (A6), cancelamento (A14) + testes | 5–7 dias |
| UI de raciocínio e controles no painel (3.3.2–3.3.4), privacidade, PNGs | 3–4 dias |
| ADR-073/076 (exposição em Release e raciocínio) | 1 dia |
| Recomendação e download (S1–S3) | 3–4 dias |
| Homologação (4.2–4.4) por pacote × hardware (inclui raciocínio ligado/desligado e comparação com Claude Code/Copilot) | 6–9 dias |
| Publicação, changelog, rollback | 2 dias |
| **Total** | **≈ 8–11 semanas** de uma pessoa (38–53 dias; antes 5–8 semanas — +11–14 dias pelo raciocínio, pela UI e pela exposição no painel); o NPU pode ser cortado sem bloquear T0/T1/T2/T4 |
