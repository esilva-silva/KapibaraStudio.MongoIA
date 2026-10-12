# PR 3 — Catálogo completo de tools (com schemas de saída) e execução real com permissões em fixture

> Especificação para outra IA implementar no repositório do Kapibara Studio. É autocontida: leia-a inteira antes de começar.
> Escrita em 2026-10-10 sobre o checkout `F:\source\esilva-silva\EsilvaSoft.KapibaraStudio`, HEAD `0dbb4dd`.
> Objetivo de negócio: permitir que o pipeline de treino gere **respostas de tools reais** (não simuladas) para as trajetórias do agente T4 (Fase 04 do plano em `C:\IA\models\KapibaraStudio.MongoIA\docs\`) e valide o simulador Python `kapisim` contra o registry real (meta: ≥ 99% de concordância em 500 chamadas).
> Depende de: **PR 2** (`pr-2-contrato-prompt-agente.md`) para a descrição das tools e o formato de `ToolResponse` congelados. Pode começar a parte de catálogo (§3.1) em paralelo.

## 1. Contexto

O registry de tools (`src/EsilvaSoft.KapibaraStudio.Application/Agents/AgentToolRegistry*.cs`, interface `IAgentToolRegistry`) expõe por provider: `GetInProcessDescriptors(providerId)`, `GetInProcessInputSchemaJson(providerId, name)`, `GetInProcessOutputSchemaJson(providerId, name)` e `InvokeAsync(...)`. A ADR-070 aprovou a decisão **D6**: o provider `local` recebe **todas as tools vigentes do produto**, filtradas por modo, permissões, grants e confirmações; **nenhuma consulta ou escrita direta em MongoDB** pelo agente (ADR-066: o agente propõe a edição, a pessoa executa, o agente lê a saída capturada).

Tools vigentes (constantes em `AgentToolRegistry.cs`; conferir quais têm descriptor/handler hoje): `list_connections`, `list_databases`, `list_collections`, `get_indexes`, `get_search_indexes`, `get_cached_schema`, `get_workspace_context`, `get_query_results`, `get_query_diagnostics`, `create_workspace_file`, `propose_file_edit` (+ `approve`, interna). Nomes **removidos** que ainda existem como constantes e não podem voltar: `mongo_find`, `mongo_count`, `sample_documents`, `mongo_find_one`, `get_document`, `mongo_distinct`, `mongo_explain`, `get_collection_schema` e as escritas de `AgentToolRegistry.LegacyNames.cs`.

Estado do KapiLab (`tools/KapiLab`, ver `tools/KapiLab/README.md`):

- `catalog export --provider <id> --surface in-process|session-channel` → `kapilab-agent-catalog-v1`, `scope=provider-maximum`, **`complete=false`**, sem `approve`;
- `catalog export-plan` (plano sintético), `catalog check` (drift → código 7), `catalog plans` (política pura);
- `catalog invoke` executa casos pelo `IAgentRuntime` e registry reais, mas a allowlist só tem **2 cenários** (`list_connections` permitido; `list_databases` negado por `PermissionMissing`), com `grantsSupported=false`, `mongoDocumentsAccessed=false`, profile sintética `127.0.0.1:1`;
- `agent replay` mantém só o caso v1 `list_connections`.

Itens da rastreabilidade: **K1–K4** e **A13a–d** (`docs/phases/phase-07a-kapicoder-mongo/rastreabilidade-f7a.md`), **F7B-06, F7B-17, F7B-20** (`docs/phases/phase-07b-kapilab/rastreabilidade-f7b.md`).

## 2. Regras do repositório (obrigatórias)

- Leia `CLAUDE.md`, `AGENTS.md` da raiz e de cada diretório antes de editá-lo; docs em **pt-BR**.
- **Segurança é o requisito principal**: nenhum caminho novo pode conceder grants fora do laboratório, ler o LiteDB/cofre do usuário, relaxar a política de produção ou executar query/escrita MongoDB por tool. Grants do laboratório existem **só** no host do KapiLab, com LiteDB temporário, e ficam marcados como sintéticos nos relatórios.
- MongoDB de fixture: o MongoDB Community **8.0.32 no WSL**, acessível em `mongodb://127.0.0.1:27017` (o KapiLab já aceita `KAPILAB_MONGO_URI` apenas `mongodb://` com host loopback explícito). Usar **somente** bancos com prefixo `kapi_sim_`; nunca tocar em `sample_mflix` ou outros bancos. Sem `$search` (o Community não tem `mongot`): `get_search_indexes` deve devolver o resultado/erro real que a IDE devolve nesse servidor.
- KapiLab: build/teste por `./scripts/lab/build_kapilab.ps1 -Backend Cpu -Test` e `-Backend WinML`; lockfiles por backend; saídas só em `data/lab`, `reports/lab`, `tmp`.
- Não mudar o comportamento de produção do agente, exceto correções de bug que este PR revelar (cada uma com teste e nota na rastreabilidade).

## 3. Entregáveis

### 3.1 Catálogo completo e canônico (K1–K4, F7B-06)

- `catalog export --provider local --surface in-process --mode <modo>` passa a produzir **`complete=true`** quando: todas as tools expostas ao provider no modo têm descriptor, **descrição** (a mesma do prompt, decidida no PR 2), **input schema** e **output schema** versionados; `approve` continua omitida e explicada.
- Para cada tool: `name`, `description`, `inputSchema`, `outputSchema`, `inputSchemaSha256`, `outputSchemaSha256`, `confirmationCategory` (`AgentProductToolNames.CategoryOf`), `outputScope` (`AgentToolOutputScopes`), quotas/limites (`AgentToolInvocationQuota`), `maxResultBytes` aplicado ao local, modos em que aparece, e `errorCodes` possíveis.
- Ferramentas sem output schema hoje: **criar o schema** a partir do tipo real de retorno (sem mudar o retorno), com teste que valida N saídas reais contra o schema.
- Garantir que nenhum nome removido (lista na §1) aparece em nenhum modo/provider; teste explícito.
- `catalog check` continua retornando 7 em drift e passa a comparar também descrições e output schemas.

### 3.2 Host de fixture com grants de laboratório (A13a–d, F7B-17)

- Novo modo do KapiLab, p.ex. `catalog invoke --fixture <fixture.json>` (ou comando `agent tools run`), que compõe `AddKapibaraStudioInfrastructure` com **LiteDB temporário** e:
  - uma `ConnectionProfile` de laboratório apontando para `KAPILAB_MONGO_URI` (loopback);
  - **grants sintéticos** concedidos pelo próprio host conforme o caso (metadata, leitura de documentos capturados, criação de arquivo), marcados `grantsSynthetic=true`;
  - um **estado de IDE sintético**: aba ativa (conexão › banco › coleção), arquivo do workspace num diretório temporário, **resultados e diagnósticos de query capturados** (para `get_query_results`/`get_query_diagnostics`) produzidos executando a query do caso pelo **mesmo serviço que a IDE usa ao executar uma aba** (não pelo agente), e o cache de schema (`get_cached_schema`) preenchido pelo mesmo serviço de inferência da IDE.
- Executa as chamadas pelo **mesmo caminho de produção**: `IAgentRuntime` → registry → política/grants/aprovação/auditoria → handler → `LocalAgentPromptBuilder.BuildToolResponse` (PR 2) com `MaxToolResultBytes` do pacote.
- Casos cobrindo **todas** as tools vigentes: sucesso, permissão negada, grant expirado/revogado, aprovação negada, argumento inválido (schema), quota excedida, resultado truncado, conexão inexistente, banco/coleção inexistente, `create_workspace_file`/`propose_file_edit` gerando **proposta** (nunca escrita direta no disco do usuário).
- Saída `kapilab-tool-invocation-report-v2`: por caso, `tool`, `arguments`, `status`, `errorCode`, decisão de política/auditoria, e o **`toolResponse` exato** (string que entraria na mensagem `tool` do prompt) — permitido porque os dados vêm só do fixture sintético `kapi_sim_*`; recusar se a URI/banco não for de laboratório.

### 3.3 Fixture versionado e semeadura

- `tools/KapiLab/Fixtures/kapi-sim/`:
  - `domains/*.json` — 3–6 domínios sintéticos pequenos (p.ex. loja, logística, IoT) com coleções, documentos fictícios (sem PII real), índices;
  - `seed.ps1`/`seed.py` (ou comando `kapilab fixture seed --uri …`) que **recria** os bancos `kapi_sim_<domínio>` (drop + insert + createIndex) de forma determinística, com hash do conteúdo;
  - `cases/*.json` — os casos de invocação da §3.2 (≥ 200 casos; ≥ 1 por tool × resultado).
- O pipeline de treino terá dezenas de domínios próprios; este fixture é o **contrato de referência** para provar o caminho.

### 3.4 Replay e paridade com o simulador Python (F7B-20)

- `agent replay` aceita trajetórias `kapilab-agent-trajectory-v1` (sequência de tool calls com respostas esperadas) e reexecuta cada call no host de fixture, comparando `toolResponse` (exato, ou por campos com tolerância documentada para campos voláteis como datas/ids gerados).
- Script `scripts/lab/parity_kapisim.py` que roda **500 chamadas** dos casos contra o host real e contra um simulador Python (o `kapisim` fica no workspace de treino; o script recebe o caminho do simulador como argumento) e produz a taxa de concordância. Meta: **≥ 99%**, e **0 casos** em que o simulador permite algo que a IDE nega (violação de permissão nunca é tolerada pelo percentual).

## 4. Critérios de aceite

- [ ] `catalog export --provider local` com `complete=true` em cada modo, todas as tools com descrição + input/output schema + hashes; nenhum nome removido exposto (teste).
- [ ] Output schemas validados contra saídas reais (≥ 20 por tool) no fixture.
- [ ] Host de fixture com grants sintéticos e estado de IDE sintético; ≥ 200 casos cobrindo todas as tools e todos os resultados da §3.2; relatório com `toolResponse` exato.
- [ ] Nenhuma escrita em MongoDB pelas tools; nenhum banco fora de `kapi_sim_*` acessado (teste que aponta para outro banco e espera recusa).
- [ ] `agent replay` com trajetórias v1; `parity_kapisim.py` documentado (a meta de 99% é verificada quando o simulador existir).
- [ ] README do KapiLab, runbook, rastreabilidade 7A (K1–K4, A13a–d) e 7B (F7B-06/17/20) e evidências atualizados.

## 5. Fora de escopo

- Gerar o dataset de treino (é do pipeline externo).
- Escrever o `kapisim` Python (é do pipeline externo; este PR só fornece o contrato e o comparador).
- Ferramentas de escrita MongoDB, execução de query pelo agente, Atlas Search executável.

## 6. Riscos

- `get_query_results`/`get_query_diagnostics` dependem do estado de execução da aba: se o serviço de execução da IDE exigir UI/Desktop, extraia a parte de domínio para uma API pública testável antes de usá-la no host (refatoração aditiva, com teste de não-regressão).
- O `catalog invoke` atual afirma `grantsSupported=false`; ao introduzir grants de laboratório, garanta que o código de grant **só** é alcançável pelo host do KapiLab (teste que o Desktop não consegue registrar o provedor de grants sintéticos).
- VSTest bloqueado pelo testhost neste ambiente: não marcar aceite sem execução comprovada.
