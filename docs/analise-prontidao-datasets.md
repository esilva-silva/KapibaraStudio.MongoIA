# Prontidão para gerar os datasets — análise do Kapibara Studio (2026-10-10)

Checkout analisado (somente leitura): `F:\source\esilva-silva\EsilvaSoft.KapibaraStudio`, HEAD `0dbb4dd` ("Add KapiLab CLI, tooling, tests, and docs"). Desde a análise anterior (`dba8546`) entraram 3 commits: Fase 6 (admin/import-export), **`d91d6c4` agente local com Qwen3, tool calling e raciocínio** e **`0dbb4dd` KapiLab** (134 arquivos). Fontes de estado: `docs/phases/phase-07a-kapicoder-mongo/rastreabilidade-f7a.md` (78 itens, "implementação parcial, sem homologação integral; VSTest bloqueado pela conexão com testhost") e `docs/phases/phase-07b-kapilab/rastreabilidade-f7b.md` (F7B-00…24, K0/K1 parciais).

Legenda: ✅ pronto para uso no dataset · 🟡 existe, mas falta algo para servir ao dataset · ❌ falta.

## 1. Resumo

| Dataset | Pode começar? | Bloqueio principal |
|---|---|---|
| **Autocomplete FIM (T0/T1/T3, professor/T2)** — [Fase 03](fase-03-dataset-autocomplete-validacao.md) | **Sim**, com o renderer Python da v1 (`compact/contracts.py`) — o contrato `editor-context-v1` não mudou | Falta a **prova de paridade em 2.000 registros** (F7B-05) e o **export do catálogo de conhecimento** do editor para os geradores Python |
| **Chat curto (T1/T2, contrato FIM `local-agent-fim-v0`)** | **Decidir antes** | Com o agente Qwen3 na IDE, o chat por FIM só roda para pacotes **sem** metadata de agente. Precisa decidir se T1/T2 ainda treinam chat ou só autocomplete |
| **Agente com tools e raciocínio (T4/Agent-Lite)** — [Fase 04](fase-04-dataset-agente-tools.md) | **Não** (só o simulador e os geradores podem começar) | O contrato de prompt do agente ainda não está congelado nem exportável (A1/PC1/PC2 "Planejado"); faltam descrições reais das tools e a execução real das tools com grants em fixture (`catalog invoke` cobre só 2 casos, sem grants e sem MongoDB) |

## 2. O que já está pronto na IDE

| Necessidade do dataset | Estado | Onde |
|---|---|---|
| Contrato `editor-context-v1` (FIM) real | ✅ | `kapilab contract render --contract editor-context-v1` usa `AutocompleteContextBuilder`/`ModelPrefix` reais (saída só com **hashes**, sem texto) |
| Tokenização pelo adapter real da IDE | ✅ | `kapilab contract tokenize --package` (catálogo + `IModelAdapter` + `ITokenizer` + prompt builder reais) e `contract diff` (código 11 em divergência) |
| Captura do chat legado `local-agent-fim-v0` | ✅ | `contract render --contract local-agent-fim-v0` grava o `ModelGenerationRequest` real (hashes) |
| Catálogo de tools exportável | 🟡 | `kapilab catalog export --provider local --surface in-process` → `kapilab-agent-catalog-v1`, `complete=false`, sem `approve`; `catalog check` detecta drift (código 7) |
| Decisão de tools do agente local (D6) | ✅ | ADR-070: **todas as tools vigentes** disponíveis ao local, filtradas por modo/permissões/grants/confirmações |
| Agente local Qwen3 com tool calling e raciocínio | 🟡 | `LocalAgentProvider` (676 linhas): `Chat/Streaming/ToolCalling/TurnPlan`, parser `qwen-hermes` incremental (`LocalAgentToolCallParser`), orçamento por passo/turno, `ForceCloseText` vindo do metadata, retry único em `InvalidToolCallFormat` |
| Adapter `qwen3` e metadata de agente/raciocínio | 🟡 | `Qwen3ModelAdapter` (sem FIM), blocos `agent`/`reasoning`/`generation.agent`/`runtime.minimumGenAi` — "compilado; smoke GenAI pendente" (C2/C5) |
| Gerador de amostras sintéticas | 🟡 | `kapilab samples generate` usa os catálogos reais do editor, mas o README diz: **"Essas amostras não constituem dataset de treinamento"** (máx. 10.000/lote, `execution=skipped`, sem MongoDB) |
| Validação de pacotes e gates | ✅ | `model validate --strict-kapi --verify-hashes`, `report evaluate/merge` (`kapilab-gates-v1`) |
| Bench pelo caminho real da IDE | 🟡 | `bench autocomplete/chat` reais; só smoke CPU com pesos reais; GPU sem prova |

## 3. O que falta para gerar os datasets

### 3.1 Bloqueadores na IDE (precisam de PR no Kapibara Studio)

| # | Falta | Por que bloqueia | Item de origem | Estado no checkout |
|---|---|---|---|---|
| I1 | **Contrato de prompt do agente congelado e exportável** — `LocalAgentPromptBuilder` (A1), renderer de fallback do template Qwen3 + **200 goldens** (PC2), captura do prompt real (PC1) | O dataset do T4 precisa ser renderizado **byte a byte** como o runtime monta o prompt (system, `<tools>`, histórico sem `<think>`, `tool` messages). Hoje isso está espalhado em `LocalAgentProvider` (`BuildMessages`, `BuildSystemPrompt`, `BuildUserContent`, `BuildToolsJson`) | A1, PC1, PC2, TA5 | **Planejado** |
| I2 | **`kapilab contract render --contract local-agent-qwen3`** (render/tokenize/diff do turno do agente com tools e raciocínio ligado/desligado) | Sem ele não há paridade Python × IDE para o dataset de agente. Hoje o KapiLab só renderiza `editor-context-v1` e `local-agent-fim-v0` | F7B-04/18 | Pendente |
| I3 | **Descrições reais das tools no prompt** | `BuildToolsJson` usa `["description"] = tool.Name`: o modelo vê só o nome. Se o dataset usar descrições ricas e a IDE não, há divergência de contrato; se usar só o nome, o modelo perde semântica. Decidir e congelar | A1 / K1 | Código atual: descrição = nome |
| I4 | **Catálogo canônico completo** (`complete=true`) com schemas de entrada **e saída** e descrições, por provider/modo | O gerador de trajetórias e o simulador `kapisim` precisam dos schemas de saída reais para fabricar `tool` responses fiéis | K1–K4, F7B-06 | K1–K4 "Planejado"; export atual `complete=false` |
| I5 | **Execução real das tools com grants em fixture** (`catalog invoke` ampliado: todas as tools, grants positivos, leitura de resultados/diagnósticos capturados, truncamento ≤ `MaxToolResultBytes`) | As respostas `{"status","result","error","truncated"}` do dataset devem sair do registry real, não de um mock. Hoje: só `list_connections` e `list_databases`, `grantsSupported=false`, `mongoDocumentsAccessed=false` | F7B-17/20, A13a–d | Parcial |
| I6 | **Export do catálogo de conhecimento do editor** (operadores, estágios, métodos do console, tipos EJSON, Atlas Search) em JSON versionado | Os geradores Python das Fases 03/04 não podem inventar APIs; o `samples generate` usa esses catálogos internamente mas não os exporta e não serve como dataset | F7B-11 (K2) | Não existe comando |
| I7 | **Smoke do template/tokenizer Qwen3 com pesos reais** (R7, C3 `ApplyChatTemplate`) | Confirma que o template do pacote (`chat_template.jinja`) renderiza igual ao HF; sem isso, os goldens podem divergir do runtime | R7, C3, C8 | Planejado / parcial |
| I8 | **Suíte de testes executável** | VSTest bloqueado pela conexão com o testhost: as evidências de contrato não podem ser reexecutadas | E1–E4 | Bloqueado |

### 3.2 Decisões pendentes (do usuário)

| # | Decisão | Impacto no dataset |
|---|---|---|
| Q1 | **T1/T2 ainda treinam o chat curto `local-agent-fim-v0`?** O painel de agente agora usa Qwen3 com tools; o caminho FIM só vale para pacotes sem metadata de agente | Se não, a Fase 03 vira **só autocomplete** (≈ −20% de volume e de horas de GPU no T1/T2). Se sim, mantém os 20% de chat no contrato FIM |
| Q2 | **Descrição das tools no prompt**: nome (como hoje) ou descrição curta do produto (I3) | Define o texto de `<tools>` em todas as trajetórias |
| Q3 | **`ForceCloseText`**: texto exato do fechamento forçado do raciocínio (vem do metadata do pacote) | Precisa ser fixado antes de gerar as amostras de orçamento estourado |
| Q4 | `kapi-context-v2` (C6): manter `editor-context-v1` | A IDE mantém v1 como legado e v2 só se a avaliação justificar → **recomendo congelar v1** para a v2 dos modelos |

### 3.3 Fora da IDE (workspace de treino — não depende de PR)

| # | Falta | Fase |
|---|---|---|
| W1 | Fixtures dos domínios sintéticos no **MongoDB 8.0.32 do WSL** (`127.0.0.1:27017`, bancos `kapi_fx_*`) para **exec-valid**; `$search` só validação estática (Community não tem `mongot`) | 03/04 |
| W2 | Geradores Python (FIM com cortes por AST, contexto de editor, schema, chat/negativos) consumindo o export do catálogo (I6) | 03 |
| W3 | Pipeline de validação de 9 etapas (sanitização → sintaxe → API → campos → execução → semântica → orçamento → dedup → decontaminação) | 03 |
| W4 | Conjunto **cego** escrito à mão (400 + 360 do agente) — o KapiLab já recusa o caminho do blind | 03/04 |
| W5 | `kapisim` (cópia Python das tools) validado contra o registry real com ≥ 99% em 500 chamadas — depende de I4/I5 | 04 |
| W6 | Renderer Python do turno do agente idêntico ao I1/I2 (paridade por hash via `contract diff`) | 04 |
| W7 | Destilação pelo professor (KapiTeacher-Mongo-7B) — depende do SFT do professor (Fase 02) | 02/03 |

## 4. Ordem sugerida

1. **Agora, sem esperar a IDE:** W1 (fixtures no MongoDB do WSL), W2/W3 para **autocomplete** usando o renderer da v1 (o contrato `editor-context-v1` não mudou), W4 (blind). Rodar `contract render`+`tokenize` em uma amostra para conferir os hashes.
2. **PR 1 na IDE (pequeno, destrava a Fase 03):** I6 export do catálogo de conhecimento + a prova de paridade de 2.000 registros (F7B-05). Decidir Q1 e Q4.
3. **PR 2 na IDE (destrava a Fase 04):** I1 + I2 + I3 (contrato do agente congelado, render/tokenize no KapiLab, goldens PC2), decidir Q2/Q3, I7 smoke Qwen3 com pesos reais.
4. **PR 3 na IDE:** I4 + I5 (catálogo completo e invocação real com grants em fixture), então W5/W6 e geração das trajetórias.
5. Em paralelo: I8 (destravar o VSTest) — sem ele nenhuma evidência de contrato é reexecutável.
