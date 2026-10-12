# Briefing compartilhado dos agentes de planejamento

> **Registro histórico (premissas iniciais).** Foi superado em partes: professor → Qwen2.5-Coder 14B/7B e pastas `BaseRepos`/`Repos` ([_briefing-troca-professor.md](_briefing-troca-professor.md)), treino 100% local, T4 com raciocínio e NPU no Vivobook X1-26-100 ([README](README.md)).

> Arquivo de trabalho do orquestrador. Todos os agentes leem este briefing antes de escrever sua fase.

## Objetivo do programa

Treinar a família de modelos **KapiCoder-Mongo** para o Kapibara Studio (IDE desktop .NET para MongoDB):

- autocomplete (FIM/ghost text) que entende o contexto do editor (aba, conexão › banco › coleção, schema inferido, console mongosh/DSL) e sugere código MongoDB;
- chat/edição curta (pedido → proposta de query com diff);
- **modo agente local** com as ferramentas do Kapibara (tool calling).

Tudo roda localmente via **ONNX Runtime GenAI** (hoje 0.15.2 no app; builds `WinML` (CPU + DirectML), `Cpu`, `Cuda`).

## Professor

`C:\IA\deepseek-coder-6.7b-base` — DeepSeek-Coder 6.7B **base** (LlamaForCausalLM, vocab 32.256, 16k ctx, FIM nativo `<｜fim▁begin｜>` / `<｜fim▁hole｜>` / `<｜fim▁end｜>`). Não é instruct e não sabe chamar ferramentas. Tokenizer diferente dos students prováveis (Qwen 151k) → destilação por logits só é possível entre mesmo vocabulário; caso contrário destilação por dataset (sequence-level).

## Camadas (tiers) — nomes fixos usados em todos os documentos

| Tier | Nome | Alvo de hardware | Uso principal |
|---|---|---|---|
| T0 | KapiCoder-Mongo-Nano (~0,5B) | CPU qualquer, < 2 GB RAM | autocomplete rápido |
| T1 | KapiCoder-Mongo-Small (~1,5B) | CPU boa / iGPU / DirectML | autocomplete + chat curto |
| T2 | KapiCoder-Mongo-Pro (~7B, até ~14B) | GPU dedicada 16 GB (AMD RX 7800 XT, DirectML) | autocomplete de alta qualidade + chat |
| T3 | KapiCoder-Mongo-NPU | NPU Hexagon do Snapdragon X (X Elite/X Plus, "X1"; e X2 quando aplicável) via QNN EP | autocomplete/chat com baixo consumo |
| T4 | KapiCoder-Mongo-Agent | GPU 16 GB (e variante menor para NPU/CPU se viável) | modo agente local (tool calling multi-turno) |

A Fase 01 escolhe os modelos base de cada tier; as demais fases se referem ao tier (T0…T4) e citam os candidatos da Fase 01 quando necessário.

## Estrutura do plano (arquivos em `C:\IA\models\KapibaraStudio.MongoIA\docs\`)

| Arquivo | Conteúdo |
|---|---|
| `README.md` | Índice, visão geral, cronograma, dependências entre fases (orquestrador) |
| `fase-00-visao-premissas.md` | Objetivos, premissas, lições da v1, critérios globais (orquestrador) |
| `fase-01-selecao-modelos-base.md` | Escolha dos modelos base por tier |
| `fase-02-professor.md` | Preparação do professor DeepSeek 6.7B (adaptação de domínio, contrato, uso na destilação) |
| `fase-03-dataset-autocomplete-validacao.md` | Material de treino e validação do autocomplete/chat |
| `fase-04-dataset-agente-tools.md` | Material de treino com as ferramentas do Kapibara (modo agente) |
| `fase-05-treinamento.md` | Receitas de treino/destilação por tier |
| `fase-06-avaliacao-benchmarks.md` | Avaliação, benchmarks, gates de aprovação |
| `fase-07-exportacao-onnx.md` | Exportação/quantização ONNX: CPU, DirectML, QNN/NPU |
| `fase-08-integracao-kapibara.md` | Integração na IDE (adapters, metadata, agente local), homologação, release |

Use links relativos entre esses arquivos (ex.: `[Fase 03](fase-03-dataset-autocomplete-validacao.md)`).

## Fontes locais obrigatórias (leia o que for relevante para sua fase)

- Pipeline anterior (v1 compacto, lições aprendidas): `C:\IA\models\SlopStudio.MongoAI-Compact\` — `docs\01..07*.md`, `docs\INTEGRATION.md`, `teacher\TEACHER.md`, `scripts\*.py`, `configs\`, `reports\`.
- Repositório da IDE: `F:\source\esilva-silva\EsilvaSoft.KapibaraStudio\` (leia o `CLAUDE.md`/`AGENTS.md`; docs em pt-BR):
  - `docs\21-autocomplete-local.md`, `docs\23-onnx-slopcoder.md`, `docs\26-ia-local-multimodelo.md`, `docs\auto-complite\*.md`, `docs\20-console.md`, `docs\10-decisoes-arquiteturais.md`.
  - Código de IA local: `src\EsilvaSoft.KapibaraStudio.Infrastructure.LocalAi*`, `src\EsilvaSoft.KapibaraStudio.LocalAi.Core`, `src\EsilvaSoft.KapibaraStudio.Autocomplete.Core`.
  - Agente: `src\EsilvaSoft.KapibaraStudio.Application\Agents\` (`AgentToolRegistry*.cs`, `AgentProductToolNames.cs`, `LocalAgentProvider.cs`, `AgentRuntime*.cs`), `src\EsilvaSoft.KapibaraStudio.Infrastructure.Agents`, `src\EsilvaSoft.KapibaraStudio.McpServer`.
- Não edite nada no repositório da IDE nem no pipeline v1: somente leitura.

## Fatos já conhecidos da v1

- Students v1: Qwen2.5-Coder 0.5B/1.5B Base, LoRA r=64, destilação por dataset do SlopCoder 6.7B (DeepSeek ajustado). Resultados chat livre 120 pedidos: 0.5B=67, 1.5B=87; cego(40): 20/29. ~1 em 4 pedidos de chat ainda errados.
- Exportações DirectML precisam ser feitas com provider `dml` no builder; INT4 DML mantém embedding fora da quantização. Exportação CPU falha em DirectML (`DmlFusedNode`).
- Treino em RX 7800 XT 16 GB com ROCm no Windows (PyTorch ROCm); memória compartilhada causa offload silencioso → limitar VRAM; cross-entropy em blocos; gradient checkpointing; professor em NF4.
- NPU/QNN e CUDA nunca foram homologados.
- `LocalModelAiChatService` foi removido (ADR-055); verifique o estado atual do chat/agente no código antes de assumir contratos.

## Regras de escrita

- Português do Brasil; identificadores, comandos e nomes de arquivo em inglês.
- Markdown com tabelas, checklists (`- [ ]`) de entregáveis, critérios de saída (gates) mensuráveis, riscos e mitigação.
- Seja concreto: caminhos de pastas, formatos JSONL com exemplos, hiperparâmetros iniciais, comandos/scripts propostos (nomes), estimativas de tempo/VRAM.
- Diferencie o que é **fato verificado** (cite o arquivo/fonte/URL) do que é **proposta/hipótese a validar**.
- Para informações de mercado/modelos/runtimes (versões, suporte QNN, licenças), pesquise na web e cite as URLs; hoje é 2026-10-08.
- Estrutura sugerida de cada fase: Objetivo · Entradas · Atividades (passo a passo) · Artefatos/estrutura de pastas · Critérios de saída · Riscos · Dependências · Estimativa.
