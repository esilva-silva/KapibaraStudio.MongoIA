# Fase 00 · Visão, premissas e critérios globais

## Objetivo do programa

Produzir a família **KapiCoder-Mongo** — modelos locais, em ONNX, especializados em MongoDB para o Kapibara Studio:

1. **Autocomplete contextual** (FIM / ghost text): entende a aba (conexão › banco › coleção), o schema inferido, os índices e o console (mongosh/DSL) e sugere a continuação correta.
2. **Chat curto / edição**: pedido em pt-BR ou en → proposta de query com diff revisável.
3. **Modo agente local**: o modelo usa as ferramentas do Kapibara (listar conexões/bancos/coleções, índices, schema em cache, contexto do workspace, resultados/diagnósticos capturados, propor edição de arquivo…) em múltiplos turnos, com **modo de raciocínio** (thinking) e respeitando permissões e confirmações. É oferecido no painel **Agente IA** como provedor local, ao lado do Claude CLI — não é só autocomplete inline. O agente **não executa queries**: propõe a edição, a pessoa executa e o agente lê a saída capturada (ADR-066 removeu `mongo_find`/`mongo_explain` e similares — ver [Fase 04](fase-04-dataset-agente-tools.md)).

## Tiers

| Tier | Nome | Hardware-alvo | Capacidades | Prioridade |
|---|---|---|---|---|
| T0 | KapiCoder-Mongo-Nano (~0,5B) | Qualquer CPU, < 2 GB RAM | autocomplete | Alta |
| T1 | KapiCoder-Mongo-Small (~1,5B) | CPU boa / DirectML | autocomplete + chat | Alta |
| T2 | KapiCoder-Mongo-Pro (o próprio KapiTeacher: Qwen2.5-Coder-7B) | GPU 16 GB (RX 7800 XT, DirectML) | autocomplete + chat de alta qualidade | Média |
| T3 | KapiCoder-Mongo-NPU | NPU Hexagon do Snapdragon X X1-26-100 (QNN), Vivobook de 16 GB compartilhados (~3 GB para o modelo) | autocomplete + chat, baixo consumo | Média |
| T4 | KapiCoder-Mongo-Agent | GPU 16 GB (variante reduzida se viável) | agente do painel Agente IA: raciocínio (thinking) + tool calling multi-turno, contexto 16k | Média |

Escolha dos modelos base: [Fase 01](fase-01-selecao-modelos-base.md).

## Estratégia professor → aluno

- **Professor (revisão 2026-10-08b):** **Qwen2.5-Coder-7B Base** (escolhido em 2026-10-09 pela sonda de QLoRA da [Fase 01 §6.6](fase-01-selecao-modelos-base.md); o 14B não coube no QLoRA local), Apache-2.0, FIM nativo, **mesmo tokenizer dos students T0/T1/T3**. O DeepSeek-Coder 6.7B base (`C:\IA\deepseek-coder-6.7b-base`) fica só como referência no bake-off. A [Fase 02](fase-02-professor.md) adapta o professor ao domínio e ao contrato da IDE (KapiTeacher-Mongo) antes de ele ensinar; o KapiTeacher também é o T2.
- **Destilação sequence-level** (o professor gera, ranqueia e filtra exemplos) **mais KD por logits** para T0/T1, que compartilham o tokenizer Qwen2.5-Coder com o professor.
- **O professor não faz chat, raciocínio nem tool calling.** Traços de raciocínio do T4 vêm de auto-destilação do próprio modelo base com rejeição por execução e de um professor de raciocínio auxiliar offline ([Fase 04](fase-04-dataset-agente-tools.md)).
- **Chat e tool calling:** Esses dados vêm de geradores programáticos validados por execução real em MongoDB e por parsers ([Fase 03](fase-03-dataset-autocomplete-validacao.md), [Fase 04](fase-04-dataset-agente-tools.md)).
- **ONNX Runtime GenAI** em todos os alvos: CPU, DirectML, QNN ([Fase 07](fase-07-exportacao-onnx.md)).

## Lições da v1 (`C:\IA\models\SlopStudio.MongoAI-Compact`) — fatos verificados

| Lição | Consequência neste plano |
|---|---|
| Qwen2.5-Coder 0.5B/1.5B destilados funcionaram e já têm adapter na IDE | São linha de base obrigatória em todos os benchmarks |
| Chat livre: 0.5B = 67/120, 1.5B = 87/120; cego 20/40 e 29/40; ~1 em 4 pedidos de chat errados | Meta mínima: superar esses números com significância |
| Professor ajustado acertava sintaxe em modo instrução só em 41% | Chat/agente não dependem do professor |
| Exportação CPU quebra no DirectML (`DmlFusedNode`); INT4 DML precisa embedding fora da quantização | Exportar por alvo, nunca reutilizar pacote CPU na GPU |
| ROCm no Windows faz offload silencioso para memória compartilhada | Limitar VRAM, cross-entropy em blocos, gradient checkpointing |
| NPU/QNN e CUDA nunca homologados | T3 tem gate de viabilidade antes de investir em treino específico |
| Treinar exatamente no contrato de prompt da IDE foi decisivo | Contratos são extraídos do código atual, nunca reescritos à mão |

## Premissas

- Hardware de treino: Ryzen 9 7900 + RX 7800 XT 16 GB (ROCm). Sem nuvem: o calendário único da GPU está na [Fase 05](fase-05-treinamento.md).
- Hardware de homologação: Ryzen CPU, RX 7800 XT (DirectML) e **o notebook Snapdragon X disponível (16 GB de memória total, compartilhada com a NPU)** para T3/Agent-Lite. O treino é **100% local** no Ryzen 9 7900 + 64 GB RAM + RX 7800 XT; não há orçamento para GPU na nuvem.
- **MongoDB para validação por execução:** Community 8.0.32 no WSL (Ubuntu), `mongodb://127.0.0.1:27017`, bancos de fixture com prefixo `kapi_fx_`/`kapi_eval_`/`kapi_sim_`; `$search` sem execução (validação estática) até haver `mongot`.
- Pesos, datasets e checkpoints ficam fora do Git do Kapibara; os pacotes publicados vão para o Hugging Face (`esilva/...`) com licença compatível com a base.
- O repositório da IDE não é alterado por este plano; mudanças necessárias viram propostas/ADRs ([Fase 08](fase-08-integracao-kapibara.md)).
- Nenhuma ferramenta de escrita em MongoDB é ensinada ao agente (ausente de propósito no produto). **Verificado no código (commit `dba8546`, 2026-10-08):** `insert_one`, `update_one`, `delete_one`, `create_index` e `drop_index` só existem como constantes históricas (`AgentToolRegistry.LegacyNames.cs`) e flags de compatibilidade (`AgentToolExposure.AgentWriteToolRelease`), sem descriptor, schema nem handler; nenhum consumidor os recebe — nem o MCP server (Claude Code/CLIs externas, que lista só descriptors do registry via broker), nem o Copilot, nem o agente local. Detalhes na [Fase 04 §3.4](fase-04-dataset-agente-tools.md).

## Critérios globais de sucesso

- [ ] Cada tier supera a linha de base v1 (ou o modelo base sem ajuste, para tiers novos) em autocomplete e chat, com teste estatístico pareado ([Fase 06](fase-06-avaliacao-benchmarks.md)).
- [ ] 0 APIs/ferramentas inventadas nos conjuntos cegos acima do limite definido na Fase 06.
- [ ] Cada pacote ONNX passa a paridade com PyTorch e o harness com as DLLs reais da IDE no hardware-alvo.
- [ ] T4 conclui as tarefas do conjunto de agente por execução real, sem violar permissões.
- [ ] Com raciocínio ligado, T4 supera o modo sem raciocínio com significância e respeita o orçamento de raciocínio ([Fase 06](fase-06-avaliacao-benchmarks.md)).
- [ ] Dataset, scripts e relatórios reprodutíveis a partir de manifestos com hashes.

## Estrutura de pastas do workspace

Decisão do usuário (2026-10-08): modelos **baixados** do Hugging Face (bases e professores) ficam em `BaseRepos\`; cada modelo **criado** pelo programa tem seu próprio workspace em `Repos\`. O restante (dados, scripts, contratos, simulador) é compartilhado na raiz.

```text
C:\IA\models\KapibaraStudio.MongoIA\
├── docs\                 este plano
├── BaseRepos\            modelos baixados do https://huggingface.co (somente leitura após o download)
│   ├── base-repos.lock.json   repo id, revisão (commit SHA), SHA-256 por arquivo, licença, data do download
│   ├── Qwen2.5-Coder-14B\     reprovado como professor (J03b); pode ser apagado
│   ├── Qwen2.5-Coder-7B\      professor (KapiTeacher-Mongo-7B)
│   ├── Qwen2.5-Coder-1.5B\    base T1 (e T3)
│   ├── Qwen2.5-Coder-0.5B\    base T0 (e T3)
│   ├── Qwen3-8B\              base T4
│   ├── Qwen3-1.7B\ Qwen3-0.6B\   bases Agent-Lite
│   ├── Qwen3-14B-GGUF\        professor auxiliar de instrução/raciocínio (llama.cpp, Q4_K_M)
│   ├── Qwen3-4B-Thinking-2507\   alternativa T4 (só no bake-off)
│   └── deepseek-coder-6.7b-base\ referência histórica (cópia/junção de C:\IA\deepseek-coder-6.7b-base)
├── Repos\                um workspace por modelo criado
│   ├── KapiTeacher-Mongo-7B\      professor adaptado = T2 Pro
│   ├── KapiCoder-Mongo-Nano\      T0
│   ├── KapiCoder-Mongo-Small\     T1
│   ├── KapiCoder-Mongo-NPU\       T3 (exporta pesos do Nano/Small para QNN)
│   ├── KapiCoder-Mongo-Agent\     T4
│   └── KapiCoder-Mongo-Agent-Lite\
│       (estrutura interna de cada workspace)
│       ├── README.md              model card (base, dados, gates, licença Apache-2.0 + aviso de modificação)
│       ├── LICENSE
│       ├── model.lock.json        base usada (aponta para BaseRepos + revisão), dataset_version, commit dos scripts
│       ├── configs\               YAML de treino/KD/QAT/export deste modelo
│       ├── adapters\              LoRA por run
│       ├── checkpoints\           checkpoints de treino (retomada a cada 30 min)
│       ├── merged\                pesos bf16 mesclados por versão
│       ├── gguf\                  conversões para llama.cpp (avaliação/geração em lote)
│       ├── exports\               pacotes ONNX por alvo (<família>-ONNX-<variante>)
│       └── reports\               avaliação, paridade, gates.json
├── configs\              configurações compartilhadas (gates.yaml, mistura de dados, ambiente)
├── scripts\              geração, validação, treino, export, benchmark (comuns a todos os modelos)
├── contracts\            prompts/schemas extraídos da IDE (versionados por hash)
├── data\                 sources · schemas · fixtures · generated · validated · splits · eval · agent · distilled · calib · kd (Fases 02–04)
├── bakeoff\              amostras, scripts e relatórios do bake-off (Fase 01)
├── tools\                consoles .NET: ContractProbe, KapiToolCatalogExport, KapiRegistryHarness, GuidanceSmoke, ide-harness
├── sim\                  kapisim do agente (Fase 04)
├── cache\                caches de HF/ROCm/llama.cpp (descartáveis)
├── reports\              relatórios do programa (comparações entre modelos, calendário da GPU)
└── env\                  scripts de fixtures do MongoDB (WSL), lockfiles Python
```

Regras:

- **Download** só por `scripts/download_base.py <repo_id> --revision <sha>` (usa `huggingface_hub.snapshot_download`, apenas `*.safetensors`, `*.json`, tokenizer, `LICENSE`, `README.md`; nunca `.bin`/pickle) e registra tudo em `base-repos.lock.json`. Pastas de `BaseRepos\` nunca são escritas por treino.
- **Repos\<modelo>** é autocontido para publicação: o que for para o Hugging Face (`esilva/...`) sai de `exports\` + `README.md` + `LICENSE`. Cada workspace pode virar um repositório Git próprio (pesos via Git LFS ou fora do Git); a raiz versiona docs, scripts, configs e contratos.
- **Disco [H]:** `BaseRepos\` ≈ 110–130 GB (14B ~30 GB, Qwen3-8B ~16 GB, 7B ~15 GB, Qwen3-4B ~8 GB, GGUF 14B ~9 GB, menores ~10 GB, DeepSeek de referência ~13,5 GB); Pico ~620–720 GB em C: ([Fase 05 §9.7](fase-05-treinamento.md); 831 GB livres em 2026-10-08) → **pouca folga**: apagar checkpoints intermediários após o merge aprovado e, se necessário, mover `cache\` e `data\kd` para E: com junção (`mklink /J`). F: não recebe dados.
