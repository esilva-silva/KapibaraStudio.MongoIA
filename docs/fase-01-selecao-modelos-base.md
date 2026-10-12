# Fase 01 · Seleção dos modelos base por tier

> Escopo: escolher, para cada tier (T0…T4), **um modelo base primário e uma alternativa**, com critérios explícitos, evidência de mercado
> (pesquisada em 2026-10-08) e um experimento curto de pré-seleção (*bake-off*) que confirma ou derruba a escolha antes de gastar GPU com treino.
> Legenda usada no documento: **[F]** = fato verificado (arquivo local ou URL citada) · **[H]** = hipótese/proposta a validar no bake-off.
> **Revisão 2026-10-08b — troca do professor:** o professor deixa de ser o DeepSeek-Coder-6.7B (agora só **referência/controle** do bake-off) e passa a ser o **Qwen2.5-Coder-14B Base** (alternativo: **7B**), escolhido por nova sonda de QLoRA (job **J03b**, §6.6; regra 14B × 7B). Consequências: tudo Qwen (§5.3) e mesmo tokenizer professor/students (C6 recalculado, §5.1); **T2 = KapiTeacher-Mongo (14B ou 7B) DML INT4**, adapter `qwen2`; T0/T1 perdem a alternativa DeepSeek-1.3B; o piloto de KD vira **A (sequence-level) × B (+ KD por logits do KapiTeacher)** (braço C e GOLD saem, §6.5); o caminho DeepSeek da IDE deixa de ser usado por qualquer tier novo (§7); estimativa +4–6 h de GPU pela sonda (§12). Decisões detalhadas em `_briefing-troca-professor.md`.

## 1. Objetivo

1. Definir o modelo base de T0 Nano, T1 Small, T2 Pro, T3 NPU e T4 Agent.
2. Escolher o **professor** (Qwen2.5-Coder-14B × 7B Base; DeepSeek-Coder-6.7B só como referência) e fechar a questão estratégica Qwen × DeepSeek: **tudo Qwen** (§5.3, §6.6).
3. Entregar à [Fase 02](fase-02-professor.md), [Fase 05](fase-05-treinamento.md), [Fase 07](fase-07-exportacao-onnx.md) e
   [Fase 08](fase-08-integracao-kapibara.md) uma lista fechada de bases, tokenizers, formatos de prompt e riscos de exportação.

## 2. Entradas

| Entrada | Onde | O que usamos |
|---|---|---|
| Lições da v1 | `C:\IA\models\SlopStudio.MongoAI-Compact\docs\01-visao-geral.md`, `07-onnx-selecao.md` | Qwen2.5-Coder 0.5B/1.5B Base como students; destilação só por dataset (vocabulários 32k × 152k) **[F]**; regra de seleção (menor INT4 dentro do orçamento: TTFT médio ≤ 800 ms, p95 ≤ 1.500 ms em CPU, RAM ≤ 4 GB, ≤ 2,5 GB em disco, perda de quantização ≤ 0,3 APT) **[F]**; DML exige export com `-e dml` e embedding fora do INT4 por causa de `tie_word_embeddings` → `GatherBlockQuantized` **[F]** |
| Resultados v1 | `07-onnx-selecao.md` | 0.5B INT4 CPU: TTFT 215 ms, 1,4 GB RAM; 1.5B-full DML-FP16: TTFT 156 ms, chat livre 87/120 **[F]** |
| IA local da IDE | `F:\source\esilva-silva\EsilvaSoft.KapibaraStudio\docs\26-ia-local-multimodelo.md`, `23-onnx-slopcoder.md` | Um modelo carregado por vez; `ChatModel` separado existe mas troca descarrega o anterior; NPU (QNN/OpenVINO/VitisAI) só "preparado", nenhum build distribui **[F]** |
| Adapters | `src\EsilvaSoft.KapibaraStudio.Infrastructure.LocalAi.OnnxAdapter\{IModelAdapter,QwenCoderModelAdapter,DeepSeekCoderModelAdapter,ModelAdapters}.cs` | `QwenCoderModelAdapter.CanHandle` aceita **somente** `model.type == "qwen2"`; `DeepSeekCoderModelAdapter` aceita `llama` + `slopcoder_manifest.json` e usa `DeepSeekModelTokenizer` (.NET próprio, paridade em 547 strings) **[F]** |
| Agente local | `src\EsilvaSoft.KapibaraStudio.Application\Agents\LocalAgentProvider.cs` | O provider local **não declara tool calling** (`SubmitToolResultAsync` lança `NotSupportedException`); usa o prefixo FIM/código, sem template de chat **[F]** |
| Professor (referência histórica) | `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\deepseek-coder-6.7b-base` (origem: `C:\IA\deepseek-coder-6.7b-base`) | LlamaForCausalLM, vocab 32.256, 16k ctx, FIM `<｜fim▁begin｜>`/`<｜fim▁hole｜>`/`<｜fim▁end｜>` **[F]** (briefing). **Revisão 2026-10-08b:** deixa de ser treinado; fica como controle zero-shot do bake-off e fonte das medições v1 (560 tok/s, 10,18 GB). O professor passa a ser Qwen2.5-Coder-14B Base (alt. 7B), baixado do HF e fixado por SHA |

## 3. Critérios de seleção ponderados

Pontuação 1–5 por critério; nota final = Σ(peso × nota)/5 (máximo 100). Os pesos mudam por tier porque o uso muda (autocomplete × agente).

| # | Critério | Como medir | Peso T0/T1 | Peso T2 | Peso T3 | Peso T4 |
|---|---|---|---:|---:|---:|---:|
| C1 | Qualidade de código / FIM no domínio | Bake-off zero-shot FIM Mongo (APT, EM, syntax, Mongo-valid) + benchmarks públicos (HumanEval-infilling) | 20 | 25 | 15 | 10 |
| C2 | FIM nativo treinado (não só tokens no vocabulário) | Model card/relatório + teste do bake-off | 15 | 10 | 10 | 0 |
| C3 | Licença comercial sem restrição relevante | Apache-2.0/MIT = 5; licenças com cláusulas de uso (DeepSeek, Llama, Gemma) = 3; não comercial = **eliminatório** | 10 | 10 | 10 | 10 |
| C4 | Tamanho / latência / memória no hardware alvo | TTFT, tok/s, RAM, KV-cache por token | 15 | 10 | 15 | 10 |
| C5 | Exportabilidade: builder ORT GenAI + DirectML (+ QNN para T3) | Arquitetura no `builder.py`; export fp32/INT4 CPU e `-e dml` sem erro; receita Olive QNN existente | 10 | 15 | 25 | 10 |
| C6 | Tokenizer compatível com o professor (KD por logits) | Mesmo tokenizer do KapiTeacher (Qwen2.5-Coder, ~151.665 tokens reais **[F a confirmar]**) = 5; cross-tokenizer (ULD/GOLD) = 2 | 10 | 5 | 5 | 0 |
| C7 | Tool calling (formato nativo, BFCL) | Model card + bake-off de tools | 0 | 5 | 5 | 30 |
| C8 | Contexto útil (≥ 4k para autocomplete, ≥ 16k para agente) | config + teste | 5 | 5 | 5 | 10 |
| C9 | Multilinguismo pt-BR/en (pedidos de chat) | Bake-off de 40 pedidos pt-BR | 5 | 5 | 5 | 5 |
| C10 | Esforço de integração na IDE | Reaproveita adapter existente = 5; novo adapter simples = 3; novo build/runtime = 1 | 10 | 10 | 5 | 10 |
| C11 | Raciocínio híbrido (thinking ligável/desligável no **mesmo** peso, orçamento controlável, tool calling com raciocínio) | Model card/relatório + bake-off ligado × desligado (§6.3). Híbrido nativo com chave dura e branda = 5; só chave dura = 4; thinking-only ou instruct-only (exige 2 modelos) = 3/2; sem raciocínio = 1 | 0 | 0 | 0 | 15 |
| | **Total** | | 100 | 100 | 100 | 100 |

> Revisão 2026-10-08 (requisito novo: o T4 é opção do painel "Agente IA" ao lado do Claude CLI e precisa de **modo de raciocínio**): pesos do T4 rebalanceados — C1 10 → 5, C7 30 → 25, C11 = 15 (a coluna T4 anterior somava 95, não 100).

Regras eliminatórias (independem da nota):
- licença não comercial (CC-BY-NC, *qwen-research*, MNPL) → fora como base (pode ser usado só como referência de avaliação);
- arquitetura ausente do builder do ORT GenAI **e** sem receita Olive → fora (exigiria exportador próprio);
- não cabe no alvo de memória do tier em INT4.

## 4. Pesquisa de mercado e runtimes (2026-10-08)

### 4.1 ONNX Runtime GenAI: o que o builder exporta

- **[F]** O `builder.py` atual (branch `main`) mapeia, entre outras: `LlamaForCausalLM`, `MistralForCausalLM`, `Mistral3ForConditionalGeneration`,
  `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`, `Phi3ForCausalLM`, `Phi4MMForCausalLM`,
  `GraniteForCausalLM`, `GraniteMoeHybridForCausalLM`, `Gemma2/3ForCausalLM`, `Gemma4*`, `SmolLM3ForCausalLM`, `Lfm2*`, `Ernie4_5ForCausalLM`, `GptOss`, `Olmo`,
  `Nemotron`. **`Starcoder2ForCausalLM` não está mapeado**; DeepSeek-V2/V3 (MLA/MoE) também não.
  Providers do `-e`: `cpu`, `cuda`, `dml`, `webgpu`, `NvTensorRtRtx` — **sem `qnn`**.
  Fonte: <https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/src/python/py/models/builder.py>,
  <https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/README.md>.
- **[F]** Versões: 0.15.2 (07/ago, a da IDE), **0.16.0 (22/set: continuous batching, speculative decoding, tool calling, Qwen3.5 dense/MoE)**,
  **0.17.0 (28/set: prefix caching, WinML zip para x64 e arm64)**, 0.17.1 (29/set, NuGet; as datas do NuGet na [Fase 07 §2.3](fase-07-exportacao-onnx.md) diferem das tags do GitHub — alvo de upgrade da IDE = **0.17.1**); 0.13.0 trouxe "QNN stateful models"; 0.14.0 removeu o empacotamento QNN dos
  pipelines do GenAI e adicionou documentação do QNN EP. Fonte: <https://github.com/microsoft/onnxruntime-genai/releases>.
- **[H]** A IDE está presa em 0.15.2. Qwen2/Llama/Qwen3/Phi3 já são suportados nessa versão; Qwen3.5 (híbrido Gated DeltaNet) exige ≥ 0.16 e não
  há evidência de que os operadores recorrentes rodem em DirectML ou QNN.

### 4.2 Snapdragon X (Hexagon NPU) via QNN

- **[F]** O caminho oficial para LLM em NPU no ORT GenAI é **pré-compilado**: binários de contexto QNN (`*.bin`, EPContext) + wrappers ONNX +
  `quantizer.onnx`/`dequantizer.onnx`/`position-processor.onnx`, gerados com Qualcomm AI Hub (`qai_hub_models ... export --device "Snapdragon X Elite CRD"`),
  em Linux/WSL, com horas de compilação. Fonte: <https://onnxruntime.ai/docs/genai/howto/build-models-for-snapdragon.html>.
- **[F]** Receita Olive para modelo próprio (`olive-recipes/Qwen-Qwen2.5-1.5B-Instruct/QNN/config.json`): `CaptureSplitInfo` → `GptqModel` (4 bits, simétrico,
  rotação Hadamard, `lm_head` 8 bits/grupo 32) → `ModelBuilder` (int4) → `MatMulNBitsToQDQ` → `GraphSurgeries` → `OnnxStaticQuantization`
  (**ativações uint16**, pesos uint8 nos demais, calibração 256 amostras, `max_seq_len` 4096) → `SplitModel` → `StaticLLM` (`context_length: 64`, batch 1)
  → `EPContextBinaryGenerator` (`soc_model: "60"`, `weight_sharing: true`). Ou seja, **w4a16 estático, grafo estático, contexto pré-compilado por SoC**.
  Fonte: <https://raw.githubusercontent.com/microsoft/olive-recipes/main/Qwen-Qwen2.5-1.5B-Instruct/QNN/config.json>.
- **[F]** O repositório `microsoft/olive-recipes` tem receitas QNN para: Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, Qwen3-0.6B (AI Toolkit),
  Phi-4-mini-instruct, Llama-3.2-1B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B, Phi-3/3.5-mini, Llama-3.1-8B (com `x_elite_config.json` e
  `x2_elite_config.json`). **Não há receita QNN para Qwen2.5-Coder** (só OpenVINO/TRT-RTX). Fonte: <https://github.com/microsoft/olive-recipes>.
- **[F]** Qualcomm AI Hub: Llama-3.2-3B-Instruct em **w4a16** (parte em w8a16), **contexto 4096**, suportado em Snapdragon X Elite e X2 Elite
  (<https://aihub.qualcomm.com/models/llama_v3_2_3b_instruct>); Phi-3.5-mini com teto de 4096 tokens (<https://aihub.qualcomm.com/models/phi_3_5_mini_instruct>);
  Llama-3.2-1B (<https://aihub.qualcomm.com/models/llama_v3_2_1b_instruct>). Microsoft publica `qwen2.5-7b-instruct-qnn-npu` no catálogo Foundry
  (<https://ai.azure.com/catalog/models/qwen2.5-7b-instruct-qnn-npu>). Nenhum Qwen2.5-Coder QNN pré-compilado encontrado.
- **[F]** Windows ML baixa e registra o QNN EP pelo `ExecutionProviderCatalog` (`EnsureAndRegisterCertifiedAsync`) em dispositivos Qualcomm compatíveis.
  Fontes: <https://ryzenai.docs.amd.com/projects/WinML/en/latest/winml_ep.html> (espelho da documentação Windows ML),
  <https://support.microsoft.com/en-us/servicing/os/windows/ai-components/2026/01/windows-execution-provider-components>.
- **[F]** Foundry Local (v0.7, set/2025) anunciou NPU Intel/AMD e "trabalho contínuo" em Qualcomm (<https://devblogs.microsoft.com/foundry/foundry-local-meets-more-silicon/>).
  Phi Silica é modelo do sistema (não treinável por nós) — irrelevante como base.
- **Implicações [H]:** T3 deve ser **≤ ~2B** (precedentes oficiais vão até 3–8B, mas com custo de memória compartilhada e TTFT), **contexto 2–4k**,
  prompt processado em blocos de 64 (o que favorece prompts curtos de autocomplete), **um pacote por SoC** (X Elite `soc_model 60`; X2 Elite separado).
- **[F] Hardware de homologação disponível (usuário):** ASUS Vivobook 16 X1607QA, Snapdragon X **X1-26-100** (entrada de linha: 8 núcleos Oryon até 2,97 GHz,
  Adreno X1-45 ≈ 1,7 TFLOPS, **NPU Hexagon 45 TOPS e LPDDR5x ≈ 135 GB/s — iguais à X Elite**), **16 GB compartilhados**, Windows 11 Home
  (<https://www.asus.com/br/laptops/for-home/vivobook/asus-vivobook-16-x1607q/techspec/>; specs de terceiros e veredito de suporte QNN na [Fase 07 §2.3](fase-07-exportacao-onnx.md)).
  O chip **não consta** da lista de requisitos do QNN EP do Windows ML (só X1E/X1P) e o `soc_model` do die (Purwa, HTP v73 **[H forte]**) não é público →
  o S-NPU-0 da Fase 07 prova o caminho antes do bake-off NPU. **Implicação [H]:** a NPU deve render como a da X Elite, mas o orçamento de memória do modelo
  é **≤ 3 GB (teto 4 GB)** e a CPU de fallback é ~1,5–2× mais lenta → o 0.5B ganha peso no T3 (ver §5.2).

### 4.3 Famílias candidatas

| Família / modelo | Tamanhos relevantes | Licença | FIM | Tool calling | Arquitetura (builder) | Vocab | Observações | Fonte |
|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-Coder** Base/Instruct | 0.5B, 1.5B, 7B, 14B (3B = *qwen-research*, **excluído**) | Apache-2.0 (exceto 3B) | **Sim, treinado** | Instruct: sim (template Hermes `<tool_call>`) | `Qwen2ForCausalLM` ✔ (IDE ✔) | ~151.665 reais (`vocab_size` 152.064 em 7B/14B; 151.936 em 0.5B/1.5B) **[F a confirmar]** | Base da v1; **professor (14B, alt. 7B) e students T0/T1/T3**; HumanEval base: 0.5B 28,0 · 1.5B 43,9 · 7B 61,6 · 14B 64,0 **[F]** relatório técnico | <https://arxiv.org/abs/2409.12186>, <https://huggingface.co/Qwen/Qwen2.5-Coder-3B> |
| **DeepSeek-Coder** v1 Base/Instruct | 1.3B, 6.7B (professor), 33B | DeepSeek License (uso comercial permitido, com restrições de uso) | **Sim, treinado** | Não | `LlamaForCausalLM` ✔ (IDE ✔ via manifesto; **sem tier novo**) | 32.256 (era o do professor antigo; hoje só referência) | 1.3B: MHA (sem GQA), 16k ctx, treino EN/ZH; HumanEval base 1.3B 34,8 **[F]** (tabela do relatório Qwen2.5-Coder) | <https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base> |
| **Qwen3** (denso, abr/2025) | 0.6B, 1.7B, 4B, 8B, 14B | Apache-2.0 | Tokens FIM existem no vocabulário; **treino FIM não confirmado** | **Sim, forte** (`<tool_call>`) | `Qwen3ForCausalLM` ✔ (IDE ✘: `model.type = qwen3`) | 151.669 | **Raciocínio híbrido no mesmo peso**: `enable_thinking` (chave dura) + `/think`/`/no_think` (chave branda); `<think>` = id 151667, `</think>` = 151668; 32k nativo (131k com YaRN). BFCL-v3 do Qwen3-8B: **68,1 com thinking × 60,2 sem** **[F]** (tab. 17/18 do relatório técnico) | <https://huggingface.co/Qwen/Qwen3-8B>, <https://arxiv.org/abs/2505.09388>, <https://github.com/QwenLM/Qwen3/discussions/1277> |
| **Qwen3 2507** | 4B-Instruct-2507 / **4B-Thinking-2507**, 30B-A3B-Instruct/**Thinking-2507** (30,5B total, 3,3B ativos), 235B-A22B | Apache-2.0 | Não | Sim | 4B: `Qwen3ForCausalLM` ✔; 30B-A3B: `Qwen3MoeForCausalLM` (fora do orçamento 16 GB como student) | 151.669 (mesmo do Qwen3) | **Não é híbrido**: a linha separa Instruct (sem thinking) e Thinking (só thinking; o template já abre `<think>`; saída recomendada até 32k tokens). 4B-Thinking-2507: BFCL-v3 71,2 e TAU2-Retail 53,5 (Qwen3-4B original: 65,9 / 38,6) **[F]**. 30B-A3B-Thinking-2507: BFCL-v3 72,4, LiveCodeBench v6 66,0 **[F]** → candidato a **professor de raciocínio offline** ([Fase 04 §6.5](fase-04-dataset-agente-tools.md)) | <https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507>, <https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507> |
| **Qwen3.5** small | 0.8B, 2B, 4B, 9B (fev–mar/2026) | Apache-2.0 | Não documentado | Sim (parser `qwen3_coder`) | `Qwen3_5ForConditionalGeneration` (≥ 0.16) | 248.320 | Híbrido Gated DeltaNet + atenção, encoder de visão embutido, 262k ctx; 2B propenso a loops de thinking. 9B: thinking ligado por padrão, desligável só por `enable_thinking=false` (**sem** `/think`/`/no_think`); BFCL-V4 66,1, TAU2 79,1 **[F]** | <https://huggingface.co/Qwen/Qwen3.5-9B>, <https://huggingface.co/Qwen/Qwen3.5-2B>, <https://artificialanalysis.ai/articles/qwen3-5-small-models> |
| **Qwen3.6** | 27B denso, 35B-A3B MoE (abr/2026); **nenhum tamanho ≤ 9B** encontrado até 2026-10-08 | Apache-2.0 | — | Sim (`qwen3_coder`); opção `preserve_thinking` para manter raciocínio entre turnos | híbrido DeltaNet + MoE + visão (fora do builder 0.15.x e do orçamento 16 GB) | 248k (família 3.5) | Só serve como **professor de raciocínio offline** (SWE-bench Verified 73,4 no 35B-A3B **[F]**); vocabulário ≠ Qwen3 (sem KD por logits) | <https://huggingface.co/Qwen/Qwen3.6-35B-A3B> |
| **Qwen3-Coder** | 30B-A3B, Next 80B-A3B (MoE) | Apache-2.0 | Sim (Next) | Sim | MoE — fora do orçamento 16 GB em qualidade aceitável | — | Sem tamanho pequeno oficial; no HumanEval-Infilling o Qwen2.5-Coder-3B (64,7%) supera Qwen3-Coder-30B (64,2%) e Next (41,1%) **[F]** | <https://arxiv.org/pdf/2605.04894>, <https://arxiv.org/pdf/2603.00729> |
| **Granite 4.0** (Nano/Micro) | 350M, 1B (versões H = híbrido Mamba-2, e não-H = transformer) | Apache-2.0 | Não confirmado | Sim (BFCLv3 forte para o tamanho) | `GraniteForCausalLM` ✔ / `GraniteMoeHybrid` ✔ | 100k | Bom candidato agente pequeno; código Mongo fraco **[H]** | <https://huggingface.co/blog/ibm-granite/granite-4-nano> |
| **Phi-4-mini-instruct** | 3.8B | MIT | Não | Sim (nativo) | `Phi3ForCausalLM` ✔ | 200k | **Receita Olive QNN existe** | <https://github.com/microsoft/olive-recipes> |
| **Llama 3.2** | 1B, 3B | Llama 3.2 Community | Não | Sim (3B razoável) | `LlamaForCausalLM` ✔ | 128k | **Pré-compilado AI Hub (w4a16, 4k)** | <https://aihub.qualcomm.com/models/llama_v3_2_3b_instruct> |
| **Gemma 3 / 3n** | 270M, 1B, 4B | Gemma Terms | Não (CodeGemma sim, antigo) | Fraco/por prompt | `Gemma3ForCausalLM` ✔ | 262k | Vocabulário enorme pesa em modelos pequenos | builder.py |
| **SmolLM3** | 3B | Apache-2.0 | Não | Sim | `SmolLM3ForCausalLM` ✔ | 128k | Reasoning dual, 64k ctx | builder.py |
| **StarCoder2** | 3B, 7B, 15B | BigCode OpenRAIL-M | Sim | Não | **✘ não mapeado** | 49k | Eliminado (exportador) | builder.py |
| **Mistral**: Codestral / Devstral / Ministral 3 | Codestral 22B (MNPL, não comercial); Devstral Small 24B (Apache); Ministral 3 3B/8B/14B (Apache, dez/2025) | variada | Codestral sim | Ministral 3: nativo | `Mistral3ForConditionalGeneration` ✔ | 131k | Codestral eliminado (licença); Devstral 24B INT4 ≈ 13–14 GB sem folga em 16 GB; Ministral 3 8B é alternativa agente **[H]** | <https://www.datalearner.com/en/ai-models/pretrained-models/ministral-3-14b-25-12> |
| **Hammer 2.1 / xLAM-2** (fine-tunes de tool calling sobre Qwen2.5) | 0.5B–7B | **CC-BY-NC-4.0** | — | Excelentes no BFCL | `qwen2` | 151k | **Eliminados** como base (não comercial); úteis como referência de formato/avaliação | <https://huggingface.co/MadeAgents/Hammer2.1-1.5b>, <https://huggingface.co/Salesforce/xLAM-2-3b-fc-r> |

### 4.4 KD por logits: o que o vocabulário permite

- **[F a confirmar no `tokenizer.json`/`config.json` dos checkpoints]** Mesmo vocabulário do professor = **toda a família Qwen2.5-Coder** (KapiTeacher 14B/7B e students 0.5B/1.5B/T3): ~151.665 tokens reais; `vocab_size` 152.064 (7B/14B) × 151.936 (0.5B/1.5B). Para KD, restringir os logits às entradas reais do tokenizer e mascarar as linhas de padding de professor e aluno ([Fase 05 §6.3](fase-05-treinamento.md)).
- **[F]** (revisão 2026-10-08b) A destilação **cross-tokenizer** (ULD, Boizard et al., TMLR 2025, <https://arxiv.org/abs/2402.12030>; **GOLD** no TRL, `trl.experimental`, ainda experimental — <https://huggingface.co/docs/trl/gold_trainer>) **deixa de ser necessária**: o motivo que a justificava (professor DeepSeek de vocabulário 32.256 × students Qwen) desapareceu. Fica citada só como alternativa caso um tier futuro use outro tokenizer.

### 4.5 Custo de KV-cache (calculado a partir de `config.json`, fp16) **[H: confirmar no bake-off]**

| Modelo | Camadas | KV heads × head_dim | KV/token | KV @ 2.048 | KV @ 16k |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-0.5B | 24 | 2 × 64 | 12 KB | 24 MB | 0,2 GB |
| Qwen2.5-Coder-1.5B | 28 | 2 × 128 | 28 KB | 56 MB | 0,45 GB |
| DeepSeek-Coder-1.3B | 24 | 16 × 128 (MHA) | **192 KB** | **384 MB** | 3,0 GB |
| Qwen2.5-Coder-7B | 28 | 4 × 128 | 56 KB | 112 MB | 0,9 GB |
| Qwen2.5-Coder-14B (KapiTeacher/T2) | 48 | 8 × 128 | 192 KB | 384 MB | 3,0 GB (~0,8 GB a 4k) |
| Qwen3-8B | 36 | 8 × 128 | 144 KB | 288 MB | 2,3 GB |

O DeepSeek-Coder-1.3B tem embedding/`lm_head` muito menores (32k × 2048) que o Qwen 1.5B (151k × 1536), mas o KV-cache MHA é ~7× maior: no
autocomplete com 2k de contexto isso é aceitável; em NPU (memória e grafo estático) e em agente com contexto longo, não.

**Contexto alvo do T4 = 16k tokens** (o raciocínio consome contexto: blocos `<think>` das chamadas do turno corrente ficam no histórico até a resposta final — [Fase 04 §5.6](fase-04-dataset-agente-tools.md)). Qwen3-8B INT4 DML ≈ 5 GB de pesos + 2,3 GB de KV a 16k ≈ 7,5–8 GB **[H]**: cabe na RX 7800 XT. Qwen3-4B-Thinking-2507 (36 camadas, 8 KV heads × 128) tem o mesmo KV/token do 8B **[H: conferir `config.json`]**, logo não economiza KV, só pesos.

## 5. Matriz de decisão por tier

Notas pré-bake-off (hipóteses **[H]** baseadas em §4); o bake-off (§6) as substitui por medições.

### 5.1 Pontuação dos finalistas

| Candidato | Tier | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | Nota |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-0.5B Base | T0 | 3 | 5 | 5 | 5 | 5 | 5 | 1 | 4 | 3 | 5 | — | **89** |
| Qwen3-0.6B Base | T0 | 2 | 2 | 5 | 5 | 4 | 2 | 3 | 4 | 4 | 3 | — | 65 |
| DeepSeek-Coder-1.3B Base (referência; saiu do plano) | T0 | 3 | 5 | 3 | 3 | 4 | 2 | 1 | 4 | 2 | 4 | — | 68 |
| Qwen2.5-Coder-1.5B Base | T1 | 4 | 5 | 5 | 4 | 5 | 5 | 1 | 4 | 3 | 5 | — | **90** |
| DeepSeek-Coder-1.3B Base (referência; saiu do plano) | T1 | 3 | 5 | 3 | 4 | 4 | 2 | 1 | 4 | 2 | 4 | — | 71 |
| Qwen2.5-Coder-7B Base (KapiTeacher-7B) | T2 | 4 | 5 | 5 | 4 | 4 | 5 | 3 | 4 | 3 | 5 | — | 85 |
| Qwen2.5-Coder-14B Base (KapiTeacher-14B) | T2 | 5 | 5 | 5 | 2 | 3 | 5 | 3 | 4 | 4 | 5 | — | 84 |
| DeepSeek-Coder-6.7B (referência; não treinado) | T2 | 4 | 5 | 3 | 4 | 3 | 2 | 1 | 4 | 2 | 4 | — | 70 |
| Qwen2.5-Coder-1.5B (QNN) | T3 | 4 | 5 | 5 | 4 | 3 | 5 | 1 | 3 | 3 | 2 | — | **73** |
| Qwen2.5-Coder-0.5B (QNN) | T3 | 3 | 5 | 5 | 5 | 3 | 5 | 1 | 3 | 3 | 2 | — | 73 (empate; desempate por C1) |
| Llama-3.2-1B (QNN pré-compilável) | T3 | 2 | 1 | 3 | 4 | 5 | 1 | 2 | 3 | 3 | 2 | — | 62 |
| **Qwen3-8B (híbrido, thinking ligado por padrão)** | T4 | 3 | 1 | 5 | 3 | 4 | 0 | 5 | 4 | 4 | 3 | 5 | **85** |
| Qwen3-4B-Thinking-2507 | T4 | 3 | 1 | 5 | 4 | 4 | 0 | 5 | 5 | 4 | 3 | 3 | 83 |
| Qwen3.5-9B (controle; só HF até a IDE ≥ 0.16) | T4 | 3 | 0 | 5 | 3 | 2 | 0 | 5 | 5 | 4 | 1 | 4 | 76 |
| Ministral-3-8B-Instruct | T4 | 3 | 1 | 5 | 3 | 3 | 0 | 4 | 5 | 4 | 2 | 2 | 69 |
| Qwen2.5-Coder-7B-Instruct | T4 | 4 | 3 | 5 | 4 | 4 | 0 | 3 | 4 | 3 | 4 | 1 | 67 |

Notas de T0–T3 recalculadas na revisão 2026-10-08b com C6 = 5 para os Qwen (mesmo tokenizer do KapiTeacher Qwen) e C6 = 2 para o DeepSeek (cross-tokenizer); a escolha do **tamanho** do professor (14B × 7B) não sai desta matriz e sim da regra da §6.6 (a nota de T2 só mostra que os dois ficam à frente do DeepSeek-6.7B). Notas T4 recalculadas com os pesos revisados (C1 5, C7 25, C11 15). O Qwen2.5-Coder-7B-Instruct cai de alternativa para último colocado: **não tem modo de raciocínio**, o que agora é requisito do painel "Agente IA". O Qwen3-4B-Thinking-2507 fica perto (BFCL-v3 superior ao Qwen3-8B com thinking, 71,2 × 68,1 **[F]**), mas é **thinking-only**: desligar o raciocínio exigiria um segundo modelo (4B-Instruct-2507) e o pacote/slot extra na IDE; e não aceita orçamento por chave branda.

### 5.2 Escolhas

| Tier | Primário | Alternativa | Justificativa curta |
|---|---|---|---|
| **T0 Nano** | **Qwen2.5-Coder-0.5B Base** | Qwen3-0.6B Base (controle do bake-off; só entra se o primário reprovar G1/G3) | FIM treinado, Apache, KV mínimo, adapter `qwen2` pronto, v1 provou 215 ms TTFT em CPU e 1,4 GB RAM **[F]**; **mesmo tokenizer do KapiTeacher** → KD por logits no plano base. *Alternativa DeepSeek-Coder-1.3B removida (revisão 2026-10-08b): sua razão de ser era o mesmo tokenizer do professor, que agora é dos Qwen* |
| **T1 Small** | **Qwen2.5-Coder-1.5B Base** | Qwen3-1.7B Base (controle do bake-off; só entra se o primário reprovar G1/G3) | Melhor base pública de 1–2B com FIM e Apache (HumanEval 43,9 × 34,8 do DeepSeek 1.3B) **[F]**; v1 chegou a 87/120 no chat livre **[F]**; DML-FP16 homologado na IDE **[F]**; mesmo tokenizer do KapiTeacher (KD por logits no plano base). *Alternativa DeepSeek-Coder-1.3B removida pelo mesmo motivo do T0* |
| **T2 Pro** | **KapiTeacher-Mongo (14B ou 7B)** — o professor adaptado da [Fase 02](fase-02-professor.md) (base **Qwen2.5-Coder-14B** se a regra da §6.6 aprovar; senão **7B**), exportado com `-e dml` INT4, adapter `qwen2` **já existente** — **revisão 2026-10-08b** | **KapiTeacher-Mongo-7B** (se o 14B perder a regra da §6.6 ou reprovar o TTFT/G4 como T2; nesse caso o 7B vira também o professor, ver [Fase 05 §9.5](fase-05-treinamento.md)) | **Por que o professor:** sem GPU na nuvem, um QLoRA extra de 7–14B custa 40–50+ h de GPU + uma trilha de avaliação/exportação; o KapiTeacher já é treinado no mesmo contrato FIM/chat da IDE e usa o `QwenCoderModelAdapter` **[F]** (§7), o mesmo contrato de prompt dos demais tiers. **14B INT4 DML** ≈ 8,5–9 GB + KV 192 KiB/token (~0,8 GB a 4k): cabe em 16 GB; TTFT maior que o do 7B (a medir no G4) **[H]**; **7B INT4** ≈ 5 GB. Gate G2.8 (papel de T2, [Fase 02](fase-02-professor.md)) continua: o professor só vira T2 se passar os gates de T2 da [Fase 06](fase-06-avaliacao-benchmarks.md). O DeepSeek-Coder-6.7B não é mais candidato (referência do bake-off) |
| **T3 NPU** | **Qwen2.5-Coder-1.5B** (mesmos pesos ajustados do T1, exportados por receita Olive QNN adaptada de `Qwen2.5-1.5B-Instruct/QNN`) | Qwen2.5-Coder-0.5B (pesos do T0, mesma receita) | Mesma arquitetura (`Qwen2ForCausalLM`, `tie_word_embeddings`) da receita oficial w4a16 → menor risco de exportação; reaproveita treino de T1/T0 (sem modelo extra). Contexto 2.048 (máx. 4.096 da receita). 0.5B é a rede de segurança se a perda de qualidade w4a16 ou a memória/latência do 1.5B falharem. **Revisão com o hardware real (Vivobook X1-26-100, 16 GB compartilhados):** a NPU é a mesma da X Elite e ambos cabem (1.5B w4a16 ≈ 1,4–1,9 GB de processo; 0.5B ≈ 0,6–0,9 GB, [Fase 07 §5.5.1](fase-07-exportacao-onnx.md)), então a escolha não muda por capacidade, mas a **ordem** muda: o **0.5B é exportado e homologado primeiro** (menor risco, autocomplete) e o 1.5B só vira o T3 distribuído se passar os gates de latência **e** memória da [Fase 06](fase-06-avaliacao-benchmarks.md) no X1-26-100 **[proposta]**. Llama-3.2-1B/3B pré-compilado (ou um modelo QNN do Foundry Local) fica só como **controle de hardware** (prova que a máquina/QNN EP funcionam — S-NPU-0) |
| **T4 Agent** | **Qwen3-8B em modo híbrido, thinking ATIVO por padrão** no painel "Agente IA" (`enable_thinking=true`, blocos `<think>…</think>` do template nativo), com opção de desligar (`/no_think` ou `enable_thinking=false`) e **orçamento de raciocínio** configurável (padrão 1.024 tokens por geração, máx. 4.096; ao estourar, o runtime força `</think>` e o modelo segue para a resposta/tool call — [Fase 04 §5.6](fase-04-dataset-agente-tools.md)); contexto alvo **16k**. Variante **Agent-Lite**: **Qwen3-1.7B** (CPU/DML) com thinking curto **opcional** (orçamento 256–512) / **Qwen3-0.6B** (NPU, receita QNN existe) **sem thinking**. No notebook de homologação (X1-26-100, 16 GB) cabem os dois: 0.6B w4a16 4k ≈ 1,1–1,5 GB na NPU; 1.7B `int4` 8k ≈ 2,7–3,0 GB na **CPU** (padrão do Lite nessa máquina); 1.7B na NPU (4k, ≈ 2,0–2,5 GB) só como experimento ([Fase 07 §5.5.1/§5.7.4](fase-07-exportacao-onnx.md)) | **Qwen3-4B-Thinking-2507** **[H]** (raciocínio-only; desligar exige par com 4B-Instruct-2507). Fallback sem raciocínio: Qwen2.5-Coder-7B-Instruct (compartilha o slot de T2) | Único base Apache ≤ 8B com raciocínio **híbrido no mesmo peso** + tool calling forte: o thinking ganha +7,9 pontos no BFCL-v3 (68,1 × 60,2) **[F]**; formato `<tool_call>` Hermes/ChatML, mesmo tokenizer-base do Qwen2.5, 32k ctx nativo; `Qwen3ForCausalLM` já no builder 0.15.x. A linha 2507 separou instruct e thinking (sem híbrido 8B), o Qwen3.5-9B exige runtime ≥ 0.16 com DeltaNet sem evidência em DML, e não há Qwen3.6 ≤ 9B (§4.3) — **nenhuma evidência forte para trocar o primário**; 4B-Thinking-2507 entra no bake-off (§6.3). Custo aceito: T4 ≠ T2 (a IDE carrega **um modelo por vez** **[F]**), mitigado pela carga sob demanda do modo agente |

### 5.3 Questão estratégica: Qwen × DeepSeek

**Conclusão (revisão 2026-10-08b): tudo Qwen. O DeepSeek-Coder fica só como referência/controle no bake-off.**

| Argumento | Qwen2.5-Coder / Qwen3 | DeepSeek-Coder v1 |
|---|---|---|
| Base pública | Mais forte em todos os tamanhos (2024); Qwen3 adiciona tool calling forte | 2023, sem sucessor denso pequeno; V2-Lite é MoE com outro tokenizer |
| KD por logits com o professor (KapiTeacher Qwen) | **Nativo** (mesmo tokenizer; top-k cacheado `topk-v1`) | Só cross-tokenizer (GOLD/ULD) — **não é mais necessário** |
| Escada de tamanhos | 0.5 / 1.5 / 7 / 14 (+ Qwen3 0.6 / 1.7 / 4 / 8) | Só 1.3 / 6.7 / 33 — **não há ~0.5B** |
| Tool calling / chat pt-BR | Sim / razoável | Não / fraco (treino EN/ZH) |
| IDE | Adapter `qwen2` com tokenizer nativo do GenAI, um só contrato de prompt para todos os tiers de autocomplete | Adapter existe, mas com tokenizer .NET próprio (mais manutenção) |
| NPU | Receita Olive QNN da mesma arquitetura | Sem receita QNN; KV MHA grande |
| Licença | Apache-2.0 (sem a cláusula de "Derivatives" da licença DeepSeek) | DeepSeek License (permite comercial, cláusulas de uso) |

O único ganho que o DeepSeek tinha (KD por logits no mesmo vocabulário) agora existe também nos Qwen, porque o professor é Qwen. Portanto:
1. **Todos os tiers em Qwen**; destilação sequence-level + **KD por logits do KapiTeacher** (plano base de T0/T1, [Fase 05 §6.3](fase-05-treinamento.md)). GOLD/ULD **não é necessário** (fica citado só como alternativa).
2. **Piloto A × B** (§6.5): A = Qwen 1.5B só sequence-level; B = A + KD por logits do KapiTeacher.
3. **DeepSeek-Coder-6.7B** (`BaseRepos\deepseek-coder-6.7b-base`) e o 1.3B ficam só como controle zero-shot do bake-off (§6.1, §6.6); não são mais treinados nem empacotados.

## 6. Experimento de pré-seleção (bake-off)

Tudo zero-shot (sem treino), exceto o piloto de KD (§6.5). Objetivo: eliminar candidatos ruins em ~1 semana, não ranquear finamente.

### 6.1 Candidatos por bancada

| Bancada | Modelos |
|---|---|
| FIM/autocomplete | Qwen2.5-Coder 0.5B/1.5B/7B/14B Base, DeepSeek-Coder 6.7B Base (**referência/controle**; 1.3B opcional), Qwen3 0.6B/1.7B/4B/8B Base, Granite-4.0-1B, Qwen3.5-2B (controle; só HF) |
| Seleção do professor (§6.6) | DeepSeek-Coder-6.7B (referência) × Qwen2.5-Coder-7B × Qwen2.5-Coder-14B: zero-shot FIM Mongo + **sonda de QLoRA (J03b)** |
| Tool calling | Qwen3 0.6B/1.7B/4B-Instruct-2507/8B, Qwen2.5-Coder-1.5B/7B-Instruct, Phi-4-mini-instruct, Granite-4.0-1B/H-1B, Ministral-3-8B-Instruct, Llama-3.2-3B-Instruct, SmolLM3-3B; Hammer2.1-1.5b (só referência, NC) |
| Tool calling **com raciocínio** (T4/Agent-Lite) | Qwen3-8B e Qwen3-1.7B em **thinking ligado × desligado** (mesmo peso), Qwen3-4B-Thinking-2507, Qwen3.5-9B (controle, só HF), SmolLM3-3B (reasoning dual); orçamentos 256/512/1.024/4.096 |
| Exportabilidade | todos os aprovados nas duas bancadas acima |
| NPU (Vivobook X1-26-100, 16 GB) | Qwen2.5-Coder-0.5B (primeiro) e 1.5B via receita Olive QNN; Qwen3-0.6B (Agent-Lite); Llama-3.2-1B pré-compilado ou modelo QNN do Foundry Local (controle) |

### 6.2 Conjuntos de amostra

```text
C:\IA\models\KapibaraStudio.MongoIA\
├── bakeoff\
│   ├── data\
│   │   ├── fim_mongo_300.jsonl        # 300 casos do bench v1 (mongosh, EJSON, Atlas Search, DSL) no contrato ModelPrefix
│   │   ├── chat_ptbr_en_40.jsonl      # 40 pedidos (20 pt-BR, 20 en) do conjunto cego v1
│   │   ├── tools_kapibara_60.jsonl    # 60 casos com schemas reais de AgentToolRegistry (single, multi, sem-tool, arg inválido)
│   │   └── tools_reasoning_20.jsonl   # 20 tarefas multi-passo no kapisim piloto (explain capturado, multi-estágio, ambiguidade) p/ thinking on × off
│   ├── scripts\                       # bakeoff_fim.py, bakeoff_tools.py, bakeoff_export.py, bakeoff_latency.py, bakeoff_report.py
│   └── reports\<data>\                # results.jsonl, matrix.md, export_log\*.txt
```

Exemplo `fim_mongo_300.jsonl` (texto neutro; cada adapter monta seu formato FIM):

```json
{"id":"fim-0042","prefix":"db.orders.aggregate([\n  { $match: { status: \"paid\" } },\n  { $group: { _id: \"$customerId\", total: { ","suffix":" } } }\n])","reference":"$sum: \"$amount\"","tags":["aggregate","group"]}
```

Exemplo `tools_kapibara_60.jsonl` (nomes reais vindos de `AgentProductToolNames.cs`, schemas exportados do registro — ver [Fase 04](fase-04-dataset-agente-tools.md)):

```json
{"id":"tool-007","lang":"pt-BR","tools":"<schemas JSON do AgentToolRegistry>","messages":[{"role":"user","content":"Quais índices a coleção orders do banco shop tem?"}],"expected":[{"name":"get_indexes","arguments":{"connectionId":"<uuid>","database":"shop","collection":"orders"}}]}
```

O agente **não** executa consultas (ADR-066): não há ferramenta de contagem/consulta; pedidos como "quantos pedidos pagos?" esperam `propose_file_edit` com a consulta ou resposta em texto ([Fase 04 §3.4](fase-04-dataset-agente-tools.md)).

### 6.3 Métricas e procedimento

| Medição | Script | Métrica | Execução |
|---|---|---|---|
| FIM zero-shot | `bakeoff_fim.py --model <hf_id> --format qwen-fim|deepseek-fim --n 300` | APT (tokens aceitos, métrica v1), EM, Syntax-valid, Mongo-valid, tamanho médio | HF Transformers bf16 na RX 7800 XT (ROCm), greedy, 32 tokens |
| Chat curto | `bakeoff_fim.py --chat` | intenção correta (rubrica v1), % respostas em código puro | idem, 256 tokens |
| Tool calling | `bakeoff_tools.py --template chatml-hermes|phi4|granite|mistral` | nome correto, JSON válido, schema-válido, AST-match de argumentos, abstenção correta (caso sem tool), pt-BR × en | template oficial de cada modelo (`apply_chat_template(tools=...)`) |
| Tool calling com raciocínio | `bakeoff_tools.py --thinking on|off --think-budget 256|512|1024|4096 --seeds 3` (mesmos 60 casos + 20 casos multi-passo do `kapisim` piloto: diagnóstico de plano do explain capturado, consulta multi-estágio, pedido ambíguo) | as métricas acima **pareadas** ligado × desligado; tokens de raciocínio médio/p95; taxa de estouro do orçamento; coerência pensamento→ação (a ação anunciada no `<think>` é a emitida); idioma do raciocínio; TTFA (tempo até a 1ª ação) | thinking: amostragem recomendada pelo Qwen (T=0,6, top-p 0,95, top-k 20; **greedy é desaconselhado no thinking** **[F]** model card Qwen3-8B); non-thinking: T=0,7, top-p 0,8; 3 seeds |
| Exportabilidade | `bakeoff_export.py` | builder 0.15.2 **e** 0.17.0: fp32 CPU, INT4 CPU, `-e dml` FP16/INT4 (sem `GatherBlockQuantized`); primeiro token 100% igual ao HF em 12 prompts; tokenizer nativo do GenAI = HF em 400 prompts (100% IDs, FIM tokens únicos) | venv `.venv-artifacts` + `.venv-winml` (como na v1) |
| Latência | `bakeoff_latency.py` | TTFT médio/p95, tok/s, pico RAM (CPU Ryzen 9 7900, `intra_op_num_threads=0`), TTFT/tok/s DML | 100 prompts reais, um processo por vez |
| NPU (opcional) | Olive `olive run --config qnn_qwen25coder_1.5b.json` + `bakeoff_latency.py --provider qnn` | compila? TTFT, tok/s, Δ APT vs. fp32, **pico de memória com a IDE aberta** | Vivobook X1-26-100 (HTP v73; `soc_model` lido no S-NPU-0 da [Fase 07](fase-07-exportacao-onnx.md) — a receita usa 60 = X Elite) |

### 6.4 Critérios de saída do bake-off (gates)

| Gate | Regra |
|---|---|
| G1 FIM | Para T0–T3, candidato só segue se APT zero-shot ≥ 85% do melhor candidato do mesmo tier e Syntax-valid ≥ 70%. Qwen3 Base só substitui Qwen2.5-Coder se superar em APT **e** EM (prova de FIM treinado) |
| G2 Tools | Para T4: nome correto ≥ 80%, JSON/schema-válido ≥ 90%, abstenção ≥ 70% no conjunto de 60, diferença pt-BR × en ≤ 10 pp. Primário = maior AST-match entre os que passam e cabem em 16 GB INT4 |
| G2b Raciocínio | Para T4 (zero-shot, orçamento 1.024): com thinking, AST-match/sucesso multi-passo ≥ sem thinking (não-inferior; o ganho com significância só é exigido após o treino, [Fase 06 §8](fase-06-avaliacao-benchmarks.md)); estouro do orçamento ≤ 15%; formato `<think>…</think>` + `<tool_call>` parseável ≥ 95%; o candidato precisa **desligar** o raciocínio sem segundo modelo (senão C11 ≤ 3). Agent-Lite 1.7B: idem com orçamento 512, e o modo desligado decide se o thinking curto fica opcional ou é removido |
| G3 Export | fp32 CPU e `-e dml` (FP16 ou INT4) exportam e geram no builder **0.15.2** (versão da IDE) com paridade de primeiro token 100% e tokenizer 100%. Falha só no 0.15.2 mas sucesso no 0.17.0 → candidato fica condicionado ao upgrade da IDE ([Fase 08](fase-08-integracao-kapibara.md)) |
| G4 Latência | T0: TTFT médio ≤ 300 ms CPU INT4, RAM ≤ 2 GB. T1: ≤ 800/1.500 ms CPU (regra v1) e ≤ 250 ms DML. T2: ≤ 400 ms DML, VRAM estimada ≤ 10 GB. T4: ≤ 1.000 ms DML com 4k de prompt e decodificação ≥ 30 tok/s (com thinking, a latência até a 1ª ação ≈ TTFT + tokens de raciocínio ÷ tok/s; ex.: 300 tokens a 40 tok/s ≈ +7,5 s **[H]**). T3: TTFT ≤ 500 ms com prompt de 512 tokens. *Limites de pré-seleção (eliminatórios, zero-shot); os gates de adoção, mais estritos, são os da [Fase 06 §8](fase-06-avaliacao-benchmarks.md)* |
| G5 Decisão | Relatório `bakeoff\reports\<data>\matrix.md` com a matriz da §5.1 recalculada por medições; escolha final assinada no [README](README.md). Se o primário de algum tier falhar, a alternativa assume sem nova rodada |

### 6.5 Piloto de KD (sequence-level × KD por logits do KapiTeacher), orçamento fixo

*Revisão 2026-10-08b: o antigo piloto A/B/C (Qwen × DeepSeek) vira **A × B**; o braço C (DeepSeek-1.3B) e o braço GOLD saem.*

- Dados: 20k exemplos de autocomplete + 3k de chat da [Fase 03](fase-03-dataset-autocomplete-validacao.md) (subset estratificado do pool T1), val 2k.
- Braço A: Qwen2.5-Coder-1.5B, LoRA r=64, SFT sequence-level (receita v1).
- Braço B: o mesmo SFT + **KD por logits do KapiTeacher** (mesmo tokenizer): cache `topk-v1` revisado (k=32, IDs **uint32**, logprobs fp16, massa residual fp16 → **194 B/posição**, [Fase 02 §3.5](fase-02-professor.md)), **reaproveitado do cache do pool T1** gerado junto com o scoring (J13: sem forward extra); perda `α·CE + (1−α)·τ²·KL` (forward KL, α 0,3–0,5, τ 1–2; [Fase 05 §6.3](fase-05-treinamento.md)). Variantes de B opcionais: τ=2,0 e α=0,3.
- 1 época cada, mesmo número de tokens; avaliação no `val` 2k + chat 40 (o `bench` da Fase 03 fica reservado à Fase 06).
- **Gate:** B entra como receita padrão de T0/T1 ([Fase 05](fase-05-treinamento.md)) se B ≥ A + 0,15 APT **ou** intenção de chat ≥ A + 3 pp, sem perder > 3 pp em pt-BR. Senão, T0/T1 seguem só sequence-level (a perda KD fica desligada; o cache já existe, sem custo extra).
- Custo estimado **[H]**: 20k + 3k exemplos ≈ 9 M tokens por braço → A ~1,1 h (2,4k tok/s **[F]** v1), cada B ~1,3 h (leitura do cache + KL, ~+10%), avaliação ~1 h por braço → **~8–10 h** no total (job J15 da [Fase 05 §9.2](fase-05-treinamento.md), depois do J13/J12). Tudo local.

### 6.6 Seleção do professor: DeepSeek-6.7B (referência) × Qwen2.5-Coder-7B × Qwen2.5-Coder-14B — job **J03b**

*Nova seção (revisão 2026-10-08b). Decisão do usuário: o professor é da família Qwen2.5-Coder Base; resta decidir o tamanho. Esta seção é a regra única; a [Fase 02](fase-02-professor.md) e a [Fase 05 §9](fase-05-treinamento.md) apenas a consomem.*

| Candidato | Papel | Params | HumanEval base **[F a confirmar no relatório 2409.12186]** | Licença | Observação |
|---|---|---:|---:|---|---|
| DeepSeek-Coder-6.7B Base | **referência/controle** (não treinado, não empacotado) | 6,7 B | ~47,6 | DeepSeek License | 560 tok/s e 10,18 GB no QLoRA v1 **[F]**; FIM `<｜fim▁*｜>`, vocab 32.256 |
| Qwen2.5-Coder-7B Base | **alternativo** | 7,6 B | ~61,6 | Apache-2.0 | 28 camadas, 4 KV heads **[F a confirmar]**; pico QLoRA ~10–11 GB, ~500–560 tok/s **[H]** |
| Qwen2.5-Coder-14B Base | **primário** | 14,7 B | ~64 | Apache-2.0 | 48 camadas, hidden 5120, 8 KV heads, embeddings não atrelados **[F a confirmar]**; pico QLoRA ~13–14,5 GB (no limite), ~250–300 tok/s **[H]** |

Critérios (todos medidos nos mesmos 300 casos `fim_mongo_300` e no mesmo ambiente do bake-off):
1. **Qualidade zero-shot** (FIM-acc/APT, EM, Syntax-valid, **exec-valid** nos casos com fixture) nos três, com o formato FIM nativo de cada um (`qwen-fim` × `deepseek-fim`; IDs FIM do Qwen2.5-Coder `<|fim_prefix|>` 151659, `<|fim_middle|>` 151660, `<|fim_suffix|>` 151661, `<|fim_pad|>` 151662, `<|endoftext|>` 151643 **[F a confirmar]**). Faz parte do J01.
2. **Sonda de QLoRA (J03b)**: 1–2 h em **cada** Qwen (o DeepSeek não repete: referência v1), NF4 double-quant nos lineares, embedding e `lm_head` em bf16, LoRA r=64 (14B: r=32 se o pico estourar), gradient checkpointing, CE em blocos, `limit_vram` 14,8 GB, dados do subset `train` da Fase 03, micro-batch de 4k e de 8k tokens (≥ 100 passos após 30 de aquecimento, `probe_throughput.py` da [Fase 05 §8](fase-05-treinamento.md)). Mede **tok/s**, **VRAM dedicada de pico** e **VRAM compartilhada** (qualquer valor > 0,2 GB = offload silencioso = reprova, lição v1). **Custo: ~4–6 h de GPU** (carga dos pesos, 2 micro-batches × 2 modelos, merge de descarte).
3. **Exportabilidade/latência do T2** (parte do J03): `-e dml` INT4 do 14B e do 7B com o builder 0.15.2 (arquitetura `Qwen2ForCausalLM` ✔): VRAM do pacote + KV a 4k e TTFT (gate G4 T2: ≤ 400 ms DML, VRAM ≤ 10 GB).

**Regra de decisão (briefing, item 2):** o professor é o **14B** se **(a)** VRAM de pico ≤ 14 GB com micro-batch ≥ 4k tokens, sem paginação para a memória compartilhada; **(b)** throughput ≥ ~250 tok/s; e **(c)** vantagem zero-shot ≥ 3 p.p. de FIM-acc **ou** de exec-valid sobre o 7B. **Senão, 7B.** Falha em (a) ou (b) é eliminatória do 14B; falha só em (c) devolve o 7B (mais barato: ≈ 60 h de GPU a menos no programa, [Fase 05 §9](fase-05-treinamento.md)). O resultado é gravado em `reports\teacher_choice.json` (id HF, SHA, nome KapiTeacher-Mongo-14B/-7B, números da sonda) e repassado às Fases 02, 05, 06, 07 e 08. A decisão é tomada **antes do J04/J08** (semana S2–S3): depois que o CPT do professor começa, trocar custa horas já gastas.

Risco registrado **[H]**: o 14B pode passar (a)–(c) e ainda reprovar o TTFT/G4 como **T2**; nesse caso o 14B continua professor (dados/destilação) e o T2 passa a ser o KapiTeacher-7B treinado à parte (O4 da [Fase 05 §9.4](fase-05-treinamento.md)), ou o T2 cai para o T1 em GPU.

#### Resultado do J03b e do FIM zero-shot — 2026-10-09 **[F]**

Bancada FIM (1.000 casos fim+completion do benchmark v1, contrato `editor-context-v1`; students bf16, candidatos a professor NF4; `bakeoff/reports/2026-10-09-fim/`):

| Modelo | chars aceitos | EM | syntax | mongo_valid | TTFT HF (ms) |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-0.5B (bf16) | 3,94 | 0,053 | 0,539 | 0,404 | 15 |
| Qwen2.5-Coder-1.5B (bf16) | 5,24 | 0,101 | 0,629 | 0,547 | 31 |
| DeepSeek-Coder-6.7B (NF4, referência) | 6,01 | 0,173 | 0,679 | 0,646 | 363 |
| Qwen2.5-Coder-7B (NF4) | 5,73 | 0,162 | 0,782 | 0,735 | 313 |
| Qwen2.5-Coder-14B (NF4) | 6,08 | 0,166 | 0,804 | 0,772 | 576 |

O APT por token do DeepSeek (2,72 × 2,22 do 14B) era artefato do vocabulário de 32k; em caracteres ele empata com o 14B e perde **12,6 pp de mongo_valid** (p < 0,001). 14B × 7B: mongo_valid +3,7 pp (McNemar p = 0,038), chars aceitos +0,36 [IC95 +0,01; +0,71] → critério (c) atendido.

Sonda de QLoRA (`bakeoff/scripts/probe_qlora.py`, NF4 double-quant, LoRA em todos os lineares, checkpointing, CE em blocos, PagedAdamW 8-bit, teto de VRAM 14,14 GB; `reports/teacher_probe*.log`):

| Modelo | micro-batch | r | tok/s | pico alocado | status |
|---|---:|---:|---:|---:|---|
| 7B | 4.096 | 64 | **360** | 8,22 GB | ok |
| 7B | 8.192 | 64 | **309** | 10,52 GB | ok |
| 14B | 2.048 | 64 | 245 | 12,33 GB | ok (abaixo do mínimo de 4k) |
| 14B | 4.096 | 64 | — | — | OOM |
| 14B | 4.096 | 32 | 75 | 13,35 GB (reservado 14,04) | no teto, alocador degradado |
| 14B | 8.192 | 32 | — | — | OOM |

**Decisão: professor = Qwen2.5-Coder-7B (KapiTeacher-Mongo-7B)** — o 14B falha (a) e (b), eliminatórios. Registro em `reports/teacher_choice.json`. Consequências: T2 = KapiTeacher-Mongo-7B (DML INT4 ~5 GB); o 7B treina a **309–360 tok/s**, abaixo dos 500–560 estimados → as horas de CPT/SFT do professor na [Fase 05 §9](fase-05-treinamento.md) sobem ~1,5× (SFT ~47 M tokens ≈ 36–42 h por época, não 24 h). exec-valid ainda não medido (fixtures da Fase 03); `mongo_valid` é a validação estática da v1.

## 7. Impactos para a IDE

| Item | Tiers | Impacto | Esforço [H] |
|---|---|---|---|
| Qwen2.5-Coder (`qwen2`) | T0, T1, **T2 (KapiTeacher 14B/7B)**, T3 | **Nenhum adapter novo**: `QwenCoderModelAdapter` + `QwenFimPromptBuilder` + tokenizer nativo do GenAI. Tokens FIM `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`; paradas `<|endoftext|>`, `<|im_end|>`, `<|fim_pad|>` | 0 |
| Qwen3 (`qwen3`) | T4 (e T0/T1 se vencer o G1) | `CanHandle` só aceita `qwen2` → **generalizar o adapter** para `qwen2|qwen3` (ou `Qwen3ModelAdapter`); verificar tokens FIM no `tokenizer.json`; adicionar `<think>`/`</think>` à rejeição de marcadores **no autocomplete** (no agente eles passam a ser esperados) | 0,5–1 dia |
| Raciocínio (thinking) | T4, Agent-Lite 1.7B | Separar o bloco `<think>…</think>` (ids 151667/151668) antes do parser de `<tool_call>`; exibir o raciocínio no painel "Agente IA" como bloco recolhível; chave liga/desliga (`enable_thinking` no template; `/no_think` como chave branda); **orçamento** contado em tokens com fechamento forçado de `</think>` pelo runtime; remover raciocínio de turnos anteriores ao re-renderizar o histórico (regra do template — invalida o prefix cache a partir da última resposta final); amostragem não-greedy no thinking. Detalhes de contrato: [Fase 04 §5.6](fase-04-dataset-agente-tools.md); implementação: [Fase 08](fase-08-integracao-kapibara.md) | Fase 08 |
| DeepSeek-Coder (`llama`) — `DeepSeekCoderModelAdapter`, `DeepSeekModelTokenizer` | **nenhum tier novo** (revisão 2026-10-08b) | O caminho DeepSeek da IDE deixa de ser usado por qualquer tier novo (T2 passa a `qwen2`). O código existente permanece só para pacotes já instalados; remover ou manter como legado é decisão da [Fase 08](fase-08-integracao-kapibara.md). Nada de manifesto `slopcoder_manifest.json`/paridade de 547 vetores novo para o plano | 0 |
| Template de chat / tool calling | T4 | `LocalAgentProvider` hoje não tem template de chat nem tools **[F]**: precisa `IChatPromptBuilder` (ChatML + bloco `<tools>` + parser de `<tool_call>{json}</tool_call>`), nova capacidade `tools` em `kapibarastudio-model.json` (nomes desconhecidos são ignorados hoje **[F]**) e implementar `SubmitToolResultAsync` (`<tool_response>`). Avaliar tool calling/constrained decoding do ORT GenAI 0.16 | Fase 08: 5–8 dias |
| Phi-4-mini / Granite / Ministral (só se vencerem G2) | T4 | Novo adapter por família (`phi3`, `granite`, `mistral`), templates próprios | 1–2 dias cada |
| NPU | T3 | Build **win-arm64** WinML já existente + registro do QNN EP pelo `ExecutionProviderCatalog` (ou pacote `onnxruntime-qnn`), pacote com binários EPContext por SoC, `hardware: ["npu"]`, `recommendedContextTokens ≤ 2048` | Fase 07/08: 5–10 dias no Vivobook X1-26-100 (disponível) |
| Upgrade ORT GenAI 0.15.2 → 0.17.x | todos (opcional) | Prefix caching (útil para o cabeçalho de contexto do autocomplete), tool calling, Qwen3.5; exige novos lockfiles WinML/Cpu/Cuda | Fase 08: 2–3 dias |
| Um modelo por vez | T2 + T4 | Se T4 ≠ T2, usar `ChatModel` separado implica descarregar/recarregar ao alternar autocomplete × agente (e a v1 registrou queda do processo ao carregar o segundo modelo DML **[F]**). Mitigação: carregar T4 só sob demanda do painel "Agente IA"; Qwen2.5-Coder-7B-Instruct compartilhando o slot de T2 fica só como fallback **sem raciocínio** (não atende o requisito de thinking) | decisão |

## 8. Entregáveis

- [ ] `bakeoff\data\fim_mongo_300.jsonl`, `chat_ptbr_en_40.jsonl`, `tools_kapibara_60.jsonl`, `tools_reasoning_20.jsonl`
- [ ] Relatório thinking ligado × desligado (T4 e Agent-Lite) com orçamentos, tokens de raciocínio, estouro e TTFA (gate G2b)
- [ ] Scripts `bakeoff_fim.py`, `bakeoff_tools.py`, `bakeoff_export.py`, `bakeoff_latency.py`, `bakeoff_report.py`
- [ ] `bakeoff\reports\<data>\matrix.md` (matriz §5.1 com medições) e `results.jsonl`
- [ ] Log de exportação por candidato (builder 0.15.2 e 0.17.0, CPU e DML)
- [ ] Relatório do piloto de KD (A × B) com teste do sinal pareado
- [ ] Relatório da sonda de QLoRA do professor (J03b: tok/s, VRAM dedicada/compartilhada, zero-shot FIM nos três) e `reports\teacher_choice.json` com a decisão 14B × 7B (§6.6)
- [ ] `BaseRepos\base-repos.lock.json`: id HF, revisão (commit SHA), licença, SHA-256 dos pesos, tokenizer, formato de prompt, por tier
- [ ] Decisão registrada no [README](README.md) e repassada às Fases 02–08

## 9. Critérios de saída da Fase 01

1. Cada tier tem primário + alternativa com G1–G4 aprovados (ou justificativa escrita para exceção).
2. Licenças conferidas e registradas em `BaseReposase-repos.lock.json` (o `LICENSE` de cada repo fica em `BaseRepos\<repo>\`) (Apache-2.0 para todas as bases Qwen; a DeepSeek License só se arquiva como registro da referência, sem obrigação de pacote).
3. Revisões HF fixadas por SHA em `BaseRepos\base-repos.lock.json`.
4. Para T3: um pacote QNN compilado e medido no Vivobook X1-26-100 (0.5B no mínimo; 1.5B se couber nos gates), ou — se o S-NPU-0 mostrar que o QNN EP não registra nesse chip — o registro explícito "T3 bloqueado: QNN EP sem suporte ao X1-26-100" com a receita pronta.
5. Professor (14B × 7B) fechado pela sonda J03b (§6.6) e decisão Qwen × DeepSeek registrada (tudo Qwen); receita de KD (A × B) fechada pelo gate do piloto §6.5.

## 10. Riscos e mitigação

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Qwen3 Base não tem FIM treinado | Alta | T4 não serve para autocomplete | T4 só agente; FIM fica com Qwen2.5-Coder (T0–T2) |
| Qwen3 não exporta para DML no builder 0.15.2 | Média | T4 sem GPU | Testar no G3; upgrade 0.17 condicionado; alternativa Qwen2.5-Coder-7B-Instruct (`qwen2`, homologado) |
| Receita QNN não fecha para Qwen2.5-Coder (ops, memória, precisão w4a16) | Média | T3 atrasa | Partir da receita oficial Qwen2.5-1.5B (mesma arquitetura); fallback 0.5B; QAT/GPTQ com dados Mongo na calibração |
| Único hardware NPU é um Snapdragon X **X1-26-100** (entrada de linha), fora da lista oficial do QNN EP do Windows ML (só X1E/X1P), com 16 GB compartilhados | Média | T3 não homologável ou limitado por memória | S-NPU-0 na semana 1 (registro do EP, `soc_model`, LLM de controle); contexto gerado no próprio dispositivo se o `soc_model` não for conhecido; orçamento ≤ 3 GB por modelo; sem serviço de nuvem (requisito local de 2026-10-08); sem evidência não há distribuição (regra v1) |
| Pacote QNN é por SoC (X1-26-100 homologado; X1E/X1P/X2 não testados) | Certa | Pacote declara só o SoC testado | `soc_model`/`htp_arch` por configuração; metadata com SoC homologado |
| KD por logits do KapiTeacher sem ganho sobre sequence-level | Média | Baixo | Gate do braço B (§6.5); a perda KD é desligável e o cache top-k sai de graça do scoring. GOLD/ULD não é mais necessário |
| Professor 14B no limite de VRAM / lento (pico 13–14,5 GB, 250–300 tok/s **[H]**) | Média | Médio (calendário e offload) | Sonda J03b antes de qualquer job longo; regra 14B × 7B (§6.6); "trocar 14B por 7B" é o corte mais cedo da ordem de corte ([Fase 05 §9.5](fase-05-treinamento.md)) |
| Licença DeepSeek com restrições de uso | Eliminado | — | Nenhum tier novo usa DeepSeek; todos os modelos treinados ficam Apache-2.0 (atribuição e aviso de modificação mantidos) |
| T2 e T4 diferentes forçam troca de modelo na IDE | Alta | UX/estabilidade DML | Carregar T4 só sob demanda do painel "Agente IA"; fallback sem raciocínio = 7B Coder-Instruct compartilhando o slot do T2 |
| Qwen3.5 (mais novo) supera tudo, mas híbrido DeltaNet/visão | Média | Escolha "envelhece" | Mantido em watch-list; reavaliar quando DML/QNN suportarem os operadores e a IDE estiver ≥ 0.16 |
| Thinking não melhora o agente no domínio (ou piora formato) e só custa latência | Média | Requisito do painel sem ganho | Bake-off G2b ligado × desligado; gate de ganho com significância na Fase 06; se reprovar, o padrão do painel vira desligado (o mesmo peso serve os dois modos) |
| Latência do raciocínio na RX 7800 XT (8B INT4 DML) inaceitável no painel | Média | UX ruim frente ao Claude CLI | Orçamento padrão 1.024 com fechamento forçado; raciocínio conciso no treino ([Fase 05 §6.4b](fase-05-treinamento.md)); medir TTFA no G4; alternativa 4B-Thinking-2507 |
| Loops/repetição no thinking com decodificação greedy (aviso do model card) | Alta se greedy | Estouro de orçamento, saída degenerada | Amostragem recomendada (T=0,6/top-p 0,95/top-k 20) só no modo thinking; `presence_penalty` se o runtime suportar; orçamento força o fim |

## 11. Dependências

- **Entrada de:** [Fase 00](fase-00-visao-premissas.md) (orçamentos globais), [Fase 03](fase-03-dataset-autocomplete-validacao.md) (subset para bake-off e piloto), [Fase 04](fase-04-dataset-agente-tools.md) (schemas das tools para `tools_kapibara_60.jsonl`), [Fase 02](fase-02-professor.md) (professor adaptado e cache top-k `topk-v1` para o braço B do piloto).
- **Saída para:** [Fase 05](fase-05-treinamento.md) (bases, escolha do professor J03b e receita KD), [Fase 06](fase-06-avaliacao-benchmarks.md) (baselines zero-shot), [Fase 07](fase-07-exportacao-onnx.md) (arquiteturas, providers, receita QNN), [Fase 08](fase-08-integracao-kapibara.md) (adapters, template de chat, build ARM64/QNN, upgrade do GenAI).

## 12. Estimativa

| Atividade | Duração | GPU |
|---|---|---|
| Montar amostras e scripts do bake-off | 2 dias | — |
| FIM + chat zero-shot (~11 modelos; DeepSeek-6.7B só como referência) | 1 dia | ~6 h (14B em 4 bits) |
| Tool calling (~12 modelos) | 1 dia | ~4 h |
| Tool calling com raciocínio (ligado × desligado, 4 orçamentos, 3 seeds; ~5 modelos) | 1 dia | ~6 h |
| Exportabilidade e latência (CPU/DML, 0.15.2 e 0.17.0) | 2 dias | ~6 h |
| **Seleção do professor — sonda de QLoRA 7B × 14B (J03b, §6.6)** | 0,5–1 dia | **~4–6 h** |
| Piloto KD (A × B + 2 variantes de B; cache reaproveitado do scoring J13) | 2 dias | ~8–10 h (só depois do SFT do professor e do scoring, S7) |
| NPU no Vivobook X1-26-100 (S-NPU-0 + bake-off 0.5B/1.5B/0.6B) | 2–3 dias | compilação Olive local (horas, CPU x64) |
| Relatório e decisão | 0,5 dia | — |
| **Total** | **~10–13 dias úteis** (≈ 8 sem NPU) | **~35–40 h de GPU local** (J01–J03 + **J03b** + J15 da [Fase 05 §9.2](fase-05-treinamento.md)); tool calling/raciocínio do bake-off no llama.cpp para caber |
