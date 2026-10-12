# Briefing — troca do professor (revisão 2026-10-08b)

Decisão do usuário (aprovada no chat): o professor deixa de ser o DeepSeek-Coder-6.7B base e passa a ser da família **Qwen2.5-Coder Base**. Treino continua **100% local** (Ryzen 9 7900, 64 GB RAM, RX 7800 XT 16 GB ROCm). Todas as fases devem ficar coerentes com o que segue.

## Decisões

1. **Professor primário: Qwen2.5-Coder-14B Base** (Apache-2.0). **Alternativo: Qwen2.5-Coder-7B Base** (Apache-2.0). **DeepSeek-Coder-6.7B base** (`C:\IA\deepseek-coder-6.7b-base`) fica só como **referência histórica/controle** no bake-off; não é mais treinado.
   - Nome do professor adaptado: **KapiTeacher-Mongo-14B** (ou **KapiTeacher-Mongo-7B** se o alternativo vencer). Use "KapiTeacher-Mongo" quando o tamanho não importar.
2. **Escolha 14B × 7B** no bake-off da Fase 01 (novo job de sonda): zero-shot FIM MongoDB nos três (DeepSeek 6.7B, Qwen 7B, Qwen 14B) + **sonda de QLoRA de 1–2 h** em cada Qwen medindo tok/s e VRAM dedicada sem offload para memória compartilhada.
   - Regra: **14B** se (a) VRAM pico ≤ 14 GB com micro-batch ≥ 4k tokens sem paginação, (b) throughput ≥ ~250 tok/s e (c) vantagem zero-shot ≥ 3 p.p. de FIM-acc ou exec-valid sobre o 7B. Senão, **7B**.
3. **Por que** (registrar como justificativa, números marcados como [F a confirmar no relatório técnico do Qwen2.5-Coder, arXiv 2409.12186]): HumanEval base ~47,6 (DS-6.7B) × ~61,6 (Qwen 7B) × ~64 (Qwen 14B); FIM nativo (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`), contexto 32k; licença Apache-2.0 elimina a cláusula de "Derivatives" da licença DeepSeek; **mesmo tokenizer dos students T0/T1/T3** (Qwen2.5-Coder) → KD por logits passa a valer para eles; um único contrato de prompt/adapter (`qwen2`) na IDE para todos os tiers de autocomplete.
4. **Fatos de arquitetura (conferir no `config.json` na verificação do checkpoint; marcar [F a confirmar])**:
   - 14B: 48 camadas, hidden 5120, 40 heads de atenção, **8 KV heads** (GQA), head_dim 128, vocab_size 152.064, 32k contexto, rope_theta 1e6, embeddings **não** atrelados.
   - 7B: 28 camadas, hidden 3584, 28 heads, **4 KV heads**, vocab_size 152.064, 32k.
   - Students 0.5B/1.5B: vocab_size 151.936 (embeddings atrelados). O tokenizer é o mesmo (~151.665 tokens reais); para KD, restringir logits às entradas reais do tokenizer e mascarar as linhas de padding de professor e aluno.
   - IDs FIM do Qwen2.5-Coder: `<|fim_prefix|>` 151659, `<|fim_middle|>` 151660, `<|fim_suffix|>` 151661, `<|fim_pad|>` 151662, `<|endoftext|>` 151643 [F a confirmar]. O contrato da IDE é o `QwenFimPromptBuilder` (já existente).
5. **Custos estimados [H] na RX 7800 XT** (QLoRA NF4, gradient checkpointing, CE em blocos):
   - 14B: pesos NF4 das camadas ~7 GB + embedding e `lm_head` em bf16 ~3,1 GB (bitsandbytes não quantiza) ≈ 10 GB; pico com LoRA r=64 e micro-batch 4–8k ≈ 13–14,5 GB (**no limite**); throughput ~250–300 tok/s (≈ 560 × 6,7/14,7). CPT 15–20 M tokens ≈ 15–22 h; SFT ~47 M tokens ≈ 45–52 h por época. Forward para scoring/top-k ≈ 900 tok/s.
   - 7B: ≈ 5,6 GB de pesos (camadas NF4 + embed/lm_head bf16 ~2,2 GB), pico ~10–11 GB; ~500–560 tok/s (igual ao DeepSeek 6.7B medido na v1); forward ~1.900 tok/s.
   - **Scoring e cache top-k numa única passada de forward** sobre o pool do T1 (~110 M tokens): 14B ≈ 34 h; 7B ≈ 16 h.
   - Impacto no calendário (Fase 05 §9): com **14B** ≈ +60–75 h de GPU no plano base (≈ 540–575 h no total); com **7B** ≈ +5–15 h (KD agora no plano base).
6. **KD por logits entra no plano base para T0 e T1** (e T3 herda os pesos do T0/T1). Cache `topk-v1` revisado: IDs **uint32** (vocab > 65.536), k=32, logprobs fp16, massa residual fp16 → 32×4 + 32×2 + 2 = **194 B/posição**; só posições de resposta; ~230 mil exemplos × ~25 tokens ≈ 5,8 M posições ≈ **1,1 GB**. Perda: α·CE + (1−α)·τ²·KL (forward KL no autocomplete, τ=1–2, α 0,3–0,5 inicial).
   - O **student DeepSeek-Coder-1.3B (`T1-ds`) e o braço C do piloto saem do plano** (motivo: a vantagem de mesmo tokenizer agora existe nos Qwen). O piloto de destilação da Fase 01 vira: **A** = Qwen 1.5B só sequence-level; **B** = A + KD por logits do KapiTeacher (mesmo tokenizer). GOLD/ULD (KD entre tokenizers) deixa de ser necessário.
7. **T2 Pro = o próprio KapiTeacher** (continua a regra "economizar um treino"): se 14B → T2 = KapiTeacher-Mongo-14B **DML INT4** (~8,5–9 GB + KV ~192 KiB/token → ~0,8 GB a 4k; cabe em 16 GB; TTFT maior que o 7B, a medir); se 7B → T2 = KapiTeacher-Mongo-7B DML INT4 (~5 GB). **Adapter `qwen2` já existente** (sai o caminho DeepSeek da IDE para o T2). Gate G2.8 (papel de T2) continua.
8. O que **não muda**: T4 = Qwen3-8B híbrido com raciocínio; professor de raciocínio/instrução auxiliar = Qwen3-14B local no llama.cpp (Qwen3-30B-A3B opcional); geradores programáticos + validação por execução; Agent-Lite; NPU no Vivobook X1-26-100; documentação MongoDB fora do treino; treino 100% local.
9. **Licença**: todos os modelos treinados ficam Apache-2.0 (bases Qwen2.5-Coder/Qwen3). Remover a obrigação de "Anexo A da licença DeepSeek" dos pacotes; manter atribuição e aviso de modificação exigidos pela Apache-2.0.
10. Llama.cpp: o merged do professor é convertido para GGUF Q8_0 (14B ≈ 15,7 GB → **não cabe inteiro com folga**; usar **Q6_K (~12 GB)** ou Q5_K_M para geração em lote com o 14B, validando top-1 ≥ 97% contra o HF NF4; 7B usa Q8_0 ~8 GB).

## Regras de edição
- Edições coerentes com o estilo, marcações [F]/[H] e numeração existentes. Registrar no topo de cada fase alterada uma linha "**Revisão 2026-10-08b — troca do professor:** …" resumindo o que mudou.
- Não editar fora de `C:\IA\models\KapibaraStudio.MongoIA\docs`. Não editar o repositório da IDE.

## Estrutura de pastas (decisão do usuário, 2026-10-08) — vale para todas as fases

- Modelos baixados do Hugging Face (bases e professores): `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\<nome-do-repo>\` (ex.: `BaseRepos\Qwen2.5-Coder-14B\`, `BaseRepos\Qwen3-8B\`, `BaseRepos\Qwen3-14B-GGUF\`, `BaseRepos\deepseek-coder-6.7b-base\`), com `BaseRepos\base-repos.lock.json`.
- Cada modelo criado tem workspace próprio: `C:\IA\models\KapibaraStudio.MongoIA\Repos\<modelo>\` (`KapiTeacher-Mongo-14B`, `KapiCoder-Mongo-Nano`, `-Small`, `-NPU`, `-Agent`, `-Agent-Lite`) com `configs\ adapters\ checkpoints\ merged\ gguf\ exports\ reports\ README.md LICENSE model.lock.json`.
- Substituem as antigas pastas raiz `teacher\`, `students\`, `exports\`, `artifacts\`, `models\` (bases.lock.json → `BaseRepos\base-repos.lock.json`). Continuam na raiz: `docs\ configs\ scripts\ contracts\ data\ bakeoff\ tools\ sim\ cache\ reports\ env\`.
- Referência completa: [Fase 00 — Estrutura de pastas](fase-00-visao-premissas.md#estrutura-de-pastas-do-workspace).
