# Fase 05 · Treinamento e destilação por tier

> Receitas de treino dos students KapiCoder-Mongo (T0…T4) na RX 7800 XT, com estimativas medidas na v1 e critérios de saída.
> Convenção: **[fato]** = verificado em arquivo/URL citado · **[proposta]** = decisão deste plano · **[hipótese]** = a validar com sonda/medição.
> **Revisão 2026-10-08 (requisito definitivo): sem orçamento para GPU na nuvem.** Treino, geração de dados, destilação, avaliação, calibração e exportação rodam **só** na máquina local (Ryzen 9 7900 12 núcleos, 64 GB DDR5-4800, RX 7800 XT 16 GB com o venv ROCm da v1; discos C: ~830 GB e E: ~1,6 TB livres; **F: não recebe dados nem checkpoints**). Uma GPU para todos os jobs: o calendário único está na §9.
> **Revisão 2026-10-08b — troca do professor:** o professor passa a ser o **KapiTeacher-Mongo-14B** (Qwen2.5-Coder-14B Base; alternativo **-7B**, decidido pela sonda **J03b** da [Fase 01 §6.6](fase-01-selecao-modelos-base.md)) e tem o **mesmo tokenizer** dos students Qwen. Efeitos nesta fase: **KD por logits entra no plano base de T0/T1** (§6.3: cache `topk-v1` uint32, 194 B/posição, gerado na mesma passada do scoring; perda `α·CE + (1−α)·τ²·KL`); `T1-ds`/`T0-ds` e o opcional O3 saem; **T2 = KapiTeacher 14B ou 7B** (adapter `qwen2`); §9 refeita: plano base ≈ **540–575 h** de GPU com o 14B (≈ **480–515 h** com o 7B), novo job J03b, scoring + top-k numa só passada (J13), perda KD ≈ +10% no treino de T0/T1, e "trocar 14B por 7B" como corte cedo da ordem de corte (§9.5). As medições v1 do DeepSeek-6.7B (§3) continuam como fatos históricos e base das extrapolações.

> **Resultado 2026-10-09 (J03b):** o professor escolhido é o **Qwen2.5-Coder-7B → KapiTeacher-Mongo-7B** (o 14B não coube com micro-batch ≥ 4k e treinou a 75 tok/s; [Fase 01 §6.6](fase-01-selecao-modelos-base.md)). Vale o cenário **7B** deste documento; o 7B mediu **309–360 tok/s** (não 500–560), então as horas de CPT/SFT do 7B aqui devem ser multiplicadas por ~1,5.

## 1. Objetivo

Transformar os modelos base escolhidos na [Fase 01](fase-01-selecao-modelos-base.md) em checkpoints especializados por tier, usando:

- o professor adaptado KapiTeacher-Mongo-14B (ou -7B; [Fase 02](fase-02-professor.md)) para destilação sequence-level e KD por logits (mesmo tokenizer Qwen2.5-Coder dos students T0/T1/T3);
- os datasets de autocomplete/chat validados ([Fase 03](fase-03-dataset-autocomplete-validacao.md)) e de agente/tools ([Fase 04](fase-04-dataset-agente-tools.md));
- um processo reprodutível (manifests, versões pinadas, regras de adoção registradas antes do treino), entregando checkpoints mesclados para a [Fase 06](fase-06-avaliacao-benchmarks.md) (avaliação) e a [Fase 07](fase-07-exportacao-onnx.md) (ONNX).

## 2. Entradas

| Entrada | Origem | Observação |
|---|---|---|
| Modelos base por tier (revisão HF fixada por SHA) | [Fase 01](fase-01-selecao-modelos-base.md) | decisão da Fase 01: T0 Qwen2.5-Coder-0.5B Base (alt. Qwen3-0.6B Base); T1 Qwen2.5-Coder-1.5B Base (alt. Qwen3-1.7B Base); piloto de KD A × B (A sequence-level; B + KD por logits do KapiTeacher); T2 **KapiTeacher-Mongo-14B ou -7B** (o próprio professor adaptado, sem treino extra nesta fase; alt. KapiTeacher-7B treinado à parte, **condicionado ao calendário**, §7); T3 = pesos de T1/T0; T4 **Qwen3-8B híbrido com thinking ligado por padrão** (desligável, orçamento 1.024/máx. 4.096, contexto 16k; alt. Qwen3-4B-Thinking-2507; fallback sem raciocínio Qwen2.5-Coder-7B-Instruct), Agent-Lite Qwen3-1.7B (thinking curto opcional) / 0.6B (sem thinking) |
| KapiTeacher-Mongo-14B (ou -7B; merged bf16 + NF4 + GGUF Q6_K/Q8_0) e cache `topk-v1` | [Fase 02](fase-02-professor.md) | contrato FIM Qwen (`<|fim_prefix|>`/`<|fim_suffix|>`/`<|fim_middle|>`); não faz chat livre nem tool calling |
| `data/tiers/<ver>/<tier>/` e `data/splits/<ver>/val/` + manifest com SHA-256 | [Fase 03](fase-03-dataset-autocomplete-validacao.md) | FIM, completion, chat, correção; peso `weight` e `label_source` da destilação |
| `data/agent/sft/{train,val}.jsonl`, `data/agent/lite/`, `data/agent/dpo/`, `data/agent/general_thinking/` | [Fase 04](fase-04-dataset-agente-tools.md) | trajetórias multi-turno no formato de tool calling do template do modelo, com `reasoning_content` (70% com thinking), amostras irmãs sem thinking e dados gerais com thinking para preservação ([Fase 04 §5.6, §6.5](fase-04-dataset-agente-tools.md)) |
| Contratos de prompt extraídos da IDE (`contracts/*.json`, hash) | [Fase 03](fase-03-dataset-autocomplete-validacao.md) / [Fase 08](fase-08-integracao-kapibara.md) | o treino usa exatamente o contrato do código atual (lição v1) |
| Pipeline v1 (somente leitura) | `C:\IA\models\SlopStudio.MongoAI-Compact\scripts\train_student.py`, `compact\env.py` | base do novo `train_sft.py` (CE em blocos, teto de VRAM, checkpoints atômicos, fingerprint) |

## 3. Linha de base medida na v1 (fatos usados nas estimativas)

Todos medidos na mesma máquina (Ryzen 9 7900 + RX 7800 XT 16 GB, Windows 11, ROCm nativo).

| Execução | Método | Dados | Throughput | VRAM pico | Duração | Fonte |
|---|---|---|---:|---:|---:|---|
| Qwen2.5-Coder-0.5B | LoRA r=64/α=128, base bf16, micro-batch 16k tokens | 100k ex. × 2 épocas, 22,7 M tokens/época (2,48 M de resposta) | ~6,8 k tok/s | 6,95 GB | 2,01 h (3.150 passos) | `docs\05-treino.md` |
| Qwen2.5-Coder-0.5B v1.1/v1.2 | continuação, lr 6e-5 | 34k / 64k ex. | ~6,7 k tok/s | 5,6 GB | 23 / 45 min | `docs\05-treino.md` |
| Qwen2.5-Coder-1.5B-full | LoRA r=64, continuação 1 época | 100k ex., 22,7 M tokens | ~2,4 k tok/s | 8,9 GB | 2,82 h (1.699 passos) | `adapters\student-1.5b-full\training_summary.json` |
| DeepSeek-Coder-6.7B (professor v1) | QLoRA NF4 double-quant, r=32/α=64, micro-batch 12.288 tokens × acum. 4 | 50k ex. × 2 épocas, 14,8 M tokens/época | ~560 tok/s | 10,18 GB | 14,78 h (2.084 passos) | `C:\SlopStudio.MongoAI-artifacts\adapters\mongodb-autocomplete-v1\training.yaml` e `training_summary.json` |
| Professor 6.7B NF4, scoring (forward) | — | 4,56 M tokens | ~1.970 tok/s | — | 38,6 min | `docs\04-destilacao.md` |
| Professor 6.7B NF4, geração greedy | lotes de 16, 64 tokens novos | 1.500 prompts | 0,5–0,9 ex./s | — | ~47 min | `docs\04-destilacao.md` |
| *Referência externa (outra unidade da mesma placa)*: llama.cpp, Llama-2-7B Q4_0, RX 7800 XT 280 W, Windows | `llama-bench` ROCm / Vulkan | pp512 / tg128 | ROCm: ~2.150–2.300 / ~96–101 tok/s; Vulkan: ~2.060–2.150 / ~96 tok/s (fluxo único) | — | — | [llama.cpp discussion #15021](https://github.com/ggml-org/llama.cpp/discussions/15021) **[fato externo]** |

A última linha é a base das estimativas de **geração em lote** deste plano (rollouts, best-of-N, avaliação de checkpoints): o HF Transformers em NF4 gera a 0,5–0,9 ex./s, lento demais para dezenas de milhões de tokens; o servidor do llama.cpp com slots paralelos (`-np`) e cache de prompt é o motor de inferência em lote local (§4.2, opção C).

**Extrapolações para o KapiTeacher Qwen (revisão 2026-10-08b) [hipótese; a sonda J03b e `probe_throughput.py` as confirmam]** a partir da linha do professor v1 (6,7B: 560 tok/s, 10,18 GB, forward ~1.970 tok/s): **14B** QLoRA NF4 ≈ **250–300 tok/s** (≈ 560 × 6,7/14,7), pesos NF4 das camadas ~7 GB + embedding e `lm_head` em bf16 ~3,1 GB (bitsandbytes não quantiza) ≈ 10 GB, pico com LoRA r=64 e micro-batch 4–8k ≈ **13–14,5 GB (no limite)**; CPT 15–20 M tokens ≈ 15–22 h; SFT ~47 M tokens ≈ 45–52 h por época; forward ≈ 900 tok/s. **7B** ≈ 5,6 GB de pesos (camadas NF4 + embedding/`lm_head` bf16 ~2,2 GB), pico ~10–11 GB, ~500–560 tok/s (= DeepSeek 6.7B medido), forward ~1.900 tok/s. **Scoring + cache top-k numa única passada de forward** sobre o pool do T1 (~110 M tokens): 14B ≈ 34 h; 7B ≈ 16 h.

Lições da v1 que viram regra nesta fase **[fato]** (`docs\02-ambiente-gpu.md`, `docs\05-treino.md`):

1. **Memória compartilhada do Windows = offload silencioso.** Sem teto, o 0.5B a 16k tokens/micro-batch "usou" 28,6 GB numa placa de 16 GB; o professor bf16 caiu para < 1 ex./s. Teto obrigatório `min(fração × total, livre − reserva)` → erro OOM explícito.
2. **Cross-entropy em blocos de 1.024 posições** com recomputação: o vocabulário Qwen (151.936) torna os logits fp32 maiores que a VRAM.
3. **Gradient checkpointing** sempre ligado nos students.
4. **Loss só nos tokens da resposta**, ponderada por `weight`.
5. **Mais chat no mix degrada grupos pequenos do autocomplete** (Indexes/Scripts/Extended JSON: −0,20 APT, IC95% [−0,39; −0,04] somando v1.1 e v1.2). Chat + correção ≤ 25% do mix; replay estratificado por grupo.
6. **Famílias sintéticas não generalizam para pedidos livres** (cego 22/40 → 22/40 na v1.2); o que moveu o chat livre foi mais orçamento de treino no 1.5B (cego 25 → 29/40). Diversidade real de instruções vem da [Fase 03](fase-03-dataset-autocomplete-validacao.md), não de mais épocas.
7. **Regras de adoção registradas antes do treino** (`selection.yaml → rounds.*`) evitaram adotar rodadas sem ganho mensurável. Mantido.

## 4. Ambiente

### 4.1 Estado do ROCm para a RX 7800 XT (gfx1101) em 2026-10

| Item | Situação | Fonte |
|---|---|---|
| Windows nativo, matriz oficial AMD | ROCm 7.2.1 · PyTorch 2.9 · Python 3.12 · Windows 11; arquiteturas `gfx1201, gfx1200, gfx1100, gfx1101` (a RX 7800 XT não aparece nominalmente; RX 7700 sim, mesmo gfx1101). A página afirma que **o stack ROCm completo ainda não é suportado no Windows** — o PyTorch traz componentes ROCm embutidos. | [Windows support matrices](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html) **[fato]** |
| Matriz unificada (preview) | lista RX 7800 XT / gfx1101 com Windows 11 25H2 | [ROCm 7.10.0 preview compatibility matrix](https://rocm.docs.amd.com/en/7.10.0-preview/compatibility/compatibility-matrix.html) **[fato, preview]** |
| WSL2 | RX 7800 XT listada; ROCm 7.2.1 com PyTorch 2.9.1, Ubuntu 22.04/24.04, Adrenalin 26.1.1; caminho novo via ROCDXG | [WSL support matrices](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html), [WSL how-to](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html) **[fato]** |
| bitsandbytes no ROCm | 0.50.x publica wheels ROCm (inclusive Windows) e declara suporte a RDNA de consumo | [bitsandbytes 0.50.0 installation](https://huggingface.co/docs/bitsandbytes/v0.50.0/installation) **[fato]** |
| Runtime realmente validado na v1 | `torch 2.10.0+rocm7.14.0a20260611` (HIP 7.14.60850, nightly TheRock), Transformers 4.57.6, PEFT 0.17.1, Accelerate 1.12.0, bitsandbytes 0.50.2 (`libbitsandbytes_rocm714.dll`), Python 3.12.10, driver 32.0.31041.1004; AOTriton experimental ligado | `reports\env\environment.json` **[fato]** |
| Atenção eficiente (SDPA flash/mem-efficient) em gfx1101 | O SDPA do PyTorch ROCm usa kernels do **AOTriton**. As notas de versão tiram o gfx1100 do estado experimental na 0.12b e publicam um pacote de imagens `gfx110x` na 0.14.2b; **o gfx1101 não é citado nominalmente**. No Windows o AOTriton não compila imagens sozinho (usa as de um build Linux). A v1 rodou com `attn_implementation: sdpa` e a variável de AOTriton experimental ligada, mas **não registrou qual kernel foi despachado** (sequências curtas, média 227 tokens). Alternativas: backend Triton do FlashAttention (CDNA e RDNA) via `triton-windows` (comunitário, experimental) ou WSL2 com o PyTorch ROCm Linux oficial | [AOTriton releases](https://github.com/ROCm/aotriton/releases), [AOTriton README](https://github.com/ROCm/aotriton), [FlashAttention README](https://github.com/Dao-AILab/flash-attention), [guia comunitário ROCm 7 + Triton no Windows](https://github.com/guinmoon/rocm7-triton-flash-attention-sage-attention-bnb) **[fato]**; disponibilidade real no gfx1101/Windows **[hipótese]**, decidida pela sonda `probe_attention.py` (§6.4c) |

### 4.2 Decisão de ambiente **[proposta]**

| Opção | Uso | Motivo |
|---|---|---|
| **A. Windows nativo com o venv ROCm da v1, clonado e pinado** (primária) | SFT, KD, QLoRA, DPO/ORPO, QAT simulada | É a única combinação já medida ponta a ponta (treino 0.5B/1.5B/6.7B). Não atualizar no meio do programa. |
| B. WSL2 Ubuntu 24.04 + ROCm 7.2.1 + PyTorch 2.9.1 (oficial) | fallback quando uma biblioteca depender de Triton/kernels Linux (ex.: Liger, torchao QAT, TRL com kernels) **e plano B da atenção eficiente do T4 a 16k** (§6.4c) | Matriz oficial lista a placa **[fato]**; desempenho e offload de memória em WSL2 **[hipótese]**: medir com `probe_throughput.py` antes de usar. |
| C. **llama.cpp** (binários Windows `win-rocm` ou `win-vulkan` das [releases](https://github.com/ggml-org/llama.cpp/releases), versão fixada no manifesto) | inferência **em lote**: rollouts de raciocínio e professor auxiliar (Fase 04), greedy/best-of-N do professor (Fase 02), dados gerais com thinking, pares DPO, avaliação intermediária de checkpoints (Fase 06) | `llama-server -np 8–16 --cont-batching` com cache de prompt por slot; modelos convertidos para GGUF a partir do merged: **Q8_0** para o 7B (~8 GB) e os students/T4; para o **KapiTeacher-14B**, Q8_0 (~15,7 GB) **não cabe com folga** em 16 GB → **Q6_K (~12 GB)** ou Q5_K_M para a geração em lote (≈ bf16 em qualidade **[hipótese]**: paridade de top-1 ≥ 97% contra HF NF4+adapter em 200 prompts antes de usar). Throughput de referência na §3 **[fato externo]**; vazão agregada com slots paralelos **[hipótese]**, medida por `probe_gen_throughput.py`. Não há GPU na nuvem em nenhuma etapa. |

Regras:

- [ ] `env/requirements-train-rocm-win.lock` (gerado por `uv pip compile`/`pip freeze`) com hashes; `env/llama.cpp.lock` com tag do build, backend (ROCm/Vulkan) e SHA-256 dos binários.
- [ ] `scripts/check_env.py` (portado da v1): exige GPU esperada por nome (ignora a iGPU que o ROCm expõe como `cuda:1`), `torch.version.hip` ou CUDA, matmul fp32/bf16; sai com código 2. Sem fallback CPU.
- [ ] `limit_vram`: `max_vram_fraction: 0.94`, `vram_reserve_gb: 1.6` → teto ~14,8 GB na RX 7800 XT (v1: 14,84 GB). Nenhum outro job de GPU em paralelo (lição: o best-of-N v1 foi empurrado para memória compartilhada por um benchmark paralelo). Trava de GPU por arquivo (`tmp/gpu.lock`) em todos os scripts.
- [ ] Caches (`HF_HOME`, `TRITON_CACHE_DIR`, `MIOPEN_USER_DB_PATH`, `TMP`) dentro do workspace (`cache\`, `tmp\`). Modelos baixados em `BaseRepos\` e artefatos grandes de cada modelo (checkpoints, merged, GGUF, ONNX) no workspace dele em `Repos\<modelo>\` ([Fase 00](fase-00-visao-premissas.md#estrutura-de-pastas-do-workspace)), tudo em C: (estimativa de disco na §9.7; se faltar espaço, `cache\` e `cache\kd\` vão para E: por junção `mklink /J`); **nada em F:** (24 GB livres).
- [ ] Toda execução de GPU passa pela fila `scripts/gpu_queue.ps1` (§9.6), que detém `tmp/gpu.lock`; llama.cpp, treino, avaliação DML e benchmarks nunca rodam ao mesmo tempo.

### 4.3 Reprodutibilidade **[proposta]**

| Item | Como |
|---|---|
| Seeds | `seed` por run no YAML; plano de lotes determinístico por `random.Random(f"{seed}-plan-{epoch}")` e decisões por exemplo via hash estável do `id` (como `stable_unit` da v1); `torch.manual_seed`, `numpy`, `random`. |
| Determinismo de kernel | ROCm/SDPA não garantem determinismo bit a bit **[hipótese]**: tolerância documentada — repetir 200 passos de um run deve reproduzir a loss com desvio relativo < 0,5%; medido uma vez por ambiente (`scripts/repro_check.py`). |
| Manifest de run | `Repos\KapiCoder-Mongo-<tier>\<run>\run_manifest.json`: SHA-256 do config, dos JSONL de treino/validação e do contrato; revisão do base; lockfile; driver/ROCm/HIP; nome da GPU; teto de VRAM aplicado; git hash do workspace; horário de início; regra de adoção (hash de `configs/gates.yaml`). |
| Retomada | `--resume auto` só aceita checkpoint com fingerprint idêntico (config, dados, revisão do base) — já implementado na v1. |
| Regras pré-registradas | `configs/rounds.yaml` com a regra de adoção de cada rodada, commitado **antes** do treino (hash no manifest). |

## 5. Visão geral dos estágios por tier

| Estágio | T0 Nano | T1 Small | T2 Pro | T3 NPU | T4 Agent |
|---|---|---|---|---|---|
| (1) Adaptação contínua de domínio (DAPT) | não | opcional | = CPT do professor ([Fase 02 §3.3 A](fase-02-professor.md)) | herda do T0/T1 | não (parte do instruct) |
| (2) SFT/destilação sequence-level FIM + chat | **sim** | **sim** | = SFT do KapiTeacher (Fase 02 §3.3 B); nada extra treina aqui | **sim** (= receita T0/T1) | sim (parte do mix) |
| (3) KD por logits do KapiTeacher (mesmo tokenizer Qwen) | **sim, plano base** (gate do braço B, Fase 01 §6.5) | **sim, plano base** | não (é o próprio professor) | herda os pesos de T0/T1 | não |
| (4) SFT de tool calling | não | não | não | não | **sim** |
| (4b) SFT de raciocínio (thinking/no_think, 16k) | não | não | não | não | **sim** (no mesmo run do estágio 4; Agent-Lite 1.7B: thinking curto) |
| (5) Preferência DPO/ORPO por execução | não (custo > ganho em autocomplete) | opcional | opcional | não | opcional (pares de trajetória verificados por execução + penalidade de raciocínio longo) |
| (6) QAT / ajuste para INT4 | se o INT4 reprovar na Fase 06 | se o INT4 reprovar | **sim** (o INT4 DML é o alvo de 16 GB) | **sim, w4a16** | sim, se INT4 |

Ordem de execução **[proposta]**: T2 = KapiTeacher (14B ou 7B) sai pronto da Fase 02 (só exportação/avaliação aqui) → T0 → T1 (reaproveita receita, risco baixo) → T3 sonda de viabilidade (em paralelo, CPU/NPU) → T4 → Agent-Lite. O T4 só começa com T1 aprovado na Fase 06 e com o dataset da Fase 04 congelado. Ordem e horas de GPU de todos os jobs: §9.

## 6. Receitas por estágio

### 6.1 Estágio 1 — Adaptação contínua de domínio (opcional)

Objetivo: dar ao student vocabulário MongoDB/mongosh/EJSON/DSL do Kapibara antes do SFT. Na v1 não houve DAPT; o ganho é **[hipótese]**, justificada só se a sonda mostrar perplexidade do base alta no corpus de domínio.

| Item | Valor inicial |
|---|---|
| Corpus | `data/dapt/` (montado como o corpus CPT da [Fase 02 §3.3](fase-02-professor.md)): código MongoDB com licença compatível (sem texto da documentação oficial, CC BY-NC-SA — Fase 03 §3), scripts mongosh, DSL do Kapibara, exemplos EJSON; 50–150 M tokens |
| Replay geral | 30% de código geral (ex.: subconjunto licenciado de The Stack v2 / dados de pré-treino públicos) para não esquecer |
| Formato | documentos inteiros com EOS, **packing** em sequências de 2.048 (aqui o packing é seguro: loss em todos os tokens) e 50% das sequências convertidas em FIM (PSM) com os tokens do base |
| LR | LoRA r=128: 1e-4; full FT (T0): 3e-5; warmup 1%, cosseno até 10% |
| Parada | 1 época; adotar se a perplexidade de domínio cair ≥ 15% e a de código geral subir ≤ 3% |
| Script | `scripts/train_dapt.py` (ou `train_sft.py --objective lm --packing`) |

Gate de entrada **[proposta]**: rodar `scripts/probe_domain_ppl.py` no base; se o ganho de perplexidade em 10 M tokens de sonda for < 10%, pular o estágio.

### 6.2 Estágio 2 — SFT / destilação sequence-level (autocomplete FIM + chat)

Mecânica herdada da v1 **[fato]**: o professor pontua (log-prob da referência), gera greedy e best-of-N, confirma/pondera (peso 0,85 na discordância), filtra os 5% mais imprevisíveis e adiciona variações; política `replace_policy: none` (trocar rótulo pelo log-prob piorava). Na v2 a destilação é executada na [Fase 02](fase-02-professor.md)/[Fase 03](fase-03-dataset-autocomplete-validacao.md); esta fase consome o JSONL resultante.

Formato de exemplo (JSONL de treino, já renderizado no contrato do student) **[proposta]** — ilustrativo: o texto real é gerado por `scripts/data/render_contract.py` nos contratos `editor-context-v1`/`local-agent-fim-v0` da [Fase 03 §2](fase-03-dataset-autocomplete-validacao.md) (o chat local é FIM com sufixo vazio, sem papel `system`):

```json
{"id":"fim-000123","task":"fim","group":"aggregation","lang":"mongosh","weight":1.0,"label_source":"reference+teacher_agree",
 "prompt":"<|fim_prefix|>// conn: local › shop › orders\n// schema: {_id:ObjectId, status:string, total:double, createdAt:date}\ndb.orders.aggregate([\n  { $match: { status: 'paid' } },\n  { $group: { _id: '$customerId', <|fim_suffix|> } }\n])<|fim_middle|>",
 "completion":"total: { $sum: '$total' }<|endoftext|>"}
{"id":"chat-004511","task":"chat_edit","group":"pt_chat","lang":"pt-BR","weight":1.0,"label_source":"generator_validated",
 "messages":[{"role":"system","content":"<contrato de chat extraído da IDE>"},{"role":"user","content":"só os pedidos pagos de março, do mais caro pro mais barato"}],
 "completion":"db.orders.find({ status: 'paid', createdAt: { $gte: new Date('2026-03-01'), $lt: new Date('2026-04-01') } }).sort({ total: -1 })"}
```

Mistura de dados por tier **[proposta]** (proporção em exemplos; v1 usava 55/25/15/5 FIM/completion/chat/correção):

| Componente | T0 | T1 | T2 | T3 | T4 (estágio 4) |
|---|---:|---:|---:|---:|---:|
| FIM (PSM, com cabeçalho de contexto 92% / sem 8%) | 62% | 50% | 45% | = T0 ou T1 | 5% |
| Completion (sufixo vazio) | 28% | 22% | 20% | idem | — |
| Chat/edição curta (pt-BR/en ~50/50) | 5%¹ | 18% | 22% | idem | 15% |
| Correção | 2% | 5% | 6% | idem | — |
| Replay de código geral / instruções gerais | 3% | 5% | 7% | idem | ver §6.4 |

¹ T0 é só autocomplete; os 5% de chat preservam o contrato caso a IDE ofereça chat no Nano, sem prometer qualidade.

Regras de mistura:

- Chat + correção ≤ 25% em T0–T2 (lição 5). Grupos pequenos (Indexes, Scripts, Extended JSON, DSL) com **piso** de 4% cada no autocomplete, independentemente da amostragem.
- Curriculum **[proposta]**: época 1 com distribuição natural; na última 20% dos passos, sobre-amostrar 1,5× os exemplos `hard` (baixa concordância do professor) e os grupos com pior APT na validação intermediária. Sem curriculum por comprimento (o length grouping já agrupa).
- **Volume dimensionado ao orçamento de GPU** **[proposta]**: T0 ~160k exemplos (~62 M tokens/época), T1 ~230k (~100 M), T4-chat 10k ⊂ T1; o T0 é subconjunto do T1 para que um só *scoring* do professor sirva aos dois ([Fase 03 §4.6](fase-03-dataset-autocomplete-validacao.md)). Amostragem **estratificada** por categoria × papel de cursor × idioma, com sobre-amostragem dos exemplos `hard` (baixo logprob do professor) e dos grupos pequenos, em vez de mais épocas ou mais exemplos fáceis.
- Comprimento: `sequence_length` 2.048 e `context_tokens` 2.048 (contrato da IDE: contexto recomendado 2048, 32 tokens no autocomplete, 256 no chat — `kapibarastudio-model.json` **[fato]**, `docs\23-onnx-slopcoder.md`); resposta de chat até 240 tokens.
- **Sem packing no SFT** **[proposta]**: v1 usou length grouping (sequência média 227 tokens, p95 363) com bom aproveitamento; packing sem máscara bloco-diagonal contamina atenção entre exemplos, e SDPA varlen no ROCm/Windows não está validado. Reavaliar só se `probe_throughput.py` mostrar padding > 25%.

Hiperparâmetros iniciais:

| | T0 Nano (~0,5B) | T1 Small (~1,5B) | KapiTeacher-7B (treinado na Fase 02; referência de receita; à parte só no O4) | KapiTeacher-14B (plano base, Fase 02; ~250–300 tok/s, no limite de VRAM) |
|---|---|---|---|---|
| Método | **A/B: LoRA r=64 (linha v1) × full FT** | LoRA r=64, α=128 (v1); experimento: r=128 | QLoRA NF4 double-quant, r=64, α=128 | QLoRA NF4, r=32, α=64 |
| Módulos | q,k,v,o,gate,up,down | idem | idem | idem |
| Dropout | 0,05 | 0,05 | 0,05 | 0,05 |
| LR | LoRA 2e-4 · full 2e-5 | 1,5e-4 | 1e-4 | 8e-5 |
| Schedule | warmup 3%, cosseno até 10% | idem | idem | idem |
| Otimizador | AdamW (β 0,9/0,95, wd 0; full: wd 0,01) | AdamW | paged AdamW 8-bit (bnb) | paged AdamW 8-bit |
| Tokens/passo (efetivo) | 32.768 | 20.480–32.768 | ~49k (12.288 × acum. 4, como o professor v1) | ~32k (8.192 × acum. 4) |
| Micro-batch | 16.384 tokens | 16.384 tokens | 12.288 tokens (≤ 12 seq.) | 4.096–8.192 tokens (definido pela sonda J03b) |
| Épocas | 2 | 2 | 1 | 1 |
| Grad clip | 1,0 | 1,0 | 1,0 | 1,0 |
| Precisão | bf16 | bf16 | base NF4, compute bf16 | idem |
| Checkpointing / CE em blocos | sim / 1.024 | sim / 1.024 | sim / 1.024 | sim / 512 |

Notas:

- **LoRA × full FT.** T0 full FT cabe em 16 GB **[hipótese, estimativa]**: 0,5B × (2 B pesos bf16 + 2 B gradiente + 8 B estados Adam fp32) ≈ 6 GB + ativações (~4 GB a 16k tokens com checkpointing) ≈ 10–11 GB. T1 full FT exigiria ~18 GB só de estados → apenas com Adam 8-bit e ainda no limite; manter LoRA (v1 provou). Professor 7B e T4 (8B): QLoRA é a única opção local (o professor v1 com QLoRA r=32 usou 10,2 GB **[fato]**); **KapiTeacher-14B** (professor primário): pesos NF4 das camadas ~7 GB + embedding e `lm_head` em bf16 ~3,1 GB ≈ 10 GB, pico com LoRA r=64 e micro-batch 4–8k ≈ 13–14,5 GB (**no limite**) **[hipótese]**, confirmado ou derrubado pela sonda J03b (regra 14B × 7B na Fase 01 §6.6); r=32 se apertar.
- **Ponto de partida:** T0/T1/T2 partem do **base** (FIM nativo); a variante de chat do T2 pode partir do instruct se a Fase 01 preferir — nesse caso o FIM precisa existir no instruct (Qwen2.5-Coder-Instruct mantém os tokens FIM; qualidade de FIM do instruct **[hipótese]** a sondar).
- Avaliação intermediária a cada ~10% dos passos: loss de validação (1.000 ex.) + geração em 200 ex. (v1 usava 60 e oscilava demais) + 40 casos de chat escritos à mão **não-cegos** (o cego nunca é usado durante o treino).

### 6.3 Estágio 3 — KD por logits do KapiTeacher (T0 e T1, plano base)

*Revisão 2026-10-08b: o professor é Qwen2.5-Coder-14B (alt. 7B) e tem o mesmo tokenizer dos students T0/T1 (T3 herda os pesos). O estágio deixa de ser uma variante `-ds` condicional e entra no plano base; o student DeepSeek-Coder-1.3B (`T1-ds`) saiu do plano.* Também se aplica ao estágio 6 (auto-destilação bf16 → fake-quant), onde professor e student sempre compartilham tokenizer.

**KD offline com cache top-k** **[proposta]** — evita manter professor e student na mesma VRAM:

1. Cache gerado na [Fase 02 §3.5](fase-02-professor.md) por `scripts/teacher/distill.py topk` (`scripts/build_kd_cache.py` é só um atalho) **na mesma passada de forward do scoring** do pool do T1 (~110 M tokens): 14B ≈ **34 h**, 7B ≈ **16 h** (job J13, §9.2). Formato `topk-v1` revisado: **só nas posições da resposta**, top-k=32, IDs **uint32** (vocabulário > 65.536), logprobs fp16, massa residual fp16 → 32 × 4 + 32 × 2 + 2 = **194 B/posição**; ~230 mil exemplos × ~25 tokens ≈ 5,8 M posições ≈ **1,1 GB** em disco.
2. **Vocabulário** **[a confirmar no `config.json`/`tokenizer.json`]**: `vocab_size` 152.064 no professor e 151.936 nos students 0.5B/1.5B, mesmo tokenizer (~151.665 tokens reais). Restringir os logits às entradas reais do tokenizer e **mascarar as linhas de padding** do professor e do aluno antes do softmax; o top-k e o bucket "resto" são calculados sobre o suporte real.
3. `scripts/train_sft.py --kd-cache <dir>`: perda combinada por posição de resposta (**forward KL**):

   `L = α · CE(y) + (1 − α) · τ² · KL(p_T^τ ‖ p_S^τ)` restrita ao suporte top-k + bucket "resto" (massa residual).

| Hiperparâmetro | Inicial | Varredura |
|---|---|---|
| α (peso da CE com o rótulo validado) | 0,3–0,5 (0,5 no braço B do piloto) | {0,3; 0,5} no piloto, {0,3; 0,5; 0,7} só se o B vencer |
| τ (temperatura) | 1–2 (1,0 no B) | {1,0; 2,0} |
| Direção | **forward KL** (cobre a massa do professor; adequado a FIM curto e determinístico) | reverse KL / JSD(β=0,5) em chat (mode-seeking reduz alucinação em geração longa — [MiniLLM](https://arxiv.org/abs/2306.08543), [GKD](https://arxiv.org/abs/2306.13649)) |
| Peso por exemplo | `weight` da destilação (0,85 na discordância); exemplos `teacher_disagree` usam α=1 (só CE) — o professor discorda do rótulo validado | — |
| Chat | o professor **não** sabe chat livre (4,5% de intenção na v1 **[fato]**): KD só em FIM/completion; chat com CE pura | — |
| Custo | ≈ **+10%** no tempo de treino de T0/T1 (leitura do cache + KL no suporte top-k): T0 ~12 → ~13 h, T1 ~23 → ~26 h (J17/J18) | — |

Gate do estágio: a perda KD fica **ligada por padrão** em T0/T1 só se o braço B do piloto ([Fase 01 §6.5](fase-01-selecao-modelos-base.md)) vencer o A (B ≥ A + 0,15 APT ou chat ≥ A + 3 pp, sem perder > 3 pp em pt-BR); senão T0/T1 seguem em sequence-level e o cache apenas fica arquivado (custo já pago, é produto do scoring). Regra pré-registrada em `configs/rounds.yaml`. GOLD/ULD (KD entre tokenizers) **não é necessário**.

### 6.4 Estágio 4 — SFT de tool calling (T4)

| Item | Proposta |
|---|---|
| Ponto de partida | modelo **pós-treinado híbrido** do tier: **Qwen3-8B** (primário, Fase 01), que já tem template de tool calling **e** os dois modos de raciocínio; partir do base exigiria reensinar chat e thinking. Alternativa: Qwen3-4B-Thinking-2507 (só thinking). Qwen2.5-Coder-7B-Instruct só como fallback sem raciocínio. Agent-Lite: Qwen3-1.7B (thinking curto opcional) / Qwen3-0.6B (sem thinking) ([Fase 04 §9](fase-04-dataset-agente-tools.md)) |
| Formato | Hermes/Qwen `<tool_call>` + JSON + `</tool_call>` (quebra de linha ao redor do JSON) / `<tool_response>` ([Fase 04 §5](fase-04-dataset-agente-tools.md)), com as 11 ferramentas extraídas dos descriptors do registry (`kapibara-tool-catalog.json`): `list_connections`, `list_databases`, `list_collections`, `get_indexes`, `get_search_indexes`, `get_cached_schema`, `get_workspace_context`, `get_query_results`, `get_query_diagnostics`, `propose_file_edit`, `create_workspace_file`. `mongo_find`/`mongo_count`/`mongo_explain`/`get_collection_schema` e as escritas foram removidas (ADR-066) e só aparecem como negativos |
| Loss | só nos turnos do assistente (raciocínio, chamadas e respostas finais); saídas de ferramenta e system mascarados; pesos do raciocínio na §6.4b |
| Contexto | alvo de inferência **16.384 tokens** (definições de ferramentas + 3–8 passos com saídas truncadas + raciocínio do turno corrente; p95 da Fase 04 ≤ 13k). No treino local: **bucketing por comprimento** até 16k (≈ 85% das amostras ≤ 8k) e, se a sonda reprovar 16k, treino em 8k + fase curta de extensão — receita e impacto na inferência em §6.4c. Sem truncamento silencioso de trajetórias |
| Método | **local (única opção):** QLoRA NF4 double-quant r=64, α=128, lr 1e-4 (r=32 e lr 5e-5 se houver regressão geral ou do thinking nativo), **1 época** de ~80 M tokens, ~64k tokens/passo efetivo, gradient checkpointing com **offload das ativações para a RAM** a partir de 12k, MLP em blocos, CE só nas posições com loss — §6.4c |

Mistura do T4 **[proposta]** (proporção em tokens de treino; revisada para raciocínio e **redimensionada ao orçamento local**: 1 época ≈ 80 M tokens):

| Componente | % | Volume | Thinking | Função |
|---|---:|---|---|---|
| Trajetórias multi-turno Kapibara validadas por execução (Fase 04), inclusive S14–S16 | 53 | 8.000 × ~5,3k ≈ 42 M | 70% com / 30% sem (amostras irmãs) | tarefa principal |
| Chamada única + seleção de ferramenta, incluindo **"nenhuma ferramenta necessária"** | 15 | 4.000 × ~3k ≈ 12 M | 50/50 | precisão de seleção, evitar chamada desnecessária |
| Permissões/recusas: pedidos de escrita, `propose_file_edit` com confirmação, injeção em saída de ferramenta | 10 | 2.500 × ~3,2k ≈ 8 M | 50/50 | segurança ([Fase 06](fase-06-avaliacao-benchmarks.md) §segurança), inclusive no raciocínio |
| Chat/edição Mongo do estágio 2 (mesmo contrato) | 9 | 10.000 × ~0,7k ≈ 7 M | sem (bloco vazio) | não perder a capacidade de query |
| **Dados gerais com thinking auto-gerados** pelo Qwen3-8B (código/raciocínio com resposta verificável; [Fase 04 §6.5](fase-04-dataset-agente-tools.md)) | 8 | 3.000 × ~2k ≈ 6 M | com | **preservar o raciocínio nativo** |
| Instruções gerais + tool calling genérico com licença compatível | 5 | 3.000 × ~1,3k ≈ 4 M | sem | evitar esquecimento catastrófico do instruct |
| FIM | 0 | — | — | removido: o Qwen3 não tem FIM treinado confirmado e o autocomplete fica com T0–T2 (Fase 01 §10) |

≈ 50% dos tokens com thinking. Nenhum componente fica 100% num modo só, exceto o chat, para não colapsar a chave liga/desliga. Seleção por qualidade/dificuldade antes do corte de volume: S3/S14–S16 ≥ 15% das trajetórias, S1/S12 fáceis limitados a 10%, dedup de tarefas quase idênticas (mesma consulta-alvo normalizada) e preferência por trajetórias com recuperação de erro verificada.

### 6.4b Estágio 4b — SFT de raciocínio (T4), no mesmo run do estágio 4

| Item | Proposta |
|---|---|
| Renderização | `render_chat.py --template qwen3 --thinking-from-sample`: cada amostra é renderizada com `enable_thinking` = `thinking.enabled`; sequência **append-only** igual à do runtime ([Fase 04 §5.6](fase-04-dataset-agente-tools.md)); trajetórias multi-turno desdobradas por turno do usuário (raciocínio de turnos anteriores removido, como no template); 10% com `/think`/`/no_think` no texto do usuário |
| Máscara de loss | `<think>`, conteúdo do raciocínio, `</think>`, chamada e resposta = peso 1 nas amostras da fonte (a1) da Fase 04; raciocínio da racionalização (a2) e do professor (b) = 0,8; traços de template (c) = 0,5 (estilo repetitivo); ação sempre 1. Bloco vazio das amostras sem thinking = parte do prompt (peso 0). Passo errado de recuperação **e** seu raciocínio = 0. Na amostra de fechamento forçado, o texto de parada inserido pelo runtime = 0 (o modelo não o gera) e a ação seguinte = 1 |
| Orçamento | cada amostra traz `thinking.budget`; nenhum raciocínio de treino passa do orçamento padrão (1.024), exceto 10% com orçamento 2.048–4.096 (casos S14/S15 difíceis) e os 5% de fechamento forçado. O modelo aprende a ser conciso; o corte acontece no runtime |
| Mistura thinking/no_think | ver a tabela acima (≈ 50/50 em tokens; 70/30 nas trajetórias). Monitorar no eval intermediário: % de gerações com `<think>` não vazio quando `enable_thinking=true` sem chave branda (alvo ≥ 97%) e 0% quando desligado |
| Preservação do raciocínio nativo | ~8% dos tokens com dados gerais com thinking (§6.4); LoRA (não full FT); eval intermediário a cada 10% dos passos em 100 problemas gerais com thinking (subset do EvalPlus **não** usado no gate) e no BFCL-subset; parar/baixar lr se cair > 3 pp em relação ao base |
| Comprimento | buckets 2k/4k/8k/12k/16k (§6.4c), sem packing (máscara bloco-diagonal não validada no ROCm); CE em blocos de 1.024 só nas posições com loss |
| Amostragem no eval | thinking: T=0,6, top-p 0,95, top-k 20 (greedy desaconselhado pelo model card); sem thinking: T=0,7/top-p 0,8 (recomendação). 3 seeds nos dois modos, como na [Fase 06 §4.6](fase-06-avaliacao-benchmarks.md); greedy só como diagnóstico |
| Agent-Lite 1.7B | mesmo mecanismo, contexto 4.096, orçamento 256–512, 50% com thinking; LoRA r=64 local. 0.6B: só amostras sem thinking (bloco vazio) |

### 6.4c Receita local do T4 na RX 7800 XT (16 GB) — Qwen3-8B QLoRA com raciocínio

Arquitetura **[fato: `config.json` do Qwen3-8B, [Fase 01 §4.5](fase-01-selecao-modelos-base.md), [model card](https://huggingface.co/Qwen/Qwen3-8B)]**: 36 camadas, hidden 4.096, intermediate 12.288, 32 heads de query / 8 KV × 128, vocabulário 151.936, `tie_word_embeddings=false`, 32.768 tokens de contexto nativo (131k com YaRN).

| Item | Valor **[proposta]** | Motivo |
|---|---|---|
| Base | NF4 double-quant nos lineares (≈ 3,7 GB); embedding e `lm_head` em bf16 (2 × ≈ 1,24 GB) | mesma técnica do professor v1 (QLoRA, 10,2 GB a 12k tokens/micro-batch **[fato]**, §3) |
| LoRA | r=64, α=128 em `q,k,v,o,gate,up,down` (~175 M parâmetros); se a VRAM apertar: r=32 (−0,8 GB) ou só `q,k,v,o,down` | adapters pequenos o bastante para checkpoint a cada 30 min |
| Otimizador | paged AdamW 8-bit (bitsandbytes 0.50.2, já validado na v1), páginas na RAM de 64 GB; estados ≈ 0,35 GB | offload completo dos estados para a CPU ganharia < 1 GB e custaria passo de otimizador na CPU: não compensa |
| Gradient checkpointing | por camada, sempre; nos buckets 12k/16k os checkpoints das camadas são **descarregados para a RAM** de forma assíncrona (implementação própria ~30 linhas com `torch.autograd.graph.saved_tensors_hooks` e cópia não bloqueante para memória *pinned*), a mesma ideia do *offloaded gradient checkpointing* ([Unsloth long context](https://unsloth.ai/blog/long-context): +1,9% de tempo relatado) | a 16k os 36 checkpoints ocupam 36 × 16.384 × 4.096 × 2 B ≈ **4,8 GB**; na RAM, ~4,8 GB de ida e volta pelo PCIe 4.0 por micro-batch (~0,3–0,5 s em ~60 s de passo) **[estimativa]**; *pinned memory* no ROCm/Windows **[hipótese]** |
| MLP em blocos (*tiled MLP*) | `gate/up/down` em 4 fatias de sequência no bucket 16k (2 no 12k), com recomputação por fatia | tensor intermediário a 16k: 16.384 × 12.288 × 2 B ≈ 0,4 GB, ~4 vivos no backward; técnica descrita em [Unsloth 500K context](https://unsloth.ai/docs/blog/500k-context-length-fine-tuning) |
| Cross-entropy | `lm_head` + CE **só nas posições com loss** (turnos do assistente, ~15–25% das posições), em blocos de 1.024 com recomputação | bloco: 1.024 × 151.936 × 4 B ≈ 0,62 GB (+ gradiente); poupa ~75% do custo do `lm_head` de 0,62 B parâmetros |
| Atenção | **kernel eficiente obrigatório** (flash ou mem-efficient). Sonda `scripts/probe_attention.py` (fwd + bwd, bf16, causal, 32/8 heads, head_dim 128, 8k e 16k) na ordem: (1) venv da v1 com `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` + `sdpa_kernel(FLASH_ATTENTION)` e `EFFICIENT_ATTENTION`; (2) backend Triton do FlashAttention via `triton-windows`; (3) WSL2 + PyTorch ROCm oficial (ambiente B, §4.2); (4) atenção "math" **em blocos de query** de 1.024 com checkpoint por bloco (sempre funciona; atenção ~2–3× mais lenta) | com atenção "math" inteira a matriz de scores é 32 × L² × 2 B: 4 GB por camada a 8k e 16 GB a 16k — não cabe. Estado do AOTriton no gfx1101 = **[hipótese]** (§4.1) |
| Lotes | buckets 2k / 4k / 8k / 12k / 16k (limites superiores); **8.192 tokens por micro-batch** até o bucket 8k (ex.: 4 × 2k, 2 × 4k, 1 × 8k), 1 × 12k e 1 × 16k; acumulação até ~64k tokens por passo; buckets embaralhados (não é curriculum por comprimento) | padding < 10%; mesmos tokens/passo em todos os buckets |
| Agenda | **1 época** (~80 M tokens ≈ 1.250 passos); warmup 3%, cosseno até 10%; eval intermediário a cada 10% (§6.4b); checkpoints a 25/50/75/100% avaliados como **curva de dados** (substitui os runs separados 10k/15k/20k trajetórias) | dado novo vale mais que segunda época; a curva diz se a 2ª rodada com mais trajetórias compensa (§9.5) |
| Teto e guarda | `limit_vram` 14,8 GB; VRAM compartilhada > 0,2 GB no `metrics.jsonl` aborta o run (lição v1); checkpoint a cada 30 min (§9.6) | — |

**VRAM estimada** **[estimativa; confirmar com `probe_throughput.py --steps 30` por bucket antes do run]**:

| Componente (GB) | Bucket 8k, sem offload | Bucket 16k, offload + MLP em blocos | 16k **sem** offload |
|---|---:|---:|---:|
| Pesos NF4 + embedding/`lm_head` bf16 | 6,2 | 6,2 | 6,2 |
| LoRA r=64 (pesos e gradientes fp32) + Adam 8-bit | 1,7 | 1,7 | 1,7 |
| Checkpoints das camadas | 2,4 | ~0,3 (camada corrente + buffer de cópia) | 4,8 |
| Pico de recomputação de uma camada **ou** bloco de CE (o maior) | 1,2 | 1,2 | 2,5 |
| Contexto HIP, alocador, fragmentação, gradiente do *hidden* | 1,0 | 1,3 | 1,3 |
| **Pico** | **~12,5** | **~10,7–12** | **~16,5 → não cabe** |

**Tempo** **[estimativa]**: ≤ 8k ≈ 70% dos tokens (~56 M) a ~380–450 tok/s (560 tok/s do 6,7B v1 **[fato]**, ajustado para 7,0 B de parâmetros não-embedding e para o custo da atenção a 4–8k) ≈ **35–41 h**; 12k–16k ≈ 30% (~24 M) a ~230–300 tok/s (a 16k a atenção soma ~50–70% do custo dos lineares) ≈ **22–29 h**. **Total ≈ 57–70 h (faixa 52–80 h) ≈ 3–4 dias contínuos de GPU**, sem contar sondas (~4 h) e merge/validação (~2 h).

**Plano B — se a sonda reprovar 16k** (sem kernel eficiente acima de 8k, ou pico > 14,8 GB mesmo com offload):

1. Fase principal com teto **8.192**: amostras ≤ 8k inteiras; trajetórias > 8k viram amostras **de prefixo** (a sequência append-only termina no último passo completo que cabe; loss só nos passos contidos; nenhum passo cortado no meio). Os passos finais das trajetórias longas (sobretudo S3/S14/S15) ficam fora da fase principal. ~60 M tokens a ~400 tok/s ≈ **40–45 h**.
2. **Fase curta de extensão** a 12k–16k: ~700 trajetórias longas completas + 20% de replay ≤ 8k ≈ 7–8 M tokens, lr 3e-5, com o melhor kernel disponível (eficiente: ~250 tok/s ≈ 8–9 h; atenção "math" em blocos: ~100–150 tok/s ≈ 14–22 h).
3. Se nem a extensão couber: o T4 fica treinado só até 8k (máximo viável) e o impacto é tratado como abaixo.

**Inferência a 16k treinando com sequências menores** **[análise]**:

- **Continua possível.** LoRA não muda RoPE nem a arquitetura, e o Qwen3-8B já suporta 32.768 tokens nativos **[fato: model card]**; o pacote ONNX mantém `context_length` 16.384 ([Fase 07 §5.7](fase-07-exportacao-onnx.md)).
- **O que não é garantido** **[hipótese]**: o comportamento **aprendido no SFT** (formato `<tool_call>`, regras do produto, concisão do raciocínio) nas posições 8k–16k vira extrapolação; ali atua o prior do instruct, que não conhece as ferramentas do Kapibara tão bem. Risco típico: perder instruções do system ou o formato no fim de trajetórias longas.
- **Medição obrigatória** na [Fase 06](fase-06-avaliacao-benchmarks.md): task success e violação de formato estratificados por comprimento do contexto (≤ 8k / 8–12k / 12–16k) no test sintético; aceitar se o bin 12–16k cair ≤ 5 pp em relação ao ≤ 8k. Se reprovar: a [Fase 08](fase-08-integracao-kapibara.md) compacta resultados de ferramenta antigos para manter o turno em ≤ ~10k e o metadata recomenda 8–12k; a extensão maior fica para a rodada seguinte.

### 6.5 Estágio 5 — Preferência (opcional): DPO/ORPO com pares do validador

| Item | Proposta |
|---|---|
| Geração de pares | `scripts/build_pref_pairs.py`: para cada prompt de chat/agente do treino, 6 amostras do modelo SFT (T=0,8, top-p 0,95); execução no MongoDB Docker da [Fase 03](fase-03-dataset-autocomplete-validacao.md) (mesmo usado pela Fase 06); **chosen** = resultado igual ao da referência; **rejected** = sintaxe válida mas resultado diferente, campo inexistente no schema, ferramenta errada/argumento inválido. Só prompts com ≥ 1 de cada. Meta: 10–20k pares. |
| DPO | β=0,1; LoRA lr 5e-6 (full: 5e-7); 1 época; log-probs de referência **pré-computados** (`--ref-logps cache`) para não carregar dois modelos — [DPO](https://arxiv.org/abs/2305.18290) |
| ORPO | alternativa sem modelo de referência: λ=0,1, lr 8e-6 — [ORPO](https://arxiv.org/abs/2403.07691) |
| Proteção | mistura 20% de exemplos SFT (CE) no lote; parada se o APT de validação cair > 0,05 |
| Script | `scripts/train_pref.py` (TRL `DPOTrainer`/`ORPOTrainer` se rodar no ambiente A; senão implementação própria ~150 linhas) |
| Adoção | somente se melhorar o conjunto cego da Fase 06 com teste pareado p < 0,05 e sem regressão de autocomplete |
| **T4 com raciocínio** | Pares do mesmo prompt gerados pelo checkpoint SFT no `kapisim` com thinking ligado (k=6, T=0,6) e **verificados por execução** (`verify.py` + `filter_reasoning.py` da Fase 04). Tipos de par: (i) sucesso × falha; (ii) sucesso coerente × raciocínio incoerente com a ação, vazamento no raciocínio ou ferramenta fora do plano "pensada"; (iii) **penalidade de raciocínio longo**: entre dois sucessos, chosen = raciocínio ≤ mediana do cenário, rejected = > 2× a mediana ou com fechamento forçado (≤ 30% dos pares, para não premiar brevidade extrema). Um sucesso nunca é rejected diante de uma falha mais curta. Meta local: **2–3k pares** (antes 5–8k), só prompts cujas duas respostas cabem em **≤ 8k** (o par dobra a memória de ativações); geração dos pares no llama.cpp (k=4, ~12 h), log-probs de referência pré-computados em NF4 (~5 h), DPO β=0,1 ou ORPO λ=0,1 (~15 h) — **~32 h no total, primeiro item da ordem de corte (§9.5)**. Adoção: task success não-inferior (margem 1 pp), tokens de raciocínio −15% ou mais e segurança intacta (Fase 06). **[hipótese, fora da v2]** RL on-policy (GRPO) com recompensa = sucesso por execução − λ·tokens de raciocínio no `kapisim` |

### 6.6 Estágio 6 — QAT / ajuste para INT4 e NPU (w4a16)

Evidência v1 **[fato]** (`docs\06-benchmark.md`): INT4 custa mais ao chat livre que ao benchmark — 1.5B-full merged → INT4 CPU perdeu 8 e ganhou 1 caso (p = 0,039; cego 29 → 24); INT8 e DML-FP16 não perderam nada; DML-INT4 85/120. Logo:

1. Primeiro tentar formatos sem QAT (INT8 CPU, FP16/INT4 DML) na [Fase 07](fase-07-exportacao-onnx.md); QAT só quando o tier **precisa** de INT4 (T2 7B em 16 GB com folga, T3 NPU, T0 < 2 GB RAM) e o PTQ reprovar no gate de paridade da [Fase 06](fase-06-avaliacao-benchmarks.md).
2. **QAT por auto-destilação** **[proposta]**: o mesmo modelo SFT bf16 é o professor (tokenizer idêntico → KD por logits do §6.3); o student tem fake-quant nos pesos lineares com o mesmo esquema do exportador (MatMulNBits, blocos de 32, assimétrico; **embedding e `lm_head` fora da quantização**, porque `GatherBlockQuantized` não roda no DirectML **[fato]**, `docs\23-onnx-slopcoder.md`). STE no backward; 0,3–0,5 época do mix do estágio 2; lr 1e-5 (full) ou LoRA r=16 lr 5e-5 sobre pesos congelados fake-quant; α=0,3 (KL domina).
3. **NPU w4a16** (T3): pesos 4 bits, ativações 16 bits estáticas. Além do fake-quant de pesos, simular ativações int16 por tensor com faixas calibradas (`scripts/calibrate_act_ranges.py`, 512 prompts representativos da Fase 03, mesmos que a Fase 07 usará no PTQ do QNN). Modelos w4a16 de LLM para Snapdragon X Elite via QNN já existem publicamente (ex.: [Qwen2.5-7B-Instruct w4a16](https://huggingface.co/Advantech-EIOT/qualcomm_qwen_qwen2.5-7B_Instruct)) **[fato]**; o fluxo ONNX Runtime GenAI + QNN para um modelo próprio **não** foi verificado **[hipótese]** — gate de viabilidade antes deste estágio.
4. Opção **[hipótese]**: rotações de Hadamard (QuaRot/SpinQuant) para reduzir outliers antes do w4a16, se o PTQ simples perder > 0,3 APT.
5. Treinar o T3 com o **mesmo comprimento de contexto estático** que o grafo QNN usará (ex.: 1.024 ou 2.048), pois a NPU exige formas fixas.

Implementação: `scripts/train_qat.py` com fake-quant próprio (~100 linhas, não depende de torchao — compatibilidade de torchao com ROCm/Windows **[hipótese]**).

## 7. Receitas consolidadas por tier

### T0 · KapiCoder-Mongo-Nano (~0,5B)

- Base: Qwen2.5-Coder-0.5B Base (linha v1). Alternativa: Qwen3-0.6B Base (controle do bake-off). *A alternativa DeepSeek-Coder-1.3B (`T0-ds`) foi removida na revisão 2026-10-08b: sua razão era o mesmo tokenizer do professor, que agora é dos Qwen.*
- Estágios: 2 (A/B LoRA r=64 × full FT) → 3 (KD por logits do KapiTeacher, plano base, ~+10% de tempo) → 6 se INT4 reprovar.
- Dados: ~160k exemplos (≈ 62 M tokens/época: ~370 tokens/ex. no autocomplete com cabeçalho de contexto, Fase 03 §4.6; a v1 tinha 227 sem cabeçalho), mix T0, subconjunto do T1.
- Objetivo de RAM < 2 GB na IDE: INT4 ~0,5 GB de pesos; pico de RAM v1 do 0.5B INT4 = 1,5 GB **[fato]**.

### T1 · KapiCoder-Mongo-Small (~1,5B)

- Base: Qwen2.5-Coder-1.5B Base (linha v1: 87/120 com orçamento completo); alternativa Qwen3-1.7B Base (controle). *`T1-ds` (DeepSeek-Coder 1.3B + KD) removido na revisão 2026-10-08b; o piloto da [Fase 01 §6.5](fase-01-selecao-modelos-base.md) agora é A × B.*
- Estágios: 2 (LoRA r=64, 2 épocas **do zero** sobre ~230k exemplos ≈ 100 M tokens/época, ~23 h puro e **~26 h com a perda KD** — a v1 mostrou que o orçamento completo foi o que moveu o chat livre) → 3 (KD por logits do KapiTeacher, plano base, gate do braço B) → 5 opcional → 6 se necessário.

### T2 · KapiCoder-Mongo-Pro (KapiTeacher-Mongo 14B ou 7B)

- **Primário [proposta, revisões 2026-10-08 e 2026-10-08b]: o próprio KapiTeacher-Mongo** da [Fase 02](fase-02-professor.md), exportado como T2 — **14B** se a sonda J03b aprovar a regra da [Fase 01 §6.6](fase-01-selecao-modelos-base.md), **7B** caso contrário. Ele já é treinado no mesmo contrato FIM da IDE (`editor-context-v1`/`kapi-context-v2`) e no chat curto `local-agent-fim-v0`, e o **adapter `qwen2` já existe** (sai o caminho DeepSeek da IDE para o T2) **[fato: Fase 01 §7]**. Economiza um QLoRA extra de 7–14B (~40–50+ h de GPU) e uma trilha inteira de avaliação/exportação. DML INT4: **14B** ≈ 8,5–9 GB + KV ~192 KiB/token (~0,8 GB a 4k) → cabe em 16 GB, TTFT maior que o do 7B (a medir, G4 T2); **7B** ≈ 5 GB **[hipótese: medir]**. Não há mais o custo de tokenizer diferente nem de licença DeepSeek.
- Para servir de T2, a mistura do SFT do professor acompanha a do T2 (mais chat e correção, Fase 02 §3.3 B) e o gate do professor passa a incluir os gates do T2 da [Fase 06](fase-06-avaliacao-benchmarks.md) (gate G2.8 da Fase 02).
- **Alternativa condicionada ao calendário (O4, §9.4):** se o professor for o 14B e ele reprovar o TTFT/VRAM/gates como T2, o T2 passa a ser o **KapiTeacher-Mongo-7B treinado à parte** (CPT + SFT ≈ 35 h; Qwen2.5-Coder-7B, estágio 2 da Fase 02) ou cai para o T1 em GPU. Se a troca 14B → 7B já tiver sido feita como corte (§9.5, item 2), o O4 deixa de existir.
- Estágio 6 (INT4 DML é o pacote-alvo): primeiro PTQ/AWQ (Fase 07); QAT por auto-destilação só se o INT4 reprovar a paridade (0,1–0,15 época, ~12 h; ~16 h no 14B).
- Merge (vale para professor e alternativa): dequantizar o base para bf16 e mesclar o adapter em bf16 na **CPU** (64 GB de RAM; nunca sobre NF4; o bf16 do 14B ≈ 29,5 GB cabe); validar (ver §10).

### T3 · KapiCoder-Mongo-NPU

- Não tem treino próprio até passar o **gate de viabilidade** (Fase 07): o base do T0/T1 **sem ajuste** exportado para QNN, carregado e gerando no Snapdragon X com ONNX Runtime GenAI, com TTFT/tokens/s medidos.
- Depois: checkpoint SFT do T0 ou T1 → estágio 6 w4a16 com contexto estático. Sem estágio próprio de SFT.

### T4 · KapiCoder-Mongo-Agent

- Base: Qwen3-8B híbrido, thinking ligado por padrão (alt. Qwen3-4B-Thinking-2507; fallback sem raciocínio Qwen2.5-Coder-7B-Instruct); Agent-Lite Qwen3-1.7B (thinking curto opcional) / 0.6B (sem thinking) se a Fase 06 mostrar task success aceitável.
- Estágios: 4 + 4b (tool calling + raciocínio, **local**, receita §6.4c: QLoRA NF4, buckets até 16k ou 8k + extensão, 1 época ~80 M tokens, ~57–70 h) → 5 opcional (local, ≤ 8k, §6.5) (pares verificados por execução, com penalidade de raciocínio longo) → 6 se INT4. A QAT/paridade INT4 deve medir os dois modos: o raciocínio longo é mais sensível à quantização **[hipótese]** (v1: INT4 custou mais em geração livre).
- Agent-Lite 1.7B: estágios 4 + 4b locais (contexto 4k, 12k exemplos ≈ 30 M tokens, 2 épocas ≈ 8–9 h).

## 8. Estimativas de tempo e VRAM na RX 7800 XT

Base das estimativas: throughputs da §3 (tokens totais, prompt + resposta). "Dados" = tokens por época. **[estimativa]** salvo indicação.

> **Conflito de volume resolvido (revisão local 2026-10-08):** o volume passa a ser dimensionado pelo **orçamento da GPU local** (§9), e não o contrário. O `train_pool` da Fase 03 continua com ~1,0 M exemplos validados (geração e validação são CPU, baratas); o que encolhe é o que passa pela GPU:
>
> | Conjunto | Antes | Agora (tokens/época) | Como |
> |---|---|---|---|
> | T0 ([Fase 03 §4.6](fase-03-dataset-autocomplete-validacao.md)) | 310k ex. | ~160k ex. ≈ **62 M**, 2 épocas | subconjunto estratificado do T1 |
> | T1 | 370k ex. | ~230k ex. ≈ **100 M**, 2 épocas | estratificado + sobre-amostragem de `hard` e grupos pequenos |
> | T2 | 340k ex. (Qwen 7B) | **0 aqui**: T2 = KapiTeacher 14B/7B (SFT do professor, ~105k ex. ≈ 47 M, [Fase 02](fase-02-professor.md)) | alternativa KapiTeacher-7B à parte (O4) só se sobrar calendário |
> | T4 agente ([Fase 04 §10.1](fase-04-dataset-agente-tools.md)) | 30k trajetórias, ~160 M de agente; ~250 M no T4 completo; nuvem | **8.000 trajetórias verificadas** (mín. 6.000) ≈ 42 M; **T4 completo ≈ 80 M, 1 época**, local | menos trajetórias porém verificadas por execução, seleção por dificuldade, curva de dados dentro do run |
> | Agent-Lite | 18k ex. ≈ 45 M | 12k ex. ≈ **30 M**, 2 épocas | derivado do T4 |
>
> Regras: 2 épocas só onde é barato (T0/T1, lição v1 de que o orçamento completo moveu o chat livre do 1.5B); nos modelos de 7–8B, **1 época + seleção por qualidade/dificuldade**; a curva de dados é medida nos checkpoints de 25/50/75/100% do próprio run.

| Tier / estágio | Método | Dados/época | Throughput | VRAM | Tempo por época | Épocas | Total local |
|---|---|---:|---:|---:|---:|---:|---:|
| T0 SFT | LoRA r=64 | 62 M | 6,8 k tok/s **[fato v1]** | ~7 GB **[fato v1]** | 2,5 h | 2 | **~5 h** |
| T0 SFT | full FT | 62 M | ~4,5–5,5 k tok/s **[hipótese]** | ~10–11 GB | ~3,4 h | 2 | ~7 h |
| T0 DAPT (opcional) | LoRA r=128 / full | 50 M | ~6 k tok/s | ~8–11 GB | 2,3 h | 1 | ~2,3 h |
| T1 SFT | LoRA r=64 | 100 M | 2,4 k tok/s **[fato v1]** | ~9 GB **[fato v1]** | 11,6 h | 2 | **~23 h** |
| Scoring + cache top-k `topk-v1` do pool T1 (J13; **plano base**) | KapiTeacher NF4, uma passada de forward | ~110 M tokens | 14B ~900 tok/s · 7B ~1.900 tok/s **[hipótese]** (v1 DeepSeek 6.7B: 1.970 **[fato]**) | ~10–13 GB (14B) · ~6 GB (7B) | 14B ~34 h · 7B ~16 h | 1 | **~34 h (7B: ~16 h)**, uma vez; substitui o scoring de 16 h + o cache opcional de ~15 h |
| T0/T1 SFT + KD por logits (§6.3) | LoRA r=64, perda `α·CE + (1−α)·τ²·KL` | 62 M / 100 M | ~−10% sobre o SFT puro **[hipótese]** | como o SFT puro (+1,1 GB de cache em disco) | +10% | 2 | T0 ~13 h · T1 ~26 h (embutido em J17/J18) |
| T2 = KapiTeacher 14B/7B | treinado na [Fase 02](fase-02-professor.md) (14B: CPT ~18 h + SFT ~45 h + it.2 ~12 h; 7B: ~11 + ~25 + ~9 h) | — | — | — | — | — | **0 h aqui** (QAT condicional ~12–16 h) |
| T2 alt. KapiTeacher-7B à parte (O4, condicional) | QLoRA NF4 r=64 | CPT 15–20 M + SFT ~47 M | ~500–560 tok/s | ~10–11 GB | ~35 h | 1 | ~35 h |
| KapiTeacher-14B (SFT it.1, na Fase 02) | QLoRA NF4 r=32–64 | ~47 M | ~250–300 tok/s **[hipótese, sonda J03b]** | ~13–14,5 GB (limite) | ~45–52 h | 1 | contado na §9.2 (J09) |
| T4 sondas | `probe_attention.py` + `probe_throughput.py` por bucket | — | — | — | — | — | ~4 h |
| T4 tools + raciocínio | QLoRA NF4 r=64, buckets até 16k (§6.4c) | **~80 M** | ~330–400 tok/s médio **[hipótese]** | ~12,5 GB (8k) · ~11–12 GB (16k com offload) **[estimativa]** | 57–70 h | 1 | **~57–70 h (faixa 52–80 h)** |
| T4 plano B (8k + extensão) | idem, teto 8k + extensão 12k–16k | ~60 M + 7–8 M | ~400 / 100–250 tok/s | ≤ 12,5 GB | — | 1 | ~50–65 h |
| T4 DPO/ORPO com raciocínio (opcional) | LoRA, ref. pré-computada, ≤ 8k × 2 | 2–3k pares ≈ 20 M | ~380 tok/s | ~13–14 GB | ~15 h | 1 | ~32 h com geração dos pares e log-probs de referência |
| Agent-Lite 1.7B (tools + thinking curto) | LoRA r=64, 4k ctx | ~30 M (12k ex. × ~2,5k) | ~1,8–2,2 k tok/s **[hipótese]** | ~10 GB | ~4,2 h | 2 | **~8–9 h** |
| DPO (T1, opcional) | LoRA, ref. pré-computada | 8k pares × ~600 tok × 2 | ~2 k tok/s | ~10 GB | ~1,5 h | 1 | ~5 h com geração de amostras |
| QAT (condicional) | fake-quant + KD | 0,1–0,5 época | ~60% do SFT **[hipótese]** | +1–2 GB | — | — | T0 ~1–2 h · T1 ~5–10 h · T2 ~12 h · T4 ~10 h |

Sondas obrigatórias antes de cada run longo: `scripts/probe_throughput.py --config <yaml> --steps 30` grava tok/s, VRAM dedicada e compartilhada (via contadores do Windows) em `reports/probes/`.

## 9. Orçamento e calendário da GPU local

Não existe GPU alugada no programa. Esta seção é a **fonte única** de horas de GPU de todas as fases; as estimativas das Fases 01–08 apontam para cá.

### 9.1 Premissas

- **Uma GPU, um job por vez** (`tmp/gpu.lock`, §4.2). Média de **19 h/dia efetivas** (faixa 18–20): jobs longos à noite e nos fins de semana; ~4–6 h/dia de janela livre para uso diário, testes DML da IDE ou pausa (§9.8).
- Horas = tempo de parede com a GPU ocupada **[estimativa]**, a partir dos throughputs da §3, incluindo sondas e merges/validações. Reserva de **15%** para reexecução, gate reprovado e travamentos.
- Semanas = semanas do programa (S1 = início da Fase 01). O caminho crítico é **humano** (geradores da Fase 03, `kapisim`/planejador/filtros da Fase 04): se ele atrasar, os jobs dependentes deslizam juntos e a GPU fica livre para os opcionais (§9.4).
- **Em paralelo na CPU** (12 núcleos/24 threads, 64 GB), sem disputar a GPU: geração programática e validação por execução no Docker (MongoDB 8.0 + atlas-local, 8 workers), dedup/MinHash/decontaminação, renderização e tokenização, execução das ferramentas do `kapisim` durante os rollouts (a GPU gera, a CPU executa), filtros de raciocínio, merges bf16 (RAM), conversão GGUF, exportações ONNX fp32/INT4/INT8 CPU (a fp32 do 8B pede ≥ 33 GB de RAM: não rodar com o professor MoE carregado em RAM). Benchmarks de **latência CPU** (Fases 06/07) exigem a máquina sem carga → só nas janelas sem job.

### 9.2 Todos os jobs de GPU do programa

Revisão 2026-10-08b: a coluna de horas traz o **cenário base com o KapiTeacher-14B**; entre parênteses, o cenário com o **7B** ("7B: x") quando difere. A escolha 14B × 7B sai do J03b (Fase 01 §6.6).

| ID | Job | Fase | Motor | Horas GPU | Depende de | Semana | Em paralelo na CPU |
|---|---|---|---|---:|---|---|---|
| J01 | Bake-off FIM + chat zero-shot (~11 modelos; DeepSeek-6.7B só como referência) | [01](fase-01-selecao-modelos-base.md) | HF ROCm | 6 | amostras do bake-off | S1–S2 | ContractProbe, Docker MongoDB, extração do catálogo |
| J02 | Bake-off tool calling + raciocínio ligado × desligado | 01 | llama.cpp / HF | 10 | catálogo extraído, `kapisim` piloto (20 tarefas) | S2 | idem |
| J03 | Bake-off exportabilidade + latência DML (inclui `-e dml` INT4 do 14B e do 7B) | 01 | ORT DML | 6 | J01 | S2 | exportações CPU do builder |
| **J03b** | **Seleção do professor: sonda de QLoRA de 1–2 h em cada Qwen2.5-Coder (7B e 14B) — tok/s, VRAM dedicada e compartilhada sem offload; aplica a regra 14B × 7B (Fase 01 §6.6)** | 01 | QLoRA NF4 | **5** (4–6) | J01 (zero-shot FIM dos três), `probe_throughput.py` | S2 | `teacher_choice.json`, download/merge dos pesos |
| J04 | Professor: verificação do checkpoint (`config.json`, tokenizer, IDs FIM) + baseline zero-shot | [02](fase-02-professor.md) | HF NF4 | 8 (7B: 7) | J03b, `val`/`blind` iniciais da Fase 03 | S2–S3 | geradores da Fase 03 |
| J05 | Sondas dos professores auxiliares (Qwen3-14B; Qwen3-30B-A3B-Thinking-2507 com `--n-cpu-moe`), 50 prompts cada | 02/[04](fase-04-dataset-agente-tools.md) | llama.cpp | 2 | GGUFs | S3 | idem |
| J06 | Re-medição dos modelos v1 na suíte v2 | [06](fase-06-avaliacao-benchmarks.md) | ORT | 3 | suítes da Fase 06 | S3 | — |
| J07 | Paráfrases/back-translation do chat (Qwen3-14B `/no_think`) | [03](fase-03-dataset-autocomplete-validacao.md) | llama.cpp | 5 | chat gerado e validado | S4 | validação por execução das paráfrases |
| J08 | Professor CPT (15–20 M tokens) | 02 | QLoRA | 18 (15–22) (7B: 11) | corpus CPT, J03b | S4 | validação/dedup da Fase 03 |
| J09 | Professor SFT it.1 (~105k ex., ~47 M tokens, 1 época) | 02 | QLoRA | 45 (45–52) (7B: 25) | J08, `train` congelado | S5 | `kapisim`, planejador |
| J10 | Student-baseline sem professor (T1, 100k ex., 1 época; referência do G2.4) | 05 | LoRA | 5 | `train` | S5 | idem |
| J11 | Avaliações do professor (it.1 e it.2) | 02 | HF NF4 | 9 (7B: 7) | J09 / J12 | S5–S6 | execução das saídas no Docker |
| J12 | Professor SFT it.2 (correção curta, ~0,3 época) | 02 | QLoRA | 12 (7B: 9) | J11 | S6 | — |
| J13 | Destilação: ***scoring* + cache top-k `topk-v1` do pool T1 numa só passada de forward** (230k ex., ~110 M tokens; uint32, 194 B/posição, ~1,1 GB) | 02/03 | HF NF4 | 34 (7B: 17) | professor aprovado (G2.1–G2.5) | S6 | merge bf16 do professor + GGUF (CPU) |
| J14 | Destilação: greedy (40k difíceis) + best-of-4 (10k chat/correção) | 02/03 | llama.cpp Q6_K (14B) / Q8_0 (7B) | 12 (7B: 10) | J13 | S6–S7 | validação por execução das candidatas |
| J15 | Piloto KD A × B (+ 2 variantes de B), cache top-k reaproveitado do J13 (Fase 01 §6.5) | 01 | LoRA | 10 | J12, J13 | S7 | montagem dos datasets T0/T1 |
| J16 | Agente: linha de base Qwen3-8B (com/sem thinking) + piloto de 1.000 rollouts + verbalização pelo professor | 04 | llama.cpp | 14 | `kapisim`, planejador, filtros | S7 | `verify.py`, inspeção humana |
| J17 | T0 SFT (LoRA + braço full FT), com perda KD (+10%) | 05 | LoRA / full | 13 | J14, J15 | S7–S8 | — |
| J18 | T1 SFT, 2 épocas, com perda KD (+10%) | 05 | LoRA | 26 | J14, J15 | S8 | trajetórias do planejador (CPU) |
| J19 | Rollouts de raciocínio (a1): ~5k tarefas, k adaptativo 2→4 | 04 | llama.cpp Q8_0 | 27 | J16 | S8–S9 | `kapisim` executa as chamadas; `verify`/`filter_reasoning` |
| J20 | Racionalização (a2) + professor auxiliar (b) Qwen3-14B nas tarefas difíceis restantes | 04 | llama.cpp | 17 (4 + teto 13) | J19 | S9 | idem |
| J21 | Dados gerais com thinking (preservação) | 04 | llama.cpp | 8 | — | S9 | testes unitários das respostas |
| J22 | Avaliação T0/T1 (checkpoints + pacotes) | 06 | HF / ORT | 12 | J17, J18 | S9 | exportação CPU de T0/T1 |
| J23 | Export DML T0/T1 + paridade + benchmark | [07](fase-07-exportacao-onnx.md) | ORT DML | 4 | J22 | S9 | — |
| J24 | T4 sondas (atenção, VRAM por bucket, 30 passos) | 05 | QLoRA | 4 | dataset T4 congelado | S10 | — |
| J25 | **T4 SFT** (§6.4c) | 05 | QLoRA | **57–70** (52–80) | J24, J20, J21 | S10–S11 | Fase 08 na IDE, eval à mão |
| J26 | Agent-Lite 1.7B | 05 | LoRA | 9 | J20 | S11 | merge T4, GGUF |
| J27 | T2 = KapiTeacher (14B/7B): export `dml-int4` + `int4` CPU + calibração AWQ/GPTQ + paridade + benchmark | 07 | ORT DML / ROCm | 15 (7B: 12) | J12 | S11 | export CPU |
| J28 | Avaliação T2 | 06 | ORT DML | 10 (7B: 8) | J27 | S11 | — |
| J29 | Avaliação intermediária do T4 (GGUF Q8_0, 1 seed, 500 casos, por faixa de contexto) + merge/validação | 06/05 | llama.cpp | 10 | J25 | S11 | — |
| J30 | Export T4 `dml-int4` + calibração AWQ + paridade com raciocínio + VRAM/TTFA | 07 | ORT DML | 16 | J29 | S12 | export fp32 CPU |
| J31 | Avaliação final T4 (pacote, 2 modos × 3 seeds) + Agent-Lite | 06 | ORT DML | 30 | J30 | S12 | — |
| J32 | T3: QAT w4a16 + calibração (só após o gate QNN com hardware) | 05/07 | HF ROCm | 4 | Snapdragon X | S13+ | Olive x64 (CPU) |
| J33 | Homologação na IDE (testes `Explicit` DML, `KAPI_AGENT_*`) | [08](fase-08-integracao-kapibara.md) | ORT DML | 8 | pacotes aprovados | S13–S15 | — |
| — | **Subtotal** | | | **≈ 470–500** (7B: ≈ 415–445) | | | |
| — | Reserva de 15% | | | ≈ 70–75 (7B: ≈ 65–70) | | S13–S14 | |
| — | **Total do plano base** | | | **≈ 540–575 h ≈ 28–30 dias de GPU a 19 h/dia** (**7B: ≈ 480–515 h ≈ 25–27 dias**) | | **S1–S14** | |

Variação sobre o plano anterior (professor DeepSeek-6.7B, 470–500 h): **14B ≈ +65–70 h de subtotal** (sonda +5, CPT +8, SFT it.1 +21, it.2 +4, avaliações do professor +3, scoring + top-k +18, greedy +2, export/avaliação T2 +5, piloto de KD −4, perda KD em T0/T1 +4, verificação +2), **7B ≈ +10–12 h** (sonda +5, perda KD +4 e pequenos acréscimos, piloto −4). O custo do cache top-k opcional (~15 h) e o do `T1-ds` (~22 h) saíram do plano.

### 9.3 Calendário por semana (plano base, KapiTeacher-14B)

| Semana | Jobs | Horas de GPU | Observação |
|---|---|---:|---|
| S1–S2 | J01–J03, **J03b**, início de J04 | ~31 | GPU quase ociosa: o trabalho é montar scripts, Docker, contratos; **J03b decide 14B × 7B no fim da S2** |
| S3 | J04, J05, J06 | ~13 | |
| S4 | J07, J08 | ~23 | |
| S5 | J09, J10, J11 | ~53 | J09 cruza uma noite + fim de semana |
| S6 | J11, J12, J13 | ~52 | J13 (scoring + top-k, ~34 h) em fim de semana |
| S7 | J14, J15, J16, J17 | ~53 | |
| S8 | J18, J19 | ~53 | |
| S9 | J20–J23 | ~41 | dataset T4 congelado no fim da semana |
| S10–S11 | J24, **J25** | ~65–75 | **bloco de 3–4 dias contínuos** (começar numa sexta à noite) |
| S11 | J26–J29 | ~44 | |
| S12 | J30, J31 | ~46 | |
| S13–S14 | reserva; opcionais (§9.4) | até ~75 | |
| S13–S15 | J32, J33 | ~12 | homologação e release (Fases 07/08) |

Com o 7B, S4–S8 perdem ~50 h (S4 ~16, S5 ~31, S6 ~32) e o programa encerra a GPU na S13. Pico semanal ≈ 75 h (S10–S11), abaixo do teto de ~133 h/semana: **a GPU não é o gargalo**, exceto no bloco contínuo do T4. Com todos os opcionais (+~235 h, §9.4), o uso da GPU se estende até ~S16.

### 9.4 Jobs opcionais/condicionais (fora do plano base)

| ID | Job | Horas | Condição |
|---|---|---:|---|
| O1 | DPO/ORPO do T4 (pares ≤ 8k, ref. pré-computada) | ~32 | T4 abaixo do gate de task success ou raciocínio prolixo |
| O2 | DPO do T1 | ~5 | T1 marginal no cego |
| O3 | *(removido na revisão 2026-10-08b)* — o `T1-ds` e o cache top-k opcional deixaram de existir: a KD por logits e o cache `topk-v1` estão no plano base (J13/J15/J17/J18) | — | — |
| O4 | T2 alternativo = KapiTeacher-7B treinado à parte (CPT ~11 + SFT ~25 + export ~12 + avaliação ~8) | ~55 | professor 14B reprovou o TTFT/VRAM/gates de T2 **e** há folga (inexistente se o corte "14B → 7B" já foi feito) |
| O5 | QAT (T0 ~2, T1 ~7, T2 ~12–16, T4 ~10) | ~31–35 | INT4 reprovou a paridade |
| O6 | 2ª rodada do T4 com ~12k trajetórias (rollouts ~15 h + treino ~70 h) | ~85 | curva 25→100% ainda subindo na Fase 06 |
| O7 | Professor (b) Qwen3-30B-A3B-Thinking-2507 com experts na RAM | 10–20 | Qwen3-14B reprovou nas tarefas S14–S16 (Fase 04 §6.5) |
| O8 | Estágio C do professor (best-of-4 + rodada curta) | ~8 | só com ganho ≥ 1 pp exec-valid previsto |
| O9 | DAPT do T0/T1 | 2–6 | sonda de perplexidade de domínio ≥ 10% |

### 9.5 Ordem de corte (o que se sacrifica primeiro se o calendário ficar longo)

1. **O1/O2 — DPO/ORPO** (T4 e T1): −37 h. O SFT verificado por execução já carrega o sinal principal.
2. **Trocar o professor 14B pelo 7B** *(novo, revisão 2026-10-08b — corte **cedo**, antes de qualquer corte de dados)*: −~60–65 h (14B ≈ 540–575 h → 7B ≈ 480–515 h), mantendo volumes, gates e verificação por execução; só perde a margem de qualidade do 14B (a regra da Fase 01 §6.6 exige ≥ 3 p.p. de vantagem, logo a perda esperada é pequena). Tem de ser decidido **antes do J08** (semana S4): depois que o CPT/SFT do 14B começa, as horas já gastas não voltam. Efeitos colaterais: T2 = KapiTeacher-7B (INT4 ~5 GB, TTFT menor) e O4 deixa de existir.
3. **O4 — T2 alternativo KapiTeacher-7B à parte**: T2 = KapiTeacher sem discussão (se ele reprovar, o T2 sai da v2 e o T1 cobre o autocomplete GPU).
4. **KD por logits no T0/T1 e piloto J15**: desligar a perda KD (−4 h em J17/J18) e cortar o piloto (−10 h; −14 h se o corte vier antes da S7); T0/T1 seguem sequence-level. O cache top-k continua sendo produzido no J13 (sem custo extra).
5. **J26 — Agent-Lite 1.7B** (−9 h de treino, −~10 h de avaliação/exportação): no Vivobook X1-26-100 ele roda só em CPU ARM64 INT4 (a NPU fica com o 0.6B, [Fase 07 §5.5.1](fase-07-exportacao-onnx.md)); o 0.6B cobre o Agent-Lite.
6. **J20 (b) — professor de raciocínio auxiliar**: fica só auto-destilação (a1 + a2) + traços do planejador (c) (−13 h).
7. **J12 — 2ª iteração do SFT do professor** e braço full FT do T0 (−12 h no 14B / −9 h no 7B e −7 h).
8. **Fase de 16k do T4 → plano B** (8k + extensão curta, §6.4c) e avaliação intermediária com 1 seed em 300 casos (−10–15 h).
9. Último recurso: T4 com 6.000 trajetórias (mínimo) e T1 com 1,5 época (−15 h e −6 h).

**Nunca se corta:** verificação por execução, gates da Fase 06 nos pacotes finais (inclusive os 3 seeds × 2 modos do T4), conjunto cego, paridade de exportação, reserva para reexecutar um job longo que travou.

### 9.6 Checkpoints e retomada (jobs de vários dias)

- **Treino:** checkpoint atômico por **tempo** (a cada 30 min) além de 5% dos passos — adapter, otimizador, scheduler, RNG, posição no plano de lotes; perda máxima numa queda = 30 min. Retenção: 3 últimos + `best` + `final`. Cópia diária do último checkpoint para o outro disco (C: ↔ E:). `--resume auto` só aceita fingerprint idêntico (§4.3).
- **Geração em lote** (scoring, greedy, best-of-N, rollouts, avaliação): saída em shards *append-only* por `id` com índice `done.idx`; seed por item derivada do `id` (a retomada reproduz o resultado); shard sem hash final é descartado e refeito.
- **Fila `scripts/gpu_queue.ps1`** sobre `configs/gpu_queue.yaml` (job, comando, dependências, janela permitida noite/dia, `max_restarts`): detém `tmp/gpu.lock`, grava *heartbeat* e encadeia os jobs da §9.2 sem intervenção. **Watchdog:** se o processo morrer ou o `metrics.jsonl`/shard não avançar por 15 min (travamento de driver, *device lost*/TDR, servidor llama.cpp caído), mata o processo, espera `check_env.py` verde, relança com `--resume auto` até 3 vezes; na 3ª falha segue para o próximo job sem dependência e registra em `reports/gpu_queue/incidents.md` + notificação local.
- **Pausa cooperativa:** `gpu_queue.ps1 -Pause` cria `tmp/gpu.pause`; o treino salva checkpoint no passo seguinte e sai, a geração fecha o lote corrente → GPU livre em ~2–5 min para uso diário; `-Resume` retoma.
- **Energia e sistema** (decisões do usuário, não automatizadas): nobreak para o PC; impedir suspensão e reinício por atualização durante jobs longos (horário ativo/pausa de atualizações); opcionalmente uma tarefa agendada que reabre a fila após reinício.

### 9.7 Disco

| Item | Estimativa | Onde |
|---|---:|---|
| Modelos do bake-off e bases (HF cache; inclui Qwen2.5-Coder-14B ≈ 30 GB e 7B ≈ 15 GB) | 230–280 GB no pico do bake-off; **~110–130 GB** depois de apagar os eliminados após o G5 | `BaseRepos\` |
| GGUFs (Qwen3-8B Q8_0 ~8,7 GB, Qwen3-14B Q4_K_M ~9 GB, 30B-A3B Q4_K_M ~18,6 GB, professor Q6_K ~12 GB (14B) / Q8_0 ~8 GB (7B), checkpoints do T4 para avaliação) | ~65 GB | `BaseRepos\Qwen3-14B-GGUF\` + `Repos\<modelo>\gguf\` |
| Merged bf16 (professor × 2 ≈ 59 GB com o 14B, T4 16,4 GB, T0/T1, alternativos) | ~110 GB | `Repos\<modelo>\merged\` |
| Checkpoints de treino (retidos) | ~20 GB | `Repos\<modelo>\checkpoints\` (cópia diária do último checkpoint em E:, só como backup) |
| Dados (`train_pool`, rollouts brutos, scoring, cache `topk-v1` ~1,1 GB) | 50–80 GB | `C:` (workspace) |
| ONNX (fp32 8B ≈ 33 GB, variantes por tier) | ~120 GB | `Repos\<modelo>\exports\` (fp32 intermediário apagado após a quantização aprovada) |
| Docker (imagens + volumes de fixtures) | ~20 GB | `C:` |
| **Total** | **~620–720 GB no pico** (~500–560 GB em regime, após limpezas) | **Tudo em C:** sob `C:\IA\models\KapibaraStudio.MongoIA` (decisão do usuário; ~830 GB livres em 2026-10-08) → folga pequena no pico do bake-off. Mitigação: baixar candidatos do bake-off em lotes e apagar os eliminados; se o livre cair abaixo de 100 GB, mover `cache\` e `cache\kd\` para E: com junção `mklink /J` (o caminho lógico continua em C:). **F: nada** |

### 9.8 Uso diário, temperatura e desgaste

- A GPU fica ocupada ~540–575 h em ~14 semanas (média ~40 h/semana; pico ~75 h na semana do T4; com o 7B, ~480–515 h em ~13 semanas). Janela diária livre de ~4–6 h (ex.: 13h–18h) para testes DML da IDE, Fase 08 e uso pessoal; jobs > 24 h (J09, J13, J19, J25) começam na sexta à noite.
- **[proposta, decisão do usuário]** Ligar o monitor na saída de vídeo da placa-mãe (o Ryzen 9 7900 tem gráficos integrados; o ROCm já o expõe como `cuda:1`, ignorado pelo `check_env.py`): libera ~0,5–1 GB de VRAM da RX 7800 XT e evita que o desktop dispute a GPU ou provoque TDR durante o treino.
- Temperatura: registrar junção/hotspot, potência e clock em CSV (sensores do driver via Adrenalin/HWiNFO, ou `amd-smi` se disponível no Windows **[hipótese]**); a fila aciona a pausa cooperativa com junção ≥ 100 °C por 5 min. Limite de potência −10% no Adrenalin é opção do usuário (perda de vazão estimada em 3–5% **[hipótese]**). Limpeza de poeira antes do bloco S5–S12.
- Energia: ~280 W da GPU + ~120 W do resto × ~560 h ≈ **225 kWh** (7B: ~500 h ≈ 200 kWh) **[estimativa]**.
- Desgaste: ~560 h de carga sustentada equivalem a alguns meses de jogo intenso; o risco prático está em ventoinhas e pasta térmica, não no chip **[hipótese]**.

## 10. Checkpoints, merge e rastreio de experimentos

### Checkpoints

- Atômicos (escrita em `tmp` + rename) a cada ~5% dos passos **e a cada 30 min** (o que vier primeiro): adapter/pesos, otimizador, scheduler, RNG, posição no plano de lotes — como na v1; regras de retomada de jobs de vários dias na §9.6.
- Retenção: 3 últimos + `best` (menor loss de validação) + `final`. Exportar `best` por padrão; se a validação de geração discordar da loss (v1 viu oscilação com 60 exemplos), a escolha é feita pela [Fase 06](fase-06-avaliacao-benchmarks.md) entre `best` e `final`.

### Merge

- `scripts/merge_adapter.py`: base bf16 + adapter → `Repos\KapiCoder-Mongo-<tier>\<run>\merged\` (safetensors + tokenizer + `generation_config.json` + `run_manifest.json`).
- Validação do merge **[proposta]**, ampliando a v1 (4 prompts **[fato]**): em 200 prompts de validação, token top-1 igual entre base+adapter e merged em ≥ 99,5% das posições da resposta e Δloss < 0,5% relativo. Para QLoRA, comparar o merged bf16 contra base NF4+adapter: o desvio esperado é maior; se Δloss > 1%, fazer 0,05–0,1 época de "healing" LoRA r=16 sobre o merged (carregado em NF4 a partir do merged bf16, ~3–6 h para o 8B). Merges de 7–8B rodam na CPU (64 GB de RAM), sem ocupar a GPU.

### Rastreio

- Fonte da verdade: `metrics.jsonl` por run (passo, loss, lr, tok/s, VRAM dedicada/compartilhada, grad-norm, métricas de eval) — formato da v1.
- **MLflow local** (`mlflow` com backend `file:./mlruns`, sem servidor externo) como visualização opcional: `scripts/log_to_mlflow.py` importa os JSONL. W&B não é usado (exige conta/sincronização); se usado, apenas `WANDB_MODE=offline`.
- `reports/training/runs.csv`: uma linha por run (tier, estágio, config hash, dados hash, tempo, tok/s, VRAM pico, best step, val loss, decisão).

### Estrutura de pastas

```text
C:\IA\models\KapibaraStudio.MongoIA\
├── configs\train\
│   ├── t0-sft-lora.yaml          t0-sft-full.yaml        t0-qat.yaml
│   ├── t1-sft-lora.yaml          t1-sft-kd.yaml          t1-dpo.yaml       (t0-sft-kd.yaml análogo)
│   ├── t2alt-sft-qlora.yaml      t2-qat.yaml             (T2 = KapiTeacher 14B/7B; t2alt = KapiTeacher-7B à parte, só O4)
│   ├── t3-w4a16-qat.yaml
│   ├── t4-tools-qlora.yaml       t4-tools-qlora-8k.yaml   t4-ext16k.yaml   (buckets até 16k / plano B)
│   ├── t4-pref-reasoning.yaml    t4lite-tools-thinking.yaml
│   ├── gpu_queue.yaml            (fila única de jobs de GPU, §9.6)
│   └── rounds.yaml               (regras de adoção pré-registradas)
├── scripts\
│   ├── check_env.py  probe_throughput.py  probe_attention.py  probe_gen_throughput.py  probe_domain_ppl.py  repro_check.py
│   ├── gpu_queue.ps1             (fila + lock + watchdog + pausa cooperativa)
│   ├── to_gguf.ps1               (merged → GGUF Q8_0 + paridade top-1 contra HF)
│   ├── train_sft.py              (SFT/DAPT/KD/tools; evolução de train_student.py da v1)
│   ├── build_kd_cache.py (atalho para teacher/distill.py topk)  build_pref_pairs.py  calibrate_act_ranges.py
│   ├── train_pref.py  train_qat.py  merge_adapter.py
│   ├── make_run_manifest.py  log_to_mlflow.py  compare_rounds.py
│   └── run_tier.ps1              (orquestra: check_env → probe → treino → merge → smoke eval; uma etapa de GPU por vez)
├── Repos\KapiCoder-Mongo-<tier>\<run>\   (um workspace por tier: Nano, Small, NPU, Agent, Agent-Lite; T2 = Repos\KapiTeacher-Mongo-14B\)
│   ├── checkpoints\step-000400\ ...
│   ├── adapters\{best,final}\
│   ├── merged\
│   ├── metrics.jsonl
│   └── run_manifest.json
├── cache\kd\<teacher-hash>\<dataset-hash>\   (top-k logits)
└── reports\training\<run>\training_report.md · runs.csv · probes\
```

Comando típico **[proposta]**:

```powershell
powershell -File scripts/run_tier.ps1 -Config configs/train/t1-sft-lora.yaml -Stage sft -Merge -SmokeEval
python scripts/train_sft.py --config configs/train/t1-sft-kd.yaml --kd-cache cache/kd/<hash> --resume auto
```

## 11. Atividades (passo a passo)

- [ ] 1. Clonar o venv da v1 para `env\.venv-train` sem alterar versões; gerar lockfile; `check_env.py` verde.
- [ ] 2. Portar `train_student.py` → `train_sft.py` (multi-tarefa, `--objective {sft,lm}`, `--kd-cache`, template de chat/tools, máscara por turno).
- [ ] 3. `repro_check.py` e `probe_throughput.py` em T0/T1 (30 passos cada); `probe_attention.py` e `probe_throughput.py` do T4 por bucket (8k/12k/16k); `probe_gen_throughput.py` do llama.cpp (Qwen3-8B Q8_0, `-np` 4/8/16); atualizar §8, §6.4c e §9 com números medidos.
- [ ] 3a. Confirmar com a sonda J03b (Fase 01 §6.6) o professor 14B × 7B; registrar `reports\teacher_choice.json` e propagar a §9.
- [ ] 3b. `gpu_queue.ps1` + `configs/gpu_queue.yaml` com os jobs da §9.2; teste de retomada (matar o processo no meio de um run de 1 h e retomar).
- [ ] 4. Registrar `rounds.yaml` (regras de adoção) e `configs/gates.yaml` da Fase 06 antes do primeiro treino.
- [ ] 5. T0: SFT LoRA e full FT; merge; smoke eval; entregar à Fase 06.
- [ ] 6. T1: SFT LoRA 2 épocas com a perda KD por logits do KapiTeacher (cache `topk-v1` do J13; ligada se o braço B do piloto vencer).
- [ ] 7. T3: aguardar gate de viabilidade QNN (Fase 07); então QAT w4a16 do checkpoint escolhido.
- [ ] 8. T2: entregar o KapiTeacher (14B ou 7B; merged bf16 da Fase 02) à Fase 07/06 como T2; KapiTeacher-7B à parte só se o 14B reprovar como T2 e a §9 tiver folga (O4).
- [ ] 9. T4: sondas; escolher receita 16k (buckets) ou plano B (8k + extensão) pela sonda; SFT de tool calling + raciocínio (estágios 4 + 4b, local, 1 época ~80 M tokens); curva de dados pelos checkpoints de 25/50/75/100%; merge; GGUF para a avaliação intermediária. Agent-Lite 1.7B.
- [ ] 10. Opcionais da §9.4 (DPO/ORPO, QAT, DAPT), conforme resultados da Fase 06 e a ordem de corte §9.5.

## 12. Critérios de saída (gates desta fase)

| Gate | Critério mensurável |
|---|---|
| Ambiente | `check_env.py` verde; `repro_check.py` com desvio de loss < 0,5%; nenhum run com VRAM compartilhada > 0,2 GB nos `metrics.jsonl` |
| Reprodutibilidade | 100% dos runs com `run_manifest.json` completo e fingerprint verificado na retomada |
| Cache KD | `topk-v1` cobre 100% dos `id` do pool T1; em 1.000 posições, Σ(top-k) + massa residual = 1 ± 1e-3; linhas de padding de professor e aluno mascaradas (teste unitário) |
| Orçamento de GPU | horas reais de cada job registradas em `reports/gpu_queue/ledger.csv`; desvio > 30% sobre a §9.2 dispara revisão da ordem de corte (§9.5) antes do job seguinte |
| T4 local | sonda aprovada (pico ≤ 14,8 GB e 0 GB compartilhada nos buckets usados; kernel de atenção registrado no manifest); se o teto for 8k, a extensão e a medição por faixa de contexto (§6.4c) são obrigatórias |
| Convergência | loss de validação do `best` ≤ 0,98 × loss no primeiro eval; sem NaN/inf; grad-norm estável |
| Merge | top-1 idêntico ≥ 99,5% e Δloss < 0,5% (LoRA) ou < 1% (QLoRA, após healing se preciso) |
| Smoke eval | 200 exemplos de validação: syntax valid ≥ 0,90; 0 respostas conversacionais em autocomplete; tool calls com JSON válido ≥ 95% (T4); T4 com thinking: `<think>…</think>` bem formado ≥ 99%, obediência à chave (≥ 97% com raciocínio não vazio quando ligado, 0% quando desligado), p95 de tokens de raciocínio por geração ≤ 900 |
| Entrega | checkpoint mesclado + manifest por tier entregue à [Fase 06](fase-06-avaliacao-benchmarks.md); aprovação final é da Fase 06 |

## 13. Riscos e mitigação

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Atualização de driver/ROCm quebra o ambiente validado | média | alto | venv e driver congelados durante o programa; WSL2 oficial (ROCm 7.2.1) como plano B |
| Offload silencioso para memória compartilhada | alta (Windows) | alto | `limit_vram`, trava de GPU, monitorar VRAM compartilhada no `metrics.jsonl` |
| QLoRA 7–14B lento demais local (T4 ~57–70 h; professor 14B ~45–52 h por época de SFT, 7B ~25 h) | alta | médio | volumes dimensionados ao orçamento (§8), 1 época + seleção por dificuldade, T2 = KapiTeacher (sem segundo treino), "trocar 14B por 7B" como corte cedo (§9.5, item 2), ordem de corte §9.5 |
| KapiTeacher-14B no limite de VRAM (pico 13–14,5 GB) ou com offload silencioso | média | alto | sonda J03b (Fase 01 §6.6) com VRAM compartilhada > 0,2 GB = reprova; micro-batch 4k; r=32; cair para o 7B |
| Incompatibilidade de merge NF4 → bf16 | média | médio | validação de merge; healing LoRA curto |
| Sem kernel de atenção eficiente no gfx1101/Windows (AOTriton não confirmado) | média | alto (T4 16k) | sonda `probe_attention.py` na S2 (antes de qualquer job longo); Triton FlashAttention; WSL2; atenção em blocos; plano B 8k + extensão (§6.4c) |
| Treino ≤ 8k degrada o agente em contextos 8–16k | média | médio | medição por faixa de contexto na Fase 06; compactação de resultados no runtime (Fase 08); extensão maior na rodada seguinte |
| Vazão do llama.cpp em lote abaixo do previsto (rollouts, scoring, avaliação) | média | médio (calendário) | `probe_gen_throughput.py` na S2; k adaptativo; teto de horas por job na fila; cortar (b) e DPO primeiro |
| **Superaquecimento/desgaste e indisponibilidade da GPU para uso diário** (~560 h de carga em ~14 semanas; bloco de 3–4 dias no T4) | alta (indisponibilidade) / baixa (dano) | médio | §9.8: janela diária livre, pausa cooperativa, jobs > 24 h no fim de semana, monitor no iGPU, log de temperatura com pausa ≥ 100 °C, limite de potência opcional, limpeza; checkpoints a cada 30 min e watchdog (§9.6) |
| Queda de energia / travamento de driver (TDR) no meio de job de vários dias | média | médio | checkpoint por tempo, shards *append-only*, watchdog com relançamento automático, nobreak (§9.6) |
| Disco cheio no meio de um run | média | alto | tudo em C: (~830 GB livres) com pico estimado de 620–720 GB (§9.7): bake-off em lotes, limpeza dos eliminados, fp32 ONNX e checkpoints intermediários apagados após aprovação; `gpu_queue.ps1` recusa iniciar job com < 100 GB livres; junção para E: como válvula; F: proibido |
| Mais chat degrada grupos pequenos do autocomplete (visto na v1) | alta | médio | teto de 25% de chat, piso por grupo, gate por grupo na Fase 06 |
| SFT de tools causa esquecimento (chat/código geral) | média | alto | ~22% de replay geral + Mongo (chat 9%, dados gerais com thinking 8%, instruções gerais 5%); regressão geral medida na Fase 06 |
| SFT degrada o raciocínio nativo do Qwen3 ou colapsa a chave thinking/no_think | média | alto | ~8% de dados gerais com thinking, mistura em todos os componentes, LoRA, eval intermediário de obediência à chave e de problemas gerais com thinking; parada antecipada |
| Raciocínio aprende a ser prolixo (custo de latência no painel) | média | médio | seleção do rollout mais curto aceito na Fase 04, orçamento no treino, DPO com penalidade de comprimento |
| Treino a 16k não cabe localmente mesmo com offload | média | médio | plano B 8k + extensão curta; r=32 / menos módulos LoRA; máximo viável documentado (§6.4c) |
| KD por logits sem ganho sobre sequence-level | média | baixo | perda KD desligável (CE pura) pela regra pré-registrada do braço B do piloto; o cache sai de graça do scoring |
| INT4/w4a16 perde chat livre (v1: −7 casos líquidos) | alta | alto | preferir INT8/FP16; QAT por auto-destilação; gate de paridade da Fase 06 |
| Fluxo QNN para modelo próprio inviável | média | alto (T3) | gate de viabilidade antes de qualquer treino específico de T3 |
| Bibliotecas (TRL, torchao, Liger) sem suporte ROCm/Windows | média | baixo | implementações próprias curtas (DPO, fake-quant); WSL2 |
| Overfitting em famílias sintéticas sem ganho em pedidos livres (v1.2) | alta | médio | adoção decidida só por conjunto cego novo (Fase 06) |

## 14. Dependências

- **Precisa de:** [Fase 01](fase-01-selecao-modelos-base.md) (bases e licenças), [Fase 02](fase-02-professor.md) (KapiTeacher e saídas da destilação), [Fase 03](fase-03-dataset-autocomplete-validacao.md) e [Fase 04](fase-04-dataset-agente-tools.md) (datasets com manifest), [Fase 06](fase-06-avaliacao-benchmarks.md) (gates registrados antes do treino), [Fase 07](fase-07-exportacao-onnx.md) (gate de viabilidade QNN e esquema exato de quantização para o QAT).
- **Entrega para:** [Fase 06](fase-06-avaliacao-benchmarks.md) (checkpoints mesclados) e [Fase 07](fase-07-exportacao-onnx.md) (checkpoints aprovados + conjunto de calibração).

## 15. Estimativa da fase

| Bloco | GPU local | Calendário |
|---|---:|---:|
| Ambiente, port do script, sondas (inclui T4 por bucket e llama.cpp), fila de GPU | ~6 h | 4–5 dias |
| Student-baseline sem professor (G2.4) | ~5 h | 1 dia |
| T0 (LoRA + full, com perda KD) | ~13 h | 2 dias |
| T1 (SFT + KD por logits) | ~26 h | 2–3 dias |
| T3 QAT w4a16 (após gate) | ~4 h | 2 dias |
| T2 = KapiTeacher 14B/7B | 0 h aqui (treino na Fase 02; alternativa KapiTeacher-7B à parte condicional ~35 h) | 0,5 dia |
| T4 (tools + raciocínio, local, §6.4c) | sondas ~4 h + **~57–70 h** (faixa 52–80) + merge/GGUF | 6–8 dias (bloco contínuo de 3–4 dias) |
| Agent-Lite 1.7B (thinking curto) | ~9 h | 1–2 dias |
| Opcionais (§9.4: DPO, QAT, DAPT, T2 alternativo, 2ª rodada do T4) | 0–235 h | conforme a ordem de corte |
| **Total desta fase** | **~125–140 h de GPU local** no plano base (sem nuvem; o scoring + top-k J13 e o piloto J15 contam nas Fases 02/01, §9.2) | **~6–7 semanas**, intercaladas com as Fases 04/06 (S6–S13) |
| **Programa inteiro (§9)** | **≈ 540–575 h de GPU com o KapiTeacher-14B (≈ 28–30 dias a 19 h/dia); ≈ 480–515 h com o 7B**; até ~810 h (14B) com todos os opcionais | **GPU ocupada de S1 a S14** (S1–S13 com o 7B; +2–3 semanas com opcionais) |
