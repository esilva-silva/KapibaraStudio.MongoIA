# Fase 02 — Preparação do professor (KapiTeacher-Mongo-14B / 7B)

> Depende de: [Fase 00](fase-00-visao-premissas.md) (premissas), [Fase 01](fase-01-selecao-modelos-base.md) (students por tier e escolha 14B × 7B), [Fase 03](fase-03-dataset-autocomplete-validacao.md) (dados e conjuntos de avaliação).
> Alimenta: [Fase 03](fase-03-dataset-autocomplete-validacao.md) (rótulos e ranking), [Fase 04](fase-04-dataset-agente-tools.md) (o professor pontua só código; agente usa outra estratégia), [Fase 05](fase-05-treinamento.md) (destilação), [Fase 06](fase-06-avaliacao-benchmarks.md) (gates).

Convenção: **[F]** = fato verificado (fonte citada) · **[H]** = hipótese/proposta a validar nesta fase.

> **Revisão local 2026-10-08:** sem GPU na nuvem — CPT, SFT, avaliação, destilação e professor auxiliar rodam na RX 7800 XT, com horas no calendário único da [Fase 05 §9](fase-05-treinamento.md). **O KapiTeacher passa a ser também o T2 Pro** (economiza o QLoRA de um segundo 7B; [Fase 05 §7](fase-05-treinamento.md)): a mistura do SFT acompanha a do T2 e o professor precisa passar também nos gates de T2 da [Fase 06](fase-06-avaliacao-benchmarks.md).
>
> **Revisão 2026-10-08b — troca do professor:** o professor deixa de ser o DeepSeek-Coder-6.7B base e passa a ser **Qwen2.5-Coder-14B Base** (primário) ou **Qwen2.5-Coder-7B Base** (alternativo, escolhido na Fase 01 pela regra do §1). DeepSeek-6.7B vira só referência/controle (medido, não treinado). Mudanças: checkpoints, tokenizer e contrato FIM Qwen (`QwenFimPromptBuilder`); QLoRA com VRAM/tempo separados para 14B e 7B; opção D do §3.4 removida; **KD por logits entra no plano base para T0/T1** (cache `topk-v1` com IDs uint32, 194 B/posição, extraído na mesma passada do scoring); GGUF Q6_K/Q5_K_M para o 14B; licença Apache-2.0 (some o Anexo A DeepSeek); horas recalculadas nos dois cenários (§8).

> **Resultado 2026-10-09 (J03b):** o professor escolhido é o **Qwen2.5-Coder-7B → KapiTeacher-Mongo-7B** (o 14B não coube com micro-batch ≥ 4k e treinou a 75 tok/s; [Fase 01 §6.6](fase-01-selecao-modelos-base.md)). Vale o cenário **7B** deste documento; o 7B mediu **309–360 tok/s** (não 500–560), então as horas de CPT/SFT do 7B aqui devem ser multiplicadas por ~1,5.

## 1. Objetivo

Transformar o checkpoint **cru** `Qwen/Qwen2.5-Coder-14B` (primário) ou `Qwen/Qwen2.5-Coder-7B` (alternativo) em **KapiTeacher-Mongo-14B** (ou **-7B**): um professor de domínio MongoDB/Kapibara que, **no contrato de prompt atual da IDE**, supera de forma mensurável o base zero-shot e os students v1, antes de ser usado para gerar, ranquear ou fornecer logits aos students T0…T4. "KapiTeacher-Mongo" designa qualquer um dos dois quando o tamanho não importa.

Checkpoints a baixar do Hugging Face (download autorizado pelo usuário no briefing; **revisão fixada por SHA de commit**, **somente safetensors**, sem `.bin`/pickle) **[H]**:

| Modelo | Repositório | Pasta proposta |
|---|---|---|
| Primário | `Qwen/Qwen2.5-Coder-14B` | `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-14B` |
| Alternativo | `Qwen/Qwen2.5-Coder-7B` | `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-7B` |
| Referência (não treinado) | `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\deepseek-coder-6.7b-base` (referência; cópia de `C:\IA\deepseek-coder-6.7b-base`) | — |

**Por que a troca (briefing item 3)** — números marcados **[F a confirmar no relatório técnico do Qwen2.5-Coder, arXiv 2409.12186]**:

1. HumanEval base ~47,6 (DeepSeek-6.7B) × ~61,6 (Qwen 7B) × ~64 (Qwen 14B): o ponto de partida é mais forte.
2. FIM nativo (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`) e contexto de 32k.
3. **Apache-2.0** elimina a cláusula de "Derivatives" da licença DeepSeek (destilação e dados sintéticos) e o Anexo A nos pacotes.
4. **Mesmo tokenizer dos students T0/T1/T3** (Qwen2.5-Coder): KD por logits passa a valer para eles, sem GOLD/ULD.
5. Um único contrato de prompt/adapter (`qwen2`) na IDE para todos os tiers de autocomplete, inclusive o T2 (que é o próprio KapiTeacher).

**Escolha 14B × 7B** (briefing item 2; o job de sonda é da [Fase 01](fase-01-selecao-modelos-base.md)): 14B se (a) VRAM pico ≤ 14 GB com micro-batch ≥ 4k tokens sem paginação, (b) throughput ≥ ~250 tok/s e (c) vantagem zero-shot ≥ 3 p.p. de FIM-acc ou exec-valid sobre o 7B; senão, 7B. Este documento descreve os dois cenários; onde os valores diferem, há uma coluna/linha por tamanho.

Não se parte do SlopCoder-Mongo-6.7B-v1. Motivos:

1. O professor v1 foi treinado em um contrato de chat que **não existe mais** (`LocalModelAiChatService`, removido pela ADR-055) **[F]** `docs/10-decisoes-arquiteturais.md` (IDE) e `src/.../Application/Agents/LocalAgentProvider.cs`.
2. O professor v1 **não era melhor que os students** no benchmark compacto: APT 3,16 no professor × 3,47 (0.5B) e 3,53 (1.5B) nos mesmos 500 exemplos; chat intent 4,5% **[F]** `SlopStudio.MongoAI-Compact\README.md`, `reports\benchmark\teacher-v1\report.md`. Um professor que não supera o aluno não acrescenta sinal, só custo.
3. A concordância greedy professor × referência foi 43% (FIM) e 38% (completion); a política final da v1 (`replace_policy: none`) quase não usou rótulos do professor (541 de 100.000 exemplos com `teacher_agree`) **[F]** `docs\04-destilacao.md` (v1).
4. Além disso, o v1 é DeepSeek (vocab 32.256, FIM e licença próprios): partir dele anularia a razão da troca (tokenizer Qwen e Apache-2.0).

## 2. Entradas

| Entrada | Caminho/fonte | Observação |
|---|---|---|
| Checkpoint primário | `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-14B` (de `Qwen/Qwen2.5-Coder-14B`) | Qwen2ForCausalLM; **48 camadas, hidden 5120, 40 heads, 8 KV heads (GQA), head_dim 128, vocab_size 152.064, 32k de contexto, rope_theta 1e6, embeddings não atrelados** **[F a confirmar no config.json]** |
| Checkpoint alternativo | `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-7B` (de `Qwen/Qwen2.5-Coder-7B`) | **28 camadas, hidden 3584, 28 heads, 4 KV heads, vocab_size 152.064, 32k** **[F a confirmar no config.json]** |
| Pesos | `model-*.safetensors` + `model.safetensors.index.json` | **somente safetensors** (sem unpickle); `revision=<SHA>` fixada no `base.json` |
| Tokenizer | `tokenizer.json` / `tokenizer_config.json` do Qwen2.5-Coder (BPE byte-level, ~151.665 tokens reais) | IDs FIM: `<\|fim_prefix\|>` 151659, `<\|fim_middle\|>` 151660, `<\|fim_suffix\|>` 151661, `<\|fim_pad\|>` 151662, `<\|endoftext\|>` 151643 **[F a confirmar]**; students 0.5B/1.5B têm vocab_size 151.936 (embeddings atrelados), mesmo tokenizer **[F a confirmar]** |
| Licença | `LICENSE` do repositório (Apache-2.0) **[F a confirmar no model card]** | sem cláusula sobre derivados/dados sintéticos; manter `LICENSE`, atribuição e aviso de modificação nos artefatos |
| Contrato FIM da IDE | `F:\source\esilva-silva\EsilvaSoft.KapibaraStudio\src\EsilvaSoft.KapibaraStudio.Application\QwenFimPromptBuilder.cs` | PSM sem BOS: `[<\|fim_prefix\|>] + prefixo + [<\|fim_suffix\|>] + sufixo + [<\|fim_middle\|>]`; exige que cada marcador codifique para **um** token (senão `InvalidDataException`); orçamento `contextTokens − 3`, sufixo até 25% (`budget / 4`), prefixo preenche o resto e é truncado pela esquerda **[F]** (lido; somente leitura) |
| Adapter/tokenizer .NET | `src\EsilvaSoft.KapibaraStudio.Infrastructure.LocalAi.OnnxAdapter\QwenCoderModelAdapter.cs` (cria `OnnxModelTokenizer`, exige FIM nos `added_tokens` do tokenizer) e `OnnxModelTokenizer.cs` | é o tokenizer .NET que a IDE usa para Qwen **[F]** |
| Contrato de contexto | `AutocompleteContextBuilder.cs` + `AiContext\EditorContextV1Contract.cs` | `editor-context-v1` congelado (cabeçalho CRLF, `ModelPrefix` com `\n` literal; a variante de treino DeepSeek que substituía `AVAILABLE COMMANDS` precisa ser reavaliada para o contrato Qwen) **[F]** |
| Contrato de chat local atual | `Application\Agents\LocalAgentProvider.cs` | chat = FIM com **sufixo vazio**, prefixo `/* Answer the instruction with code. ... */` contendo JSON `{instruction, context}`; greedy; até 256 tokens (`DefaultTokens`); sem tool calling ("O agente local não solicita tools") **[F]** |
| Referência (não treinado) | `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\deepseek-coder-6.7b-base` (referência) | LlamaForCausalLM, 32 camadas, hidden 4096, vocab 32.256, FIM 32015/32016/32017, licença DeepSeek v1.0; entra só como controle no baseline (§3.2) e no bake-off da Fase 01 **[F]** `config.json`, `tokenizer.json` |
| Dados e benchmarks | [Fase 03](fase-03-dataset-autocomplete-validacao.md) | `train`/`val`/`test`/`bench`/`blind` com manifesto |
| Ambiente GPU | venv ROCm da v1 | RX 7800 XT gfx1101 15,98 GB, PyTorch 2.10.0+rocm7.14, Transformers 4.57.6, PEFT 0.17.1, bitsandbytes 0.50.2 **[F]** `SlopStudio.MongoAI-Compact\docs\02-ambiente-gpu.md` |

## 3. Atividades

### 3.1 Verificação do checkpoint (dia 1)

Script: `scripts/teacher/verify_checkpoint.py` → `Repos\KapiTeacher-Mongo-14B\reports\checkpoint_verification.json` (um por checkpoint Qwen; o DeepSeek de referência já foi verificado na v1).

- [ ] **Integridade:** SHA-256 de cada `.safetensors` comparado aos hashes LFS publicados no Hugging Face para a revisão fixada (API do Hub / `oid sha256` dos ponteiros LFS). Registrar o SHA da revisão e os hashes no manifesto (`Repos\KapiTeacher-Mongo-14B\model.lock.json`).
- [ ] **Config:** conferir no `config.json` os campos da tabela acima (camadas, hidden, heads, KV heads, `rope_theta`, `tie_word_embeddings`, `max_position_embeddings`) — todos hoje **[F a confirmar]**. `vocab_size` 152.064 > ~151.665 tokens reais do tokenizer → as linhas do padding da `lm_head` são preenchimento. **Mascarar essas linhas** (−inf) em geração, em scoring e em top-k de logits; para KD com students (vocab 151.936) mascarar o padding de professor **e** de aluno e restringir às entradas reais comuns.
- [ ] **Tokenizer:**
  - os IDs FIM (151659/151660/151661) e `<|endoftext|>` 151643 batem com os da tabela e cada marcador codifica para **um** ID;
  - o professor, os students Qwen e o tokenizer da IDE compartilham o mesmo `tokenizer.json` (hash igual);
  - o pré-tokenizer trata `\r`/`\n` de forma consistente; o cabeçalho `editor-context-v1` usa CRLF fixo e o código do editor pode vir com CRLF (Windows) — os dois casos entram nos testes;
  - **paridade de IDs Python × C#** em 400 prompts (mesmo critério `tokenizer_samples: 400` da v1 **[F]** `configs\export.yaml` v1): builder Python (PSM Qwen, orçamento `contextTokens − 3`, sufixo ≤ 25%) × `QwenFimPromptBuilder` com o `OnnxModelTokenizer` da IDE, via `tools/ContractProbe` (.NET, referência somente leitura às DLLs da IDE) — ver [Fase 03 §4.1](fase-03-dataset-autocomplete-validacao.md).
- [ ] **FIM sanity:** 20 prompts fixos (`db.Users.find({` com sufixo `})`, chave dentro de `$group`, valor de `$match`); a saída deve parar em `<|endoftext|>` ou no eco do sufixo, sem gerar marcadores `<|fim_*|>`.
- [ ] **Janela:** perplexidade em 3 comprimentos (2k, 8k, 32k) para confirmar o contexto nativo de 32k; o contrato da IDE usa 2.048 tokens por padrão (`AutocompleteSettings.ContextTokens = 2048` **[F]**), então 4k basta no treino (e é o que cabe no 14B).

### 3.2 Baseline zero-shot no domínio (dias 1–3)

Script: `scripts/teacher/eval_teacher.py --model {deepseek-6.7b,qwen-7b,qwen-14b} --contract {editor-context-v1,kapi-context-v2,local-agent-fim-v0}` → `Repos\KapiTeacher-Mongo-14B\reports\baseline_base\`.

Medir **três** bases: DeepSeek-Coder-6.7B (controle histórico), Qwen2.5-Coder-7B e Qwen2.5-Coder-14B. O zero-shot FIM é o critério (c) da regra 14B × 7B do §1 (briefing item 2); o **job de sonda** (zero-shot + QLoRA de 1–2 h em cada Qwen, medindo tok/s e VRAM dedicada sem offload) **pertence à [Fase 01](fase-01-selecao-modelos-base.md)** — esta fase consome o resultado e só roda o baseline completo no(s) checkpoint(s) escolhido(s) e no DeepSeek de controle.

| Conjunto (da [Fase 03](fase-03-dataset-autocomplete-validacao.md)) | n | Métricas |
|---|---|---|
| `val` autocomplete (estratificado por papel do cursor e categoria) | 2.000 | EM, FIM-acc, APT@32, prefix match, sintaxe, **exec-valid** (executa no MongoDB Docker), **campos inexistentes** (alucinação contra o schema) |
| `val` chat curto (contrato `local-agent-fim-v0`) | 500 | intenção, exec-equivalência com a referência (mesmo resultado nos fixtures), respostas conversacionais/Markdown |
| `blind` (escrito à mão) | 400 (200 autocomplete + 200 chat, Fase 03 §4.6) | mesmas métricas; só leitura final |
| Generic-JS FIM (forgetting) | 500 | EM/FIM-acc em JS/TS/JSON fora do domínio (código MIT do próprio repositório da IDE e de projetos permissivos) |

Referências históricas (outro conjunto/contrato/modelo, só orientação) **[F]**: DeepSeek base cru na v1 — APT 2,93, EM 0,139, sintaxe 0,657, Mongo 0,911 (`teacher_inventory.md` v1); SlopCoder-1.5B-full — FIM-acc 0,425, EM autocomplete 0,318, chat intent 0,94 (`reports\benchmark\student-1.5b-full-merged\report.md`).

Também medir no baseline: os **students base da Fase 01** (T0/T1, agora Qwen) no mesmo conjunto — o gate G2.4 compara professor × melhor student.

### 3.3 Adaptação de domínio (dias 4–14)

Toda a adaptação é **QLoRA** (base em NF4, double quant, compute bf16), igual à v1 **[F]**: bf16 completo (14B ≈ 29,5 GB; 7B ≈ 15,2 GB **[H]**) não cabe com ativações nos 16 GB e o Windows pagina para memória compartilhada sem erro. Vale para o 14B e para o 7B.

#### Estágio A — CPT leve (continued pre-training) **[H]**

Objetivo: vocabulário e estilo do domínio (DSL do Console, EJSON, estágios/operadores, Atlas Search), mantendo o infilling nativo.

| Item | Valor inicial |
|---|---|
| Corpus | 15–20 M tokens: snippets validados da [Fase 03](fase-03-dataset-autocomplete-validacao.md) (Console/mongosh/JSON), documentos EJSON dos fixtures, catálogo da IDE renderizado como comentários JSDoc (texto próprio, MIT), scripts de exemplo da documentação da IDE; **sem** texto da documentação oficial MongoDB (licença CC BY-NC-SA 3.0, ver Fase 03 §2) |
| Replay | 10–15% de código genérico JS/TS/JSON permissivo para conter esquecimento |
| Transformação FIM | 50% dos documentos em **PSM no formato Qwen** (`<\|fim_prefix\|>` prefixo `<\|fim_suffix\|>` sufixo `<\|fim_middle\|>` alvo `<\|endoftext\|>`, sem BOS, idêntico ao `QwenFimPromptBuilder`) com corte aleatório por caractere; 50% causal |
| Sequência | 4.096 tokens com *packing* e máscara de atenção por documento |
| LoRA | r=64, alpha=128, dropout 0,05, alvos `q,k,v,o,gate,up,down` |
| Otimização | AdamW (paged 8-bit), lr 1e-4, cosine, warmup 2%, 1 época, ~16k tokens/passo efetivo |
| Script | `scripts/teacher/train_teacher.py --stage cpt --config configs/teacher_cpt.yaml` |

#### Estágio B — SFT no contrato FIM da IDE

Objetivo: o professor responde exatamente no formato que os students vão aprender.

| Item | Valor inicial |
|---|---|
| Dados | `train` da Fase 03 renderizado no formato **Qwen FIM (PSM)**, com os **mesmos textos de contexto** dos students: `editor-context-v1` (compatibilidade com pacotes atuais), o contrato candidato `kapi-context-v2` (decidido com a IDE; ver Fase 03 §2) e o chat `local-agent-fim-v0` |
| Mistura | **alinhada ao T2** (o professor é o T2 Pro): 48% FIM · 18% completion (sufixo vazio) · 20% chat curto (pt-BR/en ~50/50) · 6% correção · 4% negativos (API inexistente → forma correta) + 4% replay de código genérico; chat + correção ≤ 26% (lição v1 de degradação de grupos pequenos) |
| Tamanho | **~105 mil exemplos**, ~47 M tokens por época (~43 M de prompt + ~4 M de resposta; ~450 tokens/ex. na mistura acima, **estimado com o tokenizer DeepSeek — reconferir com o tokenizer Qwen** **[H]**); amostragem estratificada com sobre-amostragem de `hard` e grupos pequenos, como os students ([Fase 03 §4.6](fase-03-dataset-autocomplete-validacao.md)) |
| Perda | só nos tokens do *middle* + `<\|endoftext\|>`; contexto mascarado; CE calculada **em blocos** (o logit completo 4k × 152k não cabe) |
| LoRA | r=64, alpha=128 (continua o adapter do estágio A **ou** novo adapter sobre o merge do A — decidir pelo `val`) |
| Otimização | lr 1e-4 → 1e-5 cosine, **1 época** (it.1: 14B ~45–52 h · 7B ~24 h); it.2 = correção curta de ~0,3 época (14B ~15 h · 7B ~8 h) só se o `val` ou os gates de T2 apontarem lacuna; early stopping pelo `val` exec-valid |
| Script | `scripts/teacher/train_teacher.py --stage sft --config configs/teacher_sft.yaml` |

#### Estágio C — rejeição/auto-destilação (opcional, 1 rodada) **[H]**

O professor do estágio B gera best-of-4 no `train_pool`; só candidatas **exec-válidas e com campos válidos** e que discordam da referência entram como variações (peso 0,5). Só adotar se o `val` subir ≥ 1 ponto de exec-valid sem perder EM.

#### VRAM e tempo (estimativas **[H]**, medir no primeiro passo; briefing item 5)

QLoRA NF4, gradient checkpointing, CE em blocos. O bitsandbytes **não** quantiza embedding e `lm_head`: ficam em bf16 (embeddings não atrelados no 14B e no 7B **[F a confirmar]**).

| Item | 14B | 7B |
|---|---|---|
| Pesos | camadas NF4 ~7 GB + embed e `lm_head` bf16 ~3,1 GB ≈ **10 GB** | camadas NF4 + embed/`lm_head` bf16 ~2,2 GB ≈ **5,6 GB** |
| Micro-batch | **4k tokens** (4–8k no máximo; subir só se o pico permitir), `limit_vram` ligado | 8–12k tokens, `limit_vram` como na v1 |
| VRAM pico (LoRA r=64) | **~13–14,5 GB — no limite** | ~10–11 GB |
| Throughput treino | **~250–300 tok/s** (≈ 560 × 6,7/14,7) | **~500–560 tok/s** (igual ao DeepSeek 6.7B da v1, ~560 tok/s com r=32 e 12.288 tokens/micro-batch **[F]**, [Fase 05 §3](fase-05-treinamento.md)) |
| CPT 15–20 M tokens | **~15–22 h** | **~8–10 h** |
| SFT 47 M tokens (1 época) | **~45–52 h** | **~24 h** |
| Forward (scoring/top-k) | ~900 tok/s | ~1.900 tok/s |

| Fase comum | Recurso | Tempo |
|---|---|---|
| Merge bf16 (adapter → pesos) | **CPU** (64 GB de RAM; não ocupa a GPU) — o merged do 14B tem ~29,5 GB, cabe na RAM **[H]** | 20–40 min (14B) · 10–20 min (7B) |
| Conversão para GGUF (geração em lote no llama.cpp) | CPU; 14B → **Q6_K (~12 GB)** ou Q5_K_M, pois o Q8_0 (~15,7 GB) não cabe com folga; 7B → Q8_0 (~8 GB) | ~10–20 min + paridade top-1 ≥ 97% contra o HF NF4 em 200 prompts |
| Geração greedy NF4 (HF) | 14B ~10–11 GB + lotes · 7B ~8 GB, lotes de 16 | o HF NF4 do DeepSeek fazia 0,5–0,9 exemplo/s **[F]** v1; por isso a geração em lote vai para o llama.cpp |

Regra herdada da v1 **[F]**: nunca rodar benchmark ou outro processo GPU em paralelo ao professor (na v1 isso empurrou o professor para a memória compartilhada e interrompeu o best-of-N). No 14B, com pico de 13–14,5 GB, a regra é ainda mais crítica.

Se o 14B passar a paginar ou ficar abaixo de ~250 tok/s no primeiro passo medido, aplicar o plano de contingência do §6 (cair para o 7B) em vez de insistir.

### 3.4 Professor instrução/agente — vale a pena?

O Qwen2.5-Coder **Base** não segue instruções nem faz tool calling de forma confiável; na v1, mesmo o DeepSeek ajustado teve chat intent 4,5% e sintaxe 41% no modo instrução **[F]** v1 `docs\04-destilacao.md`. Opções:

| Opção | Custo | Ganho esperado | Licença | Decisão |
|---|---|---|---|---|
| A. Geradores programáticos + validação por execução real em MongoDB (Docker) | baixo/médio | chat programático saturou em ~94–98% na v1, mas pedidos livres ficaram em 56–72% **[F]** v1 README → falta **diversidade de linguagem** | sem restrição | **Adotada** (base de tudo) |
| B. SFT do KapiTeacher no contrato de chat `local-agent-fim-v0` (§3.3 B) | incluído | professor passa a **pontuar** e propor código para chat curto | Apache-2.0 | **Adotada** — mas só para código, não para escrever o pedido em linguagem natural |
| C. Professor auxiliar maior, instruct, **local** — candidato **Qwen3-14B** em `/no_think` (Apache-2.0, <https://huggingface.co/Qwen/Qwen3-14B>; o mesmo modelo do professor de raciocínio auxiliar da [Fase 04 §6.5](fase-04-dataset-agente-tools.md)); alternativa Qwen3-Coder-30B-A3B-Instruct (Apache-2.0, tool calling nativo, 256k ctx **[F]** <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct>) com experts na RAM | baixo/médio (~5 h de GPU para as paráfrases) | paráfrases pt-BR/en de pedidos, pedidos "livres" ancorados em schema, propostas de trajetórias de agente | Apache-2.0, sem restrição de uso das saídas | **[H] Adotar condicionalmente** (gate G2.6) |
| D. Qwen2.5-Coder-14B-Instruct como professor de instrução | médio (mais um modelo de 14B para baixar e rodar em GGUF; ~9 GB em Q4_K_M) | segue instruções de código, mas é da geração 2.5 (sem modo de raciocínio), **não resolve tool calling** e sobrepõe a opção C (Qwen3-14B, mais novo e já previsto para a Fase 04) | Apache-2.0 | **[H] Não adotada**: sem ganho sobre a opção C e sem resolver o agente; só reconsiderar como fonte extra de paráfrases se o Qwen3-14B reprovar no G2.6 |

(A antiga opção D, DeepSeek-Coder-6.7B-instruct, saiu: o modelo não é mais usado.)

Regras para a opção C **[H]**:

1. O auxiliar **nunca** gera rótulo sem validação: toda saída passa pelo mesmo pipeline da Fase 03 (parser, execução real, checagem de campos) e, para agente, pela execução real das ferramentas do Kapibara ([Fase 04](fase-04-dataset-agente-tools.md)).
2. Uso principal: **gerar o lado da instrução** (o pedido do usuário) a partir de uma query já validada (*back-translation*), e paráfrases de pedidos existentes. O código de referência continua programático/validado.
3. Execução local, sem nuvem: Qwen3-14B GGUF Q4_K_M (~9 GB) **inteiro na VRAM** via llama.cpp (build Windows ROCm ou Vulkan), 4–8 slots paralelos. Alternativa: Qwen3-Coder-30B-A3B-Instruct Q4_K_M (~18,6 GB) com parte dos experts na RAM de 64 GB (`--n-cpu-moe N` / `-ot`); numa placa de 16 GB com DDR5 variantes 30B-A3B com 25–30% na CPU geraram ~51–57 tok/s contra ~62 tok/s do Qwen3-14B inteiro na GPU ([benchmark 16 GB](https://www.glukhov.org/post/2026/01/choosing-best-llm-for-ollama-on-16gb-vram-gpu/), RTX 4080 + DDR5-6000) **[F externo]**; na RX 7800 XT com DDR5-4800 espera-se menos **[H]**. Sonda de 50 prompts de cada (job J05 da [Fase 05 §9.2](fase-05-treinamento.md)) antes de planejar volume.
4. Nunca rodar ao mesmo tempo que o KapiTeacher na GPU.

### 3.5 Modos de uso do professor na destilação

Script único com subcomandos: `scripts/teacher/distill.py {score,generate,best_of_n,topk}`; saídas em `data/distilled/<dataset_version>/teacher-<teacher_version>/`.

#### Sequence-level (todos os tiers)

| Modo | Como | Quando o rótulo muda |
|---|---|---|
| **Scoring** | logprob médio e mínimo da referência, só nas posições da resposta, no contrato do professor | nunca muda sozinho; usado para filtrar (remover os 5% mais imprevisíveis **quando** o professor também discorda — regra v1 **[F]**) e para *curriculum* |
| **Greedy** | temperatura 0, até 96 tokens (autocomplete) / 256 (chat) com o pós-processamento da IDE (corte de marcadores e de eco do sufixo) | concordância (igual ou extensão) → peso 1,0 |
| **Best-of-N** | N=4, T=0,7, top-p 0,95; candidatas validadas por parser + execução + campos; ranking por **(exec-valid, equivalência semântica com a referência, logprob médio normalizado pelo comprimento)** | substitui a referência **só** se a referência falha na validação e a candidata passa; caso contrário candidata válida e equivalente vira variação (peso 0,5) |

Lição da v1 **[F]**: trocar rótulo por margem de logprob (0,35) causou trocas prejudiciais (`insertOne({` → `aggregate` inteiro). Na v2 a troca exige **evidência de execução**, nunca só logprob.

**Escopo e motor locais** **[H]** (dimensionados ao orçamento de GPU, [Fase 05 §9](fase-05-treinamento.md)):

| Modo | Escopo | Motor | GPU 14B | GPU 7B |
|---|---|---|---:|---:|
| Scoring **+ top-k na mesma passada** | só o **pool do T1** (~230 mil exemplos, ⊇ T0 e T4-chat; ~110 M tokens), não o `train_pool` de ~1,0 M | HF NF4, **um único forward** que grava logprob da referência e o cache `topk-v1` das posições de resposta (forward ~900 tok/s no 14B, ~1.900 tok/s no 7B) | **~34 h** | **~16 h** |
| Greedy | ~40 mil exemplos selecionados pelo scoring (20% de menor logprob + amostra estratificada de controle) | **llama.cpp** com o merged em GGUF (14B **Q6_K/Q5_K_M**, 7B Q8_0), 16 slots; o HF NF4 em lotes de 16 faz 0,5–0,9 ex./s **[F]** v1 (DeepSeek), o llama.cpp deve fazer ~3–5 ex./s no 7B e menos no 14B **[H]** | ~15 h (greedy + best-of-4) | ~10 h (greedy + best-of-4) |
| Best-of-4 | ~10 mil prompts de chat/correção em que a referência falhou, o professor discordou ou o exemplo é `hard` | llama.cpp (mesmo GGUF) | (incluído acima) | (incluído acima) |

O llama.cpp serve só para **gerar candidatas**; logprob, aceitação e troca de rótulo continuam decididos por execução e pelo scoring em HF. Antes de usar o GGUF: **top-1 igual ao HF NF4 em ≥ 97% das posições em 200 prompts** (Q6_K/Q5_K_M no 14B, briefing item 10) e tokenização Qwen idêntica (marcadores FIM como um ID cada). Se o Q6_K do 14B falhar o limiar, subir para Q8_0 com offload parcial só para a geração (mais lento) ou usar o 7B (§6).

#### KD por logits (T0/T1; T3 herda os pesos)

Com o professor e os students no **mesmo tokenizer Qwen2.5-Coder**, KD por logits entra no **plano base** para T0 e T1 (briefing item 6); T3 herda os pesos do T0/T1. Não depende mais de nenhum student DeepSeek: o `T1-ds` (DeepSeek-Coder-1.3B) e o braço C do piloto saem do plano, e GOLD/ULD (KD entre tokenizers) deixa de ser necessário. O piloto da Fase 01 vira **A** = Qwen 1.5B só sequence-level × **B** = A + KD por logits do KapiTeacher.

Perda: `α·CE + (1−α)·τ²·KL` (forward KL no autocomplete, τ=1–2, α 0,3–0,5 inicial) **[H]**. Os logits do professor (vocab 152.064) e do aluno (151.936) são restringidos às entradas reais do tokenizer, com as linhas de padding mascaradas dos dois lados.

Formato de armazenamento (`topk-v1`, revisado — IDs **uint32** porque o vocab > 65.536):

| Campo | Tipo | Observação |
|---|---|---|
| `example_id` | string | chave no JSONL da Fase 03 |
| `positions` | uint16[n] | posições da resposta (middle + `<\|endoftext\|>`) |
| `topk_ids` | **uint32**[n, k] | vocab 152.064 > 65.536 → 4 bytes por ID |
| `topk_logprobs` | float16[n, k] | log-softmax em T=1, linhas de padding mascaradas |
| `tail_mass` | float16[n] | massa fora do top-k (para KL com "bucket" residual) |
| `teacher_hash` | string | SHA-256 do merged + versão do contrato |

Shards `safetensors` de ~512 MB + índice Parquet (`example_id → shard, offset`). Script: `distill.py topk --k 32 --positions response` (na prática executado junto com `score`, na mesma passada).

Custo de disco (k=32: 32×4 + 32×2 + 2 = **194 bytes/posição**) **[H]**:

| Escopo | Posições | Disco |
|---|---:|---:|
| Só respostas, ~230 mil exemplos × ~25 tokens (pool do T1) — **recomendado** | ~5,8 M | **~1,1 GB** |
| Só respostas, k=64 (64×4 + 64×2 + 2 = 386 B) | ~5,8 M | ~2,2 GB |
| Só respostas, 400 mil exemplos × ~25 tokens, k=32 | ~10 M | ~1,9 GB |
| Prompt + resposta (300 tokens/ex.), k=32 | ~69 M (230 mil ex.) | ~13,4 GB |

Recomendação: só respostas, k=32, sobre o pool do T1, **na mesma passada do scoring** (custo marginal só de escrita; as horas são as do scoring: 14B ~34 h, 7B ~16 h). O piloto da Fase 01 (braço B) usa só 20k exemplos: 14B ~1,0 h · 7B ~0,5 h do forward **[H]**. Antes de gerar em escala, medir o KL NF4 × bf16 em 200 prompts: se > 0,02 nat/token, extrair logits em bf16 com lotes menores (no 14B isso exige CPU offload parcial e é lento; avaliar antes de comprometer 34 h).

### 3.6 Gates do professor (precisa superar antes de ensinar)

| Gate | Critério mensurável | Conjunto |
|---|---|---|
| G2.1 Checkpoint | hashes = HF (revisão fixada); `config.json` e IDs FIM = tabela do §2; paridade de IDs Python × C# (`QwenFimPromptBuilder` + tokenizer .NET da IDE) = 100% em 400 prompts; FIM sanity 20/20 sem marcadores; padding do vocab mascarado | §3.1 |
| G2.2 Ganho sobre o base | FIM-acc ≥ base **do mesmo tamanho** + 15 p.p.; exec-valid autocomplete ≥ 0,95; campos inexistentes ≤ 2% | `val` |
| G2.3 Chat curto (código) | exec-equivalência ≥ 0,85 no `val` chat e ≥ 0,75 no `blind`; 0% Markdown/conversacional | `val`, `blind` |
| G2.4 Supera os alunos | FIM-acc e exec-valid ≥ melhor student **treinado sem professor** (baseline da Fase 05, mesmo dado) + 5 p.p.; se não, a destilação vira só filtragem/validação (documentar) | `val` |
| G2.5 Sem esquecimento grave | queda ≤ 3 p.p. no Generic-JS FIM em relação ao base | §3.2 |
| G2.6 Professor auxiliar (se usado) | ≥ 70% das saídas passam no validador; revisão humana de 200 amostras com ≥ 90% "pedido natural e fiel à query" | Fase 03/04 |
| G2.7 Contaminação | 0 exemplos de `test`/`bench`/`blind` no treino do professor (verificação do manifesto) | manifesto |
| G2.8 Papel de T2 (não bloqueia o uso como professor) | o **próprio KapiTeacher (14B ou 7B)** exportado em **DML INT4** passa nos gates de T2 da [Fase 06](fase-06-avaliacao-benchmarks.md): APT × T1 em caracteres, chat livre/cego, exec match, TTFT e VRAM no adapter `qwen2` da IDE (14B: ~8,5–9 GB + KV ~192 KiB/token → ~0,8 GB a 4k, cabe em 16 GB, TTFT maior que o 7B, **a medir**; 7B: ~5 GB). Reprovado → T2 sai da v2 ou, no cenário 14B, cai para o KapiTeacher-7B se houver folga no calendário ([Fase 05 §9.4](fase-05-treinamento.md), O4) | Fase 06 |

O `blind` é lido **uma** vez por versão de professor candidata (no máximo 3 leituras no total da fase), para não virar conjunto de ajuste.

## 4. Artefatos e estrutura de pastas

```text
C:\IA\models\KapibaraStudio.MongoIA\
  configs\teacher_cpt.yaml, teacher_sft.yaml, teacher_distill.yaml
  scripts\teacher\verify_checkpoint.py, eval_teacher.py, build_cpt_corpus.py,
                  train_teacher.py, merge_teacher.py, distill.py, aux_teacher_paraphrase.py
  Repos\KapiTeacher-Mongo-14B\   # workspace do professor (T2)
    model.lock.json                         # ponteiros + SHA da revisão HF + hashes dos checkpoints (pesos não são copiados)
    adapters\kapiteacher-cpt-v1\  kapiteacher-sft-v1\
    merged\KapiTeacher-Mongo-14B-v1\  # bf16, com LICENSE Apache-2.0 e aviso de modificação (ou KapiTeacher-Mongo-7B-v1)
  data\distilled\<dataset_version>\teacher-<teacher_version>\
    scoring.parquet, greedy.jsonl, best_of_n.jsonl, topk\shard-*.safetensors, topk\index.parquet
  Repos\KapiTeacher-Mongo-14B\reports\checkpoint_verification.json, baseline_base\, cpt\, sft\, gates.md, kl_nf4_bf16.json
BaseRepos\Qwen2.5-Coder-14B\  BaseRepos\Qwen2.5-Coder-7B\   # checkpoints baixados (safetensors, revisão fixada)
```

Entregáveis:

- [ ] `checkpoint_verification.json` aprovado (G2.1) para o(s) checkpoint(s) Qwen
- [ ] baseline zero-shot dos três bases (DeepSeek-6.7B controle, Qwen 7B, Qwen 14B) e dos students base (`Repos\KapiTeacher-Mongo-14B\reports\baseline_base\`)
- [ ] adapters CPT/SFT + merged `KapiTeacher-Mongo-14B-v1` (ou `-7B-v1`) com model card, `LICENSE` **Apache-2.0**, atribuição ao Qwen2.5-Coder e **aviso de modificação**
- [ ] `gates.md` com G2.1–G2.7
- [ ] artefatos de destilação (scoring, greedy, best-of-N, top-k) com manifesto e hashes
- [ ] relatório de viabilidade do professor auxiliar (throughput, taxa de aprovação, amostra revisada)
- [ ] nota para a [Fase 08](fase-08-integracao-kapibara.md): todos os modelos treinados ficam Apache-2.0 (bases Qwen2.5-Coder/Qwen3); manter atribuição e aviso de modificação; nenhum Anexo A DeepSeek nos pacotes

## 5. Critérios de saída

- G2.1–G2.5 e G2.7 aprovados; G2.6 aprovado **ou** professor auxiliar descartado com registro.
- G2.8 avaliado na Fase 06 depois da exportação (job J27–J28 da [Fase 05 §9.2](fase-05-treinamento.md)); não atrasa a destilação.
- Se G2.4 falhar após 2 iterações de SFT, a Fase 05 segue com destilação **só por filtragem/validação** (o professor pontua e filtra, não rotula; o cache top-k não é usado) — decisão registrada em `gates.md`.
- Artefatos de destilação versionados, com hash do professor e do dataset no manifesto.

## 6. Riscos e mitigação

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Professor não supera students (repete v1) | média | alto | G2.4 explícito; CPT + SFT no contrato atual; plano B só filtragem |
| Offload silencioso para memória compartilhada (Windows/ROCm), pior no 14B (pico 13–14,5 GB) | alta | médio | `limit_vram` (v1), processo GPU exclusivo, OOM explícito, micro-batch 4k no 14B |
| **14B não cabe / é lento demais** (pico > 14 GB, paginação ou < ~250 tok/s) | média | alto | critérios (a)/(b) da regra do §1 medidos na sonda da Fase 01; se violados na sonda ou no primeiro passo do CPT → **cair para o KapiTeacher-7B** (pico ~10–11 GB, ~500–560 tok/s, calendário ~50 h menor); nunca treinar o 14B com paginação |
| **GGUF do 14B em Q8_0 não cabe** (~15,7 GB) | alta | médio | usar **Q6_K (~12 GB)** ou Q5_K_M para greedy/best-of-N; validar top-1 ≥ 97% contra o HF NF4; se reprovar, subir o quant com offload parcial ou usar o 7B (Q8_0 ~8 GB) |
| bitsandbytes/ROCm instável em versão nova | média | alto | congelar o venv da v1; não atualizar durante a fase |
| Esquecimento de FIM/código genérico no CPT | média | médio | 50% FIM (PSM Qwen) no CPT, replay 10–15%, G2.5 |
| Contrato de contexto v2 muda depois do SFT | média | médio | dados em formato estruturado (Fase 03) e re-render; SFT curto o bastante para repetir (no 14B, repetir custa ~45–52 h: congelar o contrato antes) |
| Padding do vocab (152.064 vs ~151.665 reais; students 151.936) corrompe KD/top-k | média | médio | mascarar padding de professor e aluno; teste de alinhamento de IDs no G2.1 |
| Cache top-k/KL com NF4 distorcido no 14B | baixa | médio | medir KL NF4 × bf16 em 200 prompts antes de extrair; extração é a mesma passada do scoring (~34 h no 14B) — um erro custa a passada inteira |
| Auxiliar lento demais localmente (30B-A3B com experts na RAM) | média | baixo | primário já é o Qwen3-14B inteiro na VRAM; sonda J05; volume de paráfrases limitado (≤ 30% do chat) |
| Professor como T2 fraco em pt-BR (treino EN/ZH) ou TTFT alto do 14B em DML INT4 | média | médio | mistura do SFT alinhada ao T2 (20% chat, metade pt-BR); gate G2.8 com TTFT/VRAM medidos; alternativa KapiTeacher-7B condicionada ao calendário |
| Disputa de GPU com outros jobs (bake-off, rollouts, students) | alta | médio | fila única e calendário da [Fase 05 §9](fase-05-treinamento.md); nada roda ao lado do professor |
| NF4 distorce logits do top-k | baixa | médio | medir KL NF4 × bf16 antes de extrair |

## 7. Dependências

- [Fase 01](fase-01-selecao-modelos-base.md): **decide 14B × 7B** (job de sonda: zero-shot dos três bases + QLoRA de 1–2 h em cada Qwen, regra do §1) e fixa os students Qwen por tier; o piloto de destilação A × B usa o KapiTeacher.
- [Fase 03](fase-03-dataset-autocomplete-validacao.md): `train`, `val`, `blind`, validadores e fixtures Docker precisam existir antes do §3.2.
- IDE: definição do contrato `kapi-context-v2` (ou permanência em `editor-context-v1`) antes do SFT final; adapter `qwen2` e `QwenFimPromptBuilder` já existentes (somente leitura).
- Download dos checkpoints Qwen do Hugging Face (revisão fixada) antes do §3.1; ~29,5 GB (14B) + ~15,2 GB (7B) de safetensors **[H]**.

## 8. Estimativa

Dois cenários, conforme a escolha da Fase 01. GPU em horas, **[H]** (vazões do §3.3 e §3.5; medir no primeiro passo).

**Cenário A — KapiTeacher-Mongo-14B (primário)**

| Bloco | Esforço humano | GPU |
|---|---:|---:|
| Verificação + baseline (3 bases) + avaliações do professor | 3–4 dias | ~10 h (J04, avaliações) |
| Corpus CPT + CPT | 2 dias | 15–22 h (J08) |
| SFT it.1 (45–52 h) + it.2 curta (~15 h) | 5–6 dias | 60–67 h (J09, J11, J12) |
| Destilação: scoring + top-k na mesma passada (pool T1) | 2 dias | ~34 h (J13) |
| Destilação: greedy + best-of-N no llama.cpp (Q6_K/Q5_K_M) | 1–2 dias | ~15 h (J14) |
| Professor auxiliar (sondas + paráfrases, Qwen3-14B local) | 2–3 dias | ~7 h (J05, J07; fora da janela do KapiTeacher) |
| **Total** | **14–18 dias úteis** | **~135–150 h de GPU** (sem o auxiliar; **~142–157 h** com ele) |

**Cenário B — KapiTeacher-Mongo-7B (alternativo)**

| Bloco | Esforço humano | GPU |
|---|---:|---:|
| Verificação + baseline (3 bases) + avaliações do professor | 3–4 dias | ~10 h |
| Corpus CPT + CPT | 2 dias | 8–10 h |
| SFT it.1 (~24 h) + it.2 curta (~8 h) | 4–5 dias | ~32 h |
| Destilação: scoring + top-k na mesma passada | 2 dias | ~16 h |
| Destilação: greedy + best-of-N (Q8_0) | 1–2 dias | ~10 h |
| Professor auxiliar | 2–3 dias | ~7 h |
| **Total** | **12–16 dias úteis** | **~76–78 h** sem o auxiliar; **~85–95 h** com o auxiliar e folga de 5–10 h para repetições |

Comparação com o plano anterior (DeepSeek-6.7B, ~85–90 h, KD só opcional): o 7B custa cerca de +0 a +5 h e já inclui o KD; o 14B custa cerca de +50–60 h sobre o 7B. Impacto no calendário da [Fase 05 §9](fase-05-treinamento.md) (briefing item 5): 14B ≈ +60–75 h de GPU no plano base (≈ 540–575 h no total); 7B ≈ +5–15 h. Sem nuvem.
