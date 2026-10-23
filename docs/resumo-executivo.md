# KapiCoder-Mongo — resumo executivo do estado (2026-10-22)

Documento de leitura rápida sobre onde o programa está, o que foi provado, o que **não** foi provado e o que depende de decisões ou da IDE.
Os números abaixo vêm dos relatórios em `reports/` e `Repos/*/reports/`; cada seção aponta a fonte.

## 1. Em uma frase

Os modelos de autocomplete de MongoDB **T0 (0.5B)** e **T1 (1.5B)** foram treinados com dados sintéticos validados por execução, destilados de um professor 7B,
exportados para ONNX e medidos; **funcionam bem no conjunto sintético**, mas **ainda não há prova de qualidade em dados reais nem medição na IDE**, e três mudanças
na IDE (cache de prefixo, correção do EOS e contrato do agente) bloqueiam os gates de latência e a Fase 04.

## 2. Estado por fase

| Fase | Estado | Evidência |
|---|---|---|
| 00 Visão e estrutura | Concluída (workspace Git, 25 commits) | `docs/fase-00-*.md` |
| 01 Modelos base | Parcial: bake-off FIM e professor 7B decididos; sondas de atenção do T4 e exportabilidade feitas; falta tool calling/raciocínio do Qwen3 e piloto de KD | `reports/teacher_choice.json`, `reports/probe/` |
| 02 Professor | **Concluída com exceções** (ver §4) | `reports/teacher_adoption.json` |
| 03 Dataset de autocomplete | **Concluída tecnicamente** (v2c); faltam cego humano, revisão e licenças | `reports/datasets/v2c-validation-report.md` |
| 04 Dataset do agente (T4) | **Não iniciada**: depende dos PRs 2 e 3 na IDE | `docs/prs/` |
| 05 Treinamento | Professor, T0 e T1 treinados; T4 só sondado (16k reprovado, plano B 8k+12k r32) | `Repos/*/reports/` |
| 06 Avaliação | Harness estatístico pronto; gates em rascunho; cego não selado | `configs/gates.yaml`, `scripts/eval/` |
| 07 Exportação ONNX | **T0 e T1 exportados** (14 pacotes) e medidos em proxy; NPU/QNN não feita | `reports/export/onnx_export_t0_t1.md` |
| 08 Integração na IDE | Não iniciada; 4 PRs especificados, nenhum implementado | `docs/prs/pr-1..4-*.md` |

## 3. Resultados principais

### Dataset (v2c, `data/validated/v2c`, fora do Git)
- Treino 1.247.125 · val 45.381 · test 28.455 · bench 16.938; 110 schemas sintéticos em 14 domínios (education e legal só em avaliação).
- Validado por execução real no MongoDB 8.0.32 (WSL); decontaminação por conteúdo (`ngram_min_content: 2`); **G3.3 passa**.
- **Pendentes:** conjunto cego escrito por humano (há 40 rascunhos de agente, não selados), revisão humana (G3.7), licenças (G3.9).

### Professor KapiTeacher-Mongo-7B (adapter `sft_v2d/final`)
- FIM acc +19,85 p.p. sobre o 7B base, Generic-JS +15,2 p.p., contaminação 0, campos inexistentes com schema visível 1,88% (limite 2%, margem de 0,12 p.p.).
- **v2d é estatisticamente equivalente ao v2c**; a escolha se apoia em margem mínima.

### Alunos (val v2c, n=2.000, mesmos ids)

| | FIM acc | Exec-valid | Exec ÷ teto | Campos `full` | Generic-JS |
|---|---:|---:|---:|---:|---:|
| 0.5B base | 0,076 | 0,213 | — | 4,13% | 0,340 |
| **T0 Nano** (distilado) | 0,403 | 0,582 | 0,931 | 2,06% | 0,518 |
| 1.5B base | 0,142 | 0,316 | — | 3,76% | 0,406 |
| **T1 Small** (distilado) | 0,411 | 0,595 | **0,951** | **1,58%** | 0,632 |
| Professor 7B v2d | 0,410 | 0,576 | 0,921 | 1,88% | 0,710 (7B base 0,558) |

- T1 × T0: exec-valid +1,3 p.p. (p=0,0016); FIM acc +0,75 p.p., **não significativo**.
- Distilado × controle gold (T0): exec-valid +1,7 p.p. (Holm p=0,019); FIM acc −0,85 p.p. n.s.
- **G2.4 do professor reprova (−0,15 p.p.)**: um 0,5B treinado direto no gold iguala o professor 7B neste conjunto. O papel do professor passou a **filtro/validação**.

### Exportação ONNX (onnxruntime-genai 0.15.2, `model.type=qwen2`)
- Melhores variantes: **T0 int4** (392 MB, CPU), **T0/T1 dml-fp16** (GPU; T1: TTFT 148/170 ms, 104 tok/s). T1 int4 perde qualidade e fica fora.
- Qualidade ONNX × HF bf16 indistinguível no T0 e no T1 int8/fp16; T1 int4 perde 1,4–1,8 p.p.
- **TTFT em CPU (proxy):** sem cache, reprova (T0 258/474 ms; T1 667/1.101 ms). Com **cache de prefixo por aba** o protótipo passa (T0 92/221; T1 250/497), com saída idêntica,
  desde que a taxa de acerto do cache fique ≥ ~95%.

## 4. O que NÃO está provado (leia antes de decidir)

1. **Tudo foi medido em dado sintético** (templates + fixtures). O 0,5B treinado no gold iguala o professor 7B, o que indica que o conjunto satura por dados, não por tamanho.
   Sem o **cego humano** não sabemos quanto generaliza para consultas reais.
2. **G2.2 do professor não está aprovado:** exec-valid ÷ teto = 0,92 (< 0,95). O aceite é uma exceção registrada; o limite de campos foi recalibrado para `context_kind=full` **depois** de ver a análise de erros (decisão registrada antes do retreino, mas é recalibração).
3. **O teto do validador é 0,6255:** o parser do Console cobre só um subconjunto; parte do que parece erro do modelo é limite do validador.
4. **Sem schema visível o erro de campo continua em 11–14%**; o treino não resolveu isso. A melhoria real depende de a IDE enviar o schema da coleção.
5. **O cache de prefixo só foi validado em protótipo Python;** a taxa de acerto real depende do uso na IDE (o dataset sintético embaralha o contexto e dá só ~0 pares reutilizáveis no envelope inteiro).
6. **Gates oficiais de TTFT/RAM/paridade não foram medidos pelo KapiLab** (exige token de GPU que só o usuário pode liberar); os números acima são proxy e `qualified=false`.
7. **`gates.yaml` está em rascunho** (4–5 limiares sem número; o de paridade tokenizer/prompt/greedy está em aberto) e o baseline `base_v1_0.5b` na suíte v2 não existe, então `T0.apt_*` e `exec_match` ficam sem avaliação.

## 5. Bloqueios e quem desbloqueia

| # | Bloqueio | Efeito | Quem |
|---|---|---|---|
| B1 | **Bug do EOS na IDE** (`OnnxLocalModelRuntime` duplica o último token; concordância IDE×Python 12–13%) | Reprova `python_agreement` em todos os pacotes | IDE (PR) ou contorno no pacote (decisão do usuário) |
| B2 | **Cache de prefixo por aba** (`docs/prs/pr-4-*.md`) | Sem ele os gates de TTFT em CPU reprovam | IDE (PR) |
| B3 | **PR 1** (catálogo de conhecimento e paridade byte a byte), **PR 2** (contrato do agente), **PR 3** (execução real de tools) | Paridade `contract diff`; **Fase 04 e o T4 inteiros** | IDE (PRs) |
| B4 | **Conjunto cego** (400 itens humanos) e revisão de 200 amostras | G3.6/G3.7; única medida fora do sintético | Usuário |
| B5 | **Token de GPU do KapiLab** | Gates oficiais de GPU | Usuário |
| B6 | **Vivobook X1-26-100** (NPU, QNN) | T3 inteiro | Usuário (hardware) |
| B7 | **Q1:** T1/T2 ainda treinam o chat curto? | Escopo do chat (hoje fora) | Usuário |

## 6. Decisões em aberto do usuário

1. Corrigir o EOS na IDE, aplicar o contorno no `genai_config.json` do pacote, ou os dois.
2. Redefinir o gate de TTFT em CPU como "gate de sessão com hit ≥ 95%" (exige B2) ou manter o gate sem cache (reprova).
3. Variante do T0: int4 (392 MB) ou int8-emb4 (633 MB); recomendado int4.
4. Q1 (chat curto), e se o KD por logits (adiado) entra no ciclo seguinte.
5. Confirmar `configs/rounds.yaml` (rascunho) e congelar `configs/gates.yaml`.
6. Qual tier vai primeiro para o usuário final: **T1 DML FP16** (GPU, passa os gates proxy) e **T0 int4** (CPU, depende do cache).

## 7. Incidentes e lições registrados

- **Uso da GPU integrada:** o J12b rodou horas na iGPU por confiar no índice `Vulkan0`; a seleção do dispositivo agora é por nome com guarda de velocidade (`scripts/distill/generate_teacher.py`).
  Todo job de GPU deve confirmar o adaptador (RX 7800 XT, LUID verificável por contadores).
- **Reinício da máquina** matou o treino do professor uma vez; a retomada por checkpoint funcionou sem perda (loss reproduzida).
- **Fila da GPU** (`scripts/gpu_queue.ps1`): bug do heartbeat corrigido; `-Status` só funciona no PowerShell 7 (`pwsh`).
- **Limites de uso** derrubaram subagentes várias vezes; os trabalhos foram retomados sem refazer o que estava íntegro.

## 8. Recursos e reprodutibilidade

- **Código e relatórios:** Git local, 25 commits (último: protótipo do cache de prefixo). Pesos, dumps, GGUF, pacotes ONNX e `data/` ficam fora do Git.
- **Disco:** ~650 GB livres em C: após merged (15 GB), 3 GGUF (28 GB) e pacotes ONNX.
- **Horas de GPU gastas (aprox.):** professor SFT v2c 21 h + v2d 8 h, destilação ~20 h, T0 3,4 h (+3,4 h controle), T1 11,8 h, avaliações ~6 h (estimativas a partir dos logs da fila, não somadas por job).
- **Para reproduzir:** `Repos/*/README_job.md`, `scripts/*/README*.md`, `configs/gpu_queue.yaml` e `reports/` guardam comandos, seeds e hashes.

## 9. Próximos passos recomendados (ordem)

1. **Usuário:** escrever ou revisar o cego humano (B4) e avaliar T0/T1 nele; é a medida que decide se o trabalho serve na prática.
2. **IDE:** abrir os PRs de EOS e do cache de prefixo (B1/B2), e o PR 1 (paridade); sem eles os gates de CPU e a Fase 08 não avançam.
3. **Decidir** as seis questões de §6 e congelar `gates.yaml`.
4. **Depois do cego e do PR 1:** reavaliar a decisão T0 × T1 e, se o desempenho real for menor que o sintético, voltar ao corpus (schemas visíveis, mais variedade) antes de gastar GPU.
5. **Só então** iniciar a Fase 04 (T4 agente), que depende dos PRs 2 e 3.
