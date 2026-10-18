# Job de treino do KapiCoder-Mongo-Small (T1, Qwen2.5-Coder-1.5B): distilado x controle (J15 x J10)

Estado: **preparado e enfileirado, NAO iniciado** (so smoke de 20 passos + retomada; `reports\student\smoke_report.md` na raiz). Base `BaseRepos\Qwen2.5-Coder-1.5B`
(rev. `df3ce67c…`), workspace `Repos\KapiCoder-Mongo-Small`. Escopo: **autocomplete FIM** `editor-context-v1` (chat curto fora ate a Q1; KD por logits adiado).
Spec: `docs\fase-05-treinamento.md` §6.2/§7/§9, gates `configs\gates.yaml` (T1) e `configs\gates.teacher.yaml` (G2.4), regra `configs\rounds.yaml` (draft).
Python (venv ROCm): `$PY = "C:\SlopStudio.MongoAI-artifacts\.venv\Scripts\python.exe"`, sempre com `-I`.

## 0. O que existe

| artefato | caminho |
|---|---|
| corpus (gitignored) | `data\student\{distill,gold}\t1\` (10 shards, `val\`, `manifest.json`); selecao em `data\student\selection\t1.jsonl`; relatorio `reports\student\corpus_report.md` |
| construtor / treino / avaliacao | `scripts\student\{build_student_corpus,train_student,eval_student}.py`; configs `configs\sft_distill_v1.yaml`, `sft_gold_v1.yaml`; `configs\student_corpus.yaml` |
| fila | `J17a`, `J15-T1-distill`, `J17-T1-distill-eval`, `J10-T1-gold`, `J10e-T1-gold-eval` em `configs\gpu_queue.yaml` |
| lock | `model.lock.json` (atualizar com `scripts\student\write_model_lock.py` apos o treino) |

**Corpus T1**: 229.834 exemplos (217.306 Mongo + 12.528 replay), 63,7 M tokens/epoca (media 277 tok/ex.), 4.563 linhas de <= 16.384 tokens, 2.281 passos/epoca de ~32,8k tokens
-> **4.562 passos em 2 epocas (127,4 M tokens)**. Mongo: fim_ctx 68,6%, completion 26,3%, fim_raw 5,1% (gold nos dois bracos); replay = 12,1% dos tokens. Mesmos ids/ordem/linhas nos dois bracos;
54,1% dos exemplos Mongo tem rotulo do professor diferente do gold. Balanco dos estratos (cut_kind x categoria x context_kind) a <= 0,7 p.p. do pool de 300k.
Os tokens ficam abaixo dos ~100 M da spec porque o prompt medido tem ~270 tokens (a spec supunha ~370); o volume em exemplos segue a spec.

## 1. Metodo e hiperparametros (iguais nos dois bracos)

**LoRA r=64, alpha=128, dropout 0,05** em q,k,v,o,gate,up,down sobre a base bf16, AdamW (betas 0,9/0,95, wd 0), LR **1,5e-4**, warmup 3%, cosseno ate 10%, 32.768 tokens/passo, 2 epocas.
Por que nao full FT (medido): mestre fp32 + AdamW 8-bit estoura o teto de 14,1 GB no 1o passo; bf16 puro estoura no 3o passo (12,8 GB, 830 tok/s) e perde atualizacoes pequenas. LoRA: **2,29 k tok/s, 9,2 GB**.
Saida: adapter `adapters\<run>\{best,final}` e modelo bf16 mesclado em `merged\<run>` (usado na avaliacao). Horas por braco: **~15,5 h otimista (2,29 k tok/s) a ~17,7 h (2,0 k)**; com evals 16,0-18,2 h; `budget.hours: 22`.
Corte: `--total-steps 3422` (1,5 epoca) nos DOIS bracos = -3,9 a -4,4 h por braco.

## 2. Antes de iniciar

Igual ao Nano (`Repos\KapiCoder-Mongo-Nano\README_job.md` §2): GPU livre, commitar `gates.yaml`/`rounds.yaml`/`student_corpus.yaml`/`sft_*_v1.yaml`, confirmar `rounds.yaml`,
`build_student_corpus.py --verify-only`, `train_student.py --config ...\sft_distill_v1.yaml --dry-run --tok-s 2290`, MongoDB no WSL (+ `wsl -d Ubuntu -e sleep infinity`) e Node para as avaliacoes.

## 3. Iniciar / monitorar / retomar

```powershell
.\scripts\gpu_queue.ps1 -Status ; .\scripts\gpu_queue.ps1 -Start -DryRun
.\scripts\gpu_queue.ps1 -Start -Only J17a-eval-base-T1     # zero-shot 1,5B (~40 min)
.\scripts\gpu_queue.ps1 -Start -Only J15-T1-distill        # ~16-18 h (de noite/fim de semana)
.\scripts\gpu_queue.ps1 -Start -Only J10-T1-gold           # controle, mesmos hiperparametros, ~16-18 h
.\scripts\gpu_queue.ps1 -Start                             # tudo que esta ready, na ordem do yaml
```
Manual: `& $PY -I scripts\student\train_student.py --config Repos\KapiCoder-Mongo-Small\configs\sft_distill_v1.yaml --resume auto` (idem `sft_gold_v1.yaml`).
Monitorar: `Get-Content Repos\KapiCoder-Mongo-Small\reports\sft_distill_v1\metrics.jsonl -Tail 3` (~13 s/passo, ~2.300 tok/s, pico ~9,2 GB), `...\eval_history.jsonl -Tail 1` (a cada 10% dos passos),
`Get-Content logs\gpu_queue\J15-T1-distill.log -Wait -Tail 20`. Pausa cooperativa: `.\scripts\gpu_queue.ps1 -Pause` (codigo 75) / `-Resume`; queda/kill: relanca com `--resume auto` (fingerprint identico; perda <= 30 min).
Como o T1 e longo, comece numa sexta a noite (Fase 05 §9.8) e deixe a janela diaria livre para uso (`-Pause`).

## 4. Avaliar

```powershell
& $PY -I scripts\student\eval_student.py run --tier t1 --model BaseRepos\Qwen2.5-Coder-1.5B --name base-qwen2.5-coder-1.5b                      # J17a
& $PY -I scripts\student\eval_student.py run --tier t1 --model Repos\KapiCoder-Mongo-Small\merged\sft_distill_v1 --name sft_distill_v1 --base-name base-qwen2.5-coder-1.5b   # J17
& $PY -I scripts\student\eval_student.py run --tier t1 --model Repos\KapiCoder-Mongo-Small\merged\sft_gold_v1 --name sft_gold_v1 --base-name base-qwen2.5-coder-1.5b      # J10e
& $PY -I scripts\student\eval_student.py compare --tier t1 --distill sft_distill_v1 --gold sft_gold_v1 --base base-qwen2.5-coder-1.5b
```
Mesma amostra do professor (2.000 ids do val v2c + 500 Generic-JS), `measures.json` e `gates.json` do tier T1 em `reports\eval\<name>\`. **Ficam `unevaluated`** (nunca aprovados): chat livre, cego v2, `legacy_v1_120`,
`exec_match_chat`, PT-BR x EN, HumanEval+, TTFT/RAM/decode do KapiLab (DML FP16/CPU INT8) e `T1.blind_v2_vs_base` (todo); `T1.apt_vs_t0` so com `--t0-predictions <predictions.jsonl do T0 aprovado>`.
**G2.4**: `compare` gera `reports\eval\compare\g2_4_teacher_measures.json` (professor - aluno gold, em p.p.); use o aluno gold de maior FIM acc entre T0 e T1 e rode `check_gates --tier TEACHER --gates configs\gates.teacher.yaml`.

## 5. Criterios de decisao distilado x controle

Os de `Repos\KapiCoder-Mongo-Nano\README_job.md` §7 / `configs\rounds.yaml` (exec-valid / teto +2 p.p. com IC95% > 0; FIM acc nao-inferior -2 p.p.; campos inexistentes em `full` <= gold + 0,5 p.p.; Generic-JS <= -3 p.p.
vs zero-shot). Adotar o distilado so se todos valerem; senao, rotulo gold com o professor so como filtro (e o J10 vira o T1 de referencia). Sem o J10 o G2.4 do professor fica sem referencia: nao cortar.

## 6. Decisoes pendentes do usuario

LoRA no T1 (full FT nao cabe); corte do T1 para 1,5 epoca (`--total-steps 3422`, nos dois bracos) se o total passar de ~40 h; ligar o J10b (controle T0) antes; confirmar `configs\rounds.yaml`.
