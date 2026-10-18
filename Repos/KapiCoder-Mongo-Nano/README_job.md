# Job de treino do KapiCoder-Mongo-Nano (T0, Qwen2.5-Coder-0.5B): distilado x controle

Estado: **preparado e enfileirado, NAO iniciado** (nenhum treino completo foi executado; so smoke de 20 passos + retomada, `reports\student\smoke_report.md` na raiz).
Base `BaseRepos\Qwen2.5-Coder-0.5B` (rev. `8123ea2e…`), workspace `Repos\KapiCoder-Mongo-Nano`. Escopo: **autocomplete FIM** `editor-context-v1` (chat fora ate a Q1;
KD por logits adiado). Spec: `docs\fase-05-treinamento.md` §6.2/§7/§9, gates `configs\gates.yaml` (T0) e `configs\gates.teacher.yaml` (G2.4), regra `configs\rounds.yaml` (draft).

Python (venv ROCm; nao instalar torch novo): `$PY = "C:\SlopStudio.MongoAI-artifacts\.venv\Scripts\python.exe"`, sempre com `-I`.

## 0. O que existe

| artefato | caminho |
|---|---|
| corpus (gitignored) | `data\student\{distill,gold}\t0\` (7 shards `.npz` + `.meta.jsonl`, `val\`, `manifest.json`); selecao em `data\student\selection\t0.jsonl` |
| relatorio do corpus | `reports\student\corpus_report.md` (composicao, balanco, isolamento G2.7) |
| construtor | `scripts\student\build_student_corpus.py` + `configs\student_corpus.yaml` (deterministico, ~90 s, CPU) |
| treino | `scripts\student\train_student.py` (estende `scripts\teacher\train_teacher.py`) + `configs\sft_distill_v1.yaml` / `sft_gold_v1.yaml` |
| avaliacao | `scripts\student\eval_student.py` (run / measures / compare) sobre `scripts\teacher\eval_teacher.py`, `scripts\eval\{compare_models,check_gates}.py` |
| fila | jobs `J16a`, `J14`, `J16`, `J10b` (opcional), `J10be` (opcional) em `configs\gpu_queue.yaml` |
| lock | `model.lock.json` (base + revisao, corpus, professor, hashes; atualizar com `scripts\student\write_model_lock.py` apos o treino) |

**Corpus T0**: 159.865 exemplos (151.180 Mongo + 8.685 replay), 44,2 M tokens/epoca (media 276 tok/ex.), 3.169 linhas de <= 16.384 tokens, 1.584 passos/epoca de ~32,8k tokens
-> **3.168 passos em 2 epocas (88,3 M tokens)**. Mongo: fim_ctx 68,5% (prompt exato que o professor viu), completion 26,4% (sufixo vazio; alvo = rotulo + sufixo original),
fim_raw 5,1% (gold nos dois bracos); replay JS/TS = 12,1% dos tokens. Os dois bracos tem os **mesmos ids, ordem e linhas**: so o rotulo de fim_ctx/completion muda
(professor x `middle` original; 54,3% dos exemplos Mongo tem rotulo diferente). T0 e subconjunto do T1.

## 1. Metodo e hiperparametros (iguais nos dois bracos)

**Full fine-tuning** com pesos mestre fp32 + autocast bf16, AdamW fp32 (fused, betas 0,9/0,95, wd 0,01), LR de pico **2e-5**, warmup 3%, cosseno ate 10%, grad clip 1,0,
32.768 tokens/passo (2 linhas de 16.384), checkpointing de ativacoes, CE em blocos so no alvo, 2 epocas (a 2a reembaralha as linhas, seed fixa). Medida no smoke:
**8,2 k tok/s, 11,5 GB** de pico (LoRA r=64: 6,3 k tok/s, 4,4 GB) -> full FT e mais rapido e cabe; justifica a escolha. Saida: modelo completo bf16 em `merged\<run>` (sem adapter).
Alternativa LoRA (decisao do usuario): `--set model.method=lora --set optim.lr=2.0e-4 --set optim.weight_decay=0` (lembre `2.0e-4`, com ponto decimal, no PyYAML).

Horas: **~3,0 h otimista (8,2 k tok/s) a ~3,9 h conservador (6,3 k)** por braco; com evals/checkpoints/export 3,3-4,2 h; `budget.hours: 9`.

## 2. Antes de iniciar (checklist)

1. GPU livre (nenhum llama-server/benchmark; `tmp\gpu.lock` ausente). Sem suspensao/reinicio por update. (Ao lancar qualquer llama-server/llama-bench, selecione a GPU pelo NOME "RX 7800 XT", nunca pelo indice.)
2. **Commitar** `configs\gates.yaml`, `configs\rounds.yaml`, `configs\student_corpus.yaml` e os `sft_*_v1.yaml` antes do primeiro passo (o hash de `gates.yaml`/`rounds.yaml` vai para o `run_manifest.json`) e **confirmar a regra `rounds.yaml` (status draft)**.
3. Corpus integro e isolado:
   ```powershell
   & $PY -I scripts\student\build_student_corpus.py --verify-only        # ids/ordem/linhas iguais entre bracos; ids em train.ids.txt e fora de val/test/bench
   & $PY -I scripts\student\train_student.py --config Repos\KapiCoder-Mongo-Nano\configs\sft_distill_v1.yaml --dry-run --tok-s 8200
   ```
   Para regenerar (deterministico): `& $PY -I scripts\student\build_student_corpus.py` (aborta em qualquer vazamento; reescreve tambem T1).
4. Para as avaliacoes: MongoDB dos fixtures no ar (`wsl -d Ubuntu -u root -e systemctl start mongod` e uma sessao `wsl -d Ubuntu -e sleep infinity` aberta) e Node no PATH.

## 3. Iniciar

Pela fila (recomendado; lock, watchdog 15 min, retomada automatica):

```powershell
.\scripts\gpu_queue.ps1 -Status
.\scripts\gpu_queue.ps1 -Start -DryRun
.\scripts\gpu_queue.ps1 -Start -Only J16a-eval-base-T0    # baseline zero-shot (~30 min), uma vez
.\scripts\gpu_queue.ps1 -Start -Only J14-T0-distill       # treino do braco distilado (~3-4 h)
.\scripts\gpu_queue.ps1 -Start -Only J16-T0-distill-eval  # avaliacao + medidas + gates T0
```
`-Start` sem `-Only` roda tudo que esta `ready` (inclui os jobs do T1). O controle `J10b-T0-gold` (e `J10be-T0-gold-eval`) esta `optional`: mude `status: optional` para `ready` em
`configs\gpu_queue.yaml` para liga-lo (depende do J16).

Manual (mesmo `tmp\gpu.lock`, sem watchdog):
```powershell
& $PY -I scripts\student\train_student.py --config Repos\KapiCoder-Mongo-Nano\configs\sft_distill_v1.yaml --resume auto
& $PY -I scripts\student\train_student.py --config Repos\KapiCoder-Mongo-Nano\configs\sft_gold_v1.yaml --resume auto
```
Saidas: checkpoints `checkpoints\<run>\step-XXXXXX` (2 ultimos, modelo fp32 + otimizador + RNG, a cada 30 min/5% dos passos), `reports\<run>\{metrics.jsonl,eval_history.jsonl,run_manifest.json,heartbeat.json}`,
modelo final `merged\<run>\` (bf16, `export_info.json`).

## 4. Monitorar

```powershell
Get-Content Repos\KapiCoder-Mongo-Nano\reports\sft_distill_v1\metrics.jsonl -Tail 3     # loss, lr, tok/s, VRAM por passo (~3,5 s/passo)
Get-Content Repos\KapiCoder-Mongo-Nano\reports\sft_distill_v1\eval_history.jsonl -Tail 1 # val_loss FIM, EM/normalized/syntax/mongo_valid, monitor Generic-JS (gjs_*)
Get-Content logs\gpu_queue\J14-T0-distill.log -Wait -Tail 20 ; Get-Content reports\gpu_queue\incidents.md
```
Saude: loss finita; `vram_peak_gb` ~11,5; `tok_s_real` ~8.000; `val_loss` caindo a cada 10% dos passos; `gjs_normalized_match` nao cai > 1,5 p.p. entre avaliacoes (sinal para pausar).
O treino aborta com loss/grad nao finita ou OOM (teto 14,1 GB; nunca usa memoria compartilhada).

## 5. Pausar e retomar

`.\scripts\gpu_queue.ps1 -Pause` -> salva checkpoint no passo seguinte e sai com codigo 75; `-Resume` relanca com `--resume auto`. Queda/kill: a fila relanca ate 3 vezes (perda <= 30 min).
`--resume auto` pega o ultimo `step-*` COMPLETE e **so aceita fingerprint identico** (base, manifesto do corpus, hiperparametros); mudou algo -> outro `--run-name`.
Corte de horas: `--total-steps N` (cosseno recalculado e registrado em `schedule_changes`) ou `budget.hours`.

## 6. Avaliar (gates T0 e comparacao pareada)

```powershell
$E = "Repos\KapiCoder-Mongo-Nano\reports\eval"
& $PY -I scripts\student\eval_student.py run --tier t0 --model BaseRepos\Qwen2.5-Coder-0.5B --name base-qwen2.5-coder-0.5b                 # J16a (zero-shot)
& $PY -I scripts\student\eval_student.py run --tier t0 --model Repos\KapiCoder-Mongo-Nano\merged\sft_distill_v1 --name sft_distill_v1 --base-name base-qwen2.5-coder-0.5b   # J16
& $PY -I scripts\student\eval_student.py run --tier t0 --model Repos\KapiCoder-Mongo-Nano\merged\sft_gold_v1 --name sft_gold_v1 --base-name base-qwen2.5-coder-0.5b      # J10be (se rodou o J10b)
& $PY -I scripts\student\eval_student.py compare --tier t0 --distill sft_distill_v1 --gold sft_gold_v1 --base base-qwen2.5-coder-0.5b                                 # CPU, ~3-5 min
```
`run` usa os **mesmos 2.000 ids** do professor (`base-v2c\sample_ids.json`, val v2c, contrato `editor-context-v1`) e os 500 cortes Generic-JS do baseline do 7B; gera
`<name>\{summary.json,predictions.jsonl,report.md,measures.json,gates.json}`: FIM acc (normalized match), exec-valid (estrito), exec-valid / teto, campos inexistentes por
`context_kind`, syntax, mongo_valid, APT, Generic-JS, conversacional, e `check_gates --tier T0`. **Itens sem fonte ficam `unevaluated`** (nunca aprovados): TTFT/RAM/disco/paridade
(KapiLab/export), `T0.apt_vs_base`, `T0.apt_gain`, `T0.apt_per_group` e `T0.exec_match_autocomplete` (exigem o `base_v1_0.5b` re-medido na suite v2: passe
`--baseline-v1-predictions <predictions.jsonl>` quando existir; o zero-shot do Qwen NAO e o `base_v1`, so gera metricas `info_*`). `exit 6` do gate = algum bloqueante falhou; `7` = nenhum falhou mas ha nao avaliados.

**G2.4 do professor**: `compare ... --gold <run>` grava `reports\eval\compare\g2_4_teacher_measures.json` (`fim_acc_vs_best_student_pp`, professor - aluno gold); use o aluno gold de MAIOR FIM acc entre T0 e T1 e rode
`& $PY -I scripts\eval\check_gates.py --tier TEACHER --gates configs\gates.teacher.yaml --measures <json>` (>= +5 p.p.; senao a destilacao vira so filtragem).

## 7. Criterios de decisao distilado x controle (resumo de `configs\rounds.yaml`, draft)

O braco **distilado** vence o controle (gold) se, nos mesmos exemplos (bootstrap pareado 10.000, McNemar, Holm): (1) exec-valid / teto sobe >= +2 p.p. com IC95% do delta > 0; (2) FIM acc nao-inferior (limite
inferior do IC95% > -2 p.p.); (3) campos inexistentes em `context_kind=full` <= gold + 0,5 p.p.; (4) Generic-JS nao cai > 3 p.p. vs zero-shot. Se nao: o professor fica so como filtro/validador
(rotulo gold). Nota: o distilado imita o professor (que concorda com o gold em ~44%), entao a FIM acc contra o gold tende a ser menor que a do controle - por isso o criterio principal e exec-valid.

## 8. Decisoes pendentes do usuario

1. Full FT (padrao, medido) x LoRA no T0 (+~4 h para A/B de qualidade); 2. ligar o J10b (controle T0, ~3,3 h) para ter o efeito do professor na escala T0 antes do T1; 3. confirmar `configs\rounds.yaml`;
4. cortes do calendario total (T1 1,5 epoca etc.): ver `reports\student\smoke_report.md`; 5. fornecer as predicoes do `base_v1_0.5b` na suite v2 se quiser avaliar `T0.apt_*`/`exec_match`.
