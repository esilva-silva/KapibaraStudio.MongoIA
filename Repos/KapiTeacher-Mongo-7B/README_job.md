# Job de adaptacao do KapiTeacher-Mongo-7B (SFT QLoRA, corpus v2c)

Estado: **preparado e enfileirado, NAO iniciado**. Base `BaseRepos\Qwen2.5-Coder-7B` (rev. `0396a761…`), workspace
`Repos\KapiTeacher-Mongo-7B`. Spec: `docs\fase-02-professor.md` §3.3 (B), §3.6; `docs\fase-05-treinamento.md` §4, §6.2, §9.6.

Python (venv ROCm do bake-off; nao instalar torch novo): `$PY = "C:\SlopStudio.MongoAI-artifacts\.venv\Scripts\python.exe"`, sempre com `-I`.

## 0. O que existe

| artefato | caminho |
|---|---|
| construtor do corpus | `scripts\teacher\build_teacher_corpus.py` + `Repos\KapiTeacher-Mongo-7B\configs\corpus_v2c.yaml` |
| corpus (gitignored) | `data\teacher\v2c\` (14 shards `train-*.npz` + `.meta.jsonl`, `val\`, `manifest.json`); relatorio `reports\corpus_manifest.md` |
| treino | `scripts\teacher\train_teacher.py` + `configs\sft_v2c.yaml` |
| fila | `scripts\gpu_queue.ps1` + `configs\gpu_queue.yaml` (estado em `reports\gpu_queue\state.json`, incidentes em `reports\gpu_queue\incidents.md`) |
| avaliacao | `scripts\teacher\eval_teacher.py` (agora com `--adapter` e `--data-version v2c`), `scripts\teacher\teacher_measures.py`, `configs\gates.teacher.yaml`, `scripts\eval\check_gates.py` |
| merge | `scripts\teacher\merge_teacher.py` (CPU, bf16) |
| smoke | `reports\smoke_train.md` |

Numeros do corpus (`reports\corpus_manifest.md`): 139.258 exemplos, **37.146.949 tokens** (media 267/ex.; 14,1% de tokens com perda),
9.681 linhas de <= 4.096 tokens, **806 passos** de ~49k tokens (12 linhas). Mistura: FIM `editor-context-v1` 40%, FIM cru 10%,
completion (sufixo vazio, derivado) 20%, chat 15% + multi-turno 3% + explain 2%, correcao 6%, negativos 4%, **replay 0%** (sem fonte local licenciada).

## 1. Antes de iniciar (checklist)

1. GPU livre: nada mais usando a placa (benchmark/llama.cpp/DML); `tmp\gpu.lock` ausente ou obsoleto. Sem suspensao/reinicio por update.
2. **Commitar** `configs\gates.yaml`, `configs\gates.teacher.yaml`, `Repos\...\configs\sft_v2c.yaml` e `corpus_v2c.yaml` antes do primeiro passo
   (Fase 05 §4.3: o hash de `gates.yaml` vai para o `run_manifest.json`).
3. Corpus integro e isolado (G2.7):
   ```powershell
   & $PY -I scripts\teacher\build_teacher_corpus.py --verify-only        # relê os .meta.jsonl x data\splits\v2c\*.ids.txt
   & $PY -I scripts\teacher\train_teacher.py --dry-run --tok-s 500        # plano: passos, tokens, horas
   ```
   Para regenerar (deterministico, ~45 s): `& $PY -I scripts\teacher\build_teacher_corpus.py` (aborta se houver qualquer vazamento train x val/test/bench).
4. Para o job de avaliacao (J04b/J11): MongoDB dos fixtures no ar (`wsl -- true`, manter uma sessao WSL aberta) e Node no PATH.
5. Decisoes pendentes do usuario: ver §7.

## 2. Iniciar

Pela fila (recomendado; detem o lock, tem watchdog e retomada):

```powershell
.\scripts\gpu_queue.ps1 -Status
.\scripts\gpu_queue.ps1 -Start -DryRun                         # mostra os comandos, nao executa
.\scripts\gpu_queue.ps1 -Start -Only J09-teacher-sft-v2c       # so o treino (~22 h)
.\scripts\gpu_queue.ps1 -Start                                 # J04b (baseline, ~35 min) -> J09 (treino) -> J11 (avaliacao)
```

Manual (mesmo lock `tmp\gpu.lock`; sem watchdog):

```powershell
& $PY -I scripts\teacher\train_teacher.py --config Repos\KapiTeacher-Mongo-7B\configs\sft_v2c.yaml --resume auto
```

Recomendado deixar a janela livre ao iniciar em uma sexta a noite (Fase 05 §9.8). Saidas: checkpoints em
`Repos\KapiTeacher-Mongo-7B\checkpoints\sft_v2c\step-XXXXXX` (3 ultimos), adapter `adapters\sft_v2c\{best,final}`,
metricas em `reports\sft_v2c\{metrics.jsonl,eval_history.jsonl,run_manifest.json,heartbeat.json}`.

## 3. Monitorar

```powershell
Get-Content Repos\KapiTeacher-Mongo-7B\reports\sft_v2c\metrics.jsonl -Tail 3          # loss, lr, tok/s, VRAM por passo (~2,5 min/passo)
Get-Content Repos\KapiTeacher-Mongo-7B\reports\sft_v2c\eval_history.jsonl -Tail 1     # val_loss (fim/chat), EM, syntax, mongo_valid
Get-Content logs\gpu_queue\J09-teacher-sft-v2c.log -Wait -Tail 20
.\scripts\gpu_queue.ps1 -Status
Get-Content reports\gpu_queue\incidents.md                                            # watchdog, falhas, abandono
```

Sinais de saude: loss finita; `vram_peak_gb` <= ~9–10 GB; `tok_s_real` ~ 500; val_loss (FIM) caindo a cada 10% dos passos. O treino
aborta sozinho com loss/grad nao finita ou OOM (o alocador e limitado a 14,1 GB: nunca pagina para memoria compartilhada).

## 4. Pausar, retomar, recuperar

- **Pausa cooperativa** (liberar a GPU em ~3 min): `.\scripts\gpu_queue.ps1 -Pause` -> o treino salva checkpoint no passo seguinte e sai
  com codigo 75 (`status: paused`); `.\scripts\gpu_queue.ps1 -Resume` remove `tmp\gpu.pause` e a fila relanca com `--resume auto`.
- **Queda/kill/travamento**: a fila mata e relanca com `--resume auto` ate 3 vezes (watchdog: 15 min sem avanco de `metrics.jsonl` ou
  `heartbeat.json`; antes de relancar espera `scripts\teacher\check_gpu.py` passar); perda maxima ~30 min (checkpoint por tempo) ou 5% dos passos.
  Na 3a falha o job vira `failed`, vai para `incidents.md` e a fila segue. Manual: repetir o comando `--resume auto`.
- `--resume auto` pega o ultimo `step-*` com marcador `COMPLETE` e **so aceita fingerprint identico** (revisao/hash do base, manifest do
  corpus, LoRA/otimizador/seed/acumulacao). Mudou algo de treino -> use outro `--run-name` (nao `--resume none` na mesma pasta).
  Smoke desta preparacao: kill no passo 19, retomada do 16, loss dos passos 17–19 reproduzida a 5 casas.
- Parada por horas: `budget.hours` (34 h de treino acumulado, entre retomadas) -> checkpoint + adapter `final` + exit 0, `status: stopped_budget`.

## 5. Avaliar (G2.2–G2.5, G2.7)

Os gates do professor (Fase 02 §3.6) estao em `configs\gates.teacher.yaml` (tier `TEACHER`, mesmo formato do `gates.yaml`). O J11 da fila faz a
geracao; os passos manuais:

```powershell
$E = "Repos\KapiTeacher-Mongo-7B\reports\eval"
# baseline do base no MESMO val v2c (J04b) e adapter final na MESMA amostra (J11)
& $PY -I scripts\teacher\eval_teacher.py --model BaseRepos\Qwen2.5-Coder-7B --data-version v2c --split val --contract editor-context-v1 --n 2000 --name base-v2c --out-root $E
& $PY -I scripts\teacher\eval_teacher.py --model BaseRepos\Qwen2.5-Coder-7B --adapter Repos\KapiTeacher-Mongo-7B\adapters\sft_v2c\final `
      --data-version v2c --split val --contract editor-context-v1 --n 2000 --name sft_v2c-final --ids-from $E\base-v2c\sample_ids.json --out-root $E
& $PY -I scripts\teacher\teacher_measures.py --adapter-summary $E\sft_v2c-final\summary.json --base-summary $E\base-v2c\summary.json --out $E\sft_v2c-final\measures.json
& $PY -I scripts\eval\check_gates.py --tier TEACHER --gates configs\gates.teacher.yaml --measures $E\sft_v2c-final\measures.json --out $E\sft_v2c-final\gates.json
```

- **G2.2** (FIM-acc +15 p.p. sobre o base, exec-valid >= 0,95, campos inexistentes <= 2%) e **G2.7** saem do comando acima. `exit 7` do
  `check_gates` = nenhum reprovado mas ha gate nao avaliado.
- **G2.3** (chat exec-equivalencia, val/blind), **G2.4** (vs. melhor student sem professor = J10) e **G2.5** (queda no Generic-JS FIM) **ainda nao
  tem avaliador/conjunto**: ficam `unevaluated` (nunca aprovados). `--student-summary`, `--generic-base/--generic-adapter` do
  `teacher_measures.py` aceitam os summaries quando existirem. O `blind` so e lido com `blind_tool.py` (<= 3 leituras na fase).
- Selecao de checkpoint: `adapters\sft_v2c\best` = menor val_loss; `final` = fim do plano. Comparar os dois no val antes do blind.
- O val v2c e de schemas `eval` (nunca vistos no treino); nao usar `test`/`bench` para ajuste.

## 6. Fundir o adapter (merge bf16) e orcamento/cortes

```powershell
& $PY -I scripts\teacher\merge_teacher.py --adapter Repos\KapiTeacher-Mongo-7B\adapters\sft_v2c\final --out Repos\KapiTeacher-Mongo-7B\merged\KapiTeacher-Mongo-7B-v1
```
CPU, ~15 GB de RAM, nao usa a GPU (pode rodar com a fila ocupada). Grava safetensors bf16, tokenizer, `LICENSE` (Apache-2.0),
`MODIFICATION_NOTICE.md` e `merge_manifest.json` (hashes). Validar a paridade antes da destilacao: avaliar o merged com `eval_teacher.py --model <merged> --load nf4`
(o 7B em bf16 = 15,2 GB nao cabe no teto de 14,1 GB) e comparar com NF4+adapter, ou via GGUF Q8_0; a diferenca vem de fundir o
adapter em pesos bf16 que nao sao os NF4 do treino.

**Horas** (medido no smoke: 506 tok/s reais, 7,8 s por linha de 4k; corpus 37,15 M tokens):

| cenario | treino | + evals/ckpts | total |
|---|---:|---:|---:|
| medido (506 tok/s) | 20,4 h | ~1,6 h (10 evals de ~8 min + final ~13 min) | **~22 h** |
| conservador (360 tok/s, sonda J03b) | 28,7 h | ~1,6 h | **~30 h** |
| Fase 05 §9.2 (J09 7B, 47 M tok a 560 tok/s) | — | — | 25 h |

O recorte padrao (1 epoca do corpus inteiro, `budget.hours: 34`) cabe na janela de 24–36 h mesmo no cenario conservador. Se a janela tiver de
ser menor, **ordem de corte** (do mais barato em qualidade ao mais caro):

1. Menos avaliacao: `--set eval.every_pct_steps=20 --set eval.gen_examples=50` (−0,7 h).
2. `--set data.max_train_tokens=30000000` (806 -> 651 passos, −3,5 h); depois `24000000` (−6 h). Fazer antes de iniciar; durante o run use
   `--total-steps N` (o cosseno e re-parametrizado a partir do passo atual; registrado em `schedule_changes` do manifest).
3. Reconstruir o corpus com menos `completion` (derivado) e `fim_raw` (`mix` do `corpus_v2c.yaml`; ~45 s) e retreinar do zero.
4. Ultimo recurso: parar por `budget.hours` e usar `best`/o ultimo checkpoint (LR nao chegou ao piso; registrar).

Nunca cortar: verificacao G2.7, avaliacao final, checkpoint por tempo. Para **crescer** (usar mais da janela): `--target-examples 190000` no
construtor (~50 M tokens, ~28 h medido / ~39 h conservador — acima de 36 h no pior caso).

## 7. Decisoes do usuario antes de iniciar

1. **CPT (estagio A) fica de fora?** Este job e so o estagio B (SFT). A mistura ja tem 10% de FIM cru (+ ~3% sem cabecalho) mantendo o infilling nativo; o CPT (7B: ~11 h, J08) so
   se justifica se o val_loss/G2.2 ficar curto. Recomendado: pular e medir.
2. **Replay de codigo generico (G2.5)**: sem fonte local licenciada, `replay: none`. Sem replay nem conjunto Generic-JS FIM o G2.5 nao e
   mensuravel; fornecer ~4% (~5,5 mil ex.) de JS/TS/JSON MIT/Apache e o conjunto de 500 FIM, ou aceitar o risco (LoRA r=64, 1 epoca, base congelada).
3. **Contrato do chat** (`teacher-chat-v1`): JSON com escape .NET (`\u00E3`, `\u0022`) e linha extra `Colecoes disponiveis`. Se o ContractProbe/IDE
   divergir, mudar `chat.json_escaping`/`include_known_collections` e regerar (45 s) **antes** de treinar.
4. Corte de tokens/horas (§6) e se o J04b (baseline do base no v2c, ~35 min) roda antes do J09 (esta primeiro na fila; use `-Only` para pular).
5. Energia/sistema: nobreak, sem suspensao, monitor na iGPU (libera ~0,5–1 GB de VRAM) — decisoes suas (Fase 05 §9.6/§9.8).
