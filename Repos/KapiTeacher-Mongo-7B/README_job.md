# Job de adaptacao do KapiTeacher-Mongo-7B (SFT QLoRA, corpus v2c)

Estado: **preparado e enfileirado, NAO iniciado** (v2c + replay de 12%, sem CPT; decisoes do usuario em §7). Base `BaseRepos\Qwen2.5-Coder-7B` (rev. `0396a761…`), workspace
`Repos\KapiTeacher-Mongo-7B`. Spec: `docs\fase-02-professor.md` §3.3 (B), §3.6; `docs\fase-05-treinamento.md` §4, §6.2, §9.6.

Python (venv ROCm do bake-off; nao instalar torch novo): `$PY = "C:\SlopStudio.MongoAI-artifacts\.venv\Scripts\python.exe"`, sempre com `-I`.

## 0. O que existe

| artefato | caminho |
|---|---|
| construtor do corpus | `scripts\teacher\build_teacher_corpus.py` + `Repos\KapiTeacher-Mongo-7B\configs\corpus_v2c.yaml` |
| corpus (gitignored) | `data\teacher\v2c\` (13 shards `train-*.npz` + `.meta.jsonl`, `val\` incl. `val_gen_generic.jsonl`, `manifest.json`); relatorio `reports\corpus_manifest.md` |
| replay + G2.5 (gitignored) | `data\replay\v1\` (`replay_train.jsonl`, `generic_js_fim_eval.jsonl` = 500 cortes, `generic_js_fim_monitor.jsonl`, `manifest.json`, `files.jsonl`, `LICENSES.md`); construtor `scripts\data\build_replay.py` + `scripts\data\replay_cuts.py`; baseline `reports\baseline_base\Qwen2.5-Coder-7B-generic-js\` |
| treino | `scripts\teacher\train_teacher.py` + `configs\sft_v2c.yaml` |
| fila | `scripts\gpu_queue.ps1` + `configs\gpu_queue.yaml` (estado em `reports\gpu_queue\state.json`, incidentes em `reports\gpu_queue\incidents.md`) |
| avaliacao | `scripts\teacher\eval_teacher.py` (agora com `--adapter` e `--data-version v2c`), `scripts\teacher\teacher_measures.py`, `configs\gates.teacher.yaml`, `scripts\eval\check_gates.py` |
| merge | `scripts\teacher\merge_teacher.py` (CPU, bf16) |
| smoke | `reports\smoke_train.md` |

Numeros do corpus (`reports\corpus_manifest.md`): 129.568 exemplos, **37.118.069 tokens** (media 287/ex.; 18,5% de tokens com perda),
9.737 linhas de <= 4.096 tokens, **811 passos** de ~49k tokens (12 linhas). Mistura (pool nao-replay, por exemplos): FIM `editor-context-v1` 40%,
FIM cru 10%, completion (sufixo vazio, derivado) 20%, chat 15% + multi-turno 3% + explain 2%, correcao 6%, negativos 4%; **mais replay de codigo
generico = 12,0% dos tokens** (4,45 M tokens, 7.219 trechos de 4.429 arquivos / 555 pacotes npm MIT/Apache-2.0/BSD/ISC; 50/50 FIM PSM cru com
middle <= 96 tokens / trechos causais em sequencia). O Mongo caiu ~12% (123 k exemplos nao-replay) para manter o total em ~37 M tokens.
Fonte do replay: so `node_modules` ja instalados no disco (nada baixado); atribuicao em `data\replay\v1\LICENSES.md`.

## 1. Antes de iniciar (checklist)

1. GPU livre: nada mais usando a placa (benchmark/llama.cpp/DML); `tmp\gpu.lock` ausente ou obsoleto. Sem suspensao/reinicio por update.
2. **Commitar** `configs\gates.yaml`, `configs\gates.teacher.yaml`, `Repos\...\configs\sft_v2c.yaml` e `corpus_v2c.yaml` antes do primeiro passo
   (Fase 05 §4.3: o hash de `gates.yaml` vai para o `run_manifest.json`).
3. Corpus integro e isolado (G2.7):
   ```powershell
   & $PY -I scripts\teacher\build_teacher_corpus.py --verify-only        # relê os .meta.jsonl x data\splits\v2c\*.ids.txt
   & $PY -I scripts\teacher\train_teacher.py --dry-run --tok-s 500        # plano: passos, tokens, horas
   ```
   Para regenerar (deterministico, ~45 s): `& $PY -I scripts\teacher\build_teacher_corpus.py` (aborta se houver qualquer vazamento train x val/test/bench
   ou replay x generic_js_fim_eval). O replay (`data\replay\v1`) e reconstruido com `& $PY -I scripts\data\build_replay.py` (~1 min; redescobre
   os node_modules do disco; a saida depende do que esta instalado: **nao regenerar depois do baseline sem refazer o baseline generic-js**).
4. Para o job de avaliacao (J04b/J11): MongoDB dos fixtures no ar (`wsl -- true`, manter uma sessao WSL aberta) e Node no PATH.
5. Decisoes pendentes do usuario: ver §7.

## 2. Iniciar

Pela fila (recomendado; detem o lock, tem watchdog e retomada):

```powershell
.\scripts\gpu_queue.ps1 -Status
.\scripts\gpu_queue.ps1 -Start -DryRun                         # mostra os comandos, nao executa
.\scripts\gpu_queue.ps1 -Start -Only J09-teacher-sft-v2c       # so o treino (~22 h; ate 34 h)
.\scripts\gpu_queue.ps1 -Start                                 # J04b (baseline, ~35 min) -> J09 (treino) -> J11 (val v2c) -> J11b (G2.5 generic-js)
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
- **G2.5** (adapter - base >= -3 p.p. em FIM acc no `generic_js_fim_eval`, 500 cortes de pacotes npm holdout; syntax = node `vm.Script`; sem Mongo):
  baseline do base (NF4) ja medido em `reports\baseline_base\Qwen2.5-Coder-7B-generic-js\summary.json`. Apos o treino (job J11b):
  ```powershell
  & $PY -I scripts\teacher\eval_teacher.py --model BaseRepos\Qwen2.5-Coder-7B --adapter Repos\KapiTeacher-Mongo-7B\adapters\sft_v2c\final `
        --data-version generic-js --split eval --contract fim-raw --n 500 --load nf4 --name sft_v2c-final-generic-js `
        --ids-from Repos\KapiTeacher-Mongo-7B\reports\baseline_base\Qwen2.5-Coder-7B-generic-js\sample_ids.json --out-root $E
  & $PY -I scripts\teacher\teacher_measures.py --adapter-summary $E\sft_v2c-final\summary.json --base-summary $E\base-v2c\summary.json `
        --generic-base Repos\KapiTeacher-Mongo-7B\reports\baseline_base\Qwen2.5-Coder-7B-generic-js\summary.json `
        --generic-adapter $E\sft_v2c-final-generic-js\summary.json --out $E\sft_v2c-final\measures.json
  ```
  Durante o treino o eval periodico reporta `val_loss_generic` e `gjs_*` (100 cortes de monitor de OUTROS arquivos dos mesmos pacotes holdout;
  nunca seleciona o `best`). Queda de FIM acc generico > 1,5 p.p. no monitor ja e sinal para pausar e reavaliar (LR/replay).
- **G2.3** (chat exec-equivalencia, val/blind) e **G2.4** (vs. melhor student sem professor = J10) **ainda nao tem avaliador/conjunto**: ficam
  `unevaluated` (nunca aprovados). `--student-summary` do `teacher_measures.py` aceita o summary quando existir. O `blind` so e lido com
  `blind_tool.py` (<= 3 leituras na fase).
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

**Horas** (medido no smoke: 506 tok/s reais, 7,8 s por linha de 4k; corpus 37,10 M tokens com replay, 811 passos):

| cenario | treino | + evals/ckpts | total |
|---|---:|---:|---:|
| medido (506 tok/s) | 20,4 h | ~2,3 h (10 evals de ~12 min incl. 50 cortes generic-js + final ~21 min) | **~22,7 h** |
| conservador (360 tok/s, sonda J03b) | 28,6 h | ~2,3 h | **~30,9 h** |
| Fase 05 §9.2 (J09 7B, 47 M tok a 560 tok/s) | — | — | 25 h |

CPT (estagio A, ~11 h, J08) **pulado**: `stage_a_cpt: skip`. O recorte padrao (1 epoca do corpus inteiro, `budget.hours: 34`) cabe na janela de 24–36 h mesmo no cenario conservador (folga de ~3 h). Se a janela tiver de
ser menor, **ordem de corte** (do mais barato em qualidade ao mais caro):

1. Menos avaliacao: `--set eval.every_pct_steps=20 --set eval.gen_examples=50 --set eval.generic_gen_examples=25` (−1,2 h).
2. `--set data.max_train_tokens=30000000` (811 -> ~650 passos, −3,5 h; o corte e por prefixo aleatorio e mantem a proporcao de replay); depois `24000000` (−6 h). Fazer antes de iniciar; durante o run use
   `--total-steps N` (o cosseno e re-parametrizado a partir do passo atual; registrado em `schedule_changes` do manifest).
3. Reconstruir o corpus com menos `completion` (derivado) e `fim_raw` (`mix` do `corpus_v2c.yaml`; ~45 s) e retreinar do zero. Nao cortar o replay
   (so ha ~4,45 M tokens locais; reduzi-lo piora o G2.5).
4. Ultimo recurso: parar por `budget.hours` e usar `best`/o ultimo checkpoint (LR nao chegou ao piso; registrar).

Nunca cortar: verificacao G2.7, avaliacao final (Mongo e generic-js), checkpoint por tempo. Para **crescer** (usar mais da janela):
`--target-examples 170000` no construtor (~46 M tokens nao-replay; o replay local e fixo em ~4,45 M tokens: cai para ~9,7% do total).

## 7. Decisoes do usuario antes de iniciar

1. ~~CPT (estagio A)~~ **Decidido: pulado** (`stage_a_cpt: skip`); o infilling nativo e protegido por 10% de FIM cru + 12% de replay (50% FIM PSM).
   So reconsiderar se val_loss/G2.2 ficarem curtos ou se o G2.5 reprovar.
2. ~~Replay de codigo generico (G2.5)~~ **Decidido e montado**: 12,0% dos tokens, so fonte local licenciada (`data\replay\v1`, 555 pacotes
   MIT/Apache-2.0/BSD/ISC; nada baixado) + conjunto `generic_js_fim_eval` de 500 cortes com baseline medido. Risco residual: a fonte tem so
   ~4,5 M tokens (sem repeticao) e e dominada por bibliotecas Node/TS compiladas. Para crescer (exige autorizacao de download; npm, licencas MIT/ISC/BSD/Apache,
   ~0,3–1,5 M tokens cada, +8–10 M com ~20 pacotes): `react`, `vue`, `express`, `axios`, `d3`, `three`, `jquery`, `underscore`, `moment`, `chalk`,
   `commander`, `yargs`, `uuid`, `debug`, `async`, `eslint`, `prettier` (MIT), `typescript` (Apache-2.0; arquivos > 2.000 linhas sao descartados), `core-js`.
   Evitar pacotes do dominio MongoDB (`mongodb`, `mongoose`) para nao borrar o "generico".
3. ~~Contrato do chat~~ **Mantido `teacher-chat-v1`** (JSON com escape .NET, linha `Colecoes disponiveis`). Se o ContractProbe/IDE divergir, mudar
   `chat.json_escaping`/`include_known_collections` e regerar (45 s) **antes** de treinar.
4. Teto de **34 h** mantido (`budget.hours`). Falta decidir so se o J04b (baseline do base no v2c, ~35 min) roda antes do J09 (esta primeiro na fila;
   use `-Only` para pular).
5. Energia/sistema: nobreak, sem suspensao, monitor na iGPU (libera ~0,5–1 GB de VRAM) — decisoes suas (Fase 05 §9.6/§9.8).

## 8. Iteracao 2 (J09b, `sft_v2d`): preparada, NAO iniciada

Origem: `reports\error_analysis_sft_v2c.md`. Decisoes do usuario (2026-10-15, antes de retreinar): (1) o gate de campos inexistentes e medido em
`context_kind=full` (`G2.2.nonexistent_fields_full`, <= 2%, bloqueante); a taxa geral (`G2.2.nonexistent_fields`) e as de `none`/`irrelevant`
ficam informativas; (2) mais dados `none`/`irrelevant`, com alvo conservador.

**Referencia da it.1 com o gate novo** (`reports\eval\sft_v2c-final\{measures,gates}.json`; antigos em `*.pre_full_gate.json`): full 2,06% (34/1.647; **reprova por 0,06 p.p.**),
none 12,26% (n=155), irrelevant 14,14% (n=198), geral 4,05%. `exec_valid_vs_ceiling` 0,9265 segue reprovado.

### 8.1 Corpus v2d (`configs\corpus_v2d.yaml`, `data\teacher\v2d`, relatorio `reports\corpus_manifest_v2d.md`)
- So `train` do v2c; val/test/bench/blind nunca; G2.7 aprovado (0 vazamentos). `val_seed` igual ao da v2c (val_loss comparavel). Seed de treino novo (20261015).
- 48.381 exemplos, 13,49 M tokens, 295 passos. Mistura por componente e replay 12% iguais a v2c.
- none+irrelevant: 28,2% dos FIM com cabecalho (v2c ~17%): 5.518 cortes `full` re-renderizados com `gen_editor_context.py --force-context-kind` (mesma funcao `make_context(force_kind=)`).
- Conservador: cortes de none/irrelevant cujo `middle` tem campo/colecao/banco do schema invisivel no prefixo/sufixo sao descartados (pool: 78 mil none/irrelevant em fim_ctx e 117 mil em completion); `full` com campo invisivel nao e convertido.
- Pesos: ObjectBody/multiline, StatementStart, PropertyKey, FieldReferenceString x2; saas/iot/gaming/logistics x1,5.
- Reconstruir: `& $PY -I scripts\teacher\build_teacher_corpus.py --config Repos\KapiTeacher-Mongo-7B\configs\corpus_v2d.yaml --report Repos\KapiTeacher-Mongo-7B\reports\corpus_manifest_v2d.md` (~45 s); `--verify-only` confere G2.7.
- Auditoria de schemas (so relatorio): `reports\schema_audit_v2d.md` (`scripts\teacher\audit_schemas.py`).

### 8.2 Job J09b (`configs\sft_v2d.yaml`)
Aquecimento (`model.init_adapter` = `adapters\sft_v2c\final`; LoRA identico, otimizador/LR/passo do zero; o sha256 do adapter entra no fingerprint). LR de pico **5e-5**
(metade da it.1), cosseno ate 5e-6, warmup 3%: o adapter ja domina sintaxe/comprimento/G2.5; com ~1/3 dos tokens da it.1 e Adam novo, 1e-4 arriscaria esquecer
(G2.5) e 5e-5 ainda move o suficiente os pesos de campo. `budget.hours: 16`; mesmas avaliacoes periodicas; saidas `adapters\sft_v2d\{best,final}`.
Horas: 7,4 h (506 tok/s) / 10,4 h (360 tok/s) de treino + ~2,3 h de avaliacoes = **~9,7 h / ~12,7 h**.
Smoke (`--run-name sft_v2d_smoke`, 1 linha/passo): carregou o adapter (sum|lora_B| != 0), 12 passos + eval curto, retomou do passo 12 ate 16; checkpoints/adapter do smoke removidos.

Iniciar: `.\scripts\gpu_queue.ps1 -Start -Only J09b-teacher-sft-v2d` (depois J11c val v2c e J11d generic-js, `waiting`), ou o comando manual
`& $PY -I scripts\teacher\train_teacher.py --config Repos\KapiTeacher-Mongo-7B\configs\sft_v2d.yaml --resume auto`. Commitar antes os configs/gates.

### 8.3 Avaliar e decidir
```powershell
$E = "Repos\KapiTeacher-Mongo-7B\reports\eval"
& $PY -I scripts\teacher\teacher_measures.py --adapter-summary $E\sft_v2d-final\summary.json --base-summary $E\base-v2c\summary.json `
      --generic-base Repos\KapiTeacher-Mongo-7B\reports\baseline_base\Qwen2.5-Coder-7B-generic-js\summary.json --generic-adapter $E\sft_v2d-final-generic-js\summary.json `
      --corpus-manifest data\teacher\v2d\manifest.json --out $E\sft_v2d-final\measures.json
& $PY -I scripts\eval\check_gates.py --tier TEACHER --gates configs\gates.teacher.yaml --measures $E\sft_v2d-final\measures.json --out $E\sft_v2d-final\gates.json
& $PY -I scripts\eval\compare_models.py <args do script: v2c x v2d, mesmas ids, pareado>   # ver --help
```
Aceitar o v2d se: `G2.2.nonexistent_fields_full` <= 2%, `exec_valid_vs_ceiling` >= 0,95 (ou melhora clara sobre 0,9265), `G2.2.fim_acc_gain` >= +15 p.p. sem queda
significativa vs v2c no pareado, G2.5 >= -3 p.p. e G2.7 = 0. Informativos: taxas none/irrelevant (esperado cair) e fim_acc em none/irrelevant (pode cair: o alvo agora e conservador).
Se o full continuar > 2% ou o G2.5 cair, nao treinar a it.3 as cegas: reanalisar (`error_analysis`) e considerar mais dados `full` em PropertyKey/multiline.
Riscos: modelo mais conservador em none/irrelevant; `fim_raw` (10%) ainda contem cortes com campo invisivel; contextos `irrelevant` forcados nao reproduzem a conexao `local` dos snippets `pool` (~2,5% diferem do v2c).
