# Job atual — KapiCoder-Mongo-Nano / sft_v3

Configuracao vigente: `Repos/KapiCoder-Mongo-Nano/configs/sft_v3.yaml`; SHA256 `54fc8718fb07857ded4cf8937c6b785ef4599545cb001924d8cb67cc41b7ecdc`. Seed `20261018`; base e corpus sao os da configuracao, sem iniciar de pesos v1.

Comando de reproducao (somente apos autorizacao de GPU/treino):

```powershell
& 'C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe' '-I' 'scripts/student/train_student.py' '--config' 'Repos/KapiCoder-Mongo-Nano/configs/sft_v3.yaml' '--resume' 'auto'
```

Comando extraido do job `J20-T0-v3` em `configs/gpu_queue.yaml` (inclui cortes CLI vigentes, quando aplicaveis). Usar a fila `scripts/gpu_queue.ps1` e confirmar a RX 7800 XT por contadores; nunca selecionar por indice. O comando acima identifica o job, nao autoriza executa-lo. Relatorios: `Repos/KapiCoder-Mongo-Nano/reports/sft_v3/run_manifest.json`, `reports/student/v3_decision_t0.json` e `reports/datasets/homolog-v3-eval-t0.json`.

Limitacoes de independencia, criterios e pendencias: `reports/student/correcoes-coordenadas-20261026.md`.

---

## Historico v1 (nao usar para reproduzir o v3)

# Job de treino do KapiCoder-Mongo-Nano (T0, Qwen2.5-Coder-0.5B): distilado x controle

Estado: **preparado e enfileirado, NAO iniciado** (nenhum treino completo foi executado; so smoke de 20 passos + retomada, `reports\student\smoke_report.md` na raiz).
Base `BaseRepos\Qwen2.5-Coder-0.5B` (rev. `8123ea2e…`), workspace `Repos\KapiCoder-Mongo-Nano`. Escopo: **autocomplete FIM** `editor-context-v1` (chat fora ate a Q1;
KD por logits adiado). Spec: `docs\fase-05-treinamento.md` Â§6.2/Â§7/Â§9, gates `configs\gates.yaml` (T0) e `configs\gates.teacher.yaml` (G2.4), regra `configs\rounds.yaml` (draft).

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

## v3. Retreino com o corpus v3 (J20-T0-v3 + J22-T0-v3-eval): PREPARADO, NAO INICIADO

**Por que.** Na homologacao v1 (`reports/datasets/homolog-v1-eval.md`, schemas `hom` nunca vistos) o Nano inventa campos que nao existem no schema:
contexto `full` 15,35% (bases ~7%). O retreino testa se mais schemas novos + mais "copiar o nome do texto visivel" reduzem isso **sem perder FIM acc / exec ÷ teto**.
Inicio do ZERO a partir do Qwen base (run `sft_v3`; nunca do adapter/modelo anterior); hiperparametros IDENTICOS ao `sft_distill_v1.yaml`
(so mudam `run.name`/`run.arm`/`data.corpus_dir`/`budget.hours`; teste `scripts/student/tests/test_queue_v3.py`).

**Corpus** (`scripts/student/build_student_corpus_v3.py` + `configs/student_corpus_v3.yaml`; detalhes e funis em `reports/student/corpus_v3_report.md`; gramatica em `reports/datasets/v3-grammar-report.md`):
`data/student/v3/t0/` - 238,453 exemplos (225,835 cortes + 12,618 replay), **64,999,846 tokens/epoca**, 4,704 linhas, 2,352 passos/epoca.
Por fonte (proveniencia por exemplo em `source`/`label_source` nos `.meta.jsonl`): destilado atual `accepted-v2c-teacher` 130,984 (rotulo do professor; `fim_raw` gold),
cortes do LLM Qwen3-14B `llm-qwen3-14b` 58,808 (gold), gramatica dos 61 schemas v3 `train` `grammar-v3` 36,043 (gold), replay JS/TS 12,618 (12.0% dos tokens).
LLM + gramatica = 42% dos cortes; **none+irrelevant = 39.1%** (v2c: ~29%), sempre conservador (o rotulo so usa nomes visiveis); por fonte (%) {'accepted': {'full': 66.9, 'irrelevant': 17.8, 'none': 15.3}, 'grammar': {'full': 50.5, 'irrelevant': 25.1, 'none': 24.4}, 'llm': {'full': 53.9, 'irrelevant': 23.6, 'none': 22.4}}.
Fidelidade ao schema: cortes `full` cujo rotulo usa campo/colecao ausente do prompt inteiro entram com p=0,25 (~1% restante). **Sem destilacao nova**: rotulo gold nos dados novos (exato; o efeito da
destilacao foi pequeno), professor so no destilado atual. Balanco: cut_kind/categoria no relatorio do corpus. Decontaminado (ngram_min_content 2) contra val/test/bench v2c, v3-eval **e a homologacao selada**
(achado: o J18 original nao foi decontaminado n-grama contra a homologacao por caminho errado em `llm_to_fim.homolog_docs`; ~24% do LLM e ~4% do destilado saem aqui).

**Treino.** 2 epocas = **4,704 passos**, 129,999,692 tokens reais -> **4.46 h** a 8100 tok/s (5.73 h a 6300 tok/s conservador); com a velocidade do smoke v3
(8484 tok/s) **4.3 h**. `budget.hours: 8`. Meta da tarefa: 4-6 h.

**Smoke ja feito** (20 passos + retomada do passo 12, `--no-export`; `reports/student/smoke_v3_metrics.json`): 8484 tok/s reais (mediana 9013), 3.1 s/passo,
pico de VRAM alocada 11.58 GB (reservada 12.97 GB), 201 s de GPU no total (<= 8 min). GPU confirmada pelos contadores do Windows (`scripts/export/gpu_probe.ps1`): o pid do treino
ocupou 13,4 GB dedicados e ~99,8% do engine no adaptador da RX 7800 XT (`luid ...12602`); a iGPU ficou em 141 MB e 0%. Pastas `smoke_v3_*` apagadas.

### Antes de iniciar
1. GPU livre (`tmp\gpu.lock` ausente; nenhum llama-server/benchmark). MongoDB do WSL no ar e `wsl -d Ubuntu -e sleep infinity` aberto; Node no PATH (avaliacoes).
2. Corpus integro: `& $PY -I scripts\data\llm\run_guarded.py scripts\student\build_student_corpus_v3.py --verify-only --tiers t0` e
   `& $PY -I scripts\student\train_student.py --config Repos\KapiCoder-Mongo-Nano\configs\sft_v3.yaml --dry-run --tok-s 8100` (para regenerar: o mesmo script sem `--verify-only`, ~5 min, CPU, deterministico; SEMPRE via `run_guarded.py`).
3. **Commitar** `configs\student_corpus_v3.yaml` e `Repos\KapiCoder-Mongo-Nano\configs\sft_v3.yaml` antes do primeiro passo (o hash entra no `run_manifest.json`).

### Iniciar
```powershell
.\scripts\gpu_queue.ps1 -Status
.\scripts\gpu_queue.ps1 -Start -Only J20-T0-v3
.\scripts\gpu_queue.ps1 -Start -Only J22-T0-v3-eval   # avaliacao + decisao (~1,5-2,5 h)
```
O J22 grava `reports\student\v3_gate_t0.go` SO se o veredito for GO; sem esse arquivo o J21 (T1) fica `blocked` e a fila nao gasta GPU no T1.
Manual: `& $PY -I scripts\student\train_student.py --config Repos\KapiCoder-Mongo-Nano\configs\sft_v3.yaml --resume auto`. Monitorar/pausar/retomar: como nas secoes anteriores (mesmos arquivos, run `sft_v3`).

### Avaliar (`scripts\student\eval_v3.py --tier t0`)
(a) `eval_student.py run --name sft_v3` no val v2c com a MESMA amostra de 2.000 ids do professor + Generic-JS (500) + medidas/gates; (b) `eval_homolog.py run --name t0_v3` na homologacao v1 (6.030 cortes; execucao do texto real;
leitura de `data/homolog` so pela API autorizada, com log) e `report --out-name homolog-v3-eval-t0` (base x atual x v3, pareado, IC95 bootstrap, McNemar-Holm); (c) val pareado atual x v3 e base x v3 (`reports\student\v3_val_compare_t0.json`);
(d) `decide_v3.py` -> `reports\student\v3_decision_t0.json/.md`. "Atual" = `sft_distill_v1`.

### Criterios de decisao (aplicados por `scripts\student\decide_v3.py`)
- **C1 (meta)**: campos inexistentes em `full` na homologacao **< 15,35%** (modelo atual); ideal **<= base + 1 p.p. (~7-8%)**.
- **C2 (melhora confirmada)**: delta pareado de `no_nonexistent_field` no grupo `full` com **limite inferior do IC95 > 0**.
- **C3 (guarda na homologacao)**: perda de FIM acc (normalized_match) e de exec ÷ teto **<= 1,0 p.p.** contra o atual.
- **C4 (guarda no val/Generic-JS)**: perda no val v2c (2.000 ids) <= 2,0 p.p. em FIM acc e exec ÷ teto; Generic-JS nao cai > 3 p.p. vs zero-shot (G2.5).
- Veredito: **GO** (C1..C4) | INCONCLUSIVO (C1+C3+C4, sem C2) | SEM_MELHORA (C1 falha) | REJEITADO (C3 ou C4 violados).

### Se nao houver melhora (regra)
Qualquer veredito diferente de GO = **nao gastar mais GPU neste retreino**: nao rodar o passo seguinte (T1), nao fazer rodadas extras de hiperparametro/dados sem nova decisao do usuario, **manter o `sft_distill_v1` como modelo publicado**,
registrar o resultado (`reports\student\v3_decision_t0.md`, tabela em `reports\datasets\homolog-v3-eval-t0.md`) e no `docs\resumo-executivo.md` como resultado negativo/inconclusivo. GO: promover `merged\sft_v3`
(atualizar `model.lock.json` com `scripts\student\write_model_lock.py`) e liberar o J21.

### Riscos
- **Hipotese nao garantida**: a causa pode ser capacidade/generalizacao (o relatorio v1 mostra que `full` nao reduz o erro: 15,3% em full x 14,7% em none), nao so dados; o teste e barato no T0 (~4,5 h), por isso o T0 vai antes.
- Menos destilado efetivo (~58% dos cortes) e 12% de replay: a FIM acc no val v2c pode cair ligeiramente (C4 mede).
- Escopo do filtro de fidelidade: remove ~4-7% dos cortes cujo rotulo cita nomes que o modelo "deveria adivinhar"; pode baixar FIM acc em prompts reais com contexto incompleto.
- Homologacao e do mesmo LLM/gramatica (nao humana) e cortes `grammar` compartilham templates com o treino; uma unica seed de treino (sem variancia entre treinos).
- A homologacao v1 dos modelos ATUAIS foi medida com treino sem decontaminacao n-grama contra a homologacao (achado acima): a comparacao atual x v3 pode subestimar a melhora real ou superestimar o atual.
- Disco: `merged\sft_v3` + checkpoints (T0 full FT ~2 GB fp32 x2; T1 LoRA pequeno). Queda de energia: retoma do ultimo checkpoint (perda <= 30 min).
