# Job atual — KapiCoder-Mongo-Small / sft_v3

Configuracao vigente: `Repos/KapiCoder-Mongo-Small/configs/sft_v3.yaml`; SHA256 `20a67dc5334c753dc161cfbade0b8fb6eddf7bc57ff252855981e4731c39847c`. Seed `20261018`; base e corpus sao os da configuracao, sem iniciar de pesos v1.

Comando de reproducao (somente apos autorizacao de GPU/treino):

```powershell
& 'C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe' '-I' 'scripts/student/train_student.py' '--config' 'Repos/KapiCoder-Mongo-Small/configs/sft_v3.yaml' '--total-steps' '4617' '--resume' 'auto'
```

Comando extraido do job `J21-T1-v3` em `configs/gpu_queue.yaml` (inclui cortes CLI vigentes, quando aplicaveis). Usar a fila `scripts/gpu_queue.ps1` e confirmar a RX 7800 XT por contadores; nunca selecionar por indice. O comando acima identifica o job, nao autoriza executa-lo. Relatorios: `Repos/KapiCoder-Mongo-Small/reports/sft_v3/run_manifest.json`, `reports/student/v3_decision_t1.json` e `reports/datasets/homolog-v3-eval-t1.json`.

Limitacoes de independencia, criterios e pendencias: `reports/student/correcoes-coordenadas-20261026.md`.

---

## Historico v1 (nao usar para reproduzir o v3)

# Job de treino do KapiCoder-Mongo-Small (T1, Qwen2.5-Coder-1.5B): distilado x controle (J15 x J10)

Estado: **preparado e enfileirado, NAO iniciado** (so smoke de 20 passos + retomada; `reports\student\smoke_report.md` na raiz). Base `BaseRepos\Qwen2.5-Coder-1.5B`
(rev. `df3ce67c…`), workspace `Repos\KapiCoder-Mongo-Small`. Escopo: **autocomplete FIM** `editor-context-v1` (chat curto fora ate a Q1; KD por logits adiado).
Spec: `docs\fase-05-treinamento.md` Â§6.2/Â§7/Â§9, gates `configs\gates.yaml` (T1) e `configs\gates.teacher.yaml` (G2.4), regra `configs\rounds.yaml` (draft).
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

Igual ao Nano (`Repos\KapiCoder-Mongo-Nano\README_job.md` Â§2): GPU livre, commitar `gates.yaml`/`rounds.yaml`/`student_corpus.yaml`/`sft_*_v1.yaml`, confirmar `rounds.yaml`,
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
Como o T1 e longo, comece numa sexta a noite (Fase 05 Â§9.8) e deixe a janela diaria livre para uso (`-Pause`).

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

Os de `Repos\KapiCoder-Mongo-Nano\README_job.md` Â§7 / `configs\rounds.yaml` (exec-valid / teto +2 p.p. com IC95% > 0; FIM acc nao-inferior -2 p.p.; campos inexistentes em `full` <= gold + 0,5 p.p.; Generic-JS <= -3 p.p.
vs zero-shot). Adotar o distilado so se todos valerem; senao, rotulo gold com o professor so como filtro (e o J10 vira o T1 de referencia). Sem o J10 o G2.4 do professor fica sem referencia: nao cortar.

## 6. Decisoes pendentes do usuario

LoRA no T1 (full FT nao cabe); corte do T1 para 1,5 epoca (`--total-steps 3422`, nos dois bracos) se o total passar de ~40 h; ligar o J10b (controle T0) antes; confirmar `configs\rounds.yaml`.

## v3. Retreino com o corpus v3 (J21-T1-v3 + J23-T1-v3-eval): PREPARADO, NAO INICIADO

**Por que.** Na homologacao v1 (`reports/datasets/homolog-v1-eval.md`, schemas `hom` nunca vistos) o Small inventa campos que nao existem no schema:
contexto `full` 9,19% (bases ~7%). O retreino testa se mais schemas novos + mais "copiar o nome do texto visivel" reduzem isso **sem perder FIM acc / exec ÷ teto**.
Inicio do ZERO a partir do Qwen base (run `sft_v3`; nunca do adapter/modelo anterior); hiperparametros IDENTICOS ao `sft_distill_v1.yaml`
(so mudam `run.name`/`run.arm`/`data.corpus_dir`/`budget.hours`; teste `scripts/student/tests/test_queue_v3.py`).

**Corpus** (`scripts/student/build_student_corpus_v3.py` + `configs/student_corpus_v3.yaml`; detalhes e funis em `reports/student/corpus_v3_report.md`; gramatica em `reports/datasets/v3-grammar-report.md`):
`data/student/v3/t1/` - 309,571 exemplos (292,812 cortes + 16,759 replay), **86,000,018 tokens/epoca**, 6,156 linhas, 3,078 passos/epoca.
Por fonte (proveniencia por exemplo em `source`/`label_source` nos `.meta.jsonl`): destilado atual `accepted-v2c-teacher` 169,831 (rotulo do professor; `fim_raw` gold),
cortes do LLM Qwen3-14B `llm-qwen3-14b` 76,248 (gold), gramatica dos 61 schemas v3 `train` `grammar-v3` 46,733 (gold), replay JS/TS 16,759 (12.0% dos tokens).
LLM + gramatica = 42% dos cortes; **none+irrelevant = 34.8%** (v2c: ~29%), sempre conservador (o rotulo so usa nomes visiveis); por fonte (%) {'accepted': {'full': 72.0, 'irrelevant': 14.9, 'none': 13.2}, 'grammar': {'full': 52.6, 'irrelevant': 23.9, 'none': 23.4}, 'llm': {'full': 57.7, 'irrelevant': 21.7, 'none': 20.6}}.
Fidelidade ao schema: cortes `full` cujo rotulo usa campo/colecao ausente do prompt inteiro entram com p=0,25 (~1% restante). **Sem destilacao nova**: rotulo gold nos dados novos (exato; o efeito da
destilacao foi pequeno), professor so no destilado atual. Balanco: cut_kind/categoria no relatorio do corpus. Decontaminado (ngram_min_content 2) contra val/test/bench v2c, v3-eval **e a homologacao selada**
(achado: o J18 original nao foi decontaminado n-grama contra a homologacao por caminho errado em `llm_to_fim.homolog_docs`; ~24% do LLM e ~4% do destilado saem aqui).

**Treino.** 1,5 epoca (`--total-steps`, corte do job; o yaml diz 2 epocas) = **4,617 passos**, 129,000,027 tokens reais -> **15.65 h** a 2290 tok/s (17.92 h a 2000 tok/s conservador); com a velocidade do smoke v3
(2388 tok/s) **15.0 h**. `budget.hours: 20`. Meta da tarefa: <= 18 h (conservador 17.9 h).

**Smoke ja feito** (20 passos + retomada do passo 12, `--no-export`; `reports/student/smoke_v3_metrics.json`): 2388 tok/s reais (mediana 2425), 11.84 s/passo,
pico de VRAM alocada 9.17 GB (reservada 12.2 GB), 275 s de GPU no total (<= 8 min). GPU confirmada pelos contadores do Windows (`scripts/export/gpu_probe.ps1`): o pid do treino
ocupou 11,2 GB dedicados e ~96,5% do engine no adaptador da RX 7800 XT (`luid ...12602`); a iGPU ficou em 141 MB e 0%. Pastas `smoke_v3_*` apagadas.

### Antes de iniciar
1. GPU livre (`tmp\gpu.lock` ausente; nenhum llama-server/benchmark). MongoDB do WSL no ar e `wsl -d Ubuntu -e sleep infinity` aberto; Node no PATH (avaliacoes).
2. Corpus integro: `& $PY -I scripts\data\llm\run_guarded.py scripts\student\build_student_corpus_v3.py --verify-only --tiers t1` e
   `& $PY -I scripts\student\train_student.py --config Repos\KapiCoder-Mongo-Small\configs\sft_v3.yaml --dry-run --tok-s 2290 --total-steps 4617` (para regenerar: o mesmo script sem `--verify-only`, ~5 min, CPU, deterministico; SEMPRE via `run_guarded.py`).
3. **Commitar** `configs\student_corpus_v3.yaml` e `Repos\KapiCoder-Mongo-Small\configs\sft_v3.yaml` antes do primeiro passo (o hash entra no `run_manifest.json`).

### Iniciar
```powershell
.\scripts\gpu_queue.ps1 -Status
.\scripts\gpu_queue.ps1 -Start -Only J21-T1-v3
.\scripts\gpu_queue.ps1 -Start -Only J23-T1-v3-eval   # avaliacao + decisao (~1,5-2,5 h)
```
O J21 so inicia com `reports\student\v3_gate_t0.go` (GO do T0); forcar o T1 sem GO e decisao do usuario (criar o arquivo e `-Start -Only J21-T1-v3`).
Manual: `& $PY -I scripts\student\train_student.py --config Repos\KapiCoder-Mongo-Small\configs\sft_v3.yaml --total-steps 4617 --resume auto`. Monitorar/pausar/retomar: como nas secoes anteriores (mesmos arquivos, run `sft_v3`).

### Avaliar (`scripts\student\eval_v3.py --tier t1`)
(a) `eval_student.py run --name sft_v3` no val v2c com a MESMA amostra de 2.000 ids do professor + Generic-JS (500) + medidas/gates; (b) `eval_homolog.py run --name t1_v3` na homologacao v1 (6.030 cortes; execucao do texto real;
leitura de `data/homolog` so pela API autorizada, com log) e `report --out-name homolog-v3-eval-t1` (base x atual x v3, pareado, IC95 bootstrap, McNemar-Holm); (c) val pareado atual x v3 e base x v3 (`reports\student\v3_val_compare_t1.json`);
(d) `decide_v3.py` -> `reports\student\v3_decision_t1.json/.md`. "Atual" = `sft_distill_v1`.

### Criterios de decisao (aplicados por `scripts\student\decide_v3.py`)
- **C1 (meta)**: campos inexistentes em `full` na homologacao **< 9,19%** (modelo atual); ideal **<= base + 1 p.p. (~7-8%)**.
- **C2 (melhora confirmada)**: delta pareado de `no_nonexistent_field` no grupo `full` com **limite inferior do IC95 > 0**.
- **C3 (guarda na homologacao)**: perda de FIM acc (normalized_match) e de exec ÷ teto **<= 1,0 p.p.** contra o atual.
- **C4 (guarda no val/Generic-JS)**: perda no val v2c (2.000 ids) <= 2,0 p.p. em FIM acc e exec ÷ teto; Generic-JS nao cai > 3 p.p. vs zero-shot (G2.5).
- Veredito: **GO** (C1..C4) | INCONCLUSIVO (C1+C3+C4, sem C2) | SEM_MELHORA (C1 falha) | REJEITADO (C3 ou C4 violados).

### Se nao houver melhora (regra)
Qualquer veredito diferente de GO = **nao gastar mais GPU neste retreino**: nao rodar o passo seguinte, nao fazer rodadas extras de hiperparametro/dados sem nova decisao do usuario, **manter o `sft_distill_v1` como modelo publicado**,
registrar o resultado (`reports\student\v3_decision_t1.md`, tabela em `reports\datasets\homolog-v3-eval-t1.md`) e no `docs\resumo-executivo.md` como resultado negativo/inconclusivo. GO: promover `merged\sft_v3`
(atualizar `model.lock.json` com `scripts\student\write_model_lock.py`) e registrar no resumo executivo.

### Riscos
- **Hipotese nao garantida**: a causa pode ser capacidade/generalizacao (o relatorio v1 mostra que `full` nao reduz o erro: 15,3% em full x 14,7% em none), nao so dados; o teste e barato no T0 (~4,5 h), por isso o T0 vai antes.
- Menos destilado efetivo (~58% dos cortes) e 12% de replay: a FIM acc no val v2c pode cair ligeiramente (C4 mede).
- Escopo do filtro de fidelidade: remove ~4-7% dos cortes cujo rotulo cita nomes que o modelo "deveria adivinhar"; pode baixar FIM acc em prompts reais com contexto incompleto.
- Homologacao e do mesmo LLM/gramatica (nao humana) e cortes `grammar` compartilham templates com o treino; uma unica seed de treino (sem variancia entre treinos).
- A homologacao v1 dos modelos ATUAIS foi medida com treino sem decontaminacao n-grama contra a homologacao (achado acima): a comparacao atual x v3 pode subestimar a melhora real ou superestimar o atual.
- Disco: `merged\sft_v3` + checkpoints (T0 full FT ~2 GB fp32 x2; T1 LoRA pequeno). Queda de energia: retoma do ultimo checkpoint (perda <= 30 min).
