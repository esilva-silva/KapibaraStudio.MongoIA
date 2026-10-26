# MODEL_CARD_REVIEW - KapiCoder-Mongo-Small (checklist interno, NAO sobe)

Revise **antes de qualquer publicacao**. O card e `README.md` neste diretorio; os pesos publicados sao `merged/sft_v3` (v3), nada mais.
Estado: **PREVIEW PRIVADO**, nada enviado (sem rede, sem token). Recomendacao: `reports/publish/hf_preview_readiness.md`.

## 1. Decisoes

Ja decididas por voce (nao reabrir sem motivo): namespace `esilva`; repositorios **privados** `esilva/KapiCoder-Mongo-Small` (safetensors) e, **depois**, `esilva/KapiCoder-Mongo-Small-ONNX`;
sem versao no nome (versoes por tag git, primeira `v0.1.0-preview`); professor, dataset e GGUF **nao** sao publicados; `evaluation/` leva so agregados; ONNX so em repositorio separado e so depois.

Ainda abertas:
- [ ] **Contato / canal de feedback.** O card nao traz e-mail nem issue tracker (e o staging recusa e-mails). Decidir se acrescenta algo (nao usar e-mail pessoal sem querer).
- [ ] **Manter ou tirar o `model-index`.** Hoje tem 3 blocos (val v2c sintetico, Generic-JS, homologacao v3), todos "Internal evaluation", `verified: false`. O Hub mostra isso como "evaluation results"; se pesar demais, remova o bloco do front matter
      e ajuste `verify_card_numbers.py` (checagem "model-index ausente") e `tests/test_cards.py` (`test_model_index_*`) juntos.
- [ ] **Marca "Mongo" no nome** (`KapiCoder-Mongo-*`). Conferir as diretrizes de marca da MongoDB, Inc. para o uso de "Mongo"/"MongoDB" em nome de modelo/projeto de terceiros. Isto e um ponto a conferir, **nao e parecer juridico**; o card usa "MongoDB" so como descricao da tarefa e nao afirma afiliacao.
- [ ] **Nome "Kapibara Studio"** aparece no card (e o consumidor do modelo) e **"Qwen"** (atribuicao, com a frase de que nao ha endosso). Confirmar que podem ser citados.
- [ ] **Contrato `editor-context-v1`.** O card descreve publicamente o cabecalho (campos, limites 4096/1024/2048, comandos do dialeto Console) e traz o construtor de prompt (identico a `examples/example_fim.py`). Confirmar que pode ser divulgado e sob qual licenca (o card nao restringe alem da Apache-2.0).
- [ ] **Citar o professor nao publicado** (`KapiTeacher-Mongo-7B`) e o gerador `Qwen3-14B` (Apache-2.0): o card os descreve como parte da origem dos rotulos/consultas. Confirmar.
- [ ] **REPLAY_ATTRIBUTION.md** (555 pacotes, ~46 KB) sobe junto: ok ou reduzir a um resumo por licenca? (E-mails de autores foram removidos.)
- [ ] **Tom do card.** Sem superlativos e sem comparacao com modelos que nao medimos (so base zero-shot, checkpoint interno anterior nao publicado e, na descricao, o professor).

## 2. Pendencias do projeto que o card declara

- Nenhuma medida com consultas escritas por humanos (cego humano descartado em 2026-10-22). A homologacao e sintetica, escrita por Qwen3-14B e gramatica, e **nao e totalmente independente do treino v3**
  (o corpus tem consultas do mesmo gerador e templates de gramatica sobre outros schemas); o card diz que o ganho em consultas escritas pelo LLM provavelmente superestima o ganho em consultas humanas.
- Uma unica execucao de treino por tier (semente 20261018) e uma execucao greedy de avaliacao: sem variancia entre treinos.
- Campos inexistentes em schema novo (`full`): v3 7.50% contra base 7.15% e checkpoint interno anterior 9.19%; o card manda validar nomes de campo antes de executar.
- Gates oficiais (KapiLab, token de GPU, PRs da IDE) pendentes; ONNX v3 sem medidas proprias (por isso o card ONNX diz que nao ha avaliacao e a publicacao ONNX fica para depois).
- `README_job.md` do workspace e `model.lock.json` (braco v3 ausente; `updated_utc 2026-10-18`) nao foram atualizados por mim; do README so a linha "Estado". As fontes do v3 sao `reports/sft_v3/run_manifest.json` e `configs/sft_v3.yaml`.

## 3. Afirmacao do card <-> fonte (resumo; a lista completa, gerada e conferida por script, esta em `SOURCES.md`)

| Afirmacao no card | Fonte | Conferido por script |
|---|---|---|
| Tabelas do val sintetico (FIM acc, exec-valid, exec / teto, syntax, Generic-JS, campos inexistentes) para base, v1 interno e v3 | `reports/eval/<base, sft_distill_v1, sft_v3>/measures.json` (+ `summary.json`) | sim |
| Homologacao (tabelas, lacuna val->hom, deltas pareados, `grammar` x `llm`, alerta de campos inexistentes) e `model-index` | `reports/datasets/homolog-v3-eval-t1.json`, `homolog-v1-manifest.json`, `reports/student/v3_decision_t1.json` | sim (blocos `homolog-*`, `key-caveats`) |
| Treino v3 (exemplos, tokens, passos, horas, metodo, parametros treinaveis, base e revisao) | `reports/sft_v3/run_manifest.json`, `configs/sft_v3.yaml`, `model.lock.json` | sim |
| Composicao do corpus (58% / 26% / 16%, replay 12%, none+irrelevant, decontaminacao contra a homologacao) | `reports/student/corpus_v3_report.md` | sim (trechos) |
| Homologacao nao totalmente independente; ganho provavelmente superestimado; uma semente | inferencia a partir da composicao do corpus (descritivo) | nao (marcado em `SOURCES.md`) |
| Exemplo executavel (saida do exemplo em CPU) | `reports/publish/example_fim_run.txt`; cabecalho testado contra o golden | sim (pytest) |
| Licenca, aviso de modificacao, atribuicao Qwen | `LICENSE` (sha igual ao da base), `NOTICE.md`, `BaseRepos/Qwen2.5-Coder-*/README.md` | parcial |
| Atribuicao do replay | `data/replay/v1/LICENSES.md` -> `REPLAY_ATTRIBUTION.md` | sim (`build_replay_attribution.py --check`) |

## 4. Diferencas entre o pedido e as fontes (o card segue as fontes)

- **Horas**: `elapsed_train_h` = 15.63 h (tempo de treino acumulado, sem avaliacoes); as previsoes anteriores do plano nao sao usadas.
- **"58% do corpus destilado do professor"**: a fonte (`accepted-v2c-teacher`) usa o rotulo do professor so em `fim_ctx`/`completion`; os cortes `fim_raw` desse bloco mantem a referencia (gold). O card descreve assim.
- **Metodo**: T0 = full fine-tuning (2 epocas); T1 = LoRA r=64 mesclado, 1,5 epoca (plano de 2 epocas cortado). Passos deste tier: 4617.
- **Checkpoint v1 na tabela**: aparece so como linha comparativa ("not released"). A homologacao dele foi medida sem decontaminacao contra a homologacao (pode estar levemente otimista, segundo `corpus_v3_report.md`).
- **Diferenca val v1 -> v3** e pequena; o card nao afirma significancia no val (so ha pareado na homologacao).
- **Tag `onnx`** removida do repositorio principal (so safetensors); o staging recusa a tag `onnx` nele.

## 5. Antes de publicar (passo a passo, tudo local ate o ultimo)

1. `python scripts/publish/verify_card_numbers.py` -> OK nos dois tiers; `python scripts/publish/build_replay_attribution.py --check`.
2. `python -m pytest scripts/publish`.
3. `python scripts/publish/stage_hf_package.py --tier both --write-manifest` e `--stage --clean` (monta `cache/hf_stage/<tier>/`, fora do Git).
4. (opcional) `KAPI_PUBLISH_RUN_EXAMPLE=1 KAPI_PUBLISH_PYTHON=<python com torch> python -m pytest scripts/publish -k real_example`.
5. `python scripts/publish/upload_to_hf.py --tier small` (so imprime os comandos exatos e confere o staging contra o manifesto).
6. So com sua aprovacao e sua credencial (HF_TOKEN ou `hf auth login` feito por voce): repetir com `--i-have-user-approval` (`--user-is-logged-in` se usar login). Cria o repositorio **privado**, sobe, cria a tag `v0.1.0-preview` e compara os sha256 do Hub com o manifesto.
7. Abrir o repositorio privado no Hub, conferir o card renderizado e a tag; so entao pensar em ONNX (outro repositorio) ou em publico.
