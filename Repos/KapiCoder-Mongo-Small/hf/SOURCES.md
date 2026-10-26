# SOURCES - KapiCoder-Mongo-Small (apendice interno, NAO sobe para o Hugging Face)

Cada numero ou afirmacao do `README.md` deste diretorio e conferido por `scripts/publish/verify_card_numbers.py`; esta tabela e gerada por ele
(`--write-sources`). Caminhos relativos a raiz do workspace. Pesos publicados: `merged/sft_v3` (v3); o checkpoint v1 (`sft_distill_v1`) aparece so como linha comparativa, nao publicado.

## Tabelas de metricas (celula = chave do JSON)

| Linha | Arquivo (measures.json) | Chaves |
|---|---|---|
| Qwen2.5-Coder-1.5B, zero-shot base | `Repos/KapiCoder-Mongo-Small/reports/eval/base-qwen2.5-coder-1.5b/measures.json` | `fim_acc`, `exec_valid_autocomplete`, `exec_valid_vs_ceiling`, `syntax_valid`, `generic_js_fim_acc`, `nonexistent_fields_full_rate_pct`, `nonexistent_fields_irrelevant_rate_pct`, `nonexistent_fields_none_rate_pct` |
| Previous internal checkpoint, v1 (not released) | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_distill_v1/measures.json` | `fim_acc`, `exec_valid_autocomplete`, `exec_valid_vs_ceiling`, `syntax_valid`, `generic_js_fim_acc`, `nonexistent_fields_full_rate_pct`, `nonexistent_fields_irrelevant_rate_pct`, `nonexistent_fields_none_rate_pct` |
| **KapiCoder-Mongo-Small** (this model, v3) | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_v3/measures.json` | `fim_acc`, `exec_valid_autocomplete`, `exec_valid_vs_ceiling`, `syntax_valid`, `generic_js_fim_acc`, `nonexistent_fields_full_rate_pct`, `nonexistent_fields_irrelevant_rate_pct`, `nonexistent_fields_none_rate_pct` |

Coerencia cruzada: `summary.json` (`headline.normalized_match`, `exec_valid`, `syntax_valid`) e `<run>-generic-js/summary.json` batem com `measures.json`.

## Fatos de treino (v3)

| Fato | Fonte |
|---|---|
| base, revisao | `Repos/KapiCoder-Mongo-Small/model.lock.json`, `Repos/KapiCoder-Mongo-Small/reports/sft_v3/run_manifest.json` |
| exemplos, replay, tokens por epoca, passos, tokens vistos, horas, metodo, parametros treinaveis, bibliotecas | `Repos/KapiCoder-Mongo-Small/reports/sft_v3/run_manifest.json` |
| hiperparametros, semente (20261018) | `Repos/KapiCoder-Mongo-Small/configs/sft_v3.yaml` |
| pesos publicados (sha256) | `Repos/KapiCoder-Mongo-Small/merged/sft_v3/export_info.json` |
| composicao do corpus v3 (58% destilado do professor, 26% Qwen3-14B gold, 16% gramatica v3 gold, replay 12%, none+irrelevant), decontaminacao contra a homologacao | `reports/student/corpus_v3_report.md` |
| decisao GO do retreino (campos inexistentes em `full`: antes -> v3 -> base) | `reports/student/v3_decision_t1.{md,json}` |
| dataset v2c (funil, dialetos, decontaminacao) | `reports/datasets/v2c-validation-report.md` |
| decisoes e limites (cego descartado, teto 0,6255, 110 schemas) | `docs/resumo-executivo.md` |
| professor (rotulos do corpus destilado) | `Repos/KapiTeacher-Mongo-7B/reports/eval/sft_v2d-final/summary.json` |
| atribuicao do replay | `data/replay/v1/LICENSES.md` -> `REPLAY_ATTRIBUTION.md` |
| base: licenca e revisao | `BaseRepos/Qwen2.5-Coder-*/README.md`, `BaseRepos/base-repos.lock.json` |
| contrato `editor-context-v1` | `data/contracts/editor-context-v1.golden`, `scripts/data/render_contract.py` (exemplo do card testado contra o golden) |

## Homologacao v1 com os pesos v3 (fora da distribuicao; autor Qwen3-14B/gramatica, nao humano)

Tabelas e blocos de leitura do card (`homolog-table`, `homolog-fields-table`, `homolog-gap-table`, `homolog-reading`, `homolog-pt`) e o `model-index` da homologacao sao renderizados/conferidos por `verify_card_numbers.py` a partir de `reports/datasets/homolog-v3-eval-t1.json` (agregados; nenhum item de `data/homolog` e lido). Percentuais com 2 casas = o valor de 4 casas do JSON.

| Celula do card | Chave em `reports/datasets/homolog-v3-eval-t1.json` |
|---|---|
| FIM acc [IC95] | `models.<m>.overall.normalized_match`, `normalized_match_ci95` |
| Exec-valid, sintaxe | `models.<m>.overall.exec_valid`, `syntax_valid` |
| Exec ÷ teto [IC95]; teto | `models.<m>.overall.exec_over_ceiling`, `exec_over_ceiling_ci95`, `ceiling_exec_valid` |
| Campos inexistentes por `context_kind` | `models.<m>.by_context_kind.<c>.nonexistent_fields_rate` (conferido contra `reports/student/v3_decision_tN.json`) |
| Lacuna val -> hom | `val_v2c.<m>.{normalized_match, exec_over_ceiling, nonexistent_fields_full}` vs `models.<m>.overall` / `by_context_kind.full` |
| Deltas pareados, IC95, McNemar-Holm | `pairs.<b>_vs_<a>.context_kind.overall[]` e `pairs.<b>_vs_<a>.{context_kind,author}.groups.<g>.metrics[]` (`delta`, `ci_low`, `ci_high`, `p_holm`) |
| `grammar` x `llm` | `models.<m>.by_author.<g>.exec_over_ceiling`; pares `author.groups` |
| Conjunto selado (28 schemas, n=6.030, sha `aca0aae5...`, menor corte 269) | `reports/datasets/homolog-v1-manifest.json` (conferido contra o eval.json) |
| Protocolo e limites (autor, executar != acertar, teto 93,1% com 418 itens, n por celula, greedy unico) | `reports/datasets/homolog-v1-eval.md` (protocolo da v1; o v3 reaproveita o mesmo conjunto e o mesmo harness, JSON `format: homolog-v1-eval-1`) |

Os `v1-*-report.md` em `reports/datasets/` descrevem o dataset v1 e NAO foram usados para nenhum numero do card.

## Afirmacoes de prosa (trecho do card <-> trecho da fonte)

| Trecho no card | Fonte | Trecho na fonte |
|---|---|---|
| 110 schemas in 14 domains | `docs/resumo-executivo.md` | 110 schemas sintéticos em 14 domínios |
| 61 synthetic v3 training schemas | `reports/student/corpus_v3_report.md` | gramatica dos 61 schemas v3 `train` |
| (MIT, Apache-2.0, BSD, ISC) | `data/replay/v1/LICENSES.md` | permissivas (MIT, Apache-2.0, BSD-2/3-Clause, ISC) |
| 0.6255 | `docs/resumo-executivo.md` | O teto do validador é 0,6255 |
| Hui et al. | `BaseRepos/Qwen2.5-Coder-0.5B/README.md` | arXiv:2409.12186 |
| The 7B teacher | `Repos/KapiTeacher-Mongo-7B/reports/eval/sft_v2d-final/summary.json` | "normalized_match": 0.41 |
| LLM + grammar = 42.0% | `reports/student/corpus_v3_report.md` | LLM + gramatica v3 = **42.0%** |
| 58.0% | `reports/student/corpus_v3_report.md` | \| accepted-v2c-teacher \| teacher_seq (fim_raw: gold) \| |
| 26.0% | `reports/student/corpus_v3_report.md` | \| llm-qwen3-14b \| gold \| |
| 16.0% | `reports/student/corpus_v3_report.md` | \| grammar-v3 \| gold \| |
| 12.0% of the tokens | `reports/student/corpus_v3_report.md` | = **12.0%** dos tokens |
| first training run decontaminated against the sealed homologation set | `reports/student/corpus_v3_report.md` | o v3 e o primeiro treino decontaminado contra a homologacao |
| may be slightly optimistic | `reports/student/corpus_v3_report.md` | a homologacao v1 deles pode estar levemente otimista |
| hyper-parameters are those of the previous run | `Repos/KapiCoder-Mongo-Small/configs/sft_v3.yaml` | Hiperparametros IDENTICOS ao sft_distill_v1.yaml |
| prints `'total'` | `reports/publish/example_fim_run.txt` | saida: 'total' |
| same SHA-256, see `provenance/model_lock.json` | `BaseRepos/base-repos.lock.json` | c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 |
| seed 20261018 | `Repos/KapiCoder-Mongo-Small/configs/sft_v3.yaml` | seed: 20261018 |
| 600 examples of this validation split | `Repos/KapiCoder-Mongo-Small/configs/sft_v3.yaml` | 600 FIM do val v2c |
| 0 write executions | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_v3/measures.json` | "write_or_destructive_executions": 0 |
| 2 s limit | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_v3/summary.json` | "max_time_ms": 2000 |
| seed 20261008 | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_v3/summary.json` | "seed": 20261008 |
| AMD Radeon RX 7800 XT | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_v3/summary.json` | "device_name": "AMD Radeon RX 7800 XT" |
| One AMD Radeon RX 7800 XT | `Repos/KapiCoder-Mongo-Small/reports/sft_v3/run_manifest.json` | 7800 XT |
| transformers 4.57.6 | `Repos/KapiCoder-Mongo-Small/reports/sft_v3/run_manifest.json` | "transformers": "4.57.6" |
| up to 96 new tokens | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_v3/summary.json` | "max_new_tokens": 96 |
| 2048-token context | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_v3/summary.json` | "context_tokens": 2048 |
| (the IDE caps it at 32 tokens) | `Repos/KapiCoder-Mongo-Small/exports/KapiCoder-Mongo-Small-ONNX-int4/kapibarastudio-model.json` | "maxTokens": 32 |
| `Mongo Console JavaScript` and `json` | `reports/datasets/v2c-validation-report.md` | dialetos {'console': 889205, 'json': 111410} |
| 28 schemas | `reports/datasets/homolog-v1-eval.md` | 28 schemas `hom` |
| Qwen3-14B | `reports/datasets/homolog-v1-eval.md` | Autor Qwen3-14B / gramatica |
| `aca0aae5` | `reports/datasets/homolog-v1-manifest.json` | "set_sha256": "aca0aae5 |
| telecom and adtech | `reports/datasets/homolog-v1-eval.md` | telecom/adtech |
| 418 items | `reports/datasets/homolog-v1-eval.md` | 418 itens |
| smallest `cut_kind` has n=269 | `reports/datasets/homolog-v1-eval.md` | menor `cut_kind` tem n=269 |
| single greedy run | `reports/datasets/homolog-v1-eval.md` | Greedy unico |
| McNemar | `reports/datasets/homolog-v1-eval.md` | McNemar exato |
| 95% CI | `reports/datasets/homolog-v1-eval.md` | Wilson (FIM) e bootstrap |
| no write command was forwarded | `reports/datasets/homolog-v1-eval.md` | nenhum comando de escrita encaminhado |
| RX 7800 XT | `reports/datasets/homolog-v1-eval.md` | Todas as geracoes na RX 7800 XT |
| greedy decoding, up to 96 new tokens | `reports/datasets/homolog-v1-eval.md` | greedy, 96 tokens, contexto 2048 |
| 229,834 examples | `Repos/KapiCoder-Mongo-Small/model.lock.json` | "examples": 229834 |
| 292,812 MongoDB examples + 16,759 | `reports/student/corpus_v3_report.md` | (292,812 cortes Mongo + 16,759 replay) |
| 34.8% of the MongoDB cuts | `reports/student/corpus_v3_report.md` | **none+irrelevant = 34.8%** |
| 169,831 | `reports/student/corpus_v3_report.md` | \| accepted-v2c-teacher \| teacher_seq (fim_raw: gold) \| 169,831 \| |
| 76,248 | `reports/student/corpus_v3_report.md` | \| llm-qwen3-14b \| gold \| 76,248 \| |
| 46,733 | `reports/student/corpus_v3_report.md` | \| grammar-v3 \| gold \| 46,733 \| |
| LoRA (r=64 | `Repos/KapiCoder-Mongo-Small/configs/sft_v3.yaml` | r: 64 |
| alpha=128, dropout 0.05 | `Repos/KapiCoder-Mongo-Small/configs/sft_v3.yaml` | alpha: 128 |
| peak LR 1.5e-4 | `Repos/KapiCoder-Mongo-Small/configs/sft_v3.yaml` | lr: 1.5e-4 |
| dropout 0.05 | `Repos/KapiCoder-Mongo-Small/configs/sft_v3.yaml` | dropout: 0.05 |
| 15.63 hours | `Repos/KapiCoder-Mongo-Small/reports/sft_v3/run_manifest.json` | "elapsed_train_h": 15.63 |
| cut to 1.5 epochs | `Repos/KapiCoder-Mongo-Small/configs/sft_v3.yaml` | 1,5 epoca |

## Sem fonte numerica (descritivo; revisar)

- **Homologacao nao totalmente independente do treino v3** (mensagem obrigatoria): decorre da composicao do corpus v3 (26% de consultas do Qwen3-14B, mesmo gerador do conjunto de homologacao, e 16% da gramatica, familia de templates da parte `grammar`), documentada em `corpus_v3_report.md`; os cortes de treino que compartilham n-gramas longos com a homologacao foram removidos por decontaminacao e os 28 schemas dela nunca entraram no treino. O quanto isso infla o resultado nao foi medido.
- "The gain on LLM-written queries probably overstates the gain on queries written by people": inferencia; nao ha medida com consultas humanas.
- Homologacao: "interpretation, not tested" (consultas mais faceis para quem nao treinou) e "why is a hypothesis that was not tested" (schema visivel nao reduziu a taxa) sao hipoteses marcadas como tal no card.
- Homologacao: "the base more often fails to parse, so base rates cover fewer suggestions" decorre da definicao da fonte (campo inexistente so conta quando o comando chega ao servidor) e das taxas de sintaxe; o efeito sobre a taxa nao foi quantificado.
- "Single training run" (uma semente por tier): a semente 20261018 esta no yaml; que nao houve repeticao e a ausencia de qualquer outro run v3 nos relatorios.
- "No personal data by construction": README dos workspaces ("dados sinteticos ... nao contem dados pessoais") e rejeicao de 17 registros com padroes proibidos na etapa sanitize do v2c. Nao ha revisao humana (G3.7 nao feita); o card diz isso.
- "Behaviour around destructive operations was neither trained nor tested": ausencia de qualquer teste desse tipo nos relatorios; so ha `write_or_destructive_executions: 0` (guarda de leitura).
- "Not evaluated: bias or toxicity, long-context beyond 2048, multi-turn, adversarial": ausencia nos relatorios.
- REPLAY_ATTRIBUTION.md lista os 555 pacotes de `data/replay/v1/LICENSES.md`; o replay do corpus v3 vem do mesmo `data/replay/v1` (`configs/student_corpus.yaml`, `replay.source_dir`), mas nao verifiquei pacote a pacote que o v3 usou todos (a lista pode ser um superconjunto do realmente usado).
- Qual checkpoint do run (final ou `best`) foi mesclado em `merged/sft_v3` nao foi conferido; os hashes dos pesos publicados vem de `export_info.json`.
- Nome do repositorio (`esilva/...`), contato e visibilidade: decisao do usuario (ver `MODEL_CARD_REVIEW.md`).
- O exemplo do card foi executado localmente em CPU contra os pesos do staging (sem rede) e a funcao `build_prompt` e testada contra os SHA do golden.
