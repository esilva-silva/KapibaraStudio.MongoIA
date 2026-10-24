# SOURCES - KapiCoder-Mongo-Nano (apendice interno, NAO sobe para o Hugging Face)

Cada numero ou afirmacao do `README.md` deste diretorio e conferido por `scripts/publish/verify_card_numbers.py`; esta tabela e gerada por ele
(`--write-sources`). Caminhos relativos a raiz do workspace.

## Tabelas de metricas (celula = chave do JSON)

| Linha | Arquivo (measures.json) | Chaves |
|---|---|---|
| Qwen2.5-Coder-0.5B, zero-shot base | `Repos/KapiCoder-Mongo-Nano/reports/eval/base-qwen2.5-coder-0.5b/measures.json` | `fim_acc`, `exec_valid_autocomplete`, `exec_valid_vs_ceiling`, `syntax_valid`, `generic_js_fim_acc`, `nonexistent_fields_full_rate_pct`, `nonexistent_fields_irrelevant_rate_pct`, `nonexistent_fields_none_rate_pct` |
| **KapiCoder-Mongo-Nano** (this model) | `Repos/KapiCoder-Mongo-Nano/reports/eval/sft_distill_v1/measures.json` | `fim_acc`, `exec_valid_autocomplete`, `exec_valid_vs_ceiling`, `syntax_valid`, `generic_js_fim_acc`, `nonexistent_fields_full_rate_pct`, `nonexistent_fields_irrelevant_rate_pct`, `nonexistent_fields_none_rate_pct` |
| 0.5B trained on original labels (control, not released) | `Repos/KapiCoder-Mongo-Nano/reports/eval/sft_gold_v1/measures.json` | `fim_acc`, `exec_valid_autocomplete`, `exec_valid_vs_ceiling`, `syntax_valid`, `generic_js_fim_acc`, `nonexistent_fields_full_rate_pct`, `nonexistent_fields_irrelevant_rate_pct`, `nonexistent_fields_none_rate_pct` |

Coerencia cruzada: `summary.json` (`headline.normalized_match`, `exec_valid`, `syntax_valid`) e `<run>-generic-js/summary.json` batem com `measures.json`.

## Fatos de treino

| Fato | Fonte |
|---|---|
| base, revisao, exemplos, tokens por epoca | `Repos/KapiCoder-Mongo-Nano/model.lock.json` |
| passos, tokens vistos, horas, metodo, parametros treinaveis, bibliotecas | `Repos/KapiCoder-Mongo-Nano/reports/sft_distill_v1/run_manifest.json` |
| hiperparametros | `Repos/KapiCoder-Mongo-Nano/configs/sft_distill_v1.yaml` |
| pesos publicados (sha256) | `Repos/KapiCoder-Mongo-Nano/merged/sft_distill_v1/export_info.json` |
| composicao do corpus, replay 12%, rotulo difere do gold | `reports/student/corpus_report.md` |
| dataset v2c (funil, dialetos, decontaminacao) | `reports/datasets/v2c-validation-report.md` |
| decisoes e limites (cego descartado, teto 0,6255, sem schema 11-14%) | `docs/resumo-executivo.md` |
| professor, G2.4, papel de filtro | `reports/teacher_adoption.json`, `Repos/KapiTeacher-Mongo-7B/README_job.md` |
| atribuicao do replay | `data/replay/v1/LICENSES.md` -> `REPLAY_ATTRIBUTION.md` |
| base: licenca e revisao | `BaseRepos/Qwen2.5-Coder-*/README.md`, `BaseRepos/base-repos.lock.json` |
| contrato `editor-context-v1` | `data/contracts/editor-context-v1.golden`, `scripts/data/render_contract.py` (exemplo do card testado contra o golden) |
| ONNX: variantes, tamanhos, paridade, latencia proxy, bug do EOS | `reports/export/onnx_export_t0_t1.md`, `scripts/export/README.md`, `kapibarastudio-model.json` / `kapicoder_manifest.json` de cada pacote |

## Homologacao v1 (fora da distribuicao; autor Qwen3-14B/gramatica, nao humano)

Tabelas e blocos de leitura do card (`homolog-table`, `homolog-fields-table`, `homolog-gap-table`, `homolog-reading`, `homolog-pt`) e o `model-index` da homologacao sao renderizados/conferidos por `verify_card_numbers.py` a partir de `reports/datasets/homolog-v1-eval.json` (agregados; nenhum item de `data/homolog` e lido). Percentuais com 2 casas = o valor de 4 casas do JSON (o `.md` da fonte arredonda para 1 casa).

| Celula do card | Chave em `reports/datasets/homolog-v1-eval.json` |
|---|---|
| FIM acc [IC95] | `models.<m>.overall.normalized_match`, `normalized_match_ci95` |
| Exec-valid, sintaxe | `models.<m>.overall.exec_valid`, `syntax_valid` |
| Exec ÷ teto [IC95]; teto | `models.<m>.overall.exec_over_ceiling`, `exec_over_ceiling_ci95`, `ceiling_exec_valid` |
| Campos inexistentes por `context_kind` | `models.<m>.by_context_kind.<c>.nonexistent_fields_rate` |
| Lacuna val -> hom | `val_v2c.<m>.{normalized_match, exec_over_ceiling, nonexistent_fields_full}` vs `models.<m>.overall` / `by_context_kind.full` |
| Deltas pareados, IC95, McNemar-Holm | `pairs.<b>_vs_<a>.context_kind.overall[]` e `pairs.<b>_vs_<a>.author.groups.<g>.metrics[]` (`delta`, `ci_low`, `ci_high`, `p_holm`) |
| `grammar` x `llm` | `models.<m>.by_author.<g>.exec_over_ceiling`; pares `author.groups` |
| Conjunto selado (28 schemas, n=6.030, sha `aca0aae5...`, menor corte 269) | `reports/datasets/homolog-v1-manifest.json` (conferido contra o eval.json) |
| Protocolo e limites (autor, executar != acertar, teto 93,1% com 418 itens, n por celula, greedy unico) | `reports/datasets/homolog-v1-eval.md` |

Os `v1-*-report.md` em `reports/datasets/` descrevem o dataset v1 e NAO foram usados para nenhum numero do card.

## Afirmacoes de prosa (trecho do card <-> trecho da fonte)

| Trecho no card | Fonte | Trecho na fonte |
|---|---|---|
| Holm-adjusted p=0.019 | `docs/resumo-executivo.md` | Holm p=0,019 |
| difference -0.85 percentage points | `reports/teacher_adoption.json` | "fim_acc_delta_pp": -0.85 |
| failed (-0.15 points) | `reports/teacher_adoption.json` | "value_pp": -0.15 |
| The 7B teacher | `Repos/KapiTeacher-Mongo-7B/reports/eval/sft_v2d-final/summary.json` | "normalized_match": 0.41 |
| 110 schemas in 14 domains | `docs/resumo-executivo.md` | 110 schemas sintéticos em 14 domínios |
| 12.12% | `reports/student/corpus_report.md` | 12.12% |
| (MIT, Apache-2.0, BSD, ISC) | `data/replay/v1/LICENSES.md` | permissivas (MIT, Apache-2.0, BSD-2/3-Clause, ISC) |
| 0.6255 | `docs/resumo-executivo.md` | O teto do validador é 0,6255 |
| Hui et al. | `BaseRepos/Qwen2.5-Coder-0.5B/README.md` | arXiv:2409.12186 |
| `kapilab-kapi-model-v1` | `reports/export/onnx_export_t0_t1.md` | kapilab-kapi-model-v1 |
| does not append that token | `scripts/export/README.md` | o acrescenta a sequencia |
| without a workaround | `scripts/export/README.md` | os pacotes entregues NAO foram alterados |
| 0 write executions | `Repos/KapiCoder-Mongo-Nano/reports/eval/sft_distill_v1/measures.json` | "write_or_destructive_executions": 0 |
| 2 s limit | `Repos/KapiCoder-Mongo-Nano/reports/eval/sft_distill_v1/summary.json` | "max_time_ms": 2000 |
| seed 20261008 | `Repos/KapiCoder-Mongo-Nano/reports/eval/sft_distill_v1/summary.json` | "seed": 20261008 |
| AMD Radeon RX 7800 XT | `Repos/KapiCoder-Mongo-Nano/reports/eval/sft_distill_v1/summary.json` | "device_name": "AMD Radeon RX 7800 XT" |
| One AMD Radeon RX 7800 XT | `Repos/KapiCoder-Mongo-Nano/reports/sft_distill_v1/run_manifest.json` | 7800 XT |
| transformers 4.57.6 | `Repos/KapiCoder-Mongo-Nano/reports/sft_distill_v1/run_manifest.json` | "transformers": "4.57.6" |
| up to 96 new tokens | `Repos/KapiCoder-Mongo-Nano/reports/eval/sft_distill_v1/summary.json` | "max_new_tokens": 96 |
| 2048-token context | `Repos/KapiCoder-Mongo-Nano/reports/eval/sft_distill_v1/summary.json` | "context_tokens": 2048 |
| (the IDE caps it at 32 tokens) | `Repos/KapiCoder-Mongo-Nano/exports/KapiCoder-Mongo-Nano-ONNX-int4/kapibarastudio-model.json` | "maxTokens": 32 |
| `Mongo Console JavaScript` and `json` | `reports/datasets/v2c-validation-report.md` | dialetos {'console': 889205, 'json': 111410} |
| 28 schemas | `reports/datasets/homolog-v1-eval.md` | 28 schemas `hom` |
| Qwen3-14B | `reports/datasets/homolog-v1-eval.md` | Autor Qwen3-14B / gramatica |
| `aca0aae5` | `reports/datasets/homolog-v1-manifest.json` | "set_sha256": "aca0aae5 |
| telecom and adtech | `reports/datasets/homolog-v1-eval.md` | telecom/adtech |
| 418 items | `reports/datasets/homolog-v1-eval.md` | 418 itens |
| smallest `cut_kind` has n=269 | `reports/datasets/homolog-v1-eval.md` | menor `cut_kind` tem n=269 |
| single greedy run | `reports/datasets/homolog-v1-eval.md` | Greedy unico |
| NF4 | `reports/datasets/homolog-v1-eval.md` | NF4+LoRA |
| McNemar | `reports/datasets/homolog-v1-eval.md` | McNemar exato |
| 95% CI | `reports/datasets/homolog-v1-eval.md` | Wilson (FIM) e bootstrap |
| no write command was forwarded | `reports/datasets/homolog-v1-eval.md` | nenhum comando de escrita encaminhado |
| RX 7800 XT | `reports/datasets/homolog-v1-eval.md` | Todas as geracoes na RX 7800 XT |
| greedy decoding, up to 96 new tokens | `reports/datasets/homolog-v1-eval.md` | greedy, 96 tokens, contexto 2048 |
| (151,180 MongoDB examples + 8,685 | `reports/student/corpus_report.md` | (151,180 Mongo + 8,685 replay) |
| 54.27% | `reports/student/corpus_report.md` | 54.27% |
| \| `onnx/int4` \| 392 MB | `reports/export/onnx_export_t0_t1.md` | `int4` 392 MB |
| \| `onnx/dml-fp16` \| 959 MB | `reports/export/onnx_export_t0_t1.md` | `dml-fp16` 959 MB |
| identical in 79.0% of prompts | `reports/export/onnx_export_t0_t1.md` | \| int4 \| 392 \| 79,0 \| |
| identical in 97.5% of prompts | `reports/export/onnx_export_t0_t1.md` | \| dml-fp16 \| 959 \| 97,5 \| |
| within 0.8 points of the bf16 weights, none significant (p >= 0.34) | `reports/export/onnx_export_t0_t1.md` | \|Δ\| ≤ 0,8 pp, todos p ≥ 0,34 |
| 285 ms on average and 480 ms at p95 | `reports/export/onnx_export_t0_t1.md` | \| 285/480 \| |
| 109/124 ms | `reports/export/onnx_export_t0_t1.md` | **109/124** |
| internal target of 150/250 ms | `reports/export/onnx_export_t0_t1.md` | T0 150/250 ms |
| exec-valid is 1.7 points higher | `docs/resumo-executivo.md` | exec-valid +1,7 p.p. |
| FIM accuracy 0.4115 against 0.403 | `Repos/KapiCoder-Mongo-Nano/reports/eval/compare/sft_distill_v1_vs_sft_gold_v1.md` | \| exact_match \| 2000 \| 0.4115 \| 0.4030 \| |

## Sem fonte numerica (descritivo; revisar)

- Homologacao: "the gap of the trained model may be understated" e "why is a hypothesis that was not tested" sao interpretacao do relatorio da fonte (`homolog-v1-eval.md`, secoes Interpretacao e Contexto), marcadas como hipotese no card; nao ha experimento que as teste.
- Homologacao: "the base more often fails to parse, so base rates cover fewer suggestions" decorre da definicao da fonte (campo inexistente so conta quando o comando chega ao servidor) e das taxas de sintaxe; o efeito sobre a taxa nao foi quantificado.
- "No personal data by construction": README dos workspaces ("dados sinteticos ... nao contem dados pessoais") e rejeicao de 17 registros com padroes proibidos na etapa sanitize do v2c. Nao ha revisao humana (G3.7 nao feita); o card diz isso.
- "Behaviour around destructive operations was neither trained nor tested": ausencia de qualquer teste desse tipo nos relatorios; so ha `write_or_destructive_executions: 0` (guarda de leitura).
- "Not evaluated: bias or toxicity, long-context beyond 2048, multi-turn, adversarial": ausencia nos relatorios.
- Nome do repositorio (`esilva/...`), contato e visibilidade: decisao do usuario (ver `MODEL_CARD_REVIEW.md`).
- O exemplo do card foi executado localmente em CPU contra os pesos `merged` (sem rede) e a funcao `build_prompt` e testada contra os SHA do golden.
