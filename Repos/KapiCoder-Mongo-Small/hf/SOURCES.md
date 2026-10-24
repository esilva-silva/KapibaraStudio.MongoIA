# SOURCES - KapiCoder-Mongo-Small (apendice interno, NAO sobe para o Hugging Face)

Cada numero ou afirmacao do `README.md` deste diretorio e conferido por `scripts/publish/verify_card_numbers.py`; esta tabela e gerada por ele
(`--write-sources`). Caminhos relativos a raiz do workspace.

## Tabelas de metricas (celula = chave do JSON)

| Linha | Arquivo (measures.json) | Chaves |
|---|---|---|
| Qwen2.5-Coder-1.5B, zero-shot base | `Repos/KapiCoder-Mongo-Small/reports/eval/base-qwen2.5-coder-1.5b/measures.json` | `fim_acc`, `exec_valid_autocomplete`, `exec_valid_vs_ceiling`, `syntax_valid`, `generic_js_fim_acc`, `nonexistent_fields_full_rate_pct`, `nonexistent_fields_irrelevant_rate_pct`, `nonexistent_fields_none_rate_pct` |
| **KapiCoder-Mongo-Small** (this model) | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_distill_v1/measures.json` | `fim_acc`, `exec_valid_autocomplete`, `exec_valid_vs_ceiling`, `syntax_valid`, `generic_js_fim_acc`, `nonexistent_fields_full_rate_pct`, `nonexistent_fields_irrelevant_rate_pct`, `nonexistent_fields_none_rate_pct` |

Coerencia cruzada: `summary.json` (`headline.normalized_match`, `exec_valid`, `syntax_valid`) e `<run>-generic-js/summary.json` batem com `measures.json`.

## Fatos de treino

| Fato | Fonte |
|---|---|
| base, revisao, exemplos, tokens por epoca | `Repos/KapiCoder-Mongo-Small/model.lock.json` |
| passos, tokens vistos, horas, metodo, parametros treinaveis, bibliotecas | `Repos/KapiCoder-Mongo-Small/reports/sft_distill_v1/run_manifest.json` |
| hiperparametros | `Repos/KapiCoder-Mongo-Small/configs/sft_distill_v1.yaml` |
| pesos publicados (sha256) | `Repos/KapiCoder-Mongo-Small/merged/sft_distill_v1/export_info.json` |
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
| 0 write executions | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_distill_v1/measures.json` | "write_or_destructive_executions": 0 |
| 2 s limit | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_distill_v1/summary.json` | "max_time_ms": 2000 |
| seed 20261008 | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_distill_v1/summary.json` | "seed": 20261008 |
| AMD Radeon RX 7800 XT | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_distill_v1/summary.json` | "device_name": "AMD Radeon RX 7800 XT" |
| One AMD Radeon RX 7800 XT | `Repos/KapiCoder-Mongo-Small/reports/sft_distill_v1/run_manifest.json` | 7800 XT |
| transformers 4.57.6 | `Repos/KapiCoder-Mongo-Small/reports/sft_distill_v1/run_manifest.json` | "transformers": "4.57.6" |
| up to 96 new tokens | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_distill_v1/summary.json` | "max_new_tokens": 96 |
| 2048-token context | `Repos/KapiCoder-Mongo-Small/reports/eval/sft_distill_v1/summary.json` | "context_tokens": 2048 |
| (the IDE caps it at 32 tokens) | `Repos/KapiCoder-Mongo-Small/exports/KapiCoder-Mongo-Small-ONNX-int4/kapibarastudio-model.json` | "maxTokens": 32 |
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
| (217,306 MongoDB examples + 12,528 | `reports/student/corpus_report.md` | (217,306 Mongo + 12,528 replay) |
| 54.11% | `reports/student/corpus_report.md` | 54.11% |
| alpha=128, dropout 0.05 | `Repos/KapiCoder-Mongo-Small/configs/sft_distill_v1.yaml` | alpha: 128 |
| peak LR 1.5e-4 | `Repos/KapiCoder-Mongo-Small/configs/sft_distill_v1.yaml` | lr: 1.5e-4 |
| dropout 0.05 | `Repos/KapiCoder-Mongo-Small/configs/sft_distill_v1.yaml` | dropout: 0.05 |
| 3,422 steps | `Repos/KapiCoder-Mongo-Small/README_job.md` | `--total-steps 3422` (1,5 epoca) |
| \| `onnx/dml-fp16` \| 2,964 MB | `reports/export/onnx_export_t0_t1.md` | `dml-fp16` 2.964 MB |
| \| `onnx/int8` \| 2,578 MB | `reports/export/onnx_export_t0_t1.md` | `int8` 2.578 MB |
| identical in 96.5% of prompts | `reports/export/onnx_export_t0_t1.md` | \| dml-fp16 \| 2964 \| 96,5 \| |
| identical in 94.5% of prompts | `reports/export/onnx_export_t0_t1.md` | \| int8 \| 2578 \| 94,5 \| |
| 0.4 points below the bf16 weights (not significant, p=0.50) | `reports/export/onnx_export_t0_t1.md` | 0,408 (−0,4; 0,50) |
| (1,090 MB and 952 MB for CPU, 1,293 MB for DirectML) | `reports/export/onnx_export_t0_t1.md` | `int4` 1.090 MB, `int4-rtn` 952 MB |
| fell by 1.4 to 1.8 points (p between 0.004 and 0.04) | `reports/export/onnx_export_t0_t1.md` | **−1,8; 0,004** |
| lost 1.4 to 1.8 FIM-accuracy points | `reports/export/onnx_export_t0_t1.md` | **−1,4; 0,039** |
| 148 ms on average and 170 ms at p95 | `reports/export/onnx_export_t0_t1.md` | **148/170** |
| (104 tokens/s) | `reports/export/onnx_export_t0_t1.md` | \| 104 \| |
| 697 ms on average and 1192 ms at p95 | `reports/export/onnx_export_t0_t1.md` | 697/1192 |
| internal target of 500/650 ms | `reports/export/onnx_export_t0_t1.md` | T1 500/650 |
| FIM accuracy is 0.75 points higher (not significant) | `docs/resumo-executivo.md` | FIM acc +0,75 p.p., **não significativo** |
| exec-valid 1.3 points higher (p=0.0016) | `docs/resumo-executivo.md` | exec-valid +1,3 p.p. (p=0,0016) |
| (1.58% against 2.06%) | `Repos/KapiCoder-Mongo-Nano/reports/eval/sft_distill_v1/measures.json` | "nonexistent_fields_full_rate_pct": 2.06 |
| No control trained on the original reference labels | `reports/teacher_adoption.json` | o controle gold do T1 (J10) fica fora |
| FIM accuracy 0.403 against 0.4115 | `Repos/KapiCoder-Mongo-Nano/reports/eval/compare/sft_distill_v1_vs_sft_gold_v1.md` | \| exact_match \| 2000 \| 0.4115 \| 0.4030 \| |

## Sem fonte numerica (descritivo; revisar)

- Homologacao: "the gap of the trained model may be understated" e "why is a hypothesis that was not tested" sao interpretacao do relatorio da fonte (`homolog-v1-eval.md`, secoes Interpretacao e Contexto), marcadas como hipotese no card; nao ha experimento que as teste.
- Homologacao: "the base more often fails to parse, so base rates cover fewer suggestions" decorre da definicao da fonte (campo inexistente so conta quando o comando chega ao servidor) e das taxas de sintaxe; o efeito sobre a taxa nao foi quantificado.
- "No personal data by construction": README dos workspaces ("dados sinteticos ... nao contem dados pessoais") e rejeicao de 17 registros com padroes proibidos na etapa sanitize do v2c. Nao ha revisao humana (G3.7 nao feita); o card diz isso.
- "Behaviour around destructive operations was neither trained nor tested": ausencia de qualquer teste desse tipo nos relatorios; so ha `write_or_destructive_executions: 0` (guarda de leitura).
- "Not evaluated: bias or toxicity, long-context beyond 2048, multi-turn, adversarial": ausencia nos relatorios.
- Nome do repositorio (`esilva/...`), contato e visibilidade: decisao do usuario (ver `MODEL_CARD_REVIEW.md`).
- O exemplo do card foi executado localmente em CPU contra os pesos `merged` (sem rede) e a funcao `build_prompt` e testada contra os SHA do golden.
