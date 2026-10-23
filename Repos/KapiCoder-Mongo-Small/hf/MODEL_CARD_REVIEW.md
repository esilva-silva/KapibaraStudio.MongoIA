# MODEL_CARD_REVIEW - KapiCoder-Mongo-Small (checklist interno, NAO sobe)

Revise **antes de qualquer publicacao**. O card e `README.md` neste diretorio. Tudo abaixo e local: nada foi enviado, nenhum token foi usado.
Estado: **PREVIEW**. Recomendacao: criar o repositorio **privado** primeiro (`reports/publish/hf_preview_readiness.md`).

## 1. Decisoes que dependem de voce

- [ ] **Nome do repositorio.** O card e o manifesto assumem `esilva/KapiCoder-Mongo-Small`; o card tambem linka o irmao `esilva/KapiCoder-Mongo-Nano`. Se o namespace/nome mudar: editar o `MODEL_ID` do exemplo, o link do irmao e `suggested_repo_id` (regerar o manifesto).
- [ ] **Publico ou privado.** Recomendado: privado primeiro. `upload_to_hf.py` cria privado por padrao; publico exige `--public`.
- [ ] **Incluir ONNX?** Padrao do manifesto: **nao** (so safetensors bf16, ~2,9 GB). Se sim, variantes sugeridas: `dml-fp16` (2.964 MB, GPU) e `int8` (2.578 MB, CPU); as INT4 perdem 1,4 a 1,8 p.p. de FIM acc e ficam fora. O card ja descreve `onnx/<variante>/`; se NAO incluir ONNX, apagar a tabela "ONNX variants", o aviso do EOS e as linhas de latencia em "Formats" (a tag `onnx` e o teste do staging esperam que o card exista com ela: ajuste os dois juntos).
- [ ] **GGUF.** Nao incluido (pasta `gguf/` do workspace vazia; nenhuma metrica de GGUF dos alunos). Nao oferecer sem medir.
- [ ] **Dataset card / dataset publico?** Recomendado: **nao** publicar dataset agora (nenhum dataset sera enviado; G3.9 licencas pendente; replay npm nao e redistribuido; sem revisao humana G3.7). O card diz "training set is not published". Se quiser publicar, e outro trabalho (licencas, revisao, anonimizacao).
- [ ] **Contato / canal de feedback.** O card nao traz e-mail nem issue tracker. Decidir se adiciona (cuidado: nao colocar e-mail pessoal sem querer).
- [ ] **Contrato `editor-context-v1`.** O card descreve publicamente o formato do cabecalho (campos, limites 4096/1024/2048, comandos do dialeto Console) e traz um construtor de prompt. Confirmar que o texto do contrato pode ser divulgado e sob qual licenca (o card nao restringe alem da Apache-2.0 do modelo). Origem: `data/contracts/editor-context-v1.golden`, `scripts/data/render_contract.py`.
- [ ] **`model-index`.** Mantido (permitido pelo pedido: so metricas internas, rotuladas sinteticas, `verified: false`). O Hub mostra um widget de "evaluation results"; se achar que da peso demais a numeros sinteticos, remova o bloco `model-index` do front matter (o teste de numeros passa a ignorar so se voce tambem tirar a checagem em `verify_card_numbers.py`).
- [ ] **Nome do produto "Kapibara Studio"** aparece no card (e o consumidor). Confirmar que pode ser citado.
- [ ] **Citar o professor nao publicado** (`KapiTeacher-Mongo-7B`): o card o descreve como nao publicado. Confirmar.
- [ ] **REPLAY_ATTRIBUTION.md** (555 pacotes, ~46 KB) sobe junto: ok ou reduzir a um resumo por licenca? (E-mails de autores foram removidos das linhas de copyright.)
- [ ] **Tom do card.** Sem superlativos e sem comparacao com modelos que nao medimos (so base zero-shot, controle gold do T0, professor 7B e o irmao, todos medidos).

## 2. Pendencias do projeto que o card declara (nao sao decisoes, mas precisam estar de acordo com sua leitura)

- Homologacao fora da distribuicao: **TBD** (cego humano descartado em 2026-10-22; nao ha outra medida fora do sintetico).
- Gates oficiais (KapiLab, token de GPU, PRs da IDE) pendentes: `reports/publish/hf_preview_readiness.md`.
- README.md / README_job.md do workspace ainda dizem "treino ainda nao executado" (**desatualizados**; nao foram alterados por pedido). O `model.lock.json` esta com `updated_utc 2026-10-18` e `scripts_dirty: true` (a reproducao exata depende de arvore com alteracoes locais; o card nao promete reproducao).

## 3. Afirmacao do card <-> fonte (resumo; a lista completa, gerada e conferida por script, esta em `SOURCES.md`)

| Afirmacao no card | Fonte | Conferido por script |
|---|---|---|
| Tabelas (FIM acc, exec-valid, exec / teto, syntax, Generic-JS, campos inexistentes por `context_kind`) | `Repos/.../reports/eval/<run>/measures.json` (+ `summary.json`) | sim (`verify_card_numbers.py`) |
| `model-index` | mesmas medidas | sim |
| Base, revisao, exemplos, tokens/epoca | `model.lock.json` | sim |
| Passos, tokens vistos, horas, metodo, parametros treinaveis | `reports/sft_distill_v1/run_manifest.json` | sim |
| LoRA r=64 alpha=128, LR 1,5e-4, 1,5 epoca (3.422 passos) | `configs/sft_distill_v1.yaml`, `README_job.md` | sim (trechos) |
| 12,12% de replay, % de rotulos que diferem do gold | `reports/student/corpus_report.md` | sim |
| Dialetos Console/JSON no treino | `reports/datasets/v2c-validation-report.md` (G3.3) | sim |
| 110 schemas / 14 dominios; teto 0,6255; sem schema 11-14% (ver 4); cego descartado | `docs/resumo-executivo.md` | sim (trechos) |
| Professor: FIM acc 0,41; G2.4 reprovado (-0,15 p.p.); papel de filtro | `Repos/KapiTeacher-Mongo-7B/.../summary.json`, `reports/teacher_adoption.json` | sim |
| T1 x T0: FIM acc +0,75 p.p. (n.s.), exec-valid +1,3 p.p. (p=0,0016) | `Repos/KapiCoder-Mongo-Small/reports/eval/cmp_t0_vs_t1.md`, `docs/resumo-executivo.md` | sim |
| ONNX: tamanhos, paridade, latencia proxy, bug do EOS | `reports/export/onnx_export_t0_t1.md`, `scripts/export/README.md` | sim |
| Licenca Apache-2.0, aviso de modificacao, atribuicao Qwen | `LICENSE`, `BaseRepos/Qwen2.5-Coder-*/README.md`, `kapibarastudio-model.json` | parcial (LICENSE confere por sha com o da base) |
| Atribuicao do replay | `data/replay/v1/LICENSES.md` -> `REPLAY_ATTRIBUTION.md` | sim (`build_replay_attribution.py --check`) |
| Exemplo executavel | executado localmente em CPU contra `merged/sft_distill_v1` (sem rede); `build_prompt` testado contra os SHA do golden | sim (pytest) |
| "Sem dados pessoais por construcao" | README dos workspaces + etapa `sanitize` do v2c; **sem revisao humana** (o card diz isso) | nao (descritivo) |
| "Comportamento com operacoes destrutivas nao foi treinado nem testado" | ausencia nos relatorios; `write_or_destructive_executions: 0` | nao (ausencia) |

## 4. Diferencas entre o pedido e as fontes (o card segue as fontes)

- **"11-14% de campos inexistentes sem schema visivel"**: no KapiCoder-Mongo-Small os numeros sao `irrelevant` 7,58% e `none` 10,97% (e `full` 1,58%). O intervalo 11-14% vale para o T0 (e para o controle gold e o professor), nao para os dois tiers; o card usa os numeros reais do tier.
- **Metodo**: T0 = full fine-tuning; T1 = LoRA r=64 mesclado, **1,5 epoca** (plano de 2 epocas cortado em 3.422 passos). O card descreve o que cada tier fez.
- **Horas**: T0 3,33 h (o controle gold, +3,4 h, nao e publicado); T1 11,74 h. Sao `elapsed_train_h` do `run_manifest.json` (tempo de treino acumulado, sem avaliacoes).
- **Hardware**: RX 7800 XT esta no `run_manifest.json` e nos `summary.json`; o card cita tambem `transformers 4.57.6` e PyTorch ROCm.
- **Distilado x controle**: so existe controle gold no T0; para o T1 o card diz que o efeito da destilacao **nao foi medido**.

## 5. Antes de publicar (passo a passo, tudo local ate o ultimo)

1. `python scripts/publish/verify_card_numbers.py` -> deve imprimir OK nos dois tiers.
2. `python scripts/publish/build_replay_attribution.py --check`.
3. `python -m pytest scripts/publish/tests -q` (os testes de hash completo dos pesos: `KAPI_PUBLISH_FULL_HASH=1`).
4. `python scripts/publish/stage_hf_package.py --tier small` (dry-run; sem `--no-hash`). Para gravar o manifesto depois de qualquer edicao do card: `--write-manifest`.
5. `python scripts/publish/stage_hf_package.py --tier small --stage <DIR vazio> [--include-onnx]`.
6. `python scripts/publish/upload_to_hf.py --tier small --stage-dir <DIR>/small` (imprime o comando). So depois da sua aprovacao: `--i-have-user-approval` com `HF_TOKEN` no ambiente (o token e seu; o script nunca o aceita por argumento).
7. Reabrir o card no Hub e conferir: front matter (licenca, base_model), tabela, exemplo, links.

## 6. Observacoes de verificacao

- O exemplo do card foi rodado em CPU (`CUDA_VISIBLE_DEVICES=-1`) com os pesos `merged` e imprimiu uma continuacao curta (T0: `total`; T1: `_id`) para o prompt do exemplo. Isso e um teste de funcionamento, nao uma metrica.
- Ao carregar o tokenizer localmente com transformers 4.57.6 aparece o aviso "incorrect regex pattern ... Mistral"; os arquivos do tokenizer sao byte a byte os da base Qwen e o GenAI nativo tokeniza igual ao HF em 500/500 prompts (`reports/export/onnx_export_t0_t1.md`). Nao investigado alem disso; confira no Hub com a versao atual do transformers.
- `tokenizer_config.json` e `generation_config.json` vem do `merged/`; nada com caminho local sobe (varredura no staging). `export_info.json` e `train_state.json` sao recusados de proposito.
