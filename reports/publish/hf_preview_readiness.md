# Prontidao para publicar T0 Nano e T1 Small (pesos v3) em PREVIEW PRIVADO no Hugging Face

Data: 2026-10-26 (substitui a versao de 2026-10-23/24, que tratava os pesos v1 `sft_distill_v1`). **Nada foi enviado**: sem rede, sem token, sem upload, sem commit; nenhum pacote foi instalado.
Tudo abaixo e preparacao local e verificavel. O staging de verdade esta em `cache/hf_stage/{nano,small}/` (fora do Git, ignorado por `cache/` no `.gitignore`).

Fontes dos numeros (cada afirmacao do card esta em `Repos/*/hf/SOURCES.md`): `reports/datasets/homolog-v3-eval-t{0,1}.json`, `reports/datasets/homolog-v1-manifest.json`, `reports/student/v3_decision_t{0,1}.{md,json}`,
`reports/student/corpus_v3_report.md`, `Repos/*/reports/eval/sft_v3/{summary,measures}.json`, `Repos/*/reports/sft_v3/run_manifest.json`, `Repos/*/configs/sft_v3.yaml`, `docs/resumo-executivo.md`. Nenhum item de `data/homolog` foi lido.

## 1. Recomendacao

**Publicar apenas como PREVIEW PRIVADO**, os dois tiers, repositorios `esilva/KapiCoder-Mongo-Nano` e `esilva/KapiCoder-Mongo-Small`, so safetensors bf16 + tokenizer + card, tag `v0.1.0-preview`.
Repositorios ONNX (`esilva/KapiCoder-Mongo-{Nano,Small}-ONNX`) ficam para **depois**: o suporte no staging e no upload esta pronto e desligado por padrao (`--target onnx`), mas os pacotes v3 ainda estao sendo produzidos/qualificados por outra frente e nao tem medida propria.
**Nao tornar publico** enquanto nao houver medida com consultas escritas por humanos ou uma decisao consciente de aceitar so a avaliacao sintetica, e enquanto a IDE nao garantir schema no contexto e validacao de campos antes de executar.

## 2. Numeros do card (avaliacao interna, dado sintetico; `verified: false`)

Val v2c sintetico (n=2.000):

| | FIM acc | Exec-valid | Exec / teto | Campos inexistentes `full` / `irrelevant` / `none` | Generic-JS |
|---|---|---|---|---|---|
| T0 Nano v3 (v1 interno; base) | 0,405 (0,403; 0,076) | 0,5945 (0,582; 0,213) | 0,9504 (0,9305; 0,3405) | 1,34 / 8,59 / 9,03 % (v1: 2,06 / 11,11 / 12,90) | 0,530 (0,518; 0,340) |
| T1 Small v3 (v1 interno; base) | 0,410 (0,4105; 0,142) | 0,6015 (0,595; 0,316) | 0,9616 (0,9512; 0,5052) | 1,21 / 5,56 / 7,10 % (v1: 1,58 / 7,58 / 10,97) | 0,606 (0,632; 0,406) |

Homologacao v1 com os pesos v3 (6.030 cortes, 28 schemas novos, escrita por Qwen3-14B e gramatica, **nao humana**, **nao totalmente independente do treino v3**):

| | FIM acc | Exec / teto | Campos inexistentes `full` | `irrelevant` | `none` |
|---|---|---|---|---|---|
| T0 v3 | 42,34% (v1 33,10%; base 12,62%) | 86,17% (v1 74,43%; base 39,54%) | **10,81%** (v1 15,35%; base 7,02%) | 9,09% (base 8,40%) | 9,18% (base 10,91%) |
| T1 v3 | 45,37% (v1 36,97%; base 18,74%) | 89,26% (v1 84,44%; base 52,81%) | **7,50%** (v1 9,19%; base 7,15%) | 8,23% (base 9,61%) | 8,74% (base 10,04%) |

Leitura (o card diz o mesmo): no T0 o v3 ainda inventa mais campos que a base em schema novo com schema visivel (IC95 pareado exclui zero); no T1 esta proximo da base (IC95 pareado inclui zero, sem teste de equivalencia).
O FIM acc da homologacao ficar acima do val no v3 **nao** e evidencia de melhor generalizacao: o corpus v3 tem consultas do mesmo gerador LLM e templates de gramatica sobre outros schemas, entao o ganho em consultas escritas pelo LLM provavelmente superestima o ganho em consultas humanas. Uma so semente de treino por tier.

## 3. O que esta pronto

| Item | Onde | Estado |
|---|---|---|
| Model cards v3 (en primeiro, pt-BR curto), front matter YAML, `model-index` rotulado interno/sintetico (`verified: false`), bloco "Read this first" | `Repos/KapiCoder-Mongo-{Nano,Small}/hf/README.md` | pronto; tabelas e blocos renderizados a partir dos JSON e conferidos por script |
| Aviso de modificacao e atribuicao ao Qwen (e ao gerador Qwen3-14B) | `hf/NOTICE.md` + `LICENSE` (sha igual ao da base) | pronto |
| Atribuicao do replay npm (555 pacotes, sem e-mails) | `hf/REPLAY_ATTRIBUTION.md` | pronto (lista a populacao de `data/replay/v1`; ver §6) |
| Arvore do repositorio principal (raiz plana; `provenance/`, `evaluation/`, `examples/`) | gerada por `stage_hf_package.py`; qualquer arquivo fora da arvore autorizada e erro | pronto; 16 arquivos (T0 953,5 MB; T1 2,9 GB) |
| `provenance/model_lock.json` (base+revisao, hashes, corpus v3, commit de treino, sem caminhos locais), `provenance/upload_manifest.json` (sha256 de tudo), `evaluation/{val_v2c,homologation_v1}_summary.json` (so agregados), `examples/example_fim.py` | gerados no staging | pronto |
| Manifesto interno com tamanhos e sha256 | `Repos/*/hf/upload_manifest.json` | gerado e conferido (`--check-manifest`); muda se qualquer script de `scripts/publish` ou card mudar: regenerar antes de subir |
| Suporte ONNX separado (pastas `cpu-int4/`, `cpu-int8/`, `dml-fp16/` no Nano; `cpu-int8/`, `dml-fp16/` no Small; `export_manifest.json` sanitizado; cards `hf/ONNX_README.md`, `hf/ONNX_NOTICE.md`) | `stage_hf_package.py --target onnx`, `upload_to_hf.py --target onnx` | pronto, **desligado por padrao**, nao incluido em nenhum manifesto |
| Upload protegido: so imprime criar-repo-privado, subir, tag `v0.1.0-preview` e verificacao (sha256 do manifesto x Hub); executa so com `--i-have-user-approval` **e** credencial do proprio usuario (`HF_TOKEN` ou `--user-is-logged-in`) | `scripts/publish/upload_to_hf.py` | pronto, nunca executado |
| Conferencia dos numeros do card contra os JSON (inclui `homolog-v3-eval-*.json`, `v3_decision_*`, summaries v3) | `scripts/publish/verify_card_numbers.py` | OK nos dois tiers; falha se o card divergir |
| Testes | `scripts/publish/tests` | 82 passam, 2 pulados (execucao real do exemplo, opt-in); nenhum teste usa rede |

## 4. Exemplo executado de verdade (CPU, sem rede)

`examples/example_fim.py` do staging, pesos reais (`merged/sft_v3`, sha256 conferidos), prompt `editor-context-v1` (cabecalho testado contra `data/contracts/editor-context-v1.golden`), `HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1`,
python 3.12.10, torch 2.10 (ROCm, sem dispositivo visivel), transformers 4.57.6, bf16, greedy, 32 tokens: **T0 `'total'`, T1 `'total'`** (cerca de 5 s cada, com o modelo ja em disco). Registro: `reports/publish/example_fim_run.txt`.
O transformers 4.57.6 imprime um aviso sobre "regex pattern" ao carregar o tokenizer; os arquivos do tokenizer sao identicos aos da base (mesmo sha256 do lock), e o card menciona isso.

## 5. Verificacoes feitas (2026-10-26)

- `verify_card_numbers.py`: OK nos dois tiers (tabelas, model-index, fatos de treino, afirmacoes de prosa, frases obrigatorias, homologacao x `v3_decision_*`).
- `stage_hf_package.py --tier both` (dry-run com hash) e `--stage --clean`: OK; sem caminho local, token, e-mail, dado de treino ou arquivo do outro workspace (varredura de texto, JSON gerado e cabecalho do safetensors); arvore == a autorizada; sha256 dos pesos == `export_info.json`; LICENSE == o da base; front matter valido; exemplo do card == arquivo publicado.
- `upload_to_hf.py --tier nano|small` (dry-run): staging confere com o manifesto; imprime os comandos; nada enviado. `--i-have-user-approval` sem credencial: recusado (testado).
- `--target onnx` em dry-run sem hash nos pacotes v3 existentes no momento: plano OK para o Nano (3 variantes); nao e parte da publicacao agora.
- `pytest scripts/publish`: 82 passed, 2 skipped. `build_replay_attribution.py --check`: OK.

## 6. Riscos e afirmacoes sem fonte numerica

1. **Reputacao.** Os numeros (FIM acc 0,40-0,41, homologacao 42-45%) vem de dado sintetico e de consultas escritas por LLM; o card repete o rotulo no topo, nas tabelas e no resumo pt-BR, mas a citacao fora de contexto continua possivel.
2. **Campos inexistentes** em schema novo (T0 acima da base; T1 proximo): a mitigacao e o aviso do card ("valide nomes de campo antes de executar") e a IDE enviar o schema.
3. **Independencia da homologacao.** Inferencia a partir da composicao do corpus (`corpus_v3_report.md`); o quanto o v3 se beneficia disso nao foi medido.
4. **Uma semente** por tier: diferencas v1 -> v3 nao tem repeticao.
5. **Sem fonte numerica (descritivo)**: "sem dados pessoais por construcao"; "comportamento com operacoes destrutivas nao treinado nem testado"; "nao avaliado: vies, contexto longo, multi-turno, adversarial"; "ganho em consultas do LLM provavelmente superestima o humano"; qual checkpoint do run (final ou `best`) virou `merged/sft_v3` nao foi conferido; `REPLAY_ATTRIBUTION.md` lista os 555 pacotes de `data/replay/v1` e nao conferi pacote a pacote se o replay v3 usou todos (pode ser superconjunto). Lista completa em `SOURCES.md`.
6. **Irreversibilidade.** Mesmo privado, o upload e um ato externo; por isso so com sua aprovacao e credencial.
7. **Higiene local** (fora do meu escopo): `model.lock.json` sem o braco v3 e `README_job.md` desatualizados; copias `merged/sft_distill_v1` e `sft_gold_v1` ficam fora do staging.

## 7. O que voce ainda precisa decidir

- **Contato / canal de feedback no card** (hoje nenhum; o staging recusa e-mails, entao qualquer contato entra como link ou texto sem e-mail, ou pelo Hub).
- **Manter ou tirar o `model-index`** (3 blocos "Internal evaluation", `verified: false`); como retirar esta em `MODEL_CARD_REVIEW.md` §1.
- **Atencao a marca "Mongo" no nome** (`KapiCoder-Mongo-*`): ponto a conferir com as diretrizes de marca da MongoDB, Inc. (nao e parecer juridico); o card usa MongoDB so para descrever a tarefa e nao afirma afiliacao. Alternativa se houver duvida: renomear antes de criar o repositorio (muda `TIER_SPECS`, o card e o exemplo).
- Citar **"Kapibara Studio"**, **"Qwen"** e o contrato publico `editor-context-v1`; manter os 555 pacotes em `REPLAY_ATTRIBUTION.md` ou resumir por licenca.
- Quando (e se) publicar o ONNX: pacotes v3 completos e qualificados, decisao do EOS do GenAI 0.15.2 na IDE, e o que o card ONNX pode afirmar sobre qualidade (hoje: nada, sem medida propria).

## 8. Como seguir (tudo local ate o ultimo passo)

```
python scripts/publish/verify_card_numbers.py
python -m pytest scripts/publish
python scripts/publish/stage_hf_package.py --tier both --write-manifest
python scripts/publish/stage_hf_package.py --tier both --stage --clean
python scripts/publish/upload_to_hf.py --tier nano       # so imprime
python scripts/publish/upload_to_hf.py --tier small
# somente com a sua aprovacao e a sua credencial:
python scripts/publish/upload_to_hf.py --tier nano --i-have-user-approval [--user-is-logged-in]
```
