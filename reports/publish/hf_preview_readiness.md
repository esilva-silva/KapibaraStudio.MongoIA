# Prontidao para publicar T0 Nano e T1 Small em PREVIEW no Hugging Face

Data: 2026-10-23. **Nada foi enviado**: sem rede, sem token, sem upload; nenhum pacote foi instalado. Tudo abaixo e preparacao local e verificavel.
Fontes dos numeros: `docs/resumo-executivo.md`, `Repos/*/reports/eval/*`, `reports/export/onnx_export_t0_t1.md` (cada afirmacao do card esta em `Repos/*/hf/SOURCES.md`).

## 1. O que esta pronto

| Item | Onde | Estado |
|---|---|---|
| Model card bilingue (en primeiro, pt-BR curto), front matter YAML, `model-index` rotulado interno/sintetico (`verified: false`) | `Repos/KapiCoder-Mongo-{Nano,Small}/hf/README.md` | pronto, conferido por script |
| Aviso de modificacao e atribuicao ao Qwen | `hf/NOTICE.md` + `LICENSE` (sha igual ao da base) | pronto |
| Atribuicao do replay npm (555 pacotes, sem e-mails de autores) | `hf/REPLAY_ATTRIBUTION.md` (gerado de `data/replay/v1/LICENSES.md`) | pronto |
| Checklist de revisao e apendice afirmacao <-> fonte | `hf/MODEL_CARD_REVIEW.md`, `hf/SOURCES.md` | pronto (internos, nao sobem) |
| Manifesto de upload com tamanhos e sha256 (pacote padrao + variantes ONNX opcionais) | `hf/upload_manifest.json` | gerado; T0 0,95 GB, T1 3,1 GB (so safetensors + tokenizer + card) |
| Staging em dry-run, varredura de caminhos/segredos/dados, conferencia de hashes | `scripts/publish/stage_hf_package.py` | pronto |
| Envio protegido (imprime o comando; so executa com `--i-have-user-approval` **e** `HF_TOKEN`) | `scripts/publish/upload_to_hf.py` | pronto, nunca executado |
| Conferencia dos numeros do card contra os JSON de avaliacao | `scripts/publish/verify_card_numbers.py` | OK nos dois tiers |
| Testes | `scripts/publish/tests` | 48 passam (inclui golden do contrato, recusa sem aprovacao/token, nenhuma rede) |

Numeros do card (avaliacao interna, dado sintetico, n=2.000, val v2c): T0 FIM acc 0,403, exec-valid 0,582, exec/teto 0,9305, campos inexistentes 2,06% (`full`) / 11,11% (`irrelevant`) / 12,90% (`none`), Generic-JS 0,518;
T1 0,4105, 0,595, 0,9512, 1,58% / 7,58% / 10,97%, 0,632. Base zero-shot: T0 0,076/0,213; T1 0,142/0,316.

## 2. O que falta (e o card declara)

- **Homologacao fora da distribuicao: TBD.** O cego humano foi descartado em 2026-10-22; nao existe nenhuma medida fora do sintetico. O card diz isso em secao propria e nao preenche.
- **Gates oficiais:** KapiLab sem token de GPU (B5), TTFT de CPU reprova sem cache de prefixo (B2), bug do EOS na IDE e nos pacotes ONNX (B1), `gates.yaml` em rascunho, `qualified=false` em todas as medidas de ONNX.
- **PRs da IDE** (1 a 4) nao implementados; Fase 08 nao iniciada. O card nao promete integracao.
- **Dataset:** G3.7 (revisao humana) e G3.9 (licencas) pendentes; por isso o dataset nao e publicado e o card diz "sem revisao humana das amostras".
- **T1 sem controle gold:** o efeito da destilacao no 1,5B nao foi medido (no 0,5B foi pequeno: exec-valid +1,7 p.p., FIM acc -0,85 p.p. n.s.; o professor 7B empata com os alunos).
- **Higiene dos workspaces (fora do meu escopo, nao alterada):** `README.md`/`README_job.md` dos dois workspaces ainda dizem "treino nao executado"; `model.lock.json` tem `scripts_dirty: true`. Corrigir antes de apontar alguem para o repositorio local.
- **Decisoes suas** (lista em `MODEL_CARD_REVIEW.md` §1): nome `esilva/...`, privado/publico, incluir ONNX, dataset card, contato, divulgacao do contrato `editor-context-v1`, manter ou nao o `model-index`.

## 3. Riscos de publicar agora

1. **Reputacao.** Exec-valid 0,58-0,60 e FIM acc 0,40-0,41 soam fortes e sao de dado sintetico gerado por templates; o conjunto satura (o 0,5B empata com o professor 7B). Quem copiar a tabela sem o rotulo pode tomar como qualidade em consultas reais. O card repete o rotulo no topo, na tabela e no resumo pt-BR, mas o risco de citacao fora de contexto permanece.
2. **Uso fora do escopo.** Pessoas podem tentar chat, agente, dialeto mongosh ou executar sugestoes sem revisao. O modelo foi treinado so em Console JavaScript e JSON; comportamento com operacoes destrutivas nao foi treinado nem testado.
3. **Campos inexistentes sem schema visivel:** 11-13% no T0 e 8-11% no T1 (`irrelevant`/`none`); o treino nao resolveu e depende de a IDE enviar nomes de campos.
4. **ONNX:** os pacotes saem com o bug do EOS do GenAI 0.15.2 (token final repetido se o chamador ler `GetSequence(0)[^1]`); o T1 int4 perde 1,4-1,8 p.p. de FIM acc (nao oferecido); latencia de CPU e so proxy e fica acima da meta interna. Publicar ONNX agora e publicar artefatos "candidato nao qualificado".
5. **Licenca e atribuicao:** Apache-2.0 com aviso de modificacao e sem endosso Qwen; replay de 555 pacotes npm (permissivos) listado, codigo nao redistribuido. Conferir o nome "Kapibara Studio" e a divulgacao do formato do contrato com quem e dono da IDE.
6. **Irreversibilidade:** um repositorio publico pode ser clonado/espelhado; apagar depois nao desfaz. Por isso privado primeiro.
7. **Reproducao:** `scripts_dirty: true` no lock e dataset nao publicado; o card nao promete reproducao.

## 4. Recomendacao

**Privado primeiro**, os dois tiers juntos, **so safetensors bf16 + tokenizer + card** (sem ONNX e sem GGUF):

1. Voce revisa os dois `MODEL_CARD_REVIEW.md` e decide os itens da §1 de la.
2. `verify_card_numbers.py`, `pytest scripts/publish/tests`, `stage_hf_package.py --write-manifest` (se o card mudou), `--stage` em diretorio vazio.
3. Com sua aprovacao explicita e seu `HF_TOKEN`: `upload_to_hf.py ... --i-have-user-approval` (cria o repo **privado**).
4. Reabrir o card no Hub e conferir; so entao pensar em publico.

Antes de tornar publico: (a) ao menos uma medida nao sintetica ou aceitar o `TBD` conscientemente (alternativa do resumo: consultas reais do proprio usuario, sem selar); (b) decidir o EOS (IDE ou contorno no `genai_config.json`) se for incluir ONNX; (c) corrigir os README dos workspaces; (d) decidir sobre dataset card e contato. ONNX pode entrar numa segunda etapa, com `onnx/dml-fp16` (T0 e T1) e `onnx/int4` (T0) / `onnx/int8` (T1) apenas, ja descritos no card.
