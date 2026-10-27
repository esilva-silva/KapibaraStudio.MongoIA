# KapiCoder-Mongo — resumo executivo (26/10/2026)

T0 e T1 v3 possuem avaliacao sintetica e pacotes ONNX locais. A correcao de EOS pertence a IDE; nao serao geradas variantes eosfix. Qualidade real, latencia nativa em sessao e selecao efetiva de GPU continuam exigindo homologacao propria.

## 1. Estado e fontes

| Frente | Estado comprovado / pendencia | Fonte |
|---|---|---|
| Treino | T0/T1 sft_v3 executados; uma semente | `Repos/*/reports/sft_v3/run_manifest.json` |
| Avaliacao | GO v3 preservado; a meta ideal T0 nao foi atingida | `reports/student/v3_decision_t0.json`, `v3_decision_t1.json` |
| Exportacao | Pacotes v3 locais; evidencia proxy separada da IDE | `reports/export/`, `scripts/export/README.md` |
| Hardware | D1–D5 documentadas; latencias ruidosas precisam nova sessao ociosa | `docs/mapa-compatibilidade-gpu.md`, `reports/hwmap/phase2_measurements.md` |
| IDE PR1 | Aceite de catalogo/paridade existente | IDE: `docs/phases/phase-07b-kapilab/aceite-pr-1.md` |
| IDE PR2 | Contrato local-agent-qwen3-v1 existente | IDE: `docs/architecture/local-agent-qwen3-v1.md`, ADR-072 |
| IDE PR3 | Host/catalogo/replay implementados; sucesso Atlas e paridade externa pendentes | IDE: `docs/phases/phase-07b-kapilab/aceite-pr-3.md` |
| Fase 04 / T4 | Nao iniciar enquanto os aceites externos estiverem pendentes | Fontes PR2/PR3 acima |
| Publicacao | Cards/manifests verificados localmente; nenhum upload nesta rodada | `tmp/corrections-stage-validation.log`, `scripts/publish/verify_card_numbers.py` |

## 2. Resultados v3 [M]

| Tier | FIM anterior → v3 | Exec/teto anterior → v3 | Campos full v3 / base | Meta ideal |
|---|---|---|---|---|
| T0 | 0.3310 → 0.4234 | 0.7443 → 0.8617 | 10.81% / 7.02% | nao atingida |
| T1 | 0.3697 → 0.4537 | 0.8444 → 0.8926 | 7.50% / 7.15% | atingida |

Fontes de cada linha: `reports/datasets/homolog-v3-eval-t0.json` e `homolog-v3-eval-t1.json`; criterios em `reports/student/v3_decision_t{0,1}.json`. Delta = novo - anterior: positivo e ganho. Os campos legados *_loss_pp usam sinal inverso.

| Generic-JS | Base → v3 | Delta [M] |
|---|---|---|
| KapiCoder-Mongo-Nano | 34.0% → 53.0% | +19.0 pp |
| KapiCoder-Mongo-Small | 40.6% → 60.6% | +20.0 pp |

Fontes: `Repos/*/reports/eval/{base-qwen2.5-coder-*-generic-js,sft_v3-generic-js}/summary.json`: mesmos 500 itens e hash de dataset. Nao ha evidencia da regressao de −19/−20 pp mencionada no handoff antigo.

## 3. Limitacoes de independencia

Tudo acima e sintetico. A correcao historica do guard v1/v1 e a falta de decontaminacao J18 ate a reconstrucao v3 foram relatadas no handoff A4. O v3 continua usando gerador Qwen e templates de gramatica compartilhados. Nao ha cego humano nem prova de generalizacao real. Fontes e limites historicos: `reports/student/correcoes-coordenadas-20261026.md`; nao atribuir uma porcentagem de otimismo sem medir.

## 4. Integracao e decisoes

- T0 int4 e o padrao escolhido. EOS sera corrigido somente na IDE; sem eosfix.
- Cache CPU e opt-in e exige homologacao declarada no pacote. A prova inicial de cauda de um token divergiu; a IDE passou a reprocessar no minimo 32 tokens. Prompts curtos fazem prefill completo. Paridade sintetica nao basta para aprovar qualidade/TTFT em sessao.
- Gate CPU exige hit real >=95%, com hit/miss/mistura reportados. Sem essa evidencia o gate continua nao avaliado; nao promover pelo prototipo Python.
- GenAI 0.16/0.17 sao candidatos a avaliacao; selecao efetiva de GPU exige prova por contadores. Nenhuma mudanca de versao e aprovada apenas por aceitar device_id.
- Gates/rounds, segunda semente, chat T1/T2, hardware indisponivel, contato/model-index/nome no card e upload continuam como decisoes posteriores. Nenhum treino/upload nesta rodada.

## 5. Rastreabilidade e proximos passos

Erros T0 por dominio/corte, propostas de dados, limiares e custo da segunda semente: `reports/student/correcoes-coordenadas-20261026.md`. READMEs dos jobs apontam ao sft_v3 e ao hash real da configuracao; o v1 foi preservado como historico.

Os relatorios de decisao foram reemitidos a partir dos agregados existentes, preservando GO. O lock possui --require-clean; nao declarar scripts_dirty=false nem commitar alteracoes de terceiros para fabricar limpeza. scripts_commit deve identificar o commit que contem os scripts, anterior ao commit que grava o lock. Os locks finais foram regenerados com --require-clean a partir do commit revisado 2b0ee093a5cf224a0d9d2d07ab5d0c6349ad2df6, com scripts_dirty=false e hashes do braço v3; o commit de gravação dos locks é posterior a essa referência.


## 6. Contexto historico preservado para os cards

O registro anterior deste resumo descreve o v2c como **110 schemas sintéticos em 14 domínios**; esta rodada nao recontou schemas. Fonte historica: versao Git anterior deste documento; o relatorio `reports/datasets/v2c-validation-report.md` detalha dominios e splits. O teto do validador é 0,6255 no val v2c usado para essas comparacoes; fonte medida: `Repos/KapiCoder-Mongo-Nano/reports/eval/sft_v3/summary.json`, campo reference_ceiling.exec_valid. Nao misturar esse teto com a homologacao v1.
