# Correcoes coordenadas MongoIA / IDE — 26/10/2026

Nenhum treino, download de pesos ou upload foi iniciado. Fontes: agregados existentes; [M] indica avaliacao anterior, nao nova medicao.

## Sinais, criterios e qualidade

`decide_v3.py` calcula perda = anterior - novo. Uma perda negativa e ganho. C3 e Generic-JS nao apresentam as regressoes descritas no handoff. GO preservado.

### T0 [M]

Fonte: `reports/datasets/homolog-v3-eval-t0.json`.

FIM 0.3310 → 0.4234; exec/teto 0.7443 → 0.8617.

| Grupo | n | Campos inexistentes (%) |
|---|---:|---:|
| by_cut_kind/key | 959 | 9.59 |
| by_cut_kind/member | 600 | 6.33 |
| by_cut_kind/multiline | 480 | 22.92 |
| by_cut_kind/operator | 841 | 7.49 |
| by_cut_kind/partial | 600 | 14.83 |
| by_cut_kind/rest | 720 | 6.39 |
| by_cut_kind/stage | 720 | 19.03 |
| by_cut_kind/stmt_start | 269 | 10.41 |
| by_cut_kind/value | 841 | 1.19 |
| by_domain/adtech | 3057 | 10.14 |
| by_domain/telecom | 2973 | 10.19 |

As marginais acima nao sao o cruzamento exclusivo com `full`: nao permitem atribuir causalidade ou contar erros exclusivos. Proposta: estratificar os erros full por corte/dominio via API de homologacao, reforcar supervisao de nomes presentes/ausentes em dados de treino e validar sem reutilizar exemplos hom. Sem novo treino nesta rodada.

### T1 [M]

Fonte: `reports/datasets/homolog-v3-eval-t1.json`.

FIM 0.3697 → 0.4537; exec/teto 0.8444 → 0.8926.

| Grupo | n | Campos inexistentes (%) |
|---|---:|---:|
| by_cut_kind/key | 959 | 6.88 |
| by_cut_kind/member | 600 | 4.00 |
| by_cut_kind/multiline | 480 | 18.96 |
| by_cut_kind/operator | 841 | 5.23 |
| by_cut_kind/partial | 600 | 11.33 |
| by_cut_kind/rest | 720 | 6.81 |
| by_cut_kind/stage | 720 | 13.89 |
| by_cut_kind/stmt_start | 269 | 8.92 |
| by_cut_kind/value | 841 | 1.07 |
| by_domain/adtech | 3057 | 7.72 |
| by_domain/telecom | 2973 | 8.04 |

As marginais acima nao sao o cruzamento exclusivo com `full`: nao permitem atribuir causalidade ou contar erros exclusivos. Proposta: estratificar os erros full por corte/dominio via API de homologacao, reforcar supervisao de nomes presentes/ausentes em dados de treino e validar sem reutilizar exemplos hom. Sem novo treino nesta rodada.

## Limitacoes de independencia

O guard historico usou caminho v1/v1 e desativou check_record/check_text; J18 nao teve decontaminacao n-gram contra homologacao ate a reconstrucao v3. Os numeros v1 podem ser otimistas. O v3 usa gerador LLM da familia Qwen e templates de gramatica compartilhados: nao constitui avaliacao humana independente. Fonte historica: `docs/handoff-codex-correcoes.md` A4; validar a proveniencia historica antes de quantificar impacto. As metricas nao demonstram generalizacao real.

## Integracao existente

PR1 possui aceite em `docs/phases/phase-07b-kapilab/aceite-pr-1.md` da IDE. PR2 possui contrato `docs/architecture/local-agent-qwen3-v1.md` (ADR-072). PR3 possui host real, catalogo e replay; `aceite-pr-3.md` explicita pendencias de sucesso Atlas e paridade externa >=99%. Nao declarar a Fase 04 desbloqueada apenas pela existencia dos contratos.

## Decisoes desta rodada

EOS corrigido somente na IDE; sem variantes eosfix. T0 int4 padrao. Cache CPU opt-in com gate de sessao e hit medido >=95%; reportar hit/miss e mistura, sem promover pelo proxy. GenAI: avaliar atualizacao antes de escolher versao; sem hooks DXGI nem escrita no registro.

## Gates propostos [E] — nao congelados

- T1.blind_v2_vs_base: margem proposta 0 pp, IC95 pareado; continua unevaluated porque nao ha cego humano.
- T3.quality.parity_w4a16: booleano 1 somente se todos os gates common.parity.w4a16 aplicaveis passarem.
- T3.value.ttft_p95_vs_cpu_arm64: razao p95 NPU/CPU <1, com IC95 do ganho >0, mesmo aparelho/modelo.
- T3.selection.1_5B_vs_0_5B: booleano 1 se a regra de selecao existente for integralmente atendida; nao substituir os gates componentes.
- common.parity.tokenizer_prompt_greedy: booleano 1 exige prompts/ids identicos e concordancia greedy >=99,5% no mesmo pacote, opcoes e parada.
- baseline base_v1_0.5b: criar medicao na suite v2 com checkpoint/revisao identificados e mesmos ids; nao substituir por zero-shot Qwen nem inventar APT/exec_match.

## Segunda semente T0 — apenas plano [E]

Repetir a configuracao v3 a partir da mesma base, corpus e hashes, seed 20261019, novo run/output sem sobrescrever sft_v3. Estimativa documentada na config: 4,5–5,8 h de treino; reservar 8 h incluindo validacao, sem confundir teto com medida. Fonte: `Repos/KapiCoder-Mongo-Nano/configs/sft_v3.yaml`. Exige aprovacao antes de criar/enfileirar o job.

## Pendencias materiais

Locks nao podem declarar scripts_dirty=false enquanto houver alteracoes locais, inclusive alteracoes anteriores de publicacao. Nao commitar trabalho alheio para fabricar arvore limpa. Regenerar com --require-clean apos commits revisados; scripts_commit aponta ao commit anterior que contem os scripts. Staging/cards devem ser validados sem sobrescrever as alteracoes de publicacao ja existentes.


## Validacao cruzada de runtime [M]

O comparador Python de geracao esta em scripts/export/generation_reference.py. GenAI 0.15.2 do Python e da IDE tinha o mesmo hash, mas a DLL ORT carregada diferia (Python 1.30). A comparacao controlada usa --ort-library e registra o hash verificado pelo teste C#.

A cauda de um token mostrou divergencia sintetica. A politica atual reprocessa pelo menos 32 tokens (prompts menores fazem prefill completo). T0 passou na prova estrita de oito prompts, incluindo prompts longos com reuso real; a suite KapiLab Cpu passou 713 testes com essa prova ativada. Fontes: IDE artifacts/f7b/test-results/Cpu/kapilab-Cpu.trx e reports/export/generation-reference-t0.json. Nao comprova os 500 prompts de qualidade, FIM/exec-valid, RAM ou gate de TTFT ocioso. Nenhum pacote ganhou metadata de homologacao.

Cards Nano/Small e manifests locais conferiram; fonte do dry-run: tmp/corrections-stage-validation.log. Nenhum upload. Locks finais continuam pendentes de arvore limpa real.


T1 int8 tambem passou nos oito prompts com e sem cache: IDE tools/KapiLab.Tests/TestResults/prefix-real-t1.trx. GenAI 0.16.0 e 0.17.0 foram executados em probe CPU isolado: ambos cache=fresh 8/8, mas somente 5/8 saidas iguais a 0.15.2. Recomenda-se manter 0.15.2 ate medir qualidade e adaptador efetivo. Nenhuma versao foi adotada no produto.

A suite Python ampliada identificou um teste historico que exigia ausencia dos jobs v3 no state.json local, apesar dos treinos ja concluidos. O teste foi corrigido para verificar o contrato estatico do DAG e requires_files/GO, preservando todas essas assercoes e sem modificar state.json. Nao e evidencia de novo treino.

Atualizacao da prova controlada: 500 prompts preparados de cada tier passaram em Python x IDE sem cache e nas duas repeticoes com cache, com IDs exatamente iguais (100% nos conjuntos usados). Referencias generation-reference-val-t0.json e generation-reference-val-t1.json; evidencias IDE tools/KapiLab.Tests/TestResults/prefix-val500-t{0,1}.trx. Trata-se de paridade de geracao sintetica; nao e novo ensaio de task success, FIM, exec-valid, avaliacao humana ou generalizacao real. TTFT de hit/miss/misturas e RAM de quatro sessoes continuam pendentes; cache desativado no produto.

A suite Python final scripts/student/tests scripts/hwmap/tests scripts/export/tests scripts/publish/tests passou com 395 testes e 2 ignorados (tmp/corrections-python-tests-final.log). Gates rejeitam metricas nao finitas/ausentes; o detector exige prova explicita do adaptador e baseline CPU valido. Nenhuma medicao de latencia GPU/NPU foi realizada.
