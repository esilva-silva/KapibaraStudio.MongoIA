# Correção da calibração FIM na RX 7800 XT

Os resultados anteriores de QNN permanecem históricos e não homologam a política corrigida. A IDE e seu padrão Nano int4 não foram alterados nesta rodada.

A política `fim-exact-tail-v1` exige nova quantização a partir dos snapshots locais originais e nova calibração de ativação:

- GPTQ: preservar os 61.273 tokens ordenados (61.017 dos prompts + 256 separadores), em 29 blocos de 2.048 e uma cauda de 1.881, sem padding. Todos os 256 IDs devem estar integralmente cobertos.
- Ativação: os 256 prompts FIM originais, com `add_special_tokens=false`, `pad_to_max_len=false`, batch 1 e comprimentos de 8 a 549.
- A projeção JSON descarta apenas metadados heterogêneos incompatíveis com Arrow. `id`, `prompt`, `prompt_text`, `prompt_ids`, ordem e hash do fluxo são preservados.
- O adaptador aceita parâmetros compartilhados do Olive, como `task`, e registra apenas seus nomes. Esses parâmetros não alteram o fluxo fixado e validado por hash.

O preflight antigo com `get_calibration_dataset(None, ...)` não reproduziu a injeção de `task` do workflow real. A tentativa Nano v2 falhou antes da quantização por esse motivo; o relatório da falha foi preservado. O preflight seguinte precisa resolver o RunConfig real e usar o descriptor HfModel, além do DataConfig e da preparação instalada do GPTQ.

O teste real de ativação validou 256 itens pelo reader e pela guarda e executou duas inferências CPU no grafo dinâmico legado (comprimentos mínimo e máximo), com saídas finitas. Isso demonstra o contrato de entrada; a qualidade dos novos pesos e a compilação/inferência HTP continuam pendentes.

Nenhum novo treino, consulta de documentos MongoDB ou medição de latência ocorreu nesta correção. Pesos e dados de calibração não pertencem aos commits de relatórios/scripts.
