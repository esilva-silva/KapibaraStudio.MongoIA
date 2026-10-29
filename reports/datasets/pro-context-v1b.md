# Relatório — lote sintético contextual Pro, 41 mil exemplos

Gerado em 2026-10-28. Código e comandos de reprodução: `scripts/data/pro/README.md`. Nenhum treinamento ou publicação foi iniciado.

| Lote | Treino | Validação | Teste | Total |
|---|---:|---:|---:|---:|
| Contextual original (FIM + chat) | 24.000 | 3.000 | 3.000 | 30.000 |
| FIM no meio de script | 800 | 100 | 100 | 1.000 |
| FIM no meio de script, ampliado | 8.000 | 1.000 | 1.000 | 10.000 |
| **Total** | **32.800** | **4.100** | **4.100** | **41.000** |

Total por modalidade: 33.500 FIM e 7.500 chat. Os 11.000 FIM de meio de script cobrem 20 famílias; as três famílias de maior colisão são exclusivas de avaliação no lote ampliado. Os demais padrões se repetem entre divisões: esta avaliação de novos schemas e valores não substitui tarefas independentes e cegas. São dados sintéticos, majoritariamente parametrizados, com três schemas distintos; os 41 mil não representam 41 mil intenções independentes.

## Verificações e evidência

- `audit.py`: 41.000 IDs distintos, zero duplicatas exatas de entrada/alvo dentro de cada split, hashes dos manifestos corretos, schemas/grupos disjuntos entre train/val/test e 33.500 prompts FIM iguais ao resultado do renderer existente. Comprimentos máximos: prefixo 544, sufixo 109, middle 102 caracteres; **não** são contagens de tokens.
- `validate.js`: 30.000 scripts aceitos pela sintaxe JavaScript, métodos e argumentos verificados em mocks. `validate_mid.js`: 11.000 scripts reconstruídos `prefix + middle + suffix` aceitos, com 20 famílias e chamadas comparadas a referências independentes.
- MongoDB no Ubuntu/WSL: serviço ativo e ping OK. No banco novo `kapi_fx_pro_20261028a93c7212_{train,val,test}`, 30.000 consultas originais executadas sobre 10.000 documentos sintéticos, zero divergências. Para os 1.000 mid-script finais, os planos capturados do script reconstruído foram executados no banco novo `kapi_fx_pro_20261028a93c7215_mid`; para os 10.000 finais, no banco novo `kapi_fx_pro_20261028a93c7216_mid`; zero divergências em cada. As escritas foram limitadas à coleção de fixture no banco novo. Os verificadores compararam filtros, projeções, contagens, ordenação, agregações e efeitos das escritas; não executaram diretamente o script no interpretador da IDE.
- **Correção posterior:** a auditoria anterior não reconhecia chats sem `snippet_id`, portanto o resultado anterior de zero colisões era inválido para chat. Após corrigir `record_kind`, a auditoria detectou 6.000 colisões por 13-grama no contexto dos chats de treino. O job v1 foi pausado no passo 26 e seu checkpoint foi preservado; ele não deve servir de candidato limpo. O lote revisado `data/generated/pro-chat-context-v2/chat-train.jsonl` mantém os 6.000 exemplos e reestrutura seu contexto. A auditoria corrigida sobre os 32.800 exemplos de treino, val/test, avaliação histórica e legados removeu zero exemplos; hash `49a171dcb10ae1374f413fe42bc140abc3b09dbc89d74d5e9a501efac83e6f77`. O blind continua **not_sealed**. A ausência de colisão pela regra não elimina semelhanças semânticas entre templates.
- `test_chat_context_revision.py` comparou todos os 6.000 chats antigos e revisados: instruções, consultas, split e demais campos de contexto foram preservados, e os IDs e parâmetros estão presentes no novo contexto. Junto aos testes de reconhecimento de chat na descontaminação, foram 3 testes aprovados.
- `prepare_teacher.py` gerou quatro shards e listas de IDs em `data/generated/pro-teacher-input-v5`, usando o chat revisado e acrescentando `dialect: console` aos FIM para atender ao gerador `val_gen`; `verify_teacher_staging.py` confirmou contagens e isolamento de IDs, grupos e schemas.
- `validate_tokenizer.py` verificou os 41.000 exemplos com tokenizer SHA-256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`, igual ao lock, localizado numa pasta temporária do projeto da IDE. Sem violações: máximo FIM de 315 tokens de prompt e 48 de alvo, frente aos limites de 2.048 e 96; máximo de 350 tokens em prompt+alvo+EOS no chat frente a 4.096. O renderizador real do professor processou 85 amostras estratificadas de train/val, verificando posição da loss, EOS e `dialect` nos registros de geração. Isso não constrói o corpus de treino nem comprova paridade com a IDE.
- Três testes existentes do contrato passaram: golden congelado da IDE, cabeçalho/quebras de linha e janelas UTF-16. Isso verifica o renderer compartilhado, não uma execução de todos os registros no código da IDE.
- A suíte antiga `test_generators.py -x -q` teve 7 testes aprovados antes de falhar em `test_zero_writes_executed`; foi reproduzido o erro `ModuleNotFoundError: No module named 'bson'` no Python do Windows, que não tem o `pymongo` previsto em `env/requirements.txt`. Não se afirmou aprovação da suíte.

Os bancos isolados de execução foram preservados para inspeção. Além dos citados acima, tentativas anteriores criaram `kapi_fx_pro_20261028a93c7210_train` e `kapi_fx_pro_20261028a93c7213_mid` e `kapi_fx_pro_20261028a93c7214_mid`; não foram apagados.

## Limitações antes do treinamento

- O blind selado, os corpora antigos validados e os pesos não estavam presentes neste checkout. A auditoria não prova ausência de contaminação contra fontes indisponíveis; faltam paridade do lote com a IDE e execução do pipeline completo. O tokenizer fixado foi encontrado fora deste checkout e usado em leitura.
- Ainda faltam integração ao corpus tokenizado e aos treinadores dos students. Consultas e resultados sintéticos exercitam as intenções especificadas, mas não comprovam cobertura de aplicações reais, robustez geral ou compatibilidade com dados reais. Recomenda-se revisão humana e um conjunto de avaliação independente.
- A próxima fase de preparação está registrada em `docs/fase-pro-retreinamento-contextual.md`: Professor primeiro, seguido de Nano e Small, com bloqueios e gates explícitos antes de iniciar treino.
