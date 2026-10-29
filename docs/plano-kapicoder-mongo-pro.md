# Plano de execução — KapiCoder Mongo Pro 7B

Atualização: 28/10/2026. Objetivo acordado: modelo acima do Small, independente do agente, priorizando autocomplete MongoDB e também oferecendo chat sobre código. Candidato inicial: KapiTeacher-Mongo-7B, adaptado de Qwen2.5-Coder-7B Base. A promoção depende de resultados, não apenas do tamanho.

## 1. Primeira verificação já realizada

Os relatórios existentes `Repos/KapiTeacher-Mongo-7B/reports/eval/sft_v2d-final/report.md` e `Repos/KapiCoder-Mongo-Small/reports/eval/sft_v3/report.md` descrevem a mesma amostra sintética de 2.000 itens v2c, mesmo contrato e configuração de geração. No limite de 96 tokens, ambos registram FIM normalizado de 41,0%; execução válida é 57,6% para o professor e 60,2% para o Small; campos inexistentes, 3,9% e 2,1%, respectivamente. O professor foi avaliado em NF4 e o Small em BF16. Esses números são um alerta, não uma decisão definitiva: são dados sintéticos, não constituem avaliação independente real nem medem o pacote ONNX, e não estabelecem significância estatística.

## 2. Etapas e critérios

1. **Rastreabilidade** — identificar e verificar hashes do checkpoint final v2d e base 7B, dados, dependências e contratos; conferir se pesos e fixtures estão disponíveis. Preservar os artefatos existentes.
2. **Autocomplete primeiro** — comparar Small v3, professor 7B e base 7B em exemplos idênticos, com contrato de editor e limites de saída equivalentes; medir FIM, validade MongoDB e de execução, campos inexistentes e resultados por tipo de corte. Calcular diferenças pareadas e intervalos de confiança. Complementar com conjunto real ou revisado por humanos, separado dos dados de treino e checado contra contaminação. Não usar o conjunto histórico como prova independente.
3. **Chat sobre código** — comparar respostas dos mesmos candidatos em tarefas de explicação, criação e correção de consultas e código MongoDB; avaliar correção e aderência ao contexto com exemplos separados e revisão humana. Manter autocomplete como critério principal.
4. **Decisão** — promover o 7B apenas se demonstrar ganho útil em autocomplete sobre o Small, sem regressões relevantes em validade e campos, e com chat aceitável. Definir limiares antes da avaliação final; caso falhe, analisar erros e propor ajustes de dados/treino antes de uma nova rodada. Não presumir ganho pelo número de parâmetros.
5. **Exportação e integração** — após a avaliação, produzir e verificar pacote ONNX adequado, comparar sua qualidade com o checkpoint e testar execução na IDE. Medir latência, memória e experiência de uso em hardware alvo do Pro; a homologação informada para Nano/Small não cobre automaticamente o Pro.
6. **Entrega** — documentar resultados, limitações, licença e rastreabilidade; publicar somente após aprovação dos gates e decisão de entrega.

## 3. Situação e próximos passos

- [x] Escopo definido: autocomplete prioritário, chat secundário, candidato 7B.
- [x] Leitura inicial dos relatórios comparáveis existentes; resultado acima registrado.
- [ ] Auditar disponibilidade e hashes dos artefatos e preparar comparação reproduzível.
- [ ] Definir conjunto independente e limiares antes de executar a avaliação final.
- [ ] Executar avaliação pareada e avaliação de chat; registrar resultados.
- [ ] Decidir promoção ou necessidade de ajuste; depois exportar e homologar Pro.

Fontes de contexto: `docs/resumo-executivo.md`, `docs/fase-01-selecao-modelos-base.md`, `docs/fase-02-professor.md` e `docs/fase-06-avaliacao-benchmarks.md`. O README histórico do job do professor descreve um estágio anterior e não deve ser tratado sozinho como estado atual.
