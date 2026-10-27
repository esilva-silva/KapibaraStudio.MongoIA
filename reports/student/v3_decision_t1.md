# Decisao do retreino v3 - T1

**Veredito: GO** - promover o modelo v3 (merged/sft_v3); liberar o passo seguinte (J21 no T0->T1).

- Campos inexistentes em `full` (homologacao): atual 9.19% -> v3 7.50% (base 7.15%; relatorio v1: 9.19%); meta ideal (<= base + 1 pp): atingida.
- Delta pareado `no_nonexistent_field` (full): {'delta': 0.016935483870967744, 'ci_low': 0.008064516129032258, 'ci_high': 0.025537634408602152, 'p_holm': 0.0011534052466784378, 'p': 0.00016477217809691969}.
- Guarda na homologacao: perda de FIM acc -8.4 pp, de exec Ã· teto -4.82 pp (tolerancia 1.0 pp).
- val v2c (mesmos 2.000 ids): perda de FIM acc 0.05 pp, de exec Ã· teto -1.039 pp; Generic-JS vs zero-shot -20.0 pp.

| criterio | resultado |
|---|---|
| C1_full_fields_lower | ok |
| C2_improvement_confirmed_ci95 | ok |
| C3_hom_guardrail_loss_le_1pp | ok |
| C4_val_and_generic_guard | ok |

## Comparacoes sem ambiguidade de sinal

Delta = novo - referencia; positivo significa ganho nas metricas abaixo.

- hom_fim_acc: 0.3697 -> 0.4537; delta +8.400 pp.
- hom_exec_over_ceiling: 0.8444 -> 0.8926; delta +4.820 pp.
- generic_js: 0.4060 -> 0.6060; delta +20.000 pp.

Os campos legados *_loss_pp usam referencia - novo: valor negativo significa ganho, nao regressao.
