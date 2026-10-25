# Decisao do retreino v3 - T0

**Veredito: GO** - promover o modelo v3 (merged/sft_v3); liberar o passo seguinte (J21 no T0->T1).

- Campos inexistentes em `full` (homologacao): atual 15.35% -> v3 10.81% (base 7.02%; relatorio v1: 15.35%); meta ideal (<= base + 1 pp): nao atingida.
- Delta pareado `no_nonexistent_field` (full): {'delta': 0.04543010752688172, 'ci_low': 0.03413978494623656, 'ci_high': 0.05725806451612903, 'p_holm': 2.68308862218623e-13, 'p': 4.471814370310383e-14}.
- Guarda na homologacao: perda de FIM acc -9.24 pp, de exec ÷ teto -11.74 pp (tolerancia 1.0 pp).
- val v2c (mesmos 2.000 ids): perda de FIM acc -0.2 pp, de exec ÷ teto -1.998 pp; Generic-JS vs zero-shot -19.0 pp.

| criterio | resultado |
|---|---|
| C1_full_fields_lower | ok |
| C2_improvement_confirmed_ci95 | ok |
| C3_hom_guardrail_loss_le_1pp | ok |
| C4_val_and_generic_guard | ok |
