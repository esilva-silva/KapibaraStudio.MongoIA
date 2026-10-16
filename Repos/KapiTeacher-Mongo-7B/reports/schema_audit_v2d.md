# Auditoria de schemas para a 2a iteracao do professor (saas, iot, gaming, logistics)

Gerada por `scripts/teacher/audit_schemas.py` a partir de `Repos/KapiTeacher-Mongo-7B/reports/eval/sft_v2c-final/predictions.jsonl` (2000 predicoes do adapter `sft_v2c`) e de `data/schemas/<dominio>/*.json`. **Somente relatorio**: nenhum schema ou fixture foi alterado.

Como ler: os schemas de `eval`/`test` do val v2c sao os que geram as predicoes; os de treino (mesmo dominio, estilos diferentes) sao o que o modelo viu. Taxas por schema tem n pequeno (~15-30 predicoes): indicio, nao medida.

## saas

- Predicoes no val: 165; campo inexistente: **8 (4.8%)** (geral 4.0%). Schemas do dominio: atlas_saas (train, snake), delta_saas (eval, snake), kestrel_saas (train, camel), quasar_saas (eval, camel), saffron_saas (train, camel), tenantly (train, snake), willow_saas (train, pascal).

| tipo do nome inventado | n |
|---|---:|
| filho_inventado_de_objeto_existente | 3 |
| invencao | 2 |
| campo_de_colecao_irma | 1 |
| vocabulario_de_outro_schema_do_dominio | 1 |
| quase_igual (edicao <= 2) | 1 |

| schema | colecao | corte | nome inventado | tipo |
|---|---|---|---|---|
| saas/delta_saas | ChavesApi | partial | `Info.Extra.Label` | filho_inventado_de_objeto_existente |
| saas/delta_saas | Webhooks | stmt_start | `History.Change.Field` | vocabulario_de_outro_schema_do_dominio |
| saas/delta_saas | faturas | multiline | `extra.window` | filho_inventado_de_objeto_existente |
| saas/delta_saas | faturas | multiline | `history.change` | campo_de_colecao_irma |
| saas/delta_saas | registros uso | key | `dispositivo` | invencao |
| saas/delta_saas | usuarios | value | `uuid` | invencao |
| saas/quasar_saas | chavesApi | stage | `key` | quase_igual (edicao <= 2) |
| saas/quasar_saas | flagsRecurso | key | `rules.field` | filho_inventado_de_objeto_existente |

Falhas por schema: saas/delta_saas (6), saas/quasar_saas (2). Colecoes mais afetadas: saas/delta_saas.faturas (2), saas/delta_saas.ChavesApi (1), saas/delta_saas.registros uso (1), saas/delta_saas.usuarios (1), saas/delta_saas.Webhooks (1), saas/quasar_saas.chavesApi (1).

### saas/atlas_saas (snake; 7 colecoes, 90 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (3): `at` repetido em audit_events: at, attrs.at; `amount` repetido em invoices: amount, line_items.amount; audit_events: `action` ~ `actor`
- **Contenedores genericos (inner keys imprevisiveis)** (7): users.prefs -> {notify, theme, tz}; webhooks.attrs -> {enabled, label, score, window}; invoices.tags -> {}; audit_events.attrs -> {at, label, settings, size}; audit_events.attrs.settings -> {enabled, total}; audit_events.tags -> {}; tenants.settings -> {limits, locale, sso}
- **Aninhamento profundo (>= 4 niveis)** (2): tenants.settings.limits.burst.window_s; tenants.settings.limits.burst.cap

### saas/delta_saas (snake; 8 colecoes, 140 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (18): `configuracoes` repetido em inquilinos: configuracoes, attrs.audit.configuracoes; `habilitado` repetido em faturas: extra.habilitado, extra.configuracoes.habilitado; `label` repetido em inquilinos: attrs.label, attrs.audit.label; `tamanho` repetido em inquilinos: attrs.tamanho, attrs.audit.tamanho; `em` repetido em inquilinos: attrs.audit.configuracoes.em, history.em; `em` repetido em flags_recurso: attrs.configuracoes.em, history.em; `valor` repetido em faturas: valor, linha_itens.valor; `Contagem` repetido em ChavesApi: Info.Contagem, Info.Extra.Contagem ...
- **Contenedores genericos (inner keys imprevisiveis)** (8): inquilinos.attrs -> {audit, habilitado, label, tamanho}; inquilinos.attrs.audit -> {configuracoes, label, tamanho}; flags_recurso.attrs -> {configuracoes, label, mode, tamanho}; ChavesApi.Info -> {Contagem, Extra, Pontuacao}; ChavesApi.Info.Extra -> {Contagem, Extra, Tamanho}; ChavesApi.Info.Extra.Extra -> {Em, Mode}; faturas.extra -> {configuracoes, habilitado, mode, pontuacao}; usuarios.prefs -> {notify, theme, tz}
- **Aninhamento profundo (>= 4 niveis)** (7): inquilinos.configuracoes.limits.burst.window_s; inquilinos.configuracoes.limits.burst.cap; inquilinos.attrs.audit.configuracoes.pontuacao; inquilinos.attrs.audit.configuracoes.contagem; inquilinos.attrs.audit.configuracoes.em; ChavesApi.Info.Extra.Extra.Em ...
- **Campos opcionais raros (presenca < 0,35)** (4): inquilinos.attrs.tamanho (p=0.3); registros uso.cost (p=0.3); ChavesApi.Info.Extra.Extra.Mode (p=0.3); chamados.csat (p=0.3)

### saas/kestrel_saas (camel; 6 colecoes, 113 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (8): `score` repetido em usage records: info.score, info.audit.attrs.score; `total` repetido em usage records: info.total, info.audit.attrs.total; `enabled` repetido em invoices: attrs.enabled, attrs.settings.enabled; `at` repetido em invoices: attrs.at, attrs.settings.at, attrs.settings.extra.at, history.at; `at` repetido em supportTickets: messages.at, meta.at, meta.quota.at; `label` repetido em tenants: info.label, info.quota.label; `amount` repetido em invoices: amount, lineItems.amount; usage records: `cost` ~ `count`
- **Contenedores genericos (inner keys imprevisiveis)** (12): usage records.info -> {audit, score, total}; usage records.info.audit -> {at, attrs, count, enabled}; usage records.info.audit.attrs -> {score, total}; usage records.tags -> {}; tenants.settings -> {limits, locale, sso}; tenants.info -> {enabled, label, quota}; tenants.tags -> {}; invoices.attrs -> {at, enabled, settings} ...
- **Aninhamento profundo (>= 4 niveis)** (6): usage records.info.audit.attrs.score; usage records.info.audit.attrs.total; tenants.settings.limits.burst.windowS; tenants.settings.limits.burst.cap; invoices.attrs.settings.extra.at; invoices.attrs.settings.extra.total
- **Campos opcionais raros (presenca < 0,35)** (4): usage records.info (p=0.3); tenants.info (p=0.3); tenants.info.quota.label (p=0.3); invoices.history.change.to (p=0.3)

### saas/quasar_saas (camel; 5 colecoes, 61 campos, aninhamento max 3)

- **Nomes ambiguos/similares** (1): `em` repetido em inquilinos: extra.em, extra.window.em
- **Contenedores genericos (inner keys imprevisiveis)** (1): inquilinos.extra -> {em, label, pontuacao, window}
- **Campos opcionais raros (presenca < 0,35)** (2): chavesApi.history.change.from (p=0.3); inquilinos.extra.pontuacao (p=0.3)

### saas/saffron_saas (camel; 10 colecoes, 155 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (10): `at` repetido em auditEvents: at, history.at; `at` repetido em invoices: meta.at, history.at; `v` repetido em auditEvents: details.before.v, details.after.v; `score` repetido em usageRecords: meta.score, meta.limits.score; `amount` repetido em invoices: amount, lineItems.amount; variantes de estilo no schema: ['_id', 'id']; variantes de estilo no schema: ['TenantId', 'tenantId']; auditEvents: `action` ~ `actor` ...
- **Contenedores genericos (inner keys imprevisiveis)** (10): tenants.settings -> {limits, locale, sso}; tenants.tags -> {}; feature-flags.tags -> {}; supportTickets.tags -> {}; Users.Prefs -> {Notify, Theme, Tz}; auditEvents.details -> {after, before}; usageRecords.meta -> {count, limits, score}; usageRecords.tags -> {} ...
- **Aninhamento profundo (>= 4 niveis)** (2): tenants.settings.limits.burst.windowS; tenants.settings.limits.burst.cap
- **Campos opcionais raros (presenca < 0,35)** (4): subscriptions.renewsAt (p=0.3); api-keys.lastUsed (p=0.3); auditEvents.history (p=0.3); webhooks.extra.quota.count (p=0.3)

### saas/tenantly (snake; 5 colecoes, 57 campos, aninhamento max 3)

- **Nomes ambiguos/similares** (1): `email` repetido em users: email, prefs.notifications.email
- **Contenedores genericos (inner keys imprevisiveis)** (3): tenants.settings -> {limits, locale, sso}; users.prefs -> {notifications, theme}; audit_log.meta -> {changes, request_id, user_agent}
- **Campos opcionais raros (presenca < 0,35)** (1): tenants.trial_ends_at (p=0.3)

### saas/willow_saas (pascal; 8 colecoes, 126 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (11): `At` repetido em SupportTickets: Messages.At, History.At; `At` repetido em AuditEvents: Attrs.At, Attrs.Quota.Quota.At, History.At; `From` repetido em SupportTickets: Messages.From, History.Change.From; `Amount` repetido em Invoices: Amount, LineItems.Amount; `V` repetido em AuditEvents: Details.Before.V, Details.After.V; `Total` repetido em AuditEvents: Attrs.Total, Attrs.Quota.Total, Attrs.Quota.Quota.Total; `Label` repetido em AuditEvents: Attrs.Label, Attrs.Quota.Label; `Quota` repetido em AuditEvents: Attrs.Quota, Attrs.Quota.Quota ...
- **Contenedores genericos (inner keys imprevisiveis)** (6): SupportTickets.Tags -> {}; Users.Prefs -> {Notify, Theme, Tz}; Subscriptions.Tags -> {}; AuditEvents.Details -> {After, Before}; AuditEvents.Attrs -> {At, Label, Quota, Total}; Tenants.Settings -> {Limits, Locale, Sso}
- **Aninhamento profundo (>= 4 niveis)** (4): AuditEvents.Attrs.Quota.Quota.Total; AuditEvents.Attrs.Quota.Quota.At; Tenants.Settings.Limits.Burst.WindowS; Tenants.Settings.Limits.Burst.Cap
- **Campos opcionais raros (presenca < 0,35)** (7): SupportTickets.History.Change.From (p=0.3); Subscriptions.Addons (p=0.3); Subscriptions.History.Change.To (p=0.3); AuditEvents.Details (p=0.3); AuditEvents.Attrs (p=0.3); AuditEvents.History (p=0.3); Tenants.Settings (p=0.3)

## iot

- Predicoes no val: 168; campo inexistente: **12 (7.1%)** (geral 4.0%). Schemas do dominio: aurora_iot (train, camel), fjord_iot (train, snake), harbor_iot (train, snake), juniper_iot (eval, pascal), onyx_iot (train, camel), rivet_iot (train, camel), saffron_iot (eval, camel), sensorgrid (train, snake).

| tipo do nome inventado | n |
|---|---:|
| filho_inventado_de_objeto_existente | 4 |
| campo_de_colecao_irma | 3 |
| invencao | 2 |
| aninhamento_errado (folha existe em outro caminho) | 1 |
| quase_igual (edicao <= 2) | 1 |
| variante_de_estilo (caixa/underscore) | 1 |

| schema | colecao | corte | nome inventado | tipo |
|---|---|---|---|---|
| iot/juniper_iot | Alerts | key | `Meta.Settings` | filho_inventado_de_objeto_existente |
| iot/juniper_iot | Devices | multiline | `Info.Quota` | filho_inventado_de_objeto_existente |
| iot/juniper_iot | Devices | multiline | `Info.Quota.At` | aninhamento_errado (folha existe em outro caminho) |
| iot/juniper_iot | Maintenance | multiline | `PerfomedAt` | quase_igual (edicao <= 2) |
| iot/juniper_iot | Sensors | multiline | `Extra.At` | campo_de_colecao_irma |
| iot/juniper_iot | Sites | multiline | `Geo.Country` | filho_inventado_de_objeto_existente |
| iot/saffron_iot | comandos | multiline | `params.quota` | filho_inventado_de_objeto_existente |
| iot/saffron_iot | comandos | stage | `em` | campo_de_colecao_irma |
| iot/saffron_iot | comandos | stage | `params.quota.habilitado` | campo_de_colecao_irma |
| iot/saffron_iot | gateways | operator | `exists` | invencao |
| iot/saffron_iot | gateways | stage | `firstMac` | invencao |
| iot/saffron_iot | sensores | partial | `calibrated_em` | variante_de_estilo (caixa/underscore) |

Falhas por schema: iot/juniper_iot (6), iot/saffron_iot (6). Colecoes mais afetadas: iot/saffron_iot.comandos (3), iot/juniper_iot.Devices (2), iot/saffron_iot.gateways (2), iot/juniper_iot.Alerts (1), iot/juniper_iot.Maintenance (1), iot/juniper_iot.Sensors (1).

### iot/aurora_iot (camel; 8 colecoes, 137 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (18): `at` repetido em readings: at, history.at; `at` repetido em maintenance: extra.at, extra.attrs.at; `at` repetido em devices: info.at, info.quota.at, history.at; `enabled` repetido em maintenance: extra.enabled, extra.attrs.enabled; `mode` repetido em gateways: attrs.mode, attrs.stats.mode, attrs.stats.extra.mode; `size` repetido em gateways: attrs.size, attrs.stats.size; variantes de estilo no schema: ['Name', 'name']; variantes de estilo no schema: ['History', 'history'] ...
- **Contenedores genericos (inner keys imprevisiveis)** (9): Sites.Tags -> {}; readings.flags -> {}; maintenance.extra -> {at, attrs, enabled, mode}; maintenance.extra.attrs -> {at, enabled, label}; gateways.attrs -> {mode, score, size, stats}; gateways.attrs.stats -> {enabled, extra, mode, size}; gateways.attrs.stats.extra -> {count, mode}; devices.labels -> {env, owner, zone} ...
- **Aninhamento profundo (>= 4 niveis)** (2): gateways.attrs.stats.extra.mode; gateways.attrs.stats.extra.count
- **Campos opcionais raros (presenca < 0,35)** (3): readings.history.change.to (p=0.3); gateways.attrs.stats.enabled (p=0.3); devices.info.quota.label (p=0.3)

### iot/fjord_iot (snake; 10 colecoes, 140 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (12): `at` repetido em telemetry_batches: attrs.at, attrs.window.at; `count` repetido em alerts: extra.count, extra.settings.count; `total` repetido em alerts: extra.total, extra.settings.total; `total` repetido em readings: meta.total, meta.stats.total; `size` repetido em telemetry_batches: attrs.size, attrs.window.size; `enabled` repetido em sensors: extra.enabled, extra.stats.quota.enabled; `label` repetido em sensors: extra.stats.label, extra.stats.quota.label; variantes de estilo no schema: ['siteId', 'site_id'] ...
- **Contenedores genericos (inner keys imprevisiveis)** (11): devices.labels -> {env, owner, zone}; alerts.extra -> {count, settings, size, total}; alerts.extra.settings -> {count, total}; alerts.tags -> {}; readings.flags -> {}; readings.meta -> {count, enabled, stats, total}; readings.meta.stats -> {label, score, total}; readings.tags -> {} ...
- **Aninhamento profundo (>= 4 niveis)** (3): sensors.extra.stats.quota.label; sensors.extra.stats.quota.mode; sensors.extra.stats.quota.enabled
- **Campos opcionais raros (presenca < 0,35)** (3): devices.labels (p=0.3); readings.meta.stats.label (p=0.3); sensors.extra.stats.quota (p=0.3)

### iot/harbor_iot (snake; 9 colecoes, 148 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (24): `tamanho` repetido em alertas: extra.tamanho, extra.stats.tamanho; `tamanho` repetido em firmwares: info.tamanho, info.stats.window.tamanho; `total` repetido em manutencoes: extra.limits.total, extra.limits.configuracoes.total; `mode` repetido em firmwares: info.mode, info.stats.window.mode; `pontuacao` repetido em manutencoes: extra.pontuacao, extra.limits.pontuacao, extra.limits.configuracoes.pontuacao; `pontuacao` repetido em leituras: meta.pontuacao, meta.window.pontuacao; `em` repetido em leituras: em, meta.em; `em` repetido em alertas: ack.em, extra.em, extra.stats.em ...
- **Contenedores genericos (inner keys imprevisiveis)** (10): lotes_telemetria.extra -> {configuracoes, habilitado, label, tamanho}; manutencoes.extra -> {em, label, limits, pontuacao}; leituras.flags -> {}; leituras.meta -> {em, pontuacao, total, window}; alertas.extra -> {em, label, stats, tamanho}; alertas.extra.stats -> {contagem, em, tamanho}; firmwares.info -> {em, mode, stats, tamanho}; firmwares.info.stats -> {habilitado, label, pontuacao, window} ...
- **Aninhamento profundo (>= 4 niveis)** (6): manutencoes.extra.limits.configuracoes.pontuacao; manutencoes.extra.limits.configuracoes.total; manutencoes.extra.limits.configuracoes.habilitado; firmwares.info.stats.window.tamanho; firmwares.info.stats.window.mode; firmwares.info.stats.window.em
- **Campos opcionais raros (presenca < 0,35)** (2): lotes_telemetria.extra.tamanho (p=0.3); lotes_telemetria.extra.configuracoes.mode (p=0.3)

### iot/juniper_iot (pascal; 7 colecoes, 99 campos, aninhamento max 3)

- **Nomes ambiguos/similares** (11): `By` repetido em Alerts: Ack.By, History.By; `At` repetido em Alerts: Ack.At, History.At; `From` repetido em Alerts: Context.Window.From, History.Change.From; `To` repetido em Alerts: Context.Window.To, History.Change.To; `Score` repetido em Alerts: Meta.Score, Meta.Stats.Score; `Label` repetido em Maintenance: Meta.Label, Meta.Limits.Label; `Enabled` repetido em Maintenance: Meta.Enabled, Meta.Limits.Enabled; Maintenance: `Code` ~ `Cost` ...
- **Contenedores genericos (inner keys imprevisiveis)** (7): Alerts.Meta -> {Score, Stats, Total}; Alerts.Meta.Stats -> {Enabled, Label, Score}; Maintenance.Meta -> {Enabled, Label, Limits}; Maintenance.Tags -> {}; Devices.Labels -> {Env, Owner, Zone}; Devices.Info -> {Enabled, Extra, Mode}; Devices.Info.Extra -> {At, Score}
- **Campos opcionais raros (presenca < 0,35)** (3): Sensors.CalibratedAt (p=0.3); Alerts.History (p=0.3); Sites.LegacyId (p=0.3)

### iot/onyx_iot (camel; 8 colecoes, 128 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (19): `count` repetido em devices: info.count, info.window.count; `by` repetido em alerts: ack.by, history.by; `at` repetido em alerts: ack.at, history.at; `from` repetido em alerts: context.window.from, history.change.from; `to` repetido em alerts: context.window.to, history.change.to; `enabled` repetido em devices: info.enabled, info.window.enabled; `score` repetido em sensors: attrs.score, attrs.window.score, attrs.window.limits.score; `At` repetido em Gateways: Meta.At, Meta.Quota.Extra.At ...
- **Contenedores genericos (inner keys imprevisiveis)** (8): alerts.tags -> {}; devices.labels -> {env, owner, zone}; devices.info -> {count, enabled, window}; devices.info.window.settings -> {mode, size}; sensors.attrs -> {score, total, window}; Gateways.Meta -> {At, Mode, Quota}; Gateways.Meta.Quota.Extra -> {At, Enabled}; readings.flags -> {}
- **Aninhamento profundo (>= 4 niveis)** (7): devices.info.window.settings.size; devices.info.window.settings.mode; sensors.attrs.window.limits.label; sensors.attrs.window.limits.size; sensors.attrs.window.limits.score; Gateways.Meta.Quota.Extra.Enabled ...
- **Campos opcionais raros (presenca < 0,35)** (6): alerts.amountTotal (p=0.3); devices.info (p=0.3); devices.info.count (p=0.3); devices.info.enabled (p=0.3); Gateways.Meta (p=0.3); sites.externalId (p=0.3)

### iot/rivet_iot (camel; 10 colecoes, 118 campos, aninhamento max 3)

- **Nomes ambiguos/similares** (7): `at` repetido em alerts: ack.at, history.at; `by` repetido em alerts: ack.by, history.by; variantes de estilo no schema: ['VersionNo', 'versionNo']; variantes de estilo no schema: ['Tags', 'tags']; variantes de estilo no schema: ['CreatedAt', 'createdAt']; variantes de estilo no schema: ['Name', 'name']; firmware: `version` ~ `versionNo`
- **Contenedores genericos (inner keys imprevisiveis)** (7): firmware.tags -> {}; readings.flags -> {}; gateways.extra -> {count, quota, score}; sensors.tags -> {}; Sites.Tags -> {}; commands.info -> {count, score, window}; devices.labels -> {env, owner, zone}
- **Campos opcionais raros (presenca < 0,35)** (4): firmware.versionNo (p=0.3); readings.raw (p=0.3); commands.info.score (p=0.3); telemetryBatches.history (p=0.3)

### iot/saffron_iot (camel; 9 colecoes, 152 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (17): `em` repetido em leituras: em, history.em; `em` repetido em alertas: ack.em, history.em; `by` repetido em alertas: ack.by, history.by; `from` repetido em alertas: context.window.from, history.change.from; `to` repetido em alertas: context.window.to, history.change.to; `label` repetido em locais: attrs.label, attrs.configuracoes.label; `label` repetido em manutencoes: info.limits.label, info.limits.limits.label; `habilitado` repetido em locais: attrs.habilitado, attrs.configuracoes.habilitado ...
- **Contenedores genericos (inner keys imprevisiveis)** (8): leituras.flags -> {}; locais.attrs -> {configuracoes, habilitado, label}; manutencoes.info -> {habilitado, limits, pontuacao, total}; sensores.meta -> {audit, habilitado, tamanho, total}; sensores.meta.audit -> {attrs, contagem, habilitado, total}; sensores.meta.audit.attrs -> {label, mode, tamanho}; dispositivos.labels -> {dono, env, zone}; dispositivos.attrs -> {contagem, tamanho, window}
- **Aninhamento profundo (>= 4 niveis)** (5): manutencoes.info.limits.limits.pontuacao; manutencoes.info.limits.limits.label; sensores.meta.audit.attrs.tamanho; sensores.meta.audit.attrs.label; sensores.meta.audit.attrs.mode
- **Campos opcionais raros (presenca < 0,35)** (5): leituras.flags (p=0.3); sensores.meta.habilitado (p=0.3); alertas.ack (p=0.3); gateways.connected (p=0.3); dispositivos.labels.dono (p=0.3)

### iot/sensorgrid (snake; 4 colecoes, 36 campos, aninhamento max 2)

- **Contenedores genericos (inner keys imprevisiveis)** (2): devices.labels -> {floor, zone}; readings.tags -> {batch, calibrated}
- **Campos opcionais raros (presenca < 0,35)** (1): devices.battery_pct (p=0.3)

## gaming

- Predicoes no val: 163; campo inexistente: **8 (4.9%)** (geral 4.0%). Schemas do dominio: arcadia (train, camel), beacon_gam (eval, pascal), harbor_gam (train, snake), juniper_gam (train, camel), meridian_gam (train, snake), quasar_gam (train, camel), saffron_gam (eval, camel), tundra_gam (train, camel), umbra_gam (train, snake), yonder_gam (train, pascal).

| tipo do nome inventado | n |
|---|---:|
| filho_inventado_de_objeto_existente | 3 |
| campo_de_colecao_irma | 2 |
| vocabulario_de_outro_schema_do_dominio | 1 |
| aninhamento_errado (folha existe em outro caminho) | 1 |
| invencao | 1 |

| schema | colecao | corte | nome inventado | tipo |
|---|---|---|---|---|
| gaming/beacon_gam | Items | key | `Stats.Def` | vocabulario_de_outro_schema_do_dominio |
| gaming/beacon_gam | Players | stage | `Meta.Extra.Count` | campo_de_colecao_irma |
| gaming/beacon_gam | Quests | key | `Meta.Extra` | filho_inventado_de_objeto_existente |
| gaming/beacon_gam | Sessions | stage | `At` | campo_de_colecao_irma |
| gaming/saffron_gam | achievements | multiline | `info.window` | filho_inventado_de_objeto_existente |
| gaming/saffron_gam | items | key | `extra.stats.size` | aninhamento_errado (folha existe em outro caminho) |
| gaming/saffron_gam | items | stmt_start | `extra.stats.label` | filho_inventado_de_objeto_existente |
| gaming/saffron_gam | players | partial | `handshake` | invencao |

Falhas por schema: gaming/beacon_gam (4), gaming/saffron_gam (4). Colecoes mais afetadas: gaming/saffron_gam.items (2), gaming/beacon_gam.Items (1), gaming/beacon_gam.Players (1), gaming/beacon_gam.Quests (1), gaming/beacon_gam.Sessions (1), gaming/saffron_gam.achievements (1).

### gaming/arcadia (camel; 5 colecoes, 48 campos, aninhamento max 2)

- **Contenedores genericos (inner keys imprevisiveis)** (2): players.stats -> {bestStreak, kda, losses, wins}; items.attrs -> {atk, def}
- **Campos opcionais raros (presenca < 0,35)** (1): players.inventory (p=0.3)

### gaming/beacon_gam (pascal; 10 colecoes, 159 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (14): `At` repetido em Purchases: At, History.At; `Score` repetido em Purchases: Attrs.Score, Attrs.Extra.Score; `Score` repetido em Players: Meta.Score, Meta.Audit.Audit.Score; `Mode` repetido em Purchases: Attrs.Mode, Attrs.Extra.Mode; `Mode` repetido em Players: Meta.Audit.Mode, Meta.Audit.Audit.Mode; `Size` repetido em Quests: Meta.Size, Meta.Limits.Size; `Audit` repetido em Players: Meta.Audit, Meta.Audit.Audit; `Label` repetido em Achievements: Info.Label, Info.Limits.Label ...
- **Contenedores genericos (inner keys imprevisiveis)** (12): Purchases.Attrs -> {Extra, Mode, Score}; Purchases.Attrs.Extra -> {Count, Mode, Score}; Purchases.Tags -> {}; Guilds.Meta -> {Count, Enabled, Score, Window}; Players.Meta -> {Audit, Score, Total}; Players.Meta.Audit -> {Audit, Label, Mode}; Players.Meta.Audit.Audit -> {Mode, Score}; Achievements.Info -> {Enabled, Label, Limits, Size} ...
- **Aninhamento profundo (>= 4 niveis)** (2): Players.Meta.Audit.Audit.Score; Players.Meta.Audit.Audit.Mode
- **Campos opcionais raros (presenca < 0,35)** (9): Purchases.Attrs.Extra (p=0.3); Purchases.History.Change.To (p=0.3); Guilds.Meta (p=0.3); Achievements.Info.Limits (p=0.3); Achievements.History (p=0.3); Quests.Meta.Label (p=0.3); Quests.Meta.Limits (p=0.3); Quests.Meta.Limits.Count (p=0.3) ...

### gaming/harbor_gam (snake; 6 colecoes, 98 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (12): `tamanho` repetido em sessoes: info.tamanho, info.stats.tamanho; `habilitado` repetido em sessoes: info.habilitado, info.stats.habilitado; `habilitado` repetido em guildas: attrs.habilitado, attrs.attrs.habilitado; `total` repetido em jogadores: info.total, info.configuracoes.quota.total; `label` repetido em jogadores: info.label, info.configuracoes.label, info.configuracoes.quota.label; `attrs` repetido em guildas: attrs, attrs.attrs; variantes de estilo no schema: ['CriadoEm', 'criado_em']; variantes de estilo no schema: ['Stats', 'stats'] ...
- **Contenedores genericos (inner keys imprevisiveis)** (6): sessoes.info -> {habilitado, stats, tamanho}; sessoes.info.stats -> {habilitado, mode, tamanho}; jogadores.info -> {configuracoes, label, mode, total}; guildas.attrs -> {attrs, em, habilitado}; guildas.attrs.attrs -> {habilitado, pontuacao}; Itens.Stats -> {Atk, Def, Spd}
- **Aninhamento profundo (>= 4 niveis)** (3): jogadores.info.configuracoes.quota.total; jogadores.info.configuracoes.quota.label; jogadores.info.configuracoes.quota.habilitado
- **Campos opcionais raros (presenca < 0,35)** (4): missoes.etiquetas (p=0.3); sessoes.info.stats (p=0.3); jogadores.history (p=0.3); guildas.attrs (p=0.3)

### gaming/juniper_gam (camel; 8 colecoes, 119 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (6): `mode` repetido em missoes: attrs.mode, attrs.limits.quota.mode; `pontuacao` repetido em sessoes: attrs.pontuacao, attrs.audit.pontuacao; `pontuacao` repetido em jogadores: attrs.pontuacao, attrs.quota.pontuacao; `contagem` repetido em missoes: steps.contagem, attrs.contagem; missoes: `itens` ~ `steps`; jogadores: `reason` ~ `region`
- **Contenedores genericos (inner keys imprevisiveis)** (6): missoes.attrs -> {contagem, limits, mode, tamanho}; sessoes.attrs -> {audit, contagem, habilitado, pontuacao}; sessoes.attrs.audit -> {em, pontuacao}; inventarios.slots.meta -> {bound, enchant}; jogadores.attrs -> {habilitado, pontuacao, quota}; itens.stats -> {atk, def, spd}
- **Aninhamento profundo (>= 4 niveis)** (2): missoes.attrs.limits.quota.mode; missoes.attrs.limits.quota.total
- **Campos opcionais raros (presenca < 0,35)** (3): missoes.attrs (p=0.3); jogadores.attrs.quota (p=0.3); itens.stats.spd (p=0.3)

### gaming/meridian_gam (snake; 9 colecoes, 147 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (9): `em` repetido em partidas: info.em, info.limits.audit.em; `mode` repetido em itens: meta.mode, meta.limits.mode; `contagem` repetido em partidas: info.contagem, info.limits.contagem; `pontuacao` repetido em rankings: meta.pontuacao, meta.extra.pontuacao; `pontuacao` repetido em partidas: players.pontuacao, info.limits.pontuacao; `tamanho` repetido em partidas: info.tamanho, info.limits.audit.tamanho; `extra` repetido em missoes: extra, extra.extra; rankings: `meta` ~ `nota` ...
- **Contenedores genericos (inner keys imprevisiveis)** (8): rankings.meta -> {contagem, extra, pontuacao, tamanho}; rankings.meta.extra -> {label, pontuacao}; itens.stats -> {atk, def, spd}; itens.meta -> {em, limits, mode, total}; partidas.info -> {contagem, em, limits, tamanho}; partidas.info.limits.audit -> {em, habilitado, tamanho}; missoes.extra -> {extra, label, pontuacao}; missoes.extra.extra -> {em, habilitado, total}
- **Aninhamento profundo (>= 4 niveis)** (3): partidas.info.limits.audit.em; partidas.info.limits.audit.tamanho; partidas.info.limits.audit.habilitado
- **Campos opcionais raros (presenca < 0,35)** (11): guildas.external_id (p=0.3); guildas.valor_total (p=0.3); guildas.history (p=0.3); itens.meta (p=0.3); itens.meta.limits.mode (p=0.3); jogadores.friends (p=0.3); conquistas.history.change.from (p=0.3); partidas.info.limits.pontuacao (p=0.3) ...

### gaming/quasar_gam (camel; 6 colecoes, 92 campos, aninhamento max 3)

- **Nomes ambiguos/similares** (6): `label` repetido em inventories: attrs.label, attrs.audit.label; quests: `items` ~ `steps`; players: `label` ~ `level`; players: `reason` ~ `region`; inventories: `bound` ~ `count`; guilds: `label` ~ `level`
- **Contenedores genericos (inner keys imprevisiveis)** (7): items.stats -> {atk, def, spd}; players.attrs -> {enabled, label, total, window}; inventories.slots.meta -> {bound, enchant}; inventories.attrs -> {audit, label, total}; inventories.attrs.audit -> {count, label, size}; guilds.info -> {label, size, stats}; guilds.info.stats -> {at, enabled, mode}
- **Campos opcionais raros (presenca < 0,35)** (2): quests.rewards.gold (p=0.3); inventories.slots.meta.enchant (p=0.3)

### gaming/saffron_gam (camel; 7 colecoes, 137 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (11): `mode` repetido em achievements: info.mode, info.limits.mode; `at` repetido em achievements: info.limits.at, history.at; `at` repetido em quests: info.at, info.attrs.quota.at; `at` repetido em purchases: at, extra.at, extra.extra.at, history.at; `score` repetido em quests: info.score, info.attrs.quota.score; `enabled` repetido em items: extra.enabled, extra.stats.enabled; `extra` repetido em purchases: extra, extra.extra; `stats` repetido em items: stats, extra.stats ...
- **Contenedores genericos (inner keys imprevisiveis)** (13): achievements.info -> {label, limits, mode}; achievements.tags -> {}; quests.info -> {at, attrs, score, size}; quests.info.attrs -> {enabled, mode, quota}; matches.tags -> {}; inventories.slots.meta -> {bound, enchant}; inventories.tags -> {}; purchases.extra -> {at, extra, label, mode} ...
- **Aninhamento profundo (>= 4 niveis)** (2): quests.info.attrs.quota.at; quests.info.attrs.quota.score
- **Campos opcionais raros (presenca < 0,35)** (7): achievements.info.mode (p=0.3); achievements.history (p=0.3); purchases.paidCoins (p=0.3); purchases.extra.at (p=0.3); items.extra.size (p=0.3); items.amountTotal (p=0.3); items.history (p=0.3)

### gaming/tundra_gam (camel; 5 colecoes, 80 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (8): `score` repetido em leaderboards: extra.score, extra.stats.extra.score; `at` repetido em inventories: info.at, info.window.at, info.window.limits.at; `extra` repetido em leaderboards: extra, extra.stats.extra; `size` repetido em leaderboards: extra.size, extra.stats.size; `label` repetido em leaderboards: extra.label, extra.stats.label; variantes de estilo no schema: ['CreatedAt', 'createdAt']; matches: `playerId` ~ `players`; Players: `Reason` ~ `Region`
- **Contenedores genericos (inner keys imprevisiveis)** (7): matches.tags -> {}; inventories.slots.meta -> {bound, enchant}; inventories.info -> {at, enabled, window}; leaderboards.extra -> {label, score, size, stats}; leaderboards.extra.stats -> {extra, label, size, total}; leaderboards.extra.stats.extra -> {at, score}; items.stats -> {atk, def, spd}
- **Aninhamento profundo (>= 4 niveis)** (4): inventories.info.window.limits.total; inventories.info.window.limits.at; leaderboards.extra.stats.extra.at; leaderboards.extra.stats.extra.score
- **Campos opcionais raros (presenca < 0,35)** (4): matches.versionNo (p=0.3); Players.Banned (p=0.3); inventories.info.enabled (p=0.3); inventories.versionNo (p=0.3)

### gaming/umbra_gam (snake; 4 colecoes, 76 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (9): `stats` repetido em items: stats, attrs.stats, attrs.stats.stats; `size` repetido em items: attrs.size, attrs.stats.size; `label` repetido em items: attrs.label, attrs.stats.label; `enabled` repetido em items: attrs.enabled, attrs.stats.stats.enabled; `at` repetido em achievements: extra.at, extra.window.at; variantes de estilo no schema: ['createdAt', 'created_at']; quests: `items` ~ `steps`; players: `label` ~ `level` ...
- **Contenedores genericos (inner keys imprevisiveis)** (8): items.stats -> {atk, def, spd}; items.attrs -> {enabled, label, size, stats}; items.attrs.stats -> {label, size, stats}; items.attrs.stats.stats -> {at, enabled}; achievements.extra -> {at, enabled, score, window}; players.extra -> {at, audit, enabled}; players.extra.audit -> {label, score, size}; players.tags -> {}
- **Aninhamento profundo (>= 4 niveis)** (2): items.attrs.stats.stats.enabled; items.attrs.stats.stats.at
- **Campos opcionais raros (presenca < 0,35)** (4): items.attrs.stats.stats.at (p=0.3); achievements.unlockedBy (p=0.3); players.extra (p=0.3); players.extra.audit.size (p=0.3)

### gaming/yonder_gam (pascal; 10 colecoes, 152 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (17): `Mode` repetido em Players: Meta.Mode, Meta.Extra.Mode, Meta.Extra.Settings.Mode; `Mode` repetido em Guilds: Attrs.Mode, Attrs.Limits.Mode; `Mode` repetido em Matches: Mode, Extra.Mode, Extra.Limits.Mode; `Count` repetido em Players: Meta.Count, Meta.Extra.Count; `Count` repetido em Quests: Steps.Count, Info.Count, Info.Extra.Count; `Count` repetido em Matches: Extra.Count, Extra.Limits.Quota.Count; `At` repetido em Sessions: Extra.At, History.At; `Enabled` repetido em Guilds: Attrs.Limits.Enabled, Attrs.Limits.Settings.Enabled ...
- **Contenedores genericos (inner keys imprevisiveis)** (13): inventories.slots.meta -> {bound, enchant}; Players.Meta -> {Count, Extra, Mode}; Players.Meta.Extra -> {At, Count, Mode, Settings}; Players.Meta.Extra.Settings -> {Enabled, Mode}; Quests.Info -> {Count, Enabled, Extra, Score}; Quests.Info.Extra -> {Count, Size}; Sessions.Extra -> {At, Label, Score, Settings}; Sessions.Extra.Settings -> {Score, Total} ...
- **Aninhamento profundo (>= 4 niveis)** (7): Players.Meta.Extra.Settings.Mode; Players.Meta.Extra.Settings.Enabled; Guilds.Attrs.Limits.Settings.Enabled; Guilds.Attrs.Limits.Settings.At; Guilds.Attrs.Limits.Settings.Count; Matches.Extra.Limits.Quota.Count ...
- **Campos opcionais raros (presenca < 0,35)** (6): Players.Meta.Extra.At (p=0.3); Quests.Rewards.Items (p=0.3); Quests.Info.Count (p=0.3); Sessions.Extra (p=0.3); Items.Stats.Def (p=0.3); Purchases.PaidCoins (p=0.3)

## logistics

- Predicoes no val: 157; campo inexistente: **9 (5.7%)** (geral 4.0%). Schemas do dominio: borealis_log (train, snake), freightline (eval, camel), granite_log (train, camel), kestrel_log (train, camel), lumen_log (train, pascal), meridian_log (train, camel), onyx_log (train, pascal), xylem_log (train, camel), yonder_log (eval, snake).

| tipo do nome inventado | n |
|---|---:|
| vocabulario_de_outro_schema_do_dominio | 2 |
| quase_igual (edicao <= 2) | 2 |
| campo_de_colecao_irma | 2 |
| filho_inventado_de_objeto_existente | 2 |
| aninhamento_errado (folha existe em outro caminho) | 1 |

| schema | colecao | corte | nome inventado | tipo |
|---|---|---|---|---|
| logistics/freightline | carriers | key | `services` | vocabulario_de_outro_schema_do_dominio |
| logistics/freightline | carriers | stage | `contracts.rate` | quase_igual (edicao <= 2) |
| logistics/freightline | shipments | multiline | `destination.state` | quase_igual (edicao <= 2) |
| logistics/freightline | vehicles | key | `info.attrs` | vocabulario_de_outro_schema_do_dominio |
| logistics/freightline | warehouses | partial | `createdAt` | campo_de_colecao_irma |
| logistics/yonder_log | pacotes | key | `peso_kg` | campo_de_colecao_irma |
| logistics/yonder_log | pacotes | multiline | `extra.limits.habilitado` | filho_inventado_de_objeto_existente |
| logistics/yonder_log | pacotes | multiline | `extra.window` | filho_inventado_de_objeto_existente |
| logistics/yonder_log | remessas | rest | `meta.quota.extra` | aninhamento_errado (folha existe em outro caminho) |

Falhas por schema: logistics/freightline (5), logistics/yonder_log (4). Colecoes mais afetadas: logistics/yonder_log.pacotes (3), logistics/freightline.carriers (2), logistics/freightline.shipments (1), logistics/freightline.vehicles (1), logistics/freightline.warehouses (1), logistics/yonder_log.remessas (1).

### logistics/borealis_log (snake; 8 colecoes, 126 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (13): `at` repetido em routes: extra.attrs.at, history.at; `from` repetido em routes: stops.window.from, history.change.from; `to` repetido em routes: stops.window.to, history.change.to; `score` repetido em manifests: extra.score, extra.extra.score; `total` repetido em manifests: extra.extra.total, extra.extra.window.total; `extra` repetido em manifests: extra, extra.extra; `label` repetido em manifests: extra.label, extra.extra.window.label; routes: `code` ~ `done` ...
- **Contenedores genericos (inner keys imprevisiveis)** (7): drivers.meta -> {at, quota, score}; manifests.extra -> {extra, label, score}; manifests.extra.extra -> {score, size, total, window}; routes.extra -> {attrs, enabled, score}; routes.extra.attrs -> {at, mode}; shipments.tags -> {}; warehouses.tags -> {}
- **Aninhamento profundo (>= 4 niveis)** (3): manifests.extra.extra.window.label; manifests.extra.extra.window.count; manifests.extra.extra.window.total
- **Campos opcionais raros (presenca < 0,35)** (6): drivers.rating (p=0.3); drivers.meta.quota.enabled (p=0.3); manifests.extra (p=0.3); manifests.extra.extra (p=0.3); manifests.extra.extra.score (p=0.3); routes.history.change.to (p=0.3)

### logistics/freightline (camel; 5 colecoes, 57 campos, aninhamento max 3)

- **Nomes ambiguos/similares** (2): variantes de estilo no schema: ['_id', 'id']; carriers: `lane` ~ `name`
- **Campos opcionais raros (presenca < 0,35)** (1): shipments.volumeM3 (p=0.3)

### logistics/granite_log (camel; 8 colecoes, 149 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (11): `at` repetido em warehouses: attrs.limits.at, history.at; `at` repetido em vehicles: gps.at, history.at; `enabled` repetido em shipments: meta.enabled, meta.extra.enabled, meta.extra.limits.enabled; `enabled` repetido em drivers: info.enabled, info.quota.extra.enabled; `score` repetido em shipments: meta.score, meta.extra.score; `total` repetido em shipments: meta.total, meta.extra.total, meta.extra.limits.total; `total` repetido em vehicles: attrs.total, attrs.window.total; `size` repetido em drivers: info.quota.size, info.quota.extra.size ...
- **Contenedores genericos (inner keys imprevisiveis)** (11): manifests.tags -> {}; shipments.meta -> {enabled, extra, score, total}; shipments.meta.extra -> {enabled, limits, score, total}; shipments.tags -> {}; warehouses.attrs -> {enabled, label, limits, size}; warehouses.tags -> {}; carriers.tags -> {}; drivers.info -> {enabled, mode, quota, total} ...
- **Aninhamento profundo (>= 4 niveis)** (6): shipments.meta.extra.limits.size; shipments.meta.extra.limits.total; shipments.meta.extra.limits.enabled; drivers.info.quota.extra.at; drivers.info.quota.extra.enabled; drivers.info.quota.extra.size
- **Campos opcionais raros (presenca < 0,35)** (2): carriers.history (p=0.3); drivers.info.total (p=0.3)

### logistics/kestrel_log (camel; 10 colecoes, 159 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (15): `at` repetido em stockMoves: at, info.audit.at, history.at; `label` repetido em stockMoves: info.label, info.audit.attrs.label; `mode` repetido em stockMoves: info.mode, info.audit.mode; `mode` repetido em shipments: legs.mode, info.mode; `size` repetido em stockMoves: info.size, info.audit.attrs.size; `size` repetido em drivers: extra.size, extra.attrs.size; `size` repetido em manifests: info.size, info.attrs.extra.size; `enabled` repetido em drivers: extra.attrs.enabled, extra.attrs.audit.enabled ...
- **Contenedores genericos (inner keys imprevisiveis)** (15): stockMoves.info -> {audit, label, mode, size}; stockMoves.info.audit -> {at, attrs, mode}; stockMoves.info.audit.attrs -> {enabled, label, size}; tracking events.details -> {device, operator}; drivers.extra -> {attrs, score, size}; drivers.extra.attrs -> {audit, enabled, size}; drivers.extra.attrs.audit -> {count, enabled}; manifests.info -> {attrs, count, size, total} ...
- **Aninhamento profundo (>= 4 niveis)** (9): stockMoves.info.audit.attrs.label; stockMoves.info.audit.attrs.enabled; stockMoves.info.audit.attrs.size; drivers.extra.attrs.audit.count; drivers.extra.attrs.audit.enabled; manifests.info.attrs.extra.label ...
- **Campos opcionais raros (presenca < 0,35)** (5): stockMoves.info.audit.at (p=0.3); tracking events.details (p=0.3); tracking events.versionNo (p=0.3); drivers.extra.attrs (p=0.3); warehouses.history.change.to (p=0.3)

### logistics/lumen_log (pascal; 10 colecoes, 181 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (35): `At` repetido em Stock Moves: At, Extra.Window.At, Extra.Window.Window.At; `At` repetido em Packages: Meta.At, Meta.Stats.Window.At; `Size` repetido em Stock Moves: Extra.Size, Extra.Window.Window.Size; `Size` repetido em Vehicles: Info.Size, Info.Quota.Extra.Size; `Score` repetido em Vehicles: Info.Score, Info.Quota.Extra.Score; `Window` repetido em Stock Moves: Extra.Window, Extra.Window.Window; `Enabled` repetido em Packages: Meta.Stats.Enabled, Meta.Stats.Window.Enabled; `Enabled` repetido em Manifests: Extra.Attrs.Enabled, Extra.Attrs.Limits.Enabled ...
- **Contenedores genericos (inner keys imprevisiveis)** (14): Stock Moves.Extra -> {Score, Size, Window}; Packages.Meta -> {At, Score, Size, Stats}; Packages.Meta.Stats -> {Count, Enabled, Window}; Packages.Tags -> {}; trackingEvents.details -> {device, operator}; Routes.Tags -> {}; warehouses.extra -> {count, size, window}; warehouses.extra.window.audit -> {at, enabled, label} ...
- **Aninhamento profundo (>= 4 niveis)** (13): Stock Moves.Extra.Window.Window.At; Stock Moves.Extra.Window.Window.Size; Packages.Meta.Stats.Window.Enabled; Packages.Meta.Stats.Window.At; warehouses.extra.window.audit.label; warehouses.extra.window.audit.enabled ...
- **Campos opcionais raros (presenca < 0,35)** (7): Stock Moves.Batch (p=0.3); trackingEvents.details.operator (p=0.3); warehouses.extra (p=0.3); Drivers.VehicleId (p=0.3); Drivers.Rating (p=0.3); Shipments.CarrierId (p=0.3); Vehicles.Info.Quota.Extra.Total (p=0.3)

### logistics/meridian_log (camel; 9 colecoes, 178 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (22): `em` repetido em eventos rastreio: em, meta.em, meta.audit.em; `em` repetido em movimentosEstoque: em, history.em; `em` repetido em veiculos: gps.em, extra.configuracoes.configuracoes.em; `em` repetido em remessas: meta.audit.window.em, history.em; `em` repetido em motoristas: meta.configuracoes.em, history.em; `habilitado` repetido em eventos rastreio: meta.habilitado, meta.audit.habilitado; `habilitado` repetido em motoristas: meta.habilitado, meta.configuracoes.stats.habilitado; `mode` repetido em transportadoras: meta.mode, meta.window.mode ...
- **Contenedores genericos (inner keys imprevisiveis)** (12): eventos rastreio.details -> {dispositivo, operator}; eventos rastreio.meta -> {audit, em, habilitado, mode}; eventos rastreio.meta.audit -> {em, habilitado, pontuacao}; transportadoras.meta -> {contagem, mode, window}; veiculos.extra -> {configuracoes, habilitado, pontuacao}; remessas.meta -> {audit, habilitado, tamanho}; remessas.meta.audit -> {contagem, label, total, window}; motoristas.meta -> {configuracoes, contagem, habilitado, tamanho} ...
- **Aninhamento profundo (>= 4 niveis)** (13): transportadoras.meta.window.quota.em; transportadoras.meta.window.quota.label; transportadoras.meta.window.quota.total; veiculos.extra.configuracoes.configuracoes.tamanho; veiculos.extra.configuracoes.configuracoes.label; veiculos.extra.configuracoes.configuracoes.em ...
- **Campos opcionais raros (presenca < 0,35)** (7): eventos rastreio.meta (p=0.3); eventos rastreio.meta.audit.em (p=0.3); eventos rastreio.etiquetas (p=0.3); transportadoras.meta.window.quota (p=0.3); veiculos.extra.pontuacao (p=0.3); motoristas.meta.configuracoes.total (p=0.3); motoristas.history (p=0.3)

### logistics/onyx_log (pascal; 10 colecoes, 162 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (24): `Size` repetido em Packages: Meta.Size, Meta.Attrs.Size; `Label` repetido em StockMoves: Info.Label, Info.Quota.Label; `At` repetido em StockMoves: At, Info.At, History.At; `Score` repetido em Carriers: Info.Score, Info.Attrs.Score; `Enabled` repetido em Carriers: Info.Enabled, Info.Attrs.Enabled, Info.Attrs.Quota.Enabled; `Count` repetido em Carriers: Info.Count, Info.Attrs.Count, Info.Attrs.Quota.Count; `From` repetido em Routes: Stops.Window.From, History.Change.From; `To` repetido em Routes: Stops.Window.To, History.Change.To ...
- **Contenedores genericos (inner keys imprevisiveis)** (12): Manifests.Info -> {At, Label, Size, Window}; Carriers.Info -> {Attrs, Count, Enabled, Score}; Carriers.Info.Attrs -> {Count, Enabled, Quota, Score}; Packages.Meta -> {Attrs, Enabled, Size, Total}; Packages.Meta.Attrs -> {Mode, Size}; StockMoves.Info -> {At, Label, Quota}; Routes.Tags -> {}; vehicles.info -> {count, score, settings} ...
- **Aninhamento profundo (>= 4 niveis)** (5): Carriers.Info.Attrs.Quota.Enabled; Carriers.Info.Attrs.Quota.Count; vehicles.info.settings.extra.at; vehicles.info.settings.extra.total; vehicles.info.settings.extra.enabled
- **Campos opcionais raros (presenca < 0,35)** (5): Packages.Meta (p=0.3); StockMoves.Batch (p=0.3); StockMoves.Info.Label (p=0.3); Routes.History.Change.To (p=0.3); vehicles.amountTotal (p=0.3)

### logistics/xylem_log (camel; 9 colecoes, 170 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (19): `at` repetido em tracking-events: at, meta.at; `at` repetido em stockMoves: at, attrs.attrs.extra.at; `size` repetido em carriers: meta.size, meta.limits.size, meta.limits.stats.size; `size` repetido em stockMoves: attrs.size, attrs.attrs.extra.size; `total` repetido em drivers: info.total, info.window.attrs.total; `count` repetido em tracking-events: meta.count, meta.stats.count; `count` repetido em vehicles: info.count, info.quota.count; `attrs` repetido em stockMoves: attrs, attrs.attrs ...
- **Contenedores genericos (inner keys imprevisiveis)** (16): carriers.meta -> {at, limits, size}; carriers.meta.limits.stats -> {count, size}; carriers.tags -> {}; tracking-events.details -> {device, operator}; tracking-events.meta -> {at, count, label, stats}; tracking-events.meta.stats -> {count, size}; tracking-events.tags -> {}; stockMoves.attrs -> {attrs, score, size} ...
- **Aninhamento profundo (>= 4 niveis)** (6): carriers.meta.limits.stats.size; carriers.meta.limits.stats.count; stockMoves.attrs.attrs.extra.at; stockMoves.attrs.attrs.extra.size; drivers.info.window.attrs.enabled; drivers.info.window.attrs.total
- **Campos opcionais raros (presenca < 0,35)** (11): carriers.meta.limits (p=0.3); carriers.meta.limits.stats.size (p=0.3); tracking-events.meta.label (p=0.3); stockMoves.attrs (p=0.3); stockMoves.attrs.size (p=0.3); stockMoves.amountTotal (p=0.3); routes.info.attrs.label (p=0.3); drivers.vehicleId (p=0.3) ...

### logistics/yonder_log (snake; 9 colecoes, 135 campos, aninhamento max 4)

- **Nomes ambiguos/similares** (8): `from` repetido em rotas: stops.window.from, history.change.from; `to` repetido em rotas: stops.window.to, history.change.to; `mode` repetido em remessas: legs.mode, extra.mode; `extra` repetido em remessas: extra, extra.extra; `total` repetido em remessas: extra.total, extra.extra.total; `contagem` repetido em pacotes: extra.contagem, extra.limits.contagem; `em` repetido em eventos-rastreio: em, history.em; motoristas: `nome` ~ `nota`
- **Contenedores genericos (inner keys imprevisiveis)** (5): remessas.extra -> {contagem, extra, mode, total}; remessas.extra.extra -> {habilitado, total}; eventos-rastreio.details -> {dispositivo, operator}; pacotes.extra -> {contagem, limits, mode}; pacotes.extra.limits.attrs -> {em, total}
- **Aninhamento profundo (>= 4 niveis)** (2): pacotes.extra.limits.attrs.total; pacotes.extra.limits.attrs.em
- **Campos opcionais raros (presenca < 0,35)** (4): eventos-rastreio.details (p=0.3); eventos-rastreio.valor_total (p=0.3); manifestos.valor_total (p=0.3); rotas.external_id (p=0.3)

## Sintese e recomendacoes (nao aplicadas)

1. Contenedores genericos (`meta`, `info`, `extra`, `attrs`, `stats`, `seo`, `social`...) cujo conteudo muda por colecao concentram as invencoes `x.y`: o modelo sabe que o pai existe e inventa o filho. Em contexto `full` o `RESULT FIELDS` os lista; sem ele (none/irrelevant) nao ha como acertar: a v2d so treina esses cortes quando o caminho esta visivel no documento.
2. Variantes de estilo (camel x snake, `validUntil` vs `valid_until`) entre schemas do mesmo dominio e a mesma folha em caminhos diferentes (`at`, `total`, `name`) sao fonte de `aninhamento_errado`; mitigar com exemplos de PropertyKey/FieldReferenceString (sobre-amostrados x2 na v2d), nao mudando os schemas (decisao: somente relatorio).
3. Campos opcionais raros: so entram em `FieldSchema`/`RESULT FIELDS` por amostragem (`schema_subset`); quando o modelo os cita sem estarem visiveis, o corte e descartado nos contextos sem schema (v2d).

