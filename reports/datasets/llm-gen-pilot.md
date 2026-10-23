# Piloto da geração de consultas por LLM local (Qwen3-14B Q4_K_M)

Gerado em 2026-10-23 01:02 por `scripts/data/llm/pilot.py`. Artefatos: `data/generated/v3/pilot` (não versionado). Nada em volume foi gerado; este é o piloto que a decisão do usuário precisa para liberar J18/J19.

## 1. Resumo

- **Vazão:** 2,982 consultas/h geradas (96.7 tok/s de saída agregados, 8 slots, 8 consultas por chamada); validação por execução do texto: 14,575 consultas/h em 3 lotes paralelos (CPU, em paralelo com a GPU).
- **Aceitação:** 229/304 = 75.3% (funil na §3); vazio mantido em 35 das aceitas (15.3%, limite 25%).
- **Diversidade:** 157 operadores distintos e 17 estágios distintos em 229 consultas (corpus de gramática, mesma amostra: 156 / 13); 4-gramas distintos por token 0.35 vs 0.398; consultas multi-instrução 45.0% vs 17.9%.
- **Estimativa 60k aceitas:** 79,651 consultas brutas ≈ **27 h de GPU** de geração (+ validação 5 h de CPU em paralelo), ≈ 323,843 cortes FIM (5.4 por consulta, após decontaminação).

## 2. Configuração e dispositivo

- Modelo: `BaseRepos/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf` (Apache-2.0, sem download). `llama-server` b11693, `-fa on`, KV q8_0, 8 slots × 5120 tokens, `--jinja`, thinking desligado (`chat_template_kwargs.enable_thinking=false`, `/no_think`), saída restrita por `json_schema`.
- **Dispositivo escolhido por NOME:** `Vulkan0` = AMD Radeon RX 7800 XT (16368 MiB). Guarda de velocidade: prompt 839.1 t/s (mínimo 300), decode 54.1 t/s.
- **Confirmação independente (contadores do Windows, durante a geração):** `\GPU Adapter Memory(*)\Dedicated Usage` = 13.69 GB no adaptador `luid_0x00000000_0x00012602_phys_0` (os outros ≈ 0 GB); `\GPU Engine(*)` do pid do servidor: 65.0% nesse adaptador e 0% nos demais. Veredito: **ok**.

## 3. Funil de aceitação

| etapa | restam | % do bruto | rejeitadas na etapa |
|---|---:|---:|---:|
| bruto (gerado) | 304 | 100.0% |  |
| static (escrita/ddl/eval…) | 304 | 100.0% | 0 |
| syntax (node vm.Script) | 283 | 93.1% | 21 |
| dup (hash normalizado) | 283 | 93.1% | 0 |
| exec (mongosh, texto real) | 233 | 76.6% | 50 |
| fields (plano real × schema) | 229 | 75.3% | 4 |
| shape (≤2 por forma/coleção) | 229 | 75.3% | 0 |
| vazio ≤ 25% | 229 | 75.3% | 0 |

Principais motivos de rejeição:

| etapa | motivo | n |
|---|---|---:|
| exec | no_exec:  | 14 |
| syntax | Unexpected token '.' | 9 |
| syntax | Unexpected token ':' | 4 |
| fields | campo inexistente | 4 |
| syntax | Unexpected token '}' | 2 |
| exec | error: 16702: PlanExecutor error during aggregation :: caused by :: $c | 2 |
| exec | error: 31325: Invalid $project :: caused by :: Unknown expression $pus | 2 |
| exec | error: Uncaught ReferenceError: $$ROOT is not defined  | 2 |
| syntax | Unexpected end of input | 1 |
| syntax | Unexpected number | 1 |
| syntax | missing ) after argument list | 1 |
| syntax | Invalid destructuring assignment target | 1 |
| syntax | Unexpected token 'false' | 1 |
| syntax | Unexpected token '{' | 1 |

## 4. Prova de leitura apenas

- **Estática:** `common.static_check` recusa insert/update/delete/replace/drop/createIndex/bulkWrite/findAndModify/runCommand/eval/load/require/`$out`/`$merge`/`$function`/`$where`/`$accumulator`, `getSiblingDB` fora do schema, `maxTimeMS` e loops infinitos (mascarando strings, comentários e regex).
- **Proxy de rede somente-leitura (`ro_proxy.py`)** entre o mongosh e o mongod: allowlist de comandos (find, aggregate, count, distinct, getMore, killCursors, explain de leitura, listCollections/listIndexes, handshake); qualquer outro é respondido com `{ok:0, code:13}` sem chegar ao servidor; `aggregate` com `$out`/`$merge`/`$function` é recusado; bancos fora da lista são recusados; `maxTimeMS=3000` é injetado em todo comando de dados.
- Comandos que **chegaram** ao servidor: `{"ismaster": 51, "buildinfo": 8, "getparameter": 16, "atlasversion": 8, "listdatabases": 283, "listcollections": 283, "ping": 566, "aggregate": 132, "find": 127, "hello": 7, "getmore": 27, "killcursors": 17, "endsessions": 8, "count": 3, "distinct": 3}`; escritas encaminhadas: `{}`; recusados pelo proxy: `{"aggregate:db=admin": 8}`.
- **`dbHash` de cada banco tocado antes = depois:** True (38 bancos). (`opcounters` do servidor no período, só informativo — o outro agente semeia bancos em paralelo: `{'insert': 0, 'update': 0, 'delete': 0, 'query': 257, 'command': 1310, 'getmore': 27}`.)

## 5. Diversidade × corpus de gramática (v2c)

Mesma função de métrica (`metrics.py`) nas 229 consultas LLM aceitas e em 229 snippets únicos de treino do v2c (amostra determinística dos shards finais do FIM validado). “Corpus-sub” = só categorias aggregation/crud/operators/dsl (as comparáveis).

| métrica | LLM (aceitas) | corpus-sub | corpus (todas) |
|---|---:|---:|---:|
| operadores `$` distintos | 157 | 156 | 163 |
| operadores por consulta | 3.84 | 4.11 | 4.43 |
| estágios distintos | 17 | 13 | 13 |
| estágios por pipeline | 2.8 | 3.09 | 2.78 |
| conjuntos de operadores distintos / consulta | 0.655 | 0.79 | 0.755 |
| 3-gramas distintos (tokens normalizados) | 3353 | 3264 | 3391 |
| 4-gramas distintos | 4857 | 4462 | 4628 |
| 4-gramas distintos / total | 0.35 | 0.398 | 0.374 |
| formas estruturais distintas / consulta | 1.0 | 1.0 | 1.0 |
| caracteres (média) | 260.2 | 210.1 | 241.7 |
| caracteres (p90) | 414 | 343 | 425 |
| linhas (média) | 10.04 | 4.72 | 5.45 |
| aninhamento máx. (média) | 4.82 | 4.68 | 5.09 |
| aninhamento máx. (p90) | 8 | 7 | 8 |
| multiline | 0.747 | 0.664 | 0.703 |
| com comentários | 0.175 | 0.0 | 0.0 |
| multi-instrução | 0.45 | 0.179 | 0.166 |
| com const/let/var | 0.493 | 0.179 | 0.166 |
| com função/arrow | 0.026 | 0.0 | 0.0 |
| getSiblingDB/getCollection/colchetes | 0.288 | 0.472 | 0.406 |
| ObjectId/ISODate/NumberDecimal/… | 0.402 | 0.358 | 0.362 |
| aggregate | 0.441 | 0.472 | 0.48 |
| find/findOne | 0.524 | 0.297 | 0.236 |
| CRLF | 0.0 | 0.323 | 0.41 |

Estágios só no LLM: $bucketAuto, $facet, $graphLookup, $replaceRoot, $sample, $setWindowFields, $unionWith; só no corpus: $set, $sortByCount, $unset. Operadores mais usados no LLM: $gte, $sum, $group, $project, $match, $lte, $gt, $sort, $limit, $in, $exists, $lt, $regex, $avg, $count, $cond, $and, $options.

## 6. Aceitação por tarefa e persona

| tarefa | geradas | aceitas | taxa |
|---|---:|---:|---:|
| script_vars | 15 | 11 | 73.3% |
| bson_literals | 14 | 14 | 100.0% |
| range_filters | 14 | 10 | 71.4% |
| date_window | 14 | 13 | 92.9% |
| top_n | 13 | 11 | 84.6% |
| lookup_join | 13 | 7 | 53.8% |
| find_projection_sort | 13 | 12 | 92.3% |
| nested_paths | 12 | 9 | 75.0% |
| text_regex | 12 | 11 | 91.7% |
| sort_group_limit_chain | 12 | 9 | 75.0% |
| conditional_expr | 12 | 8 | 66.7% |
| cursor_pagination | 12 | 11 | 91.7% |
| date_arithmetic | 11 | 5 | 45.5% |
| dedupe | 11 | 8 | 72.7% |
| facet_dashboard | 10 | 8 | 80.0% |
| monthly_report | 9 | 6 | 66.7% |
| data_cleaning | 9 | 5 | 55.6% |
| string_ops | 9 | 1 | 11.1% |
| count_distinct | 9 | 8 | 88.9% |
| single_entity_debug | 9 | 9 | 100.0% |
| bracket_access | 8 | 8 | 100.0% |
| object_reshape | 7 | 4 | 57.1% |
| window_running | 7 | 3 | 42.9% |
| array_ops | 7 | 7 | 100.0% |
| sibling_db | 6 | 5 | 83.3% |
| union_with | 6 | 3 | 50.0% |
| graph_hierarchy | 5 | 4 | 80.0% |
| bucket_distribution | 5 | 3 | 60.0% |
| util_function | 4 | 4 | 100.0% |
| stats | 4 | 3 | 75.0% |
| sample_qa | 4 | 4 | 100.0% |
| audit_invariants | 4 | 2 | 50.0% |
| compare_fields_expr | 3 | 2 | 66.7% |
| conditional_pivot | 1 | 1 | 100.0% |

| persona | geradas | aceitas | taxa |
|---|---:|---:|---:|
| backend_dev | 30 | 19 | 63.3% |
| terse_power_user | 27 | 22 | 81.5% |
| monthly_report | 25 | 20 | 80.0% |
| junior_dev | 24 | 18 | 75.0% |
| support | 24 | 21 | 87.5% |
| bi_analyst | 24 | 21 | 87.5% |
| security_auditor | 24 | 16 | 66.7% |
| tech_writer | 22 | 17 | 77.3% |
| product_manager | 21 | 13 | 61.9% |
| prod_debug | 19 | 14 | 73.7% |
| data_engineer | 18 | 16 | 88.9% |
| sre | 17 | 13 | 76.5% |
| dba | 15 | 10 | 66.7% |
| qa_tester | 14 | 9 | 64.3% |

## 7. Cortes FIM e contexto

- 229 consultas → 229 snippets → 1374 cortes → **1236** após decontaminação oficial (`ngram_min_content: 2`) contra val/test/bench do v2c (57471 documentos) e contra a homologação ({'test': 24, 'val': 84, 'bench': 30}); removidos: 23 documentos.
- `context_kind`: {'full': 906, 'irrelevant': 183, 'none': 147} → **sem schema visível (none+irrelevant): 26.7%** (meta ≥ 25%), com alvo conservador (`field_need ≠ 2`; 381 cortes exigiriam campo invisível e ficaram só em `full`).
- `cut_kind`: {'partial': 157, 'rest': 156, 'value': 212, 'operator': 170, 'member': 156, 'key': 238, 'multiline': 62, 'stage': 57, 'stmt_start': 28}; categoria: {'dsl': 126, 'aggregation': 504, 'crud': 606}.
- Registro: `source: llm-qwen3-14b`, `author: llm`, `split_role: train`, `group_id`/`split_key` por consulta; `prompt` renderizado com o contrato `editor-context-v1` e orçamento de 2048 tokens (tokenizer Qwen2.5-Coder); G2.7: `blind_guard` em cada registro/prompt e `homolog_guard` no treino.

## 8. Ensaio da homologação (schemas `hom`; fora de `data/homolog`, sem selar)

- Geração hom: 160 consultas a 2,597/h; aceitas 111 (69.4%).
- Ensaio `build_homolog --dry-run`: 2009 itens a partir de 111 consultas LLM (+495 snippets de gramática); menor célula (cut_kind × context_kind) = 24 (alvo ≥ 50); none+irrelevant 52.0%; células em falta: 2.
  - por cut_kind: {'key': 232, 'value': 232, 'operator': 232, 'stage': 232, 'member': 232, 'partial': 232, 'rest': 232, 'multiline': 232, 'stmt_start': 153}; por context_kind: {'none': 540, 'irrelevant': 504, 'full': 965}.

## 9. Amostra: 10 consultas aceitas

**A1** — `retail-pos/v3_quasar_pos.Items_new` · graph_hierarchy · bi_analyst · extracted · 6 docs · intenção: 

```js
const pipeline = [
  {
    $group: {
      _id: '$Dept',
      itemCount: { $sum: 1 },
      avgPrice: { $avg: '$Price' }
    }
  },
  {
    $sort: { itemCount: -1 }
  }
];
console.log(db.Items_new.aggregate(pipeline).toArray());
```

**A2** — `hr/saffron_hr.Departamentos` · text_regex · terse_power_user · pretty2 · 114 docs · intenção: 

```js
const coll = db.Departamentos;
const search = 'mesa';
// Case-insensitive search on Nome and Codigo fields
const results = coll.find({$or:[{Nome:{$regex:search, $options:'i'}}, {Codigo:{$regex:search, $options:'i'}}]}).toArray();
```

**A3** — `energy/granite_nrg.contracts` · range_filters · prod_debug · lines · 5 docs · intenção: 

```js
db.getSiblingDB('granite_nrg').contracts.find({signedAt:{$gte:ISODate('2025-06-01T00:00:00Z'),$lt:ISODate('2026-01-01T00:00:00Z')},termMonths:{$in:[20,30]},green:{$ne:true}}).sort({signedAt:-1}).limit(5).toArray()
```

**A4** — `gaming/v3_vertex_gam.Sessions` · lookup_join · junior_dev · extracted · 101 docs · intenção: 

```js
const pipeline = [
  {
    $lookup: {
      from: "Players",
      localField: "PlayerId",
      foreignField: "_id",
      as: "player_info"
    }
  },
  {
    $unwind: "$player_info"
  },
  {
    $match: {
      "player_info.Level": {
        $gte: 10
      }
    }
  },
  {
    $project: {
      PlayerId: 1,
      player_info: {
        Level: 1,
        Email: 1
      }
    }
  }
];

db.Sessions.aggregate(pipeline);
```

**A5** — `logistics/v3_monolith_log.gpsPings` · monthly_report · data_engineer · short · 21 docs · intenção: 

```js
const coll = db.gpsPings;
coll.aggregate([
  {
    $group: {
      _id: {
        year: { $dateToString: { format: "%Y", date: "$at" } },
        month: { $dateToString: { format: "%m", date: "$at" } }
      },
      totalDocs: { $sum: 1 },
      avgSpeed: { $avg: "$speedKmh" }
    }
  },
  {
    $sort: { '_id.year': 1, '_id.month': 1 }
  }
]).toArray();
```

**A6** — `content/v3_orbital_cnt.subscribers` · union_with · support · pretty2 · 334 docs · intenção: 

```js
db.subscribers.aggregate([
  {
    $unionWith: {
      coll: 'subscribers',
      pipeline: [
        {
          $match: {
            email: { $regex: '.*example.com.*', $options: 'i' }
          }
        }
      ]
    }
  },
  {
    $match: {
      email: { $regex: '.*example.com.*', $options: 'i' }
    }
  }
]).toArray();
```

**A7** — `logistics/v3_saffron_log.customs` · object_reshape · product_manager · pretty2 · 5 docs · intenção: 

```js
db.customs.aggregate([
  {
    $sample: { size: 5 }
  },
  {
    $project: {
      shp_id: 1,
      status: 1,
      events: { $arrayElemAt: ["$events", 0] }
    }
  }
]).toArray()
```

**A8** — `manufacturing/v3_fjord_mfg.t_machines` · bracket_access · security_auditor · compact · 5 docs · intenção: 

```js
const coll = db.getCollection('t_machines');
const resultado = coll.find({ Status: 'down', Spec: { $exists: true } }).sort({ Commissioned: -1 }).limit(5);
console.log(resultado.toArray());
```

**A9** — `content/v3_orbital_cnt.subscribers` · audit_invariants · prod_debug · extracted · 650 docs · intenção: 

```js
const invariant = {
  $expr: {
    $and: [
      {
        $gte: ["$since", "$legacySince"]
      },
      {
        $eq: [
          "$topics.0",
          {
            $cond: [
              { $ne: ["$topics.0", null] },
              "$topics.0",
              'sem topico'
            ]
          }
        ]
      }
    ]
  }
};

db.subscribers.aggregate([{
  $match: invariant
}]).toArray();
```

**A10** — `healthcare-synthetic/v3_ion_hlt.Visits` · sort_group_limit_chain · sre · pretty4 · 4 docs · intenção: 

```js
let pipeline = [
  {
    $match: {
      At: {
        $gte: ISODate('2025-01-01T00:00:00Z'),
        $lte: ISODate('2026-09-30T23:59:59Z')
      }
    }
  },
  {
    $group: {
      _id: '$Kind',
      count: { $sum: 1 }
    }
  },
  {
    $sort: { count: -1 }
  },
  {
    $limit: 5
  },
  {
    $project: {
      _id: 0,
      kind: '$_id',
      total: '$count'
    }
  }
];
let result = db.Visits.aggregate(pipeline).toArray();
console.log(printjson(result));
```

## 10. Amostra: 10 consultas rejeitadas (com motivo)

**R1** — etapa `fields` — motivo: campo inexistente: UpdatedAt — `commerce/meridian_com.Reviews` · audit_invariants

```js
db.Reviews.aggregate([ { $match: { $expr: { $lt: ["$CreatedAt", "$UpdatedAt"] } } } ])
```

**R2** — etapa `syntax` — motivo: Unexpected token '}' — `saas/v3_onyx_saas.integrations` · dedupe

```js
db.getSiblingDB('v3_onyx_saas').integrations.aggregate([{$group:{_id:"$provider",count:{$sum:1}}},{$match:{count:{$gt:1}}},{$project:{_id:1,docs:{$push:"$_id"}}}}]
```

**R3** — etapa `exec` — motivo: error: Uncaught TypeError: db.timesheets. ... eline).limit is not a function  — `hr/prism_hr.timesheets` · date_arithmetic

```js
var today = new Date();
var lastWeek = new Date();
lastWeek.setDate(lastWeek.getDate() - 7);
var pipeline = [
  {
    $match: {
      date: {
        $gte: lastWeek,
        $lte: today
      }
    }
  },
  {
    $addFields: {
      weekDay: {
        $dateDayOfWeek: {
          date: "$date",
          unit: "dayOfWeek",
          timezone: "UTC"
        }
      }
    }
  }
];
var results = db.timesheets.aggregate(pipeline).limit(10);
```

**R4** — etapa `fields` — motivo: campo inexistente: nomeProjeto — `workspace/atlas_wks.projects` · lookup_join

```js
const resultado = db.projects.aggregate([
  { $lookup: { from: 'members', localField: 'owner', foreignField: '_id', as: 'responsavel' } },
  { $unwind: '$responsavel' },
  { $match: { 'responsavel.role': 'manager' } },
  { $project: { nomeProjeto: 1, nomeResponsavel: '$responsavel.name', status: 1 } }
]).toArray();
```

**R5** — etapa `syntax` — motivo: Unexpected token '.' — `finance/juniper_fin.cotacoes` · nested_paths

```js
printjson(db['cotacoes'].find({meta.quota.extra.habilitado: true, 'meta.quota.extra.tamanho': {$gt: NumberLong(1000000)}}).limit(2).toArray())
```

**R6** — etapa `exec` — motivo: error: 168: Invalid $project :: caused by :: Unrecognized expression '$exists' — `iot/v3_drift_iot.firmwares_v2` · conditional_expr

```js
const coll = db.firmwares_v2; coll.aggregate([{
  $project: {
    version: 1,
    is_mandatory: {
      $cond: [{
        $eq: ['$mandatory', true]
      }, 'Yes', 'No']
    },
    override_info: {
      $switch: {
        branches: [
          { case: { $exists: ['$override'] }, then: '$override.by' },
          { case: true, then: 'No override' }
        ],
        default: 'N/A'
      }
    }
  }
}]).pretty();
```

**R7** — etapa `fields` — motivo: campo inexistente: produto_id — `commerce/kestrel_com.avaliacoes` · sort_group_limit_chain

```js
db.avaliacoes.aggregate([
  { $match: { stars: { $gte: 3 } } },
  { $group: { _id: "$produto_id", count: { $sum: 1 } } },
  { $sort: { count: -1 } },
  { $limit: 5 },
  { $project: { produto_id: 1, count: 1, _id: 0 } }
]).toArray(); // top 5 produtos com mais avaliações positivas
```

**R8** — etapa `syntax` — motivo: Unexpected token ':' — `saas/willow_saas.Users` · lookup_join

```js
const coll = db.Users; coll.aggregate([{$lookup:from:'AuditEvents',localField:'_id',foreignField:'Actor',as:'audits'},{$unwind:'$audits'},{$match:{$expr:{$eq:['$audits.Actor', '$$ROOT._id']}}},{$project:{_id:1, auditAction:'$audits.Action', auditIp:'$audits.Ip'}}]).pretty();
```

**R9** — etapa `exec` — motivo: error: 40104: Unknown argument to $graphLookup: localField — `gaming/harbor_gam.sessoes` · graph_hierarchy

```js
const coll = db.sessoes;
const pipeline = [
  { $graphLookup: {
    from: 'sessoes',
    localField: 'jogador_id',
    foreignField: 'jogador_id',
    as: 'historico_sessoes',
    maxDepth: 1
  } },
  { $project: { _id: 0, jogador_id: 1, historico_sessoes: 1 } }
];
const result = coll.aggregate(pipeline).toArray();
```

**R10** — etapa `fields` — motivo: campo inexistente: permissions.cloud — `retail-pos/v3_delta_pos.operadores_v2` · nested_paths

```js
db.operadores_v2.find({
  'permissions.cloud': true,
  'papel': 'supervisor'
}, {
  _id: 0,
  nome: 1,
  papel: 1,
  'permissions.cloud': 1
})
```

## 11. Estimativas de horas e volume

Base: vazão e taxas medidas neste piloto (vale para 8 slots; ver §12 para sensibilidade).

| alvo | consultas brutas | geração (GPU) | validação (CPU, paralela) | cortes FIM |
|---|---:|---:|---:|---:|
| 60,000 aceitas (J18, treino) | 79,651 | 27 h | 5 h | 323,843 |
| homologação (J19, ≥ 5.000 cortes) | 1,442 (≈ 1,000 aceitas; regra: ≥ 1.000 consultas para ≥ 5.000 cortes com teto de 14 cortes/consulta e ≥ 50 por célula cut_kind × context_kind; o J19 pede 2.000 aceitas por folga) | 0.6 h | 0.1 h | ≥ 5.000 |

## 12. Limitações, riscos e decisões

- **Viés Qwen:** autor e alunos são da mesma família; a homologação mede estilo/estrutura novos, não consultas humanas. Estilos/idiomas “típicos de Qwen” aparecem nos dois lados.
- **Execução ≠ correção semântica:** uma consulta que executa e devolve dados pode não fazer o que a tarefa pedida diz; o filtro é de executabilidade e de campos, não de intenção.
- **Vazio:** consultas com valores inventados tendem a retornar vazio; mantido ≤ 25% e medido por tarefa.
- **Campos:** o checador usa o plano real enviado ao servidor (variáveis/funções já resolvidas) e o fluxo de campos por estágio; é conservador (pode recusar consultas válidas após `$replaceRoot`/`$unionWith`, que viram permissivas).
- **Cobertura da execução:** consultas que nunca iteram o cursor (`no_exec`) e scripts que dependem de estado entre blocos são recusados; `$search`/Atlas e `$text` ficam fora.
- **Decisões do usuário:** (1) liberar J18 (volume do treino) e J19 (homologação); (2) alvo de aceitas (60k) vs. horas de GPU; (3) usar `Qwen3-14B` também no filtro/professor? (4) ativar `run_guarded.py` nos jobs de treino existentes (guard sem editar os geradores).
