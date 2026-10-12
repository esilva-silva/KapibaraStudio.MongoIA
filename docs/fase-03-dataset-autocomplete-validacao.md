# Fase 03 — Dataset de autocomplete, chat curto e validação

**Revisão 2026-10-08b — troca do professor:** o professor passa a ser o Qwen2.5-Coder-14B Base (alt. 7B), Apache-2.0 (KapiTeacher-Mongo-14B/7B), com o mesmo tokenizer dos students; os dados do professor/T2 são renderizados no FIM Qwen (`QwenFimPromptBuilder`, igual aos students), sai a renderização DeepSeek, o ContractProbe confere só o caminho Qwen, `tokenizers` perde `deepseek-coder`, e o *scoring*/top-k do pool do T1 passa a 34 h (14B) / 16 h (7B) [H]. DeepSeek-6.7B fica só como controle histórico.

> Depende de: [Fase 00](fase-00-visao-premissas.md), [Fase 01](fase-01-selecao-modelos-base.md) (tokenizers dos students por tier).
> Alimenta: [Fase 02](fase-02-professor.md) (treino e gates do professor), [Fase 04](fase-04-dataset-agente-tools.md) (schemas, fixtures e validadores reutilizados), [Fase 05](fase-05-treinamento.md) (dados por tier), [Fase 06](fase-06-avaliacao-benchmarks.md) (`test`, `bench`, `blind`).

Convenção: **[F]** = fato verificado (fonte citada) · **[H]** = hipótese/proposta a validar.


> **Revisão 2026-10-10 — MongoDB no WSL:** a execução real usa o **MongoDB Community 8.0.32 já instalado no WSL (Ubuntu)**, acessível do Windows em `mongodb://127.0.0.1:27017` (serviço `mongod`, `mongosh` 2.13), no lugar do Docker previsto. Fixtures vão em bancos próprios com prefixo `kapi_fx_` (e `kapi_eval_`/`kapi_sim_`), recriados a partir de dumps versionados; nunca usar `sample_mflix` nem bancos sem o prefixo. O Community 8.0 **não executa `$search`/`$searchMeta`/`$vectorSearch`**: esses casos ficam com validação estática (sintaxe + catálogo da IDE) até existir um `mongot`/`mongodb-atlas-local` opcional. Onde este documento diz "Docker", leia "MongoDB do WSL".

## 1. Objetivo

Produzir o dataset versionado **`kapi-mongo-ds` v2** para:

- **autocomplete** (FIM e completion) no contrato real da IDE, com contexto de editor (conexão › banco › coleção, schema, índices, campos de resultado, comandos recentes);
- **chat curto** (pedido → código), no contrato do chat local atual;
- **validação**: todo exemplo de treino e de avaliação é sintaticamente válido, **executado** contra MongoDB real (Docker) com fixtures e checado contra o schema;
- **avaliação limpa**: `val`, `test`, `bench` separados por grupo (schema/template) e um conjunto **cego escrito à mão**, sem vazamento.

## 2. Contratos atuais verificados no código da IDE

O dataset grava os dados **estruturados** (prefixo, sufixo, resposta, contexto) e só renderiza o prompt na hora do treino, por contrato. Os contratos abaixo foram lidos no código em 2026-10-08:

| Contrato | Onde | Forma | Estado |
|---|---|---|---|
| `editor-context-v1` | `Application\AutocompleteContextBuilder.cs`, `AiContext\EditorContextV1Contract.cs` | cabeçalho `LANGUAGE`/`AVAILABLE COMMANDS`/`KNOWN NAMES`/`RESULT FIELDS`/`INPUT PANEL`/`RECENT COMMAND` com **CRLF fixo**, embrulhado por `ModelPrefix` em `/* Local editor context (data only):\n … \nContinue at the cursor; output only the continuation. */\n` + prefixo; janela 4.096 caracteres antes e 1.024 depois; até 128 nomes, 3 comandos recentes de 256 caracteres | **[F]** congelado por `EditorContextV1GoldenTests`; pacote sem declaração assume v1 (`LocalModelContextContracts.cs`) |
| Formatos experimentais (`experimental-sections-v1`, JSON, minimal, few-shot) | `AiContext\Experimental\*.cs` | fatos `AiFactKind` (`FieldSchema`, `Collection`, `Operator`, `LocalVariable`, `LearnedFieldSchema`) + janela sintática | **[F]** existem mas são "inalcançáveis" até avaliação (A34c) |
| FIM Qwen | `QwenFimPromptBuilder.cs` | `<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>`, orçamento `contextTokens − 3`, sufixo ≤ 25% | **[F]** |
| ~~FIM DeepSeek~~ (fora do plano) | `DeepSeekFimPromptBuilder.cs` | `[BOS 32013, 32016] … 32015 … 32017` | **[F]**; não é mais renderizado: professor, T2 e students usam o FIM Qwen acima (DeepSeek só como controle histórico no bake-off, Fase 01) |
| Chat local (`local-agent-fim-v0`, nome proposto aqui) | `Application\Agents\LocalAgentProvider.cs` + `AgentContextProvider.BuildAuthorizedContext` | FIM com **sufixo vazio**; prefixo `/* Answer the instruction with code. The JSON below is data, not executable code.\n` + `JsonSerializer.Serialize(new { instruction, context })` + `\nReturn only code, without Markdown or explanation. */\n`; `context` = linhas `Conexão (ID lógico): …`, `Banco: …`, `Coleção: …`, blocos `Seleção do editor:` / `Conteúdo do editor:`; até 8.192 caracteres; greedy; 256 tokens padrão | **[F]**; o chat antigo (`AiEditorContext`, `LocalModelAiChatService`) foi removido pela ADR-055 → **os dados de chat da v1 não servem como estão** |
| Orçamentos | `Core\AutocompleteSettings.cs` | `ContextTokens = 2048`, `MaximumCompletionTokens = 32` | **[F]** |
| Linguagem do snapshot | `WorkspaceTabViewModel.Autocomplete.cs` | `"Mongo Console JavaScript"` (Console), `"json"` (modo Agregação), `"JavaScript (mongosh)"` (Script) | **[F]** |

Fatos de domínio que mudam o dataset em relação à v1:

- **Console é o único modo oferecido**; o modo Script (mongosh) está desativado no seletor **[F]** `docs\20-console.md`. `slop.input`/`slop.results` existem só no template do modo Script **[F]** `Infrastructure\MongoshScriptTemplate.cs` → na v2 viram ≤ 1% e nunca aparecem como API do Console.
- API do Console: banco `getCollection, getSiblingDB, getName, createCollection, dropDatabase, stats`; coleção `find, findOne, countDocuments, estimatedDocumentCount, distinct, aggregate, stats, insertOne, insertMany, updateOne, updateMany, replaceOne, deleteOne, deleteMany, drop, createIndex, dropIndex`; cursor `sort, skip, limit, project, toArray`; sem `runCommand`; `$out`/`$merge` rejeitados no cursor de leitura **[F]** `docs\20-console.md`.
- Globais: `db`, `getConnection`, `ConnectionPool`, `ENV`, `console`, `EJSON`, `ObjectId`, `NumberLong`, `NumberDecimal`, `UUID/CGUUID/JUUID/GUUID` (exigem texto), `Date` com proxy **[F]** `Infrastructure\ConsoleBootstrap.cs`.
- Catálogo embutido `mongodb-language.v1.json`: 47 grupos, 13 snippets, 42 shapes **[F]** `src\EsilvaSoft.KapibaraStudio.Autocomplete.Core\mongodb-language.v1.json`.
- Papéis de cursor (`StatementStart`, `MemberAccess`, `IndexerString`, `CallArgument`, `PropertyKey`, `PropertyKeyString`, `PropertyValue`, `ArrayElement`, `StringArgument`, `FieldReferenceString`, `VariableReferenceString`) **[F]** `docs\auto-complite\context-engine.md`.

**Contrato de contexto para os novos students [H]:** treinar com dados renderizáveis em `editor-context-v1` **e** num contrato v2 candidato (`kapi-context-v2`, escolhido entre os formatos experimentais pela avaliação A34c da IDE, com participação da [Fase 06](fase-06-avaliacao-benchmarks.md)). Enquanto a IDE não declarar v2 em `LocalModelContextContracts.Supported`, o pacote publicado precisa funcionar em v1. Por isso os registros guardam **fatos estruturados**, não texto renderizado.

## 3. Entradas e fontes (com licença)

| Fonte | Uso | Licença | Decisão |
|---|---|---|---|
| Repositório da IDE (catálogo, shapes, docs `20-console.md`, `06-editor-bson-e-uuid.md`, corpus golden `tests\…\Ai\EditorContextV1Corpus.cs`) | gramática dos geradores, lista de API, goldens de contrato, exemplos curados | MIT **[F]** `docs\20-console.md` ("EsilvaSoft.KapibaraStudio continua MIT") | **usar** (snapshot por commit) |
| Pipeline v1 (`SlopStudio.MongoAI-Compact\datasets\source\*.jsonl`, 61.316 snippets do professor v1 + 47.270 sintéticos) | snippets reaproveitados **após revalidação** pelos validadores v2 | dados próprios | **usar** com marcação `origin: v1:*`; chat v1 só como fonte de instruções a re-renderizar |
| Conjuntos escritos à mão da v1 (`chat_handwritten*.jsonl`: 80 + 40 cegos) | regressão em `eval/legacy-v1/`, re-renderizados no contrato atual | próprios | **só avaliação** |
| Documentação oficial MongoDB (`github.com/mongodb/docs`) | consulta humana para escrever geradores e checar comportamento; **não** entra como texto de treino | CC BY-NC-SA 3.0 **[F]** <https://github.com/mongodb/docs> (link para `creativecommons.org/licenses/by-nc-sa/3.0/`) | **não copiar** para o corpus (cláusula NC/SA incompatível com distribuição do modelo **[H]** — confirmar com jurídico); nomes/assinaturas de operadores são fatos e vêm do catálogo da IDE |
| MongoDB Server em Docker (`mongo:8.0.x`, digest fixado) | execução real dos exemplos | SSPL v1 (<https://www.mongodb.com/legal/licensing/server-side-public-license>) | **usar** só como executor local, sem redistribuição **[H]** |
| `mongodb/mongodb-atlas-local` (tag fixada) | `$search`, `$vectorSearch`, `$searchMeta` com `mongot` | imagem para desenvolvimento/teste, nó único **[F]** <https://www.mongodb.com/docs/search/self-managed/current/installation/quick-start/> | **usar** para validar Atlas Search |
| mongosh | validação do dialeto mongosh (≤ 10% dos dados) | Apache-2.0 (<https://github.com/mongodb-js/mongosh>) | **usar** como executor |
| `pymongo`/`bson.json_util` | gerar EJSON canônico/relaxado | Apache-2.0 | **usar** |
| Faker (locales `pt_BR`, `en_US`) | valores sintéticos | MIT | **usar** |
| Professor auxiliar instruct local (Qwen3-14B em `/no_think`, GGUF Q4_K_M inteiro na VRAM; alt. Qwen3-Coder-30B-A3B-Instruct com experts na RAM — ver [Fase 02 §3.4](fase-02-professor.md)) | paráfrases e back-translation de pedidos | Apache-2.0 | **usar condicionalmente** (gate G2.6), sempre validado; sem GPU na nuvem |
| Datasets de exemplo da MongoDB (`sample_mflix` etc.) | — | termos não verificados | **não usar** |
| Drivers (C#, Node, Python) | fora do escopo do editor (Console JS) | Apache-2.0 | **não usar** em v2.0 |

Entregável: `data/sources/licenses.md` com cada fonte, licença, URL/commit e decisão.

## 4. Atividades

### 4.1 Snapshot dos contratos e da IDE (dias 1–2)

- [ ] `scripts/data/snapshot_ide.py --ide F:\source\esilva-silva\EsilvaSoft.KapibaraStudio` copia (somente leitura) para `data/sources/ide-snapshot/<ide_commit>/`: `mongodb-language.v1.json`, lista de API do Console extraída de `ConsoleBootstrap.cs`, corpus golden v1, e grava os hashes.
- [ ] `tools/ContractProbe` (.NET 10, console, fora do repositório da IDE, referenciando por `HintPath` as DLLs de `Application`, `Autocomplete.Core`, `LocalAi.Core` compiladas a partir do commit do snapshot):
  - `render --contract editor-context-v1|experimental-* --in records.jsonl` → texto exato do `ModelPrefix`/sufixo;
  - `render-chat` → prefixo do `LocalAgentProvider` (o método é privado: reproduzir e cobrir com teste de paridade contra `LocalAgentProviderTests`);
  - `tokenize --model <pasta>` → IDs pelo tokenizer .NET usado pela IDE (**só o caminho Qwen** `QwenFimPromptBuilder`/`qwen2`, que cobre students T0–T3 **e** o professor/T2 KapiTeacher; sem caminho DeepSeek);
  - `parse` → Acornima (o mesmo parser do Console).
- [ ] `scripts/data/render_contract.py` (Python, usado no treino) com teste de paridade byte a byte contra o ContractProbe em 2.000 registros por contrato. Detalhes a cobrir: CRLF fixo do cabeçalho v1 × `\n` literal do `ModelPrefix`; troca de `*/` por `* /`; `AVAILABLE COMMANDS` por linguagem; ~~variante `deepSeekTrainingContract`~~ (removida; renderização única no FIM Qwen para professor e students); escape de `JsonSerializer` padrão (não-ASCII como `\uXXXX`, aspas e `<>&'+` escapados) no chat.

### 4.2 Schemas sintéticos e fixtures (dias 2–5)

- [ ] `scripts/data/build_schemas.py` → `data/schemas/<domain>/<schema_id>.json`.
  - 14 domínios: commerce, saas, logistics, iot, hr, content, finance, travel, workspace, healthcare-sintético, gaming, energy; **holdout**: education, legal (só `test`/`bench`).
  - 6–10 schemas por domínio, 4–12 coleções cada; documentos aninhados (até 4 níveis), arrays de subdocumentos, tipos BSON variados (ObjectId, UUID subtipo 4 e 3, Date, Decimal128, Int64, Int32, Double, bool, null, string, arrays), campos opcionais (presença 30–100%).
  - Estilos de nome: camelCase, PascalCase, snake_case, pt-BR sem acento, coleções com hífen/espaço (força `db["x-y"]`, `getCollection("…")`, `ConnectionPool["Prod Cluster"]["Company-Db"]`).
  - Índices (simples, compostos, multikey, TTL, únicos, parciais), `$jsonSchema` em ~30% das coleções com `enum`, índices de Atlas Search/Vector em ~20%.
- [ ] `scripts/data/seed_fixtures.py --seed 20261008` → 200–2.000 documentos por coleção, determinístico; valores calibrados para que filtros típicos retornem **algum, mas não todo** documento (necessário para a equivalência semântica).
- [ ] `data/fixtures/docker-compose.yml`: `mongo:8.0.x@sha256:…` (porta 27117) e `mongodb/mongodb-atlas-local:<tag>@sha256:…` (porta 27118), só em `127.0.0.1`; dumps `mongodump --archive --gzip` por schema para restaurar em segundos.

### 4.3 Geradores (dias 4–12)

| Script | Saída | Descrição |
|---|---|---|
| `gen_snippets.py` | `raw/<ver>/snippets.jsonl.zst` | amostrador por gramática que caminha os **shapes** do catálogo (`Filter`, `Projection`, `Sort`, `Update`, `Pipeline`, `Stage`, `LookupBody`, …) tipado pelo schema: só usa campos que existem, operadores compatíveis com o tipo, estágios com a semântica de campos de saída (`$group`, `$project`, `$lookup.as`, `$unwind`). Renderiza em Console (`db.c`, `db.getCollection`, `getConnection().getDatabase().getCollection()`, `ConnectionPool.a.b.c`, `const x = …`), JSON (modo Agregação) e mongosh (≤ 10%). Variações de formatação: indentação 2/4, aspas simples/duplas, chaves com/sem aspas, CRLF em 40–70% |
| `gen_fim.py` | `generated/<ver>/fim.jsonl.zst`, `completion.jsonl.zst` | cortes por AST (Acornima via ContractProbe ou `acorn` no Node) — tabela abaixo |
| `gen_editor_context.py` | campo `editor` + `facts` | simula o snapshot: `known_names` (conexões, bancos, coleções do explorer), `result_fields` (até 128 nomes do resultado nos fixtures), `recent_commands` (≤ 3, mesma conexão/banco), `input_panel` (raro), fatos `Target`, `SchemaSubset`, `OperatorSubset`, `Indexes`, `LocalDeclarations`, `SimilarStatement`; 8% sem contexto; 10% com nomes irrelevantes de outras coleções |
| `gen_chat.py` | `generated/<ver>/chat.jsonl.zst` | intents `create` (filtro, últimos N dias, contagem, agrupamento, top N, distinct, lookup, Atlas Search, índice, update, delete, insert, troca de conexão, EJSON), `modify` (filtro, ordenação/limite, projeção, find→aggregate, conexão), `explain` (comentário `//` + código, pois o contrato pede só código), `correction`; 50% pt-BR (15% sem acento, 5% com erro de digitação, registro coloquial), 50% en; valores citados aparecem literalmente no código |
| `gen_contrastive.py` | parte do chat | **pares contrastivos** no mesmo grupo para atacar as falhas da v1 (condição invertida/omitida, parte do pedido ignorada **[F]** v1 README): "pagos" × "não pagos", "exceto cancelados", "e ordena por data", "só os 5 primeiros" |
| `aux_paraphrase.py` | parte do chat | (opcional, [Fase 02 §3.4](fase-02-professor.md)) back-translation query → pedido e paráfrases; ≤ 30% do chat; mesmo validador |
| `gen_negatives.py` | `correction.jsonl.zst`, `pref.jsonl.zst` | APIs inexistentes e erros comuns → forma correta (tabela abaixo); também pares `chosen`/`rejected` para DPO opcional da Fase 05 |

Pontos de corte FIM (alvo de distribuição no `train`, ±2 p.p.):

| Estratégia | Exemplo do cursor `▌` | Alvo |
|---|---|---:|
| Chave de filtro/projeção (`PropertyKey`, `PropertyKeyString`) | `find({ "Cliente.▌` | 16% |
| Valor tipado (`PropertyValue`), incluindo construtores BSON | `{ createdAt: { $gte: ▌ } }` | 14% |
| Dentro de operador (objeto de operadores, `$expr`, acumuladores) | `{ $sum: "$▌" }`, `{ total: { $▌` | 14% |
| Estágio de pipeline (`ArrayElement`), estágio inteiro multilinha | `aggregate([ { $match: … },\r\n  ▌` | 12% |
| Acesso a membro/navegação (`MemberAccess`, `IndexerString`, `StringArgument`) | `ConnectionPool.dev.▌`, `getCollection("▌` | 10% |
| Token parcial (corte no meio do identificador, sem alinhar ao tokenizer) | `db.Orders.agg▌` | 10% |
| Resto de linha / resto de statement | `.sort({ createdAt: -1 })▌` | 12% |
| Multilinha (objeto/estágio/argumento completo) | corpo de `$lookup` | 8% |
| Início de statement (`StatementStart`) com statements anteriores relevantes | linha nova após `const orders = …` | 4% |

Respostas limitadas a 96 tokens do student (a IDE gera 32 por padrão; treinar até 96 ensina a parar em fronteira natural). Remover exemplos sem código antes **e** depois do cursor (lição v1: 1,3% impossíveis de inferir **[F]**).

APIs inexistentes / erros (negativos e correções):

| Errado | Correto | Fonte |
|---|---|---|
| `ConnectionPull`, aliases PascalCase, `GetDatabase` | `ConnectionPool`, `getDatabase` | **[F]** `docs\21-autocomplete-local.md`, `docs\20-console.md` |
| `.pretty()`, `.count()`, `bulkWrite`, `runCommand` no Console | `toArray()`, `countDocuments`, `insertMany`/`updateMany`; sem `runCommand` | **[F]** API do Console |
| `$out`/`$merge` em leitura no Console | remover; explicar em comentário | **[F]** |
| `ObjectId(123)`, ObjectId com 23 dígitos, `UUID()` sem texto | `ObjectId("…24 hex…")`, `UUID("…")` | **[F]** `ConsoleBootstrap.cs` |
| `NumberLong(9007199254740993)` | `NumberLong("9007199254740993")` | **[F]** |
| Operadores inventados (`$contains`, `$like`, `$between`, `$notIn`, `$join`) | `$regex`/`$in`/`$gte`+`$lte`/`$nin`/`$lookup` | **[H]** lista inicial |
| Campo inexistente no schema | campo correto do schema do contexto | gerado |

### 4.4 Formato JSONL (registro estruturado, independente de contrato)

Registro FIM (`schema_version: kapi-ds-record-v2`):

```json
{
  "id": "fim-commerce.orders@3-agg.group.sum-000417-k1",
  "schema_version": "kapi-ds-record-v2",
  "type": "fim",
  "task": "autocomplete",
  "dialect": "console",
  "language": "Mongo Console JavaScript",
  "domain": "commerce",
  "schema_id": "commerce/orders@3",
  "template_id": "agg.group.sum",
  "group_id": "commerce/orders@3::agg.group.sum",
  "cursor_role": "FieldReferenceString",
  "cut": {"strategy": "operator_value", "multiline": false, "mid_token": false},
  "prefix": "db.getCollection(\"orders\").aggregate([\r\n  { $match: { status: \"paid\" } },\r\n  { $group: { _id: \"$customerId\", revenue: { $sum: \"$",
  "suffix": "\" } } },\r\n  { $sort: { revenue: -1 } }\r\n])",
  "completion": "total",
  "accept": ["total"],
  "editor": {
    "target": {"connection": "dev-cluster", "database": "shop", "collection": "orders", "confidence": "Explicit"},
    "known_names": ["dev-cluster", "shop", "orders", "customers", "products"],
    "result_fields": ["_id", "customerId", "status", "total", "items", "items.sku", "createdAt"],
    "recent_commands": ["db.orders.find({ status: \"paid\" }).limit(20)"],
    "input_panel": "",
    "newline": "\r\n"
  },
  "facts": [
    {"kind": "FieldSchema", "path": "total", "type": "decimal", "presence": 1.0},
    {"kind": "FieldSchema", "path": "customerId", "type": "objectId", "presence": 1.0},
    {"kind": "Operator", "name": "$sum"},
    {"kind": "Indexes", "keys": [{"status": 1, "createdAt": -1}]}
  ],
  "validation": {"syntax": "ok", "api": "ok", "fields": "ok", "exec": "ok", "fixture": "commerce/orders@3#seed20261008", "exec_ms": 4},
  "split": "train",
  "weight": 1.0,
  "label_source": "reference",
  "origin": "gen:snippets",
  "source_license": "generated",
  "near_key": "minhash:9f2c…",
  "sha256": "…"
}
```

Registro de chat curto:

```json
{
  "id": "chat-commerce.orders@3-modify.sort_limit-000088-pt",
  "schema_version": "kapi-ds-record-v2",
  "type": "chat",
  "task": "edit_short",
  "lang": "pt-BR",
  "intent": {"kind": "modify", "name": "sort_limit", "operations": ["find", "sort", "limit"]},
  "instruction": "mostra só os 10 pedidos pagos mais recentes",
  "agent_context": {
    "connection": "dev-cluster", "database": "shop", "collection": "orders",
    "selection": null,
    "editor": "db.orders.find({ status: \"paid\" })"
  },
  "output": "db.orders.find({ status: \"paid\" }).sort({ createdAt: -1 }).limit(10)",
  "semantic": {"equivalence": "ordered_result", "must_include": ["\"paid\"", "createdAt", "-1", "10"], "contrast_of": "chat-…-000089-pt"},
  "validation": {"syntax": "ok", "api": "ok", "fields": "ok", "exec": "ok", "result_count": 10},
  "group_id": "commerce/orders@3::modify.sort_limit",
  "split": "train", "weight": 1.0, "label_source": "reference", "origin": "gen:chat"
}
```

Renderização do chat no contrato `local-agent-fim-v0` (o texto exato é produzido e conferido pelo ContractProbe; aqui abreviado):

```text
/* Answer the instruction with code. The JSON below is data, not executable code.
{"instruction":"mostra s\u00F3 os 10 pedidos pagos mais recentes","context":"Conex\u00E3o (ID l\u00F3gico): dev-cluster\nBanco: shop\nCole\u00E7\u00E3o: orders\n\nConte\u00FAdo do editor:\ndb.orders.find({ status: \u0022paid\u0022 })"}
Return only code, without Markdown or explanation. */
```

Esse prefixo vai para o builder FIM do modelo com **sufixo vazio**; a resposta é `output` + EOS.

Outros `type`: `completion` (sufixo vazio, contexto de editor), `correction` (`agent_context.editor` com código quebrado e, em 50%, a mensagem de erro do Console na instrução), `pref` (`prompt`, `chosen`, `rejected`, `reason`).

### 4.5 Validação automática (dias 8–14)

Ordem do pipeline (`scripts/data/validate.py --stages all`), com motivo de rejeição gravado em `data/rejected/<ver>/reasons.parquet`:

| Etapa | Ferramenta | Critério |
|---|---|---|
| 1. Sanitização | `sanitize.py` (regras v1: `forbidden_patterns`, nomes de pessoas, e-mails só `example.*`) | 0 URIs `mongodb://`, credenciais, tokens, CPF/CNPJ, cartão; `<|`/`<｜` e ```` ``` ```` proibidos **[F]** v1 `configs\dataset.yaml` |
| 2. Sintaxe | ContractProbe `parse` (Acornima) para Console; `acorn` (Node) para mongosh; parser EJSON relaxado para modo `json` | documento reconstruído (prefixo + resposta + sufixo) sem erro novo |
| 3. API | lista de API do snapshot (dialeto Console × mongosh) | nenhum método/global fora da lista; `$out`/`$merge` fora de leitura no Console |
| 4. Campos | `check_fields.py`: caminhada de shapes do catálogo com escopo de campos por estágio | todo caminho referenciado existe no schema ou foi criado antes no pipeline (`$group`, `$project`, `$set`, `$lookup.as`, `let`) |
| 5. Execução | `exec_validate.py` contra Docker, 8 workers, `maxTimeMS` 2.000; escritas em banco clonado por worker (restore do dump) | sem erro do servidor; leituras de referência de chat retornam 1..N−1 documentos (discriminativo) |
| 6. Equivalência (chat/correção) | mesmo executor | resultado da referência = resultado esperado do template (conjunto, ou lista ordenada quando há `sort`); escritas comparadas pelo estado final da coleção |
| 7. Contrato/orçamento | `render_contract.py` + tokenizer do student | prompt cabe em 2.048 tokens (chat inteiro, como `RequireFullContext`); resposta ≤ 96 (autocomplete) / 256 (chat) tokens |
| 8. Dedup | `dedup_minhash.py` (`datasketch`, 128 permutações, shingles de 5 tokens normalizados) | exato por hash normalizado; quase-duplicata Jaccard ≥ 0,85 dentro do split; limite por grupo (como `max_per_group` da v1) |
| 9. Decontaminação | `decontaminate.py` | treino sem sobreposição com `test`/`bench`/`blind`/`legacy-v1`: grupos disjuntos, MinHash ≥ 0,7 removido, 13-gramas de tokens de código e instruções normalizadas sem acento removidos |

Execução de Atlas Search: `$search`/`$vectorSearch` só no container atlas-local, com índices criados pelo seed; vetores sintéticos de dimensão pequena (ex.: 64) gerados por hash determinístico **[H]**.

Executar pelo runtime do próprio Console (ContractProbe `exec`, usando `ConsoleRuntime` com sessão real) é o ideal porque prova "a IDE executaria isso" **[H]** — se a montagem de DI for inviável fora do app, traduzir a DSL para mongosh (`ConnectionPool.a.b.c` → `getSiblingDB`) e registrar a limitação.

### 4.6 Splits, conjunto cego e tamanhos

Separação **por grupo** `group_id = schema_id::template_id`, nunca por exemplo:

| Split | Grupos | Conteúdo | Tamanho alvo |
|---|---|---|---:|
| `train_pool` | 88% dos grupos dos domínios não-holdout | entrada do professor e do treino | ~1,0 M validados (de ~1,5 M candidatos) |
| `val` | 6% | seleção de checkpoint | 10.000 |
| `test` | 3% + 10% dos templates reservados | relatório interno das Fases 02/05 | 5.000 |
| `bench` | 3% + domínios holdout (40% do bench) + templates reservados | entregue à [Fase 06](fase-06-avaliacao-benchmarks.md) | 3.000 (1.800 autocomplete, 900 chat, 300 correção) |
| `blind` | escrito à mão | lido só nos gates finais | 400 |
| `legacy-v1` | 120 pedidos livres da v1 (80 original/estendido + 40 cegos), re-renderizados | comparação com v1 | 120 |

**Conjunto cego (`data/eval/blind/`):**

- [ ] 200 posições de autocomplete em scripts realistas e 200 pedidos de chat (100 pt-BR, 100 en), escritos por pessoa que **não** escreveu templates, sobre 3 schemas exclusivos do blind;
- [ ] respostas de referência executadas nos fixtures e revisadas por uma segunda pessoa; alternativas aceitáveis listadas;
- [ ] arquivo com SHA-256 no manifesto; os geradores recusam ler `data/eval/` (verificação por caminho e por hash);
- [ ] cada leitura registrada em `reports/dataset/blind_access.log` (quem, quando, modelo).

Tamanhos de treino por tier — **revisados para treino 100% local** (sem GPU na nuvem): dimensionados ao orçamento de horas da RX 7800 XT na [Fase 05 §8–§9](fase-05-treinamento.md), amostrados do `train_pool` **[H]**:

| Tier | Autocomplete (FIM + completion) | Chat curto + correção | Tokens/época (≈) | Épocas · GPU | Contrato/contexto | Observação |
|---|---:|---:|---:|---|---|---|
| T0 Nano | 150 mil | 10 mil (robustez) | 62 M | 2 · ~5 h (LoRA) | v1 + v2, ctx 1.024–2.048 | **subconjunto do T1** (um só *scoring*/top-k do professor, mesmo tokenizer Qwen) |
| T1 Small | 180 mil | 50 mil | 100 M | 2 · ~23 h | v1 + v2, ctx 2.048 | autocomplete + chat |
| T2 Pro | = dados do SFT do professor (~105 mil, [Fase 02 §3.3 B](fase-02-professor.md)) | incluído (20% chat + 6% correção) | ~47 M | 1 · ~24 h (na Fase 02) | v1 + v2, ctx 2.048–4.096 | **T2 = KapiTeacher-Mongo-14B** (ou 7B se o alternativo vencer o bake-off; Fase 01); dados renderizados no FIM Qwen; tokens/época ~47 M agora contados com o tokenizer Qwen **[H]** (a contagem de ~47 M vinha do tokenizer DeepSeek de 32k, que gera mais tokens por exemplo de código; reconferir após o render) |
| T3 NPU | = T1 | = T1 | — | — | renderizado para o contexto fixo do export QNN (ex.: 1.024) | depende da [Fase 07](fase-07-exportacao-onnx.md) |
| T4 Agent | 0 (o T4 não faz autocomplete) | 10 mil (⊂ chat do T1) | 7 M | dentro do T4 | + dados de agente da [Fase 04](fase-04-dataset-agente-tools.md) | chat template com tools |

Como o volume cai **sem perder cobertura** **[proposta]**: (1) amostragem estratificada por categoria × papel de cursor × dialeto × idioma, com os alvos de distribuição abaixo; (2) piso de 4% por grupo pequeno (Indexes, Scripts, EJSON, DSL); (3) sobre-amostragem 1,5× dos exemplos `hard` (baixo logprob do professor ou discordância greedy × referência) e corte de quase-duplicatas fáceis; (4) o `train_pool` de ~1,0 M continua existindo (gerar e validar é CPU) e serve de reserva para rodadas futuras. O professor pontua só o pool do T1 (230 mil, ⊇ T0 e T4-chat), não o `train_pool` inteiro (~1,0 M × ~370 tokens ≈ 114 h com o 14B a ~900 tok/s de forward ou ≈ 54 h com o 7B ~1.900 tok/s **[H]**; antes ≈ 52 h com o DeepSeek 6.7B).

Média estimada: ~350 tokens de prompt e ~20 de resposta por exemplo de autocomplete; ~600 e ~60 no chat **[H]**.

Distribuição alvo por categoria no `train` (±2 p.p.): aggregation 24%, crud 16%, operators 10%, atlas_search 12%, json_ejson 10%, console_dsl (conexões, `ConnectionPool`, `ENV`, cursores) 12%, indexes 5%, correção/negativos 6%, chat técnico 5%.

### 4.7 Versionamento e manifesto

- Versão `kapi-mongo-ds-MAJOR.MINOR.PATCH`: MAJOR muda schema de registro ou contrato; MINOR adiciona fontes/geradores; PATCH só filtros.
- Shards `*.jsonl.zst` de 100 mil linhas; o manifesto não aceita arquivo sem hash.
- `scripts/data/build_manifest.py` / `verify_manifest.py`:

```json
{
  "dataset": "kapi-mongo-ds",
  "version": "2.0.0",
  "created_utc": "2026-10-xxT00:00:00Z",
  "seed": 20261008,
  "generator_commit": "<git sha do workspace KapibaraStudio.MongoIA>",
  "ide_snapshot": {"commit": "<sha>", "language_json_sha256": "…", "golden_v1_sha256": "…"},
  "contracts": ["editor-context-v1", "kapi-context-v2-candidate", "local-agent-fim-v0"],
  "tokenizers": {"qwen2.5-coder": "<sha256 tokenizer.json>"},
  "fixtures": {"mongo_image": "mongo@sha256:…", "atlas_local_image": "mongodb/mongodb-atlas-local@sha256:…", "dumps_sha256": "…"},
  "files": [{"path": "splits/2.0.0/train/train-00000.jsonl.zst", "sha256": "…", "records": 100000}],
  "counts": {"train_pool": 0, "val": 0, "test": 0, "bench": 0, "blind": 400},
  "rejections": {"syntax": 0, "api": 0, "fields": 0, "exec": 0, "dedup": 0, "decontam": 0},
  "blind_sha256": "…",
  "licenses": "data/sources/licenses.md"
}
```

- O workspace `C:\IA\models\KapibaraStudio.MongoIA` ainda **não** é repositório git **[F]** → `git init` para código, configs e manifestos; dados fora do git (só o manifesto rastreia).

## 5. Estrutura de pastas e scripts

```text
C:\IA\models\KapibaraStudio.MongoIA\
  configs\dataset.yaml                      # proporções, seeds, limites, domínios holdout
  tools\ContractProbe\                      # .NET 10: render, render-chat, tokenize, parse, exec
  scripts\data\
    snapshot_ide.py  build_schemas.py  seed_fixtures.py
    gen_snippets.py  gen_fim.py  gen_editor_context.py  gen_chat.py  gen_contrastive.py
    gen_negatives.py  aux_paraphrase.py
    render_contract.py  sanitize.py  validate.py  check_fields.py  exec_validate.py
    dedup_minhash.py  decontaminate.py  split_groups.py  sample_tiers.py
    build_manifest.py  verify_manifest.py  report_dataset.py
  data\
    sources\ide-snapshot\<ide_commit>\  sources\v1\  sources\licenses.md
    schemas\<domain>\<schema_id>.json
    fixtures\docker-compose.yml  fixtures\dumps\<schema_id>.archive.gz
    raw\<ver>\  generated\<ver>\  validated\<ver>\  rejected\<ver>\
    splits\<ver>\{train_pool,val,test,bench}\
    tiers\<ver>\{t0,t1,t2,t3,t4}\
    eval\blind\  eval\legacy-v1\
    manifests\kapi-mongo-ds-<ver>.json
  reports\dataset\<ver>\{dataset_report.md, distribution.json, rejections.md, parity_contracts.json, human_review.md}
```

## 6. Critérios de saída (gates)

- [ ] **G3.1 Paridade de contrato:** 100% de igualdade byte a byte (texto) e de IDs (tokenizer .NET × Python) em 2.000 registros por contrato (`editor-context-v1`, candidato v2, `local-agent-fim-v0`).
- [ ] **G3.2 Validade:** 100% de `train_pool`/`val`/`test`/`bench` com `syntax`, `api`, `fields`, `exec` = ok; chat/correção com equivalência = ok.
- [ ] **G3.3 Volume e distribuição:** ≥ 1,0 M validados no `train_pool`; categorias, papéis de cursor, dialetos e idiomas dentro de ±2 p.p. dos alvos.
- [ ] **G3.4 Vazamento:** 0 grupos compartilhados entre splits; 0 pares treino × (`test`/`bench`/`blind`/`legacy-v1`) com MinHash ≥ 0,7 ou 13-grama comum.
- [ ] **G3.5 Privacidade:** 0 ocorrências dos padrões proibidos.
- [ ] **G3.6 Blind:** 400 itens escritos, revisados por segunda pessoa, referências executadas; hash no manifesto.
- [ ] **G3.7 Revisão humana:** 300 exemplos aleatórios do `train` com ≥ 97% corretos e naturais (chat) e 0 APIs inexistentes.
- [ ] **G3.8 Reprodutibilidade:** reconstruir 1 shard com o mesmo seed gera o mesmo SHA-256; `verify_manifest.py` sem erro.
- [ ] **G3.9 Licenças:** `licenses.md` completo; nenhuma fonte NC/sem licença no treino.

## 7. Riscos e mitigação

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Templates demais → chat saturado no benchmark e fraco em pedido livre (repete v1: ~94% programático × 56–72% livre **[F]**) | alta | alto | pares contrastivos, paráfrases validadas, blind escrito à mão como gate, `legacy-v1` |
| Reforço de chat degrada autocomplete (v1.1/v1.2: −0,20 APT em Indexes/Scripts/EJSON **[F]**) | média | alto | proporções fixas por tier; `val` por categoria; nenhum reforço sem medir autocomplete |
| Contrato v2 não decidido a tempo | média | médio | registros estruturados; v1 sempre renderizável |
| Divergência Python × IDE na renderização (CRLF, escape JSON) | média | alto | ContractProbe + G3.1 |
| Execução lenta ou instável no Docker (Windows) | média | médio | dumps restauráveis, 8 workers, timeout, cache por hash de query |
| Atlas Search local muda/limitações do preview | média | baixo | tag fixada por digest; Atlas Search só valida o que roda localmente; resto marcado `exec: skipped` e fora do `bench` |
| Licença da documentação MongoDB | certa (NC) | alto se ignorada | não usar texto da documentação; revisão jurídica do `licenses.md` |
| Vazamento do blind | baixa | alto | pasta isolada, hash, log de acesso, geradores bloqueiam o caminho |
| Valores literais impossíveis de inferir | alta | médio | `accept` com alternativas; filtro por dificuldade com o professor ([Fase 02](fase-02-professor.md)) |

## 8. Dependências

- [Fase 01](fase-01-selecao-modelos-base.md): tokenizers finais por tier (orçamentos e limites em tokens).
- [Fase 02](fase-02-professor.md): scoring/greedy/best-of-N sobre o pool do T1 (230 mil, não o `train_pool` inteiro) e cache top-k (`topk-v1`, KD por logits no plano base para T0/T1; `T1-ds` saiu do plano); professor auxiliar local para paráfrases.
- [Fase 04](fase-04-dataset-agente-tools.md): reutiliza schemas, fixtures, executor e validadores.
- [Fase 06](fase-06-avaliacao-benchmarks.md): consome `test`, `bench`, `blind`, `legacy-v1` e define métricas finais.
- IDE: decisão do contrato v2 (A34c) e build das DLLs para o ContractProbe (somente leitura).

## 9. Estimativa

| Bloco | Esforço humano | Máquina |
|---|---:|---:|
| Snapshot + ContractProbe + paridade | 3–4 dias | — |
| Schemas, fixtures, Docker | 3 dias | ~1 h |
| Geradores (snippets, FIM, contexto, chat, negativos) | 6–8 dias | ~3 h CPU |
| Validação (sintaxe, API, campos, execução) | 4 dias | 3–6 h (≈1,5 M candidatos a 100–300 exec/s **[H]**) |
| Dedup, decontaminação, splits, manifesto | 2 dias | ~2 h |
| Conjunto cego (escrita + revisão) | 4–5 dias (2 pessoas) | — |
| Paráfrases com o professor auxiliar (opcional) | 1 dia | ~5 h de GPU local (Qwen3-14B no llama.cpp; job J07) |
| Rótulos do professor sobre o pool T1 (*scoring*, greedy, best-of-N — executados pela [Fase 02 §3.5](fase-02-professor.md)) | — | ~34 h (14B) / ~16 h (7B) de GPU local numa única passada de forward com cache top-k (J13–J14) **[H]**; antes ~26 h |
| **Total** | **~22–27 dias úteis** (parte em paralelo com a Fase 02) | CPU para gerar/validar; GPU só nos jobs acima, no calendário único da [Fase 05 §9](fase-05-treinamento.md) |
