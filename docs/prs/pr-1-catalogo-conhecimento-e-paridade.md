> **Estado reconciliado em 26/10/2026:** especificacao historica; consultar codigo/ADRs e aceites atuais da IDE antes de implementar. PR1 atendido no aceite da IDE; PR2 contrato local-agent-qwen3-v1; PR3 com pendencias Atlas e paridade externa. EOS sera corrigido na IDE, sem eosfix. Cache CPU fica opt-in ate homologacao; GPU exige avaliacao de atualizacao GenAI, sem adotar versao por suposicao. [Evidencias e limites](../../reports/student/correcoes-coordenadas-20261026.md).

# PR 1 — Export do catálogo de conhecimento e prova de paridade do contrato FIM (2.000 registros)

> Especificação para outra IA implementar no repositório do Kapibara Studio. É autocontida: leia-a inteira antes de começar.
> Escrita em 2026-10-10 sobre o checkout `F:\source\esilva-silva\EsilvaSoft.KapibaraStudio`, HEAD `0dbb4dd`.
> Objetivo de negócio: **destravar a Fase 03** (dataset de autocomplete) do plano de treinamento em `C:\IA\models\KapibaraStudio.MongoIA\docs\`.

## 1. Contexto

O Kapibara Studio é uma IDE .NET 10 (Avalonia) para MongoDB, com autocomplete local por modelos ONNX (família KapiCoder-Mongo, base Qwen2.5-Coder). Um pipeline Python **externo** à IDE (`C:\IA\models\KapibaraStudio.MongoIA`) vai gerar o dataset de treino do autocomplete. Para o modelo aprender o que a IDE realmente envia, duas coisas precisam existir:

1. **Os geradores Python precisam do vocabulário real do editor** (operadores, estágios de agregação, métodos do console/mongosh, construtores EJSON, operadores Atlas Search, DSL) para não inventar APIs. Hoje esse vocabulário está dentro da IDE (`src/EsilvaSoft.KapibaraStudio.Autocomplete.Core/mongodb-language.v1.json`, `KnowledgeCatalog`, `MongoSyntaxVocabulary`) e **não é exportável**: o `kapilab samples generate` usa os catálogos internamente, mas não os publica, e seu README diz que as amostras "não constituem dataset de treinamento".
2. **O renderer Python do prompt precisa ser comprovadamente idêntico ao da IDE.** O contrato é `editor-context-v1` (cabeçalho com CRLF + `ModelPrefix` + FIM do Qwen). O KapiLab já renderiza (`contract render`, saída só com **hashes**), tokeniza pelo adapter real (`contract tokenize`) e compara (`contract diff`, código 11). Falta a **prova em 2.000 registros reais** pedida pelo item F7B-05 (`docs/phases/phase-07b-kapilab/rastreabilidade-f7b.md`), que hoje está "Parcial".

## 2. Regras do repositório (obrigatórias)

- Leia `CLAUDE.md`, `AGENTS.md` da raiz e o `AGENTS.md` de cada diretório antes de editá-lo.
- Documentação e mensagens em **pt-BR**; identificadores em inglês.
- Não mexer em autenticação, credenciais ou permissões do Claude/Copilot; não ler arquivos de credenciais.
- KapiLab fica **fora da solução principal**; build/teste dele pelo wrapper: `./scripts/lab/build_kapilab.ps1 -Backend Cpu -Test` (e `-Backend WinML`). Restore em `--locked-mode`; lockfiles por backend. Não adicione pacote NuGet sem atualizar os lockfiles corretos.
- Saídas de arquivo do KapiLab ficam restritas a `data/lab`, `reports/lab` ou `tmp` do workspace; o KapiLab recusa o conjunto cego, links/junctions e segredos — preserve essas proteções.
- Mudanças em produto (`src/`) só se forem aditivas e sem alterar comportamento do autocomplete. Prefira expor por API pública já existente.
- Registre o andamento em `docs/phases/phase-07b-kapilab/rastreabilidade-f7b.md` (F7B-05 e F7B-11) e evidências em `docs/phases/phase-07b-kapilab/evidencias-locais.md`.

## 3. Entregáveis

### 3.1 `kapilab catalog knowledge export`

Novo comando no KapiLab (`tools/KapiLab/Cli` + `tools/KapiLab/Core`):

```text
kapilab --workspace <ws> catalog knowledge export [--dialect mql|console|dsl|all] [--out data/lab/knowledge-catalog.json]
```

- Fonte: o **mesmo** catálogo que o editor usa (`KnowledgeCatalog` composto pelas `ICatalogSource` reais — `LanguageCatalogSource` sobre `mongodb-language.v1.json` — e `MongoSyntaxVocabulary`). Não reler o JSON por fora nem duplicar listas.
- Schema `kapilab-knowledge-catalog-v1`:

```json
{
  "schema": "kapilab-knowledge-catalog-v1",
  "source": { "languageFile": "mongodb-language.v1.json", "languageSha256": "…", "ideCommit": "…", "ideDirty": false },
  "symbols": [
    { "id": "op.$gte", "kind": "QueryOperator", "name": "$gte", "category": "comparison", "dialects": ["Mql","Console"],
      "detail": "…", "valueShape": "…", "parameters": [], "returns": null, "snippet": "…", "applicableTypes": [], "since": "…",
      "flags": ["…"] }
  ],
  "vocabulary": {
    "functions": ["…"], "operators": ["…"], "aggregationStages": ["…"], "atlasSearchOperators": ["…"],
    "extendedJsonTypes": ["…"], "keywords": ["…"], "dslFunctions": ["…"], "dslRoots": ["…"]
  },
  "counts": { "symbols": 0, "byKind": { } },
  "snapshotSha256": "…"
}
```

- Campos espelham `CatalogSymbol` (`Id, Kind, Name, Detail, Category, Dialects, Flags, ValueShape, Parameters, Returns, Snippet, ApplicableTypes, Since`). **Não** exportar `Field`/`Scope` com dados de usuário nem metadados de conexão (só conhecimento estático da linguagem).
- Ordenação estável (por `kind`, depois `name`, depois `id`); JSON com escrita atômica; `snapshotSha256` = SHA-256 do conteúdo serializado sem o próprio campo.
- `--dialect` filtra; padrão `all`.
- Documentar no `tools/KapiLab/README.md` (seção de comandos) e no runbook.

### 3.2 `kapilab catalog knowledge check`

```text
kapilab catalog knowledge check --expected <a.json> --actual <b.json>
```

Compara dois snapshots; retorna **código 7** se símbolos foram adicionados/removidos/alterados (mesma convenção do `catalog check` de tools), listando as diferenças por `id`. Serve para o pipeline Python bloquear treino quando o catálogo da IDE mudar.

### 3.3 Prova de paridade do contrato `editor-context-v1` em 2.000 registros (fecha F7B-05)

Objetivo: demonstrar que o renderer Python do pipeline externo produz **o mesmo prompt e os mesmos IDs** que a IDE.

- **Corpus:** `C:\IA\models\KapibaraStudio.MongoIA\bakeoff\data\v1_benchmark.jsonl` (2.000 registros reais do benchmark v1: `fim`, `completion`, `chat`, `correction`; campos `prefix`, `suffix`, `context`, `completion`, `chat_context`…). Use os 1.300 registros `fim`+`completion` para `editor-context-v1` e os 700 `chat`+`correction` para `local-agent-fim-v0` (se o renderer legado aceitar o formato; senão, documente e use só os 1.300 + gere sintéticos até 2.000 com `samples generate`).
- **Referência Python:** `C:\IA\models\KapibaraStudio.MongoIA\bakeoff\v1\scripts\compact\contracts.py` (`prompt_text`, `prompt_ids`, `model_prefix`, `QwenFim`), cópia do pipeline v1 — **somente leitura**; não copie esse código para a IDE.
- **Pacote de tokenizer:** `C:\IA\models\KapibaraStudio.MongoIA\BaseRepos\Qwen2.5-Coder-0.5B` (safetensors + tokenizer HF; o `contract tokenize` exige um pacote ONNX válido — se não houver um pacote ONNX Qwen2.5-Coder exportado, use um dos pacotes ONNX já existentes na máquina, p.ex. `C:\IA\Qwen2.5-Coder-0.5B-onnx-int4-cpu`, e registre o hash do `tokenizer.json`; o tokenizer é idêntico em toda a família Qwen2.5-Coder).
- **Procedimento** (script Python em `scripts/lab/parity_editor_context_v1.py`, usando o wrapper `scripts/lab/kapilab.py`):
  1. Converter os registros para o formato de entrada do `contract render` (documente o mapeamento de campos).
  2. `kapilab contract render --contract editor-context-v1` → hashes de prefixo/sufixo por `id`.
  3. Renderizar os mesmos registros com o Python de referência e calcular os mesmos hashes (mesma normalização de bytes UTF-8, **CRLF preservado**).
  4. `kapilab contract tokenize --package <pkg> --context-tokens 2048 --completion-tokens 32` → IDs da IDE.
  5. Tokenizar com o HF tokenizer (`transformers.AutoTokenizer`) + `prompt_ids` do Python → `kapilab-tokenized-contract-v2` equivalente.
  6. `kapilab contract diff` → **0 divergências** exigidas; qualquer divergência retorna 11 e deve ser analisada (corrigir o lado errado ou documentar a diferença como decisão).
- **Saídas:** `reports/lab/parity-editor-context-v1/{render-ide.jsonl, render-py.jsonl, tokens-ide.jsonl, tokens-py.jsonl, diff.json, summary.md}` com hashes dos insumos, versão do GenAI, commit da IDE.
- O script não pode imprimir prompts brutos no log (só hashes), coerente com a política de privacidade do KapiLab.

## 4. Critérios de aceite

- [ ] `catalog knowledge export` gera `kapilab-knowledge-catalog-v1` determinístico (duas execuções → mesmo `snapshotSha256`), cobrindo todos os `SymbolKind` presentes em `mongodb-language.v1.json`; teste com contagem por `kind` igual à do catálogo carregado pelo editor.
- [ ] `catalog knowledge check` retorna 0 sem mudança e 7 com mudança (teste com snapshot alterado).
- [ ] Nenhum dado de conexão/usuário/schema real no export (teste com catálogo que inclui fontes de metadata → só fontes estáticas aparecem).
- [ ] Paridade: **2.000 registros, 0 divergências** de hash de prompt e de IDs (`contract diff` = 0), ou divergências explicadas e corrigidas; `summary.md` com os números.
- [ ] Testes novos em `tools/KapiLab.Tests` passam em Cpu e WinML; `python -m unittest discover -s scripts/lab/tests -v` passa.
- [ ] README do KapiLab, runbook, `rastreabilidade-f7b.md` (F7B-05 → aceito com evidência; F7B-11 atualizado) e `evidencias-locais.md` atualizados.

## 5. Fora de escopo

- Novo contrato `kapi-context-v2` (decisão: **manter `editor-context-v1`**).
- Execução de queries no MongoDB (fica no pipeline Python).
- Qualquer mudança no comportamento do autocomplete do produto.

## 6. Observações e riscos

- O VSTest está bloqueado pela conexão com o testhost neste ambiente (registrado na rastreabilidade da Fase 7A). Se persistir, registre a falha e use o caminho de teste que já funcionou para o KapiLab (o README cita suítes Cpu/WinML aprovadas) — não marque aceite sem evidência executada.
- Se o renderer legado `local-agent-fim-v0` não aceitar os registros de chat da v1, não force: documente e complete os 2.000 com sintéticos.
