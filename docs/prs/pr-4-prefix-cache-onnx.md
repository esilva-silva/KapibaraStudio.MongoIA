> **Estado reconciliado em 26/10/2026:** especificacao historica; consultar codigo/ADRs e aceites atuais da IDE antes de implementar. PR1 atendido no aceite da IDE; PR2 contrato local-agent-qwen3-v1; PR3 com pendencias Atlas e paridade externa. EOS sera corrigido na IDE, sem eosfix. Cache CPU fica opt-in ate homologacao; GPU exige avaliacao de atualizacao GenAI, sem adotar versao por suposicao. [Evidencias e limites](../../reports/student/correcoes-coordenadas-20261026.md).

# PR 4 — Sessão por aba com cache de prefixo (KV) no `OnnxLocalModelRuntime` para o autocomplete em CPU

> Especificação (rascunho) para outra IA implementar no repositório do Kapibara Studio. É autocontida: leia-a inteira antes de começar. **Nada foi implementado na IDE**; o checkout de referência só foi lido.
> Escrita em 2026-10-22 sobre o checkout `F:\source\esilva-silva\EsilvaSoft.KapibaraStudio`, HEAD `7844bd7`. Evidência: `C:\IA\models\KapibaraStudio.MongoIA\reports\export\prefix_cache_ttft.md`; protótipo: `scripts\export\prefix_cache_proto.py`.
> Objetivo de negócio: fazer os gates de TTFT em CPU do autocomplete (T0 150/250 ms; T1 500/650 ms) passarem sem trocar de modelo, reaproveitando o KV do prefixo estável do prompt `editor-context-v1` (experimento R43 / AC-15).

## 1. Contexto

O prompt `editor-context-v1` + FIM Qwen tem ~280 tokens (média) e o prefill domina o TTFT em CPU (T0 int4 ≈ 258–300 ms, T1 int8 ≈ 670–750 ms). A estrutura é: `<|fim_prefix|>` + envelope (`LANGUAGE`, `AVAILABLE COMMANDS`, `KNOWN NAMES`, `RESULT FIELDS`, `INPUT PANEL`, `RECENT COMMAND`, fecho `Continue at the cursor…*/`) + prefixo de código + `<|fim_suffix|>` + sufixo + `<|fim_middle|>`. Só o envelope (≈ 71% dos tokens; o fixo por linguagem é 22%) é estável entre pedidos consecutivos de uma aba; o prefixo de código muda a cada tecla mas costuma só crescer no fim.

Hoje `OnnxLocalModelRuntime.Generate` cria um `Generator` por pedido (`max_length = input.Length + MaximumTokens`), chama `AppendTokens(input)` e `GenerateNextToken`. `ModelGenerationRequest.PrefixCache` (`PrefixCacheState.Tokens`) existe e é **inerte** (DEC-R42-PROMPTTOKENS; `AutocompleteMetrics.PrefixCacheReusedTokens` já existe).

No GenAI 0.15.2 (C#) existem `Generator.AppendTokens`, `RewindTo(int)` e `TokenCount()`; **não** existe cópia/transferência de KV entre `Generator`s nem prefix cache automático (Engine 0.17, KV paginado, não exposto). `RewindTo` não funciona com beam search (a IDE usa greedy). Portanto o cache é um **`Generator` persistente por sessão** (por aba), não um KV compartilhado.

Medido (CPU, protótipo Python, n=200, hit estável, média/p95 ms): T0 int4 24 thr sem cache 258/474 → B 92/221, digitação (C) 58/149; T1 int8 667/1101 → B 250/497, C 160/381. Equivalência greedy 100% (500 prompts × 2 tiers); FIM acc/exec-valid inalterados.

## 2. Regras do repositório (obrigatórias)

- Leia `CLAUDE.md`, `AGENTS.md` da raiz e o `AGENTS.md` de cada diretório antes de editá-lo; documentação e mensagens em pt-BR, identificadores em inglês.
- Não mexer em autenticação, credenciais ou permissões do Claude/Copilot.
- Mudança aditiva: o comportamento padrão do autocomplete **não muda** (cache desligado por padrão, AC-15: habilitado por provider só com equivalência greedy aprovada e ganho de TTFT medido). Chat, agente e IA explícita (`ChatPrompt`) nunca usam o cache.
- Não alterar o contrato `editor-context-v1` nem os goldens.
- Registre a decisão (por provider) em `docs/auto-complite/decisions.md` (AC-15) e as métricas em `AutocompleteMetrics`.

## 3. Entregáveis

### 3.1 `PrefixCacheState` / `ModelGenerationRequest`

- Substituir o tipo inerte por: `SessionKey` (string opaca: identifica a aba/documento), `Mode` (`Disabled` padrão | `Reuse`) e `CapTokens` (int?: teto de tokens reaproveitáveis; `null` = LCP inteiro). Manter `Tokens` apenas se algum chamador já o usa. Campos aditivos, nenhum obrigatório.
- O chamador do autocomplete (`AiAutocompleteProvider`/`AiGenerationPipeline`, onde `PromptTokens` é montado) preenche `SessionKey` = id da aba + conexão + banco + coleção + linguagem. O runtime **não** deve inferir a chave do texto.

### 3.2 Sessão por aba em `OnnxLocalModelRuntime`

- Novo tipo interno `PrefixSession` (dono de um `Generator` criado com `max_length = ContextTokens` (2.048), os ids do KV válido `List<int> kv`, `LastUsed`). Dicionário `SessionKey → PrefixSession` com **LRU (padrão 4 sessões por modelo)** e descarte ao `Release()`/troca de modelo/provider/fallback.
- Fluxo em `Generate` quando `Mode == Reuse` e o prompt veio de `PromptTokens` (autocomplete):
  1. `n = min(CapTokens ?? ∞, LCP(session.kv, input), input.Length - 1)` (sempre ≥ 1 token novo: o logit da última posição só existe após processar ao menos um token).
  2. `generator.RewindTo(n)`; `generator.AppendTokens(input[n..])`; `GenerateNextToken` (greedy, como hoje). `terminate_session` e o laço de parada permanecem iguais.
  3. Ao terminar (ou ser cancelado): `session.kv = input` (só o prompt; os tokens gerados não entram), exceto se houve cancelamento durante o prefill ou exceção → `session.Invalidate()` (descartar o `Generator`).
  4. Registrar `PrefixCacheReusedTokens` (= n) e o TTFT; incluir `reusedTokens` em `GeneratedChunk`/`ModelGenerationResult` para o KapiLab.
- Pedido em andamento: o serviço já serializa as gerações; um novo pedido da mesma sessão cancela o anterior e só então chama `RewindTo` (nunca com o `Generator` ainda em uso).
- Primeiro pedido de uma aba (miss): prefill completo no `Generator` da sessão (= custo do baseline, sem `Generator` por pedido a criar). Opcional: ao abrir a aba/trocar de conexão, **pré-aquecer** o envelope (política A/B) em segundo plano, sem competir com um pedido do usuário.
- Ler o token gerado por `GetNextTokens()` (ou checar `IsDone()`) em vez de `GetSequence(0)[^1]`, para não duplicar o último token no EOS (achado da Fase 07 §3); pode ser corrigido neste PR ou no PR do EOS, mas a sessão não pode depender do bug.

### 3.3 Invalidação do cache (por comparação de ids, nunca de texto)

A sessão é válida enquanto `LCP` cobrir o que se quer reaproveitar; o LCP é recalculado a cada pedido, então:
- mudou `RECENT COMMAND`/`RESULT FIELDS`/`INPUT PANEL` → o LCP para antes do trecho alterado (reaproveita cabeçalho+nomes);
- mudou `KNOWN NAMES` (conexão, lista de coleções) → LCP ≈ 62 tokens (só o cabeçalho fixo);
- mudou a linguagem/dialeto → LCP ≈ 11 tokens; criar/usar outra sessão pela `SessionKey`;
- prompt truncado pelo orçamento (o cabeçalho cai primeiro) → LCP ≈ 0, prefill completo;
- invalidar tudo: `Release()`, troca de modelo, provider/fallback (`RecoverOnCpuAsync`), `SessionOverlay` diferente, cancelamento no meio do prefill, exceção do GenAI, `Settings` que alterem o prompt;
- **nunca** reusar com `ChatPrompt`, `Temperature > 0` com `Seed` diferente ou `TopP/TopK` (o cache só vale para greedy em R43);
- o fim do prompt (sufixo e `<|fim_middle|>`) muda a cada tecla e sempre é reprocessado; a regra “≥ 1 token novo” garante isso mesmo em prompt repetido.

### 3.4 Configuração e telemetria

- `AutocompleteSettings`: `PrefixCacheMode` (padrão `Disabled`), `PrefixCacheMaxSessions` (4). Habilitar por provider via metadata do pacote (`runtime.prefixCache: reuse`) **só** para CPU até haver prova em DirectML/NPU.
- Métricas: `prefix_cache.reused_tokens` (já existe), `prefix_cache.hit_rate` (n>0), `prefix_cache.invalidations{reason}`, TTFT separado por hit/miss; memória (working set) com N sessões.
- KapiLab: `bench autocomplete` ganha `--session-mode` (pedidos de uma mesma aba em sequência: o pedido k+1 = pedido k + 1 caractere do trecho esperado, e variantes com `recent_commands` alterado) e reporta TTFT média/p95 de hit, de miss e da mistura, mais `reusedTokens`.

### 3.5 Testes

- Unitários (fake do `Generator`): sequência de chamadas `RewindTo(n)` → `AppendTokens(resto)` com `n = min(cap, LCP, len-1)`; prompt idêntico repetido → `n = len-1`; LCP 0 → prefill completo; cancelamento/exceção → sessão invalidada; LRU descarta a menos usada; troca de modelo/provider zera sessões; `Disabled` → caminho atual intacto (mesmos parâmetros e `max_length`).
- Equivalência greedy com modelo real (CPU, T0 int4 e T1 int8): 500 prompts do val v2c, saída com cache = sem cache em **≥ 99%** (o protótipo deu 100%); divergências devem trazer a margem top1−top2 do baseline; FIM acc/exec-valid em n=500 inalterados (T0 0,390/0,562; T1 0,408/0,582). Incluir cenários: hit, miss, mudança de `recent_commands`, digitação (+1 char), troca de linguagem.
- Teste de concorrência: dois pedidos seguidos da mesma sessão (o segundo cancela o primeiro) sem corrida no `Generator`.
- Teste de memória: working set com 4 sessões ≤ gate de RAM do tier (T0 ≤ 1,8 GB; T1 ≤ 3,8 GB).

## 4. Critérios de aceite (gates)

Medidos com `--session-mode`, CPU, 12 e 24 threads, n ≥ 200, aquecimento descartado, máquina ociosa:
- T0 int4 (150/250 ms): TTFT em hit estável ≤ 150/250 (protótipo: 92/221 a 24 thr; 98/207 a 12); mistura com hit ≥ 95% ≤ 150/250; reportar também 80% e 90% (o p95 falha em 90%).
- T1 int8 (500/650 ms): hit estável ≤ 500/650 (protótipo 250/497 e 235/482); hit ≥ 95% ≤ 500/650 (632 e 562).
- Taxa de acerto declarada e medida em uso real (aba com digitação contínua) ≥ 95% para valer o gate; senão reportar como informativo.
- Equivalência greedy ≥ 99%; FIM acc/exec-valid sem perda (|Δ| ≤ 0 em n=500 ou provado numérico).
- `Disabled` mantém os números atuais (258/474 e 667/1101 a 24 thr no proxy) e o mesmo comportamento; zero regressão nos testes existentes.
- Sem GPU: o cache só liga em CPU; DML/NPU permanecem desligados.

## 5. Fora de escopo

- Prefix cache automático (Engine 0.17/KV paginado), KV quantizado, qualquer mudança no contrato `editor-context-v1` ou nos limites de `recent_commands`/`result_fields` (os dados de truncamento estão no relatório, só para decisão do produto).
- Cache para chat/agente (A6 da Fase 08 trata `Generator` por turno com `RewindTo`; reutilize o mesmo `PrefixSession` depois, em outro PR).
- DirectML/NPU; beam search; correção do EOS (se não for feita aqui).

## 6. Riscos

| risco | mitigação |
|---|---|
| Taxa de acerto menor que a esperada (comandos executados mudam o envelope) | LCP parcial reaproveita cabeçalho+nomes; telemetria de hit/miss; gate declara a taxa; pré-aquecimento ao abrir a aba |
| Diferença numérica rara do prefill em dois blocos | gate de equivalência ≥ 99% por modelo/provider; interruptor `Disabled` |
| Memória por sessão (KV de 2.048 posições) | LRU de 4; medir working set; liberar na troca de modelo |
| Corrida entre cancelamento e `RewindTo` | serialização do serviço + invalidar a sessão em cancelamento durante o prefill |
| Bug do EOS (último token duplicado) contaminando `kv` | `kv` guarda só o prompt; ler `GetNextTokens()` |
| Harness de gate atual (pedidos independentes) não vê o ganho | `--session-mode` no KapiLab e gate redefinido como “sessão” |
