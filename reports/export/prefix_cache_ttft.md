# Cache de prefixo (KV) para o autocomplete em CPU — protótipo e medida

Data: 2026-10-22 · onnxruntime-genai **0.15.2** (= IDE) · Ryzen 9 7900 (12c/24t), **somente CPU** (provider CPU; `torch` nunca importado; sonda `gpu_probe.ps1` no processo: 0 MB dedicados/compartilhados e 0% de engine) ·
pacotes `int4` (T0 Nano 0,5B) e `int8` (T1 Small 1,5B) · harness: `scripts/export/prefix_cache_proto.py` (testes: `scripts/export/tests/test_prefix_cache.py`, 13 passam) ·
dados brutos: `reports/export/work/prefix_cache/*.json`, `reports/export/work/prefix_anatomy.json`, scores em `reports/export/eval/t{0,1}/*cache*` e `*trunc*`.
**PROXY, nada qualificado.** O contrato oficial não foi alterado; o checkout da IDE (F:\) só foi lido.

## 1. Resumo

- Fração cacheável do prompt (1.845 prompts do val v2c com cabeçalho, média 281 tokens): **A (cabeçalho fixo por linguagem) 62,7 tokens = 22%** (p50 24%, p90 37%); **B (cabeçalho + nomes + campos + comandos recentes, até o fim do envelope) 200 tokens = 71%** (p50 72%, p90 91%).
- TTFT em hit estável (n=200, 5 aquecimentos descartados, ordem dos modos aleatória por pedido), média/p95:
  T0 int4 sem cache 258/474 ms (24 thr) e 300/478 (12 thr) → **A 202/405 e 237/418**, **B 92/221 e 98/207**; T1 int8 sem cache 667/1101 e 747/1189 → **A 541/863 e 588/1043**, **B 250/497 e 235/482**.
- **Equivalência greedy: 100,0%** (500/500) nas políticas A e B, 100% no prefill em blocos (200/200) e na sessão de digitação C (460/460), em T0 int4 e T1 int8; o baseline repetido também é 100% idêntico. FIM acc / exec-valid em n=500 **inalterados** (T0 0,390/0,562; T1 0,408/0,582, idênticos ao pacote sem cache).
- Gates de TTFT em CPU (T0 150/250; T1 500/650): **passam com B em acerto estável (T0 e T1, 12 e 24 thr)** e com taxa de acerto ≥ 95%; com 90% falha o p95; **A não passa em nenhum tier**. O gate medido com pedidos independentes (harness atual do KapiLab) não vê o cache B.
- Recomendação: sessão por aba com `Generator` persistente + `rewind_to(LCP)` (cobre B e a digitação, política C: T0 58/149, T1 136–160/343–381 ms); cache A só como piso.

## 2. Anatomia do prompt (2.000 prompts; `prefix_cache_proto.py anatomy`)

Ids reproduzem `eval_teacher.build_ids` (500/500 idênticos aos do export). 155 registros `context_kind=none` (8%) não têm cabeçalho (só ocorrem no treino; a IDE sempre emite o cabeçalho) e ficam fora das frações; nenhum dos 1.845 teve o cabeçalho cortado pelo orçamento de 2.048. Só há as linguagens Console (1.627) e json (218) na amostra.

| segmento (tokens) | média | p50 | p90 | muda quando |
|---|---|---|---|---|
| marcador FIM + `/* Local editor context…LANGUAGE:` | 11 | 11 | 11 | nunca (global) |
| até `KNOWN NAMES:` (A, por linguagem) | 62,7 | 61 | 75 | troca de linguagem |
| `KNOWN NAMES` (lista) | 36,4 | 36 | 52 | conexão/banco/lista de coleções |
| `RESULT FIELDS` | 34,1 | 28 | 72 | painel de resultado |
| `INPUT PANEL` + `RECENT COMMAND` | 54,9 | 0 | 178 | cada comando executado / digitação no painel |
| fecho do envelope `Continue at the cursor…*/` | 12 | 12 | 12 | nunca |
| prefixo de código (até 4.096 chars) | 42,4 | 25 | 109 | cada tecla |
| sufixo + marcadores | 39,0 | 23 | 99 | edição depois do cursor |

Cacheável (tudo que vem antes da 1ª parte variável do pedido):

| política | tokens (média / p50 / p90) | fração do prompt (média / p50 / p90) |
|---|---|---|
| A0 global (todas as linguagens) | 11 / 11 / 11 | 4% |
| **A** cabeçalho fixo por linguagem | 62,7 / 61 / 75 | 22% / 24% / 37% |
| A + nomes | 99 / 98 / 116 | 36% / 37% / 57% |
| B-campos (cabeçalho+nomes+campos) | 133 / 128 / 175 | 47% / 50% / 75% |
| **B** (tudo até o fim do envelope) | 200 / 176 / 323 | 71% / 72% / 91% |

Prefixo comum **empírico** (ids reais): LCP de todos os prompts = 11 tokens; por linguagem, LCP do grupo = 61 (Console) / 75 (json) e melhor vizinho p50 68 / p90 94; por conexão, banco, schema ou coleção o LCP do grupo é 11–61 e o vizinho p50 63–68 / p90 75–93;
por (conexão, banco, coleção) o vizinho dá p50 61 / p90 65. **O dataset não mede B**: dos 844 pares na mesma chave só 2 têm o mesmo cabeçalho até os campos, porque o gerador sintético embaralha/duplica `known_names`, nomes de conexões distratoras e `recent_commands` por registro (a IDE é determinística dentro de uma aba). Por isso B é um **teto estrutural** (cabeçalho estável na aba) e o TTFT de B é de acerto estável; use as misturas da §3.
Sessão de digitação (mesmo documento, outro cursor): a amostra tem 1 corte por snippet, então a política C foi simulada (§3).

## 3. Protótipo (a API Python espelha o C#: `append_tokens`, `rewind_to`, `token_count` existem em 0.15.2)

Fluxo: **sem cache** = um `Generator` por pedido, `max_length = prompt + max_tokens`, como o `OnnxLocalModelRuntime.Generate` (TTFT = `append_tokens` + 1º `generate_next_token`; em 0.15.2 o `append_tokens` já executa o prefill). **Com cache** = `Generator` persistente (`max_length` 2.048): `rewind_to(n)` + `append_tokens(prompt[n:])` com `n = min(corte da política, LCP com o KV válido, len-1)`; o rewind está dentro do cronômetro; o “pedido anterior” (prime) fica fora. C = digitação: o pedido anterior é o prompt do registro e o novo é o mesmo +1 caractere do trecho esperado (sufixo igual), n = LCP inteiro.

Viabilidade verificada em 0.15.2: `rewind_to` funciona na CPU (com `past_present_share_buffer` true e false, `num_beams=1`), zera `is_done` e aceita `rewind_to(0)`; **não funciona com beam search** (`RewindToLength is not supported with beam search`). Não há API para copiar/exportar/importar KV entre `Generator`s; o `Engine`/`Request` do Python 0.15.2 não faz prefix caching (cada `step` pede tokens novos; sem KV paginado) e o C# 0.15.2 não o expõe. Logo:
(i) **um Generator persistente com rewind: viável e medido**; (ii) **sessão por aba = um Generator por aba mantido vivo**: viável (memória por sessão não medida: KV de 2.048 posições; precisa de LRU); (iii) **prefill em blocos de ≤ 2.048 não reduz TTFT**: `append_tokens` em blocos de 64/128 ou `chunk_size` nas search options são iguais ou piores em CPU (§5).

### Tabela principal — TTFT (ms, média/p95; n=200; 5 aquecimentos descartados)

| tier | threads | sem cache | A (hit) | B-campos (hit) | **B (hit)** | **C (digitação)** | speedup B (média) | gate | A | B | C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 int4 | 24 | 258 / 474 | 202 / 405 | 151 / 336 | **92 / 221** | 58 / 149 | 2,8× | 150/250 | falha | **passa** | passa |
| T0 int4 | 12 | 300 / 478 | 237 / 418 | 169 / 325 | **98 / 207** | 58 / 148 | 3,1× | 150/250 | falha | **passa** | passa |
| T1 int8 | 24 | 667 / 1101 | 541 / 863 | 395 / 778 | **250 / 497** | 160 / 381 | 2,7× | 500/650 | falha | **passa** | passa |
| T1 int8 | 12 | 747 / 1189 | 588 / 1043 | 413 / 797 | **235 / 482** | 136 / 343 | 3,2× | 500/650 | falha | **passa** | passa |

Tokens reaproveitados em média (de 283,7): A 62,5, B-campos 132,3, B 200,3, C 240,7 (de 284). Baseline da digitação (prompt +1 char, sem cache): T0 254/442 (24 thr) e 301/479 (12); T1 655/1109 e 747/1204. Método: ordem dos modos embaralhada por pedido (seed 20261020) para que nenhum modo herde sistematicamente a frequência/cache da CPU; carga da máquina antes de cada rodada 1–7% (Gerenciador de Tarefas e apps de fundo abertos, então “ociosa” só aproximadamente). O p95 do baseline (474) é maior que o do proxy anterior (382; 100 prompts) por amostra diferente. Threads: 24 só ajuda a média (258 × 300); o p95 quase não muda. A sonda de GPU no processo (T0, 24 thr) deu 0 MB/0% de uso.

### Mistura acerto/erro (B) — o gate depende da taxa de acerto

O erro de cache (cabeçalho mudou: comando executado, painel de resultado novo, troca de aba/conexão) cai no prefill completo (= baseline; o rewind até o LCP parcial aproveita o que sobrou, não modelado aqui). TTFT média/p95 da mistura (amostras reais de acerto e de erro):

| tier / thr | 50% | 80% | 90% | 95% | 100% |
|---|---|---|---|---|---|
| T0 int4 / 24 | 175 / 421 | 126 / 323 | 109 / 258 | 101 / 242 | 92 / 221 |
| T0 int4 / 12 | 199 / 438 | 138 / 359 | 118 / 289 | 108 / 229 | 98 / 207 |
| T1 int8 / 24 | 459 / 1003 | 334 / 826 | 292 / 694 | 271 / 632 | 250 / 497 |
| T1 int8 / 12 | 491 / 1087 | 337 / 922 | 286 / 730 | 261 / 562 | 235 / 482 |

Gate T0 (150/250): a média passa com hit ≥ 80%; o p95 exige ≥ ~92–95%. Gate T1 (500/650): passa com hit ≥ 95% (p95 632 e 562) e falha o p95 em 90% (694/730). A (que independe do estado da aba) nunca passa.

### Equivalência (n=500, greedy ≤ 96 tokens, mesmas paradas de `onnx_run.py`; 12 thr)

| tier | sem cache repetido (n=100) | A | B | blocos 128 (n=200) | C (n=460) | divergências |
|---|---|---|---|---|---|---|
| T0 int4 | 100% | 100% | 100% | 100% | 100% | 0 |
| T1 int8 | 100% | 100% | 100% | 100% | 100% | 0 |

A comparação é token a token (inclui o motivo de parada). Esperavam-se diferenças numéricas (o prefill em dois blocos usa M diferente nos MatMulNBits/atenção), mas nenhuma mudou um token em 500 prompts × 2 tiers; o harness registra a margem top1−top2 do baseline no 1º token divergente caso apareçam. Não é garantia formal: o teste de equivalência na IDE deve rodar sempre (gate ≥ 99%).

| tier | variante | FIM acc (normalized_match) | exec-valid | campos inex. | sintaxe |
|---|---|---|---|---|---|
| T0 int4 | pacote medido na Fase 07 | 0,390 | 0,562 | 0,036 | 0,986 |
| T0 int4 | sem cache (este harness) | 0,390 | 0,562 | 0,036 | 0,986 |
| T0 int4 | cache A | 0,390 | 0,562 | 0,036 | 0,986 |
| T0 int4 | cache B | 0,390 | 0,562 | 0,036 | 0,986 |
| T1 int8 | pacote medido na Fase 07 | 0,408 | 0,582 | 0,026 | 0,994 |
| T1 int8 | sem cache / cache B | 0,408 / 0,408 | 0,582 / 0,582 | 0,026 | 0,994 |

(T1 cache A: tokens 100% idênticos; o score exec não foi repetido. C não é pontuado: são prompts diferentes.)

## 4. Políticas — leitura

- **A (cache global do cabeçalho fixo)**: 22% do prompt, ganho de 15–22% no TTFT (T0 24 thr: 258→202). Simples (uma sessão por linguagem, sempre acerta) mas não fecha nenhum gate. Os nomes (`KNOWN NAMES`) vêm logo depois e variam por conexão, então o ganho para em ~62 tokens.
- **B (por conexão/banco/coleção; sessão por aba)**: 71% do prompt, TTFT cai 2,7–3,2× e passa os gates em hit estável. O ponto fraco é a taxa de acerto: `recent_commands` e `result_fields` mudam por ação do usuário. Com o rewind até o LCP real, uma mudança só em `recent_commands` ainda reaproveita cabeçalho+nomes+campos (B-campos: 47%; T0 151/336 — sozinho não passa).
- **C (generalização de B, mesma implementação)**: reaproveitar o LCP inteiro com o pedido anterior da aba. Em digitação o novo pedido só refaz sufixo + 1–2 tokens: T0 58/149, T1 136–160/343–381 (85% do prompt reaproveitado). É o que ocorre de fato ao digitar, mas o harness de gate atual (pedidos independentes) não o exercita.

## 5. Outras alavancas de TTFT em CPU (mesmo harness, só `base`, n=200, sem cache)

| alavanca | T0 int4 média / p95 (ms) | observação |
|---|---|---|
| threads 8 | 411 / 657 | |
| 12 (2 rodadas) | 300 / 478; 302 / 491 | |
| 0 (padrão da IDE) | 307 / 484 | ≈ 12 |
| 16 | 255 / 402 | |
| 20 | 238 / 391 | melhor p95 |
| 24 (2 rodadas) | 258 / 474; 265 / 453 | ruído ±8% entre rodadas |
| `past_present_share_buffer=false` (12 thr) | 309 / 500 | sem ganho |
| `num_beams=2` | ~2× mais lento (578 × 294 ms em 1 prompt) | `rewind_to` não suporta beam |
| prefill em blocos manual 64 / 128 (24 thr, entrelaçado) | 302 / 593; 288 / 559 | pior que 1 bloco (258 / 474) |
| `chunk_size` 64 / 128 (12 thr) | 321 / 510; 312 / 490 | igual ou pior |
| int8 T0 (12 / 24 thr) | 288 / 455; **217 / 347** | mais rápido que int4 em 24 thr; 1.071 MB (acima do gate de 700 MB) |
| int8-emb4 T0 (12 / 24 thr) | 275 / 442; **220 / 356** | 633 MB; qualidade igual (Fase 07) |
| T1 int4 (12 / 24 thr) | 848 / 1395; 688 / 1214 | mais lento que o int8 e perde FIM (Fase 07) |
| ORT `graph_optimization_level`, `inter_op`, `enable_cpu_mem_arena` | ±5% (ruído; 25 prompts) | `config_entries` (allow_spinning) **não é aceito** pelo `genai_config` 0.15.2 |

Nenhuma alavanca leva o T0 a 150/250 sem cache (melhor: int8-emb4 a 24 thr, 220/356). A IDE usa `intra_op_num_threads=0` (≈ 12); 20 threads daria ~−20% de média e ~−18% de p95 sem mudar pacote — a avaliar contra a carga da IDE.

### Reduzir o prompt (T0 int4, n=500, greedy, sem cache; só dado para decisão — contrato oficial inalterado)

| variante | tokens do prompt (média / p90) | FIM acc | Δpp | exec-valid | campos inex. | TTFT (média / p95) |
|---|---|---|---|---|---|---|
| contrato (cheio) | 268,2 / 405 | 0,390 | – | 0,562 | 0,036 | 282 / 474 |
| `recent_commands` ≤ 1 | 243,5 / 354 | 0,388 | −0,2 | 0,560 | 0,038 | 268 / 424 |
| sem `recent_commands` | 215,4 / 309 | 0,382 | −0,8 | 0,554 | 0,044 | 229 / 353 |
| `result_fields` ≤ 32 | 268,2 / 405 | 0,390 | 0 | 0,562 | 0,036 | 287 / 490 |
| `result_fields` ≤ 16 | 264,7 / 403 | 0,388 | −0,2 | 0,562 | 0,036 | 296 / 490 |
| sem `result_fields` | 237,0 / 370 | 0,382 | −0,8 | 0,548 | 0,058 | 276 / 494 |
| `known_names` ≤ 32 | 268,2 / 405 | 0,390 | 0 | 0,562 | 0,036 | 312 / 518 |
| `known_names` ≤ 16 | 267,5 / 405 | 0,386 | −0,4 | 0,560 | 0,036 | 308 / 521 |
| enxuto: sem recentes, campos ≤ 16, nomes ≤ 16 | 211,2 / 297 | 0,380 | −1,0 | 0,552 | 0,044 | 246 / 372 |
| enxuto-0: idem, sem campos | 183,6 / 263 | 0,374 | −1,6 | 0,514 | 0,088 | 212 / 334 |

O corte custa pouco em FIM acc (−0,2 a −1,0 pp) mas tirar `result_fields` dobra os “campos inexistentes” e derruba exec-valid a 0,514 no enxuto-0. Nenhuma redução chega perto dos gates (melhor: 212/334 ms). Os TTFT das variantes que quase não mudam o número de tokens estão no ruído (rodadas sequenciais, não entrelaçadas; a deriva da máquina é de ±10%): confie nos tokens do prompt. n=500 sem teste pareado: |Δ| ≤ 1 pp está dentro do erro de amostragem.

## 6. Limitações do GenAI 0.15.2 e riscos

- KV não é transferível entre `Generator`s: cada sessão paga o prefill do seu prefixo uma vez e guarda memória; sem prefix cache automático (Engine 0.17, KV paginado) nem em C#.
- `rewind_to` não existe com beam search; a IDE usa greedy (OK). Cancelamento (`terminate_session`) no meio do prefill deixa o KV parcial: tratar a sessão como inválida (reconstruir).
- EOS: o passo que gera `<|endoftext|>` não entra em `GetSequence` (achado da Fase 07); a sessão deve ler `GetNextTokens()` ou checar `IsDone`.
- Equivalência é empírica (100% em 500 × 2 tiers); a numérica do prefill em blocos pode mudar um token raro: manter o gate ≥ 99% e um interruptor por provider.
- Só CPU medida. DirectML tem outro modelo de KV: decidir por provider (AC-15).
- O harness de TTFT do KapiLab (pedidos independentes, 1 Generator por pedido) precisa de um modo de sessão para enxergar B/C.

## 7. Recomendação

1. Implementar na IDE **sessão por aba** com `Generator` persistente e `rewind_to(LCP)` (B+C), com A como piso ao abrir a aba; invalidar por comparação de ids (spec em `docs/prs/pr-4-prefix-cache-onnx.md`).
2. Gates CPU: com B/C, T0 (int4) e T1 (int8) passam 150/250 e 500/650 **em acerto estável**; redefinir o gate como “sessão” (taxa de acerto ≥ 95% declarada e p95 dos erros reportado à parte). Hoje, com pedidos independentes, ambos continuam falhando (T0 258/474, T1 667/1101 a 24 thr).
3. Em paralelo e independente do cache: T0 `int8-emb4` (633 MB, qualidade igual) e `intra_op_num_threads≈20`; reduzir `recent_commands` a ≤ 1 só se o produto aceitar −0,2 pp.
