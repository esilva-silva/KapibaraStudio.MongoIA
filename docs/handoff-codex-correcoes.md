# Handoff para o Codex — correções pendentes (KapiCoder-Mongo)

Estado de partida: branch `master`, último commit `58b42cb` (D1–D5 do mapa de GPU). Leia primeiro `docs/resumo-executivo.md` (desatualizado, ver C1), `docs/mapa-compatibilidade-gpu.md` e `docs/prs/`.

## 0. Como usar os subagentes Sol e Lua

> **Premissa a confirmar:** não há definição de "Sol" e "Lua 6.1" no repositório. Tratei como dois subagentes paralelos e "6.1" como o rótulo/versão do perfil. Se "6.1" for outra coisa (versão do plano, modelo), ajuste esta seção; o conteúdo das tarefas não muda.

| Agente | Domínio | Pode escrever em | Não pode tocar |
|---|---|---|---|
| **Sol** | Dados, treino, avaliação, gates (qualidade do modelo) | `scripts/data`, `scripts/student`, `scripts/eval`, `configs/gates*.yaml`, `Repos/*/reports`, `reports/` (exceto `hwmap`) | `scripts/hwmap`, `scripts/export`, `docs/prs` |
| **Lua** | Exportação, GPU/hardware, publicação, documentação de PRs da IDE | `scripts/export`, `scripts/hwmap`, `scripts/publish`, `configs/gpu_compat.yaml`, `docs/` | `scripts/data`, `scripts/student`, `scripts/eval`, `configs/gates*.yaml` |

Regras de orquestração:
1. O Codex principal só orquestra: dispara, espera, valida e commita. Sol e Lua trabalham em paralelo apenas nos blocos marcados **[paralelo]**; blocos **[serial]** esperam o anterior.
2. Cada agente devolve: arquivos alterados, comandos rodados, testes e resultado, o que ficou **[E]** (estimado) e o que é **[M]** (medido).
3. Um agente que encontrar necessidade de editar arquivo do outro agente para e reporta ao orquestrador.
4. Jobs de GPU passam pela fila (`pwsh scripts/gpu_queue.ps1`, `configs/gpu_queue.yaml`). **Só um job de GPU por vez**, e todo job confirma o adaptador RX 7800 XT (0x747E) por contadores. Nunca selecionar GPU por índice.
5. Commits pequenos por tema, mensagem em português, sem commitar pesos, `data/`, `exports/`, GGUF ou `reports/gpu_queue/state.json`.

## 1. Restrições invioláveis

- **Nada de upload ao Hugging Face** nem criação de repositório. `scripts/publish/upload_to_hf.py` só imprime comandos e exige `--i-have-user-approval`; o usuário roda.
- Não ler `data/homolog` fora da API autorizada (`homolog_guard`, com access log). Schemas `hom` (28) nunca entram em treino.
- `F:\source\esilva-silva\EsilvaSoft.KapibaraStudio` (IDE) é **somente leitura**. PRs para a IDE são apenas rascunhos em `docs/prs/`.
- Sem downloads da internet sem aprovação. MongoDB só leitura e só bancos `kapi_fx_*`.
- Não ler o token em `~/.cache/huggingface/token`.
- Não declarar qualidade fora do conjunto sintético: o cego humano foi descartado; a homologação é gerada por LLM/gramática (família Qwen), não humana.

## 2. Correções

### Bloco A — Documentação e rastreabilidade [paralelo: Lua faz A1–A3, Sol faz A4]

**A1 (Lua) — Atualizar `docs/resumo-executivo.md`.** Está datado de 2026-10-22 e ignora T0/T1 v3, ONNX v3, mapa de GPU e D1–D5. Atualizar tabela de fases, §3 (resultados v3), §5–§6 (bloqueios/decisões) e §8 (commits). Números v3 vêm de `Repos/*/reports/v3_decision_*.md` e `reports/export/`.
- Referência: T0 v3 homolog FIM 0,4234, exec÷teto 0,8617, campos inexistentes (full) 10,81% (base 7,02%); T1 v3 FIM 0,4537, exec÷teto 0,8926, campos 7,50% (base 7,15%).
- Aceite: todo número tem fonte em `reports/`; nada sem fonte.

**A2 (Lua) — Regenerar `model.lock.json`** (`scripts_dirty` stale) com árvore limpa após os commits. Aceite: lock aponta para o commit atual, `scripts_dirty=false`.

**A3 (Lua) — Limpar `README_job.md` obsoletos** em `Repos/KapiCoder-Mongo-{Nano,Small}/` para refletir `sft_v3.yaml`, o comando real e os hashes. Aceite: seguir o README reproduz o job v3 sem edições.

**A4 (Sol) — Registrar a limitação de independência da homologação.** O bug do caminho `v1/v1` em `homolog_guard` desativou `check_record/check_text`; o corpus J18 não foi decontaminado por n-gram contra a homologação até ser refeito para o v3. Consequências a documentar em `reports/` e no resumo: números do `sft_distill_v1` na homologação podem estar levemente otimistas; o v3 usa o mesmo gerador LLM e templates de gramática, logo não é totalmente independente. Aceite: parágrafo "Limitações de independência" com os itens acima e a fonte de cada um.

### Bloco B — Qualidade dos modelos [serial, Sol]

**B1 — T0 v3: campos inexistentes (full) 10,81% vs meta ≤ base+1 pp (8,02%).** A meta ideal não foi atingida (o GO veio pelos critérios C1–C4). Investigar causa (análise de erros por `cut_kind`/domínio) e propor correção de dados antes de gastar GPU. Entregar relatório + proposta; **não** iniciar treino sem aprovação do usuário.

**B2 — Regressão em Generic-JS** (−19 pp T0, −20 pp T1 vs zero-shot, no val v2c). Decidir se é aceitável (o produto é Mongo) ou se exige mistura de replay. Entregar análise e recomendação com números.

**B3 — Guarda de homologação** mostra perda de FIM acc (−9,24 pp T0, −8,4 pp T1) e exec÷teto (−11,74 / −4,82 pp) frente ao critério C3 de 1 pp. Verificar a definição do C3 em `scripts/student/decide_v3.py` e esclarecer por que o veredito é GO com essas perdas (a comparação parece ser contra o modelo anterior em conjunto diferente). Se o critério estiver mal formulado, corrigir `decide_v3.py` + teste e reemitir `v3_decision_*.md`. **Não alterar o veredito sem mostrar a conta ao usuário.**

**B4 — `configs/gates.yaml` (rascunho, 128 gates).** Preencher os 4–5 limiares sem número e o de paridade tokenizer/prompt/greedy; criar o baseline `base_v1_0.5b` na suíte v2 (hoje `T0.apt_*` e `exec_match` ficam sem avaliação). Propor valores com justificativa; **o congelamento é decisão do usuário**.

**B5 — Repetibilidade.** Resultados com semente única. Planejar (sem rodar) uma segunda semente do T0 v3 com custo de GPU estimado.

### Bloco C — Exportação e hardware [serial entre C1→C2; C3–C5 paralelos, Lua]

**C1 — Pacotes `-eosfix`.** O bug do EOS na IDE (`GetSequence(0)[^1]` duplica o último token; concordância IDE×Python 12–13%) reprova `python_agreement`. Implementar a flag `--eos-workaround` no export, gerando variante `-eosfix` **sem remover** a variante padrão. Aceite: concordância ≥ gate em teste Python que reproduz o comportamento da IDE; saída idêntica na variante padrão.

**C2 — Cache de prefixo (PR 4).** Os gates de TTFT em CPU só passam com cache de prefixo por aba (T0 105/208 ms, T1 246/526 ms com cache; sem cache T0 301/506, T1 768/1281). Manter o protótipo Python como referência e fechar o rascunho `docs/prs/pr-4-prefix-cache-onnx.md` (taxa de acerto mínima ~95%, política B). Não implementar na IDE.

**C3 — Re-medir a curva de VRAM com a máquina ociosa.** 53 de 131 registros da fase 2 e a maioria das 120 execuções da curva ficaram `noisy` (CPU > 3%). O pico de VRAM não é afetado, mas a latência é. Refazer só a latência com a máquina ociosa e marcar o que continua ruidoso.

**C4 — Detector (`scripts/hwmap/detect_gpus.py`).** (a) núcleos físicos vs. threads fora do Zen 4 (hoje validado só no Ryzen 9 7900); (b) formalizar `abort_warmup_ttft_factor` (4×, proposta) com teste; (c) cobrir casos de `device_id` ausente/desconhecido e driver sem contador de VRAM. Aceite: testes novos; os 73 atuais continuam passando.

**C5 — `docs/prs/pr-5-selecao-de-adaptador-dml.md`.** Revisar o rascunho: GenAI-WinML 0.15.2 ignora `device_id`/`device_filter`; a IDE precisa garantir a GPU certa (preferência de GPU do Windows por app, validar adaptador padrão efetivo com micro-benchmark, cair para CPU se vier iGPU/WARP, ou subir a versão do ORT GenAI). Incluir critérios de aceite testáveis.

### Bloco D — PRs da IDE (somente documentação) [Lua]

**D1** — Revisar `pr-1` (catálogo/paridade byte a byte), `pr-2` (contrato do agente), `pr-3` (tools/execução fixture): cada um com escopo, arquivos da IDE a mudar (lidos, não editados), testes e critério de aceite. Fase 04 (T4) fica **bloqueada** até os PRs 2 e 3 existirem na IDE; não iniciar.

### Bloco E — Publicação (preparação, nada enviado) [Lua]

**E1** — Rodar `verify_card_numbers.py` e `stage_hf_package.py` e confirmar que os cards refletem os números v3 da homologação. Pendentes de decisão do usuário: contato no card, manter `model-index`, nome "Mongo" (marca da MongoDB, Inc.; não é parecer jurídico). Nenhum upload.

## 3. Itens que dependem de decisão do usuário (não resolver sozinho)

1. EOS: corrigir na IDE, usar `-eosfix`, ou ambos.
2. Gate de TTFT em CPU: "gate de sessão com hit ≥ 95%" (exige PR 4) ou manter sem cache (reprova).
3. Variante T0: int4 (392 MB) ou int8-emb4 (633 MB). Recomendado int4.
4. Congelar `configs/gates.yaml` e confirmar `configs/rounds.yaml`.
5. Token de GPU do KapiLab (gates oficiais); Vivobook X1-26-100 (NPU/T3).
6. Q1: T1/T2 treinam o chat curto?
7. Qualquer upload ao Hugging Face.

## 4. Definição de pronto (por tarefa)

- Testes relevantes passam (`python -m pytest scripts/hwmap/tests`, `scripts/data/llm/tests` etc.).
- Todo número novo cita arquivo de origem e é rotulado [M] ou [E].
- Nenhum arquivo ignorado pelo Git foi forçado; nenhuma restrição da §1 foi violada.
- O agente informa explicitamente o que **não** conseguiu verificar.
