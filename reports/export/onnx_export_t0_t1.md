# Exportação ONNX dos alunos T0 Nano e T1 Small (Fase 07)

Data: 2026-10-20 · Builder `onnxruntime-genai` **0.15.2** (= IDE) · modelos de origem `merged/sft_distill_v1` (T0 0,5B full FT; T1 1,5B, LoRA fundido), bf16 ·
pacotes em `Repos/KapiCoder-Mongo-<Nano|Small>/exports/<família>-ONNX-<variante>/` (gitignored) · scripts e reprodução: `scripts/export/README.md` ·
KapiLab: `reports/export/kapilab_commands.md`. **Nada aqui é qualificado**: latência/memória são PROXY (ver §4) e nenhum gate foi marcado como aprovado oficialmente.

## 1. Pacotes gerados

T0 (8): `fp32-cpu` 1.904 MB (referência), `int4` 392 MB, `int4-rtn` 317 MB, `int4-kquantmixed` 423 MB, `int8` 1.071 MB, `int8-emb4` 633 MB, `dml-fp16` 959 MB, `dml-int4` 544 MB.
T1 (6): `fp32-cpu` 5.916 MB (referência), `int4` 1.090 MB, `int4-rtn` 952 MB, `int8` 2.578 MB, `dml-fp16` 2.964 MB, `dml-int4` 1.293 MB.

- `model.type = qwen2`; `genai_config.json` só tem `search` alterado (`do_sample=false`, `max_length=2048`, Fase 07 §6.5); `context_length` fica 32768 (builder).
- Tokenizer/vocab/merges/`generation_config.json` copiados **byte a byte** do merged (SHA-256 idêntico, testado); `chat_template.jinja` do builder removido (só pacotes de agente).
- Metadata `kapibarastudio-model.json` (schema `kapilab-kapi-model-v1`): `contextContract editor-context-v1`, `recommendedContextTokens 2048`,
  `generation.autocomplete {32, 0}`, `runtime.minimumGenAi 0.15.2`, paradas (`<|fim_prefix|>…<|fim_pad|>`), `hardware` cpu/gpu, licença Apache-2.0 + aviso de modificação,
  base (`Qwen/Qwen2.5-Coder-0.5B` @ `8123ea2e…` / `-1.5B` @ `df3ce67c…`), `datasetVersion v2c`, hash dos pesos merged. `LICENSE`, `NOTICE.md`, `README.md` em cada pacote.
- `kapicoder_manifest.json` (SHA-256 de cada arquivo, schema do KapiLab) e `export_manifest.json` (sha256 + bytes por arquivo, ferramenta/versão, comando exato, hashes do merged).
- Receitas (`build_variants.py`): `int4` = `algo_config=k_quant matmul_mixed_precision=last_matmul:int8 block_size=32 accuracy_level=4` (Fase 07 §5.2); `int4-rtn` = RTN; `int4-kquantmixed` = melhor int4
  da sonda; `int8` = contorno `op_types_to_quantize=MatMul shared_embeddings=false` (o `-p int8` puro continua gerando grafo inválido; `algo_config=rtn` com `-p int8` gera **4 bits** em silêncio, descartado);
  `int8-emb4` = `int8` + embedding em `GatherBlockQuantized` INT4 (o ORT 1.30 só aceita 4 bits em Gather) para caber nos 700 MB; `dml-fp16`; `dml-int4` com `shared_embeddings=false`.
- Testes: `scripts/export/tests` (57 passam): hashes do manifesto, `genai_config`, metadata, tokenizer byte a byte.

## 2. GPU: verificação das duas placas

LUID da RX 7800 XT nesta sessão = `0x00000000_0x00012951` (calibrado alocando 3 GB com torch-ROCm em `cuda:0`="AMD Radeon RX 7800 XT": memória dedicada do processo apareceu só nesse LUID);
iGPU = `0x00000000_0x00014bfc`. O GenAI-WinML escolhe sozinho (o `device_id` 0/1/2 não alterou nada). **Todos** os 4 pacotes DML foram verificados por contadores
(`\GPU Process Memory` e `\GPU Engine` por pid, `scripts/export/gpu_probe.ps1`): memória dedicada e utilização só no LUID da 7800 XT, **0 MB e 0% na iGPU**. A medida de latência reconfere ao final e aborta (exit 3) se
detectar a iGPU — não ocorreu. VRAM dedicada medida (processo): T0 `dml-fp16` 2.933 MB, `dml-int4` 2.530 MB; T1 `dml-fp16` 5.133 MB, `dml-int4` 3.062 MB (bem acima do tamanho do pacote: buffers/KV do DML).

## 3. Qualidade e paridade (500 prompts do val v2c, `editor-context-v1`, ctx 2048, greedy ≤96 tokens, scorer do `eval_teacher`)

Amostra: 500 ids sorteados (seed 20261020) dos 2.000 de `base-v2c/sample_ids.json`; paridade token a token nos 200 primeiros contra o HF bf16 do mesmo modelo regenerado na 7800 XT
(texto idêntico às predições armazenadas do eval_student em 100%). Δ pareado e McNemar exato (`scripts/eval/stats.py`) contra as **mesmas linhas** do HF bf16 (n=500; os números n=2.000 do enunciado:
T0 0,403/0,582; T1 0,4105/0,595). Tokenizador nativo do GenAI = HF em 500/500 prompts de cada pacote (ids idênticos, marcadores FIM como token único).

HF bf16 no recorte n=500: T0 FIM 0,384 · exec 0,564 · campos inexistentes 0,040 · sintaxe 0,992; T1 FIM 0,412 · exec 0,582 · campos 0,026 · sintaxe 0,994.

### T0 Nano (0,5B)

| variante | MB | greedy idêntico % | prefixo aceito % | FIM acc (Δpp; p) | exec-valid (Δpp; p) | campos inex. | sintaxe | TTFT proxy méd/p95 ms | KapiLab TTFT méd/p95 | tok/s | RAM pico GB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32-cpu (ref.) | 1904 | 97,0 | 98,0 | 0,382 (−0,2; 1,0) | 0,564 (0,0; 1,0) | 0,040 | 0,992 | 350/572 | – | 27 | 3,33 | – |
| int4 | 392 | 79,0 | 84,9 | 0,390 (+0,6; 0,58) | 0,562 (−0,2; 1,0) | 0,036 | 0,986 | 285/480 | 297/502 | 113 | 1,53 | – |
| int4-rtn | 317 | 70,0 | 78,7 | 0,388 (+0,4; 0,82) | 0,562 (−0,2; 1,0) | 0,048 | 0,994 | 297/492 | 302/487 | 120 | 1,39 | – |
| int4-kquantmixed | 423 | 81,5 | 86,7 | 0,392 (+0,8; 0,34) | 0,568 (+0,4; 0,69) | 0,032 | 0,990 | 287/475 | 281/476 | 102 | 1,56 | – |
| int8 | 1071 | 95,0 | 96,7 | 0,384 (0,0; 1,0) | 0,568 (+0,4; 0,50) | 0,038 | 0,992 | 262/440 | 276/462 | 93 | 1,58 | – |
| int8-emb4 | 633 | 93,0 | 95,4 | 0,386 (+0,2; 1,0) | 0,568 (+0,4; 0,50) | 0,036 | 0,996 | 263/442 | 269/442 | 92 | 1,58 | – |
| dml-fp16 | 959 | 97,5 | 98,4 | 0,382 (−0,2; 1,0) | 0,564 (0,0; 1,0) | 0,040 | 0,992 | **109/124** | – (sem token de fila) | 166 | 2,27 | 2933 |
| dml-int4 | 544 | 76,5 | 85,1 | 0,382 (−0,2; 1,0) | 0,560 (−0,4; 0,75) | 0,046 | 0,990 | 142/165 | – | 216 | 2,50 | 2530 |

### T1 Small (1,5B)

| variante | MB | greedy idêntico % | prefixo aceito % | FIM acc (Δpp; p) | exec-valid (Δpp; p) | campos inex. | sintaxe | TTFT proxy méd/p95 ms | KapiLab TTFT méd/p95 | tok/s | RAM pico GB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32-cpu (ref.) | 5916 | 96,0 | 97,4 | 0,408 (−0,4; 0,50) | 0,580 (−0,2; 1,0) | 0,028 | 0,992 | não medido | – | – | – | – |
| int4 | 1090 | 78,5 | 84,3 | 0,398 (**−1,4; 0,039**) | 0,580 (−0,2; 1,0) | 0,032 | 0,994 | 788/1317 | 787/1293 | 47 | 2,63 | – |
| int4-rtn | 952 | 73,0 | 80,5 | 0,394 (**−1,8; 0,004**) | 0,568 (**−1,4; 0,039**) | 0,038 | 0,990 | 800/1372 | 790/1282 | 49 | 2,39 | – |
| int8 | 2578 | 94,5 | 96,6 | 0,408 (−0,4; 0,50) | 0,582 (0,0; 1,0) | 0,026 | 0,994 | 697/1192 | 698/1139 | 33 | 3,01 | – |
| dml-fp16 | 2964 | 96,5 | 97,7 | 0,408 (−0,4; 0,50) | 0,582 (0,0; 1,0) | 0,026 | 0,992 | **148/170** | – | 104 | 2,26 | 5133 |
| dml-int4 | 1293 | 78,0 | 83,3 | 0,394 (**−1,8; 0,012**) | 0,574 (−0,8; 0,22) | 0,034 | 0,992 | 173/208 | – | 167 | 2,51 | 3062 |

Leitura:
- O piso de paridade greedy contra o HF **bf16** é ~96-97% mesmo para o ONNX fp32 (T0 97,0%; T1 96,0%): o bf16 de referência já difere do fp32 em ~3-4% dos prompts. Os gates `greedy_identical` (≥97/99) deveriam usar a referência fp32/mesma precisão.
- Paridade greedy: fp16 DML e fp32 ≈ piso; int8 93-95%; **int4 70-82%** (RTN o pior). Apesar disso a qualidade medida (FIM, exec-valid, campos, sintaxe) do T0 em qualquer variante é indistinguível do HF (|Δ| ≤ 0,8 pp, todos p ≥ 0,34), e ΔAPT ≥ −0,05 (T0) / −0,06 (T1 int4).
- T1: os int4 **perdem de forma significativa** FIM (−1,4 a −1,8 pp, p 0,004-0,04) e, no RTN, exec-valid (−1,4 pp); int8 e fp16 não perdem (Δ −0,4 pp, p 0,5). 1,5B sofre mais com int4 que o 0,5B (que não perde com 392 MB).
- Gates de paridade (Fase 06 §6, `common.parity.*`): passam T0 int4/int8/int8-emb4/dml-int4 (ΔAPT e Δexec dentro do piso); T0 `dml-fp16` passa (97,5 ≥ 97); T0 `fp32-cpu` falha 97,0 < 99 e T1 `dml-fp16` falha 96,5 < 97 **só por causa da referência bf16** (T1 fp32 = 96,0%); T1 `dml-int4` falha ΔAPT (−0,19 < −0,15). Top-1 *teacher-forced* e cosseno de logits (Fase 07 §7.1) **não foram medidos** (GenAI 0.15 só expõe os logits da última posição).
- **Achado de integração (afeta todos os pacotes):** a IDE (`OnnxLocalModelRuntime.Generate`) lê `GetSequence(0)[^1]` e, no GenAI 0.15.2, o passo que gera `<|endoftext|>` não acrescenta o token à sequência → a IDE **duplica o último token** de toda sugestão que termina em EOS (`'1'→'11'`, `"→""`, `({→({({`). No KapiLab (`model run autocomplete`, 100 registros): concordância de texto IDE × Python **12% (T0 int4) e 13% (T1 int8)**; numa cópia de teste com `eos_token_id=151935` (inalcançável; a IDE já para pelos seus tokens) sobe a **93% (95,8% nos registros com cabeçalho)**. Enquanto isso não for corrigido (ou o `eos_token_id` neutralizado no pacote), o gate `common.harness.python_agreement ≥ 0,995` não passa para nenhum modelo. Detalhes e arquivos: `scripts/export/README.md`, `reports/export/kapilab_agreement_*.json`.
- Contrato: `contract tokenize` do KapiLab × ids do Python: 95/100 idênticos em T0 e T1; os 5 diferentes são exatamente os `context_kind=none` (o Python/treino omite o cabeçalho; a IDE sempre emite o cabeçalho de linguagem) → a IDE nunca produzirá esse caso.

## 4. Latência e memória — PROXY (não qualificado)

Método: `onnx_run.py --mode latency` (GenAI Python 0.15.2, ids pré-tokenizados, `ClearProviders()/AppendProvider()` como a IDE, `intra_op_num_threads` padrão=0, batch 1): 3 aquecimentos descartados + **200** prompts reais do val
(mediana **260** tokens, média 272), até 32 tokens com paradas do contrato; TTFT = `append_tokens`+1º `generate_next_token`; tok/s = decodificação forçada de 64 tokens (`min_length`, 20 prompts); RAM = `peak_wset`;
um processo por pacote. A máquina **não esteve ociosa de verdade**: apps de fundo (ChatGPT, Codex, testhost/dotnet de outra sessão) mantiveram a CPU entre 1% e 7% antes das rodadas (registrado por pacote);
as medidas CPU com carga >5% ou do T1 foram repetidas até a carga ficar <5% (a rodada 1 foi guardada em `work/*_run1.json`). O proxy foi cruzado com o caminho real da IDE: `kapilab bench autocomplete --device cpu`
(apphost publicado, 203 registros, 1 aquecimento): TTFT KapiLab × proxy diferem ≤ 8% (T0 int4 297/502 × 285/480; T1 int8 698/1139 × 697/1192) — o proxy é bom para CPU. Sensibilidade a threads (T0 int4, 100 prompts, informativo):
4 thr 731/1191 ms, 8 thr 406/660, 12 thr 291/462, 24 thr **229/382** — ainda longe de 150/250.

### Gates (`configs/gates.yaml` / `gates.ide.json`) contra as medidas proxy

Rodado com `scripts/eval/check_gates.py` (via `scripts/export/gates_proxy.py`; medidas do owner kapilab com `proxy:true, qualified:false` → o check devolve `unqualified` e guarda o resultado cru em `raw_result`; `qualified=false` em todos). Resultado cru:

| tier / cenário | variante | TTFT méd (≤) | TTFT p95 (≤) | RAM GB (≤) | disco MB (≤) | tok/s (≥) | paridade |
|---|---|---|---|---|---|---|---|
| T0.cpu | int4 | 285 ✗ (150) | 480 ✗ (250) | 1,53 ✓ (1,8) | 392 ✓ (700) | – | ΔAPT +0,08 ✓ · Δexec −0,2 pp ✓ |
| T0.cpu | int4-rtn / kquantmixed | 297 / 287 ✗ | 492 / 475 ✗ | 1,39 / 1,56 ✓ | 317 / 423 ✓ | – | ✓ |
| T0.cpu | int8 | 262 ✗ | 440 ✗ | 1,58 ✓ | **1071 ✗** | – | ΔAPT −0,02 ✓ |
| T0.cpu | int8-emb4 | 263 ✗ | 442 ✗ | 1,58 ✓ | 633 ✓ | – | ΔAPT −0,02 ✓ |
| T0 DML (sem gate próprio; cenário CPU só informativo) | dml-fp16 | 109 | 124 | 2,27 | 959 | 166 | greedy 97,5 ✓ |
| T1.cpu_int8 | int8 | 697 ✗ (500) | 1192 ✗ (650) | 3,01 ✓ (3,8) | – | – | ΔAPT −0,01 ✓ |
| T1.cpu_int8 (informativo) | int4 / int4-rtn | 788 / 800 ✗ | 1317 / 1372 ✗ | 2,63 / 2,39 ✓ | – | – | ✓ |
| T1.dml_fp16 | dml-fp16 | 148 ✓ (170) | 170 ✓ (200) | – | – | 104 ✓ (90) | greedy 96,5 ✗ (≥97, ver leitura) |
| T1.dml_fp16 (informativo) | dml-int4 | 173 ✗ | 208 ✗ | – | – | 167 ✓ | ΔAPT −0,19 ✗ |

Notas: o gate T1 `cpu_int8` usa a métrica de **chat** (`ttft_chat_*`); aqui só há prompts de autocomplete de ~270 tokens (o proxy é imperfeito para esse cenário). O TTFT da CPU é dominado pelo prefill de ~270 tokens
(o grafo int4 não ajuda: 285 ms int4 × 262 ms int8 no T0). O gate T0 de CPU (150/250 ms) **não é atingível** neste Ryzen 9 7900 com prompts reais de ~270 tokens (melhor caso 229/382 com 24 threads); o gate T1 CPU idem.

## 5. KapiLab

Existe binário publicado (`artifacts/f7b/publish/{Cpu,WinML}/win-x64/…KapiLab.exe`, SHA-256 confere com `tools/kapilab.lock.json`; GenAI 0.15.2; checkout atual `7844bd71` ≠ commit do lock `b1e10b09`); nada foi compilado nem escrito em `F:\`
(workspace em `tmp/kapilab_ws`). Resultados:
- `model validate --strict-kapi --verify-hashes` nos **14 pacotes**: todos `Available`, `schemaValid`, `ideSupported`, `strictSchema kapilab-kapi-model-v1`, `hashesValid`, 14 arquivos verificados, 0 issues (`reports/export/kapilab_validate_t0|t1.json`). `--load` não existe nesta publicação; `model run autocomplete --device cpu` carregou e gerou em CPU sem fallback.
- `contract render|tokenize`: ver §3 (95/100 idênticos; os 5 = `context_kind=none`). `bench autocomplete --device cpu`: 203 amostras/pacote CPU, 0 nulos, 0 falhas, 0 fallback (`reports/export/kapilab_bench_summary.json`).
- **Bloqueios:** GPU no KapiLab exige `--via-queue` + `KAPILAB_GPU_TOKEN` da fila local — sem token, **não rodado** (comandos prontos em `kapilab_commands.md`); o gate oficial DML do T1 continua pendente. QNN/NPU: sem máquina Snapdragon nem QAIRT/Olive aqui; nenhum pacote `qnn-*`.

## 6. Recomendação

| tier | candidato a produção | por quê | alternativa |
|---|---|---|---|
| T0 CPU | **`int4`** (392 MB) — ou `int4-kquantmixed` (423 MB) | qualidade = HF (Δ ≤ 0,8 pp, p ≥ 0,34), menor RAM/disco, passa RAM/disco/paridade; falha só TTFT (todos os CPU falham) | `int8-emb4` (633 MB, greedy 93%, passa disco) se se quiser mais fidelidade numérica; descartar `int8` (1,07 GB > 700) e `int4-rtn` (pior paridade) |
| T0 GPU | **`dml-fp16`** (959 MB) | paridade 97,5%, TTFT 109/124 ms, 166 tok/s | `dml-int4` só se VRAM importar (sem ganho de TTFT) |
| T1 GPU | **`dml-fp16`** (2,96 GB) | única que passa TTFT 148/170, ≥90 tok/s (104) e não perde qualidade (Δ −0,4 pp, p 0,5; exec idêntico) | – (`dml-int4` perde −1,8 pp FIM com p=0,012 e estoura TTFT) |
| T1 CPU | **`int8`** (2,58 GB) | sem perda de qualidade; RAM 3,0 GB ✓; **TTFT 697/1192 ms não passa 500/650** | `int4` não compensa (perde −1,4 pp FIM, TTFT igual) |

Riscos: (1) bug do EOS na IDE (§3) invalida o gate de concordância IDE×Python até ser corrigido; (2) TTFT de CPU reprova em T0 e T1 — é prefill-bound; (3) int4 do T1 perde qualidade significativa; paridade greedy int4 70-82% (informativa);
(4) máquina não ociosa (1-7% de fundo) e proxy ≠ harness oficial; GPU não medida pelo KapiLab; (5) DML usa muita VRAM (3-5 GB) e RAM de pico 2,3-2,5 GB; (6) top-1 teacher-forced não medido;
(7) `search.max_length=2048` no config (Fase 07 §6.5) e `context_length` 32768 — a IDE usa seu próprio `max_length`; (8) o aluno foi treinado também com `context_kind=none` que a IDE nunca emite (sem impacto, só nota).

## 7. Decisões que precisam do usuário

1. **Corrigir o EOS na IDE** (ler `GetNextTokens()`/checar `IsDone`) **ou** aceitar o contorno `eos_token_id` neutro no `genai_config.json` dos pacotes (testado, +80 pp de concordância). Aplico no `package_variant.py` se aprovado.
2. **Gate de TTFT de CPU** (T0 150/250 ms; T1 500/650 ms): inatingível aqui com prompts de ~270 tokens → recalibrar o gate, reduzir o orçamento de contexto/cabeçalho, ou declarar T0/T1 como GPU-first com CPU best-effort.
3. **Referência de paridade** dos gates greedy: trocar bf16 por fp32 (piso real 96-97%), senão fp16 DML reprova por ruído do bf16.
4. T0 CPU: `int4` (392 MB) vs `int8-emb4` (633 MB); T1: `int4` fora (confirmar).
5. Liberar o token de GPU do KapiLab (`--via-queue`) para o gate oficial DML; decidir se se republica o KapiLab no commit atual da IDE (`7844bd71`).
