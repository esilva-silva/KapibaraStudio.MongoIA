# Fase 2 do mapa de GPU: medicoes (2026-10-26)

Gerado a partir de `reports/hwmap/phase2_raw/*.json` (bruto), `phase2_runs.jsonl` (carga observada por execucao) e `phase2_measurements.json` (agregado) por
`scripts/hwmap/phase2_report.py` + `phase2_md.py`. Maquina: Ryzen 9 7900 (12C/24T), DDR5, RX 7800 XT (1002:747E), iGPU Raphael 2 CU (1002:164E); GenAI-WinML 0.15.2;
pacotes v3. Prompts reais: `reports/export/work_v3/t{0,1}_prompts.json` (500 ids do val v2c, todos em `base-v2c/sample_ids.json`; mediana 261,5 tokens); n=100 por execucao,
3 aquecimentos descartados, 32 tokens de saida (paradas do contrato); tok/s sustentado = 64 tokens forcados (n=20 no T0, 10 no T1/CPU/variantes lentas).

## 0. Metodo, desvios e limites (leia primeiro)

1. **O GenAI-WinML 0.15.2 ignora `device_id`, `device_filter`, `performance_preference` e `device_filtering_options`** (testados: o processo sempre caiu na 7800 XT, conferido por
   contador). Para medir a iGPU *de proposito* sem mexer nas "Configuracoes graficas" do Windows, `scripts/hwmap/dxgi_select.py` remapeia, **so dentro do processo**, a ordem de
   enumeracao DXGI (`EnumAdapters1`/`EnumAdapterByGpuPreference` na vtable do `IDXGIFactory`) para por o adaptador `vendor:device` pedido na posicao 0. As travas de
   `scripts/export` (`onnx_run.py`, `measure_all.py`) **nao foram alteradas**; a iGPU so e escolhida com `scripts/hwmap/phase2_bench.py --adapter 1002:164E --allow-igpu`
   (flag explicita, gravada no JSON como `allow_igpu`; sem a flag o script recusa).
2. **Adaptador confirmado por contadores** (`\GPU Process Memory(*)\Dedicated/Shared Usage` e `\GPU Engine(*)\Utilization Percentage` por pid, `vram_sampler.ps1`): ver §7.
   Execucoes de latencia/paridade feitas antes do pedido de evidencia ficam como `pinned_by_arg_only` (a sonda pos-medida `probe_end` mostra a memoria do pid por LUID, nao o engine);
   para cada configuracao foi acrescentada uma execucao curta (n=20) com os contadores ligados durante a medida (`evid_*`); `adapter_ok` = pico de memoria > 100 MB no adaptador
   pedido e <= 50 MB / <= 1% de engine no outro. Nada foi apagado.
3. **Carga de fundo.** Regra: CPU <= 3% em 10 s antes de cada medida. O PC tem apps do usuario (ChatGPT, claude, Taskmgr, Explorer) que variaram de ~1% a ~25% durante as horas da bateria; nao foram encerrados.
   Cada execucao grava `idle_obs` (cargas de 10 s) e `noisy` (true quando nao chegou a <= 3% apos as tentativas). **Ruidosas: 53 de 131 registros bem-sucedidos**
   (quase toda a microbateria dos GPUs, T1 iGPU r2/r3, T1 cpu-int8 em 4/6/8/20 threads e padrao, e 3 dos 4 registros de paridade `evid_`); o efeito aparece na CPU
   (T1 th16: TTFT 644-754 ms entre execucoes) e e pequeno nas DML (CV de 3-4% na microbateria da 7800 XT mesmo ruidosa). Uma passada silenciosa extra da curva T1 foi tentada e **abandonada** (7-16% por > 6 min).
4. **Repeticoes.** Cada configuracao rodou 2 vezes e uma 3a quando TTFT media ou tok/s diferiram > 10% (ocorreu em `t0 dml-fp32`, `t1 cpu-int8 th16` e `t1 dml-fp16 iGPU`; na fp32 e na th16 a execucao discrepante coincide com carga de fundo alta).
5. **Limites.** Uma unica maquina; sem Pascal/Intel/outras GPUs; pacotes DML padrao usam graph capture; micro-benchmark com texto sintetico de 270 tokens; TTFT e proxy (GenAI Python, sem o caminho `LocalAiModelService` da IDE); sem medida de potencia/termica.

## A. iGPU (2 CU) x CPU x 7800 XT

| config | exec x n | TTFT media / p50 / p95 (ms) | tok/s decode (64 forcados) | tok/s (saidas curtas) | RAM pico WS (GB) | VRAM dedic. / compart. fim (MB) | var. exec-a-exec TTFT/tok/s (%) | carga CPU 10 s antes (%) | ruidosa |
|---|---|---|---|---|---|---|---|---|---|
| T0 dml-fp16 / 7800 XT | 2x100 | 104 / 103 / 115 | 173.2 | 79.9 | 1.13 | 2749 / 155 | 1.1/0.6 | 1.9, 1.1 | nao |
| T0 dml-fp16 / **iGPU** | 2x100 | 1338 / 1246 / 2223 | 23.1 | 18.7 | 3.52 | 371 / 2551 | 3.4/1.7 | 1.1, 1.4 | nao |
| T0 cpu-int4 / 24 thr | 2x100 | 216 / 214 / 359 | 106.0 | 91.0 | 1.05 | - | 0.5/0.6 | 1.0, 2.1 | nao |
| T0 cpu-int4 / 12 thr | 2x100 | 313 / 298 / 514 | 115.5 | 97.7 | 1.05 | - | 0.4/2.9 | 2.4, 2.1 | nao |
| T0 cpu-int4 / padrao (0) | 2x100 | 310 / 295 / 500 | 119.1 | 100.3 | 1.05 | - | 1.2/0.5 | 1.4, 1.3 | nao |
| T1 dml-fp16 / 7800 XT | 2x100 | 141 / 137 / 162 | 107.9 | 57.2 | 1.55 | 4947 / 158 | 0.1/0.1 | 1.2, 1.4 | nao |
| T1 dml-fp16 / **iGPU** | 3x100 | 3713 / 3557 / 6096 | 7.3 | 6.8 | 5.71 | 412 / 4854 | 6.5/17.7 | 1.3, 7.0, 8.7 | sim (2/3) |
| T1 cpu-int8 / 24 thr | 2x100 | 535 / 516 / 863 | 27.6 | 27.2 | 2.51 | - | 3.0/2.5 | 2.6, 1.1 | nao |
| T1 cpu-int8 / 12 thr | 2x100 | 786 / 759 / 1278 | 32.9 | 31.1 | 2.50 | - | 0.3/1.2 | 1.1, 1.8 | nao |
| T1 cpu-int8 / padrao (0) | 2x100 | 791 / 765 / 1258 | 27.2 | 25.8 | 2.50 | - | 1.2/2.9 | 11.8, 15.4 | sim (2/2) |

**Resposta objetiva: a iGPU de 2 CU NAO vence a CPU em nenhum caso.** T0: iGPU 1338 ms de TTFT e 23 tok/s contra CPU
216 ms (6.2x mais rapida que a iGPU) / 106 tok/s com 24 threads e 313 ms (4.3x mais rapida que a iGPU)
/ 116 tok/s com 12; T1: iGPU 3713 ms / 7.3 tok/s contra CPU int8 535 ms / 28 tok/s
(24 thr: 6.9x no TTFT, 3.8x no decode). Contra a 7800 XT a iGPU e 12.9x
mais lenta no TTFT e 7.5x no decode (T0), 26.4x e 14.8x (T1).
A extrapolacao do mapa (~0,7 s de prefill do T0 na iGPU) era otimista: o DML mediu 1,3 s. A iGPU usa so ~370-410 MB dedicados e 2,2-4,9 GB de memoria compartilhada (DDR5), e a RAM de pico do processo sobe a 3,5 GB (T0) / 5,7 GB (T1) contra ~1,1/1,6 GB na 7800 XT.
No T0 o decode da CPU (106-119 tok/s forcado) chega a 61-69% do da 7800 XT (173): o ganho da dGPU no T0 esta no prefill (104 x 216-313 ms).

## B. Precisao no DirectML (7800 XT), T0

**Causa da falha do `dml-fp32` (rc=1 nas execucoes iniciais).** O builder 0.15.2 declara como validas apenas `FP16 DML` e `INT4 DML` (mensagem "Valid precision + execution provider
combinations"; FP32 so em CPU/CUDA). `-p fp32 -e dml` (`scripts/hwmap/build_exp.py`) cai no caminho de Attention empacotada e gera um grafo sem as saidas `present.N.key/value`
(292 nos, 24 `Attention`, nenhum `GroupQueryAttention`): o ORT recusa o carregamento ("This is an invalid model. Graph output (present.0.key) does not exist in the graph", `graph.cc:1494`).
**Nao e VRAM nem limite do DML: e um export invalido do builder** [M]; o pacote invalido ficou em `...-v3-ONNX-dml-fp32-builder-invalid-exp`. Contorno experimental (monkeypatch
em memoria do builder, `MHA_PATCH` do `build_exp.py`): forcar `MultiHeadAttention` + `repeat_kv` e desligar o graph capture (o GenAI recusa graph capture sem `past_present_share_buffer`).
Isso gera um fp32 valido (1.908 MB), mas o caminho de atencao muda junto com a precisao; por isso o controle `dml-fp16-mha` (fp16, mesmo caminho MHA sem graph capture, 964 MB) isola a precisao.

| config | exec x n | TTFT media / p50 / p95 (ms) | tok/s decode (64 forcados) | tok/s (saidas curtas) | RAM pico WS (GB) | VRAM dedic. / compart. fim (MB) | var. exec-a-exec TTFT/tok/s (%) | carga CPU 10 s antes (%) | ruidosa |
|---|---|---|---|---|---|---|---|---|---|
| T0 dml-fp16 (GQA + graph capture; pacote v3) | 2x100 | 104 / 103 / 115 | 173.2 | 79.9 | 1.13 | 2749 / 155 | 1.1/0.6 | 1.9, 1.1 | nao |
| T0 dml-int4 (exp) | 2x100 | 134 / 133 / 150 | 225.9 | 89.2 | 1.26 | 2346 / 155 | 0.4/0.0 | 1.8, 1.2 | nao |
| T0 dml-fp16-mha (controle exp) | 2x100 | 117 / 117 / 131 | 9.4 | 9.3 | 0.95 | 2546 / 8 | 0.3/1.1 | 1.6, 1.5 | nao |
| T0 dml-fp32 MHA (exp) | 3x100 | 157 / 155 / 207 | 8.9 | 9.0 | 1.75 | 3862 / 8 | 17.4/18.0 | 13.7, 2.3, 1.4 | sim (1/3) |

| pacote (T0, 7800 XT) | tamanho em disco (MB) | paridade greedy identica ao HF bf16 (n=100) | 1o token igual | prefixo aceito medio | VRAM dedicada pico (n=20, contadores) |
|---|---|---|---|---|---|
| dml-fp16 | 960 | 95.0% | 100.0% | 96.6% | 2476 MB |
| dml-int4 (exp) | 548 | 72.0% | 97.0% | 81.3% | 2209 MB |
| dml-fp16-mha (exp) | 964 | 95.0% | 100.0% | 97.4% | 2489 MB |
| dml-fp32 MHA (exp) | 1908 | 94.0% | 100.0% | 96.1% | 3749 MB |

Leitura: (i) **fp32 custa ~2x em disco (1.908 x 960 MB) e +1,3 GB de VRAM (3,75 x 2,47 GB)**; no prefill custa 148 ms (execs silenciosas) contra 117 ms do fp16 no MESMO caminho MHA
(+26%), e a paridade com o HF e igual (94% x 95%, dentro do ruido de n=100). (ii) O decode do caminho MHA sem graph capture cai a ~9 tok/s tanto em fp32
(8.9) quanto em fp16 (9.4) contra 173 tok/s do fp16 GQA+graph capture: **a lentidao e do caminho de atencao, nao da precisao** (precisao custa ~5% no decode).
Conclusao para Pascal: um `dml-fp32` util exigiria um export GQA/graph-capture em fp32 que o builder 0.15.2 nao produz; o MHA improvisado reprova o criterio de decode (>= 25-60 tok/s) ate numa 7800 XT.
O beneficio do fp32 numa GPU de FP16 fraco so se mede nela (sem Pascal local): **[E]**. (iii) **dml-int4**: 134 ms de TTFT (+29% vs fp16),
226 tok/s (+30%), 548 MB (-43%), VRAM 2,2 x 2,5 GB, mas **a paridade greedy cai a 72%** (prefixo medio 81%): nao serve ao T0.

## C. Micro-benchmark de primeira execucao (executor real de `scripts/hwmap/microbench.py`)

10 execucoes por candidato, **um processo novo por execucao** (270 tokens de prompt sintetico + 32 de decodificacao; 1 aquecimento descartado + 3 repeticoes por processo, mediana). `total_s` = tempo interno (import do GenAI + carga + 4 geracoes); `parede` = medido pelo orquestrador (inclui iniciar o Python). Dispersao = desvio/media entre as 10 execucoes.

| candidato | TTFT p50 ms (media; min-max) | disp. TTFT | tok/s (media; min-max) | disp. tok/s | carga do modelo (s) | total_s (media / max) | parede s (media / max) | ruidosas |
|---|---|---|---|---|---|---|---|---|
| T1 dml-fp16 / 7800 XT | 170 (160-181) | 3.5% | 82.7 (79.4-85.7) | 1.9% | 2.7 | 5.2 / 5.3 | 6.3 / 7.5 | 10/10 |
| T0 dml-fp16 / 7800 XT | 121 (108-127) | 4.0% | 131.2 (124.0-137.7) | 3.1% | 1.7 | 3.5 / 4.1 | 4.2 / 5.4 | 10/10 |
| T0 dml-fp16 / iGPU | 1347 (1236-1407) | 5.4% | 21.2 (19.6-24.0) | 8.5% | 1.7 | 13.3 / 14.0 | 14.9 / 15.3 | 10/10 |
| T1 dml-fp16 / iGPU | 3488 (3403-3744) | 3.4% | 8.1 (7.3-8.6) | 4.9% | 2.9 | 33.0 / 35.2 | 34.4 / 36.5 | 4/10 |
| T0 cpu-int4 / CPU (thr. padrao) | 231 (220-248) | 3.3% | 125.9 (124.0-129.3) | 1.4% | 0.9 | 2.9 / 3.2 | 4.8 / 5.1 | 0/10 |
| T1 cpu-int8 / CPU (thr. padrao) | 755 (731-774) | 1.5% | 32.8 (32.0-33.9) | 1.8% | 1.7 | 8.5 / 8.6 | 10.4 / 10.5 | 0/10 |

**Tempo da calibracao:** dGPU (7800 XT) T1 = 5,2 s internos / 6,3 s de parede; T0 = 3,5 / 4,2 s; baseline T0-CPU = 2,9 / 4,8 s (T1-CPU: 8,5 / 10,4 s). Um candidato de dGPU mais o baseline T0-CPU somam ~8-11 s de
parede em sequencia: **a meta de <= 10 s por candidato e cumprida para dGPU e CPU; NAO para a iGPU lenta** (T0 13,3 s; T1 33 s), que precisa de aborto no aquecimento
(proposta no YAML, `abort_warmup_ttft_factor`, nao medida). Repetibilidade: dispersao de TTFT 1,5-5,4% e de tok/s 1,4-3,1% nas dGPU/CPU (iGPU: 5-9% no tok/s), igual com e sem
carga de fundo (a CPU silenciosa deu 3,3%/1,4% no T0): o micro-benchmark e repetivel a ~5% mesmo sob carga moderada.

**Limiares propostos (aplicados no YAML por serem [M] diretos neste PC):**

| criterio | antes [E] | agora [M] | dado que sustenta | margem |
|---|---|---|---|---|
| T0 TTFT p50 max | 350 ms | **250 ms** | 7800 XT 121 ms (max 127) | 2,0x |
| T0 TTFT GPU/CPU max | 0,7 | **0,7 (mantido)** | 121/231 = 0,52; iGPU 5,8 | 1,3x |
| T0 decode min | 25 tok/s | **60 tok/s** | 7800 XT 131 (min 124); iGPU 21 | 2,1x |
| T1 TTFT p50 max | 250 ms | **250 ms (mantido)** | 7800 XT 170 (max 181); iGPU 3.488 | 1,4x |
| T1 decode min | 40 tok/s | **40 tok/s (mantido)** | 7800 XT 82,7 (min 79,4); iGPU 8,1 | 2,0x |
| T1 TTFT GPU/CPU max | (usava 0,7) | **1,0 (novo, contra o baseline T0-CPU)** | T1/T0cpu = 170/231 = 0,74: com 0,7 a propria 7800 XT reprovaria o T1 | 1,35x |

`microbench.evaluate()` ganhou a chave `t1_accept_ttft_vs_cpu_ratio_max` (o baseline CPU continua sendo o T0 int4 do mesmo PC). Validade: so RDNA3/7800 XT e CPU Zen 4 deste PC; para outras familias
os limiares sao regra de margem, nao medida [E].

## D. Curva de threads (CPU)

IDE usa o padrao do ORT (0). `intra_op_num_threads` = N.

| threads | T0 int4 TTFT media / p95 (ms) | T0 tok/s forcado | T1 int8 TTFT media / p95 (ms) | T1 tok/s forcado | T1: carga antes (%) | T1 ruidosa |
|---|---|---|---|---|---|---|
| 4 | 884 / 1455 | 102 | 2224 / 3528 | 24 | 8.1 | 1/1 |
| 6 | 613 / 1000 | 108 | 1452 / 2358 | 26 | 6.4 | 1/1 |
| 8 | 440 / 707 | 111 | 1133 / 1800 | 26 | 20.0, 8.0 | 2/2 |
| 12 | 313 / 514 | 116 | 786 / 1278 | 33 | 1.1, 1.8 | 0/2 |
| 16 | 256 / 414 | 122 | 685 / 1131 | 29 | 1.1, 1.1, 14.0 | 1/3 |
| 20 | 219 / 350 | 120 | 608 / 972 | 25 | 9.7, 22.2 | 2/2 |
| 24 | 216 / 359 | 106 | 535 / 863 | 28 | 2.6, 1.1 | 0/2 |
| padrao | 310 / 500 | 119 | 791 / 1258 | 27 | 11.8, 15.4 | 2/2 |

Leitura: o TTFT escala quase com 1/N ate 12 threads (T1: 4 thr 2.224 ms, 6 thr 1.452, 8 thr 1.133, 12 thr 786, 24 thr 535; T0: 884, 613, 440, 313, 216) e o SMT ainda ajuda (24 thr = 0,68x o tempo
de 12 no T1, 0,69x no T0). O decode e quase plano (banda): ~100-120 tok/s no T0 e 25-33 no T1. O padrao do ORT (0) equivale a 12 threads (310 x 313 ms no T0; 791 x 786 no T1).
**Validacao do N=12 de `cpu_t1_int8_min_physical_cores`:** com 12 threads o T1 int8 fica em 786 ms de media (p95 1,28 s) e com 8 em 1,13 s (p95 1,8 s); nenhum ponto chega ao gate de autocomplete
(500/650 ms) nem com 24 threads (535/863 ms). O N=12 e **consistente com o dado [M]** (12 = 0,79 s; 8 ja e 1,4x pior) mas e uma escolha de politica sobre "T1 int8 como opcao de chat" (D4):
**valor nao alterado**. Ressalvas: medido em threads de um Zen 4 12C/24T (nao em nucleos fisicos isolados); as execucoes de 4/6/8/20 threads e o padrao do T1 foram ruidosas (carga 6-22%),
mas a curva e monotonica e as execucoes silenciosas (12 e 24 thr) batem com o padrao 1/N.

## E. VRAM de processo (7800 XT)

Pico = maior valor do amostrador de contadores (1 s) durante a geracao; `fim` = sonda apos a bateria completa (100 prompts + 20 decodificacoes forcadas). Valores do relatorio v3 (200 prompts): T1 5.133 MB, T0 2.933 MB.

| carga | dedicada pico (MB) | compartilhada pico (MB) | fim da bateria (MB, dedic./comp.) | TTFT / tok/s |
|---|---|---|---|---|
| T0 dml-fp16, prompts reais (~260 tok) | 2511 | 67 | 2749 / 155 | 104 ms / 173 tok/s |
| T1 dml-fp16, prompts reais | 4813 | 52 | 4947 / 158 | 141 ms / 108 tok/s |
| T0 dml-fp16, **contexto cheio** (prefill 1.900 + decode ate 2.048) | 5428 | 12 | - | 355 ms / 133 tok/s (3 execs) |
| T1 dml-fp16, **contexto cheio** | 7623 | 12 | - | 467 ms / 86 tok/s (3 execs) |

Leitura para D1: com prompts de autocomplete (~260 tokens) o pico medido e **2,5-2,9 GB (T0)** e **4,8-5,1 GB (T1)**; os limiares T0 >= 3.500 MB e T1 >= 5.900 MB cobrem esse regime com 0,6-0,8 GB de folga
(placa de "6 GB" reporta ~6.144 MB). **Com o contexto cheio de 2.048 tokens o pico sobe para 5,4 GB (T0) e 7,6 GB (T1)** (+2,5-2,9 GB de buffers intermediarios do prefill de ~1.900 tokens): uma placa de 6 GB
comporta o T1 so enquanto o prompt for curto, e uma de 4 GB nao comporta o T0 em contexto cheio. Nao medido: a curva pico x comprimento do prompt entre 270 e 1.900 tokens [pendente]. Os limiares D1
**nao foram alterados** (decisao do usuario); estes dados os sustentam para o regime de autocomplete e os limitam para contexto longo.

## 7. Execucao x adaptador pedido x adaptador observado (todas as execucoes DML, exceto a microbateria)

`contadores` = amostrador durante a execucao (pico; `adapter_ok` medido); `pinned_by_arg_only + probe_end` = adaptador fixado so por argumento, observado apos a medida pela memoria do pid; a execucao `evid_*` do mesmo pacote/adaptador fornece o contador. Em `parity_*` sem sonda o observado e `-` (ver `evid_parity_*`).

| execucao | pedido | observado | memoria dedic.+compart. por GPU (MB) | engine % por GPU | evidencia | adapter_ok |
|---|---|---|---|---|---|---|
| ctx_t0-dml-fp16_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=5440 | 1002:747E=93.4 | contadores | true (verify: on_requested_adapter) |
| ctx_t1-dml-fp16_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=7636 | 1002:747E=92 | contadores | true (verify: on_requested_adapter) |
| evid_lat_t0-dml-fp16-mha_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=2497 | 1002:747E=7.3 | contadores | True |
| evid_lat_t0-dml-fp16_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=2510 | 1002:747E=41.9 | contadores | True |
| evid_lat_t0-dml-fp16_igpu | 1002:164E (AMD Radeon(TM) Graphics) | 1002:164E AMD Radeon(TM) Graphics | 1002:164E=2572, 1002:747E=0 | 1002:164E=209.2 | contadores | True |
| evid_lat_t0-dml-fp32_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=3757 | 1002:747E=13.3 | contadores | True |
| evid_lat_t0-dml-int4_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=2234 | 1002:747E=34 | contadores | True |
| evid_lat_t1-dml-fp16_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=4818 | 1002:747E=57.3 | contadores | True |
| evid_lat_t1-dml-fp16_igpu | 1002:164E (AMD Radeon(TM) Graphics) | 1002:164E AMD Radeon(TM) Graphics | 1002:164E=5008, 1002:747E=0 | 1002:164E=451.2 | contadores | True |
| evid_parity_t0-dml-fp16-mha_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=2497 | 1002:747E=7.6 | contadores | True |
| evid_parity_t0-dml-fp16_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=2508 | 1002:747E=32.1 | contadores | True |
| evid_parity_t0-dml-fp32_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=3757 | 1002:747E=12.2 | contadores | True |
| evid_parity_t0-dml-int4_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=2241 | 1002:747E=20.4 | contadores | True |
| lat_t0-dml-fp16-mha_7800xt_r1 | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=2554 | 1002:747E=0 | pinned_by_arg_only + probe_end | True |
| lat_t0-dml-fp16-mha_7800xt_r2 | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=2554 | 1002:747E=0 | pinned_by_arg_only + probe_end | True |
| lat_t0-dml-fp16_7800xt_r1 | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=2904 | 1002:747E=0 | pinned_by_arg_only + probe_end | True |
| lat_t0-dml-fp16_7800xt_r2 | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=2904 | 1002:747E=0 | pinned_by_arg_only + probe_end | True |
| lat_t0-dml-fp16_igpu_r1 | 1002:164E (AMD Radeon(TM) Graphics) | 1002:164E AMD Radeon(TM) Graphics | 1002:164E=2913, 1002:747E=0 | 1002:164E=0 | pinned_by_arg_only + probe_end | True |
| lat_t0-dml-fp16_igpu_r2 | 1002:164E (AMD Radeon(TM) Graphics) | 1002:164E AMD Radeon(TM) Graphics | 1002:164E=2913, 1002:747E=0 | 1002:164E=0 | pinned_by_arg_only + probe_end | True |
| lat_t0-dml-fp32_7800xt_r1 | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=3870 | 1002:747E=0 | pinned_by_arg_only + probe_end | True |
| lat_t0-dml-fp32_7800xt_r2 | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=3870 | 1002:747E=0 | pinned_by_arg_only + probe_end | True |
| lat_t0-dml-fp32_7800xt_r3 | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=3870 | 1002:747E=0 | pinned_by_arg_only + probe_end | True |
| lat_t0-dml-int4_7800xt_r1 | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=2500 | 1002:747E=0 | pinned_by_arg_only + probe_end | True |
| lat_t0-dml-int4_7800xt_r2 | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=2500 | 1002:747E=0 | pinned_by_arg_only + probe_end | True |
| lat_t1-dml-fp16_7800xt_r1 | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=5105 | 1002:747E=0 | pinned_by_arg_only + probe_end | True |
| lat_t1-dml-fp16_7800xt_r2 | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=5105 | 1002:747E=0 | pinned_by_arg_only + probe_end | True |
| lat_t1-dml-fp16_igpu_r1 | 1002:164E (AMD Radeon(TM) Graphics) | 1002:164E AMD Radeon(TM) Graphics | 1002:164E=5236, 1002:747E=0 | 1002:164E=0 | pinned_by_arg_only + probe_end | True |
| lat_t1-dml-fp16_igpu_r2 | 1002:164E (AMD Radeon(TM) Graphics) | 1002:164E AMD Radeon(TM) Graphics | 1002:164E=5236, 1002:747E=0 | 1002:164E=0 | pinned_by_arg_only + probe_end | True |
| lat_t1-dml-fp16_igpu_r3 | 1002:164E (AMD Radeon(TM) Graphics) | 1002:164E AMD Radeon(TM) Graphics | 1002:164E=5236, 1002:747E=0 | 1002:164E=0 | pinned_by_arg_only + probe_end | True |
| parity_t0-dml-fp16-mha_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | - | - | - | pinned_by_arg_only + probe_end | - (sem sonda) |
| parity_t0-dml-fp16_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | - | - | - | pinned_by_arg_only + probe_end | - (sem sonda) |
| parity_t0-dml-fp32_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | - | - | - | pinned_by_arg_only + probe_end | - (sem sonda) |
| parity_t0-dml-int4_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | - | - | - | pinned_by_arg_only + probe_end | - (sem sonda) |
| verify_t0-dml-fp16-mha_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=1952 | 1002:747E=8.3 | contadores | True |
| verify_t0-dml-fp16_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=2578 | 1002:747E=27.7 | contadores | true (verify: on_requested_adapter) |
| verify_t0-dml-fp16_igpu | 1002:164E (AMD Radeon(TM) Graphics) | 1002:164E AMD Radeon(TM) Graphics | 1002:164E=1982, 1002:747E=0 | 1002:164E=221.8 | contadores | true (verify: on_requested_adapter) |
| verify_t0-dml-fp32_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=3179 | 1002:747E=9.4 | contadores | True |
| verify_t0-dml-int4_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=2295 | 1002:747E=29.2 | contadores | true (verify: on_requested_adapter) |
| verify_t1-dml-fp16_7800xt | 1002:747E (AMD Radeon RX 7800 XT) | 1002:747E AMD Radeon RX 7800 XT | 1002:164E=0, 1002:747E=4866 | 1002:747E=43.5 | contadores | true (verify: on_requested_adapter) |
| verify_t1-dml-fp16_igpu | 1002:164E (AMD Radeon(TM) Graphics) | 1002:164E AMD Radeon(TM) Graphics | 1002:164E=4198, 1002:747E=0 | 1002:164E=396.2 | contadores | true (verify: on_requested_adapter) |

Microbateria (60 execucoes): `microbench.py` nao coleta contadores (processo isolado de 4-33 s); cada candidato usou o mesmo par pacote+`--adapter` das execucoes `evid_lat_*` correspondentes (confirmadas acima)
e os tempos sao coerentes com o adaptador pedido (TTFT da iGPU 8-20x o da 7800 XT; o da 7800 XT igual ao das execucoes de latencia).

## 8. Execucoes falhas ou descartadas

- rc != 0 em `phase2_runs.jsonl`: verify_t0-dml-fp32_7800xt, parity_t0-dml-fp32_7800xt, lat_t0-dml-fp32_7800xt_r1, lat_t0-dml-fp32_7800xt_r2, evid_lat_t1-dml-fp16_igpu. As quatro de `dml-fp32` sao o export invalido do builder (§B), **refeitas** com o pacote MHA; a `evid_lat_t1-dml-fp16_igpu` estourou o `timeout` de 120 s do amostrador (bug do harness, corrigido) e foi repetida com sucesso.
- A passada silenciosa da curva T1 foi interrompida (a maquina nunca ficou < 3%).