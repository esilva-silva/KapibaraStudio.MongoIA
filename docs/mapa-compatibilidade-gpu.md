# Mapa de compatibilidade de GPU — qual variante de autocomplete para cada placa

> **Atualização de andamento — 28/10/2026:** o usuário informou que a validação de hardware dos modelos Nano e Small foi concluída com sucesso nos notebooks e ambientes ARM, x86, GPU e CPU adequados. Este registro não inclui a matriz detalhada nem medições; as descrições e pendências históricas abaixo devem ser lidas nesse contexto. Isso não estende o aceite aos demais modelos ou a recursos específicos não mencionados.


Data: 2026-10-26 (fase 2 de medição: `reports/hwmap/phase2_measurements.md`; curva VRAM × contexto: `reports/hwmap/vram_curve.md`; **decisões D1–D5 consolidadas em §9**, YAML schema 2) · Escopo: Windows 11/10, T0 Nano (0,5B) e T1 Small (1,5B), pacotes ONNX v3 (GenAI 0.15.2 = IDE; DirectML fp16/int4, CPU int4/int8).
Tabela de decisão legível por máquina: `configs/gpu_compat.yaml` · detector: `scripts/hwmap/detect_gpus.py` · roteiro de homologação: `docs/roteiro-homologacao-hardware.md` · seleção de adaptador na IDE: `docs/prs/pr-5-selecao-de-adaptador-dml.md`.
Contexto: `docs/fase-07-exportacao-onnx.md` (§2.3, §5), `reports/export/onnx_export_t0_t1.md`, `reports/export/prefix_cache_ttft.md`.

## 0. Convenção de evidência (leia primeiro)

Toda afirmação traz uma marca. **[F]** = fato com fonte (S1…S11 abaixo; acessadas em 2026-10-26). **[M]** = medido por nós (arquivo citado). **[E]** = estimativa/hipótese, com a lógica na própria frase; **[E] nunca é fato**.
**Quase todo este mapa, para além da RX 7800 XT, da iGPU de 2 CU e da CPU Ryzen 9 7900 deste PC, é [E]** (a fase 2 passou a medir esses três: `reports/hwmap/phase2_measurements.md`): não existe medição nossa em Pascal, Turing, Ampere, Ada, Blackwell, RDNA1/2/4, Vega, Intel ou outras iGPUs. A classe de desempenho por família é extrapolada; a **única** defesa contra erro é o micro-benchmark de primeira execução (§6) e a homologação (roteiro).
As faixas de `device_id` do YAML são [E] (de memória/pci-ids, não conferidas contra fonte citada); só `0x747E` (RX 7800 XT) e `0x164E` (iGPU Raphael) são [M] (`Win32_VideoController` deste PC).

| Id | Fonte | Fato usado |
|---|---|---|
| S1 | https://learn.microsoft.com/en-us/windows/ai/directml/dml-intro | DirectML em *sustained engineering*; suportado por todo hardware DX12; introduzido no Windows 10 1903 |
| S2 | https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html | Requisito: NVIDIA Kepler (GTX 600)+, AMD GCN 1ª ger. (HD 7000)+, Intel Haswell+, Qualcomm Adreno 600+; `device_id` = ordem de `IDXGIFactory::EnumAdapters` (o padrão 0 "nem sempre é a mais rápida") |
| S3 | https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers | Windows ML inclui CPU e DirectML (*legacy*); sob demanda (Win11 24H2/build 26100+): MIGraphX (AMD RDNA3+, "não suportado para GenAI hoje"), NvTensorRtRtx (RTX 30xx+, driver ≥ 32.0.15.5585, CUDA 12.5), OpenVINO, QNN, VitisAI, WebGPU (experimental; DX12; "~Intel 11ª ger./NVIDIA Turing+") |
| S4 | https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/run-genai-onnx-models | ORT GenAI para Windows ML é 0.x *preview*, sujeito a mudanças |
| S5 | https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html | GP100: FP16 = 2× FP32; **GP104: FP16 = 1/64 do FP32**; GP104 tem INT8 `__dp4a` |
| S6 | https://www.anandtech.com/show/10325/the-nvidia-geforce-gtx-1080-and-1070-founders-edition-review/5 | FP16 no GP104: "bom para compatibilidade e pouco mais" |
| S7 | https://docs.nvidia.com/cuda/cuda-features-archive/index.html | CUDA 13.0 removeu compilação/bibliotecas para Maxwell, Pascal e Volta; CUDA 12.x ainda compila |
| S8 | https://guru3d.com/story/nvidia-releases-geforce-58266-security-driver-for-maxwell-and-pascal-gpus/ | Maxwell/Pascal/Volta em driver R580 *security-only* (imprensa; confirmar no site NVIDIA) |
| S9 | https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/AMD-Radeon-Adrenalin-Edition-26-5-2-Polaris-Vega-Treiber-1536146/ | Polaris/Vega em driver *legacy* separado (desde 2023), atualizações raras (imprensa) |
| S10 | https://sisoftware.co.uk/?p=2108 | Navi 10 (RX 5700 XT) reporta FP16 a 2× o FP32 |
| S11 | https://www.intel.com/content/www/us/en/developer/articles/guide/windows-ml-support.html | Intel + Windows ML: iGPU Alder Lake+, NPU Core Ultra, Arc A310+ (8 GB recomendado) e B580+ (10 GB recomendado); não cita Iris Xe/UHD por nome |

Fatos já registrados em `docs/fase-07-exportacao-onnx.md` §2.3 (NuGet `Microsoft.ML.OnnxRuntimeGenAI.WinML` 0.16.0/0.17.0/0.17.1 em set/2026; EPs do Windows ML) são citados como [F fase-07].

## 1. Resumo executivo e árvore de decisão

**Conclusão.** O DirectML fp16 que já exportamos é o caminho certo para **GPU dedicada com VRAM suficiente e FP16 nativo**, **com teto de contexto por VRAM**; a CPU T0 int4 é o piso universal; **toda iGPU vai para a CPU nesta versão (D3)**. Os pontos de maior incerteza são: **Pascal (FP16 a 1/64) [E]**, **Intel Arc / outras dGPUs (DML nunca medido por nós) [E]** e **a VRAM do T1 conforme o comprimento do contexto** [M].

- **[M] Referência:** RX 7800 XT 16 GB (624 GB/s): T1 DML fp16 TTFT 148/170 ms (média/p95), 104 tok/s; T0 DML fp16 109/124 ms. CPU (Ryzen 9 7900, 12C/24T): T0 int4 285/480 ms (229/382 com 24 threads), T1 int8 697/1.192 ms. iGPU de 2 CU RDNA2 ≈ 50× mais lenta que a 7800 XT em llama.cpp Vulkan (pp 27 × 1.395 t/s; tg 2,5 × 90 t/s) — `reports/export/onnx_export_t0_t1.md` §4–§5 e medição do usuário. **[M fase 2, GenAI-WinML 0.15.2, 100 prompts reais]:** 7800 XT T0 104 ms / 173 tok/s, T1 141 ms / 108 tok/s; CPU T0 int4 216 ms (24 thr) / 313 ms (12 thr) / 106–116 tok/s, T1 int8 535 ms (24 thr) / 786 ms (12 thr) / 28–33 tok/s; **iGPU 2 CU: T0 1.338 ms / 23 tok/s, T1 3.713 ms / 7,3 tok/s** (a CPU vence a iGPU em todos os casos).
- **[M] VRAM de processo × contexto (curva, n = 10 por ponto, `reports/hwmap/vram_curve.md`):** pico dedicado **máximo** para contexto total de 302 / 544 / 800 / 1.056 / 1.568 / 2.048 tokens — **T0** 1.858 / 2.337 / 2.465 / 3.246 / 3.510 / 5.560 MB; **T1** 4.064 / 4.526 / 4.654 / 5.452 / 5.708 / **7.759 MB**. Os limiares originais (T1 ≥ ~6 GB, T0 ≥ 3,5 GB) valem para o regime de autocomplete (prompts de ~270–700 tokens); o **contexto maior é limitado por uma regra de teto por VRAM** (D1): T1 dml-fp16 só em dGPU com **≥ 8 GB** (contexto cheio de 2.048 → YAML `t1_full_min_vram_mb` 8.150 MB; 8 GB reporta ~8.176); entre **~6 e 8 GB** (6.100–8.149 MB) só T1 com **teto de 1.536 tokens e se o micro-benchmark aprovar**, senão T0; **T0 dml-fp16 com ≥ 3,5 GB** com teto proporcional (3.500 → 768, 3.650 → 1.024, 3.900 → 1.536, ≥ 5.950 → 2.048); < 3,5 GB ou sem DX12 → CPU.
- **Regra geral:** dGPU elegível e `supported ≠ false` → T1 dml-fp16 (VRAM ≥ 8.150 MB, ou 6.100–8.149 com teto e micro-benchmark) ou T0 dml-fp16 (≥ 3.500 MB, teto proporcional); família `unknown` (Pascal, Polaris, nome novo), Intel Arc e Blackwell → mesma variante **só após o micro-benchmark** (Pascal/Polaris reprovando → CPU T0 int4 direto; **sem variante fp32**, D2); **iGPU → sempre CPU** (`igpu_policy: cpu_always_until_measured`, D3; 680M/780M/880M/890M/8060S, Arc de Core Ultra e Iris Xe **não são candidatas** até haver medição real); CPU sempre T0 int4 no autocomplete; **T1 int8 em CPU só como modo opcional e declaradamente lento, escolha do usuário, nunca padrão** (≥ 12 núcleos físicos e ≥ 16 GB; não cumpre o gate de TTFT: 535 ms a 24 thr, 786 ms a 12 — D4). **Nenhuma variante nova agora (D5).** A saída do detector inclui `context_cap_tokens`.
- **Seleção do adaptador (achado de produto):** o GenAI-WinML 0.15.2 ignora `device_id`/`device_filter`/`performance_preference`/`device_filtering_options` e usa o adaptador padrão do DirectML (neste PC, a 7800 XT; num notebook híbrido, provavelmente a iGPU). A IDE deve validar o adaptador **efetivo** e cair para a CPU se for iGPU/WARP/lento: `docs/prs/pr-5-selecao-de-adaptador-dml.md`.

```
início ─ enumerar adaptadores DXGI (nunca pelo índice: vendor/device/VRAM)
  │
  ├─ descartar: software/WARP (0x1414), VM, supported=false
  │
  ├─ iGPU (qualquer uma)? ──► IGNORAR / CPU  (D3: igpu_policy cpu_always_until_measured; só uma dGPU elegível assume)
  │
  ├─ há dGPU (não integrada) elegível?  (VRAM dedicada do DXGI)
  │     ├─ VRAM ≥ 8.150 MB (placa de 8 GB+) ───► T1 dml-fp16, contexto até 2.048          ┐ família medida/alta confiança → aceita direto [E];
  │     ├─ 6.100 ≤ VRAM < 8.150 (6–8 GB) ──────► T1 dml-fp16 com teto 1.536 SÓ se o      │ senão (Pascal, Polaris, Intel, Blackwell, unknown)
  │     │                                         micro-benchmark aprovar; senão T0       │ → micro-benchmark antes de aceitar
  │     ├─ 3.500 ≤ VRAM < 6.100 ────────────────► T0 dml-fp16, teto 768/1.024/1.536/2.048 ┘ (Pascal/Polaris reprovando → CPU T0 int4, D2)
  │     └─ VRAM < 3.500 MB ────────────────────► CPU
  │
  └─ CPU: T0 cpu-int4 (autocomplete, sempre). T1 cpu-int8 só como modo opcional e lento, escolha do usuário (≥ 12 núcleos físicos e ≥ 16 GB), nunca padrão (D4)
  fallback de qualquer falha de DML/GPU ─► T0 cpu-int4
  IDE: validar o adaptador DXGI EFETIVO (PR-5) — o GenAI 0.15.2 ignora device_id
```

## 2. Tabela por família

Colunas: **DX12 / driver** (mínimo), **FP16**, **INT8/matriz**, **VRAM / banda** (faixa típica), **classe T0 / T1 [E]** (A = como a 7800 XT em ordem de grandeza; B = ~2–3× pior; C = pior que a CPU), **variante**, **riscos**. Todas as colunas de classe, VRAM e banda são [E] (conhecimento de especificação de produto, não medido), exceto onde marcado.
Mínimos de plataforma comuns [F]: Windows 10 1903+ para o DirectML (S1); DX12 (S1, S2); GenAI WinML é *preview* 0.x (S4). Driver: usar sempre o WHQL mais recente da família — não fixamos número de driver por falta de fonte [E].

### 2.1 NVIDIA (PCI 0x10DE)

| Família | DX12 / driver | FP16 | INT8 / matriz | VRAM · banda | Classe T0 / T1 | Variante (regra) | Riscos |
|---|---|---|---|---|---|---|---|
| GTX 9xx e anteriores (Maxwell/Kepler/Fermi) | DX12 FL 11_0–12_1; DML roda em Kepler+ [F S2]; Maxwell em driver *security-only* [F S8] | sem ganho | sem dp4a | 2–6 GB · 80–225 GB/s | C / C | **NÃO SUPORTADA** → CPU T0 int4 | política: abaixo da fronteira GTX 1000 [E] |
| **GTX 10 (Pascal GP10x)**, MX150/250, GT 1030 | FL 12_1; driver R580 *security-only* [F S8] | **1/64 do FP32 nas GP10x de consumo [F S5, S6]** | `dp4a` existe [F S5]; uso pelo DML [E] desconhecido | 2–11 GB · 48–484 GB/s (1060 6 GB: 192) | T0: B/C ?; T1: C ? **[E] — sem medida** | `unknown` (D2) → T1 (6.100–8.149 MB: teto 1.536; ≥ 8.150: 2.048) ou T0 (≥ 3.500 MB) **só com micro-benchmark**; **reprovou → CPU T0 int4** (sem fp32). GT 1030/MX/1050 2 GB → CPU | **maior risco do mapa**: se o DML executar FP16 nativo no Pascal, o prefill de ~270 tokens do T1 (~0,8 TFLOP) levaria > 10 s (1060: FP32 4,4 TFLOPS → FP16 ~70 GFLOPS) [E]; se converter para FP32 em shader, é aceitável [E]. Sem variante fp32 (D2; o builder 0.15.2 não gera FP32+DML válido [M]) |
| GTX 16 (Turing TU116/TU117) | FL 12_1 | 2× FP32 [E] (sem Tensor Cores) | dp4a [E] | 4–6 GB · 112–336 GB/s | T0 B; T1 B | 4 GB (1650) → **T0 dml-fp16**, teto 1.536; 6 GB (1660/Super/Ti) → faixa com teto: T1 teto 1.536 só com micro-benchmark aprovado, senão T0 | 6 GB reportado ~6.144 MB (piso do T1 = 6.100): margem pequena |
| RTX 20 (Turing) | FL 12_1 | 2× [E] | dp4a + Tensor [E] | 6–24 GB · 336–672 GB/s | A/B | T1 dml-fp16 (≥ 8.150 MB: 2.048; 6 GB: teto 1.536 + micro-benchmark) | RTX 2060 6 GB na faixa com teto |
| RTX 30 (Ampere) | FL 12_2 | sim | Tensor gen 3 | 6–24 GB · 168–936 GB/s | A (3060+) | T1 dml-fp16 | 3050 6 GB (96–168 GB/s): faixa com teto (T1 1.536 + micro-benchmark) e banda baixa — T1 decode pode cair <~40 tok/s [E] |
| RTX 40 (Ada) | FL 12_2 | sim | Tensor gen 4 | 8–24 GB · 272–1.008 GB/s | A | T1 dml-fp16 | — |
| RTX 50 (Blackwell) | FL 12_2 | sim | Tensor gen 5 | 8–32 GB · 448–1.792 GB/s | A [E] | T1 dml-fp16 **com micro-benchmark** | hardware/driver recentes, sem teste nosso; DML "legacy" pode ficar atrás [E] |
| MX/laptops Max-Q | variam | variam | variam | 2–8 GB | variam | por device_id; "unknown" → micro-benchmark | híbrido NVIDIA+iGPU: o adaptador padrão do Windows costuma ser a iGPU [F S2: "device 0 nem sempre é a mais rápida"] → sempre escolher por vendor/device/VRAM |

### 2.2 AMD (PCI 0x1002)

| Família | DX12 / driver | FP16 | INT8 / matriz | VRAM · banda | Classe T0 / T1 | Variante | Riscos |
|---|---|---|---|---|---|---|---|
| Pré-Polaris (GCN1–3, TeraScale, Fury) | FL 11_1–12_0; DML aceita GCN 1ª ger.+ [F S2] | não | não | 2–4 GB | C / C | **NÃO SUPORTADA** → CPU | drivers abandonados [E] |
| **Polaris (RX 400/500)** — *abaixo de Vega* | FL 12_0; driver *legacy* [F S9] | sem Rapid Packed Math [E] | sem dp4a [E] | 4–8 GB · 112–256 GB/s | ? / ? **[E]** | `unknown` (D2): T1 (8 GB) ou T0 **só com micro-benchmark**; reprovou → CPU T0 int4 | RX 580 é popular: homologar; sem fp32 |
| **Vega** (RX Vega 56/64, Radeon VII) | FL 12_1; driver *legacy* [F S9] | 2× (RPM) [E] | limitado [E] | 8–16 GB HBM2 · 410–1.024 GB/s | B / B (banda alta) [E] | **T1 dml-fp16** (8 GB: contexto 2.048) | driver legacy: bugs DX12 não corrigidos [E]; Vega 56 8 GB é candidata forte |
| iGPU Vega 3–11 (Raven/Picasso/Renoir/Cezanne/Barcelo) | FL 12_1 | 2× [E] | limitado | compartilhada (DDR4 38–51 GB/s) | C / C | **CPU** | ≤ ~2,2 TFLOPS e banda de DDR4: pior que a CPU [E] |
| RX 5000 (RDNA1) | FL 12_1 | 2× [F S10] | dp4a em Navi 10 incerto [E] | 4–8 GB · 112–448 GB/s | B / B [E] | 4 GB (5500 XT) → T0 (teto 1.536); 6 GB → T1 teto 1.536 + micro-benchmark; 8 GB → T1 | sem medida; sem WMMA |
| RX 6000 (RDNA2) | FL 12_1/12_2 | 2× [E] | dot4; sem WMMA | 4–16 GB · 144–512 GB/s | B–A [E] | 6400/6500 XT (4 GB) → T0 (teto 1.536); 6600/6600 XT/6650 XT (8 GB) → T1 (contexto 2.048); 6 GB (RX 6600 LE/OEM) → T1 teto 1.536 + micro-benchmark | prioridade nº 1 de homologação |
| **RX 7000 (RDNA3)** | FL 12_2 | 2× | WMMA | 8–24 GB · 288–960 GB/s | **A [M] (7800 XT)** | **T1 dml-fp16 [M]** | 7600 (288 GB/s) deve ficar ~2× atrás da 7800 XT no decode [E] |
| RX 9000 (RDNA4) | FL 12_2 | 2× | WMMA gen 2 | 8–32 GB · 320–640 GB/s | A [E] | T1 dml-fp16 com micro-benchmark | sem teste nosso |
| iGPU 610M / "Radeon Graphics" 2 CU (Raphael) | FL 12_x | sim | dot4 | compartilhada | **C [M]** (piso) | **CPU** (D3) | medida neste PC: ~50× mais lenta que a 7800 XT em llama.cpp; 13× (T0) / 26× (T1) no DML e a CPU vence em todos os casos |
| iGPU 660M/680M (RDNA2 6–12 CU) | FL 12_x | sim | dot4 | LPDDR5/DDR5 75–100 GB/s | B/C [E] | **CPU** (D3: não candidata até medir; `candidate_after_measurement` T0 dml-fp16) | empata com CPU de desktop; talvez ganhe de CPU de notebook 4–8 núcleos [E], sem medida |
| iGPU 780M/760M/880M/890M (RDNA3/3.5 8–16 CU) | FL 12_x | sim | WMMA | 90–120 GB/s | B [E] | **CPU** (D3: não candidata até medir; depois só T0, **T1 não**) | banda compartilhada limita decode do T1 a ~25–35 tok/s [E] |
| iGPU 8060S (Strix Halo, 40 CU) | FL 12_x | sim | WMMA | até ~256 GB/s; VRAM = *carve-out* configurável | B/A [E] | **CPU** (D3: não candidata até medir) | `DedicatedVideoMemory` não é confiável (0,5–96 GB) → usar RAM total quando for medida |

### 2.3 Intel (PCI 0x8086)

| Família | DX12 / driver | FP16 | INT8 / matriz | VRAM · banda | Classe T0 / T1 | Variante | Riscos |
|---|---|---|---|---|---|---|---|
| Pré-Gen9 (HD 2000–5000, Gen6–8) | DML "Haswell+" [F S2]; sem FP16 útil | não | não | compartilhada | C / C | **NÃO SUPORTADA** → CPU | — |
| UHD/HD Gen9–Gen11, UHD Xe-LP 24–32 EU (UHD 630/730/750/770) | FL 12_0/12_1 | 2× [E] | dp4a Gen11+ [E] | DDR4 38–51 GB/s; 0,4–0,8 TFLOPS | C / C [E] | **CPU** | menos compute que o piso medido (2 CU ≈ 1,1 TFLOPS) [E] |
| Iris Xe (Gen12 80–96 EU) | FL 12_1; iGPU Alder Lake+ citada no Windows ML [F S11] | 2× | dp4a | LPDDR4x/DDR4 51–68 GB/s; ~2 TFLOPS | C/B [E] | **CPU** (D3: não candidata até medir) | pode empatar com CPU de notebook fraco, sem medida |
| Arc A-series (Alchemist) | FL 12_2; Arc A310+ citada [F S11] | sim | dp4a + XMX | 6–16 GB · 186–560 GB/s | A/B [E] | A750/A770 (8–16 GB) → T1; A380 (6 GB) → faixa com teto (T1 1.536 só com micro-benchmark; banda 186 GB/s), senão T0; A310 (4 GB) → T0 teto 1.536 — sempre **com micro-benchmark** | histórico de drivers DX12 irregular [E]; DML nunca medido por nós |
| Arc B-series (Battlemage) | FL 12_2; B580+ citada [F S11] | sim | XMX | 10–12 GB · 456 GB/s | A [E] | T1 com micro-benchmark | hardware recente, sem teste |
| iGPU Arc de Core Ultra (Meteor/Arrow-H/Lunar Lake) | iGPU 12ª ger.+ [F S11] | sim | XMX (Lunar) / dp4a | LPDDR5X 90–135 GB/s; ~4–4,6 TFLOPS | B [E] | **CPU** (D3: não candidata até medir) | Arrow Lake desktop (4 Xe cores, ~2 TFLOPS) casa pelo mesmo nome e é mais fraca [E] |

## 3. iGPU × CPU

**Dado de piso [M]:** 2 CU RDNA2 (≈ 1,1 TFLOPS FP32 de pico [E]): pp 27 t/s no 7B Q5_K_M ⇒ ~0,38 TFLOPS efetivos (2·7e9·27) [E, derivado]. O mesmo cálculo na 7800 XT dá ~19 TFLOPS efetivos (53% de ~37 de pico).
**Medido na fase 2 [M] (ONNX/DML, prompts reais de ~260 tokens):** T0 dml-fp16 na iGPU TTFT 1.338 ms (p95 2.223), 23 tok/s forçado, vs CPU 216–313 ms e 106–119 tok/s e 7800 XT 104 ms / 173 tok/s; T1: iGPU 3.713 ms / 7,3 tok/s vs CPU int8 535–786 ms / 28–33 tok/s e 7800 XT 141 ms / 108 tok/s. A estimativa de "~0,7 s de prefill" abaixo era otimista (real 1,3 s). A iGPU usa ~370–410 MB dedicados + 2,2–4,9 GB compartilhados; RAM de pico 3,5 GB (T0) / 5,7 GB (T1).
**Extrapolação para o T0 [E; para 2 CU substituída pela medida acima]** (0,5B ⇒ ~2·0,5e9·270 = 0,27 TFLOP de prefill [E]): iGPU de 2 CU ≈ 0,7 s de prefill (> CPU 285 ms [M]) → CPU vence. Uma iGPU de 12 CU RDNA3 (~8 TFLOPS de pico) levaria ~0,1 s no regime do llama.cpp; mas o DML do ORT atingiu só ~2,5 TFLOPS efetivos no T0 e ~5,5 no T1 da 7800 XT [M, derivado: 109 ms e 148 ms para 0,27/0,81 TFLOP, com sobrecarga fixa de dezenas de ms] — logo uma 780M tende a **empatar** (~250–350 ms) com a CPU de desktop de 12 núcleos e a ganhar de CPU de notebook (4–8 núcleos: 400–700 ms [E, da escala 4thr 731/8thr 406 ms]). **Conclusão [E] da versão anterior (substituída pela decisão D3, §9):** a extrapolação errou por ~2× na única iGPU medida (previa 0,7 s, mediu 1,3 s), então ela **não** é base para colocar uma iGPU maior em produção. **D3: toda iGPU vai para a CPU nesta versão** (`igpu_policy: cpu_always_until_measured`); as iGPUs maiores (680M/780M/880M/890M/8060S, Arc de Core Ultra, Iris Xe) **não são candidatas** até haver medição real nelas pelo roteiro de homologação, e só então a política pode ser revista (≥ 30% de ganho sobre a CPU do mesmo PC era o critério proposto [E]).
**Banda [E, especificação]:** DDR4-3200 dual = 51 GB/s; DDR5-5600 dual = 90; LPDDR5X-7500 128-bit = 120; GDDR6 da 7800 XT = 624 [M]. O decode do T1 fp16 (3 GB lidos/token) fica limitado a ~17 tok/s em 51 GB/s, ~30 em 90, ~40 em 120 [E, 3 GB/token] — por isso **T1 em iGPU não é recomendado**. O decode do T0 (1 GB/token) = 50/90/120 tok/s [E] é suficiente.
**CPU também disputa a mesma RAM**: iGPU e CPU dividem banda; o ganho da iGPU é de compute (prefill), não de banda.
Em PC com dGPU, a iGPU é **sempre ignorada** (e sem dGPU elegível também vai para a CPU, D3) (teste `test_this_pc_ignores_igpu_and_warp_regardless_of_order`), como o contador de GPU confirmou [M] (0 MB e 0% na iGPU, `onnx_export_t0_t1.md` §2).

## 4. Mapa de NÃO SUPORTADAS (fronteira)

| Hardware | Motivo | Comportamento |
|---|---|---|
| NVIDIA Fermi/Kepler/Maxwell (GTX 400–900, GT 7xx, GTX 750) | abaixo de GTX 1000 por política [E]; DML roda em Kepler+ [F S2] mas sem FP16 acelerado; Maxwell *security-only* [F S8] | CPU T0 int4; override avançado manual |
| NVIDIA/AMD/Intel dGPU com < 3,5 GB (GTX 1050 2 GB, GT 1030, MX150/250 2 GB, 1060 3 GB) | VRAM < piso do T0 (D1: 3,5 GB; pico 1,9–2,9 GB no regime típico e 5,6 GB em contexto cheio [M]) | CPU |
| AMD pré-Polaris (GCN1–3, TeraScale, R9 200/300, Fury) | sem FP16, drivers abandonados [E]; DML aceita GCN1+ [F S2] | CPU |
| AMD Polaris (RX 4xx/5xx) e NVIDIA Pascal (GTX 10) | **fronteira "unknown"** (D2), não "não suportada": GCN4 sem FP16 empacotado, driver *legacy* [F S9]; Pascal GP10x FP16 a 1/64 [F S5] | micro-benchmark; **reprovou → CPU T0 int4** (sem variante fp32) |
| Intel pré-Gen9 / sem DX12 | sem DX12 pleno/FP16 [E] | CPU |
| Intel UHD/HD Gen9–11, UHD 24–32 EU | suportado mas "não recomendado": mais lento que a CPU [E] | CPU |
| **Toda iGPU** (inclui 680M/780M/880M/890M/8060S, Arc de Core Ultra, Iris Xe, Vega 3–11, 610M) | D3: a iGPU de 2 CU mediu 13× (T0) / 26× (T1) mais lenta que a 7800 XT e perde para a CPU em todos os casos [M]; as maiores são [E] e **não são candidatas até haver medição real** | CPU (`igpu_policy: cpu_always_until_measured`) |
| Adaptadores de software (Microsoft Basic Render Driver, WARP), Hyper-V/RDP sem GPU | sem hardware DX12 [F S1] | CPU |
| Qualcomm Adreno / NPUs (Ryzen AI, Intel NPU, Snapdragon) | fora desta versão (§7) | CPU |
| Windows < 10 1903 | DirectML introduzido no 1903 [F S1] | CPU |
| VMs/passthrough parcial, Linux/macOS | fora de escopo do build WinML | CPU |

## 5. Precisamos exportar outras variantes?

### 5.1 Resposta objetiva às quatro perguntas

**(i) O DirectML fp16 cobre tudo que é DX12?** **Em compatibilidade, sim [F S1, S2]; em *desempenho aceitável*, não.** DML carrega em qualquer GPU DX12, mas (a) FP16 lento (Pascal GP10x, Polaris) pode inviabilizá-lo [E]; (b) VRAM abaixo de ~3,5/5,9 GB [M]; (c) iGPUs são mais lentas que a CPU [M/E]. O mapa existe exatamente para isso. Custo adicional zero: o DML já está no build WinML da IDE.

**(ii) Precisamos de DML fp32 ou int8 para GPUs de FP16 fraco (Pascal)?** **Recomenda-se testar antes de exportar; não exportar já.**
- *Pascal GP10x:* FP16 a 1/64 [F S5]. O ORT/DML pode converter para FP32 (nesse caso o `dml-fp16` já serve) ou executar FP16 lento [E]. A GTX 1060 (6 GB) é o caso crítico: o T1 em fp32 pesa ~5,9 GB (`fp32-cpu` é 5.916 MB [M]) e não cabe; o **T0 em fp32 (~1,9 GB; `fp32-cpu` 1.904 MB [M])** caberia em ≥ 4 GB. O `dml-int4` (T0 544 MB; T1 1,29 GB [M]) usa ativações fp16 (não resolve o FP16 lento) e o T1 int4 perde 1,4–1,8 p.p. (fora de cogitação).
- *Medido na fase 2 [M] (7800 XT, T0; `phase2_measurements.md` §B):* o builder 0.15.2 **não exporta `dml-fp32` válido** (FP32+DML não está na lista de combinações válidas; o grafo sai sem `present.*` e o ORT recusa o carregamento). Com monkeypatch (MHA, sem graph capture) obtém-se um fp32 de 1.908 MB (2× o fp16), +1,3 GB de VRAM, prefill +~35% sobre o fp16 no mesmo caminho de atenção, paridade igual (94% × 95%), mas decode de ~9 tok/s tanto em fp32 quanto em fp16 no caminho MHA (o fp16 padrão GQA+graph capture faz 173 tok/s): a lentidão é do caminho de atenção, não da precisão (~5%). `dml-int4`: 548 MB, TTFT +29%, decode +30%, **paridade greedy 72%** (reprovado). O benefício do fp32 em Pascal continua **[E]** (sem GPU).
- *Teste recomendado em Pascal (continua [PENDENTE]):* numa GP106/GP104, rodar `onnx_run.py --mode latency` com `dml-fp16` (T0 e T1) e, **se o `dml-fp16` for lento**, gerar `dml-fp32` do T0 (`build_variants.py`, `-p fp32 -e dml`; não suportado pelo `build_variants.py` atual, que só tem `dml-fp16` e `dml-int4` [M: `scripts/export/README.md`]) e comparar. DML **int8** não existe no builder 0.15.2 (o `-p int8` gera grafo inválido e o contorno é CPU-only [M]) → descartar.
- **Decisão D2 (DECIDIDO, 2026-10-26):** Pascal = `unknown` decidido pelo micro-benchmark; **sem variante fp32** (o builder 0.15.2 não aceita FP32+DML e o remendo cai a ~9 tok/s); reprovou → CPU T0 int4.

**(iii) Backend/export extra para NVIDIA, Intel, AMD?** **Não agora.**
| Opção | O que exigiria | Ganho [E] | Custo/risco | Veredito |
|---|---|---|---|---|
| ORT GenAI **CUDA** (`Microsoft.ML.OnnxRuntimeGenAI.Cuda`) | build `Cuda` da IDE (o `.csproj` rejeita outros [F fase-07 §2.3]); CUDA Toolkit instalado + `CUDA_PATH` [F onnxruntime.ai/docs/genai/howto/install.html]; pacotes `cuda-fp16/int4` | NVIDIA 30+/40+ talvez 1,5–2× o DML [E] | DLLs da NVIDIA (cuBLAS/cuDNN, centenas de MB a GB [E]), CUDA 13 abandonou Pascal [F S7], 2º build/QA/release; DML já atinge o gate do T1 na 7800 XT [M] | **NÃO** |
| **TensorRT-RTX** (Windows ML EP) | RTX 30xx+, driver ≥ 32.0.15.5585 [F S3]; EP sob demanda Win11 24H2+; compila engines (JIT) | alto em RTX [E] | exige a IDE registrar EPs (hoje não registra [F fase-07 §2.3]); suporte a GenAI não confirmado nas fontes [E]; 24H2+ | **NÃO agora**; spike após registrar EPs |
| **OpenVINO** (Windows ML EP) | Intel iGPU 12ª ger.+/Arc [F S11]; export OpenVINO/ONNX específico | ganho em Intel iGPU/Arc [E] | sem máquina Intel de teste; GenAI+OpenVINO não validado [E]; fase-07 §5.6 já define "investigação" | **NÃO**; spike de 1 dia quando houver máquina Arc/Core Ultra |
| **MIGraphX/VitisAI** (AMD) | RDNA3+ / Ryzen AI NPU [F S3] | — | MIGraphX "não suportado para GenAI hoje" [F S3] | **NÃO** |
| **llama.cpp Vulkan/GGUF** | outro runtime (C#↔nativo), outro empacotamento, reavaliar paridade/qualidade GGUF | cobre Pascal/Polaris/iGPU/Intel/AMD velhos com um só backend; medimos Vulkan rodando na 7800 XT e na iGPU [M] | 2º stack a manter, eval próprio, licenças/binários; contradiz o contrato `kapilab-kapi-model-v1` baseado em GenAI [E] | **NÃO agora**; plano B se o micro-benchmark mostrar que DML falha em Pascal/Polaris/Intel |
| WebGPU EP (experimental) | Windows ML ≥ 2.4.66-preview; "Intel 11ª ger./NVIDIA Turing+" [F S3] | possível sucessor do DML [E] | experimental | acompanhar |

**(iv) DirectML *legacy* e o ORT GenAI 0.15.2 da IDE.** [F] DML está em *sustained engineering*: "continua suportado", sem novas funcionalidades (S1, S2); o Windows ML o inclui como *legacy* (S3); o GenAI-WinML é 0.x *preview* (S4); o NuGet já está em 0.17.1 (set/2026) [F fase-07 §2.3]. **Não encontrei (2026-10-26) data de remoção anunciada.** Risco de médio prazo: **médio** [E] — o DML não some de uma hora para outra, mas (a) correções para GPUs novas (RTX 50/RDNA4/Battlemage) podem não vir; (b) o upgrade do GenAI (0.16/0.17) pode mudar o comportamento do DML sem aviso. Recomendação:
1. manter DML como alvo GPU (única via com cobertura multi-vendor num só pacote, com gate T1 já batido [M]);
2. **fixar** GenAI 0.15.2 e chavear o cache do micro-benchmark pela versão do GenAI/driver (§6) — um upgrade da IDE reavalia sozinho;
3. antes de qualquer upgrade (R1 da Fase 08) repetir a homologação nas 3 famílias (RDNA, Ampere+, Intel);
4. registrar um spike do EP WebGPU e do registro de EPs do Windows ML como "sucessor candidato", sem criar pacote;
5. gatilhos para reabrir esta análise: aviso de remoção do DML, regressão de DML numa família coberta, ou ≥ 10% dos usuários caindo em CPU por falha de DML.

### 5.2 Recomendação de exports

| | Itens |
|---|---|
| **Manter** | T0 `dml-fp16`, T1 `dml-fp16`, T0 `int4` (CPU), T1 `int8` (CPU) — os quatro já medidos e suficientes para o mapa |
| **Adicionar** | **Nada agora (D5).** `dml-fp32` do T0 deixou de ser condicional (D2): não é gerável pelo builder 0.15.2; reavaliar quando houver builder/ORT GenAI mais novo |
| **NÃO fazer** | CUDA / TensorRT-RTX / OpenVINO / MIGraphX / GGUF como variantes agora; `dml-int4` para o T1 (−1,8 p.p., [M]); exports por vendor; pacote por família |
| **Spikes (sem pacote)** | OpenVINO numa Arc/Core Ultra; WebGPU EP; registro de EPs do Windows ML |

## 6. Micro-benchmark de primeira execução e fallback seguro (desenho)

**Quando:** no primeiro uso de uma variante GPU com `needs_microbench: true` (qualquer família `unknown`, Pascal, Polaris, Blackwell, Intel Arc, e **a faixa de 6–8 GB do T1, D1**; iGPUs não passam por ele: D3) e quando o cache invalida. Para famílias medidas/de alta confiança (RDNA3, Ampere/Ada) roda em segundo plano, sem bloquear.
**Orçamento:** 5–10 s no total, em **processo filho isolado por candidato** (um processo por candidato; o fato de carregar um 2º modelo DML no mesmo processo já derrubou o processo [F fase-07 §2.3]).
**Passos por candidato:** (1) carregar o pacote com o adaptador escolhido por LUID/vendor/device (nunca pelo índice) e conferir por contador que a memória dedicada subiu **nesse** LUID e não na iGPU (como `gpu_probe.ps1`/`onnx_run.py --mode verify` [M]); (2) 1 aquecimento descartado; (3) prefill de ~270 tokens (mediana medida 260; `microbench.prompt_tokens`) e 32 tokens de decodificação, 3 repetições; (4) mediana de TTFT e tok/s; (5) também medir a CPU T0 int4 (a mesma carga) se ainda não houver medida da CPU neste PC.
**Aceite (calibrado na fase 2, [M] só para 7800 XT/RDNA3 e CPU Zen 4 deste PC; ver `microbench` no YAML):** GPU aceita se `TTFT_p50 ≤ 250 ms` (T0; era 350 [E]) / `≤ 250 ms` (T1), `decode ≥ 60 tok/s` (T0; era 25 [E]) / `≥ 40` (T1), **e** `TTFT_gpu ≤ 0,7 × TTFT_cpu(T0 int4)` (T0) / `≤ 1,0 ×` (T1: 170/231 = 0,74; com 0,7 a própria 7800 XT reprovaria). Dados (10 execuções, processo novo cada): 7800 XT T1 170 ms (160–181) e 82,7 tok/s; T0 121 ms (108–127) e 131 tok/s; CPU T0 231 ms; iGPU T0 1.347 ms / 21 tok/s, T1 3.488 ms / 8,1 tok/s. Tempo por candidato (parede): 7800 XT T1 6,3 s, T0 4,2 s, CPU T0 4,8 s, CPU T1 10,4 s, **iGPU T0 14,9 s, T1 34 s** (a iGPU lenta estoura os 10 s ⇒ abortar no aquecimento, proposta). Dispersão entre execuções: TTFT 1,5–5,4%, tok/s 1,4–3,1% (iGPU 5–9%). Se o T1 falhar mas o T0 passar → T0 DML; se ambos falharem → CPU T0 int4.
**Cache:** JSON por usuário, chave `adapter(vendor,device,subsys,revision) + driver_version + genai_version + hash do pacote + build do Windows`; conteúdo: variante aceita, TTFT/tok/s, data. **Reavaliar** quando qualquer item da chave mudar, quando o fallback ocorrer 3 vezes seguidas, ou por botão "Recalibrar" na IDE; expira em 90 dias [E].
**Fallback seguro:** qualquer falha no caminho DML → **T0 CPU int4**, sem erro visível: falha ao criar a sessão (ex.: `80070057`/`DmlFusedNode`, como já visto com k-quant em DML [F fase-07 §2.3]), `DXGI_ERROR_DEVICE_REMOVED/HUNG` (TDR), falta de VRAM/OOM, timeout de carga (> 8 s) ou de primeira geração (> 3 s), saída vazia/lixo (checagem de paridade de 5 tokens contra a CPU), e *watchdog* em produção (TTFT p95 > 2× o calibrado em 3 sugestões seguidas → desliga a GPU na sessão). Três falhas em sessões distintas → marcar `gpu_disabled` no cache até mudar driver/GPU/versão. O usuário pode forçar GPU/CPU nas configurações.
**Referência de implementação:** `scripts/hwmap/microbench.py` (executor rodado de verdade na fase 2 e `evaluate()` testado; `--adapter vendor:device` + `--allow-igpu` medem uma iGPU de propósito via `scripts/hwmap/dxgi_select.py`, pois o GenAI-WinML 0.15.2 ignora `device_id`) e `scripts/hwmap/detect_gpus.py` (seleção por vendor/device/VRAM).

## 7. NPU (fora desta versão)

NPUs (Ryzen AI, Intel NPU, Snapdragon) **ficam fora**. Estado [F S3, fase-07 §2.3]: o Windows ML entrega por EP sob demanda (Win11 24H2/26100+) QNN (Snapdragon X Elite/Plus X1E/X1P, driver Hexagon ≥ 30.0.140.0), VitisAI (Ryzen AI; Adrenalin 25.6.3–25.9.1 + driver NPU 32.00.0203.280–297) e OpenVINO (NPU Core Ultra, Intel guide S11). O **T3 QNN** (Snapdragon X1-26-100, X1E/X1P/X2 fora da lista) é trabalho separado (Fase 07 §5.5/§8). O detector não classifica NPUs.

## 8. Medições da fase 2: feitas e ainda pendentes

Feitas neste PC (`reports/hwmap/phase2_measurements.md/.json`; n=100 prompts reais por execução, ≥ 2 execuções; carga de fundo e execuções ruidosas registradas lá):
1. **[M] iGPU 2 CU:** T0 e T1 `dml-fp16` na iGPU (selecionada por vendor/device e confirmada por contadores de memória/engine por pid) × CPU × 7800 XT (§3).
2. **[M] Precisão no DML (T0, 7800 XT):** fp16 × int4 × fp32/MHA (e controle fp16/MHA): tamanho, TTFT, tok/s, VRAM, paridade (§5.1 (ii)). Pascal real: **[E]**.
3. **[M] Micro-benchmark** (10 execuções por candidato, 6 candidatos) e limiares calibrados (§6).
4. **[M] Curva de threads** T0 int4 e T1 int8 (4–24) — N=12 consistente (§1).
5. **[M] VRAM de processo:** T0 2,5–2,9 GB, T1 4,8–5,1 GB com prompts de autocomplete; **5,4 GB (T0) / 7,6 GB (T1) com contexto de 2.048 tokens**.
6. **[M] Curva VRAM × contexto (`reports/hwmap/vram_curve.md`, 120 execuções, n = 10 por ponto):** T0 1.858 / 2.337 / 2.465 / 3.246 / 3.510 / 5.560 MB e T1 4.064 / 4.526 / 4.654 / 5.452 / 5.708 / 7.759 MB (pico máximo) para contexto total 302 / 544 / 800 / 1.056 / 1.568 / 2.048 tokens; a curva é em degraus (buffers do DML), não linear; base do `context_cap_by_vram` (D1).

Continuam **[E]**/pendentes: todo hardware fora deste PC (Pascal 1060/1070/1080, GTX 1650/1660, RTX 20/30/40/50, RX 5000/6000/9000, Vega 56/64, Polaris RX 580, Intel Arc A/B, Iris Xe, Core Ultra, 780M/680M); as faixas de `device_id`; o fp32 em GPU de FP16 fraco; o corte de aborto no aquecimento do micro-benchmark; "núcleos físicos" ≠ threads em outras microarquiteturas; iGPUs de 6–16 CU (a de 2 CU não permite extrapolar com segurança).

## 9. Decisões (D1–D5)

Todas **DECIDIDAS pelo usuário em 2026-10-26**, consolidadas em `configs/gpu_compat.yaml` (schema 2: bloco `decisions`, `decision_ref` por regra/limiar, `migration`) e em `scripts/hwmap/detect_gpus.py` (testes: `scripts/hwmap/tests`). Evidência: [M] = medida neste PC (7800 XT, iGPU Raphael 2 CU, Ryzen 9 7900); o que vale fora dele é [E].

### D1 — VRAM por regime de uso + teto de contexto por VRAM · **DECIDIDO** (2026-10-26)

- **Decisão.** Os limiares de VRAM valem para o **regime de autocomplete típico** (prompts de ~270–700 tokens). Há uma **regra de teto de contexto por VRAM**: **T1 dml-fp16 só em GPU dedicada com ≥ 8 GB** (contexto cheio de 2.048 tokens); **entre ~6 e 8 GB** o T1 só com **teto de contexto menor** e só se o **micro-benchmark** aprovar; abaixo disso **T0**; **T0 dml-fp16 com ≥ 3,5 GB** (2,5–2,9 GB medidos no regime típico) com teto proporcional; **< 3,5 GB ou sem DX12 → CPU/não suportada**.
- **Justificativa.** O pedido original (T1 em 4 GB) não se sustenta: o T1 mede 4,8–5,1 GB no regime típico e **7,6–7,8 GB com contexto cheio**; o T0 mede 2,5–2,9 GB e **5,4–5,6 GB** com contexto cheio. Uma placa de 6 GB comporta o T1 só enquanto o contexto for curto, e uma de 4 GB não comporta o T0 em contexto cheio — daí o teto por VRAM em vez de um único corte.
- **Evidência [M] (curva, `reports/hwmap/vram_curve.md`; pico dedicado máximo de 10 processos por ponto, adaptador confirmado por contadores — `adapter_ok` 120/120; prompt L e `max_length` = L + 32, em 2.048 o prompt tem 2.016):**

| contexto total (tokens) | 302 | 544 | 800 | 1.056 | 1.568 | 2.048 |
|---|---|---|---|---|---|---|
| T0 dml-fp16 — pico (MB) | 1.858 | 2.337 | 2.465 | 3.246 | 3.510 | 5.560 |
| T1 dml-fp16 — pico (MB) | 4.064 | 4.526 | 4.654 | 5.452 | 5.708 | 7.759 |
| TTFT mediano T0 / T1 (ms) | 116 / 155 | 122 / 162 | 128 / 190 | 146 / 228 | 224 / 331 | 324 / 570 |

  Os dados da fase 2 (§E de `phase2_measurements.md`) coincidem: T1 4,8–5,1 GB e T0 2,5–2,9 GB com prompts reais, 7,6 / 5,4 GB em contexto cheio. O pico cresce em degraus porque o pacote usa graph capture com KV compartilhado de tamanho `max_length` e buffers de prefill proporcionais ao prompt. A execução do bench foi **ruidosa** (apps do usuário mantiveram 4–17% de CPU, a porta de ≤ 3% raramente foi cumprida); o pico de memória varia < 10 MB entre execuções, então o ruído não afeta este número.
- **Regra aplicada (YAML `context_cap_by_vram` = maior pico medido + 400 MB, arredondado a 50 MB; teto = tokens totais prompt + saída):**

| VRAM dedicada (DXGI) | variante | `context_cap_tokens` | condição |
|---|---|---|---|
| ≥ 8.150 MB (placa de 8 GB+; 8 GB reporta ~8.176) | T1 dml-fp16 | 2.048 | família confiável → direto; `unknown`/Intel/Blackwell → micro-benchmark |
| 6.100 – 8.149 MB (6 GB reporta ~6.144) | T1 dml-fp16 | **1.536** | **só se o micro-benchmark aprovar**, senão T0 |
| 5.950 – 6.099 MB | T0 dml-fp16 | 2.048 | — |
| 3.900 – 5.949 MB (4 GB reporta ~4.096) | T0 dml-fp16 | 1.536 | — |
| 3.650 – 3.899 MB | T0 dml-fp16 | 1.024 | — |
| 3.500 – 3.649 MB | T0 dml-fp16 | 768 | piso do T0 (D1); a curva só pede 2.850 MB para 768 |
| < 3.500 MB, sem DX12, software | CPU T0 int4 | 2.048 (CPU) | — |

  `context_cap_min_useful_tokens` = 768 (envelope típico de 270–700 tokens + saída): teto menor que isso = variante inútil → próxima/CPU. **[E]:** a curva foi medida só na 7800 XT; outras GPUs/drivers devem ficar na mesma ordem de grandeza e serão confirmados na homologação; a margem de 400 MB é política.

### D2 — Pascal: `unknown` + micro-benchmark, sem variante fp32 · **DECIDIDO** (2026-10-26)

- **Decisão.** Pascal (GTX 10, MX150/250, GT 1030) = `unknown`, decidido pelo micro-benchmark; **sem variante fp32** (o builder 0.15.2 não aceita FP32+DML e o decode cai a ~9 tok/s com o remendo); se o micro-benchmark reprovar → **CPU T0 int4** (direto, sem tentar T0 DML — `microbench_fail_goes_to_cpu`). Polaris segue a mesma linha.
- **Evidência.** [M] `phase2_measurements.md` §B: `-p fp32 -e dml` gera grafo sem `present.*` (rejeitado pelo ORT); com o remendo MHA o decode fica em 8,9 tok/s (fp32) e 9,4 (fp16) contra 173 tok/s do fp16 padrão; fp32 custa 2× em disco e +1,3 GB de VRAM; paridade igual (94% × 95%). [F S5, S6] FP16 a 1/64 nas GP10x. [E] Desempenho real do `dml-fp16` em Pascal: sem GPU local.

### D3 — iGPUs sempre na CPU · **DECIDIDO** (2026-10-26)

- **Decisão.** Nesta versão **toda iGPU vai para a CPU** (`igpu_policy: cpu_always_until_measured`). As iGPUs maiores (680M/780M/880M/890M/8060S, Arc dos Core Ultra, Iris Xe) **não são candidatas** até haver medição real nelas (roteiro de homologação). As regras ficam com `gpu_variant: cpu` e `candidate_after_measurement` apenas informativo; o detector não roda micro-benchmark de iGPU e lista as iGPUs em `igpus_pending_measurement`.
- **Evidência [M]** (`phase2_measurements.md` §A, §C): iGPU de 2 CU T0 1.338 ms / 23 tok/s e T1 3.713 ms / 7,3 tok/s contra 7800 XT 104 ms / 173 e 141 ms / 108 (**12,9× e 26,4× mais lenta no TTFT; 7,5× e 14,8× no decode**) e contra CPU 216–313 ms / 106–119 (T0) e 535–786 ms / 28–33 (T1): a CPU vence em todos os casos. A extrapolação do mapa para essa iGPU (0,7 s) era otimista (real 1,3 s). Maiores: **[E]**.

### D4 — T1 em CPU · **DECIDIDO** (2026-10-26)

- **Decisão.** O autocomplete em CPU é **sempre T0 int4**. **T1 int8 em CPU só como modo opcional e declaradamente lento**, escolha do usuário, **nunca padrão**, com ≥ 12 núcleos físicos e ≥ 16 GB de RAM; **não cumpre o gate de TTFT** (T1 500/650 ms).
- **Evidência [M]** (`phase2_measurements.md` §D): T1 int8 TTFT médio 2.224 / 1.452 / 1.133 / 786 / 685 / 608 / 535 ms com 4 / 6 / 8 / 12 / 16 / 20 / 24 threads (786 ms a 12, 535 ms a 24); T0 int4 884 / 613 / 440 / 313 / 256 / 219 / 216 ms. O N = 12 é consistente com o dado (threads de um Zen 4, não núcleos físicos isolados: ressalva [E]).

### D5 — Nenhuma variante nova agora · **DECIDIDO** (2026-10-26)

- **Decisão.** Aprovado "nenhuma variante nova agora": sem CUDA / TensorRT-RTX / OpenVINO / MIGraphX / GGUF, sem `dml-int4` do T1/T0, sem `dml-fp32`. **Reavaliar quando houver builder / ORT GenAI mais novo.** Registrado em `variants_not_exported` (YAML).
- **Evidência.** [M] `dml-int4` do T0: paridade greedy 72% (prefixo 81%) contra 95% do fp16, `dml-int4` do T1 −1,8 p.p.; `dml-fp32`: grafo inválido no builder (D2). Demais opções: [E]/[F] (§5.1 (iii)).

### Achado de produto → PR 5

O GenAI-WinML 0.15.2 **ignora** `device_id`, `device_filter`, `performance_preference` e `device_filtering_options`; qualquer pedido cai no adaptador padrão do DirectML (neste PC, a 7800 XT; num notebook híbrido, provavelmente a iGPU). A IDE deve garantir a GPU certa (preferência de GPU do Windows por app, DXGI para o adaptador efetivo + micro-benchmark, queda segura para CPU, upgrade do GenAI se suportar seleção): **`docs/prs/pr-5-selecao-de-adaptador-dml.md`** (não implementado na IDE). O detector sinaliza `adapter_selection.effective_adapter_check_required`.

### Continua [E] depois das decisões

Todo hardware fora deste PC (Pascal, GTX 16, RTX 20/30/40/50, RX 5000/6000/9000, Vega, Polaris, Intel Arc, todas as iGPUs além da de 2 CU); as faixas de `device_id`; a curva de VRAM em outras GPUs/drivers; o desempenho real do `dml-fp16` em Pascal/Polaris; "núcleos físicos" ≠ threads em outras microarquiteturas; o corte de aborto do micro-benchmark (`abort_warmup_ttft_factor` 4×, proposta); a margem de 400 MB do teto de contexto (política).
