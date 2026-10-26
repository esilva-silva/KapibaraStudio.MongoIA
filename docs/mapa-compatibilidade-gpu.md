# Mapa de compatibilidade de GPU — qual variante de autocomplete para cada placa

Data: 2026-10-26 (fase 2 de medicao incluida: `reports/hwmap/phase2_measurements.md`) · Escopo: Windows 11/10, T0 Nano (0,5B) e T1 Small (1,5B), pacotes ONNX v3 (GenAI 0.15.2 = IDE; DirectML fp16/int4, CPU int4/int8).
Tabela de decisão legível por máquina: `configs/gpu_compat.yaml` · detector: `scripts/hwmap/detect_gpus.py` · roteiro de homologação: `docs/roteiro-homologacao-hardware.md`.
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

**Conclusão.** O DirectML fp16 que já exportamos é o caminho certo para **GPU dedicada com VRAM suficiente e FP16 nativo**; a CPU int4 é o piso universal; a iGPU quase nunca compensa. Os três pontos de maior incerteza são: **Pascal (FP16 a 1/64) [E]**, **Intel Arc/iGPUs (DML nunca medido por nós) [E]** e **a VRAM mínima do T1 (5,1 GB medidos, não 4 GB)** [M].

- **[M] Referência:** RX 7800 XT 16 GB (624 GB/s): T1 DML fp16 TTFT 148/170 ms (média/p95), 104 tok/s; T0 DML fp16 109/124 ms. CPU (Ryzen 9 7900, 12C/24T): T0 int4 285/480 ms (229/382 com 24 threads), T1 int8 697/1.192 ms. iGPU de 2 CU RDNA2 ≈ 50× mais lenta que a 7800 XT em llama.cpp Vulkan (pp 27 × 1.395 t/s; tg 2,5 × 90 t/s) — `reports/export/onnx_export_t0_t1.md` §4–§5 e medição do usuário. **[M fase 2, GenAI-WinML 0.15.2, 100 prompts reais]:** 7800 XT T0 104 ms / 173 tok/s, T1 141 ms / 108 tok/s; CPU T0 int4 216 ms (24 thr) / 313 ms (12 thr) / 106–116 tok/s, T1 int8 535 ms (24 thr) / 786 ms (12 thr) / 28–33 tok/s; **iGPU 2 CU: T0 1.338 ms / 23 tok/s, T1 3.713 ms / 7,3 tok/s** (a CPU vence a iGPU em todos os casos).
- **[M] VRAM de processo medida (dedicada):** T1 dml-fp16 **5.133 MB**, T0 dml-fp16 **2.933 MB** (bem acima do tamanho do pacote: buffers/KV do DML) → o limiar "≥ 4 GB → T1" do pedido original **não se sustenta**: placas de 4 GB não comportam T1. O YAML usa T1 ≥ 5.900 MB (placas de 6 GB, que o DXGI reporta ~6.144 MB) e T0 ≥ 3.500 MB. **Decisão D1 do usuário.**
- **Regra geral:** dGPU elegível e `supported ≠ false` → T1 dml-fp16 (VRAM ≥ 5.900 MB) ou T0 dml-fp16 (≥ 3.500 MB); família `unknown`/Pascal/Intel/Blackwell → mesma variante **só após o micro-benchmark**; iGPU → CPU, salvo as de ≥ ~4 TFLOPS (780M/680M/890M/Arc de Core Ultra), que são *candidatas* a T0 DML sujeitas ao micro-benchmark; CPU sempre T0 int4 no autocomplete; T1 int8 na CPU só como opção (chat/qualidade) com **≥ 12 núcleos físicos e ≥ 16 GB** (N=12 **[M fase 2]**: T1 int8 em 4/6/8/12/16/20/24 threads = 2.224/1.452/1.133/786/685/608/535 ms de TTFT médio; o 7900 reprova o gate 500/650 até com 24 threads; T0 int4: 884/613/440/313/256/219/216 ms; abaixo de 12 threads o T1 passa de ~1,1 s [M]; threads ≠ núcleos físicos isolados, ver phase2 §D).

```
início ─ enumerar adaptadores DXGI (nunca pelo índice: vendor/device/VRAM)
  │
  ├─ descartar: software/WARP (0x1414), VM, supported=false
  │
  ├─ há dGPU (não integrada) elegível?
  │     ├─ VRAM ≥ 5.900 MB ─────────► T1 dml-fp16   ┐  se família confiança alta e medida (RDNA3, Ampere/Ada...)  → aceita direto [E]
  │     ├─ 3.500 ≤ VRAM < 5.900 ────► T0 dml-fp16   ┘  senão (Pascal, Polaris, Intel, Blackwell, unknown) → micro-benchmark antes de aceitar
  │     └─ VRAM < 3.500 MB ─────────► CPU
  │
  ├─ só iGPU?  ≥ ~4 TFLOPS (780M, 680M, 890M, Arc Core Ultra) ─► candidata T0 dml-fp16 + micro-benchmark (vence a CPU por ≥ 30%? senão CPU)
  │            demais (Vega 3–11, 610M, UHD, Iris Xe*, Gen ≤ 9) ─► CPU   (*Iris Xe: candidata opcional ao micro-benchmark)
  │
  └─ CPU: T0 cpu-int4 (autocomplete). T1 cpu-int8 só se ≥ 12 núcleos físicos e ≥ 16 GB (opcional, chat)
  fallback de qualquer falha de DML/GPU ─► T0 cpu-int4
```

## 2. Tabela por família

Colunas: **DX12 / driver** (mínimo), **FP16**, **INT8/matriz**, **VRAM / banda** (faixa típica), **classe T0 / T1 [E]** (A = como a 7800 XT em ordem de grandeza; B = ~2–3× pior; C = pior que a CPU), **variante**, **riscos**. Todas as colunas de classe, VRAM e banda são [E] (conhecimento de especificação de produto, não medido), exceto onde marcado.
Mínimos de plataforma comuns [F]: Windows 10 1903+ para o DirectML (S1); DX12 (S1, S2); GenAI WinML é *preview* 0.x (S4). Driver: usar sempre o WHQL mais recente da família — não fixamos número de driver por falta de fonte [E].

### 2.1 NVIDIA (PCI 0x10DE)

| Família | DX12 / driver | FP16 | INT8 / matriz | VRAM · banda | Classe T0 / T1 | Variante (regra) | Riscos |
|---|---|---|---|---|---|---|---|
| GTX 9xx e anteriores (Maxwell/Kepler/Fermi) | DX12 FL 11_0–12_1; DML roda em Kepler+ [F S2]; Maxwell em driver *security-only* [F S8] | sem ganho | sem dp4a | 2–6 GB · 80–225 GB/s | C / C | **NÃO SUPORTADA** → CPU T0 int4 | política: abaixo da fronteira GTX 1000 [E] |
| **GTX 10 (Pascal GP10x)**, MX150/250, GT 1030 | FL 12_1; driver R580 *security-only* [F S8] | **1/64 do FP32 nas GP10x de consumo [F S5, S6]** | `dp4a` existe [F S5]; uso pelo DML [E] desconhecido | 2–11 GB · 48–484 GB/s (1060 6 GB: 192) | T0: B/C ?; T1: C ? **[E] — sem medida** | `unknown` → T1 (≥ 5.900 MB) ou T0 (≥ 3.500 MB) **só com micro-benchmark**; fallback CPU. GT 1030/MX/1050 2 GB → CPU | **maior risco do mapa**: se o DML executar FP16 nativo no Pascal, o prefill de ~270 tokens do T1 (~0,8 TFLOP) levaria > 10 s (1060: FP32 4,4 TFLOPS → FP16 ~70 GFLOPS) [E]; se converter para FP32 em shader, é aceitável [E]. Testar `dml-fp32`/`dml-int4` (§5.2) |
| GTX 16 (Turing TU116/TU117) | FL 12_1 | 2× FP32 [E] (sem Tensor Cores) | dp4a [E] | 4–6 GB · 112–336 GB/s | T0 B; T1 B | 4 GB (1650) → **T0 dml-fp16**; 6 GB (1660/Super/Ti) → T1 no limite (5.900 MB) | 6 GB reportado ~6.100 MB: margem pequena; micro-benchmark confirma |
| RTX 20 (Turing) | FL 12_1 | 2× [E] | dp4a + Tensor [E] | 6–24 GB · 336–672 GB/s | A/B | T1 dml-fp16 (≥ 5.900 MB) | RTX 2060 6 GB no limite de VRAM |
| RTX 30 (Ampere) | FL 12_2 | sim | Tensor gen 3 | 6–24 GB · 168–936 GB/s | A (3060+) | T1 dml-fp16 | 3050 6 GB (96–168 GB/s): banda baixa — T1 decode pode cair <~40 tok/s [E] |
| RTX 40 (Ada) | FL 12_2 | sim | Tensor gen 4 | 8–24 GB · 272–1.008 GB/s | A | T1 dml-fp16 | — |
| RTX 50 (Blackwell) | FL 12_2 | sim | Tensor gen 5 | 8–32 GB · 448–1.792 GB/s | A [E] | T1 dml-fp16 **com micro-benchmark** | hardware/driver recentes, sem teste nosso; DML "legacy" pode ficar atrás [E] |
| MX/laptops Max-Q | variam | variam | variam | 2–8 GB | variam | por device_id; "unknown" → micro-benchmark | híbrido NVIDIA+iGPU: o adaptador padrão do Windows costuma ser a iGPU [F S2: "device 0 nem sempre é a mais rápida"] → sempre escolher por vendor/device/VRAM |

### 2.2 AMD (PCI 0x1002)

| Família | DX12 / driver | FP16 | INT8 / matriz | VRAM · banda | Classe T0 / T1 | Variante | Riscos |
|---|---|---|---|---|---|---|---|
| Pré-Polaris (GCN1–3, TeraScale, Fury) | FL 11_1–12_0; DML aceita GCN 1ª ger.+ [F S2] | não | não | 2–4 GB | C / C | **NÃO SUPORTADA** → CPU | drivers abandonados [E] |
| **Polaris (RX 400/500)** — *abaixo de Vega* | FL 12_0; driver *legacy* [F S9] | sem Rapid Packed Math [E] | sem dp4a [E] | 4–8 GB · 112–256 GB/s | ? / ? **[E]** | `unknown`: T1 (8 GB) ou T0 com micro-benchmark; fallback CPU | RX 580 é popular: homologar |
| **Vega** (RX Vega 56/64, Radeon VII) | FL 12_1; driver *legacy* [F S9] | 2× (RPM) [E] | limitado [E] | 8–16 GB HBM2 · 410–1.024 GB/s | B / B (banda alta) [E] | **T1 dml-fp16** (≥ 5.900 MB) | driver legacy: bugs DX12 não corrigidos [E]; Vega 56 8 GB é candidata forte |
| iGPU Vega 3–11 (Raven/Picasso/Renoir/Cezanne/Barcelo) | FL 12_1 | 2× [E] | limitado | compartilhada (DDR4 38–51 GB/s) | C / C | **CPU** | ≤ ~2,2 TFLOPS e banda de DDR4: pior que a CPU [E] |
| RX 5000 (RDNA1) | FL 12_1 | 2× [F S10] | dp4a em Navi 10 incerto [E] | 4–8 GB · 112–448 GB/s | B / B [E] | 4 GB (5500 XT) → T0; 6–8 GB → T1 | sem medida; sem WMMA |
| RX 6000 (RDNA2) | FL 12_1/12_2 | 2× [E] | dot4; sem WMMA | 4–16 GB · 144–512 GB/s | B–A [E] | 6400/6500 XT (4 GB) → T0; 6600+ (8 GB) → T1 | prioridade nº 1 de homologação |
| **RX 7000 (RDNA3)** | FL 12_2 | 2× | WMMA | 8–24 GB · 288–960 GB/s | **A [M] (7800 XT)** | **T1 dml-fp16 [M]** | 7600 (288 GB/s) deve ficar ~2× atrás da 7800 XT no decode [E] |
| RX 9000 (RDNA4) | FL 12_2 | 2× | WMMA gen 2 | 8–32 GB · 320–640 GB/s | A [E] | T1 dml-fp16 com micro-benchmark | sem teste nosso |
| iGPU 610M / "Radeon Graphics" 2 CU (Raphael) | FL 12_x | sim | dot4 | compartilhada | **C [M]** (piso) | **CPU** | medida neste PC: ~50× mais lenta que a 7800 XT |
| iGPU 660M/680M (RDNA2 6–12 CU) | FL 12_x | sim | dot4 | LPDDR5/DDR5 75–100 GB/s | B/C [E] | candidata T0 dml-fp16 + micro-benchmark | empata com CPU de desktop; ganha de CPU de notebook 4–8 núcleos [E] |
| iGPU 780M/760M/880M/890M (RDNA3/3.5 8–16 CU) | FL 12_x | sim | WMMA | 90–120 GB/s | B [E] | candidata T0 dml-fp16 + micro-benchmark; **T1 não** | banda compartilhada limita decode do T1 a ~25–35 tok/s [E] |
| iGPU 8060S (Strix Halo, 40 CU) | FL 12_x | sim | WMMA | até ~256 GB/s; VRAM = *carve-out* configurável | B/A [E] | T1 dml-fp16 com micro-benchmark | `DedicatedVideoMemory` não é confiável (0,5–96 GB) → usar RAM total |

### 2.3 Intel (PCI 0x8086)

| Família | DX12 / driver | FP16 | INT8 / matriz | VRAM · banda | Classe T0 / T1 | Variante | Riscos |
|---|---|---|---|---|---|---|---|
| Pré-Gen9 (HD 2000–5000, Gen6–8) | DML "Haswell+" [F S2]; sem FP16 útil | não | não | compartilhada | C / C | **NÃO SUPORTADA** → CPU | — |
| UHD/HD Gen9–Gen11, UHD Xe-LP 24–32 EU (UHD 630/730/750/770) | FL 12_0/12_1 | 2× [E] | dp4a Gen11+ [E] | DDR4 38–51 GB/s; 0,4–0,8 TFLOPS | C / C [E] | **CPU** | menos compute que o piso medido (2 CU ≈ 1,1 TFLOPS) [E] |
| Iris Xe (Gen12 80–96 EU) | FL 12_1; iGPU Alder Lake+ citada no Windows ML [F S11] | 2× | dp4a | LPDDR4x/DDR4 51–68 GB/s; ~2 TFLOPS | C/B [E] | **CPU**; candidata opcional ao micro-benchmark (T0) | pode empatar com CPU de notebook fraco |
| Arc A-series (Alchemist) | FL 12_2; Arc A310+ citada [F S11] | sim | dp4a + XMX | 6–16 GB · 186–560 GB/s | A/B [E] | A750/A770 (8–16 GB) → T1; A380 (6 GB) → T1 elegível pela VRAM mas banda baixa (186 GB/s), o micro-benchmark decide; A310 (4 GB) → T0 — sempre **com micro-benchmark** | histórico de drivers DX12 irregular [E]; DML nunca medido por nós |
| Arc B-series (Battlemage) | FL 12_2; B580+ citada [F S11] | sim | XMX | 10–12 GB · 456 GB/s | A [E] | T1 com micro-benchmark | hardware recente, sem teste |
| iGPU Arc de Core Ultra (Meteor/Arrow-H/Lunar Lake) | iGPU 12ª ger.+ [F S11] | sim | XMX (Lunar) / dp4a | LPDDR5X 90–135 GB/s; ~4–4,6 TFLOPS | B [E] | candidata T0 dml-fp16 + micro-benchmark | Arrow Lake desktop (4 Xe cores, ~2 TFLOPS) casa pelo mesmo nome e é mais fraca [E] |

## 3. iGPU × CPU

**Dado de piso [M]:** 2 CU RDNA2 (≈ 1,1 TFLOPS FP32 de pico [E]): pp 27 t/s no 7B Q5_K_M ⇒ ~0,38 TFLOPS efetivos (2·7e9·27) [E, derivado]. O mesmo cálculo na 7800 XT dá ~19 TFLOPS efetivos (53% de ~37 de pico).
**Medido na fase 2 [M] (ONNX/DML, prompts reais de ~260 tokens):** T0 dml-fp16 na iGPU TTFT 1.338 ms (p95 2.223), 23 tok/s forçado, vs CPU 216–313 ms e 106–119 tok/s e 7800 XT 104 ms / 173 tok/s; T1: iGPU 3.713 ms / 7,3 tok/s vs CPU int8 535–786 ms / 28–33 tok/s e 7800 XT 141 ms / 108 tok/s. A estimativa de "~0,7 s de prefill" abaixo era otimista (real 1,3 s). A iGPU usa ~370–410 MB dedicados + 2,2–4,9 GB compartilhados; RAM de pico 3,5 GB (T0) / 5,7 GB (T1).
**Extrapolação para o T0 [E; para 2 CU substituída pela medida acima]** (0,5B ⇒ ~2·0,5e9·270 = 0,27 TFLOP de prefill [E]): iGPU de 2 CU ≈ 0,7 s de prefill (> CPU 285 ms [M]) → CPU vence. Uma iGPU de 12 CU RDNA3 (~8 TFLOPS de pico) levaria ~0,1 s no regime do llama.cpp; mas o DML do ORT atingiu só ~2,5 TFLOPS efetivos no T0 e ~5,5 no T1 da 7800 XT [M, derivado: 109 ms e 148 ms para 0,27/0,81 TFLOP, com sobrecarga fixa de dezenas de ms] — logo uma 780M tende a **empatar** (~250–350 ms) com a CPU de desktop de 12 núcleos e a ganhar de CPU de notebook (4–8 núcleos: 400–700 ms [E, da escala 4thr 731/8thr 406 ms]). **Conclusão [E]: só vale a pena iGPU com ≥ ~4 TFLOPS e, mesmo assim, apenas se o micro-benchmark confirmar ≥ 30% de ganho sobre a CPU do mesmo PC.**
**Banda [E, especificação]:** DDR4-3200 dual = 51 GB/s; DDR5-5600 dual = 90; LPDDR5X-7500 128-bit = 120; GDDR6 da 7800 XT = 624 [M]. O decode do T1 fp16 (3 GB lidos/token) fica limitado a ~17 tok/s em 51 GB/s, ~30 em 90, ~40 em 120 [E, 3 GB/token] — por isso **T1 em iGPU não é recomendado**. O decode do T0 (1 GB/token) = 50/90/120 tok/s [E] é suficiente.
**CPU também disputa a mesma RAM**: iGPU e CPU dividem banda; o ganho da iGPU é de compute (prefill), não de banda.
Em PC com dGPU, a iGPU é **sempre ignorada** (teste `test_this_pc_ignores_igpu_and_warp_regardless_of_order`), como o contador de GPU confirmou [M] (0 MB e 0% na iGPU, `onnx_export_t0_t1.md` §2).

## 4. Mapa de NÃO SUPORTADAS (fronteira)

| Hardware | Motivo | Comportamento |
|---|---|---|
| NVIDIA Fermi/Kepler/Maxwell (GTX 400–900, GT 7xx, GTX 750) | abaixo de GTX 1000 por política [E]; DML roda em Kepler+ [F S2] mas sem FP16 acelerado; Maxwell *security-only* [F S8] | CPU T0 int4; override avançado manual |
| NVIDIA com < 3,5 GB (GTX 1050 2 GB, GT 1030, MX150/250 2 GB, 1060 3 GB) | VRAM < T0 (2.933 MB medidos no processo) [M] | CPU |
| AMD pré-Polaris (GCN1–3, TeraScale, R9 200/300, Fury) | sem FP16, drivers abandonados [E]; DML aceita GCN1+ [F S2] | CPU |
| AMD Polaris (RX 4xx/5xx) | **fronteira "unknown"**, não "não suportada": GCN4 sem FP16 empacotado, driver *legacy* [F S9] | micro-benchmark; senão CPU |
| Intel pré-Gen9 / sem DX12 | sem DX12 pleno/FP16 [E] | CPU |
| Intel UHD/HD Gen9–11, UHD 24–32 EU | suportado mas "não recomendado": mais lento que a CPU [E] | CPU |
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
- Decisão provisória: Pascal = `unknown`/micro-benchmark; **não** criar variante nova até o teste.

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
| **Adicionar (condicional, fase 2)** | T0 `dml-fp32` **somente se** o teste em Pascal mostrar `dml-fp16` > 2× pior que a expectativa (ex.: TTFT T0 > 1 s) — ~1,9 GB, ≥ 4 GB VRAM |
| **NÃO fazer** | CUDA / TensorRT-RTX / OpenVINO / MIGraphX / GGUF como variantes agora; `dml-int4` para o T1 (−1,8 p.p., [M]); exports por vendor; pacote por família |
| **Spikes (sem pacote)** | OpenVINO numa Arc/Core Ultra; WebGPU EP; registro de EPs do Windows ML |

## 6. Micro-benchmark de primeira execução e fallback seguro (desenho)

**Quando:** no primeiro uso de uma variante GPU com `needs_microbench: true` (qualquer família `unknown`, Pascal, Polaris, Blackwell, Intel, iGPU candidata) e quando o cache invalida. Para famílias medidas/de alta confiança (RDNA3, Ampere/Ada) roda em segundo plano, sem bloquear.
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

Continuam **[E]**/pendentes: todo hardware fora deste PC (Pascal 1060/1070/1080, GTX 1650/1660, RTX 20/30/40/50, RX 5000/6000/9000, Vega 56/64, Polaris RX 580, Intel Arc A/B, Iris Xe, Core Ultra, 780M/680M); as faixas de `device_id`; a curva VRAM × comprimento do prompt (270–1.900 tokens); o fp32 em GPU de FP16 fraco; o corte de aborto no aquecimento do micro-benchmark; "núcleos físicos" ≠ threads em outras microarquiteturas; iGPUs de 6–16 CU (a de 2 CU não permite extrapolar com segurança).

## 9. Decisões do usuário

- **D1** — VRAM mínima do T1: 4 GB (pedido) × **5.900 MB (proposta [M])**. Com 4 GB o T1 estoura (5.133 MB medidos no processo).
- **D2** — Pascal: aceitar `unknown` + micro-benchmark e testar `dml-fp32` do T0 em fase 2 (recomendado) ou declarar Pascal "não suportada".
- **D3** — iGPUs candidatas (780M/680M/890M/Arc Core Ultra) com micro-benchmark, ou simplesmente CPU (mais simples, perde ganho em notebook).
- **D4** — CPU: T1 int8 como opção de chat com ≥ 12 núcleos/16 GB (proposto) ou nunca no autocomplete.
- **D5** — Aprovar o plano "nenhuma nova variante agora; DML fp32 T0 condicional".

**Dados da fase 2 que apoiam as decisões (sem decidi-las):** D1 — T1 4,8–5,1 GB de pico com autocomplete, 7,6 GB com contexto cheio; T0 2,5–2,9 / 5,4 GB. D2 — `dml-fp32` do T0 não é gerável pelo builder 0.15.2 (grafo inválido); via MHA é viável mas ~9 tok/s e 2× o tamanho; `dml-int4` perde paridade (72%). D3 — a iGPU de 2 CU perde da CPU em todos os casos; 6–16 CU não medidas. D4 — T1 int8 em CPU: 786 ms (12 thr) / 535 ms (24 thr), nenhum atinge o gate 500/650 ms. D5 — nenhuma variante nova foi necessária nas medições.
