#!/usr/bin/env python3
"""Gera reports/hwmap/phase2_measurements.md a partir de phase2_measurements.json (rode phase2_report.py antes)."""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "hwmap"))
from detect_gpus import enumerate_dxgi  # noqa: E402

LM = {a["luid"]: f"{a['vendor_id']:04X}:{a['device_id']:04X} {a['name']}" for a in enumerate_dxgi()}
D = json.loads((ROOT / "reports/hwmap/phase2_measurements.json").read_text(encoding="utf-8"))
L, M = D["latency"], D["micro"]


def ld(r):
    v = r["load_before_10s"]
    return f"{v[-1]:.1f}" if v else "-"


def row(label, k):
    v = L[k]
    runs = v["runs"]
    noisy = sum(1 for r in runs if r["noisy"])
    vr = f"{v['run_to_run_var_ttft_pct']}/{v['run_to_run_var_tps_pct']}" if v["run_to_run_var_ttft_pct"] is not None else "1 exec."
    vram = f"{round(v['vram_dedicated_mb_end'])} / {round(v['vram_shared_mb_end'])}" if v.get("vram_dedicated_mb_end") is not None else "-"
    return (f"| {label} | {len(runs)}x{v['n_prompts_per_run'][0]} | {v['ttft_ms_mean']:.0f} / {v['ttft_ms_p50']:.0f} / {v['ttft_ms_p95']:.0f} | {v['dec_forced_tps_mean']:.1f} | "
            f"{v['dec_tps_mean']:.1f} | {v['peak_wset_gb']:.2f} | {vram} | {vr} | {', '.join(ld(r) for r in runs)} | {'sim (' + str(noisy) + '/' + str(len(runs)) + ')' if noisy else 'nao'} |")


H = ("| config | exec x n | TTFT media / p50 / p95 (ms) | tok/s decode (64 forcados) | tok/s (saidas curtas) | RAM pico WS (GB) | VRAM dedic. / compart. fim (MB) | "
     "var. exec-a-exec TTFT/tok/s (%) | carga CPU 10 s antes (%) | ruidosa |\n|---|---|---|---|---|---|---|---|---|---|")
o = []
w = o.append
ns = D["noisy_summary"]
w(f"""# Fase 2 do mapa de GPU: medicoes (2026-10-26)

Gerado a partir de `reports/hwmap/phase2_raw/*.json` (bruto), `phase2_runs.jsonl` (carga observada por execucao) e `phase2_measurements.json` (agregado) por
`scripts/hwmap/phase2_report.py` + `phase2_md.py`. Maquina: Ryzen 9 7900 (12C/24T), DDR5, RX 7800 XT (1002:747E), iGPU Raphael 2 CU (1002:164E); GenAI-WinML 0.15.2;
pacotes v3. Prompts reais: `reports/export/work_v3/t{{0,1}}_prompts.json` (500 ids do val v2c, todos em `base-v2c/sample_ids.json`; mediana 261,5 tokens); n=100 por execucao,
3 aquecimentos descartados, 32 tokens de saida (paradas do contrato); tok/s sustentado = 64 tokens forcados (n=20 no T0, 10 no T1/CPU/variantes lentas).

## 0. Metodo, desvios e limites (leia primeiro)

1. **O GenAI-WinML 0.15.2 ignora `device_id`, `device_filter`, `performance_preference` e `device_filtering_options`** (testados: o processo sempre caiu na 7800 XT, conferido por
   contador). Para medir a iGPU *de proposito* sem mexer nas "Configuracoes graficas" do Windows, `scripts/hwmap/dxgi_select.py` remapeia, **so dentro do processo**, a ordem de
   enumeracao DXGI (`EnumAdapters1`/`EnumAdapterByGpuPreference` na vtable do `IDXGIFactory`) para por o adaptador `vendor:device` pedido na posicao 0. As travas de
   `scripts/export` (`onnx_run.py`, `measure_all.py`) **nao foram alteradas**; a iGPU so e escolhida com `scripts/hwmap/phase2_bench.py --adapter 1002:164E --allow-igpu`
   (flag explicita, gravada no JSON como `allow_igpu`; sem a flag o script recusa).
2. **Adaptador confirmado por contadores** (`\\GPU Process Memory(*)\\Dedicated/Shared Usage` e `\\GPU Engine(*)\\Utilization Percentage` por pid, `vram_sampler.ps1`): ver §7.
   Execucoes de latencia/paridade feitas antes do pedido de evidencia ficam como `pinned_by_arg_only` (a sonda pos-medida `probe_end` mostra a memoria do pid por LUID, nao o engine);
   para cada configuracao foi acrescentada uma execucao curta (n=20) com os contadores ligados durante a medida (`evid_*`); `adapter_ok` = pico de memoria > 100 MB no adaptador
   pedido e <= 50 MB / <= 1% de engine no outro. Nada foi apagado.
3. **Carga de fundo.** Regra: CPU <= 3% em 10 s antes de cada medida. O PC tem apps do usuario (ChatGPT, claude, Taskmgr, Explorer) que variaram de ~1% a ~25% durante as horas da bateria; nao foram encerrados.
   Cada execucao grava `idle_obs` (cargas de 10 s) e `noisy` (true quando nao chegou a <= 3% apos as tentativas). **Ruidosas: {len(ns['noisy'])} de {ns['rc0']} registros bem-sucedidos**
   (quase toda a microbateria dos GPUs, T1 iGPU r2/r3, T1 cpu-int8 em 4/6/8/20 threads e padrao, e 3 dos 4 registros de paridade `evid_`); o efeito aparece na CPU
   (T1 th16: TTFT 644-754 ms entre execucoes) e e pequeno nas DML (CV de 3-4% na microbateria da 7800 XT mesmo ruidosa). Uma passada silenciosa extra da curva T1 foi tentada e **abandonada** (7-16% por > 6 min).
4. **Repeticoes.** Cada configuracao rodou 2 vezes e uma 3a quando TTFT media ou tok/s diferiram > 10% (ocorreu em `t0 dml-fp32`, `t1 cpu-int8 th16` e `t1 dml-fp16 iGPU`; na fp32 e na th16 a execucao discrepante coincide com carga de fundo alta).
5. **Limites.** Uma unica maquina; sem Pascal/Intel/outras GPUs; pacotes DML padrao usam graph capture; micro-benchmark com texto sintetico de 270 tokens; TTFT e proxy (GenAI Python, sem o caminho `LocalAiModelService` da IDE); sem medida de potencia/termica.
""")
w("## A. iGPU (2 CU) x CPU x 7800 XT\n\n" + H)
for lab, k in [("T0 dml-fp16 / 7800 XT", "lat_t0-dml-fp16_7800xt"), ("T0 dml-fp16 / **iGPU**", "lat_t0-dml-fp16_igpu"), ("T0 cpu-int4 / 24 thr", "lat_t0-cpu-int4_th24"),
               ("T0 cpu-int4 / 12 thr", "lat_t0-cpu-int4_th12"), ("T0 cpu-int4 / padrao (0)", "lat_t0-cpu-int4_th0"), ("T1 dml-fp16 / 7800 XT", "lat_t1-dml-fp16_7800xt"),
               ("T1 dml-fp16 / **iGPU**", "lat_t1-dml-fp16_igpu"), ("T1 cpu-int8 / 24 thr", "lat_t1-cpu-int8_th24"), ("T1 cpu-int8 / 12 thr", "lat_t1-cpu-int8_th12"),
               ("T1 cpu-int8 / padrao (0)", "lat_t1-cpu-int8_th0")]:
    w(row(lab, k))
a, b, c, c12 = L["lat_t0-dml-fp16_igpu"], L["lat_t0-dml-fp16_7800xt"], L["lat_t0-cpu-int4_th24"], L["lat_t0-cpu-int4_th12"]
a1, b1, c1 = L["lat_t1-dml-fp16_igpu"], L["lat_t1-dml-fp16_7800xt"], L["lat_t1-cpu-int8_th24"]
w(f"""
**Resposta objetiva: a iGPU de 2 CU NAO vence a CPU em nenhum caso.** T0: iGPU {a['ttft_ms_mean']:.0f} ms de TTFT e {a['dec_forced_tps_mean']:.0f} tok/s contra CPU
{c['ttft_ms_mean']:.0f} ms ({a['ttft_ms_mean'] / c['ttft_ms_mean']:.1f}x mais rapida que a iGPU) / {c['dec_forced_tps_mean']:.0f} tok/s com 24 threads e {c12['ttft_ms_mean']:.0f} ms ({a['ttft_ms_mean'] / c12['ttft_ms_mean']:.1f}x mais rapida que a iGPU)
/ {c12['dec_forced_tps_mean']:.0f} tok/s com 12; T1: iGPU {a1['ttft_ms_mean']:.0f} ms / {a1['dec_forced_tps_mean']:.1f} tok/s contra CPU int8 {c1['ttft_ms_mean']:.0f} ms / {c1['dec_forced_tps_mean']:.0f} tok/s
(24 thr: {a1['ttft_ms_mean'] / c1['ttft_ms_mean']:.1f}x no TTFT, {c1['dec_forced_tps_mean'] / a1['dec_forced_tps_mean']:.1f}x no decode). Contra a 7800 XT a iGPU e {a['ttft_ms_mean'] / b['ttft_ms_mean']:.1f}x
mais lenta no TTFT e {b['dec_forced_tps_mean'] / a['dec_forced_tps_mean']:.1f}x no decode (T0), {a1['ttft_ms_mean'] / b1['ttft_ms_mean']:.1f}x e {b1['dec_forced_tps_mean'] / a1['dec_forced_tps_mean']:.1f}x (T1).
A extrapolacao do mapa (~0,7 s de prefill do T0 na iGPU) era otimista: o DML mediu 1,3 s. A iGPU usa so ~370-410 MB dedicados e 2,2-4,9 GB de memoria compartilhada (DDR5), e a RAM de pico do processo sobe a 3,5 GB (T0) / 5,7 GB (T1) contra ~1,1/1,6 GB na 7800 XT.
No T0 o decode da CPU (106-119 tok/s forcado) chega a 61-69% do da 7800 XT (173): o ganho da dGPU no T0 esta no prefill (104 x 216-313 ms).
""")
w("## B. Precisao no DirectML (7800 XT), T0\n")
w("""**Causa da falha do `dml-fp32` (rc=1 nas execucoes iniciais).** O builder 0.15.2 declara como validas apenas `FP16 DML` e `INT4 DML` (mensagem "Valid precision + execution provider
combinations"; FP32 so em CPU/CUDA). `-p fp32 -e dml` (`scripts/hwmap/build_exp.py`) cai no caminho de Attention empacotada e gera um grafo sem as saidas `present.N.key/value`
(292 nos, 24 `Attention`, nenhum `GroupQueryAttention`): o ORT recusa o carregamento ("This is an invalid model. Graph output (present.0.key) does not exist in the graph", `graph.cc:1494`).
**Nao e VRAM nem limite do DML: e um export invalido do builder** [M]; o pacote invalido ficou em `...-v3-ONNX-dml-fp32-builder-invalid-exp`. Contorno experimental (monkeypatch
em memoria do builder, `MHA_PATCH` do `build_exp.py`): forcar `MultiHeadAttention` + `repeat_kv` e desligar o graph capture (o GenAI recusa graph capture sem `past_present_share_buffer`).
Isso gera um fp32 valido (1.908 MB), mas o caminho de atencao muda junto com a precisao; por isso o controle `dml-fp16-mha` (fp16, mesmo caminho MHA sem graph capture, 964 MB) isola a precisao.
""")
w(H)
for lab, k in [("T0 dml-fp16 (GQA + graph capture; pacote v3)", "lat_t0-dml-fp16_7800xt"), ("T0 dml-int4 (exp)", "lat_t0-dml-int4_7800xt"),
               ("T0 dml-fp16-mha (controle exp)", "lat_t0-dml-fp16-mha_7800xt"), ("T0 dml-fp32 MHA (exp)", "lat_t0-dml-fp32_7800xt")]:
    w(row(lab, k))
P, E = D["parity"], D["evid"]
w("""
| pacote (T0, 7800 XT) | tamanho em disco (MB) | paridade greedy identica ao HF bf16 (n=100) | 1o token igual | prefixo aceito medio | VRAM dedicada pico (n=20, contadores) |
|---|---|---|---|---|---|""")
for lab, pk, sz, ev in [("dml-fp16", "parity_t0-dml-fp16_7800xt", 960, "evid_parity_t0-dml-fp16_7800xt"), ("dml-int4 (exp)", "parity_t0-dml-int4_7800xt", 548, "evid_parity_t0-dml-int4_7800xt"),
                        ("dml-fp16-mha (exp)", "parity_t0-dml-fp16-mha_7800xt", 964, "evid_parity_t0-dml-fp16-mha_7800xt"), ("dml-fp32 MHA (exp)", "parity_t0-dml-fp32_7800xt", 1908, "evid_parity_t0-dml-fp32_7800xt")]:
    p = P[pk]
    w(f"| {lab} | {sz} | {p['greedy_tokens_identical_pct']}% | {p['first_token_identical_pct']}% | {p['accepted_prefix_frac_mean_pct']}% | {E[ev]['vram_dedicated_mb_peak']:.0f} MB |")
f32, f16m, f16, i4 = L["lat_t0-dml-fp32_7800xt"], L["lat_t0-dml-fp16-mha_7800xt"], L["lat_t0-dml-fp16_7800xt"], L["lat_t0-dml-int4_7800xt"]
q = st.mean(r["ttft_mean"] for r in f32["runs"] if not r["noisy"])
w(f"""
Leitura: (i) **fp32 custa ~2x em disco (1.908 x 960 MB) e +1,3 GB de VRAM (3,75 x 2,47 GB)**; no prefill custa {q:.0f} ms (execs silenciosas) contra {f16m['ttft_ms_mean']:.0f} ms do fp16 no MESMO caminho MHA
(+{100 * (q / f16m['ttft_ms_mean'] - 1):.0f}%), e a paridade com o HF e igual (94% x 95%, dentro do ruido de n=100). (ii) O decode do caminho MHA sem graph capture cai a ~9 tok/s tanto em fp32
({f32['dec_forced_tps_mean']:.1f}) quanto em fp16 ({f16m['dec_forced_tps_mean']:.1f}) contra {f16['dec_forced_tps_mean']:.0f} tok/s do fp16 GQA+graph capture: **a lentidao e do caminho de atencao, nao da precisao** (precisao custa ~5% no decode).
Conclusao para Pascal: um `dml-fp32` util exigiria um export GQA/graph-capture em fp32 que o builder 0.15.2 nao produz; o MHA improvisado reprova o criterio de decode (>= 25-60 tok/s) ate numa 7800 XT.
O beneficio do fp32 numa GPU de FP16 fraco so se mede nela (sem Pascal local): **[E]**. (iii) **dml-int4**: {i4['ttft_ms_mean']:.0f} ms de TTFT (+{100 * (i4['ttft_ms_mean'] / f16['ttft_ms_mean'] - 1):.0f}% vs fp16),
{i4['dec_forced_tps_mean']:.0f} tok/s (+{100 * (i4['dec_forced_tps_mean'] / f16['dec_forced_tps_mean'] - 1):.0f}%), 548 MB (-43%), VRAM 2,2 x 2,5 GB, mas **a paridade greedy cai a 72%** (prefixo medio 81%): nao serve ao T0.
""")
w("## C. Micro-benchmark de primeira execucao (executor real de `scripts/hwmap/microbench.py`)\n")
w("10 execucoes por candidato, **um processo novo por execucao** (270 tokens de prompt sintetico + 32 de decodificacao; 1 aquecimento descartado + 3 repeticoes por processo, mediana). `total_s` = tempo interno (import do GenAI + carga + 4 geracoes); `parede` = medido pelo orquestrador (inclui iniciar o Python). Dispersao = desvio/media entre as 10 execucoes.\n")
w("| candidato | TTFT p50 ms (media; min-max) | disp. TTFT | tok/s (media; min-max) | disp. tok/s | carga do modelo (s) | total_s (media / max) | parede s (media / max) | ruidosas |\n|---|---|---|---|---|---|---|---|---|")
names = {"micro_t1-dml-fp16_7800xt": "T1 dml-fp16 / 7800 XT", "micro_t0-dml-fp16_7800xt": "T0 dml-fp16 / 7800 XT", "micro_t0-dml-fp16_igpu": "T0 dml-fp16 / iGPU",
         "micro_t1-dml-fp16_igpu": "T1 dml-fp16 / iGPU", "micro_t0-cpu-int4_cpu": "T0 cpu-int4 / CPU (thr. padrao)", "micro_t1-cpu-int8_cpu": "T1 cpu-int8 / CPU (thr. padrao)"}
for k, n in names.items():
    v = M[k]
    w(f"| {n} | {v['ttft_mean']:.0f} ({v['ttft_min']:.0f}-{v['ttft_max']:.0f}) | {v['ttft_cv_pct']:.1f}% | {v['tps_mean']:.1f} ({v['tps_min']:.1f}-{v['tps_max']:.1f}) | {v['tps_cv_pct']:.1f}% | "
      f"{v['load_s_mean']:.1f} | {v['total_s_mean']:.1f} / {v['total_s_max']:.1f} | {v['wall_s_mean']:.1f} / {v['wall_s_max']:.1f} | {sum(1 for x in v['noisy'] if x)}/10 |")
w("""
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
""")
w("## D. Curva de threads (CPU)\n\nIDE usa o padrao do ORT (0). `intra_op_num_threads` = N.\n")
w("| threads | T0 int4 TTFT media / p95 (ms) | T0 tok/s forcado | T1 int8 TTFT media / p95 (ms) | T1 tok/s forcado | T1: carga antes (%) | T1 ruidosa |\n|---|---|---|---|---|---|---|")
for th in (4, 6, 8, 12, 16, 20, 24, 0):
    t0, t1 = L[f"lat_t0-cpu-int4_th{th}"], L[f"lat_t1-cpu-int8_th{th}"]
    w(f"| {th if th else 'padrao'} | {t0['ttft_ms_mean']:.0f} / {t0['ttft_ms_p95']:.0f} | {t0['dec_forced_tps_mean']:.0f} | {t1['ttft_ms_mean']:.0f} / {t1['ttft_ms_p95']:.0f} | "
      f"{t1['dec_forced_tps_mean']:.0f} | {', '.join(ld(r) for r in t1['runs'])} | {sum(1 for r in t1['runs'] if r['noisy'])}/{len(t1['runs'])} |")
w("""
Leitura: o TTFT escala quase com 1/N ate 12 threads (T1: 4 thr 2.224 ms, 6 thr 1.452, 8 thr 1.133, 12 thr 786, 24 thr 535; T0: 884, 613, 440, 313, 216) e o SMT ainda ajuda (24 thr = 0,68x o tempo
de 12 no T1, 0,69x no T0). O decode e quase plano (banda): ~100-120 tok/s no T0 e 25-33 no T1. O padrao do ORT (0) equivale a 12 threads (310 x 313 ms no T0; 791 x 786 no T1).
**Validacao do N=12 de `cpu_t1_int8_min_physical_cores`:** com 12 threads o T1 int8 fica em 786 ms de media (p95 1,28 s) e com 8 em 1,13 s (p95 1,8 s); nenhum ponto chega ao gate de autocomplete
(500/650 ms) nem com 24 threads (535/863 ms). O N=12 e **consistente com o dado [M]** (12 = 0,79 s; 8 ja e 1,4x pior) mas e uma escolha de politica sobre "T1 int8 como opcao de chat" (D4):
**valor nao alterado**. Ressalvas: medido em threads de um Zen 4 12C/24T (nao em nucleos fisicos isolados); as execucoes de 4/6/8/20 threads e o padrao do T1 foram ruidosas (carga 6-22%),
mas a curva e monotonica e as execucoes silenciosas (12 e 24 thr) batem com o padrao 1/N.
""")
w("## E. VRAM de processo (7800 XT)\n\nPico = maior valor do amostrador de contadores (1 s) durante a geracao; `fim` = sonda apos a bateria completa (100 prompts + 20 decodificacoes forcadas). Valores do relatorio v3 (200 prompts): T1 5.133 MB, T0 2.933 MB.\n")
w("| carga | dedicada pico (MB) | compartilhada pico (MB) | fim da bateria (MB, dedic./comp.) | TTFT / tok/s |\n|---|---|---|---|---|")
V, C = D["verify"], D["ctx"]
for lab, k, fk in [("T0 dml-fp16, prompts reais (~260 tok)", "verify_t0-dml-fp16_7800xt", "lat_t0-dml-fp16_7800xt"), ("T1 dml-fp16, prompts reais", "verify_t1-dml-fp16_7800xt", "lat_t1-dml-fp16_7800xt")]:
    v, f = V[k], L[fk]
    w(f"| {lab} | {v['vram_dedicated_mb_peak']:.0f} | {v['vram_shared_mb_peak']:.0f} | {f['vram_dedicated_mb_end']:.0f} / {f['vram_shared_mb_end']:.0f} | {f['ttft_ms_mean']:.0f} ms / {f['dec_forced_tps_mean']:.0f} tok/s |")
for lab, k in [("T0 dml-fp16, **contexto cheio** (prefill 1.900 + decode ate 2.048)", "ctx_t0-dml-fp16_7800xt"), ("T1 dml-fp16, **contexto cheio**", "ctx_t1-dml-fp16_7800xt")]:
    v = C[k]
    r = v["ctx_runs"]
    w(f"| {lab} | {v['vram_dedicated_mb_peak']:.0f} | {v['vram_shared_mb_peak']:.0f} | - | {st.mean(x['ttft_ms'] for x in r):.0f} ms / {st.mean(x['tok_s'] for x in r):.0f} tok/s (3 execs) |")
w("""
Leitura para D1: com prompts de autocomplete (~260 tokens) o pico medido e **2,5-2,9 GB (T0)** e **4,8-5,1 GB (T1)**; os limiares T0 >= 3.500 MB e T1 >= 5.900 MB cobrem esse regime com 0,6-0,8 GB de folga
(placa de "6 GB" reporta ~6.144 MB). **Com o contexto cheio de 2.048 tokens o pico sobe para 5,4 GB (T0) e 7,6 GB (T1)** (+2,5-2,9 GB de buffers intermediarios do prefill de ~1.900 tokens): uma placa de 6 GB
comporta o T1 so enquanto o prompt for curto, e uma de 4 GB nao comporta o T0 em contexto cheio. Nao medido: a curva pico x comprimento do prompt entre 270 e 1.900 tokens [pendente]. Os limiares D1
**nao foram alterados** (decisao do usuario); estes dados os sustentam para o regime de autocomplete e os limitam para contexto longo.
""")
w("## 7. Execucao x adaptador pedido x adaptador observado (todas as execucoes DML, exceto a microbateria)\n")
w("`contadores` = amostrador durante a execucao (pico; `adapter_ok` medido); `pinned_by_arg_only + probe_end` = adaptador fixado so por argumento, observado apos a medida pela memoria do pid; a execucao `evid_*` do mesmo pacote/adaptador fornece o contador. Em `parity_*` sem sonda o observado e `-` (ver `evid_parity_*`).\n")
w("| execucao | pedido | observado | memoria dedic.+compart. por GPU (MB) | engine % por GPU | evidencia | adapter_ok |\n|---|---|---|---|---|---|---|")


def nm(x):
    return LM.get(x, x)


for r in D["adapter_table"]:
    ob = r["observed"] or "-"
    ob = nm(ob) if ob.startswith("0x") else ob
    mem = ", ".join(f"{nm(l).split(' ')[0]}={v:.0f}" for l, v in (r["mem_mb_by_luid"] or {}).items())
    ut = ", ".join(f"{nm(l).split(' ')[0]}={v}" for l, v in (r["util_pct_by_luid"] or {}).items())
    ev = r["evidence"]
    ev = "pinned_by_arg_only + probe_end" if ev.startswith("pinned") else ("contadores" if ev in ("counters", "counters_during_run") else ev)
    ok = r["adapter_ok"]
    if ok is None:
        ok = "- (sem sonda)" if r["run"].startswith("parity_") else "true (verify: on_requested_adapter)"
    w(f"| {r['run']} | {r['requested']} ({r['requested_name'] or ''}) | {ob} | {mem or '-'} | {ut or '-'} | {ev} | {ok} |")
w("""
Microbateria (60 execucoes): `microbench.py` nao coleta contadores (processo isolado de 4-33 s); cada candidato usou o mesmo par pacote+`--adapter` das execucoes `evid_lat_*` correspondentes (confirmadas acima)
e os tempos sao coerentes com o adaptador pedido (TTFT da iGPU 8-20x o da 7800 XT; o da 7800 XT igual ao das execucoes de latencia).

## 8. Execucoes falhas ou descartadas
""")
w(f"- rc != 0 em `phase2_runs.jsonl`: {', '.join(ns['rc_nonzero'])}. As quatro de `dml-fp32` sao o export invalido do builder (§B), **refeitas** com o pacote MHA; a `evid_lat_t1-dml-fp16_igpu` estourou o `timeout` de 120 s do amostrador (bug do harness, corrigido) e foi repetida com sucesso.")
w("- A passada silenciosa da curva T1 foi interrompida (a maquina nunca ficou < 3%).")
Path(ROOT / "reports/hwmap/phase2_measurements.md").write_text("\n".join(o), encoding="utf-8")
print(len("\n".join(o)))
