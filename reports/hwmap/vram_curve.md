# Curva VRAM de processo x comprimento do prompt (RX 7800 XT, T0/T1 dml-fp16 v3)

Gerado por `scripts/hwmap/vram_curve.py` + `vram_curve_report.py` a partir de `reports/hwmap/vram_curve_raw/*.json` (120 execucoes) em 2026-10-26. Cada execucao = **um processo novo**: carrega o pacote v3 `dml-fp16` com o adaptador fixado em 1002:747E (`dxgi_select`), liga o amostrador `vram_sampler.ps1` (contadores `GPU Process Memory`/`GPU Engine` por pid e por LUID, 1 amostra/s por ~8 s) e gera repetidamente (prefill de L tokens + 32 forcados, `max_length = L + 32`; em L = 2.048 o prompt tem 2.016 tokens para o contexto TOTAL ser 2.048; tokens dos prompts reais do val v2c concatenados). Pico = maior amostra de **memoria dedicada do processo** no LUID da 7800 XT. `adapter_ok` = pico > 100 MB no LUID pedido e <= 50 MB / <= 1% de engine na iGPU.

## Resultado (n = 10 por ponto)

| variante | prompt (tok) | contexto total (tok) | n | pico dedicado min / mediana / **max** (MB) | desvio (MB) | compart. max (MB) | TTFT mediano (ms) | adapter_ok | execucoes ruidosas |
|---|---|---|---|---|---|---|---|---|---|
| T0 dml-fp16 | 270 | 302 | 10 | 1858 / 1858 / **1858** | 0.12 | 32 | 115.5 | 10/10 | 10/10 |
| T0 dml-fp16 | 512 | 544 | 10 | 2336 / 2336 / **2337** | 0.36 | 32 | 121.5 | 10/10 | 9/10 |
| T0 dml-fp16 | 768 | 800 | 10 | 2464 / 2465 / **2465** | 0.46 | 31 | 127.7 | 10/10 | 5/10 |
| T0 dml-fp16 | 1024 | 1056 | 10 | 3246 / 3246 / **3246** | 0.25 | 30 | 145.9 | 10/10 | 3/10 |
| T0 dml-fp16 | 1536 | 1568 | 10 | 3509 / 3509 / **3510** | 0.25 | 26 | 223.7 | 10/10 | 6/10 |
| T0 dml-fp16 | 2016 | 2048 | 10 | 5543 / 5560 / **5560** | 8.23 | 23 | 324.2 | 10/10 | 9/10 |
| T1 dml-fp16 | 270 | 302 | 10 | 4064 / 4064 / **4064** | 0.0 | 26 | 155.1 | 10/10 | 10/10 |
| T1 dml-fp16 | 512 | 544 | 10 | 4526 / 4526 / **4526** | 0.12 | 26 | 161.5 | 10/10 | 2/10 |
| T1 dml-fp16 | 768 | 800 | 10 | 4654 / 4654 / **4654** | 0.09 | 24 | 190.4 | 10/10 | 3/10 |
| T1 dml-fp16 | 1024 | 1056 | 10 | 5451 / 5452 / **5452** | 0.12 | 23 | 227.9 | 10/10 | 2/10 |
| T1 dml-fp16 | 1536 | 1568 | 10 | 5707 / 5708 / **5708** | 0.15 | 21 | 330.8 | 10/10 | 7/10 |
| T1 dml-fp16 | 2016 | 2048 | 10 | 7758 / 7759 / **7759** | 0.22 | 18 | 569.5 | 10/10 | 10/10 |

## Teto de contexto por VRAM (D1)

Regra: VRAM minima = **maior pico medido + 400 MB** (driver, janela/compositor, fragmentacao), arredondada a 50 MB. Uma placa informa ao DXGI 8 GB ~ 8.176, 6 GB ~ 6.144, 4 GB ~ 4.096 MB.

| teto de contexto (tokens) | T1 pico max (MB) | T1 VRAM min (MB) | T0 pico max (MB) | T0 VRAM min (MB) |
|---|---|---|---|---|
| 270 | 4064 | 4450 | 1858 | 2250 |
| 512 | 4526 | 4950 | 2337 | 2750 |
| 768 | 4654 | 5050 | 2465 | 2850 |
| 1024 | 5452 | 5850 | 3246 | 3650 |
| 1536 | 5708 | 6100 | 3510 | 3900 |
| 2048 | 7759 | 8150 | 5560 | 5950 |

## Observacoes

- Carga de fundo: a regra era CPU <= 3% em 10 s antes de cada execucao; o PC tem apps do usuario (ChatGPT, claude, Taskmgr) que mantiveram 4-17%, entao a maioria das execucoes ficou marcada `noisy` (coluna acima). O pico de **memoria** e determinado pelo tamanho dos buffers do DML (graph capture + KV em `max_length`), nao pela carga de CPU: o desvio entre execucoes e de poucos MB.
- O pico cresce com `max_length` porque o pacote usa `past_present_share_buffer` + graph capture (o KV e alocado no tamanho maximo pedido) e o prefill aloca buffers intermediarios proporcionais ao prompt.
- A curva foi medida na 7800 XT (RDNA3, GenAI-WinML 0.15.2). A VRAM de processo depende do grafo/runtime, nao da potencia da GPU; em outras GPUs o valor deve ser da mesma ordem **[E]** ate a homologacao.
- Reproduzir: `C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe scripts/hwmap/vram_curve.py` (retomavel) e `python scripts/hwmap/vram_curve_report.py`.
