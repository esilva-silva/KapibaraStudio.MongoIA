# PR 5 — Garantir o adaptador DirectML correto (GPU dedicada) no `OnnxLocalModelRuntime`: DXGI efetivo + micro-benchmark + queda segura para CPU

> Especificação (rascunho) para outra IA implementar no repositório do Kapibara Studio. É autocontida: leia-a inteira antes de começar. **Nada foi implementado na IDE**; o checkout de referência (`F:\source\esilva-silva\EsilvaSoft.KapibaraStudio`) é somente leitura e só foi lido.
> Escrita em 2026-10-26. Evidência: `C:\IA\models\KapibaraStudio.MongoIA\reports\hwmap\phase2_measurements.md` (§0 e §7), `docs\mapa-compatibilidade-gpu.md` (Decisões D1–D5, §6), `configs\gpu_compat.yaml` (schema 2), protótipos: `scripts\hwmap\detect_gpus.py`, `dxgi_select.py`, `microbench.py`, `phase2_bench.py`.
> Objetivo de negócio: o autocomplete em GPU só vale se o DirectML rodar **na GPU dedicada certa**. Se a IDE pedir "a GPU X" e o runtime cair em silêncio na iGPU, no WARP ou numa GPU lenta, o TTFT estoura (iGPU de 2 CU: 13× mais lenta no T0 e 26× no T1 que a RX 7800 XT [M]) sem que ninguém perceba.

## 1. Achado de produto (medido)

O **GenAI-WinML 0.15.2 ignora** `device_id`, `device_filter`, `performance_preference` e `device_filtering_options` do provider `dml` (testados nesta máquina; reports/hwmap/phase2_measurements.md §0): **qualquer pedido cai no adaptador padrão do DirectML**. Neste PC o padrão é a RX 7800 XT (0x1002/0x747E), por isso a IDE "funciona" aqui; num notebook híbrido (iGPU + dGPU) o padrão costuma ser a iGPU (`S2`: o device 0 "nem sempre é a mais rápida"), e o resultado seria um autocomplete 10–25× mais lento que a CPU. Os LUIDs mudam a cada boot e o índice DXGI também pode mudar — **nunca** escolher por índice nem por LUID guardado.

Para medir a iGPU de propósito o laboratório usa `scripts\hwmap\dxgi_select.py`, que remapeia a enumeração DXGI **só dentro do processo de teste** (gancho na vtable do `IDXGIFactory`). Isso é um recurso de medição; **não** é a solução recomendada para a IDE (ver §3, opção D, risco alto).

## 2. Regras do repositório (obrigatórias)

- Leia `CLAUDE.md`, `AGENTS.md` da raiz e o `AGENTS.md` de cada diretório antes de editá-lo; documentação e mensagens em pt-BR, identificadores em inglês.
- Não mexer em autenticação, credenciais ou permissões do Claude/Copilot.
- Mudança aditiva: o comportamento de CPU não muda; sem GPU elegível o autocomplete segue em T0 `int4` na CPU. Nada que escreva no registro do Windows sem consentimento explícito do usuário (a preferência de GPU por app é **configuração do sistema**: pedir, nunca impor em silêncio).
- Não alterar o contrato `editor-context-v1` nem os goldens.

## 3. Entregáveis

### 3.1 Detecção do adaptador DXGI **efetivo**

O que importa é em qual adaptador o DML **de fato** vai criar o dispositivo, não qual a IDE "pediu".

- **A. Escolha por intenção:** enumerar com `IDXGIFactory6::EnumAdapterByGpuPreference(DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)` (e `EnumAdapters1` como fallback), ler `DXGI_ADAPTER_DESC1` (VendorId, DeviceId, SubSysId, Revision, `DedicatedVideoMemory`, `Flags & DXGI_ADAPTER_FLAG_SOFTWARE`) e classificar por `configs/gpu_compat.yaml` (vendor/device/VRAM, nunca por índice) — mesma lógica de `scripts\hwmap\detect_gpus.py` (portar ou chamar o JSON de saída como referência de comportamento). O resultado é o **adaptador pretendido** + `recommended_variant` + `context_cap_tokens`.
- **B. Adaptador padrão efetivo do processo:** o do `EnumAdapterByGpuPreference(UNSPECIFIED)` / índice 0 do processo da IDE **após** aplicar a preferência de GPU do Windows para o executável (ver 3.2). É o que o DML 0.15.2 vai usar. Comparar B com A por `(VendorId, DeviceId, LUID)` do mesmo boot.
- **C. Prova por contadores (autoridade final):** após criar a sessão, confirmar a memória dedicada do **próprio pid** no LUID pretendido com os contadores `\GPU Process Memory(pid_*_luid_*)\Dedicated Usage` e `\GPU Engine(pid_*_luid_*)\Utilization Percentage` (`vram_sampler.ps1`, `gpu_probe.ps1`; critério `adapter_ok` de `phase2_bench.py`: > 100 MB no LUID pretendido e ≤ 50 MB / ≤ 1% de engine nos demais). Se a memória subiu em outro LUID, o adaptador efetivo **não** é o pretendido.

### 3.2 Forçar a GPU certa: preferência de GPU do Windows por aplicativo

- A forma suportada de mudar o adaptador padrão de um processo, sem hack, é a **preferência de gráficos por app** do Windows (Configurações › Sistema › Tela › Elementos gráficos › "Alto desempenho"; grava `HKCU\Software\Microsoft\DirectX\UserGpuPreferences`, valor `GpuPreference=2;` para o caminho do executável). Entregar:
  - verificação (leitura) da preferência atual do executável da IDE e do host de IA (se for um processo separado);
  - caixa de diálogo/atalho "Usar a GPU de alto desempenho para o autocomplete" que **abre a página de Configurações** ou, com consentimento explícito e botão "Aplicar", grava a chave do usuário atual; nunca em silêncio; desfazer registrado; documentar no AGENTS.md;
  - após aplicar, exigir reiniciar o processo (a preferência vale na criação do dispositivo) e repetir 3.1 B/C.
- **Opção D (não recomendada):** ganchos de DXGI no processo da IDE (como `dxgi_select.py`). Intrusivo (patch de vtable), frágil entre versões do Windows/drivers, risco de antivírus; mantido só no laboratório.
- **Opção E (preferida a médio prazo):** subir o GenAI/WinML para uma versão que respeite `device_id`/`device_filter`/`device_filtering_options` e **validar por teste** (mesma bateria do §4) antes de confiar; só então a IDE passa o `device_id` do adaptador pretendido. A seleção por provider options continua exigindo C (prova por contador).

### 3.3 Validação com o micro-benchmark (e fallback seguro)

- Executar o micro-benchmark de primeira execução (`docs/mapa-compatibilidade-gpu.md` §6; `scripts\hwmap\microbench.py`, processo filho isolado, 270 tokens de prompt + 32 de decodificação, 1 aquecimento + 3 repetições) **no adaptador efetivo**, sempre que: a família é `unknown`/Pascal/Polaris/Intel/Blackwell (`needs_microbench`), a VRAM cai na faixa com teto (6–8 GB, D1) ou o adaptador efetivo ≠ o pretendido.
- Critério de aceite (YAML `microbench`, [M] só na 7800 XT/Zen 4): T0 `TTFT_p50 ≤ 250 ms`, `≥ 60 tok/s`, `TTFT_gpu ≤ 0,7 × TTFT_cpu(T0 int4)`; T1 `≤ 250 ms`, `≥ 40 tok/s`, `≤ 1,0 ×`.
- **Queda segura para CPU (T0 `int4`), sem erro visível,** quando qualquer item ocorrer: (1) o adaptador efetivo é iGPU, `Microsoft Basic Render Driver`/WARP ou `vendor` desconhecido; (2) a prova por contador (3.1 C) diz que a memória não subiu no LUID pretendido; (3) o micro-benchmark reprova ou estoura o tempo (carga > 8 s, primeira geração > 3 s, aborto no aquecimento se TTFT > 4× o teto); (4) `DXGI_ERROR_DEVICE_REMOVED/HUNG`, falha de sessão DML, OOM; (5) o *watchdog* em produção vê TTFT p95 > 2× o calibrado em 3 sugestões seguidas. Três falhas em sessões distintas → `gpu_disabled` no cache até mudar driver/GPU/versão; o usuário pode forçar GPU/CPU nas configurações. Híbrido em que a dGPU não é o padrão **e** o usuário recusa a preferência de GPU: CPU.
- **Cache** do resultado por `adapter(vendor,device,subsys,revision) + driver_version + genai_version + hash do pacote + build do Windows` (+ a preferência de GPU do executável); reavaliar quando qualquer item mudar, no 3º fallback seguido ou pelo botão "Recalibrar".
- **Teto de contexto por VRAM (D1):** a IDE aplica `context_cap_tokens` do detector ao orçamento de prompt do autocomplete (truncar o envelope `editor-context-v1` antes de estourar a VRAM): T1 só em dGPU ≥ 8 GB com contexto de até 2.048; 6–8 GB só T1 com teto menor e micro-benchmark aprovado; T0 ≥ 3,5 GB com teto proporcional; abaixo disso CPU. Os valores vêm de `context_cap_by_vram` no YAML (curva medida em `reports/hwmap/vram_curve.md`).

### 3.4 Telemetria e configuração

- Registrar (por sessão de autocomplete, sem texto do usuário): adaptador pretendido e efetivo (vendor/device, nunca o LUID persistido), `recommended_variant`, `context_cap_tokens` aplicado, resultado do micro-benchmark, motivo de qualquer queda para CPU (`igpu`, `warp`, `counter_mismatch`, `microbench_fail`, `device_lost`, `watchdog`), TTFT por adaptador.
- `AutocompleteSettings`: `GpuMode` (`Auto` padrão | `ForceCpu` | `ForceGpu` avançado), `RequireHighPerformanceGpu` (padrão `true`).
- Painel/diagnóstico: mostrar "GPU usada: …, variante …, contexto máx. … tokens" e "GPU de alto desempenho não é o padrão do Windows — [Corrigir]".

### 3.5 Testes

- Unitários (fakes de DXGI): híbrido iGPU + dGPU com padrão = iGPU → detecta divergência pretendido × efetivo e fica em CPU quando o usuário recusa a preferência; dGPU como padrão → aceita; WARP/iGPU sozinhos → CPU; ordem de enumeração e LUIDs embaralhados entre execuções não mudam a escolha (por vendor/device/VRAM); VRAM nas faixas ≥ 8 GB / 6–8 GB / 3,5–6 GB / < 3,5 GB → variante e `context_cap_tokens` esperados.
- Integração (máquina com 2 GPUs, como este PC): contadores confirmam `adapter_ok`; com a preferência "Economia de energia" para o executável, o teste **detecta** a iGPU como efetiva e a IDE cai para a CPU com o motivo registrado; com "Alto desempenho" volta à GPU. Repetir 10 inicializações com reboot entre elas (LUIDs mudam).
- Regressão: sem GPU (ou `ForceCpu`) mesmo comportamento e mesmos números de hoje; zero mudança nos testes existentes.

## 4. Critérios de aceite (gates)

- Em máquina híbrida, **0 de 20** execuções de autocomplete em GPU rodam num adaptador diferente do pretendido sem queda para a CPU registrada (prova por contador em 100% das inicializações).
- Em máquina de um adaptador (dGPU) o custo extra de validação ≤ 10 s na primeira execução (≤ 2 s com cache) e nenhum bloqueio da UI (em segundo plano, com CPU T0 int4 servindo enquanto isso).
- TTFT na GPU validada dentro do gate do tier (T0 150/250 ms; T1 500/650 ms) e `context_cap_tokens` respeitado em 100% dos pedidos (VRAM de processo ≤ VRAM − margem de 400 MB).
- Nenhuma escrita no registro/configuração do Windows sem consentimento explícito e reversível.

## 5. Fora de escopo

- Variantes novas (CUDA/TensorRT-RTX/OpenVINO/MIGraphX/GGUF, `dml-int4`, `dml-fp32`): decisão D5, reavaliar quando houver builder/ORT GenAI mais novo.
- iGPUs maiores (680M/780M/880M/890M/8060S, Arc de Core Ultra, Iris Xe): ficam na CPU (D3, `igpu_policy: cpu_always_until_measured`) até haver medição real pelo roteiro de homologação.
- NPUs, Linux/macOS, prefix cache em DML (PR 4).

## 6. Riscos

| risco | mitigação |
|---|---|
| Preferência de GPU por app é configuração do sistema (consentimento, política corporativa) | pedir ao usuário, nunca impor; queda segura para CPU quando recusada |
| O GenAI futuro passa a respeitar `device_id` e a IDE ainda assume o padrão | teste de regressão na atualização (R1 da Fase 08): repetir a bateria §3.5 com o novo GenAI antes de confiar |
| Contadores de GPU indisponíveis (políticas, Windows Server, VM) | sem prova ⇒ tratar como não validado e usar o micro-benchmark com limiar de ratio contra a CPU; sem os dois ⇒ CPU |
| Validação lenta na primeira execução | executar em segundo plano com a CPU servindo; cache por chave |
| Detecção de "padrão efetivo" divergir entre APIs DXGI de Windows diferentes | autoridade final = contador por pid (3.1 C), não a enumeração |
