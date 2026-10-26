# Roteiro de homologação de hardware (~30 min por máquina)

Objetivo: validar, em PCs diferentes (RX 6000, RTX, Intel Arc, notebooks com iGPU, Pascal, Vega…), as regras de `configs/gpu_compat.yaml` e preencher os itens [E] de `docs/mapa-compatibilidade-gpu.md`.
Cada máquina devolve **um arquivo** em `reports/hwmap/machines/<hostname>_<AAAAMMDD>.json` (formato no §5). **Somente leitura**: nada de escrita no MongoDB nem em `Repos/`.

## 0. Prioridade das máquinas
1. GTX 1060/1070/1080 (Pascal) — decide D2 e o `dml-fp32` do T0. 2. RX 6600/6700 (RDNA2). 3. RTX 3060/4060. 4. Intel Arc A/B. 5. Notebook com 780M/680M. 6. Core Ultra / Iris Xe. 7. RX Vega 56/64, RX 580. 8. GTX 1650/1660.

## 1. Pré-requisitos (checklist, ~5 min)
- [ ] Windows 10 1903+ (ideal Windows 11); anote `winver`/build.
- [ ] Driver gráfico atualizado do fabricante; anote a versão (o detector também lê).
- [ ] DirectX 12: `dxdiag` → aba Exibir → "Níveis de recurso" contém 12_0 ou superior. (DirectML já vem no Windows ≥ 1903 [F]).
- [ ] Python 3.10–3.12 com `pyyaml` (detector) e `onnxruntime-genai-winml==0.15.2` + `numpy` (micro-benchmark DML). Use um venv novo só para isto (o detector em si não instala nada além de pyyaml).
- [ ] Este repositório (somente leitura) e os pacotes ONNX v3 copiados para `Repos\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-v3-ONNX-{dml-fp16,int4}` e `Repos\KapiCoder-Mongo-Small\exports\KapiCoder-Mongo-Small-v3-ONNX-{dml-fp16,int8}` (os pacotes são gitignored; copiar do PC de referência, conferindo `kapicoder_manifest.json`).
- [ ] Energia: notebook ligado na tomada, plano "Alto desempenho", fechar apps pesados; anote a carga de CPU antes (<5%).
- [ ] Para forçar a GPU certa em notebooks híbridos: Configurações → Sistema → Tela → Elementos gráficos → o `python.exe` do venv em "Alto desempenho". O DML escolhe o adaptador sozinho; **o resultado só vale se o contador confirmar a GPU esperada** (passo 3).

## 2. Detectar (2 min)
```powershell
python scripts\hwmap\detect_gpus.py --out reports\hwmap\machines\_tmp_detect.json
```
Confira: `chosen_adapter` (vendor/device/VRAM), `recommended_variant`, `needs_microbench`, `ignored`. Anote qualquer adaptador com `rule` = `*-unknown` (device_id novo → vira linha no YAML).

## 3. Micro-benchmark e latência (15–20 min)
Rodar **um processo por pacote**, ocioso, nesta ordem. LUID = `chosen_adapter.luid` do passo 2.
```powershell
# CPU (referência da máquina)
python scripts\hwmap\microbench.py --pkg Repos\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-v3-ONNX-int4 --provider cpu --tier t0 --out reports\hwmap\machines\_t0_cpu.json
# GPU T0 e T1 (DML)
python scripts\hwmap\microbench.py --pkg Repos\KapiCoder-Mongo-Nano\exports\KapiCoder-Mongo-Nano-v3-ONNX-dml-fp16 --provider dml --tier t0 --out reports\hwmap\machines\_t0_dml.json
python scripts\hwmap\microbench.py --pkg Repos\KapiCoder-Mongo-Small\exports\KapiCoder-Mongo-Small-v3-ONNX-dml-fp16 --provider dml --tier t1 --out reports\hwmap\machines\_t1_dml.json
```
Enquanto cada DML roda, em **outro** PowerShell confirme a GPU (obrigatório; o índice DXGI não vale):
```powershell
powershell -NoProfile -File scripts\export\gpu_probe.ps1 -ProcId <PID do python> -Samples 2
```
Aceite do passo: memória dedicada > 100 MB e utilização no LUID esperado; 0% na iGPU. Se subiu na iGPU, **descarte** e refaça com a GPU alta-performance fixada (§1).
Opcional (harness de latência completo, exige `Repos\...\merged\sft_v3` com o tokenizador e `transformers`/`psutil` — por isso é só para PCs de desenvolvimento): `scripts\export\onnx_run.py --mode latency --provider dml --tier t1 --pkg <pacote> --n 100 --luid <LUID> --out ...` (usa os prompts reais de `reports\export\work_v3\t1_prompts.json`; a bandeira `on_7800xt` do JSON significa "no LUID passado em `--luid`" e a sonda ainda confere a iGPU antiga pelo LUID fixo — ignore o nome). Não alterar `scripts\export`.

Casos especiais:
- **Pascal:** além de T0/T1 `dml-fp16`, anote TTFT e `nvidia-smi`; **se TTFT do T0 > 1 s ou o T1 > 3 s, pare** (não é falha do roteiro: confirma o risco do FP16 a 1/64 [F S5]) e registre `notes`. Se `dml-fp32` do T0 existir, repita com ele.
- **iGPU (notebook sem dGPU):** rodar T0 `dml-fp16` na iGPU **e** T0 `int4` na CPU; o ganho esperado é <30% ou perda (§3 do mapa).
- **Intel Arc / RTX 50 / RDNA4:** anotar erros de criação de sessão (código `0x…`) integralmente.

## 4. O que NÃO executar
- Nada de escrita no MongoDB; nenhum script de `scripts\data`, `scripts\publish`, treino, `build_variants.py`, `package_variant.py`, `prefix_cache_proto.py`.
- Não instalar CUDA Toolkit, OpenVINO, TensorRT, Olive, nem outros drivers/EPs.
- Não editar `scripts\export`, `Repos\*\exports`, `data\`; não commitar.
- Não rodar outros benchmarks/jogos em paralelo; não usar overclock/undervolt durante a medição.

## 5. Arquivo de resultado (um por máquina)
Salvar como `reports\hwmap\machines\<hostname>_<AAAAMMDD>.json` (apagar os `_tmp*`/`_t*_*.json` depois de juntar):
```json
{
  "schema": "kapi-hwmap-machine-v1",
  "date": "2026-10-27", "hostname": "PC-DO-FULANO", "operator": "nome",
  "os": {"build": "10.0.26100", "power_plan": "Alto desempenho", "on_ac": true},
  "cpu": {"name": "...", "physical_cores": 8, "ram_gb": 32, "load_pct_before": 2},
  "detect": { "...": "conteudo completo do passo 2 (detect_gpus.py)" },
  "gpu_confirmed_by_counter": true,
  "gpu_probe": {"proc_dedicated_mb": {"0x...": 0}, "proc_engine_util_pct": {"0x...": 0}},
  "runs": [
    {"tier": "t0", "package": "v3-ONNX-int4",     "provider": "cpu", "ttft_ms_p50": 0, "decode_tok_s": 0, "ok": true},
    {"tier": "t0", "package": "v3-ONNX-dml-fp16", "provider": "dml", "ttft_ms_p50": 0, "decode_tok_s": 0, "ok": true, "vram_proc_mb": 0},
    {"tier": "t1", "package": "v3-ONNX-dml-fp16", "provider": "dml", "ttft_ms_p50": 0, "decode_tok_s": 0, "ok": true, "vram_proc_mb": 0}
  ],
  "verdict": {"variant_chosen_by_rule": "t1-dml-fp16", "variant_that_would_be_accepted": "t1-dml-fp16", "agrees_with_yaml": true},
  "errors": [], "notes": "..."
}
```
Preencher `verdict.agrees_with_yaml` comparando a variante do detector com a que passaria nos critérios do §6 do mapa (`microbench` no YAML). Qualquer divergência vira correção da regra (`evidence` passa de E para M).

## 6. Depois
Devolver o JSON (e prints de `dxdiag` só se houver erro). O mantenedor consolida em `docs/mapa-compatibilidade-gpu.md` e atualiza `configs/gpu_compat.yaml` (`evidence: M …`, `confidence`).
