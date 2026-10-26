# Exportação ONNX dos alunos v3 (T0 Nano 0,5B e T1 Small 1,5B)

Data: 2026-10-26 · builder `onnxruntime-genai` **0.15.2** (= IDE) · origem `merged/sft_v3` (T0 full FT, T1 LoRA r=64 fundido), bf16 · pacotes em
`Repos/KapiCoder-Mongo-<Nano|Small>/exports/<família>-v3-ONNX-<variante>/` (gitignored; **os pacotes antigos não foram tocados**) · reprodução: `scripts/export/README.md` (§ Geração v3, `KAPI_EXPORT_GEN=v3`) ·
comandos KapiLab: `reports/export/kapilab_commands.md`. **Tudo é PROXY/não qualificado** (latência/memória do GenAI Python; nenhum gate marcado como aprovado oficialmente).

## 1. Pacotes

T0: `int4` 392 MB, `int8` 1.071 MB (contorno `op_types_to_quantize=MatMul` do builder), `int8-emb4` 633 MB, `dml-fp16` 959 MB. T1: `int8` 2.578 MB, `dml-fp16` 2.964 MB.
**Fora (de propósito):** T1 `int4`/`dml-int4` (no v2 perdiam −1,4 a −1,8 p.p. de FIM com p ≤ 0,04, fora de cogitação), T0 `dml-int4`, `int4-rtn`/`kquantmixed`, fp32 (referência do v2; sem demanda) e qualquer `qnn-*` (sem máquina/QAIRT).

- Tokenizer, vocab, merges e `generation_config.json` byte a byte do `merged/sft_v3` (SHA-256 testado); `chat_template.jinja` removido. `genai_config`: só `search` alterado (greedy, `max_length` 2048); **`eos_token_id` = 151643 (sem o contorno do EOS)**.
- Metadata `kapibarastudio-model.json` (schema `kapilab-kapi-model-v1`): nome `…-v3 ONNX <VARIANTE>`, versão `3.0.0-preview.1`, base Qwen2.5-Coder (revisão fixada), `datasetVersion v3`, `training` (run `sft_v3`, método full/LoRA, hash dos pesos merged, hash do manifesto do corpus v3, hash da config e fingerprint do treino, commit), licença Apache-2.0 + aviso de modificação (cita o professor 7B e as consultas geradas pelo Qwen3-14B, Apache-2.0), `LICENSE`, `NOTICE.md`, `README.md`, `kapicoder_manifest.json`, `export_manifest.json` (sha256+bytes de todos os arquivos, comando exato, hashes do merged).
- **Validação com o KapiLab real** (apphost publicado Cpu, somente leitura; saídas em `reports/export/kapilab_validate_v3_t0|t1.json`): `model validate --strict-kapi --verify-hashes` nos 6 pacotes → `Available`, `schemaValid`, `ideSupported`, `strictSchema kapilab-kapi-model-v1`, `hashesValid`, 0 issues. Não rodei `bench`/`model run` do KapiLab (não pedido; GPU exige fila+token).
- Testes: `scripts/export/tests` = 101 passam (inclui 30 dos pacotes v3 com hash completo: manifestos, `genai_config`, metadata, tokenizer = merged, proveniência ligada ao `run_manifest.json`, EOS padrão 151643).

## 2. GPU (duas placas)

LUIDs mudam por boot: recalibrados nesta sessão alocando 3 GB com torch-ROCm em `cuda:0` ("AMD Radeon RX 7800 XT"): **7800 XT = `0x00000000_0x00012602`** (3.299 MB dedicados do processo), iGPU = `0x00000000_0x00014830` (0 MB). Uma 1ª tentativa de `verify` com o LUID antigo (`…12951`) foi **recusada pela trava** (exit 3) e descartada — a trava funciona. Depois, em todas as execuções DML (quality, homologação, latência) os contadores `GPU Process Memory`/`GPU Engine` por pid mostraram memória dedicada e utilização só na 7800 XT: T0 `dml-fp16` 2.933 MB, T1 `dml-fp16` 5.133 MB; **0 MB / 0% na iGPU**. Ao final não há processo usando GPU.

## 3. Qualidade e paridade vs HF bf16 do mesmo modelo v3

Val v2c, 500 ids (seed 20261020) de `base-v2c/sample_ids.json`, `editor-context-v1`, ctx 2048, greedy ≤ 96 tokens, scorer do `eval_teacher`; paridade token a token nos 200 primeiros (HF regenerado na 7800 XT = predições armazenadas em 100%). Δ pareado e McNemar exato contra as mesmas linhas do HF (n=500). Tokenizador nativo do GenAI = HF em 500/500 prompts em todos os pacotes.
HF bf16 n=500: T0 FIM 0,394 · exec 0,576 (÷teto 0,624 = 0,923) · campos 0,032 · sintaxe 0,992; T1 FIM 0,398 · exec 0,586 (÷teto 0,939) · campos 0,026 · sintaxe 0,998.

| tier | variante | MB | greedy idêntico % (prefixo aceito %) | val: FIM (Δpp; p) | val: exec (Δpp; p) · exec÷teto | val: campos inex. (Δpp) |
|---|---|---|---|---|---|---|
| T0 | int4 | 392 | 81,0 (86,2) | 0,390 (−0,4; 0,63) | 0,578 (+0,2; 1,0) · 0,926 | 0,030 (−0,2) |
| T0 | int8 | 1071 | 96,5 (97,8) | 0,396 (+0,2; 1,0) | 0,576 (0,0; 1,0) · 0,923 | 0,032 (0,0) |
| T0 | int8-emb4 | 633 | 94,0 (95,9) | 0,394 (0,0; 1,0) | 0,576 (0,0; 1,0) · 0,923 | 0,032 (0,0) |
| T0 | dml-fp16 | 959 | 97,0 (98,0) | 0,394 (0,0; 1,0) | 0,576 (0,0; 1,0) · 0,923 | 0,030 (−0,2) |
| T1 | int8 | 2578 | 97,0 (98,4) | 0,400 (+0,2; 1,0) | 0,586 (0,0; 1,0) · 0,939 | 0,026 (0,0) |
| T1 | dml-fp16 | 2964 | 98,0 (98,8) | 0,402 (+0,4; 0,50) | 0,586 (0,0; 1,0) · 0,939 | 0,026 (0,0) |

Homologação v1 (**n=500** = primeiros 500 itens por sha1(id), idêntico a `--limit 500`; acesso só por `homolog_guard.authorized`/`eval_homolog` com log em `access.log.jsonl`; prompts ids = prompt selado em 500/500; exec do texto real no Mongo RO; só agregados aqui, textos ficam em `data/homolog/results/onnx_v3_*`; `scripts/export/homolog_onnx.py`, agregados em `reports/export/homolog_onnx_v3_t0|t1.json`).
HF bf16 (mesmos 500): T0 FIM 0,426 · exec 0,818 · exec÷teto 0,874 · campos 0,092; T1 FIM 0,458 · exec 0,840 · exec÷teto 0,897 · campos 0,072 (teto 0,936).

| tier | variante | hom: FIM (Δpp; p) | hom: exec (Δpp; p) | hom: exec÷teto (Δpp) | hom: campos inex. (Δpp; p) |
|---|---|---|---|---|---|
| T0 | int4 | 0,420 (−0,6; 0,65) | 0,816 (−0,2; 1,0) | 0,872 (−0,2) | 0,098 (+0,6; 0,66) |
| T0 | int8 | 0,422 (−0,4; 0,63) | 0,814 (−0,4; 0,73) | 0,870 (−0,4) | 0,096 (+0,4; 0,69) |
| T0 | int8-emb4 | 0,426 (0,0; 1,0) | 0,814 (−0,4; 0,69) | 0,870 (−0,4) | 0,096 (+0,4; 0,63) |
| T0 | dml-fp16 | 0,422 (−0,4; 0,50) | 0,810 (−0,8; 0,22) | 0,865 (−0,9) | 0,098 (+0,6; 0,38) |
| T1 | int8 | 0,458 (0,0; 1,0) | 0,836 (−0,4; 0,63) | 0,893 (−0,4) | 0,074 (+0,2; 1,0) |
| T1 | dml-fp16 | 0,452 (−0,6; 0,38) | 0,838 (−0,2; 1,0) | 0,895 (−0,2) | 0,076 (+0,4; 0,50) |

Leitura: nenhuma variante tem diferença significativa contra o HF (menor p = 0,22; |Δ| ≤ 0,9 pp, com 2–4 itens de 500 por métrica). Os campos inexistentes sobem ≤ 0,6 pp na homologação (T0 ficou em 9,6–9,8% contra 9,2% do HF; **a meta de campos do v3 é medida no HF completo, não nestes n=500**). Paridade greedy: fp16 DML/int8 ≈ piso do bf16 (96,5–98%); `int8-emb4` 94%; `int4` do T0 81% (informativa; sem perda de qualidade medida). Top-1 teacher-forced não medido (GenAI 0.15 só expõe logits da última posição).

## 4. Latência e memória (PROXY)

`onnx_run.py --mode latency` (GenAI Python 0.15.2, ids pré-tokenizados, `ClearProviders/AppendProvider` como a IDE, threads padrão, batch 1): 3 aquecimentos descartados + **200** prompts do val (mediana **260** tokens, ≤ 32 tokens gerados); tok/s = decodificação forçada de 64 tokens; RAM = `peak_wset`; um processo por pacote. **Máquina não ociosa de verdade** (ChatGPT/Taskmgr/Codex abertos do usuário; carga de fundo 1–2% nas medidas finais; as 3 variantes CPU do T0 foram **repetidas** porque a 1ª rodada começou com 9–14% de carga — guardadas em `work_v3/*_run1.json`; T1 `int8` mediu com 1,9%). Comparação com os gates de `configs/gates.yaml`/`gates.ide.json` via `scripts/eval/check_gates.py` (`scripts/export/gates_proxy.py`; medidas do owner kapilab marcadas `proxy:true` → status `unqualified`, resultado cru abaixo).

| tier / cenário | variante | TTFT méd / p95 ms (gate) | tok/s | RAM pico GB (gate) | VRAM ded. 7800 XT | disco MB (gate) | paridade (gate) |
|---|---|---|---|---|---|---|---|
| T0.cpu | int4 | 301 / 506 ✗ (150/250) | 120 | 1,53 ✓ (1,8) | – | 392 ✓ (700) | ΔAPT +0,10 ✓ · Δexec +0,2 pp ✓ |
| T0.cpu | int8 | 279 / 462 ✗ | 90 | 1,57 ✓ | – | 1071 ✗ | ΔAPT +0,03 ✓ |
| T0.cpu | int8-emb4 | 280 / 475 ✗ | 90 | 1,58 ✓ | – | 633 ✓ | ΔAPT +0,01 ✓ |
| T0 DML (cenário CPU só informativo) | dml-fp16 | **105 / 119** | 172 | 2,27 | 2.933 MB | 959 | greedy 97,0 ✓ (≥ 97) |
| T1.cpu_int8 | int8 | 768 / 1281 ✗ (500/650) | 32 | 3,01 ✓ (3,8) | – | 2578 | ΔAPT +0,01 ✓ |
| T1.dml_fp16 | dml-fp16 | **141 / 166 ✓** (170/200) | 108 ✓ (≥ 90) | 2,26 | 5.133 MB | 2964 | greedy 98,0 ✓ |

Cache de prefixo (política B, `prefix_cache_proto.py bench`, n=200, 5 aquecimentos, ordem aleatória por pedido, só CPU, sonda de GPU = 0 MB/0%), TTFT média/p95 ms, threads padrão e 12 — **confirma o protótipo anterior**:

| pacote | sem cache | A (cabeçalho/linguagem) | **B (hit estável)** | v2 (B antes) |
|---|---|---|---|---|
| T0 int4, padrão | 324 / 523 | 254 / 450 | **105 / 208 ✓ (≤150/250)** | 92 / 221 (24 thr) |
| T0 int4, 12 thr | 318 / 506 | 249 / 440 | **102 / 211 ✓** | 98 / 207 |
| T1 int8, padrão | 802 / 1278 | 628 / 1102 | **246 / 526 ✓ (≤500/650)** | 250 / 497 (24 thr) |
| T1 int8, 12 thr | 795 / 1273 | 627 / 1080 | **246 / 519 ✓** | 235 / 482 |

(A não passa; sem cache nada passa em CPU. A equivalência greedy 100% do cache **não foi reexecutada** neste relatório; vale a do v2 — mesma API, outro peso.) O baseline do T1 mediu ~8% acima do v2 (802 × 747 ms): ruído de máquina/aplicativos de fundo.

## 5. Comparação com os pacotes antigos (`sft_distill_v1`, mesmo método)

Tamanhos idênticos por variante (mesma arquitetura). Qualidade no val n=500: HF antigo T0 0,384/0,564 → v3 0,394/0,576; T1 0,412/0,582 → 0,398/0,586 (recortes e modelos diferentes; use os Δ pareados vs HF, que são ≈ 0 nos dois). Perdas de quantização: no v2 os `int4` do T1 perdiam −1,4/−1,8 pp (significativo); o v3 T0 `int4` perde −0,4 pp (p 0,63). Latência T0 int4 301/506 × 285/480 (v2), `dml-fp16` 105/119 × 109/124; T1 int8 768/1281 × 697/1192 (≈ +10%, máquina mais carregada), `dml-fp16` 141/166 × 148/170; paridade greedy: T0 `int4` 81,0 × 79,0, `int8` 96,5 × 95,0, `dml-fp16` 97,0 × 97,5, T1 `int8` 97,0 × 94,5, `dml-fp16` 98,0 × 96,5. O achado do EOS da IDE (§7) continua valendo igual.

## 6. Recomendação

| tier | melhor variante | por quê |
|---|---|---|
| T0 CPU | **`int4`** (392 MB) | Δ val/hom ≤ 0,6 pp (p ≥ 0,63), menor RAM/disco, passa RAM/disco/paridade; TTFT CPU só passa com cache B. `int8-emb4` (633 MB) é a alternativa mais fiel (94%); `int8` estoura 700 MB |
| T0 GPU | **`dml-fp16`** (959 MB) | 105/119 ms, 172 tok/s, greedy 97,0%; hom exec −0,8 pp (p 0,22, ruído) |
| T1 GPU | **`dml-fp16`** (2,96 GB) | única que passa TTFT 141/166 e ≥ 90 tok/s (108); sem perda de qualidade |
| T1 CPU | **`int8`** (2,58 GB) | sem perda; RAM ✓; TTFT 768/1281 reprova 500/650 (passa com cache B: 246/526) |

## 7. Pendências e decisões do usuário

1. **EOS da IDE (onnxruntime-genai 0.15.2 duplica o último token ao terminar em EOS):** pacotes entregues **sem** contorno. Variante com contorno (`eos_token_id=151935`, cópia `…-eosfix` por hardlink, manifestos refeitos, metadata `eosWorkaround`): `KAPI_EXPORT_GEN=v3 python -I scripts/export/package_variant.py --tier <t0|t1> --variant <v> --eos-workaround` (testado e removido). Decidir: corrigir na IDE (`GetNextTokens()`/`IsDone`) ou aceitar o contorno.
2. Gate de TTFT de CPU (T0 150/250; T1 500/650) é inatingível sem cache de prefixo neste Ryzen com prompts de ~270 tokens; só passa com a política B (acerto estável). Recalibrar o gate, ou adotar cache por aba, ou declarar GPU-first.
3. Referência de paridade dos gates greedy: bf16 impõe piso ~96–97%.
4. KapiLab: bench CPU/GPU e `--via-queue` com token **não rodados** (GPU depende do token da fila); gate oficial DML do T1 pendente. QNN/NPU não feito. T1 int4 ficou fora (confirmar).
5. Ressalva herdada do v3: a homologação deixou de ser totalmente independente do treino v3 (mesmo gerador LLM/templates); os Δ ONNX×HF não são afetados, mas as taxas absolutas hom estão otimistas.
