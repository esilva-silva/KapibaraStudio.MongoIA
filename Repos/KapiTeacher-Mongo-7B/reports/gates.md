# Gates do professor — KapiTeacher-Mongo-7B (Fase 02 §3.6)

Gerado em 2026-10-12 21:05:10 por `scripts/teacher/write_gates.py`. Professor: Qwen2.5-Coder-7B Base (`Qwen/Qwen2.5-Coder-7B@0396a76181e1`), decisao em `reports/teacher_choice.json` (14B descartado).

## G2.1 Checkpoint — PARCIAL (`pending_ide`)

| item | status | evidencia |
|---|---|---|
| Hashes = HF (revisao fixada) | **pass** | 13 arquivos x `base-repos.lock.json`; 13 comparados com a API do HF na revisao fixada (LFS sha256 / git blob sha1); sem .bin/pickle |
| `config.json` = tabela do §2 | **pass** | 28 camadas, hidden 3584, 28 heads, 4 KV, vocab_size 152.064, 32.768 pos., rope_theta 1e6, embeddings nao atrelados (todos conferem) |
| IDs FIM/eos e tokenizer | **pass** | fim_prefix 151659, fim_middle 151660, fim_suffix 151661, fim_pad 151662, eos 151643; cada marcador = 1 token; `tokenizer.json` identico ao dos students 0.5B/1.5B (sha `c0382117ea32`); 151665 tokens reais; CRLF round-trip ok |
| FIM sanity 20 prompts | **pass** | 20/20 sem marcadores `<|fim_*|>` vazando (NF4, greedy, 64 tokens); 18/20 pararam em `<|endoftext|>` ou eco do sufixo, os demais atingiram o limite sem vazar marcador |
| Padding do vocab mascarado | **pass** | lm_head 152064 linhas x 151665 reais (399 de padding); logits[:151665] antes do argmax; nenhum ID de padding gerado; argmax sem mascara no padding: 0/20 |
| Paridade de IDs Python x C# (400 prompts) | **pending_ide** | NAO verificada: exige o `QwenFimPromptBuilder` + `OnnxModelTokenizer` da IDE via `tools/ContractProbe` (.NET), nao executado aqui. IDs esperados do lado Python gravados em `reports/parity_python_ids.jsonl` (400 prompts, sha256 em `checkpoint_verification.json`) para a comparacao futura. Sem aprovacao fingida. |
| Perplexidade 2k/8k/32k (opcional §3.1) | not_run | fora do escopo desta etapa |

**G2.1 = partial_pending_ide**: todos os itens executaveis aqui passaram; falta apenas a paridade Python x C# (depende da IDE).

## Baseline zero-shot FIM no `val` (Fase 02 §3.2)

Amostra: **n=1000** de 2830 exemplos do `val` de `data/generated/v1/fim.jsonl` (estratificado por cut_kind x dominio, seed 20261008; sha256 da amostra `9317e5b9c45c0006`, dataset `6b4065ce37dff97c`). Contrato `fim-raw` (QwenFimPromptBuilder sobre prefix/suffix, sem cabecalho de contexto, pois o v1 nao traz `context`), greedy, max 96 tokens, contexto 2048. FIM puro: chat/blind/Generic-JS nao avaliados nesta etapa.

| modelo | carga | n | EM | FIM acc. (norm.) | chars aceitos | syntax | mongo_valid (estatico) | **exec-valid** | campos inex. | exec-valid @32 | GPU (min) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-7B | nf4 | 1000 | 16.9% | 16.9% | 4.19 | 85.5% | 81.1% | 50.4% | 12.3% | 44.3% | 15.9 |
| Qwen2.5-Coder-1.5B | bf16 | 1000 | 13.0% | 13.0% | 3.95 | 76.5% | 68.9% | 40.2% | 13.7% | 35.5% | 12.3 |
| Qwen2.5-Coder-0.5B | bf16 | 1000 | 9.7% | 9.7% | 3.55 | 66.1% | 57.5% | 31.4% | 10.9% | 25.8% | 11.5 |
| deepseek-coder-6.7b-base | nf4 | 1000 | 18.5% | 18.5% | 4.54 | 91.2% | 86.2% | 52.5% | 14.5% | 45.9% | 14.7 |

Teto do validador (mesmas checagens no `middle` de referencia): syntax 100.0%, mongo_valid 100.0%, exec-valid 89.6%; o restante e limite do parser de subconjunto do Console (stage `api`), nao erro do modelo.
exec-valid = sintaxe + API do Console + campos do schema + tipos + execucao somente-leitura (maxTimeMS 2.000) em `kapi_fx_*` sobre `prefix + predicao + suffix`, com predicao nao vazia. Detalhes e por cut_kind/dominio: `baseline_base/<modelo>/{summary.json,report.md}`.

## Gates que exigem SFT

| gate | status |
|---|---|
| G2.2 Ganho sobre o base (FIM-acc +15 p.p.; exec-valid >= 0,95; campos inex. <= 2%) | nao avaliado (exige SFT) |
| G2.3 Chat curto (codigo) | nao avaliado (exige SFT) |
| G2.4 Supera os alunos treinados sem professor (+5 p.p.) | nao avaliado (exige SFT; o baseline acima so da o ponto de partida do base) |
| G2.5 Sem esquecimento grave (Generic-JS FIM) | nao avaliado (exige SFT; base do Generic-JS nao medido nesta etapa) |
| G2.6 Professor auxiliar | nao avaliado (fora de escopo) |
| G2.7 Contaminacao (0 test/bench/blind no treino) | nao avaliado (sem treino ainda) |
| G2.8 Papel de T2 (DML INT4 na IDE) | nao avaliado (exige SFT + export) |
