# Destilacao sequence-level do KapiTeacher-Mongo-7B (autocomplete FIM, `editor-context-v1`)

Escopo desta rodada: **so autocomplete FIM**. O chat curto fica FORA ate a decisao **Q1** do usuario (nenhum script aqui le `chat.jsonl` como prompt;
o chat de val/test/bench entra apenas no indice de decontaminacao).

Premissa (`reports/teacher_adoption.json`): o G2.2 **nao foi aprovado** (exec-valid/teto 0,9209 < 0,95; campo inexistente 12-14% sem schema visivel).
Logo o professor **nunca rotula as cegas**: toda candidata passa pelo validador do dataset (plano B da Fase 02 §3.5 / G2.4) e o `gold` so serve para medir.

```
data/validated/v2c (SO train)            configs/distill_v2c.yaml
        |  build_prompt_pool.py  --> data/distilled/v2c/pool/pool-NNNNN.jsonl (+ pool.manifest.json, eval_index.json, eval_docs.jsonl)
        |  generate_teacher.py   --> data/distilled/v2c/gen/gen-NNNNN.jsonl (+ done.idx, topk/*.npz opcional, gen_summary.json)
        |  filter_teacher.py     --> data/distilled/v2c/{accepted,rejected}.jsonl + filter_report.json + reports/distill/filter-report-v2c.md
```

## Passo a passo

Pre-requisitos: MongoDB do WSL no ar (`wsl -d Ubuntu -u root -e systemctl start mongod`; manter `wsl -d Ubuntu -e sleep infinity` em background durante o filtro).
Python: pool/filtro em `env\.venv\Scripts\python.exe` (tem pymongo; `KAPI_TOKENIZERS_SITE` ja e definido pelos scripts), geracao no venv ROCm
(`C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe`, torch/peft/bitsandbytes).

1. **Pool** (CPU, ~2-5 min no piloto; ~10 min + ~70 s de snippets para 600k):
   `env\.venv\Scripts\python.exe -I scripts\distill\build_prompt_pool.py` (padrao agora: 300k = 200k contexto natural + 100k re-renderizados; `--report reports\distill\pool-v2c-300k.md`).
   Piloto: `--target 1000 --out-dir data\distilled\v2c-pilot --eval-cache data\distilled\v2c`.
2. **Geracao** (GPU, uma por vez; fila: job `J12-distill-generate`):
   `C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe -I scripts\distill\generate_teacher.py --resume [--save-topk] [--samples 1]`.
   Backend alternativo (GGUF + llama.cpp Vulkan): `--backend llama-server --gguf q8_0 --slots 8` (sem top-k).
   `--model merged` usa o merge bf16 (`Repos/KapiTeacher-Mongo-7B/merged/sft_v2d-bf16`) carregado em NF4.
3. **Filtro** (CPU + Mongo somente leitura; job `J13-distill-filter`, depende do J12):
   `env\.venv\Scripts\python.exe -I scripts\distill\filter_teacher.py`.
4. Testes: `cd scripts\distill; ..\..\env\.venv\Scripts\python.exe -m pytest tests -q`.

## Garantias

- **G2.7**: o pool so le `split == train`; ids, grupos, schemas, snippets, `prompt_sha256` e pares prefixo/sufixo de val/test/bench sao
  indexados e qualquer coincidencia descarta o registro (contagem no `pool.manifest.json`, esperado 0 violacoes); dominios holdout nunca; blind via `blind_guard`
  (manifesto lido uma vez; hoje `not_sealed`). O re-render usa so snippets de treino.
- **Resposta de referencia**: vai no campo `gold`; o professor so recebe prefix/suffix/editor (teste `test_generate`).
- **Retomada/determinismo**: unidade = bloco; `done.idx` e o log de commit; shards sao truncados ao ultimo tamanho confirmado; seed por item = blake2b(seed:id:k);
  lotes dependem so do conteudo do bloco.
- **Zero escritas**: escrita/DDL/Atlas nunca vao ao driver; o guard de comandos do pymongo reporta `write_commands_seen_by_guard` (esperado `[]`).
- **Decontaminacao**: `decontaminate.run` com `ngram_min_content: 2` (lido de `configs/dataset.yaml`, assert contra o yaml da destilacao) + 2a passada de verificacao.

## Formatos

- `gen-*.jsonl`: `idx,id,k,mode,seed,raw,pred,stop,gen_tokens,kept_tokens,prompt_tokens,logprob_mean,logprob_min[,topk_pos]`.
- `topk/t-BBBBBB.npz` (opcional): `cand_idx,cand_k,n_pos,topk_ids(uint32[P,k]),topk_logprobs(fp16[P,k]),tail_mass(fp16[P])` — `topk-v1` da Fase 02 §3.5, k=32 (194 B/posicao), so posicoes da resposta (cortadas no mesmo ponto que `pred`).
- `accepted.jsonl`: registro do pool + `middle=target=pred`, `label_source: teacher_seq`, `agrees_with_gold`, `gold_exact`, `plan_equal_gold`, `gold_executable`,
  `weight` (1,0 para todos; so informativo), `exec_kind` (`live|skipped_write|skipped_static`), `teacher{mode,k,seed,logprob_*,pool_idx,adapter_sha256}`.
- `rejected.jsonl`: uma linha por candidata nao aceita (`stage`, `reason`, `pred`, `gold`).

> Nota: as secoes abaixo (piloto) foram medidas com o rank antigo baseado no gold e pool de 600k; valem como historico de vazao/qualidade.

## Piloto de 1.000 prompts (2026-10-16; 750 contexto natural + 250 re-renderizados; `data/distilled/v2c-pilot/`)

Composicao: context_kind full 79,7% / irrelevant 11,2% / none 9,1%; categorias aggregation 29,5, crud 16,8, dsl 14,3, atlas_search 13,5, json_ejson 11,0,
operators 10,7, indexes 4,2; gold executavel (`ok`) 73,6% (o resto e escrita/Atlas, so estatico). Isolamento: 3 prompts descartados por prefixo+sufixo
identico a avaliacao; 0 violacoes no pool final.

| backend | slots | vazao (greedy + 1 amostra T=0,2) | greedy so | 600k prompts |
|---|---:|---:|---:|---:|
| HF NF4 + adapter (lotes por comprimento, top-k ligado) | lote 32 | **0,73 ex/s** (2.630 ex/h; 21 tok/s de decodificacao) | ~1,3 ex/s (est.) | ~228 h (greedy+amostra) |
| GGUF Q8_0, llama-server Vulkan | 8 | **2,25 ex/s** (8,1k ex/h) | **4,3 ex/s (15,5k ex/h)** | 74 h / **~39 h** |
| GGUF Q8_0 | 16 | 2,2 ex/s (sem ganho) | - | - |
| GGUF Q5_K_M | 16 | 1,6 ex/s | - | ~104 h (pior) |

O HF e dominado pela decodificacao (cauda de 96 passos por lote, ~0,3 s/passo com LoRA nao fundida sobre NF4); prefill 262 tokens/prompt em media e barato.
Q8_0 e a opcao recomendada para sequence-level (J12). A amostra extra custa ~2x porque o llama.cpp nao compartilha o prefill entre requisicoes: prefira
greedy no volume todo e, depois, uma 2a passada so nos prompts em que a greedy foi rejeitada (~5,5%).

**Q8_0 x HF NF4 (mesmos 1.000 prompts)**: a predicao greedy e IDENTICA em apenas 56,8% dos prompts (Q5_K_M 58,6%, 500 prompts); as divergencias sao quase todas
posicoes de alta entropia (valores literais, escolha de campo, ordem de estagio), com logprob medio parecido; a qualidade e equivalente:

| metrica | HF NF4 | Q8_0 |
|---|---:|---:|
| greedy valida (todas as etapas 1-7) | 94,5% | 94,8% |
| prompts aceitos (apos dedup+decontam) | 91,9% | 92,1% |
| concorda com o gold (executavel) | 41,7% | 40,6% |
| aceitas que divergem do gold | 52,2% | 52,3% |
| decontam removeu | 31 | 39 |

Filtro (HF, 1.000 prompts): greedy valida em 94,5% (full 97,1%, none 86,8%, irrelevant 82,1%); campo inexistente na greedy: full 2,0%, none 7,7%, irrelevant 11,6%
(coerente com o G2.2); rejeicoes principais: campo inexistente, resposta incompleta, decontam (3,4%: MinHash e 13-grama). Igual ao gold em 40,4% dos prompts
com gold executavel e 52% das aceitas divergem do gold (validas): o professor e uma *variacao validada*, nao uma copia do gold. Escritas vistas pelo guard: 0.
Projecao para 600k: ~551k aceitos (alvo T1 230k / T0 160k = folga de 2,4x; da para reduzir o pool para ~300k sem perder o alvo).

## Custos de disco (600k prompts)

- pool: ~1,05 GB; gen (greedy): ~0,2 GB (com amostra ~0,4 GB); accepted ~1,3 GB; rejected ~0,15-0,3 GB; `eval_docs.jsonl` 68 MB.
- **top-k (k=32, uint32 + fp16 + fp16 = 194 B/posicao)**: medido 0,0052 GB por 1k exemplos com 2 candidatas (5,6 MB/1.000 prompts; ~14 posicoes/candidata)
  => **~3,1 GB para 600k** (1,6 GB so greedy; ~0,7 GB se so os ~230k aceitos do T1 forem mantidos). So via HF (`--backend hf --save-topk`), ~228 h para o pool todo:
  nao compensa antes do piloto A x B decidir se o KD vale; alternativa barata: rodar o HF com `--save-topk` so sobre o subconjunto T0/T1 ja aceito (~230k prompts, ~48 h).

## Riscos

1. Vazao do HF (228 h) torna o top-k do pool completo inviavel; KD exige HF e um recorte (T0/T1 aceitos).
2. Greedy do GGUF difere do HF em ~43% dos prompts (entropia alta); aceitavel para sequence-level, mas o top-k e a logprob de um backend nao valem para o outro.
3. `agrees_with_gold` so em ~42%: a distribuicao do aluno se afasta do gold sintetico; o rank (`exact_gold` primeiro) usa o gold para escolher entre candidatas
   validas (como na Fase 02 §3.5) — se preferirem rotulo cego, trocar `filter.rank` por `[logprob_mean]`.
4. Campo inexistente continua 8-12% sem schema visivel: o filtro os remove, logo `none/irrelevant` ficam sub-representados nos aceitos (acc 82-84% x 94% em full).
5. O filtro depende do MongoDB do WSL (parou entre sessoes antes) e de ~4 GB de RAM no decontam por registro.
6. GPU compartilhada com o desktop (~3,6 GB de outro processo): lotes HF grandes (24k tokens) estouraram a VRAM e ficaram 10x mais lentos; o piloto usou 8k tokens (pico 8,4 GB).
7. Nao fundir adapter em NF4; o `merged` bf16 (NF4 na carga) deve acelerar o HF ~20-30% (nao medido: ainda nao existe no disco).

## Decisoes do usuario (2026-10-17) e estado atual

1. **Pool de 300k** prompts de treino (200k contexto natural + 100k re-renderizados full 50% / irrelevant 25% / none 25%): construido em
   `data/distilled/v2c/pool` (`reports/distill/pool-v2c-300k.md`): none+irrelevant = 86.247 (28,7%; none 40.837, irrelevant 45.410); cobertura por cut_kind/dominio/cursor_role
   dentro de 0,27 p.p. do treino; 0 violacoes G2.7 (1.882 candidatos descartados por prefixo+sufixo/prompt iguais a avaliacao); blind `not_sealed`; decontam `ngram_min_content: 2`.
2. **Geracao so greedy em GGUF Q8_0** (llama-server): ~4,3 ex/s => ~20 h (J12).
3. **Ranking SEM gold**: `filter.rank: [stages_passed]` (mais etapas do validador vencidas; empate => greedy). O gold e so medicao/auditoria; `agrees_with_gold`,
   `gold_exact` e `plan_equal_gold` sao **apenas informativos** e o `weight` e 1,0 para todos (nao ha mais 0,85). A selecao so ve candidatas validas, entao o rank so decide entre greedy e amostra.
4. **2a passada (J12b)**: `generate_teacher.py --backend llama-server --retry-from data/distilled/v2c --samples 1` gera so amostras (T=0,2, seed por item) dos prompts cuja greedy foi
   rejeitada em etapa de validacao e sem aceita (~5,5%, ~16k prompts, ~1 h); depois o J13b refaz o filtro com `--gen gen gen-retry` em `data/distilled/v2c-final`. Opcional.
5. **KD por logits adiado**: `--save-topk` segue implementado (so backend HF), nao usado.
6. **Chat curto fora** (Q1 pendente).

## Fila (nenhum iniciado)

| job | status | horas | depende de |
|---|---|---:|---|
| J12-distill-generate (Q8_0 greedy, 300k) | ready | ~20 (conservador 28) | pool pronto |
| J13-distill-filter (passada 1) | waiting | ~1,5 | J12 |
| J12b-distill-retry (amostras dos rejeitados) | waiting | ~1,2 | J13 |
| J13b-distill-filter-final | waiting | ~1,5 | J12b |

Total ~24 h (22 h sem a 2a passada). Disco: pool ~0,5 GB, gen ~0,1 GB, accepted ~0,7 GB.
