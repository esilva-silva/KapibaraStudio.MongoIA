# KapiCoder-Mongo-Small (T1)

Autocomplete FIM de MongoDB (e, apos a decisao Q1, chat curto) para o Kapibara Studio, ~1,5B parametros (DML FP16 / CPU INT8). **Em preparacao: treino ainda nao executado.**

| | |
|---|---|
| Base | `Qwen/Qwen2.5-Coder-1.5B` (revisao `df3ce67c0e24480f20468b6ef2894622d69eb73b`, Apache-2.0), em `BaseRepos\Qwen2.5-Coder-1.5B`; ver `model.lock.json` |
| Metodo | SFT sequence-level de autocomplete FIM `editor-context-v1`, LoRA r=64 (alpha 128) mesclado em bf16, 2 epocas; dois bracos pareados: **distilado** (alvo do KapiTeacher-Mongo-7B) e **controle gold** (J10, referencia do G2.4) |
| Dados | `data\student\{distill,gold}\t1` (229.834 exemplos, 63,7 M tokens/epoca; T0 e subconjunto): prompts de treino do dataset v2c (split `train`), alvos do professor validados por execucao somente leitura, replay JS/TS permissivo (12% dos tokens); `reports\student\corpus_report.md` (raiz) |
| Gates | `configs\gates.yaml` tier T1 (Fase 06 §8; so os itens do Python aplicaveis ao autocomplete sao avaliados aqui, o resto fica `unevaluated`); resultados em `reports\eval\<run>\gates.json` quando existirem |
| Job | `README_job.md` (iniciar, monitorar, retomar, avaliar, criterios distilado x controle) |
| Estado | T1 treinado de novo no corpus v3 (`sft_v3`, LoRA r=64, 1,5 epoca, 309.571 exemplos, 15,63 h, uma semente); o v1 (`sft_distill_v1`) fica so como checkpoint interno nao publicado. Avaliado no val v2c sintetico e na homologacao v1 (6.030 cortes de schemas novos, escrita por LLM/gramatica, nao humana e **nao totalmente independente do treino v3**): FIM acc 45,4% contra 18,7% da base; campos inexistentes em schema novo com schema visivel 7,5% contra 7,15% da base (proximo da base). Card e staging de **preview privado** em `hf/` e `cache/hf_stage/` (`esilva/KapiCoder-Mongo-Small`, tag `v0.1.0-preview`); nada publicado |

Contrato de uso (IDE): prompt `QwenFimPromptBuilder` com cabecalho `editor-context-v1` (contexto de 2.048 tokens); tokenizer identico ao do professor. Escopo desta rodada: autocomplete; chat curto fora ate a Q1.

Licenca: Apache-2.0 (`LICENSE`, herdada do Qwen2.5-Coder). Versao **modificada** (ajustada) do modelo base: incluir aviso de modificacao e atribuicao ao publicar (modelo de texto em
`Repos\KapiTeacher-Mongo-7B\merged\sft_v2d-bf16\MODIFICATION_NOTICE.md`). Dados sinteticos sem dados pessoais; replay de pacotes MIT/Apache-2.0/BSD/ISC (`data\replay\v1\LICENSES.md`).
