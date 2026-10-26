# KapiCoder-Mongo-Nano (T0)

Autocomplete FIM de MongoDB (Mongo Console / mongosh / pipelines JSON) para o Kapibara Studio, ~0,5B parametros, pensado para CPU (INT4/INT8). **Em preparacao: treino ainda nao executado.**

| | |
|---|---|
| Base | `Qwen/Qwen2.5-Coder-0.5B` (revisao `8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301`, Apache-2.0), em `BaseRepos\Qwen2.5-Coder-0.5B`; ver `model.lock.json` |
| Metodo | SFT sequence-level de autocomplete FIM `editor-context-v1`, full fine-tuning (mestre fp32 + autocast bf16), 2 epocas; dois bracos pareados: **distilado** (alvo do KapiTeacher-Mongo-7B) e **controle gold** |
| Dados | `data\student\{distill,gold}\t0` (159.865 exemplos, 44,2 M tokens/epoca): prompts de treino do dataset v2c (split `train`), alvos do professor validados por execucao somente leitura, replay de codigo JS/TS permissivo (12% dos tokens); detalhes em `reports\student\corpus_report.md` (raiz) |
| Gates | `configs\gates.yaml` tier T0 (Fase 06 §8); G2.4 do professor em `configs\gates.teacher.yaml`; resultados em `reports\eval\<run>\gates.json` quando existirem |
| Job | `README_job.md` (iniciar, monitorar, retomar, avaliar, criterios distilado x controle) |
| Estado | T0 treinado de novo no corpus v3 (`sft_v3`, full fine-tuning, 238.453 exemplos, 4,38 h, uma semente); o v1 (`sft_distill_v1`) fica so como checkpoint interno nao publicado. Avaliado no val v2c sintetico e na homologacao v1 (6.030 cortes de schemas novos, escrita por LLM/gramatica, nao humana e **nao totalmente independente do treino v3**): FIM acc 42,3% contra 12,6% da base; campos inexistentes em schema novo com schema visivel 10,8% contra 7,0% da base (ainda acima: validar nomes de campo antes de executar). Card e staging de **preview privado** em `hf/` e `cache/hf_stage/` (`esilva/KapiCoder-Mongo-Nano`, tag `v0.1.0-preview`); nada publicado |

Contrato de uso (IDE): prompt `QwenFimPromptBuilder` com cabecalho de contexto do editor (`editor-context-v1`, contexto de 2.048 tokens, ate 32 tokens no autocomplete); tokenizer identico ao do professor e dos demais
Qwen2.5-Coder. Escopo: autocomplete; chat curto fora ate a decisao Q1.

Licenca: Apache-2.0 (`LICENSE`, herdada do Qwen2.5-Coder). Este modelo e uma versao **modificada** (ajustada) do modelo base; aviso de modificacao e atribuicao ao publicar: ver `Repos\KapiTeacher-Mongo-7B\merged\sft_v2d-bf16\MODIFICATION_NOTICE.md`
como modelo do texto. Os dados sinteticos (schemas ficticios) nao contem dados pessoais; o replay vem de pacotes npm MIT/Apache-2.0/BSD/ISC (`data\replay\v1\LICENSES.md`).
