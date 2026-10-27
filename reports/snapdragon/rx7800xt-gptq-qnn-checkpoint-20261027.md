# Nano GPTQ e preparação QNN em 27/10/2026

Experimento isolado; sem alteração do produto, dos snapshots publicados ou dos ambientes originais. Nenhum peso neste commit. Nano CPU int4 continua padrão da IDE.

## GPTQ concluído

RX 7800 XT confirmada como único dispositivo visível, job serializado pelo lock compartilhado. GPTQModel 4.2.0, Torch ROCm 2.10.0, Transformers 4.57.6, Olive 0.13.0; backend Torch, Hadamard com FWHT Torch local ao processo. 29 blocos de 2048 tokens (59.392 tokens efetivos, originados de 256 prompts), 328,016 s no passe GPTQ, 333,203 s totais; pico alocado 4.561.904.640 bytes. Artefato validado: 798 tensores, sem valores flutuantes não finitos, mesmos IDs de tokenizer nos 456 prompts fixados. Bytes serializados do tokenizer diferem.

Avaliação de referência dos 200 itens: APT 2,565 e acerto exato 40,5%, contra BF16 2,585 e 40,0%. Delta APT -0,020; acerto +0,5 ponto percentual. Não representa execução ONNX/QNN nem qualificação em dados reais. Modo quality-only: latências e throughput removidos porque a preparação CPU podia ocorrer em paralelo. As latências do baseline antigo usam runner v1 e não devem ser comparadas diretamente com o runner posterior.

Recarga exigiu device_map explícito cuda:0 (ROCm) e Torch eager; falhas anteriores de seleção rocm e Triton ausente foram preservadas. Não houve alteração de pesos para a recarga.

## Preparação QNN em andamento

Ambiente CPU separado: ORT 1.31.1, plugin oficial QNN 2.7.0 realmente registrado, GenAI 0.18.0, Transformers 5.3.0, Optimum 2.1.0, ONNX 1.23.2. GPU oculta. O plugin expõe QNN CPU neste host; isso não demonstra HTP/NPU. A IDE permanece GenAI 0.15.2.

A ponte process-local Olive0.13/GenAI0.18 usa parse_extra_options oficial e loaders nativos; não altera arquivos instalados. ModelBuilder, conversão MatMulNBitsToQDQ e cirurgias concluíram. A tentativa v5 parou antes da calibração porque kv_cache_dim permaneceu simbólico. A v6 fixa somente esse símbolo em 64, comprovado pela configuração Nano (896/14); sequência passada permanece dinâmica.

A v6 chegou à coleta efetiva de entradas em CPU, com input_ids [1,2048] e KV [1,2,0,64]. Helper QNN get_qnn_qdq_config observado: ativações QUInt16, pesos QUInt8, 24 overrides QUInt16 e CPUExecutionProvider explícito. A execução está em andamento: não declarar calibração, StaticLLM, compilação HTP ou qualidade NPU concluídas. A receita v6 está anexada com caminhos locais para auditoria.

Uma cópia derivada do GPTQ altera apenas max_position_embeddings para 2048; origem e pesos preservados. Isso ainda requer inspeção dos shapes estáticos/configuração final e geração para demonstrar capacidade real. search.max_length isoladamente não é prova. Falhas de load_kwargs nulo, API/layout GenAI e metadados KV estão preservadas, sem transformá-las em aprovação.

Ainda pendem artefatos estáticos, validação, transferência e compilação no Snapdragon, equivalência/qualidade, seleção efetiva da NPU e medições manuais. Nenhum benchmark em CI e nenhuma modificação em TDR, DXGI ou preferências gráficas.

## Verificação dos scripts

54 testes passaram e 1 foi ignorado (inspeção opcional de wheel Olive ausente) no ambiente científico isolado, com GPU oculta. A tentativa inicial no Python mínimo não coletou a suíte por falta de NumPy; nenhum pacote foi instalado nele.

Inspeção adicional do código oficial GenAI 0.15.2: create_model também depende de hf_details/parse_extra_options; o módulo quantized_model antigo existe, mas o patch amplo Olive precisa de validação independente para GPTQ misto. Não há evidência de que trocar o builder resolva o símbolo KV do leitor Olive. A v6 atual prossegue sem substituir ambientes. Fontes: https://github.com/microsoft/onnxruntime-genai/blob/v0.15.2/src/python/py/models/builder.py e https://github.com/microsoft/onnxruntime-genai/blob/v0.15.2/src/python/py/models/quantized_model.py .

## Contrato de transferência e execução previsto

Auditoria do StaticLLM Olive0.13 confirmou: context_length64 é o bloco de prefill; iterator usa sequência1 e model.context_length permanece2048. O grafo anterior à divisão tem24GQA, RoPEcos/sin[2048,32], KVfloat32[B,2,past/total,64] e split_assignments coerentes. A saída final ainda precisa ser inspecionada.

StaticLLM gera context.onnx e iterator.onnx compartilhando external_data do transformer; transformer.onnx pode ser apagado enquanto transformer.onnx.data continua necessário. O pacote deve incluir TODOS os arquivos referenciados por locations, não apenas dados com nome igual ao grafo.

O EPContextBinaryGenerator do Olive compila os grupos context/iterator em HTP e preserva embeddings/lm_head fora desses grupos. O desenho é híbrido: embeddings Gather4bit e lm_head MatMulNBits8bit podem executar em CPU. Desabilitar fallback nas sessões compiladas do transformer não prova todo o LLM na NPU. O compilador ARM deve receber somente os grafos dos grupos context/iterator, preservando os componentes CPU e registrando profiling por componente.

## Preparação concluída e transferência experimental

A v6 completou 256 entradas em CPU (1053,44 s de iteração), OnnxStaticQuantization em 1061,843 s, Split em 1,078 s e StaticLLM em 0,969 s. O wrapper falhou depois de emitir os artefatos porque Olive remove output_dir do dicionário. A falha foi preservada; o wrapper agora captura o caminho antes da chamada. Nenhuma repetição da calibração foi necessária.

O source mantém hardlinks e 25 referências antigas ao cache; o checker ONNX 1.23 recusou links múltiplos por segurança. A proteção foi mantida: o pacote foi copiado para um diretório novo, com recursos próprios e Composite relativo. Os quatro grafos da cópia passaram full_check, com floats finitos, escalas Q/DQ positivas, nenhuma referência fora do pacote e paridade do tokenizer em 456 prompts. Escalas assinadas dos pesos GatherBlockQuantized foram quantificadas separadamente das escalas de ativação.

Controle CPU com GenAI 0.18: o prompt de 361 tokens recebe padding até 384. O orçamento de 393 produzia apenas 9 tokens novos; com orçamento de 416, produziu 32 tokens plausíveis, começando pelo ID 1666. O runner agora reserva espaço após padding, registra padded_prompt_tokens/max_length_requested e recusa estouro sem truncamento. Esse controle não substitui a avaliação dos 200 casos ou a medição na NPU.

O pacote v7 contém 30 arquivos mais export_manifest.json. LICENSE, NOTICE e REPLAY_ATTRIBUTION foram verificadas contra o snapshot fixado. Publicação: https://huggingface.co/esilva/KapiCoder-Mongo-Nano-ONNX/tree/39a55f1c17411d8d2cfa0167095bff5250f7c98e/qnn-prepared . Branch experimental/snapdragon-qnn-rx7800xt-20261027. O download público dos 31 arquivos foi concluído e todos os hashes conferiram. O manifesto remoto também corresponde ao local. A publicação é preparação W4A16; a compilação de EPContext HTP ainda depende do Snapdragon. O catálogo e o padrão Nano CPU int4 permanecem como estavam.

Antes de compilar no ARM, verificar o manifesto e preservar pipeline/model_config relativo. make_qnn_recipe.rebase_composite resolve os recursos para o diretório baixado. O compilador direto deve receber context.onnx e iterator.onnx com compartilhamento de pesos; embeddings/head permanecem componentes CPU. Ainda pendem geração de EPContext HTP, profiling QNN, avaliação dos 200 itens na NPU, limites próximos da capacidade, cancelamento e paridade. Nenhuma inferência HTP foi realizada neste host.

Os relatórios dentro de audit registram os estados observados durante a preparação; o relatório estrutural aprovado e o manifesto de transferência são as evidências finais dessa cópia. A avaliação de qualidade publicada é a referência GPTQ, não uma avaliação do pipeline ONNX na NPU.

Suíte dos scripts: 62 testes passaram e um foi ignorado por ausência do wheel opcional de inspeção. Benchmarks permanecem manuais. Os repositórios originais IDE b2164ab e MongoIA d01e04b estavam limpos na conferência.

## Derivação de janela 32 e limite 2016+32

A janela 64 exige padding de 2016 para 2048 antes de gerar e, portanto, não preserva o contrato lógico de 2016 tokens de entrada mais 32 de saída. Uma nova derivação executou apenas StaticLLM sobre uma cópia do SplitModel existente, com janela 32, sem repetir GPTQ ou calibração. Saída e cache novos preservam o pacote v7 e o source anterior.

Os hashes de context.onnx.data, embeddings.onnx e lm_head.onnx são idênticos aos da janela 64. A auditoria parametrizada --expected-window 32 aprovou quatro grafos, interfaces context[1,32,896]/iterator[1,1,896], contexto configurado 2048, KV2x64 e paridade dos 456 prompts.

O controle curto gerou os mesmos 32 IDs da janela 64. Um controle sintético separado, fora do corpus de qualidade, recebeu exatamente 2016 tokens e gerou 32 novos até sequência 2048, sem padding adicional ou truncamento. Não é qualificação na NPU. Suíte atual:71 testes passaram e um foi ignorado.

Pacote local: C:/IA/models/KapiSnapdragonArtifacts/nano-qnn-window32-transfer-v1. O manifesto e os 27 itens foram verificados. Plano exato em nano-qnn-window32-publication-plan-rx7800xt.json. A tentativa de publicar essa derivação foi rejeitada pela revisão automática por exigir autorização específica para o payload experimental; nenhum upload da janela 32 ocorreu. Publicação e verificação pública aguardam aprovação humana. O pacote anterior permanece publicado em 39a55f1c17411d8d2cfa0167095bff5250f7c98e.

## Janela 32 com tokenizer original: pacote local v2

O empacotador agora oferece --preserve-tokenizer-source, mantendo compatibilidade com os experimentos anteriores. Quando habilitado, verifica e copia byte a byte tokenizer.json, tokenizer_config.json e os auxiliares presentes no snapshot fixado. No Nano c03, merges.txt e vocab.json também foram preservados. Os quatro arquivos conferem por tamanho e SHA-256. Arquivos originais de modelo/configuração de geração não são copiados sobre a configuração derivada.

O inspetor oferece --require-original-tokenizer, mantendo os 456 checks de IDs e exigindo hashes iguais dos arquivos originais. O pacote v2 passou no gate estrito, nos quatro checkers ONNX e nas interfaces de janela 32. config.json.max_position_embeddings, genai.model.context_length e search.max_length permanecem 2048. O model_max_length32768 do tokenizer original não substitui a capacidade do runtime. O template extraído corresponde ao texto do original.

Controles novos na CPU: 32 IDs idênticos no caso curto; o caso sintético com 2016 tokens gerou os mesmos 32 IDs do controle anterior e terminou em 2048. Nenhum novo GPTQ ou lote de calibração foi executado. Pesos e pacotes anteriores foram preservados. Suíte:80 testes passaram e um foi ignorado.

Pacote final local para aprovação: C:/IA/models/KapiSnapdragonArtifacts/nano-qnn-window32-transfer-v2.31 arquivos,436014016bytes; manifesto a5f13fa94f3ef449c79bbb3c59bf725728d21b31d14c961b5ac160bfd9bc1b72. O plano nano-qnn-window32-original-tokenizer-publication-plan-rx7800xt.json substitui o plano v1. Nenhum upload desse pacote foi tentado; a publicação permanece bloqueada pela revisão automática anterior, aguardando autorização específica do payload experimental atualizado. Compilação e homologação HTP seguem pendentes.
