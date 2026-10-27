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
