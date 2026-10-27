# Small W4A16: preparação local em 27/10/2026

Snapshot fixado em 4ba57bc1f56a5b3cbe486dd314321ce4cdb2cae6, com arquivos e 456 IDs auditados. Foram mantidos os 256 prompts de calibração e 200 de avaliação do Nano, com os mesmos hashes. Trabalho isolado no Esilva-Pc; IDE, Nano e snapshots originais preservados. Sem novo treinamento, download de pesos ou publicação.

GPTQ na RX 7800 XT com verificação do nome, dispositivo único e lock compartilhado: 861,031 s totais, 854,297 s no passe GPTQ, pico alocado pelo Torch de 6.688.331.264 bytes. A estimativa anterior de 14.254.388.936 bytes era extrapolação pelo tamanho dos pesos e superestimou o valor medido. Foram 85 chamadas FWHT, incluindo o controle inicial. Os 256 prompts produziram 29 blocos efetivos de 2048 tokens (59.392 tokens).

Checkpoint aprovado: 930 tensores, floats finitos, hashes, shards e caminhos verificados e 456 IDs do tokenizer iguais. A cópia para contexto 2048 modifica somente a configuração derivada; pesos e arquivos originais conferiram antes e depois.

Referência GPTQ dos 200 casos: acerto exato de 40% contra 42% no BF16 (delta de -2 pontos percentuais); APT de 2,525 contra 2,55 (delta de -0,025). A comparação inclui mudança de precisão e quantização, sem isolar causalidade. Não é aprovação de qualidade. A execução paralela à preparação CPU usou quality-only, sem TTFT ou decode para qualificação. JavaScript gerado não foi executado e o conjunto é sintético.

A preparação CPU completou ModelBuilder, conversão QDQ, cirurgias e fixação de kv_cache_dim=128. Está calibrando as ativações com o helper QNN real, CPUExecutionProvider e 28 overrides QUInt16. O Small tem 28 camadas, hidden 1536, 12 heads e KV 2×128. Janela StaticLLM de 32 e contexto 2048 estão previstos. O contador no checkpoint JSON é uma observação parcial; calibração e pacote final ainda não concluíram.

Ainda pendem calibração dos 256 prompts, Split/StaticLLM, controles CPU de 32 tokens e de 2016+32, auditoria de grafos e tokenizer original byte a byte, manifesto de pacote e avaliação física HTP. Energia e custo monetário não foram medidos. Nenhum pacote Small será publicado antes de aprovação do artefato concreto.

Utilitários e testes pertinentes: 100 passaram e um foi ignorado. Execução manual de modelos permanece fora de CI.

## Cobertura efetiva do GPTQ auditada

A seleção continua com 256 prompts e a calibração de ativações é um fluxo separado de 256 entradas. A reprodução do pré-processador instalado revelou que o JOIN do GPTQ conserva 29 blocos de 2048 tokens e descarta a cauda de 1881 tokens: 248 IDs têm todos os tokens representados, um ID tem 177/290 tokens e sete IDs são omitidos. Desses 248, 220 permanecem em um único bloco e 28 atravessam fronteiras; cobertura integral de tokens não significa contexto indiviso.

O histórico registrou contagens/comprimentos, não hashes ou IDs. A atribuição por ID é reconstrução verificável com código e entradas fixados, detalhada em gptq-calibration-token-attribution-rx7800xt.json; não é captura histórica dos tokens. A perda decorre do JOIN, sem truncamento individual, embaralhamento ou filtro posterior dos 29 blocos.

Não há flag suportada no Olive0.13 para conservar a cauda. Uma política futura explícita poderia emitir o bloco final de 1881 tokens (30 lotes, mesmos 256 IDs, sem padding); sua preparação foi validada apenas em CPU. Isso mudaria os pesos e exigiria repetir GPTQ, calibração de ativações e avaliação. Não foi executado e não garante melhora de qualidade. Se o requisito for manter cada prompt indiviso, será necessária outra política de agrupamento.

O job Small atual foi preservado. As amostras de memória mostram RSS próximo de 37 GB e memória privada de 40,7 GB, com margem de commit de 11,8 GB e paginação instantânea zerada nas amostras observadas. Não existe checkpoint configurado dos ranges parciais; interromper perderia a calibração em curso. CalibMaxIntermediateOutputs não foi aplicado: na versão instalada, o caminho inspecionado limpa a coleção antes de computar ao atingir o limite, sem equivalência demonstrada.

## Política de tokens da calibração de ativações

A reprodução do pré-processamento instalado (Olive0.13/Transformers5.3) confirma os 256 prompts raw sem alteração de IDs ou truncamento, mas a receita atual add_special_tokens=True envolve cada linha como `None ` + prompt + ` <|endoftext|>`, porque bos_token é None. São quatro tokens extras úteis por linha:62.041 úteis e462.247 de padding em256lotes de2048. Essa é preservação como subsequência, não paridade estrita da entrada antes do padding. O histórico não capturou hashes dos tokens; os relatórios de tokenização são reconstruções, distintos do trace de entradas realmente processadas.

O parâmetro suportado add_special_tokens=False foi comprovado somente em memória:256/256 linhas com paridade estrita,61.017 tokens úteis e463.271 de padding. Existe proposta local não executada em activation-raw-policy-proposal-rx7800xt.json. Corrigir só esse wrapper exige repetir quantização/calibração de ativações e avaliação, mas não GPTQ. Corrigir a cauda JOIN é outra alteração e exige repetir ambas as fases. Nenhuma causalidade ou melhora de qualidade foi demonstrada. O job e os pacotes atuais foram preservados; essa limitação deve acompanhar a variante experimental.

## Encerramento da revisão legada

O trace encerrou em input_complete com256 entradas; calibração iterada3571,422s, OnnxStaticQuantization3587,094s e preparaçãoCPU total3636,078s. Split eStaticLLM concluíram. Os arquivos permanecem em small-qnn-prepare-rx7800xt-v1/qnn-output-v1, com contexto2048, janela32,28camadas eKV2x128. Esta revisão fica preservada como legado, não como pacote final: JOIN e wrapper de ativações precisam da política corrigida. Não houve publicação ou transferência desses pesos. Nenhum job novo será lançado sem os scripts/pins e preflight da revisão corrigida.
