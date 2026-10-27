# Small W4A16: preparação local em 27/10/2026

Snapshot fixado em 4ba57bc1f56a5b3cbe486dd314321ce4cdb2cae6, com arquivos e 456 IDs auditados. Foram mantidos os 256 prompts de calibração e 200 de avaliação do Nano, com os mesmos hashes. Trabalho isolado no Esilva-Pc; IDE, Nano e snapshots originais preservados. Sem novo treinamento, download de pesos ou publicação.

GPTQ na RX 7800 XT com verificação do nome, dispositivo único e lock compartilhado: 861,031 s totais, 854,297 s no passe GPTQ, pico alocado pelo Torch de 6.688.331.264 bytes. A estimativa anterior de 14.254.388.936 bytes era extrapolação pelo tamanho dos pesos e superestimou o valor medido. Foram 85 chamadas FWHT, incluindo o controle inicial. Os 256 prompts produziram 29 blocos efetivos de 2048 tokens (59.392 tokens).

Checkpoint aprovado: 930 tensores, floats finitos, hashes, shards e caminhos verificados e 456 IDs do tokenizer iguais. A cópia para contexto 2048 modifica somente a configuração derivada; pesos e arquivos originais conferiram antes e depois.

Referência GPTQ dos 200 casos: acerto exato de 40% contra 42% no BF16 (delta de -2 pontos percentuais); APT de 2,525 contra 2,55 (delta de -0,025). A comparação inclui mudança de precisão e quantização, sem isolar causalidade. Não é aprovação de qualidade. A execução paralela à preparação CPU usou quality-only, sem TTFT ou decode para qualificação. JavaScript gerado não foi executado e o conjunto é sintético.

A preparação CPU completou ModelBuilder, conversão QDQ, cirurgias e fixação de kv_cache_dim=128. Está calibrando as ativações com o helper QNN real, CPUExecutionProvider e 28 overrides QUInt16. O Small tem 28 camadas, hidden 1536, 12 heads e KV 2×128. Janela StaticLLM de 32 e contexto 2048 estão previstos. O contador no checkpoint JSON é uma observação parcial; calibração e pacote final ainda não concluíram.

Ainda pendem calibração dos 256 prompts, Split/StaticLLM, controles CPU de 32 tokens e de 2016+32, auditoria de grafos e tokenizer original byte a byte, manifesto de pacote e avaliação física HTP. Energia e custo monetário não foram medidos. Nenhum pacote Small será publicado antes de aprovação do artefato concreto.

Utilitários e testes pertinentes: 100 passaram e um foi ignorado. Execução manual de modelos permanece fora de CI.
