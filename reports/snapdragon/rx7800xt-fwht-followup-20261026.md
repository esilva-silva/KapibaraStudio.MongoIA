# Seguimento FWHT isolado — RX 7800 XT

Os baselines BF16 foram compartilhados no commit3568c3e, branch codex/snapdragon-rx7800xt. Este seguimento não altera esses400pedidos nem a IDE; testa uma adaptação experimental após a falha da extensão Hadamard ausente. Nenhum peso Nano/Small foi quantizado ou publicado.

## Prova numérica

O helper torch_fwht.py implementa x @ H.T * scale na última dimensão potência2, sem padding implícito, com acúmuloFP32 e retorno ao dtype original. Fonte e semântica da API: https://github.com/Dao-AILab/fast-hadamard-transform . Não é a implementaçãoCUDA original e não há alegação de desempenho equivalente.

A referência densa foi construída independentemente por paridade popcount emCPU/float64. Passaram72comparações CPU/RX (N16/32/64/128, FP32/FP16/BF16, escalas1/sqrt(N),1,0,25), preservando dtype/device/shape e input. Erro absoluto máximo contra a referência arredondada: {'torch.bfloat16': 0.0, 'torch.float16': 0.0, 'torch.float32': 2.384185791015625e-07}. Relatório fwht-rx7800xt-dense-reference.json. Os testesCPU também cobrem não-contíguo, N1, batchvazio, gradiente, acúmulo de baixa precisão e rejeições. São testes de correção, não benchmark. São opcionais em ambientes sem Torch; o lock de ferramentas comum não ganha uma dependência pesada implícita.

## Microteste GPTQ na RX

probe_gptq_fwht.py roda em subprocesso próprio. Ele registra alias temporário fast_hadamard_transform para o helper, faz binding explícito no namespace hadamard_utils do GPTQModel4.2 e seleciona BACKEND.TORCH. Nenhum arquivo do pacote instalado é modificado. HIP_VISIBLE_DEVICES=0 e os guardas exigem uma única RX7800XT; o enum GPTQ aceita cuda, enquanto o dispositivo atual é fixado emcuda:0 após a verificação por nome.

A primeira tentativa falhou pelo enum literal cuda:0; o erro está preservado em gptq-rx7800xt-fwht-device-enum-error.json. Depois do mapeamento explícito, o microteste passou: quatro prompts64tokens, modeloQwen2 aleatório1camada/hidden64, rotaçãoHadamard, pesos4bits e head8bits, 6.843s, picoGPU474599424bytes. As três chamadas FWHT registraram tensoresFP32 emcuda:0. O tipo quantizado é TorchQuantLinear; o empacotamento final armazena os tensores emCPU, o que não indica uso da iGPU. Relatório gptq-rx7800xt-fwht-probe.json inclui hashes do helper e wrapper. Essa prova pequena não prevê qualidade, RAM/tempo do modelo completo ou validadeQNN.

## Dimensões e próximos gates

A inspeção get_hadK aceita hidden896 doNano (K28/bloco32) e hidden1536 doSmall (K12/bloco128). O helper isolado não aceita intermediate4864 doNano; porém essa chamada intermediária está comentada no rotate_mlp_output doGPTQ4.2, portanto o achado NÃO é prova de bloqueio do caminho ativo. O relatório gptq-rotation-shape-compatibility-rx7800xt.json faz essa distinção. As rotações ativas usam hidden_size e dimensão de head.

Antes de um job completo, ainda é necessário verificar as dimensões reais no fluxo carregado, geração/qualidade após quantização, export estático e compatibilidade ModelBuilder/Olive/runtime. Não foi iniciada quantização longa, nem foi gerado pacoteW4A16/QNN funcional. A compilação final continua pertencendo ao notebook e não deve assumir soc_model60. O testeCPU sem rotação anteriormente medido continua sendo alternativa distinta.
