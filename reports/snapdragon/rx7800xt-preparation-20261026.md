# Preparação no Esilva-Pc com RX 7800 XT — 26/10/2026

Execução autorizada pelo usuário neste chat. Checkout isolado `C:/IA/models/KapibaraStudio.MongoIA-snapdragon-rx7800xt`, branch local `codex/snapdragon-rx7800xt`; master original e IDE preservados. Os baselines consumiram o runner e4542ea; depois dos processos encerrados, o checkout recebeu f52ee70 e seus 37 testes passaram. Nenhum treino, consulta MongoDB, quantização completa ou publicação de pesos foi executado.

## Identidade e seleção real

Esilva-Pc, RAM física 67.815.215.104 bytes. Windows lista RX 7800 XT e iGPU AMD Radeon(TM) Graphics. PyTorch ROCm 2.10.0+rocm7.14.0a20260611, HIP 7.14.60850, enumera RX no índice0 (17.163.091.968 bytes de VRAM) e iGPU no índice1. O job identifica a RX pelo nome, aplica HIP_VISIBLE_DEVICES=0 em processo novo, exige exatamente uma GPU visível e executa com `cuda:0 --expected-gpu RX 7800 XT`. O termo cuda é a API PyTorch/ROCm, não uma GPU NVIDIA. Os relatórios registram nome real, HIP, dtype BF16 e dispositivo efetivo dos tensores. Não houve fallback para iGPU. Jobs foram serializados pelo lock `C:/IA/models/KapibaraStudio.MongoIA/tmp/gpu.lock`, liberado ao terminar.

## Fontes e amostra

Revisões Nano c03bfe15ba1ff4e5b58a4b0953349eca82bb43f1 e Small4ba57bc1f56a5b3cbe486dd314321ce4cdb2cae6. Pesos públicos existentes foram copiados para snapshots próprios em C:/IA/models/KapiPublishedSnapshots;34/34 arquivos conferidos contra árvores pinadas GitSHA/LFS SHA-256 e manifesto local. Foram baixados apenas3038bytes de .gitattributes. Nenhum peso foi baixado ou alterado.

O prepare_samples.py regenerou256calibração e200avaliação, com grupos/schemas separados e até549tokens. SHA256 calibração179736eed969a8a961af6d3fc410a3eead24306234d65ccf245c20cb16978b60; avaliação43b6eb37732209fb89747f309477c651d11fb6eb0d22b9d23a007a90b2793bc8. São exatamente os hashes da amostra do notebook. Código JavaScript gerado não foi executado.

## Baselines BF16 — medidas proxy

| Modelo | Pedidos completos | APT | Match exato aparado | TTFT p50/p95 (ms) | Mediana decode (tokens/s) |
|---|---:|---:|---:|---:|---:|
| Nano |200/200|2,585|40%|20,53 /31,12|62,59|
| Small|200/200|2,550|42%|39,48 /58,24|46,11|

Não há erros nos400pedidos; dtype torch.bfloat16, modelo e inputs em RX7800XT. APT/match usam o comparador fornecido contra as referências sintéticas. Arquivos nano-baseline-rx7800xt.json e small-baseline-rx7800xt.json preservam IDs e saídas para comparação no notebook. O resumo é rx7800xt-baseline-summary.json.

Estas medidas incluem o primeiro pedido, sem exclusão de warmup. Runner v1 captura o fim depois do decode final; o runner v2 posterior alterou essa convenção. Não comparar diretamente a velocidade de decode v1/v2 como se fossem a mesma medida. Isso não homologa IDE, dados reais ou latência alvo; não foi um benchmark de release/CI. O PyTorch emitiu aviso de caminho SDPA/AOTriton experimental no AMD; nenhuma variável experimental foi ativada para esconder ou modificar esse comportamento.

## GPTQ/Hadamard e alternativa CPU

Ambiente isolado env/.venv-snapdragon-rx-probe lê os pacotes do venv ROCm original por .pth explícito e instala somente as dependências próprias. Torch/Transformers originais permaneceram2.10.0+rocm7.14.0a20260611/4.57.6. GPTQModel4.2.0, Olive0.13.0, pip check passou. Fonte GPTQ SHA256 0dff183893268e15cd683a30e08acaf0b03073d55aa44f7076380ee87094f0ae. O pacote foi construído sem extensões CUDA (BUILD_CUDA_EXT=0); isso não comprova suporte a kernels GPTQ nativos em Windows/ROCm. Versões efetivas: rx7800xt-probe-environment.json.

A sonda aleatória de uma camada, quatro prompts de64tokens, falhou primeiro por UnicodeEncodeError do logbar/cp1252 (gptq-rx7800xt-probe.json). A repetição com PYTHONUTF8=1/PYTHONIOENCODING=utf-8 passou desse ponto e falhou em4,843s com ModuleNotFoundError: fast_hadamard_transform; pico GPU188.892.160bytes, apenas RX visível (gptq-rx7800xt-probe-utf8.json). Não é uma prova de impossibilidade de portar o kernel; é a falha efetivamente observada neste ambiente.

A documentação/fonte oficial do Fast Hadamard Transform não forneceu uma rota Windows/ROCm comprovada nesta revisão: [instalador](https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/setup.py) e [distribuições PyPI](https://pypi.org/project/fast-hadamard-transform/#files). Não foi feita instalação cega de CUDA para AMD. No GPTQModel a rotação chama .cuda(); apenas configurar device=cpu não equivale a um pipeline CPU. [Fonte4.2.0](https://github.com/ModelCloud/GPTQModel/blob/v4.2.0/gptqmodel/quantization/rotation/hadamard_utils.py).

A alternativa limitada scripts/snapdragon/probe_gptq_cpu_alternative.py passou em4,39s com quatro prompts de64tokens, modelo aleatório, rotação=None, lm_head=False, backend Torch, cache GPU desabilitado e todas as saídas/tensores em CPU. HIP_VISIBLE_DEVICES=-1 foi aplicado antes do import; torch confirmou zero GPUs. É uma receita diferente: não comprova Hadamard/W4A16 QNN, qualidade ou custo de Nano/Small completos. Não extrapolar esse tempo para um job completo.

WSL2 Ubuntu existe e estava Running. O Python padrão /usr/bin/python3 não encontrou Torch; nenhuma instalação foi feita, e outros venvs não foram pesquisados exaustivamente. Relatório wsl-rx7800xt-inventory.json. Preparação estática W4A16 e compilação final no notebook permanecem pendentes; não há artefato QNN funcional para publicar.

## Seguimento experimental

A falta da extensão original foi contornada somente em microteste isolado com FWHT puroPyTorch, referência densa CPU/RX e backendTorch. Resultado e limitações: rx7800xt-fwht-followup-20261026.md. Isso não constitui quantização completa ou pacoteQNN.
