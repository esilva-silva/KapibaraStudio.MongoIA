# Dataset sintético contextual Pro — instruções e avaliação

Lote complementar para o candidato KapiTeacher-Mongo-7B: todos os nomes, dados, coleções e esquemas são sintéticos. Os dados anteriores não são alterados. Arquivos gerados em `data/generated` são ignorados pelo Git; guarde os três manifestos e os scripts ao transportar o lote.

## Geração e auditoria

Em diretórios vazios:

```powershell
python scripts/data/pro/build.py
python scripts/data/pro/build_mid.py
python scripts/data/pro/build_mid.py --expanded
python scripts/data/pro/fix_chat_context.py
node scripts/data/pro/validate.js data/generated/pro-context-v1b data/generated/pro-chat-context-v2/chat-train.jsonl
node scripts/data/pro/validate_mid.js data/generated/pro-mid-script-v4 data/generated/pro-mid-script-v4/verified-plans.jsonl
node scripts/data/pro/validate_mid.js data/generated/pro-mid-script-extra-v4 data/generated/pro-mid-script-extra-v4/verified-plans.jsonl --expanded
python scripts/data/pro/audit.py --train-chat data/generated/pro-chat-context-v2/chat-train.jsonl
python scripts/data/pro/prepare_teacher.py --out data/generated/pro-teacher-input-v5 --train-chat data/generated/pro-chat-context-v2/chat-train.jsonl
python scripts/data/pro/verify_teacher_staging.py --root data/generated/pro-teacher-input-v5
python scripts/data/pro/validate_tokenizer.py --tokenizer CAMINHO\tokenizer.json --source data/generated/pro-teacher-input-v5/source
```

Os geradores recusam sobrescrever arquivos; o verificador de mid-script também recusa sobrescrever planos. Os manifestos incluem SHA-256 e contagem. Para repetir em outro local use `--out` e compare os hashes. O arquivo `verified-plans.jsonl` é derivado, não faz parte do corpus de treino, e deve ser gerado novamente pelo verificador antes da validação no servidor.

O resultado revisado contém 41.000 exemplos: 32.800 treino, 4.100 validação e 4.100 teste; 33.500 FIM e 7.500 chat. Use o arquivo `pro-chat-context-v2/chat-train.jsonl` no lugar do chat original de treino: o contexto original apresentou colisões pela regra oficial de 13-gramas após correção do detector. São três esquemas sintéticos, um por divisão. O conjunto complementar contém 11.000 FIM de conclusão no meio de script (8.800/1.100/1.100). As 20 famílias incluem filtros, projeções, agregação, inserção e atualização; os cortes exploram aspas, vírgulas, objetos, arrays e sufixos existentes. Três famílias de mid-script ficam reservadas exclusivamente à avaliação no lote expandido. As demais famílias são compartilhadas; valores diferentes não transformam exemplos em tarefas independentes. Avalie também com cenários independentes e conjunto cego.

O FIM usa o renderer `editor-context-v1` e Qwen do repositório. Comentários são dados de entrada da tarefa; não são instruções para o avaliador. Chat segue a estrutura do renderer do professor e as explicações são comentários JavaScript. `audit.py` confere hashes, IDs, grupos e schemas, duplicatas exatas dentro de cada split e renderização de todos os prompts FIM. `validate.js` e `validate_mid.js` recompõem os scripts, verificam sintaxe e conferem os argumentos contra expectativas independentes. O último emite planos capturados que são executados pelo verificador MongoDB.

## MongoDB

Os scripts `validate_mongo.js` e `validate_mid_mongo.js` criam bancos novos com prefixo `kapi_fx_pro_`, recusam nomes já existentes e não acessam dados de aplicações. `validate_mid_mongo.js` apaga e reenvia apenas documentos da coleção de fixture do banco novo; executa os planos capturados, incluindo as escritas de exemplo. Ele nunca executa os comentários de usuário como instruções para o avaliador. Bancos criados ficam disponíveis para inspeção; não execute sobre um banco de dados existente.

Para repetir a validação no Ubuntu/WSL, use caminhos Linux para os JSONL e um identificador novo com 12 a 40 letras minúsculas/dígitos:

```bash
export KAPI_PRO_RUN=umidentificadornovo123
export KAPI_PRO_DIR=/caminho/para/data/generated/pro-context-v1b
mongosh --quiet /caminho/para/scripts/data/pro/validate_mongo.js
export KAPI_PRO_RUN=outroidentificadornovo123
export KAPI_PRO_PLAN_FILES=/caminho/pro-mid-script-v4/verified-plans.jsonl:/caminho/pro-mid-script-extra-v4/verified-plans.jsonl
mongosh --quiet /caminho/para/scripts/data/pro/validate_mid_mongo.js
```

A validação semântica dos 30 mil originais executa as consultas completas. Para os 11 mil de mid-script, o Node executa o script reconstruído em uma API simulada, compara as chamadas esperadas e captura os argumentos; o MongoDB executa esses planos e verifica resultados e efeitos nas fixtures. Isto não é uma execução direta do script reconstruído no interpretador da IDE.

A preparação cria shards no padrão de nome e listas de IDs aceitos pelo indexador do professor; acrescenta `dialect: console` nos registros FIM preparados, exigido para `val_gen`. O indexador real foi executado sem GPU e conferiu contagens, papéis e isolamento. `validate_tokenizer.py` aceita somente o tokenizer com o hash fixado no lock; mede limites nos 41 mil registros e exercita o renderizador real do professor em amostras de treino e validação. Isso é apenas staging e não é um corpus tokenizado ou aprovado. A auditoria de descontaminação do relatório cobre os 41 mil exemplos e os JSONL legados locais, com a regra oficial de conteúdo. O conjunto cego não estava selado e os outros corpora validados não estavam presentes neste checkout. Não implica ausência de colisões contra dados indisponíveis. Antes do treino: executar guarda blind quando existir, descontaminar contra todos os corpora de avaliação preservados, verificar modelos fixados e executar paridade da IDE; completar a integração ao pipeline oficial e realizar revisão humana dos exemplos.
