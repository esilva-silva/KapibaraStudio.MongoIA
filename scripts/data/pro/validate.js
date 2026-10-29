// Validação independente da sintaxe e dos argumentos das chamadas geradas.
const fs = require('fs');
const readline = require('readline');
const vm = require('vm');
const path = require('path');

const schemas = {
  train: ['projetos_demo', 'projetoId', 'sistema', 'situacao', 'prioridade'],
  val: ['iniciativas_demo', 'codigoProjeto', 'sistemaOrigem', 'estado', 'nivel'],
  test: ['carteira_demo', 'idProjeto', 'sistemaResponsavel', 'status', 'grau']
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
async function main() {
  const dir = process.argv[2];
  const trainChat = process.argv[3];
  if (!dir) throw Error('Uso: node validate.js DIRETORIO');
  const ids = new Set(), schemasBySplit = {}, counts = {};
  for (const split of Object.keys(schemas)) for (const kind of ['fim','chat']) {
    const name = `${kind}-${split}.jsonl`;
    let count = 0;
    const file = trainChat && name === 'chat-train.jsonl' ? trainChat : path.join(dir,name);
    const input = readline.createInterface({input:fs.createReadStream(file), crlfDelay:Infinity});
    for await (const line of input) {
      const r = JSON.parse(line);
      if (ids.has(r.id) || r.split !== split || r.type !== kind || !r.synthetic) throw Error(`metadados: ${r.id}`);
      ids.add(r.id); (schemasBySplit[split] ??= new Set()).add(r.schema_id);
      const i = Number(r.id.split('-').pop()), n = Math.floor(i / 6), c = i % 6;
      const [coll, pid, sys, state, rank] = schemas[split];
      const id = `PRJ-${String(n + {train:0,val:10000,test:20000}[split]).padStart(5,'0')}`, system = `SIS-${split.toUpperCase()}-${String(n % 71).padStart(2,'0')}`;
      const status = ['ativo','pausado','concluido'][n % 3], p = n % 5 + 1;
      const calls = [];
      const cursor = {sort: arg => {calls.push(['sort',arg]); return cursor;}, limit: arg => {calls.push(['limit',arg]); return cursor;}};
      const collection = {};
      for (const method of ['find','findOne','countDocuments']) collection[method] = (...args) => {calls.push([method,...args]); return cursor;};
      const db = {getCollection: name => {calls.push(['collection',name]); return collection;}};
      const code = kind === 'fim' ? r.prefix+r.middle+r.suffix : r.output;
      new vm.Script(code).runInNewContext({db}, {timeout:100});
      const expected = [
        [['findOne',{[pid]:id,[sys]:system}]],
        [['find',{[sys]:system,[state]:status}]],
        [['countDocuments',{[sys]:system,[rank]:{$gte:p}}]],
        [['find',{[sys]:system,[state]:{$ne:status}}]],
        [['find',{[sys]:system}],['sort',{[rank]:-1}],['limit',5]],
        [['find',{[pid]:id,[sys]:system},{[pid]:1,[state]:1,_id:0}]]
      ][c];
      if (!eq(calls, [['collection',coll],...expected])) throw Error(`consulta incorreta: ${r.id} ${JSON.stringify(calls)}`);
      if (kind === 'fim' && (!r.middle || r.target !== r.middle || !r.prompt.includes('<|fim_middle|>'))) throw Error(`FIM: ${r.id}`);
      if (kind === 'chat' && (!r.output.startsWith('// ') || !r.context.editor)) throw Error(`chat: ${r.id}`);
      count++;
    }
    counts[name] = count;
  }
  for (const a of Object.keys(schemasBySplit)) for (const b of Object.keys(schemasBySplit))
    if (a !== b && [...schemasBySplit[a]].some(x => schemasBySplit[b].has(x))) throw Error('schema entre splits');
  console.log(JSON.stringify({counts, ids:ids.size, sintaxe_e_argumentos:'ok', schemas_disjuntos:'ok'},null,2));
}
main().catch(e => {console.error(e); process.exitCode=1;});
