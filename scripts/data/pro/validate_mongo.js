// Executar em mongosh, depois de validate.js. Cria três bancos novos de fixtures.
// KAPI_PRO_DIR é o diretório Linux dos JSONL; KAPI_PRO_RUN é um identificador único.
const fs = require('fs');
const path = require('path');
async function main() {
const dir = process.env.KAPI_PRO_DIR;
const run = process.env.KAPI_PRO_RUN;
if (!dir || !/^[a-z0-9]{12,40}$/.test(run || '')) throw Error('Defina KAPI_PRO_DIR e KAPI_PRO_RUN (12 a 40 caracteres alfanuméricos minúsculos).');
const schema = {
  train: ['projetos_demo', 'projetoId', 'sistema', 'situacao', 'prioridade', 4000, 0],
  val: ['iniciativas_demo', 'codigoProjeto', 'sistemaOrigem', 'estado', 'nivel', 500, 10000],
  test: ['carteira_demo', 'idProjeto', 'sistemaResponsavel', 'status', 'grau', 500, 20000]
};
const names = Object.keys(schema).map(s => `kapi_fx_pro_${run}_${s}`);
const existing = new Set(db.getMongo().getDBNames());
if (names.some(n => existing.has(n))) throw Error('Um banco de destino já existe; escolha outro KAPI_PRO_RUN.');
const summary = {run, databases:names, fixtures:0, checked:0, bySplit:{}, errors:[]};
for (const [split, [collection, pid, sys, state, rank, nmax, offset]] of Object.entries(schema)) {
  const database = db.getSiblingDB(`kapi_fx_pro_${run}_${split}`);
  const docs = [], bySystem = new Map();
  for (let n = 0; n < nmax; n++) {
    const system = `SIS-${split.toUpperCase()}-${String(n % 71).padStart(2, '0')}`;
    const item = {_id:n*2+1, [pid]:`PRJ-${String(n+offset).padStart(5,'0')}`, [sys]:system,
                  [state]:['ativo','pausado','concluido'][n%3], [rank]:n%5+1};
    // Mesmo ID em outro sistema: detecta omissão da condição de sistema.
    docs.push({...item, _id:n*2, [sys]:'OUTRO-'+system}, item);
    if (!bySystem.has(system)) bySystem.set(system, []);
    bySystem.get(system).push(item);
  }
  database.getCollection(collection).insertMany(docs, {ordered:true});
  summary.fixtures += docs.length;
  summary.bySplit[split] = {checked:0, cases:[0,0,0,0,0,0]};
  for (const kind of ['fim','chat']) {
    const data = fs.readFileSync(path.join(dir, `${kind}-${split}.jsonl`), 'utf8');
    for (const line of data.split('\n')) {
      if (!line) continue;
      const r = JSON.parse(line), i = Number(r.id.split('-').pop()), n = Math.floor(i/6), c = i%6;
      const source = kind === 'fim' ? r.prefix+r.middle+r.suffix : r.output;
      const pos = source.lastIndexOf('db.getCollection(');
      if (pos < 0) throw Error(`sem consulta: ${r.id}`);
      const expr = source.slice(pos).trim().replace(/;$/, '');
      // Os métodos do shell são assíncronos dentro de Function, fora da transformação
      // automática do mongosh. Aguarde cada etapa antes de encadear a próxima.
      let value;
      if (c === 4) {
        const marker = ').sort(';
        const at = expr.indexOf(marker);
        if (at < 0 || !expr.endsWith(').limit(5)')) throw Error(`encadeamento inesperado: ${r.id}`);
        const findExpr = expr.slice(0, at + 1);
        const sortExpr = expr.slice(at + marker.length, -').limit(5)'.length);
        value = await Function('db', `return (async()=>{const c=await ${findExpr}; const s=await c.sort(${sortExpr}); return await s.limit(5);})()`)(database);
      } else {
        value = await Function('db', `return (async()=>await (${expr}))()`)(database);
      }
      const actual = value && typeof value.toArray === 'function' ? await value.toArray() : value;
      const all = bySystem.get(`SIS-${split.toUpperCase()}-${String(n%71).padStart(2,'0')}`);
      const id = `PRJ-${String(n+offset).padStart(5,'0')}`, status = ['ativo','pausado','concluido'][n%3];
      const matchId = all.find(d => d[pid] === id);
      let ok = false;
      const sameIds = (a,b) => JSON.stringify(a.map(x=>x._id).sort((x,y)=>x-y)) === JSON.stringify(b.map(x=>x._id).sort((x,y)=>x-y));
      if (c === 0) ok = actual && actual._id === matchId._id;
      if (c === 1) ok = sameIds(actual, all.filter(d => d[state] === status));
      if (c === 2) ok = actual === all.filter(d => d[rank] >= n%5+1).length;
      if (c === 3) ok = sameIds(actual, all.filter(d => d[state] !== status));
      if (c === 4) {
        const expectedRanks = all.map(d=>d[rank]).sort((a,b)=>b-a).slice(0,5);
        ok = actual.length === expectedRanks.length && actual.every((d,j)=>d[sys] === matchId[sys] && d[rank] === expectedRanks[j]);
      }
      if (c === 5) ok = actual.length === 1 && actual[0][pid] === id && actual[0][state] === matchId[state] && Object.keys(actual[0]).length === 2;
      summary.checked++; summary.bySplit[split].checked++; summary.bySplit[split].cases[c]++;
      if (!ok) { summary.errors.push(r.id); if (summary.errors.length >= 10) throw Error(JSON.stringify(summary)); }
    }
  }
}
print(JSON.stringify(summary, null, 2));
if (summary.errors.length) throw Error('resultados divergentes');
}
main();
