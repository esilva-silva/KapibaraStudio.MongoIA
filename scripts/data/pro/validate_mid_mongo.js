// Executa os planos capturados do JavaScript reconstruído em um banco novo e isolado.
// Antes: validate_mid.js para cada lote, gerando os verified-plans.jsonl.
const fs = require('fs');
async function main() {
  const run=process.env.KAPI_PRO_RUN, list=process.env.KAPI_PRO_PLAN_FILES;
  if (!/^[a-z0-9]{12,40}$/.test(run || '') || !list) throw Error('KAPI_PRO_RUN ou KAPI_PRO_PLAN_FILES inválido');
  const name=`kapi_fx_pro_${run}_mid`;
  if ((await db.getMongo().getDBNames()).includes(name)) throw Error('Banco de destino já existe.');
  const database=await db.getSiblingDB(name);
  const result={database:name, checked:0, cases:Array(20).fill(0), bySplit:{train:0,val:0,test:0}, errors:[]};
  for (const filename of list.split(':')) {
    for (const line of fs.readFileSync(filename,'utf8').split('\n')) {
      if (!line) continue;
      const p=JSON.parse(line), {pid,sys,state,rank,ident,system,level,scenario:c}=p;
      const coll=await database.getCollection('fixtures');
      await coll.deleteMany({}); // Somente a coleção criada neste banco novo.
      const docs=[
        {_id:1,[pid]:ident,[sys]:'OUTRO-'+system,[state]:'pausado',[rank]:level,meta:{regiao:'sul',canal:'web'}},
        {_id:2,[pid]:ident,[sys]:system,[state]:'ativo',[rank]:level,meta:{regiao:'sul',canal:'web'},observacao:'synthetic'},
        {_id:3,[pid]:ident+'-OUTRO',[sys]:system,[state]:'concluido',[rank]:level+1,meta:{regiao:'norte',canal:'loja'}}
      ];
      await coll.insertMany(docs);
      const [method,...args]=p.calls[0];
      let actual=await coll[method](...args);
      if (p.calls.length===2) actual=await actual.sort(p.calls[1][1]);
      if (actual && typeof actual.toArray==='function') actual=await actual.toArray();
      const ids = a => a.map(x=>x._id).sort().join(',');
      const projected = a => a.length===2 && a.every(x=>Object.keys(x).length===2 && x[pid] && x[state]);
      let ok=false;
      if (c===0) ok=actual.length===1 && actual[0][pid]===ident && actual[0][state]==='ativo' && Object.keys(actual[0]).length===2;
      if (c===1 || c===2) ok=ids(actual)==='2,3';
      if (c===3 || c===7) ok=projected(actual);
      if (c===4 || c===5 || c===12 || c===19) ok=ids(actual)==='2';
      if (c===6) ok=actual.length===2 && actual[0]._id===3 && actual[1]._id===2;
      if ([8,9,17].includes(c)) {
        const inserted=await coll.findOne({_id:actual.insertedId});
        ok=!!actual.acknowledged && !!inserted && inserted[pid]===ident && inserted[sys]===system && inserted[state]==='ativo' && inserted[rank]===level;
        if(c===9) ok=ok && JSON.stringify(inserted.etiquetas)===JSON.stringify(['azul','verde']);
        if(c===17) ok=ok && inserted.meta.regiao==='sul' && inserted.meta.canal==='web';
      }
      if ([10,11,15,16].includes(c)) {
        const a=await coll.findOne({_id:1}), b=await coll.findOne({_id:2}), d=await coll.findOne({_id:3});
        if(c===10) ok=actual.modifiedCount===1 && a[state]==='pausado' && b[state]==='concluido' && d[state]==='concluido';
        if(c===11) ok=actual.modifiedCount===2 && a[rank]===level && b[rank]===level+1 && d[rank]===level+2;
        if(c===15) ok=actual.matchedCount===1 && a.meta.canal==='web' && b.meta.regiao==='sul' && b.meta.canal==='web' && d.meta.canal==='loja';
        if(c===16) ok=actual.modifiedCount===1 && a[state]==='pausado' && b[state]==='concluido' && d[state]==='concluido';
      }
      if(c===13) ok=ids(actual)==='2';
      if(c===14) ok=actual.length===2 && actual[0]._id===2 && actual[1]._id===3;
      if(c===18) ok=actual.length===2 && actual.some(x=>x._id==='ativo' && x.total===1) && actual.some(x=>x._id==='concluido' && x.total===1);
      result.checked++; result.cases[c]++; result.bySplit[p.split]++;
      if(!ok) { result.errors.push(p.id); if(result.errors.length>=5) throw Error(JSON.stringify(result)); }
    }
  }
  print(JSON.stringify(result,null,2));
  if(result.errors.length) throw Error('Divergência semântica.');
}
main();
