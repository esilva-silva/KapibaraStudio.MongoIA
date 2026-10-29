// Verifica reconstrução e argumentos, sem executar escritas reais. --plans grava planos verificados.
const fs = require('fs'), vm = require('vm'), path = require('path');
const dir = process.argv[2], out = process.argv[3];
const expanded = process.argv.includes('--expanded');
if (!dir || (out && fs.existsSync(out))) throw Error('Informe diretório e, opcionalmente, caminho novo de planos');
const schemas = {
 train:['projetos_demo','projetoId','sistema','situacao','prioridade',0],
 val:['iniciativas_demo','codigoProjeto','sistemaOrigem','estado','nivel',10000],
 test:['carteira_demo','idProjeto','sistemaResponsavel','status','grau',20000]
};
const seen = new Set(), counts = {}, plans = [], cases = Array(expanded ? 20 : 12).fill(0);
for (const [split,[coll,pid,sys,state,rank,offset]] of Object.entries(schemas)) {
  let count=0;
  for (const line of fs.readFileSync(path.join(dir,`fim-${split}.jsonl`),'utf8').split('\n')) {
    if (!line) continue;
    const r=JSON.parse(line), j=Number(r.id.split('-').pop()), n=1+2*j;
    const trainCases=Array.from({length:20},(_,x)=>x).filter(x=>![12,15,17].includes(x));
    const c=expanded && split==='train' ? trainCases[j%trainCases.length] : j%cases.length;
    if (seen.has(r.id) || r.split!==split || r.scenario!==c || r.target!==r.middle || !r.middle || !r.suffix || !r.prefix.includes('const resultado = ') || !r.suffix.includes('código posterior')) throw Error(`estrutura ${r.id}`);
    seen.add(r.id);
    const id=expanded ? `MID-${split.toUpperCase()}-${String(n).padStart(5,'0')}` : `PRJ-${String(n+offset).padStart(5,'0')}`, system=`SIS-${expanded?'EXTRA-':''}${split.toUpperCase()}-${String(n).padStart(5,'0')}`, level=n%5+1;
    const calls=[];
    const collection={};
    const cursor={sort:(...args)=>{calls.push(['sort',...args]); return cursor;}};
    for (const method of ['find','aggregate','insertOne','updateOne','updateMany']) collection[method]=(...args)=>{calls.push([method,...args]); return cursor;};
    const db={getCollection:name=>{calls.push(['collection',name]); return collection;}};
    new vm.Script(r.prefix+r.middle+r.suffix).runInNewContext({db},{timeout:100});
    const expected = [
      ['find',{[pid]:id,[sys]:system},{[pid]:1,[state]:1,_id:0}],
      ['find',{[sys]:system}],
      ['find',{[sys]:system,[rank]:{$gte:level}}],
      ['find',{[sys]:system},{[pid]:1,[state]:1,_id:0}],
      ['find',{[sys]:system,'meta.regiao':'sul'}],
      ['find',{[sys]:system,[state]:{$in:['ativo','pausado']}}],
      ['aggregate',[{$match:{[sys]:system}},{$sort:{[rank]:-1}}]],
      ['aggregate',[{$match:{[sys]:system}},{$project:{[pid]:1,[state]:1,_id:0}}]],
      ['insertOne',{[pid]:id,[sys]:system,[state]:'ativo',[rank]:level}],
      ['insertOne',{[pid]:id,[sys]:system,[state]:'ativo',[rank]:level,etiquetas:['azul','verde']}],
      ['updateOne',{[pid]:id,[sys]:system},{$set:{[state]:'concluido'}}],
      ['updateMany',{[sys]:system},{$inc:{[rank]:1}}],
      ['find',{[sys]:system,meta:{regiao:'sul',canal:'web'}}],
      ['find',{[sys]:system,$or:[{[state]:'ativo'},{[rank]:level}]}],
      ['find',{[sys]:system}],
      ['updateOne',{[pid]:id,[sys]:system},{$set:{meta:{regiao:'sul',canal:'web'}}}],
      ['updateMany',{[sys]:system,[state]:{$in:['ativo','pausado']}},{$set:{[state]:'concluido'}}],
      ['insertOne',{[pid]:id,[sys]:system,[state]:'ativo',[rank]:level,meta:{regiao:'sul',canal:'web'}}],
      ['aggregate',[{$match:{[sys]:system}},{$group:{_id:`$${state}`,total:{$sum:1}}}]],
      ['find',{[sys]:system,observacao:{$exists:true}}]
    ][c];
    const expectedCalls=[['collection',coll],expected,...(c===14 ? [['sort',{[rank]:1}]] : [])];
    if (JSON.stringify(calls)!==JSON.stringify(expectedCalls)) throw Error(`semântica ${r.id}: ${JSON.stringify(calls)}`);
    plans.push({id:r.id,split,scenario:c,pid,sys,state,rank,ident:id,system,level,calls:calls.slice(1)});
    count++; cases[c]++;
  }
  counts[split]=count;
}
if (JSON.stringify(counts)!==JSON.stringify(expanded ? {train:8000,val:1000,test:1000} : {train:800,val:100,test:100})) throw Error('contagens inesperadas');
if(out) fs.writeFileSync(out,plans.map(p=>JSON.stringify(p)).join('\n')+'\n',{flag:'wx'});
console.log(JSON.stringify({counts,cases,unique:seen.size,reconstrucao_e_argumentos:'ok'},null,2));
