// Verificacao de sintaxe (node vm.Script) do TEXTO de cada consulta: stdin JSONL {id,text} -> stdout JSONL {id,ok,err}.
// Aceita top-level await (o REPL do mongosh aceita) tentando de novo dentro de uma funcao async.
const vm = require('vm');
const rl = require('readline').createInterface({ input: process.stdin });
rl.on('line', (line) => {
  if (!line.trim()) return;
  const { id, text } = JSON.parse(line);
  let ok = true, err = null;
  try { new vm.Script(text); }
  catch (e1) {
    try { new vm.Script('(async()=>{' + text + '\n})'); }
    catch (e2) { ok = false; err = String(e1.message).slice(0, 200); }
  }
  process.stdout.write(JSON.stringify({ id, ok, err }) + '\n');
});
