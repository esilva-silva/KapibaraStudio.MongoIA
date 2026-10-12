// Verifica sintaxe JavaScript sem executar: vm.Script apenas compila.
// Entrada (stdin): JSONL {"id": ..., "code": "...", "kind": "script"|"mongosh"|"json"}. Saída: JSONL {"id", "ok", "error"}.
// "mongosh" aceita await no nível superior (o mongosh executa scripts como corpo assíncrono).
import { Script } from "node:vm";
import { createInterface } from "node:readline";

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of rl) {
  if (!line.trim()) continue;
  const { id, code, kind } = JSON.parse(line);
  let ok = true;
  let error = null;
  try {
    if (kind === "json") JSON.parse(code);
    else if (kind === "mongosh" && /\bawait\b/.test(code)) new Script("(async () => {\n" + code + "\n})", { filename: "snippet.js" });
    else new Script(code, { filename: "snippet.js" });
  } catch (e) {
    ok = false;
    error = String(e && e.message || e).slice(0, 200);
  }
  process.stdout.write(JSON.stringify({ id, ok, error }) + "\n");
}
