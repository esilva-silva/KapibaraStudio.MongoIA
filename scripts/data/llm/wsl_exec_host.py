#!/usr/bin/env python3
"""Executa o TEXTO real de consultas com mongosh (no WSL, python3 + stdlib) atras do proxy somente-leitura `ro_proxy.py`.

  wsl -d Ubuntu -e python3 /mnt/c/.../wsl_exec_host.py --batch batch.jsonl --out outdir [--timeout 20] [--max-time-ms 3000] [--mongo 127.0.0.1:27017]

batch.jsonl: {"id", "db", "text"} (o `text` ja com getSiblingDB reescrito para o banco real kapi_fx_*). O mongosh recebe os textos por STDIN
(semantica de REPL: o valor da ultima expressao e exibido, o cursor e iterado -> o comando de leitura realmente vai ao servidor), cada texto
dentro de um bloco `{ ... }` (isola const/let entre consultas) e entre marcadores. Saida: <out>/results.jsonl (uma linha por consulta) e
<out>/proxy_stats.json (contadores de comandos permitidos/recusados no servidor).

Resultado por consulta: status (ok | error | timeout | no_exec | blocked), error, n_docs, ms, commands [{name, db, bson}], server_errors, blocked_attempts.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ro_proxy import RoProxy  # noqa: E402

ERR_RE = re.compile(r"(Uncaught[\s\S]{0,260}|Mongosh\w*Error[^\n]*|MongoServerError[^\n]*|MongoInvalidArgumentError[^\n]*|BSON\w*Error[^\n]*)")


def chunk_text(item: dict) -> str:
    i, db, text = item["id"], item["db"], item["text"]
    return (f'db = db.getSiblingDB({json.dumps(db)});\n'
            f'print("@@S:{i}@@"); db.runCommand({{ping: 1, comment: "KAPI:S:{i}"}});\n'
            "{\n" + text + "\n}\n"
            f'print("@@E:{i}@@"); db.runCommand({{ping: 1, comment: "KAPI:E:{i}"}});\n')


class Host:
    def __init__(self, batch, out, timeout, max_time_ms, mongo, allowed_dbs):
        self.batch, self.out, self.timeout = batch, out, timeout
        self.events: "queue.Queue" = queue.Queue()
        host, port = mongo.split(":")
        self.proxy = RoProxy((host, int(port)), allowed_dbs, max_time_ms, event_cb=self.events.put)
        self.proxy.serve_forever()
        self.res: dict = {}

    def run_chunk(self, items: list[dict]) -> str | None:
        """Roda `items` num mongosh; devolve id da consulta que travou (timeout) ou None."""
        uri = f"mongodb://127.0.0.1:{self.proxy.port}/?directConnection=true&appName=kapi-llm-validate"
        p = subprocess.Popen(["mongosh", "--quiet", "--norc", uri], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
        lines: "queue.Queue" = queue.Queue()
        threading.Thread(target=lambda: [lines.put(l) for l in p.stdout] or lines.put(None), daemon=True).start()

        def feed():
            try:
                # compat com a API do Console da IDE (cursor.project): o mongosh so tem .projection(); cursor lazy, nenhum comando vai ao servidor
                p.stdin.write('{ const __c = db.getSiblingDB(%s).getCollection("x").find(); '
                              'Object.getPrototypeOf(__c).project = function (x) { return this.projection(x); }; }\n' % json.dumps(items[0]["db"]))
                for it in items:
                    p.stdin.write(chunk_text(it))
                    p.stdin.flush()
                p.stdin.write("\n")
                p.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
        threading.Thread(target=feed, daemon=True).start()
        cur, seg, t_last, t_start, hung = None, [], time.time(), {}, None
        done_ids = set()
        while True:
            try:
                ln = lines.get(timeout=1.0)
            except queue.Empty:
                if time.time() - t_last > self.timeout:
                    hung = cur or next((it["id"] for it in items if it["id"] not in done_ids), None)
                    break
                continue
            if ln is None:
                break
            t_last = time.time()
            m = re.search(r"@@S:([^@]+)@@", ln)
            if m and "print(" not in ln:
                cur, seg = m.group(1), []
                t_start[cur] = time.time()
                continue
            m = re.search(r"@@E:([^@]+)@@", ln)
            if m and "print(" not in ln:
                i = m.group(1)
                self.res.setdefault(i, {})["output"] = "".join(seg)[-4000:]
                self.res[i]["ms"] = round((time.time() - t_start.get(i, time.time())) * 1000)
                done_ids.add(i)
                cur, seg = None, []
                continue
            if cur:
                seg.append(ln)
        try:
            p.kill()
        except OSError:
            pass
        return hung

    def drain(self):
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                return
            i = ev.get("id")
            if not i:
                continue
            r = self.res.setdefault(i, {})
            if ev["ev"] == "cmd":
                r.setdefault("commands", []).append({"name": ev["name"], "db": ev["db"], "bson": ev["bson"]})
            elif ev["ev"] == "docs":
                r["n_docs"] = (r.get("n_docs") or 0) + (ev.get("n") or 0)
                r["n_reads"] = r.get("n_reads", 0) + 1
            elif ev["ev"] == "err":
                if ev.get("code") == 13:
                    r.setdefault("blocked_attempts", []).append(ev.get("errmsg"))
                else:
                    r.setdefault("server_errors", []).append({"code": ev.get("code"), "msg": ev.get("errmsg")})

    def run(self):
        items = list(self.batch)
        remaining = items
        timed_out = set()
        t0 = time.time()
        while remaining:
            hung = self.run_chunk(remaining)
            self.drain()
            if hung is None:
                break
            time.sleep(0.2)
            self.drain()
            timed_out.add(hung)
            self.res.setdefault(hung, {})["timeout"] = True
            idx = next((k for k, it in enumerate(remaining) if it["id"] == hung), len(remaining) - 1)
            remaining = remaining[idx + 1:]
        self.drain()
        time.sleep(0.3)
        self.drain()
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "results.jsonl"), "w", encoding="utf-8") as f:
            for it in items:
                r = self.res.get(it["id"], {})
                out = r.get("output", "")
                err = None
                if r.get("timeout"):
                    status = "timeout"
                elif r.get("blocked_attempts"):
                    status, err = "blocked", r["blocked_attempts"][0]
                elif r.get("server_errors"):
                    status, err = "error", f"{r['server_errors'][0]['code']}: {r['server_errors'][0]['msg']}"
                else:
                    m = ERR_RE.search(out)
                    if m:
                        status, err = "error", re.sub(r"\s+", " ", m.group(1))[:240]
                    elif "output" not in r:
                        status, err = "error", "sem_marcador_final"
                    elif not r.get("commands"):
                        status = "no_exec"
                    else:
                        status = "ok"
                f.write(json.dumps({"id": it["id"], "status": status, "error": err, "n_docs": r.get("n_docs", 0), "n_reads": r.get("n_reads", 0),
                                    "ms": r.get("ms"), "commands": (r.get("commands") or [])[:12], "output_tail": out[-600:]},
                                   ensure_ascii=False) + "\n")
        st = self.proxy.stats.to_dict()
        st["seconds"] = round(time.time() - t0, 1)
        st["queries"] = len(items)
        with open(os.path.join(self.out, "proxy_stats.json"), "w", encoding="utf-8") as f:
            json.dump(st, f)
        self.proxy.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--max-time-ms", type=int, default=3000)
    ap.add_argument("--mongo", default="127.0.0.1:27017")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.batch, encoding="utf-8") if l.strip()]
    dbs = {it["db"] for it in items}
    Host(items, a.out, a.timeout, a.max_time_ms, a.mongo, dbs).run()


if __name__ == "__main__":
    main()
