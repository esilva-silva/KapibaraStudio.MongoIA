#!/usr/bin/env python3
"""Reformula somente o contexto de chats de treino; dados originais permanecem intactos."""
import hashlib
import json
from pathlib import Path
import build

root=build.ROOT
src=root/'data/generated/pro-context-v1b/chat-train.jsonl'
out=root/'data/generated/pro-chat-context-v2'
if out.exists(): raise SystemExit('Diretório já existe')
out.mkdir(parents=True)
h=hashlib.sha256(); count=0
with src.open(encoding='utf8') as f, (out/'chat-train.jsonl').open('xb') as dest:
 for line in f:
  r=json.loads(line); i=int(r['id'].split('-')[-1]); n=i//6
  db, coll, pid, system, state, rank=build.SCHEMAS['train']
  ident=f'PRJ-{n:05d}'; sysname=f'SIS-TRAIN-{n%71:02d}'; status=('ativo','pausado','concluido')[n%3]; priority=n%5+1
  r['id']=f'pro-chatfix-train-{i:05d}'; r['origin']='gen:pro-synthetic-context-v2'
  r['context']['editor']=(f'Coleção sintética: {coll}\n{pid}: string\n{system}: string\n{state}: string\n{rank}: int\n'
                           f'ID: {ident}\nSistema: {sysname}\nSituação: {status}\nPrioridade: {priority}')
  data=(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode(); dest.write(data); h.update(data); count+=1
manifest={'chat-train.jsonl':{'count':count,'sha256':h.hexdigest()},'derived_from_sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
print(json.dumps(manifest,indent=2))
