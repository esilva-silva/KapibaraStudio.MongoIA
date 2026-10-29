#!/usr/bin/env python3
"""Congela recorte balanceado de chat do split de validação."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
source=ROOT/'data/generated/pro-context-v1b/chat-val.jsonl'; out=ROOT/'data/generated/pro-chat-eval-val-v1'
if out.exists(): raise SystemExit('Diretório de saída já existe')
pools=defaultdict(list)
for line in source.open(encoding='utf8'):
 r=json.loads(line)
 if r['split']!='val' or r['type']!='chat': raise ValueError(r['id'])
 pools[r['scenario']].append(r)
if set(pools)!=set(range(6)): raise ValueError('Famílias incompletas')
rows=[]
for case in sorted(pools):
 pool=sorted(pools[case],key=lambda r:hashlib.sha256(('pro-chat-eval-v1:'+r['id']).encode()).hexdigest())
 if len(pool)<20: raise ValueError(case)
 rows+=pool[:20]
raw=b''.join((json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode() for r in rows)
out.mkdir(parents=True); (out/'val.jsonl').write_bytes(raw)
manifest={'split':'val','examples':len(rows),'families':6,'per_family':20,'sha256':hashlib.sha256(raw).hexdigest(),
          'ids_sha256':hashlib.sha256('\n'.join(r['id'] for r in rows).encode()).hexdigest(),
          'warning':'Amostra sintética de validação; não é blind nem mede equivalência de execução.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8'); print(json.dumps(manifest,ensure_ascii=False))
