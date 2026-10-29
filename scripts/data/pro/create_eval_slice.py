#!/usr/bin/env python3
"""Seleção fixa e balanceada para comparar modelos no split val, nunca test."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'data/generated/pro-comparison-val-v1'

def main():
    if OUT.exists(): raise SystemExit('Diretório já existe; não sobrescrever.')
    sources=[('contexto',ROOT/'data/generated/pro-context-v1b/fim-val.jsonl'),
             ('meio',ROOT/'data/generated/pro-mid-script-v4/fim-val.jsonl'),
             ('meio',ROOT/'data/generated/pro-mid-script-extra-v4/fim-val.jsonl')]
    pools=defaultdict(list)
    for label,p in sources:
        for line in p.open(encoding='utf-8'):
            r=json.loads(line)
            if r['split']!='val' or r['type']!='fim': raise ValueError(r['id'])
            pools[(label,r['scenario'])].append(r)
    if len(pools)!=26: raise ValueError(f'Famílias inesperadas: {len(pools)}')
    selected=[]
    for key,rows in sorted(pools.items()):
        rows.sort(key=lambda r: hashlib.sha256(('pro-compare-v1:'+r['id']).encode()).hexdigest())
        if len(rows)<10: raise ValueError(key)
        selected.extend(rows[:10])
    OUT.mkdir(parents=True)
    raw=b''.join((json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode() for r in selected)
    (OUT/'val.jsonl').write_bytes(raw)
    manifest={'split':'val','examples':len(selected),'per_family':10,'context_families':6,'mid_families':20,
              'sha256':hashlib.sha256(raw).hexdigest(),'ids_sha256':hashlib.sha256('\n'.join(r['id'] for r in selected).encode()).hexdigest(),
              'warning':'Sintético; famílias majoritariamente vistas no treino; não é conjunto cego.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
