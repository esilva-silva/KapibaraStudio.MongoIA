#!/usr/bin/env python3
"""Extrai somente val/test/bench de corpus antigo para auditoria, sem ler o blind."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/data'))
import blind_guard

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    blind_guard.check_path(a.source); blind_guard.check_path(a.out)
    if a.out.exists(): raise SystemExit('Saída já existe')
    files=sorted(a.source.glob('fim-*.jsonl'))+[a.source/'chat.jsonl']
    if not files or any(not f.is_file() for f in files): raise SystemExit('Arquivos do corpus ausentes')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    counts=Counter(); h=hashlib.sha256()
    with a.out.open('xb') as out:
        for file in files:
            blind_guard.check_path(file)
            with file.open('rb') as f:
                for line in f:
                    r=json.loads(line)
                    if r.get('split') in ('val','test','bench'):
                        out.write(line); h.update(line); counts[r['split']]+=1
    print(json.dumps({'counts':dict(counts),'sha256':h.hexdigest(),'files':len(files)}))
if __name__=='__main__': main()
