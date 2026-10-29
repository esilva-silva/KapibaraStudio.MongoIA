#!/usr/bin/env python3
"""Empacota registros Pro em shards/IDs reconhecidos pelo indexador do professor.

Não marca os dados como aprovados, não tokeniza, não treina e não substitui o pipeline.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/data'))
import blind_guard

SOURCES=['pro-context-v1b','pro-mid-script-v4','pro-mid-script-extra-v4']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=ROOT/'data/generated/pro-teacher-input-v2')
    ap.add_argument('--exclude-train-ids',type=Path,help='IDs identificados pela descontaminação; somente treino')
    ap.add_argument('--train-chat',type=Path,help='chat de treino revisado com manifesto no mesmo diretório')
    a=ap.parse_args(); blind_guard.check_path(a.out)
    removed=set()
    if a.exclude_train_ids:
        blind_guard.check_path(a.exclude_train_ids)
        removed={line.strip() for line in a.exclude_train_ids.read_text(encoding='utf-8').splitlines() if line.strip()}
    if a.out.exists(): raise SystemExit('Diretório já existe; escolha outro --out')
    src=a.out/'source'; splits=a.out/'splits'
    src.mkdir(parents=True); splits.mkdir()
    ids={s:set() for s in ('train','val','test','bench')}; counts=Counter(); allids=set()
    for idx, name in enumerate(SOURCES):
        folder=ROOT/'data/generated'/name
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        for kind in ('fim','chat'):
            dest=src/(f'fim-{idx:05d}.jsonl' if kind=='fim' else 'chat.jsonl')
            mode='ab' if kind=='chat' and dest.exists() else 'xb'
            with dest.open(mode) as output:
                for split in ('train','val','test'):
                    is_override = idx == 0 and kind == 'chat' and split == 'train' and a.train_chat is not None
                    p=a.train_chat if is_override else folder/f'{kind}-{split}.jsonl'
                    source_manifest=json.loads((p.parent/'manifest.json').read_text(encoding='utf8')) if is_override else manifest
                    if p.name not in source_manifest: continue
                    if not p.is_file(): raise FileNotFoundError(p)
                    blind_guard.check_path(p)
                    raw=p.read_bytes()
                    if hashlib.sha256(raw).hexdigest()!=source_manifest[p.name]['sha256']: raise ValueError(f'hash inválido: {p}')
                    for line in raw.splitlines(keepends=True):
                        r=json.loads(line)
                        if not line.startswith(b'{"id":"') or r['split']!=split or r['type']!=kind or r['id'] in allids:
                            raise ValueError(f'registro inválido: {r.get("id")}')
                        if r['id'] in removed:
                            if split!='train': raise ValueError('ID de avaliação na lista de exclusão')
                            removed.remove(r['id'])
                            continue
                        allids.add(r['id']); ids[split].add(r['id']); counts[f'{split}:{kind}']+=1
                        if kind=='fim':
                            # O renderizador de val_gen do professor exige este campo.
                            # Os registros deste lote são todos Mongo Console JavaScript.
                            r['dialect']='console'
                            line=(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8')
                        output.write(line)
    if removed: raise ValueError(f'IDs de exclusão não encontrados: {len(removed)}')
    for split, values in ids.items():
        (splits/f'{split}.ids.txt').write_text(''.join(x+'\n' for x in sorted(values)),encoding='utf-8')
    manifest={'status':'staging-only-not-approved', 'counts':dict(sorted(counts.items())),
              'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(src.glob('*.jsonl'))}}
    (src/'manifest.json').write_text(json.dumps({'files':{k:{'sha256':v} for k,v in manifest['files'].items()}},indent=2)+'\n',encoding='utf-8')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
