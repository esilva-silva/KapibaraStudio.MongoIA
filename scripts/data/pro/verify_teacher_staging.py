#!/usr/bin/env python3
"""Executa o indexador real do professor e verifica seu isolamento, sem tokenizer ou GPU."""
import json
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/teacher'))
import build_teacher_corpus as teacher

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=ROOT/'data/generated/pro-teacher-input-v2')
root=parser.parse_args().root
teacher.CFG=yaml.safe_load((ROOT/'Repos/KapiTeacher-Mongo-7B/configs/corpus_v2d.yaml').read_text(encoding='utf-8'))
teacher.NAMES=None
files=sorted(p for p in (root/'source').iterdir() if teacher.SHARD_RE.match(p.name))
scans=[teacher._scan_shard(str(p)) for p in files]
ids=defaultdict(set); groups=defaultdict(set); schemas=defaultdict(set); counts=Counter()
for s in scans:
    assert not s['violations'] and not s.get('hom'), (s['path'],s['violations'])
    ids['train'].update(r[0] for r in s['train'])
    for ident, split in s['eval_ids'].items(): ids[split].add(ident)
    for split, vals in s['groups'].items(): groups[split].update(vals)
    for split, vals in s['schemas'].items(): schemas[split].update(vals)
    counts.update(s['counts'])
for split in ('train','val','test','bench'):
    expected=set((root/'splits'/f'{split}.ids.txt').read_text(encoding='utf-8').splitlines())
    assert ids[split]==expected, (split,len(ids[split]),len(expected))
for split in ('val','test','bench'):
    assert not ids['train'] & ids[split] and not groups['train'] & groups[split] and not schemas['train'] & schemas[split]
assert dict(counts)=={'train:chat':6000,'val:chat':750,'test:chat':750,'train:fim':26800,'val:fim':3350,'test:fim':3350}
print(json.dumps({'files':[p.name for p in files], 'counts':dict(counts),'teacher_indexer':'ok','split_ids_groups_schemas':'ok'},indent=2))
