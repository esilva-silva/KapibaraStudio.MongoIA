#!/usr/bin/env python3
"""Audita manifests, isolamento e renderização sem abrir o conjunto cego."""
import hashlib
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/data'))
import blind_guard
import render_contract as rc

DIRS=['pro-context-v1b','pro-mid-script-v4','pro-mid-script-extra-v4']
parser=argparse.ArgumentParser()
parser.add_argument('--train-chat',type=Path,help='substituição revisada dos chats de treino')
args=parser.parse_args()
ids=set(); records=Counter(); roles=defaultdict(set); groups=defaultdict(set)
contents=Counter(); max_len=Counter(); n=0
for name in DIRS:
    folder=ROOT/'data/generated'/name
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for file,meta in manifest.items():
        p=args.train_chat if name=='pro-context-v1b' and file=='chat-train.jsonl' and args.train_chat else folder/file
        if p != folder/file:
            meta=json.loads((p.parent/'manifest.json').read_text(encoding='utf8'))[p.name]
        blind_guard.check_path(p)
        raw=p.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==meta['sha256'], p
        rows=raw.decode('utf-8').splitlines()
        assert len(rows)==meta['count'], p
        for line in rows:
            r=json.loads(line); n+=1
            assert r['id'] not in ids, r['id']
            ids.add(r['id']); split=r['split']; roles[r['schema_id']].add(split); groups[r['group']].add(split)
            records[(split,r['type'])]+=1
            assert r['synthetic'] and r['schema_version']=='kapi-ds-record-v2'
            if r['type']=='fim':
                assert r['prefix'] and r['middle'] and r['suffix'] and r['target']==r['middle']
                prompt=rc.qwen_fim_text(rc.render_model_prefix(r['prefix'],r['editor'],r['language']),r['suffix'])
                assert r['prompt']==prompt, r['id']
                max_len['prefix']=max(max_len['prefix'],len(r['prefix']))
                max_len['suffix']=max(max_len['suffix'],len(r['suffix']))
                max_len['middle']=max(max_len['middle'],len(r['middle']))
                contents[(split,'fim',r['prefix'],r['middle'],r['suffix'])]+=1
            else:
                assert r['instruction'] and r['context']['editor'] and r['output'].startswith('// ')
                contents[(split,'chat',r['instruction'],r['context']['editor'],r['output'])]+=1
assert n==41000 and all(len(s)==1 for s in roles.values()) and all(len(s)==1 for s in groups.values())
assert all(v == 1 for v in contents.values()), 'registros idênticos na mesma divisão'
print(json.dumps({'total':n,'counts':{f'{s}:{t}':v for (s,t),v in sorted(records.items())},
 'schemas_by_split':{k:sorted(v) for k,v in roles.items()},'unique_ids':len(ids),
 'exact_duplicate_records_within_split':sum(v-1 for v in contents.values()),'max_chars':dict(max_len),
 'manifest_sha256':'ok','fim_prompt_renderer':'ok','group_and_schema_isolation':'ok'},indent=2,ensure_ascii=False))
