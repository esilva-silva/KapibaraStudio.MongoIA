#!/usr/bin/env python3
"""Valida limites com o tokenizer fixado e exercita o renderizador real do professor, sem treinar."""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import yaml
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/data'),str(ROOT/'scripts/teacher')]
import render_contract as rc
import build_teacher_corpus as teacher

EXPECTED_HASH='c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tokenizer',required=True,type=Path); ap.add_argument('--source',type=Path,default=ROOT/'data/generated/pro-teacher-input-v2/source'); args=ap.parse_args()
    digest=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    if digest != EXPECTED_HASH: raise SystemExit(f'tokenizer não corresponde ao lock: {digest}')
    tok=Tokenizer.from_file(str(args.tokenizer))
    for marker,want in teacher.EXPECTED_IDS.items():
        got=tok.encode(marker,add_special_tokens=False).ids
        if got != [want]: raise ValueError((marker,got,want))
    cfg=yaml.safe_load((ROOT/'Repos/KapiTeacher-Mongo-7B/configs/corpus_v2d.yaml').read_text(encoding='utf-8'))
    teacher.CFG=cfg; teacher._TOK=tok; teacher.BLIND_PROMPTS=frozenset()
    source=args.source
    counts=Counter(); maxima=defaultdict(int); selected_by_file={}; failures=[]
    def check_batch(batch):
        texts=[]
        for r in batch:
            if r['type']=='fim':
                prompt=r['prompt']; target=r['middle']
            else:
                prompt=teacher.render_chat_prompt(r,cfg); target=r['output']
            texts.extend((prompt,target))
        encoded=tok.encode_batch(texts,add_special_tokens=False)
        for i,r in enumerate(batch):
            p=len(encoded[2*i].ids); t=len(encoded[2*i+1].ids)
            kind=r['type']; counts[kind]+=1
            maxima[f'{kind}_prompt']=max(maxima[f'{kind}_prompt'],p)
            maxima[f'{kind}_target']=max(maxima[f'{kind}_target'],t)
            maxima[f'{kind}_total_with_eos']=max(maxima[f'{kind}_total_with_eos'],p+t+1)
            if kind=='fim' and (p>2048 or t>96): failures.append((r['id'],'fim_budget',p,t))
            if p+t+1>cfg['seq_len']: failures.append((r['id'],'teacher_budget',p,t))
    for path in sorted(p for p in source.iterdir() if teacher.SHARD_RE.match(p.name)):
        picked={}; per_group=set(); batch=[]
        with path.open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line); batch.append(r)
                if r['split'] in ('train','val'):
                    key=(r['split'],r['type'],r.get('scenario'))
                    if key not in per_group:
                        per_group.add(key); picked[r['id']]=('fim_ctx' if r['type']=='fim' else 'chat',r['split']=='val',None)
                if len(batch)>=128: check_batch(batch); batch=[]
        if batch: check_batch(batch)
        selected_by_file[path]=picked
    if counts != {'fim':33500,'chat':7500} or failures: raise ValueError((counts,failures[:10]))
    sample=0
    for path,picked in selected_by_file.items():
        result=teacher._render_shard((str(path),picked,{},20261008))
        if result['dropped'] or len(result['items'])!=len(picked): raise ValueError((path,result['dropped']))
        for row in result['items']:
            assert 0 < row['loss_start'] < len(row['ids']) and row['ids'][-1]==teacher.EXPECTED_IDS['<|endoftext|>']
            assert row['meta']['n_prompt']==row['loss_start']
            if row['meta']['kind'] == 'partial' and row.get('gen'):
                assert row['gen']['dialect']=='console'
            sample+=1
    print(json.dumps({'tokenizer_sha256':digest,'counts':dict(counts),'max_tokens':dict(maxima),
                      'fim_prompt_limit':2048,'fim_target_limit':96,'teacher_seq_limit':cfg['seq_len'],
                      'teacher_render_samples':sample,'errors':len(failures)},indent=2))

if __name__=='__main__': main()
