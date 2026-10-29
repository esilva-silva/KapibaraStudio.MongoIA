#!/usr/bin/env python3
"""Audita 13-gramas do recorte de validação contra FIM de corpus tokenizado anterior.
Não usa GPU, não altera corpus de origem; não substitui MinHash/chave/blind.
"""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/data'))
import decontaminate as dc

FIM_PRE,FIM_SUF,FIM_MID,EOS=151659,151661,151660,151643
LOCK='c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tokenizer',required=True,type=Path); ap.add_argument('--prior',required=True,type=Path); ap.add_argument('--out',required=True,type=Path); a=ap.parse_args()
    if a.out.exists(): raise SystemExit('Saída já existe')
    if hashlib.sha256(a.tokenizer.read_bytes()).hexdigest()!=LOCK: raise SystemExit('Tokenizer não corresponde ao lock')
    tok=Tokenizer.from_file(str(a.tokenizer))
    sample=ROOT/'data/generated/pro-comparison-val-v1/val.jsonl'; manifest=json.loads((sample.parent/'manifest.json').read_text(encoding='utf8'))
    if hashlib.sha256(sample.read_bytes()).hexdigest()!=manifest['sha256']: raise ValueError('Amostra divergente')
    index={}
    for line in sample.read_text(encoding='utf8').splitlines():
        r=json.loads(line)
        for gram,content in dc.features(r)['grams'].items():
            if content>=2: index.setdefault(gram,set()).add(r['id'])
    corpus=json.loads((a.prior/'manifest.json').read_text(encoding='utf8'))
    counts=Counter(); hits=set(); prior_matches=0
    for shard in corpus['shards']:
        path=a.prior/shard['file']; meta_path=a.prior/shard['meta_file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=shard['sha256']: raise ValueError(f'Shard alterado: {path}')
        if hashlib.sha256(meta_path.read_bytes()).hexdigest()!=shard['meta_sha256']: raise ValueError(f'Metadados alterados: {meta_path}')
        with np.load(path) as data:
            ids=data['ids']; offsets=data['ex_offsets']; starts=data['loss_start']
            with meta_path.open(encoding='utf8') as f:
                for i,line in enumerate(f):
                    meta=json.loads(line); counts['examples']+=1
                    seq=ids[offsets[i]:offsets[i+1]].tolist(); start=int(starts[i]); prompt=seq[:start]; target=seq[start:]
                    if FIM_PRE not in prompt or FIM_SUF not in prompt or FIM_MID not in prompt:
                        counts['non_fim_skipped']+=1; continue
                    try:
                        b=prompt.index(FIM_PRE); s=prompt.index(FIM_SUF,b+1); m=prompt.index(FIM_MID,s+1)
                    except ValueError:
                        counts['non_fim_skipped']+=1; continue
                    if target and target[-1]==EOS: target=target[:-1]
                    r={'prefix':tok.decode(prompt[b+1:s],skip_special_tokens=False),'suffix':tok.decode(prompt[s+1:m],skip_special_tokens=False),'middle':tok.decode(target,skip_special_tokens=False)}
                    counts['fim_checked']+=1
                    matched=set()
                    for gram,content in dc.features(r)['grams'].items():
                        if content>=2 and gram in index: matched.update(index[gram])
                    if matched: prior_matches+=1; hits.update(matched)
    result={'sample_sha256':manifest['sha256'],'prior_manifest_sha256':hashlib.sha256((a.prior/'manifest.json').read_bytes()).hexdigest(),
            'counts':dict(counts),'prior_fim_with_hit':prior_matches,'sample_val_ids_with_hit':len(hits),'sample_val_ids':sorted(hits),
            'scope':'13-gramas de código com >=2 tokens de conteúdo; não testa exact, chave aproximada, MinHash ou blind; apenas validação de 260 itens.'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf8') as f: json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps({k:v for k,v in result.items() if k!='sample_val_ids'},ensure_ascii=False))
if __name__=='__main__': main()
