#!/usr/bin/env python3
"""Avaliação pareada no recorte Pro de validação; executa somente com GPU livre.

Usa o gerador do avaliador do professor. As métricas Mongo são estáticas;
execução real de predições não é realizada por este comando.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/teacher'))
import eval_teacher as ev

DATA=ROOT/'data/generated/pro-comparison-val-v1/val.jsonl'
MANIFEST=DATA.parent/'manifest.json'

def request_fingerprint(model, adapter, load, sample_sha):
    model=Path(model).resolve(strict=True)
    if not model.is_dir(): raise ValueError('modelo deve ser um diretório')
    files=sorted(p for p in model.iterdir() if p.is_file() and (p.suffix in ('.safetensors','.bin') or p.name in ('config.json','tokenizer.json')))
    if not files: raise ValueError('modelo sem arquivos verificáveis')
    return {'model':str(model),'files_sha256':{p.name:ev.sha256_file(p) for p in files},
            'adapter':str(Path(adapter).resolve(strict=True)) if adapter else None,
            'adapter_sha256':ev.sha256_file(Path(adapter)/'adapter_model.safetensors') if adapter else None,
            'load':load,'sample_sha256':sample_sha,'contract':'editor-context-v1','max_new_tokens':ev.MAX_NEW_TOKENS}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model'); p.add_argument('--adapter'); p.add_argument('--name');
    p.add_argument('--load',choices=['auto','bf16','nf4'],default='auto'); p.add_argument('--resume',action='store_true'); p.add_argument('--inspect',action='store_true'); a=p.parse_args()
    meta=json.loads(MANIFEST.read_text(encoding='utf-8')); raw=DATA.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=meta['sha256']: raise ValueError('hash da amostra difere')
    rows=[json.loads(line) for line in raw.splitlines()]
    if len(rows)!=260 or any(r['split']!='val' or r['type']!='fim' for r in rows): raise ValueError('amostra inválida')
    coverage=Counter(('meio' if r.get('task_scenario')=='mid_script_suffix' else 'contexto',r['scenario']) for r in rows)
    if len(coverage)!=26 or any(n!=10 for n in coverage.values()): raise ValueError('amostra não balanceada')
    if a.inspect:
        print(json.dumps({'examples':len(rows),'families':len(coverage),'per_family':10,'sha256':meta['sha256']})); return
    if not a.model or not a.name: p.error('--model e --name são necessários sem --inspect')
    if not a.name.replace('-','').replace('_','').isalnum(): p.error('nome de execução inválido')
    for r in rows: r['dialect']='console'; r.setdefault('target',r['middle'])
    ev.DATA_VERSION='pro-val'
    ev.read_rows=lambda contract_name,split=None: rows if split=='val' else []
    ev.dataset_paths=lambda contract_name: (DATA,MANIFEST)
    a.split='val'; a.n=len(rows); a.seed=20261028; a.contract='editor-context-v1'; a.ids_from=None
    out=ROOT/'Repos/KapiTeacher-Mongo-7B/reports/eval'/a.name
    if out.exists() and not a.resume: raise SystemExit('Diretório de avaliação já existe; use outro nome ou --resume.')
    request=request_fingerprint(a.model,a.adapter,a.load,meta['sha256'])
    request_file=out/'request.json'
    if a.resume and out.exists():
        if not request_file.is_file() or json.loads(request_file.read_text(encoding='utf-8'))!=request:
            raise SystemExit('Retomada recusada: modelo, adapter ou amostra diferem ou fingerprint está ausente.')
    out.mkdir(parents=True,exist_ok=True)
    if not request_file.exists(): request_file.write_text(json.dumps(request,indent=2)+'\n',encoding='utf-8')
    ev.phase_gen(a,out)
    pred=ev.read_jsonl(out/'predictions.jsonl'); index={r['id']:r for r in rows}; groups=defaultdict(list)
    if len(pred)!=len(rows) or len({r['id'] for r in pred})!=len(rows) or {r['id'] for r in pred}!=set(index):
        raise ValueError('Predições não correspondem à amostra congelada')
    for r in pred:
        source=index[r['id']]; label='meio' if source.get('task_scenario')=='mid_script_suffix' else 'contexto'
        r['bench_group']=f"{label}:{source['scenario']}"
        groups[r['bench_group']].append(r)
    # O comparador pareado existente agrupa por bench_group; preservar o conjunto e a ordem dos IDs.
    with (out/'predictions.jsonl').open('w',encoding='utf-8',newline='\n') as handle:
        for r in pred: handle.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary=ev.phase_report(a,out)
    extra={'warning':'val sintético; execução MongoDB e blind não foram avaliados por este comando.',
           'sample_sha256':meta['sha256'],'groups':{k:ev.aggregate(v) for k,v in sorted(groups.items())}}
    (out/'pro_groups.json').write_text(json.dumps(extra,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summary['headline'],'groups':len(groups)},ensure_ascii=False))

if __name__=='__main__': main()
