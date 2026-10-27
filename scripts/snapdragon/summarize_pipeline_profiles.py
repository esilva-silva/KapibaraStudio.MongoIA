"""Summarize per-stage ORT profiling evidence for a mixed GenAI pipeline."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def summarize(report_path):
    report_path=Path(report_path).resolve(strict=True)
    report=json.loads(report_path.read_text(encoding='utf-8'))
    evidence=report['provider_evidence']; stages=evidence['intended_stages']
    if len(stages)!=4 or [s['name'] for s in stages] != ['embeddings','context','iterator','lm_head']:
        raise ValueError('Expected embeddings, context, iterator, lm_head stage order')
    summary={'schema':'kapi-pipeline-profile-summary-v1','report':str(report_path),'report_sha256':sha(report_path),
             'stages':[], 'qnn_profile_files':evidence.get('qnn_profile_files', []), 'errors':[]}
    assigned={}
    for name in evidence['profile_files']:
        path=Path(name).resolve(strict=True)
        match=re.search(r'-(\d+)_\d{4}-\d{2}-\d{2}_',path.name)
        if not match: raise ValueError('Cannot identify stage index: '+str(path))
        index=int(match.group(1))
        if index>=len(stages) or index in assigned: raise ValueError('Unexpected or duplicate profile stage')
        assigned[index]=path
    for index, stage in enumerate(stages):
        if index not in assigned:
            summary['errors'].append('Missing profile for '+stage['name']); continue
        path=assigned[index]; events=json.loads(path.read_text(encoding='utf-8'))
        counts=Counter(); ops={}
        for event in events:
            if event.get('cat') != 'Node': continue
            args=event.get('args') or {}; provider=args.get('provider')
            if not provider: continue
            counts[provider]+=1
            key=(provider,args.get('op_name','unknown'),event.get('name','unknown'))
            if key not in ops:
                ops[key]={'provider':provider,'op_name':key[1],'node_event':key[2], 'count':0,
                    'input_type_shape':args.get('input_type_shape'),'output_type_shape':args.get('output_type_shape')}
            ops[key]['count']+=1
        expected='QNNExecutionProvider' if 'qnn' in stage['providers'] else 'CPUExecutionProvider'
        valid=counts.get(expected,0)>0 and (expected!='QNNExecutionProvider' or counts.get('CPUExecutionProvider',0)==0)
        if not valid: summary['errors'].append('Unexpected execution evidence for '+stage['name'])
        summary['stages'].append({'index':index,'name':stage['name'],'intended_providers':stage['providers'],
            'profile':str(path),'profile_sha256':sha(path),'kernel_events_by_provider':dict(counts),
            'valid':valid,'nodes':list(ops.values())})
    summary['status']='verified' if not summary['errors'] else 'failed'
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out)
    if out.exists(): p.error('Output must not already exist')
    try:
        result=summarize(a.report);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(json.dumps({'status':result['status'],'errors':result['errors']},indent=2))
        return 0 if result['status']=='verified' else 1
    except Exception as exc:p.exit(1,f'Profile summary failed: {type(exc).__name__}: {exc}\n')

if __name__=='__main__':raise SystemExit(main())
