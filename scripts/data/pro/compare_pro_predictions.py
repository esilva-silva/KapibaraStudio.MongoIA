#!/usr/bin/env python3
"""Compara dois resultados Pro no mesmo recorte de validação, recusando IDs ausentes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/eval'))
import compare_models

DATA=ROOT/'data/generated/pro-comparison-val-v1/val.jsonl'
MANIFEST=DATA.parent/'manifest.json'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--baseline',required=True,type=Path); p.add_argument('--candidate',required=True,type=Path); p.add_argument('--out',required=True,type=Path); p.add_argument('--resamples',type=int,default=10000); p.add_argument('--chat',action='store_true'); a=p.parse_args()
    if a.out.exists(): raise SystemExit('Saída já existe; escolha um nome novo')
    data=ROOT/'data/generated/pro-chat-eval-val-v1/val.jsonl' if a.chat else DATA
    manifest=json.loads((data.parent/'manifest.json').read_text(encoding='utf-8'))
    raw=data.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['sha256']: raise ValueError('Hash da amostra difere do manifesto')
    refs={r['id']:r for r in (json.loads(x) for x in raw.splitlines())}
    if len(refs)!=(120 if a.chat else 260): raise ValueError('Amostra inválida')
    requests=[]
    for folder in (a.baseline,a.candidate):
        meta=json.loads((folder/'gen_meta.json').read_text(encoding='utf-8'))
        if meta['dataset']['sha256']!=manifest['sha256']: raise ValueError(f'Amostra divergente: {folder}')
        request=json.loads((folder/'request.json').read_text(encoding='utf-8'))
        if request['sample_sha256']!=manifest['sha256']: raise ValueError(f'Fingerprint da amostra diverge: {folder}')
        requests.append(request)
        rows=compare_models.load(folder/'predictions.jsonl')
        if set(rows)!=set(refs): raise ValueError(f'IDs de predição incompletos ou extras: {folder}')
        for ident,r in rows.items():
            src=refs[ident]; group=('chat' if a.chat else 'meio' if src.get('task_scenario')=='mid_script_suffix' else 'contexto')+':'+str(src['scenario'])
            if r.get('bench_group')!=group: raise ValueError(f'Grupo divergente: {ident}')
    comparable=('contract','max_new_tokens','load','render_config_sha256','render_source_sha256')
    if any(requests[0].get(k)!=requests[1].get(k) for k in comparable):
        raise ValueError('Parâmetros de geração ou contrato divergem')
    if a.chat:
        evidence=[folder/'exec_validation.json' for folder in (a.baseline,a.candidate)]
        if evidence[0].exists()!=evidence[1].exists(): raise ValueError('Só um dos lados tem execução em fixtures')
        if evidence[0].exists():
            execution=[json.loads(path.read_text(encoding='utf8')) for path in evidence]
            if not execution[0].get('fixture_sha256') or execution[0]['fixture_sha256']!=execution[1].get('fixture_sha256'):
                raise ValueError('As fixtures não são idênticas')
    result=compare_models.run(a.baseline/'predictions.jsonl',a.candidate/'predictions.jsonl',min_group_n=20 if a.chat else 10,n_resamples=a.resamples)
    result['pro_sample_sha256']=manifest['sha256']
    result['scope_note']='Validação sintética, não blind; MongoDB estático; grupos pequenos têm pouca precisão.'
    a.out.mkdir(parents=True)
    (a.out/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (a.out/'comparison.md').write_text(compare_models.to_markdown(result)+'\n',encoding='utf-8')
    print(json.dumps({'n':result['n_aligned'],'groups':len(result['groups']),'sample_sha256':manifest['sha256']}))
if __name__=='__main__': main()
