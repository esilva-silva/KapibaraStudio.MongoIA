#!/usr/bin/env python3
"""Compara planos de chat em fixtures MongoDB existentes, somente leitura.

NÃO executa JavaScript gerado: parseia o subconjunto permitido e executa apenas
find/findOne/countDocuments no banco sintético fixado. É uma aproximação por
fixtures, não substitui o blind nem prova equivalência geral.
"""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
from bson import json_util
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/data'))
import chat_validate as cv
import validate_snippets as vs

DATA=ROOT/'data/generated/pro-chat-eval-val-v1/val.jsonl'
MANIFEST=DATA.parent/'manifest.json'
DB='kapi_fx_pro_20261028a93c7212_val'
COLLECTION='iniciativas_demo'
# Calculado a partir da regra determinística de validate_mongo.js, não dos resultados do modelo.
EXPECTED_FIXTURE_SHA256='1050db95a0309227e6b48b8abcefccab50c462cfb0101b6c22c3463797ee7c75'


def parse(text):
    plan, meta=cv.parse_console(text)
    if meta.get('collection')!=COLLECTION or meta.get('database') is not None or meta.get('connection') is not None:
        raise ValueError('coleção/conexão fora do contexto autorizado')
    if plan.get('op') not in {'find','findOne','countDocuments'}: raise ValueError('operação fora da allowlist')
    for check in (cv.check_ops(plan),cv.check_structure(plan)):
        if check: raise ValueError(str(check))
    vs.scan_forbidden(plan)
    return plan


def run(coll,plan):
    op=plan['op']; filt=vs.from_ej(plan.get('filter') or {})
    if op=='countDocuments': return ('count',coll.count_documents(filt,maxTimeMS=2000))
    cur=coll.find(filt,vs.from_ej(plan['projection']) if plan.get('projection') else None,max_time_ms=2000)
    if plan.get('sort'): cur=cur.sort(list(plan['sort'].items()))
    if plan.get('skip'): cur=cur.skip(int(plan['skip']))
    lim=1 if op=='findOne' else int(plan.get('limit') or 1001)
    if lim<1 or lim>1001: raise ValueError('limite fora do orçamento')
    docs=list(cur.limit(lim))
    if len(docs)>1000: raise ValueError('resultado excede orçamento')
    serialized=[json_util.dumps(x,sort_keys=True) for x in docs]
    if op=='findOne': return ('one',serialized[0] if serialized else None)
    if not plan.get('sort'): serialized.sort()
    return ('list',serialized)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',required=True,type=Path); ap.add_argument('--out',required=True,type=Path); args=ap.parse_args()
    summary_path=args.out.with_suffix(args.out.suffix+'.summary.json')
    if args.out.exists() or summary_path.exists(): raise SystemExit('Saída já existe')
    m=json.loads(MANIFEST.read_text(encoding='utf8')); raw=DATA.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=m['sha256']: raise ValueError('hash da amostra diverge')
    rows={r['id']:r for r in (json.loads(x) for x in raw.splitlines())}
    preds={}
    for line in args.predictions.read_text(encoding='utf8').splitlines():
        r=json.loads(line)
        if r['id'] in preds: raise ValueError('ID duplicado')
        preds[r['id']]=r
    if len(rows)!=120 or set(preds)!=set(rows): raise ValueError('predições não cobrem a amostra')
    client,guard=vs.make_client('mongodb://127.0.0.1:27017')
    client.admin.command('ping')
    if DB not in client.list_database_names(): raise SystemExit('Fixture ausente; não criar automaticamente')
    coll=client[DB][COLLECTION]
    if coll.count_documents({},maxTimeMS=2000)!=1000: raise SystemExit('Fixture inesperada; não avaliar')
    fixture_hasher=hashlib.sha256()
    for doc in coll.find({},max_time_ms=2000).sort([('_id',1)]):
        fixture_hasher.update((json_util.dumps(doc,sort_keys=True)+'\n').encode('utf8'))
    fixture_sha256=fixture_hasher.hexdigest()
    if fixture_sha256!=EXPECTED_FIXTURE_SHA256: raise SystemExit('Conteúdo da fixture diverge do corpus sintético esperado')
    refs=[rows[i]['output'] for i in rows]; predictions=[preds[i]['prediction'] for i in rows]
    if not cv.node_available(): raise SystemExit('Node necessário para checar sintaxe')
    syntaxes=cv.node_syntax(refs+predictions); out=[]; stats=Counter()
    for j,(ident,r) in enumerate(rows.items()):
        referr,prederr=syntaxes[j],syntaxes[j+len(rows)]
        if referr: raise ValueError(f'referência inválida: {ident}: {referr}')
        ref=parse(r['output']); expected=run(coll,ref)
        entry={'id':ident,'bench_group':f"chat:{r['scenario']}",'reference_op':ref['op'],'exec_equiv':False,'stage':'ok'}
        try:
            if prederr: raise ValueError('sintaxe: '+prederr)
            pred=parse(preds[ident]['prediction'])
            # Respeitar tipo de retorno, ordenação e limites solicitados; depois comparar os resultados.
            if pred['op']!=ref['op'] or pred.get('sort')!=ref.get('sort') or pred.get('limit')!=ref.get('limit'):
                raise ValueError('operação, ordenação ou limite divergente')
            entry['exec_equiv']=run(coll,pred)==expected
            if not entry['exec_equiv']: entry['stage']='resultado_divergente'
        except Exception as e:
            entry['stage']='recusado'; entry['error']=f'{type(e).__name__}: {e}'[:300]
        stats[entry['stage']]+=1; out.append(entry)
    if guard.violations: raise RuntimeError('Comandos fora da allowlist observados: '+str(guard.violations))
    client.close()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf8') as f:
        for r in out: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary={'examples':len(out),'equivalent':sum(r['exec_equiv'] for r in out),'stages':dict(stats),
             'fixture':DB,'fixture_sha256':fixture_sha256,'reads':guard.reads,'forbidden_commands':len(guard.violations),
             'sample_sha256':m['sha256'],'predictions_sha256':hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
             'scores_sha256':hashlib.sha256(args.out.read_bytes()).hexdigest(),
             'scope':'equivalência somente nessas fixtures; predições parseadas como planos permitidos; nenhuma execução de JavaScript'}
    with summary_path.open('x',encoding='utf8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__': main()
