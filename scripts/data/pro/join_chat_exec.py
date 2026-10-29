#!/usr/bin/env python3
"""Junta pontuação MongoDB de leitura às predições de chat sem alterar originais."""
import argparse,hashlib,json
from pathlib import Path

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read_rows(path):
 rows={}
 for line in path.read_text(encoding='utf8').splitlines():
  r=json.loads(line)
  if r['id'] in rows: raise ValueError('ID duplicado')
  rows[r['id']]=r
 return rows

def main():
 p=argparse.ArgumentParser(); p.add_argument('--evaluation',required=True,type=Path); p.add_argument('--scores',required=True,type=Path); p.add_argument('--out',required=True,type=Path); a=p.parse_args()
 if a.out.exists(): raise SystemExit('Diretório de saída já existe')
 pred=a.evaluation/'predictions.jsonl'; meta=a.evaluation/'gen_meta.json'; request=a.evaluation/'request.json'; summ=a.scores.with_suffix(a.scores.suffix+'.summary.json')
 summary=json.loads(summ.read_text(encoding='utf8')); gen=json.loads(meta.read_text(encoding='utf8')); req=json.loads(request.read_text(encoding='utf8'))
 if summary['scores_sha256']!=sha(a.scores) or summary['predictions_sha256']!=sha(pred): raise ValueError('Hashes de scores ou predições divergem')
 if summary['sample_sha256']!=gen['dataset']['sha256'] or summary['sample_sha256']!=req['sample_sha256']: raise ValueError('Hashes de amostra divergem')
 predictions=read_rows(pred); scores=read_rows(a.scores)
 if len(predictions)!=120 or set(predictions)!=set(scores): raise ValueError('IDs não correspondem')
 for ident,r in predictions.items():
  s=scores[ident]
  if r['bench_group']!=s['bench_group'] or not isinstance(s['exec_equiv'],bool): raise ValueError('Grupo ou resultado inválido')
  r['metrics']['exec_equiv_fixtures']=s['exec_equiv']; r['exec_stage']=s['stage']
 a.out.mkdir(parents=True)
 for path in (meta,request): (a.out/path.name).write_bytes(path.read_bytes())
 with (a.out/'predictions.jsonl').open('x',encoding='utf8',newline='\n') as f:
  for r in predictions.values(): f.write(json.dumps(r,ensure_ascii=False)+'\n')
 (a.out/'exec_validation.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 print(json.dumps({'examples':len(predictions),'equivalent':sum(x['metrics']['exec_equiv_fixtures'] for x in predictions.values())}))
if __name__=='__main__': main()
