#!/usr/bin/env python3
"""Compara chat de código em validação sintética; sintaxe/validação Mongo são estáticas, sem execução de predições."""
import argparse,hashlib,json,sys,time
from collections import Counter
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/teacher'),str(ROOT/'scripts/data'),str(ROOT/'scripts/data/pro')]
from _common import Generator,gpu_lock,load_hf,sha256_file
from build_teacher_corpus import render_chat_prompt
from eval_pro_slice import request_fingerprint
import decontaminate as dc
DATA=ROOT/'data/generated/pro-chat-eval-val-v1/val.jsonl'; MANIFEST=DATA.parent/'manifest.json'

def main():
 p=argparse.ArgumentParser(); p.add_argument('--model'); p.add_argument('--adapter'); p.add_argument('--name'); p.add_argument('--load',choices=['auto','bf16','nf4'],default='nf4'); p.add_argument('--resume',action='store_true'); p.add_argument('--inspect',action='store_true'); a=p.parse_args()
 manifest=json.loads(MANIFEST.read_text(encoding='utf8')); raw=DATA.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=manifest['sha256']: raise ValueError('hash da amostra diverge')
 rows=[json.loads(x) for x in raw.splitlines()]
 if len(rows)!=120 or Counter(r['scenario'] for r in rows)!={i:20 for i in range(6)} or any(r['split']!='val' or r['type']!='chat' for r in rows): raise ValueError('amostra inválida')
 if a.inspect: print(json.dumps({'examples':len(rows),'families':6,'sha256':manifest['sha256']})); return
 if not a.model or not a.name or not a.name.replace('-','').replace('_','').isalnum(): p.error('informe --model e --name válido')
 if a.load=='auto': p.error('selecione explicitamente bf16 ou nf4 para garantir comparação igual')
 out=ROOT/'Repos/KapiTeacher-Mongo-7B/reports/eval'/a.name
 if out.exists() and not a.resume: raise SystemExit('Saída já existe; use outro nome ou --resume')
 req=request_fingerprint(a.model,a.adapter,a.load,manifest['sha256']); req['contract']='local-agent-fim-v0'; req['max_new_tokens']=256
 config_path=ROOT/'Repos/KapiTeacher-Mongo-7B/configs/corpus_pro_context_v2.yaml'
 req['render_config_sha256']=sha256_file(config_path)
 req['render_source_sha256']=sha256_file(ROOT/'scripts/teacher/build_teacher_corpus.py')
 request=out/'request.json'
 if a.resume and out.exists() and (not request.is_file() or json.loads(request.read_text(encoding='utf8'))!=req): raise SystemExit('Fingerprint difere; retomada recusada')
 out.mkdir(parents=True,exist_ok=True)
 if not request.exists(): request.write_text(json.dumps(req,indent=2)+'\n',encoding='utf8')
 predictions=out/'predictions.jsonl'; done={}
 if a.resume and predictions.exists():
  for line in predictions.read_text(encoding='utf8').splitlines():
   r=json.loads(line)
   if r['id'] in done: raise ValueError('ID duplicado')
   done[r['id']]=r
 known={r['id'] for r in rows}
 if set(done)-known: raise ValueError('IDs fora da amostra')
 cfg=yaml.safe_load(config_path.read_text(encoding='utf8'))
 from compact import syntax
 from compact.mongo_validate import validate
 with gpu_lock('eval_pro_chat:'+a.name):
  model,tokenizer,device,gpu,load_seconds=load_hf(Path(a.model).resolve(),a.load)
  if a.adapter:
   from peft import PeftModel
   model=PeftModel.from_pretrained(model,str(Path(a.adapter).resolve()),torch_device=str(device)); model.eval()
  gen=Generator(model,tokenizer,device)
  with predictions.open('a',encoding='utf8',newline='\n') as f:
   for r in rows:
    if r['id'] in done: continue
    prompt=render_chat_prompt(r,cfg); ids=tokenizer(prompt,add_special_tokens=False)['input_ids']
    result=gen.generate(ids,256,frozenset({151643}),'')
    text=gen.decode([x for x in result['tokens'] if x!=151643]).strip()
    item={'id':r['id'],'bench_group':f"chat:{r['scenario']}",'prediction':text,'reference':r['output'],
          'generated_tokens':len(result['tokens']),'stop':result['stop'],'ttft_ms':round(result['ttft']*1000,1)}
    done[r['id']]=item; f.write(json.dumps(item,ensure_ascii=False)+'\n'); f.flush()
    if len(done)%20==0: print(f"{len(done)}/{len(rows)}",flush=True)
  del model
 jobs=[]
 for r in rows:
  for label,text in (('pred',done[r['id']]['prediction']),('ref',r['output'])): jobs.append((r['id']+'|'+label,text,syntax.kind_for('console')))
 checks=syntax.check_many(jobs)
 for r in rows:
  item=done[r['id']]; pred=item['prediction']; syntactic=checks[r['id']+'|pred'][0]
  v=validate(pred,'console') if syntactic else None
  item['metrics']={'code_equal_normalized':dc.code_tokens(pred,literals='keep')==dc.code_tokens(r['output'],literals='keep'),
                   'syntax_valid':bool(syntactic),'mongo_static_valid':v.valid if v and v.validatable else (False if not syntactic else None),
                   'empty':not bool(pred)}
  item['reference_syntax_valid']=bool(checks[r['id']+'|ref'][0])
 with predictions.open('w',encoding='utf8',newline='\n') as f:
  for r in rows: f.write(json.dumps(done[r['id']],ensure_ascii=False)+'\n')
 (out/'gen_meta.json').write_text(json.dumps({'dataset':{'sha256':manifest['sha256']},'model':req,'examples':len(rows),
   'limits':'Apenas sintaxe e verificação estática; igualdade normalizada não é equivalência semântica; não avalia blind.',
   'created_utc':time.strftime('%FT%TZ',time.gmtime())},indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 print(json.dumps({'examples':len(rows),'syntax_valid':sum(x['metrics']['syntax_valid'] for x in done.values())/len(rows),
                   'code_equal_normalized':sum(x['metrics']['code_equal_normalized'] for x in done.values())/len(rows)}))
if __name__=='__main__': main()
