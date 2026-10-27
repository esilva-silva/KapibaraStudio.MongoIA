from pathlib import Path
import json,sys,time,traceback,gc
sys.path.insert(0,str(Path('scripts/snapdragon').resolve()))
import onnxruntime_genai as og
from run_sample import load_corpus,stop_ids,generation_budget
import argparse
p=argparse.ArgumentParser(description='Manual CPU control for Nano static pipeline; no HTP evidence')
p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--corpus',type=Path,default=Path(__file__).resolve().parents[2]/'data/calib/snapdragon-20261026/evaluation.jsonl')
p.add_argument('--sample-id',default='fim-legal.cobalt_leg.billing-000714-stage-000279')
args=p.parse_args();out=args.model.resolve()
metadata=json.loads((out/'genai_config.json').read_text(encoding='utf-8'))
window=metadata['model']['decoder']['sliding_window']['window_size'];capacity=metadata['model']['context_length']
report={'qualified':False,'scope':'CPU generation control only; not HTP compilation or NPU qualification','status':'starting','versions':{'genai':og.__version__}}
try:
    cfg=og.Config(str(out));cfg.clear_providers();started=time.monotonic();model=og.Model(cfg);report['load_seconds']=time.monotonic()-started
    tok=og.Tokenizer(model);items=load_corpus(args.corpus);item=next(x for x in items if x['id']==args.sample_id)
    ids=list(tok.encode(item['prompt_text']));assert ids==item['prompt_ids'];report.update(sample_id=item['id'],prompt_tokens=len(ids));padded,maximum=generation_budget(len(ids),capacity,window);params=og.GeneratorParams(model);params.set_search_options(max_length=maximum,do_sample=False)
    g=og.Generator(model,params);g.append_tokens(ids);tokens=[];stops=stop_ids(tok.encode);before=len(g.get_sequence(0));report['sequence_length_after_append']=before;report['max_length_requested']=maximum;started=time.monotonic()
    for _ in range(32):
        if g.is_done():break
        g.generate_next_token();seq=list(g.get_sequence(0))
        if len(seq)>before:
            tokens.extend(int(v) for v in seq[before:]);before=len(seq)
            if tokens[-1] in stops:break
    report.update(status='control_completed',generated_ids=tokens,generated_text=tok.decode(tokens),wall_generation_seconds=time.monotonic()-started,final_sequence_length=len(g.get_sequence(0)),generator_is_done=g.is_done())
    del g,params,tok,model;gc.collect()
except Exception as e:report.update(status='failed',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
path=args.out;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(report,indent=2))

raise SystemExit(1 if report['status']=='failed' else 0)
