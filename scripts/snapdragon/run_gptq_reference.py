"""Manual GPTQ artifact reference on the selected RX, without changing saved weights."""
import argparse,hashlib,json,time,traceback
from pathlib import Path

def run(model_path,corpus,out,device,expected_gpu,sample_id=None,quality_only=False):
    report={'quality_only':quality_only,'timing_scope':'disabled for quality-only runs alongside CPU preparation' if quality_only else 'instrumented manual control; not qualification','status':'failed','backend':'gptqmodel-torch','qualified':False,'proxy':True,'model':str(model_path),'items':[]}
    try:
        import torch
        from run_sample import verify_gpu,load_corpus,HFRunner,stop_ids,memory
        report['gpu']=verify_gpu(torch,device,expected_gpu)
        if torch.cuda.device_count()!=1:raise RuntimeError('Only selected RX must be visible')
        torch.cuda.set_device(0);torch.manual_seed(20261026);torch.cuda.reset_peak_memory_stats()
        torch.compiler.set_stance('force_eager')
        report['execution_mode']='Torch eager; no Triton compilation'
        report['device_map']={'':device}
        from gptqmodel import GPTQModel
        from gptqmodel.utils import BACKEND
        from transformers import AutoTokenizer
        started=time.monotonic()
        wrapped=GPTQModel.load(str(model_path),backend=BACKEND.TORCH,device=device,device_map={'':device},torch_dtype=torch.float16,trust_remote_code=False,local_files_only=True)
        model=wrapped.model.to(device).eval()
        report['load_seconds']=time.monotonic()-started
        tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True,trust_remote_code=False)
        runner=HFRunner.__new__(HFRunner);runner.torch=torch;runner.device=device;runner.model=model;runner.tok=tokenizer;runner.stops=stop_ids(lambda text:tokenizer.encode(text,add_special_tokens=False))
        items=load_corpus(corpus);report['corpus_sha256']=hashlib.sha256(corpus.read_bytes()).hexdigest();report['input_corpus_count']=len(items)
        selected=[x for x in items if x['id']==sample_id] if sample_id else items
        if not selected:raise ValueError('No matching control sample')
        report['provider_evidence']={'requested':device,'gpu':report['gpu'],'tensor_devices':sorted({str(p.device) for p in model.parameters()}|{str(b.device) for b in model.buffers()}),'dtype':str(next(model.parameters()).dtype),'backend':'TorchQuantLinear'}
        for item in selected:
            row={'id':item['id'],'reference':item.get('reference',item.get('target')),'suffix':item.get('suffix','')};row.update(runner.generate(item))
            if quality_only:
                for key in ('ttft_ms','elapsed_s','decode_tokens_per_second'):row[key]=None
            report['items'].append(row)
            if len(report['items'])%10==0:
                out.with_name(out.name+'.progress.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print('Processed',len(report['items']),flush=True)
        report.update(status='control_completed' if sample_id else 'completed',peak_gpu_bytes=torch.cuda.max_memory_allocated(),memory_after=memory())
    except Exception as error:report.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','corpus','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--expected-gpu',default='RX 7800 XT');p.add_argument('--sample-id');p.add_argument('--quality-only',action='store_true');a=p.parse_args();r=run(a.model.resolve(),a.corpus.resolve(),a.out.resolve(),a.device,a.expected_gpu,a.sample_id,a.quality_only);print(json.dumps(r,indent=2));raise SystemExit(1 if r['status']=='failed' else 0)
