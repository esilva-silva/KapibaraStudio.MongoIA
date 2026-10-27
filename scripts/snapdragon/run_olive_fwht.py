"""Isolated Olive GPTQ job on an explicitly selected RX GPU; no QNN activation pass here."""
import argparse,functools,hashlib,importlib.machinery,importlib.metadata,json,sys,time,traceback,types
from pathlib import Path

def run(config_path,report_path):
    import torch
    from run_sample import verify_gpu
    import torch_fwht
    report={'status':'starting','phase':'GPTQ preparation only','qualified':False,'qnn_activation_calibration':'not run',
            'adaptations':['pure Torch FWHT process-local module alias','GPTQ4.2 global binding','force BACKEND.TORCH'],
            'stages':[],'fwht_calls':0}
    started=time.monotonic()
    def save():
        report['wall_seconds']=time.monotonic()-started
        report_path.parent.mkdir(parents=True,exist_ok=True)
        report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    try:
        config=json.loads(config_path.read_text(encoding='utf-8'))
        if set(config['passes'])!={'cs','g'}:raise ValueError('This wrapper is bounded to CaptureSplitInfo and GPTQ; do not use it for QNN activation calibration')
        if config['passes']['g']['device']!='cuda' or config['passes']['g']['rotation']!='hadamard':raise ValueError('Expected explicit GPU Hadamard recipe')
        report['gpu']=verify_gpu(torch,'cuda:0','RX 7800 XT')
        if torch.cuda.device_count()!=1:raise RuntimeError('Only the RX 7800 XT may be visible')
        torch.cuda.set_device(0);torch.manual_seed(20261026);torch.cuda.reset_peak_memory_stats()
        report['versions']={name:importlib.metadata.version(name) for name in ('torch','transformers','gptqmodel','olive-ai','onnxruntime-genai','onnxruntime','datasets','optimum')}
        report['config_sha256']=hashlib.sha256(config_path.read_bytes()).hexdigest()
        report['fwht_source_sha256']=hashlib.sha256(Path(torch_fwht.__file__).read_bytes()).hexdigest()
        report['wrapper_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        adapter=types.ModuleType('fast_hadamard_transform')
        adapter.__spec__=importlib.machinery.ModuleSpec(adapter.__name__,loader=None,origin=torch_fwht.__file__)
        def transform(x,scale=1):
            if x.device.type!='cuda' or x.device.index!=0:raise RuntimeError('Hadamard input must use selected RX')
            report['fwht_calls']+=1
            return torch_fwht.hadamard_transform(x,scale)
        adapter.hadamard_transform=transform
        # Exercise the registered entrypoint before loading a real model.
        sanity=adapter.hadamard_transform(torch.tensor([1.,-1.,1.,-1.],device='cuda:0'),0.5)
        torch.testing.assert_close(sanity,torch.tensor([0.,2.,0.,0.],device='cuda:0'),rtol=0,atol=0)
        report['adapter_entrypoint_preflight']='passed'
        if 'fast_hadamard_transform' in sys.modules:raise RuntimeError('Ambiguous FHT module in current process')
        sys.modules[adapter.__name__]=adapter
        from gptqmodel.models.auto import BaseGPTQModel
        from gptqmodel.quantization.rotation import hadamard_utils
        from gptqmodel.utils import BACKEND
        hadamard_utils.fast_hadamard_transform=adapter
        original_quantize=BaseGPTQModel.quantize
        @functools.wraps(original_quantize)
        def quantize(model,dataset,*args,**kwargs):
            kwargs['backend']=BACKEND.TORCH
            lengths=[int(item['input_ids'].shape[-1]) for item in dataset]
            report['effective_gptq_calibration']={'batches':len(dataset),'min_tokens':min(lengths),'max_tokens':max(lengths),'backend':'Torch','device':str(model.quantize_config.device),'rotation':model.quantize_config.rotation}
            report['status']='quantizing';save()
            return original_quantize(model,dataset,*args,**kwargs)
        BaseGPTQModel.quantize=quantize
        from olive.passes import Pass
        original_pass_run=Pass.run
        @functools.wraps(original_pass_run)
        def pass_run(current,*args,**kwargs):
            event={'pass':type(current).__name__,'status':'running','start_wall_seconds':time.monotonic()-started};report['stages'].append(event);save()
            try:
                value=original_pass_run(current,*args,**kwargs)
                event.update(status='completed',seconds=time.monotonic()-started-event['start_wall_seconds']);save();return value
            except Exception as error:
                event.update(status='failed',error=f'{type(error).__name__}: {error}');save();raise
        Pass.run=pass_run
        report['status']='running';save()
        from olive.workflows import run as olive_run
        result=olive_run(config,tempdir=str(config_path.parent/'tmp'))
        report['workflow_result_nonempty']=bool(result)
        report['status']='workflow_returned_artifact_validation_pending' if result else 'failed'
        report['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    report['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
    save();return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    r=run(a.config.resolve(),a.report.resolve());print(json.dumps(r,indent=2));raise SystemExit(1 if r['status']=='failed' else 0)
