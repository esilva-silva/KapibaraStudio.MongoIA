"""Isolated Olive GPTQ job on an explicitly selected RX GPU; no QNN activation pass here."""
import argparse,functools,hashlib,importlib.machinery,importlib.metadata,json,struct,sys,time,traceback,types
from pathlib import Path

def validate_calibration_batches(dataset, ledger):
    if (ledger.get('policy') != 'kapi_gptq_join_tail_v1' or ledger.get('selected_rows') != 256
            or ledger.get('all_prompt_tokens_covered') is not True or ledger.get('batch_size') != 1
            or ledger.get('padding') != 'none' or ledger.get('add_special_tokens') is not False
            or ledger.get('shuffle') is not False):
        raise ValueError('Invalid GPTQ calibration ledger policy')
    batches=ledger.get('batches') or []
    if len(dataset)!=30 or len(batches)!=30 or [x.get('tokens') for x in batches] != [2048]*29+[1881]:
        raise ValueError('Expected 29 full blocks and one 1881-token tail')
    token_stream=hashlib.sha256(); observed=[]
    for index,(item,expected) in enumerate(zip(dataset,batches)):
        def flatten(key):
            value=item[key]
            if hasattr(value,'detach'): value=value.detach().cpu()
            rows=value.tolist() if hasattr(value,'tolist') else value
            if not isinstance(rows,list) or len(rows)!=1 or not isinstance(rows[0],list):
                raise ValueError('Expected a single batch dimension')
            if any(type(x) is not int for x in rows[0]): raise ValueError('Expected integer IDs and masks')
            return rows[0]
        ids=flatten('input_ids');mask=flatten('attention_mask')
        if len(ids)!=expected['tokens'] or mask != [1]*len(ids):
            raise ValueError('Batch token length or attention mask mismatch at '+str(index))
        encoded=b''.join(struct.pack('<q',x) for x in ids)
        digest=hashlib.sha256(encoded).hexdigest()
        if expected.get('index')!=index or digest!=expected.get('token_ids_sha256'):
            raise ValueError('Batch token hash mismatch at '+str(index))
        token_stream.update(encoded)
        mask_hash=hashlib.sha256(b''.join(struct.pack('<q',x) for x in mask)).hexdigest()
        observed.append({'index':index,'tokens':len(ids),'token_ids_sha256':digest,'attention_mask_sha256':mask_hash})
    stream_hash=token_stream.hexdigest()
    if stream_hash!=ledger.get('stream_token_ids_sha256') or sum(x['tokens'] for x in observed)!=ledger.get('consumed_tokens'):
        raise ValueError('Calibration stream hash or count mismatch')
    return {'matched':True,'batch_count':len(observed),'token_count':sum(x['tokens'] for x in observed),
            'stream_token_ids_sha256':stream_hash,'batches':observed}


def run(config_path,report_path,calibration_preflight=None):
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
        ledger=None
        if calibration_preflight:
            ledger_path=Path(calibration_preflight).resolve(strict=True)
            ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
            report['runtime_calibration_audit']={'preflight_sha256':hashlib.sha256(ledger_path.read_bytes()).hexdigest()}
            original_prepare=BaseGPTQModel.prepare_dataset
            @functools.wraps(original_prepare)
            def prepare_dataset(model,*args,**kwargs):
                result=original_prepare(model,*args,**kwargs)
                report['runtime_calibration_audit']['after_prepare_dataset']=validate_calibration_batches(result,ledger)
                save()
                return result
            BaseGPTQModel.prepare_dataset=prepare_dataset
        original_quantize=BaseGPTQModel.quantize
        @functools.wraps(original_quantize)
        def quantize(model,dataset,*args,**kwargs):
            kwargs['backend']=BACKEND.TORCH
            if ledger is not None:
                report['runtime_calibration_audit']['quantize_entry']=validate_calibration_batches(dataset,ledger)
                save()
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
        if ledger is not None and not all(report['runtime_calibration_audit'].get(scope,{}).get('matched') is True
                                          for scope in ('quantize_entry','after_prepare_dataset')):
            raise RuntimeError('Both runtime calibration audit hooks must complete')
        report['workflow_result_nonempty']=bool(result)
        report['status']='workflow_returned_artifact_validation_pending' if result else 'failed'
        report['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    report['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
    save();return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--calibration-preflight',type=Path);a=p.parse_args()
    r=run(a.config.resolve(),a.report.resolve(),a.calibration_preflight);print(json.dumps(r,indent=2));raise SystemExit(1 if r['status']=='failed' else 0)
