"""Standalone random-model probe with an explicit process-local pure-Torch FWHT adapter. Run in a fresh process only; no package files are modified."""
import argparse,hashlib,importlib.machinery,importlib.util,json,sys,time,traceback,types
from pathlib import Path

def run(source,calibration,device,expected_gpu):
    report={'status':'failed','scope':'random tiny model only; experimental adapters, no user-weight quantization',
            'adaptations':['process-local fast_hadamard_transform alias to torch_fwht','GPTQ4.2 global-name binding in hadamard_utils','BACKEND.TORCH explicitly selected','GPTQ DEVICE enum cuda with current device0 and only RX visible'],'fwht_calls':[]}
    start=time.monotonic()
    try:
        import torch
        from run_sample import verify_gpu
        report['gpu']=verify_gpu(torch,device,expected_gpu)
        if torch.cuda.device_count()!=1:raise RuntimeError('Only the selected physical GPU may be visible')
        torch.cuda.set_device(0)
        if importlib.util.find_spec('fast_hadamard_transform') is not None:raise RuntimeError('Probe requires no installed native FHT to avoid ambiguity')
        import torch_fwht
        report['wrapper_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report['fwht_source_sha256']=hashlib.sha256(Path(torch_fwht.__file__).read_bytes()).hexdigest()
        adapter=types.ModuleType('fast_hadamard_transform')
        adapter.__spec__=importlib.machinery.ModuleSpec(adapter.__name__,loader=None,origin=torch_fwht.__file__)
        def transform(x,scale=1):
            report['fwht_calls'].append({'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device)})
            if x.device.type!='cuda':raise RuntimeError('Expected selected HIP GPU for Hadamard rotation')
            return torch_fwht.hadamard_transform(x,scale)
        adapter.hadamard_transform=transform
        sys.modules[adapter.__name__]=adapter
        from transformers import AutoTokenizer,Qwen2Config,Qwen2ForCausalLM
        from gptqmodel import QuantizeConfig
        from gptqmodel.models.auto import MODEL_MAP,BaseGPTQModel
        from gptqmodel.quantization.rotation import hadamard_utils
        from gptqmodel.utils import BACKEND
        from probe_gptq import read_prompts,tokenize_prompts
        hadamard_utils.fast_hadamard_transform=adapter
        torch.manual_seed(0);torch.cuda.reset_peak_memory_stats()
        tokenizer=AutoTokenizer.from_pretrained(str(source),local_files_only=True,trust_remote_code=False)
        samples=tokenize_prompts(tokenizer,read_prompts(calibration))
        config=Qwen2Config(vocab_size=len(tokenizer),hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
        model=Qwen2ForCausalLM(config).to(dtype=torch.bfloat16)
        qconfig=QuantizeConfig(bits=4,group_size=-1,sym=True,lm_head=True,dynamic={'+:.*lm_head*':{'bits':8,'sym':True,'group_size':32,'desc_act':False}},device=torch.device(device).type,rotation='hadamard')
        quantized=MODEL_MAP.get('qwen2',BaseGPTQModel)(model,False,qconfig,trust_remote_code=False,model_local_path=str(source))
        quantized.quantize(samples,tokenizer=tokenizer,backend=BACKEND.TORCH)
        report.update(status='passed',sample_count=len(samples),sample_token_lengths=[int(s['input_ids'].shape[1]) for s in samples],quantized_linear_types=sorted({type(m).__module__+'.'+type(m).__name__ for m in quantized.model.modules() if 'QuantLinear' in type(m).__name__}),stored_tensor_devices=sorted({str(p.device) for p in quantized.model.parameters()}|{str(b.device) for b in quantized.model.buffers()}))
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    report['duration_seconds']=round(time.monotonic()-start,3)
    try:report['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
    except Exception:pass
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','calibration','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--expected-gpu',default='RX 7800 XT');a=p.parse_args()
    report=run(a.source,a.calibration,a.device,a.expected_gpu);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2));raise SystemExit(0 if report['status']=='passed' else 1)
