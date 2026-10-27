"""Bounded CPU-only GPTQ feasibility; no Hadamard, no QNN qualification, no user weights."""
import argparse
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

def run(source, calibration):
    result={'status':'failed','recipe':'GPTQ CPU/Torch without rotation','qualified':False,
            'scope':'random tiny model only; different from Hadamard/QNN recipe'}
    start=time.monotonic()
    try:
        import torch
        if torch.cuda.is_available() or torch.cuda.device_count():
            raise RuntimeError('CPU alternative requires no visible CUDA/HIP devices before importing GPTQModel')
        torch.set_num_threads(4)
        torch.manual_seed(0)
        result['gpu_visible_count']=torch.cuda.device_count()
        result['hip']=torch.version.hip
        result['versions']={name:importlib.metadata.version(name) for name in ('torch','gptqmodel','transformers')}
        from transformers import AutoTokenizer,Qwen2Config,Qwen2ForCausalLM
        from gptqmodel import QuantizeConfig
        from gptqmodel.models.auto import MODEL_MAP,BaseGPTQModel
        from gptqmodel.utils import BACKEND
        from probe_gptq import read_prompts,tokenize_prompts
        tokenizer=AutoTokenizer.from_pretrained(str(source),local_files_only=True,trust_remote_code=False)
        samples=tokenize_prompts(tokenizer,read_prompts(calibration))
        config=Qwen2Config(vocab_size=len(tokenizer),hidden_size=64,intermediate_size=128,num_hidden_layers=1,
                          num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
        model=Qwen2ForCausalLM(config).float().cpu().eval()
        quant_config=QuantizeConfig(bits=4,group_size=-1,sym=True,lm_head=False,device='cpu',rotation=None)
        quantized=MODEL_MAP.get('qwen2',BaseGPTQModel)(model,False,quant_config,trust_remote_code=False,model_local_path=str(source))
        quantized.quantize(samples,tokenizer=tokenizer,backend=BACKEND.TORCH,calibration_enable_gpu_cache=False)
        devices=sorted({str(p.device) for p in quantized.model.parameters()}|{str(b.device) for b in quantized.model.buffers()})
        if devices != ['cpu']: raise RuntimeError('Non-CPU tensors after quantization: '+str(devices))
        result.update(status='passed',sample_count=len(samples),sample_token_lengths=[int(s['input_ids'].shape[1]) for s in samples],tensor_devices=devices)
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    result['duration_seconds']=round(time.monotonic()-start,3)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','calibration','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=run(args.source,args.calibration)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='passed' else 1)
