"""Read-only FP16 initializer scan and one bounded HF FP16 GPU control, no export modification."""
import argparse,json,time
from pathlib import Path

def scan_onnx(package):
 import numpy as np,onnx
 graph=onnx.load_model(str(package/'model.onnx'),load_external_data=False)
 result={'initializer_count':len(graph.graph.initializer),'floating_elements':0,'nonfinite_initializers':[],'max_abs_finite':0.0}
 for tensor in graph.graph.initializer:
  dtype=onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
  if not np.issubdtype(dtype,np.floating):continue
  count=int(np.prod(tensor.dims));result['floating_elements']+=count
  if tensor.data_location==onnx.TensorProto.EXTERNAL:
   ext={v.key:v.value for v in tensor.external_data};data=np.memmap(package/ext['location'],mode='r',dtype=dtype,offset=int(ext.get('offset',0)),shape=(count,))
  else:data=onnx.numpy_helper.to_array(tensor).reshape(-1)
  bad=0;maximum=0.0
  for start in range(0,count,1048576):
   chunk=np.asarray(data[start:start+1048576]);finite=np.isfinite(chunk);bad+=int((~finite).sum())
   if finite.any():maximum=max(maximum,float(np.abs(chunk[finite]).max()))
  result['max_abs_finite']=max(result['max_abs_finite'],maximum)
  if bad:result['nonfinite_initializers'].append({'name':tensor.name,'count':bad,'dtype':str(dtype)})
 return result

def control(source,corpus,sample_id,device,expected_gpu):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from run_sample import verify_gpu,HFRunner,stop_ids
 gpu=verify_gpu(torch,device,expected_gpu)
 if torch.cuda.device_count()!=1:raise RuntimeError('Only the selected RX may be visible')
 torch.cuda.set_device(0)
 items=[json.loads(x) for x in corpus.read_text(encoding='utf-8').splitlines()]
 item=next(x for x in items if x['id']==sample_id)
 tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16).to(device).eval()
 traces=[]
 def hook(name):
  def capture(module,args,output):
   x=output[0] if isinstance(output,tuple) else output
   if not isinstance(x,torch.Tensor):return
   finite=torch.isfinite(x);count=int(finite.sum());total=x.numel()
   maximum=float(torch.where(finite,x.float().abs(),0).max()) if count else None
   traces.append({'module':name,'dtype':str(x.dtype),'device':str(x.device),'elements':total,'nonfinite':total-count,'max_abs_finite':maximum})
  return capture
 handles=[layer.register_forward_hook(hook(f'layer.{i}')) for i,layer in enumerate(model.model.layers)]
 handles += [model.model.norm.register_forward_hook(hook('final_norm')),model.lm_head.register_forward_hook(hook('lm_head'))]
 ids=tokenizer.encode(item['prompt_text'],add_special_tokens=False)
 with torch.inference_mode():
  result=model(input_ids=torch.tensor([ids],device=device),attention_mask=torch.ones((1,len(ids)),dtype=torch.long,device=device),use_cache=False)
  logits=result.logits[0,-1];next_id=int(logits.argmax())
  finite=bool(torch.isfinite(logits).all())
 for h in handles:h.remove()
 control={'sample_id':sample_id,'gpu':gpu,'dtype':str(next(model.parameters()).dtype),'first_token_id':next_id,'first_token':tokenizer.decode([next_id]),'last_position_logits_finite':finite,'trace':traces}
 if finite:
  runner=HFRunner.__new__(HFRunner);runner.torch=torch;runner.device=device;runner.tok=tokenizer;runner.model=model;runner.stops=stop_ids(lambda s:tokenizer.encode(s,add_special_tokens=False))
  control['generation_32']=runner.generate(item)
 else:control['generation_32']='not attempted because the first logits were nonfinite'
 return control

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('source','package','corpus','out'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--sample-id',default='fim-legal.cobalt_leg.billing-000714-stage-000279');p.add_argument('--device',default='cuda:0');p.add_argument('--expected-gpu',default='RX 7800 XT');a=p.parse_args()
 report={'scope':'read-only published FP16 scan and bounded HF control; no IDE/GPU qualification','onnx':scan_onnx(a.package),'hf_fp16':control(a.source,a.corpus,a.sample_id,a.device,a.expected_gpu)}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
