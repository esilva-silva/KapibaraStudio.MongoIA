"""Manual dense-reference proof for the isolated PyTorch FWHT on CPU and selected HIP GPU."""
import argparse,json,math
from pathlib import Path
import torch
from torch_fwht import hadamard_transform
from run_sample import verify_gpu

def run(device,expected_gpu):
    evidence=verify_gpu(torch,device,expected_gpu)
    report={'status':'passed','scope':'numerical transform only, not quantization or native CUDA equivalence','gpu':evidence,'items':[]}
    generator=torch.Generator(device='cpu').manual_seed(20261026)
    tolerances={torch.float32:(2e-6,2e-6),torch.float16:(1e-3,1e-3),torch.bfloat16:(8e-3,8e-3)}
    for n in (16,32,64,128):
        dense=torch.tensor([[1 if (i&j).bit_count()%2==0 else -1 for j in range(n)] for i in range(n)],dtype=torch.float64)
        for dtype in (torch.float32,torch.float16,torch.bfloat16):
            cpu=(torch.randint(-32,33,(2,3,n),generator=generator)/16).to(dtype)
            for scale in (1/math.sqrt(n),1,0.25):
                expected=(cpu.double()@dense.T*scale).to(dtype)
                for selected in ('cpu',device):
                    x=cpu.to(selected);before=x.clone()
                    actual=hadamard_transform(x,scale)
                    torch.testing.assert_close(actual.cpu(),expected,rtol=tolerances[dtype][0],atol=tolerances[dtype][1])
                    assert torch.equal(x,before) and actual.device==x.device and actual.dtype==x.dtype
                    report['items'].append({'n':n,'dtype':str(dtype),'scale':scale,'device':str(actual.device),
                        'max_abs_error_rounded_reference':float((actual.cpu().double()-expected.double()).abs().max())})
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',required=True);p.add_argument('--expected-gpu',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=run(a.device,a.expected_gpu);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print('Passed',len(report['items']),'dense CPU/GPU comparisons')
