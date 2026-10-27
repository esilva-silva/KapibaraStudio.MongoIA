"""Run one compiled QNN EPContext with deterministic zero inputs for profiling."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
from compile_qnn_context import context_files


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def version(name):
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return None


def run(model_path, out_dir):
    import numpy as np
    import onnxruntime as ort
    import onnxruntime_qnn as qnn
    model=Path(model_path).resolve(strict=True);out=Path(out_dir).resolve()
    if out.exists() or out.is_relative_to(model.parent) or model.parent.is_relative_to(out):
        raise ValueError('Output directory must be new and separate from model directory')
    binaries=context_files(model)
    out.mkdir(parents=True)
    report={'status':'failed','scope':'synthetic profiling only, no quality validation',
        'model':str(model),'model_sha256':sha(model),
        'context_binaries':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in binaries],
        'versions':{name:version(name) for name in ('onnxruntime','onnxruntime-qnn','onnx')},
        'cpu_fallback_disabled':True, 'plugin_released_by_process_exit':True}
    ep=qnn.get_ep_name();registered=False;session=None
    try:
        ort.register_execution_provider_library(ep,qnn.get_library_path());registered=True
        devices=[d for d in ort.get_ep_devices() if d.ep_name==ep and d.device.type==ort.OrtHardwareDeviceType.NPU]
        if not devices:raise RuntimeError('No QNN NPU device')
        options=ort.SessionOptions();options.log_severity_level=0;options.log_verbosity_level=4
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.enable_profiling=True;options.profile_file_prefix=str(out/'ort-profile')
        options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
        opts={'backend_path':qnn.get_qnn_htp_path(),'profiling_level':'basic', 'profiling_file_path':str(out/'qnn-profile.csv')}
        options.add_provider_for_devices(devices,opts)
        session=ort.InferenceSession(str(model),sess_options=options)
        types={'tensor(float)':np.float32,'tensor(float16)':np.float16,'tensor(uint16)':np.uint16,
               'tensor(int64)':np.int64,'tensor(int32)':np.int32,'tensor(uint8)':np.uint8}
        feeds={}; input_details=[]
        for item in session.get_inputs():
            if item.type not in types:raise ValueError('Unsupported input dtype: '+item.type)
            if any(type(dim) is not int or dim <= 0 for dim in item.shape):
                raise ValueError('Input shape is not concrete and positive: '+item.name+' '+str(item.shape))
            data=np.zeros(item.shape,dtype=types[item.type])
            if item.name=='past_seq_len':data.fill(63)
            elif item.name=='total_seq_len':data.fill(64)
            feeds[item.name]=data
            input_details.append({'name':item.name,'shape':item.shape,'type':item.type,
                                  'fill':63 if item.name=='past_seq_len' else 64 if item.name=='total_seq_len' else 0})
        report['inputs']=input_details
        values=session.run(None,feeds)
        report['outputs']=[{'name':item.name,'shape':list(value.shape), 'dtype':str(value.dtype),
                            'finite':bool(np.isfinite(value).all())} for item,value in zip(session.get_outputs(),values)]
        profile=Path(session.end_profiling()); del session;session=None
        events=json.loads(profile.read_text(encoding='utf-8'))
        counts=Counter((event.get('args') or {}).get('provider') for event in events if (event.get('args') or {}).get('provider'))
        report['ort_profile']={'path':str(profile),'sha256':sha(profile),'kernel_events_by_provider':dict(counts)}
        csv=out/'qnn-profile.csv'
        report['qnn_csv']={'path':str(csv),'exists':csv.is_file(),'sha256':sha(csv) if csv.is_file() else None,
                           'bytes':csv.stat().st_size if csv.is_file() else None}
        if not counts.get(ep) or counts.get('CPUExecutionProvider'):
            raise RuntimeError('Profile lacks exclusive QNN provider kernel evidence')
        if not all(row['finite'] for row in report['outputs']):raise RuntimeError('Nonfinite output')
        report['status']='passed' if csv.is_file() and csv.stat().st_size else 'inconclusive_csv_missing'
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        session=None
        # One-shot CLI: process exit releases the plugin after all native handles.
        # Explicit unregister can race live EP/device handles on this runtime.
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args()
    try:
        report=run(a.model,a.out_dir);print(json.dumps(report,indent=2));return 0 if report['status']=='passed' else 1
    except Exception as exc:p.exit(1,f'Compiled probe failed: {type(exc).__name__}: {exc}\n')

if __name__=='__main__':raise SystemExit(main())
