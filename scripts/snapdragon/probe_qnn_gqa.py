"""Isolate first ONNX Runtime GQA with fixed KV dimensions; optional QNN HTP run."""
import argparse
import copy
import json
from pathlib import Path, PurePosixPath


def build(source, output, dtype='fp32', heads=14):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    source = Path(source).resolve(strict=True); output = Path(output).resolve()
    if output.exists() or output.is_relative_to(source.parent):
        raise ValueError('Output must be new and outside source package')
    if heads <= 0 or heads % 2: raise ValueError('heads must be positive and divisible by two KV heads')
    model = onnx.load(str(source), load_external_data=False)
    node = copy.deepcopy(next(n for n in model.graph.node if n.op_type == 'GroupQueryAttention'))
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if attrs.get('num_heads') != 14 or attrs.get('kv_num_heads') != 2 or attrs.get('do_rotary') != 1 or attrs.get('rotary_interleaved') != 0:
        raise ValueError('Unexpected source GQA attributes')
    for attr in node.attribute:
        if attr.name == 'num_heads': attr.i = heads
    node.input[:] = ['qkv', '', '', 'past_key', 'past_value', 'past_seq_len', 'total_seq_len', 'cos_cache', 'sin_cache', '', '', '']
    node.output[:] = ['y', 'present_key', 'present_value']
    output.mkdir(parents=True)
    initializers = []
    for name in ('cos_cache', 'sin_cache'):
        tensor = next(t for t in model.graph.initializer if t.name == name)
        meta = {e.key: e.value for e in tensor.external_data}
        rel = PurePosixPath(meta['location'].replace('\\', '/'))
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe external data path')
        data_path = (source.parent / Path(*rel.parts)).resolve(strict=True)
        if not data_path.is_relative_to(source.parent): raise ValueError('External file outside source')
        with data_path.open('rb') as stream:
            stream.seek(int(meta['offset'])); raw = stream.read(int(meta['length']))
        array = np.frombuffer(raw, dtype=np.float32).reshape(tuple(tensor.dims)).copy()
        if dtype == 'fp16': array = array.astype(np.float16)
        initializers.append(numpy_helper.from_array(array, name))
    typ = TensorProto.FLOAT16 if dtype == 'fp16' else TensorProto.FLOAT
    inputs = [helper.make_tensor_value_info('qkv', typ, [1,64,(heads+4)*64]),
        helper.make_tensor_value_info('past_key', typ, [1,2,2048,64]),
        helper.make_tensor_value_info('past_value', typ, [1,2,2048,64]),
        helper.make_tensor_value_info('past_seq_len', TensorProto.INT32, [1,1]),
        helper.make_tensor_value_info('total_seq_len', TensorProto.INT32, [])]
    outputs = [helper.make_tensor_value_info('y', typ, [1,64,heads*64]),
        helper.make_tensor_value_info('present_key', typ, [1,2,2048,64]),
        helper.make_tensor_value_info('present_value', typ, [1,2,2048,64])]
    graph = helper.make_graph([node], 'gqa_fixed_shapes', inputs, outputs, initializers)
    isolated = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid('com.microsoft', 1)], ir_version=10)
    target = output / 'gqa.onnx'; onnx.save(isolated, str(target))
    return target


def run(model_path, output, dtype, heads, run_npu, affinity_htp=False):
    import numpy as np
    import onnxruntime as ort
    rng = np.random.default_rng(42); npdtype = np.float16 if dtype == 'fp16' else np.float32
    feeds = {'qkv': rng.normal(0, .05, (1,64,(heads+4)*64)).astype(npdtype),
        'past_key': np.zeros((1,2,2048,64), dtype=npdtype), 'past_value': np.zeros((1,2,2048,64), dtype=npdtype),
        'past_seq_len': np.array([[63]], dtype=np.int32), 'total_seq_len': np.array(64, dtype=np.int32)}
    report = {'model': str(model_path), 'dtype': dtype, 'heads': heads, 'run_npu': run_npu}
    options = ort.SessionOptions(); options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        cpu = ort.InferenceSession(str(model_path), options, providers=['CPUExecutionProvider'])
        expected = cpu.run(None, feeds)
        report['cpu'] = {'status':'passed', 'shapes':[list(x.shape) for x in expected], 'finite':all(bool(np.isfinite(x).all()) for x in expected)}
    except Exception as exc:
        expected = None; report['cpu'] = {'status':'failed', 'error': f'{type(exc).__name__}: {exc}'}
    if run_npu:
        import onnxruntime_qnn as qnn
        ep = qnn.get_ep_name(); ort.register_execution_provider_library(ep, qnn.get_library_path())
        try:
            devices=[d for d in ort.get_ep_devices() if d.ep_name==ep and d.device.type==ort.OrtHardwareDeviceType.NPU]
            if not devices: raise RuntimeError('No QNN NPU device')
            opts=ort.SessionOptions(); opts.log_severity_level=0; opts.log_verbosity_level=4; opts.enable_profiling=True
            opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            opts.profile_file_prefix=str(output/'ort-profile')
            opts.add_session_config_entry('session.disable_cpu_ep_fallback','1')
            ep_options = {'backend_path':qnn.get_qnn_htp_path(), 'profiling_level':'detailed',
                'profiling_file_path':str(output/'qnn-profile.csv')}
            if affinity_htp:
                affinity_path = output / 'op-affinity.json'
                affinity_path.write_text(json.dumps({'op_type': {'GroupQueryAttention': 'HTP'}}, indent=2) + '\n', encoding='utf-8')
                ep_options['op_affinity'] = str(affinity_path)
            report['op_affinity'] = ep_options.get('op_affinity')
            opts.add_provider_for_devices(devices, ep_options)
            session=ort.InferenceSession(str(model_path),sess_options=opts)
            actual=session.run(None,feeds); profile=session.end_profiling(); del session
            result={'status':'passed','profile':profile,'shapes':[list(x.shape) for x in actual], 'finite':all(bool(np.isfinite(x).all()) for x in actual)}
            if expected is not None:
                result['max_abs_error']=[float(np.max(np.abs(a.astype(np.float32)-b.astype(np.float32)))) for a,b in zip(actual,expected)]
            report['npu']=result
        except Exception as exc:
            report['npu']={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
        finally: ort.unregister_execution_provider_library(ep)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--dtype',choices=('fp32','fp16'),default='fp32');p.add_argument('--heads',type=int,default=14)
    p.add_argument('--run-npu',action='store_true')
    p.add_argument('--affinity-htp',action='store_true', help='Explicitly pin GroupQueryAttention to HTP')
    args=p.parse_args(); output=Path(args.out_dir).resolve()
    try:
        model=build(args.source,output,args.dtype,args.heads)
        report=run(model,output,args.dtype,args.heads,args.run_npu,args.affinity_htp)
        print(json.dumps(report,indent=2))
        return 0 if report['cpu']['status']=='passed' and (not args.run_npu or report['npu']['status']=='passed') else 1
    except Exception as exc:
        p.exit(1,f'GQA probe failed: {type(exc).__name__}: {exc}\n')

if __name__ == '__main__': raise SystemExit(main())
