"""Compile QNN EPContext locally; adapted from Olive 0.13 context_binary.py.
Copyright (c) Microsoft Corporation. MIT license: configs/snapdragon/LICENSE.olive.txt.
https://github.com/microsoft/Olive/blob/main/olive/passes/onnx/context_binary.py
"""
import argparse
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path, PurePosixPath


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(path, models):
    path = Path(path).resolve(strict=True)
    root = path.parent
    manifest = json.loads(path.read_text(encoding='utf-8'))
    entries = manifest.get('files')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Manifest requires a nonempty files list')
    covered = set()
    for entry in entries:
        rel = PurePosixPath(entry['path'].replace('\\', '/'))
        if rel.is_absolute() or not rel.parts or '..' in rel.parts:
            raise ValueError('Unsafe manifest path')
        resolved = (root / Path(*rel.parts)).resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError('Manifest resource outside root')
        if sha(resolved) != entry['sha256'] or resolved.stat().st_size != entry['bytes']:
            raise ValueError('Manifest mismatch: ' + entry['path'])
        covered.add(resolved)
    for model in models:
        model = Path(model).resolve(strict=True)
        if model not in covered:
            raise ValueError('Model not covered by manifest: ' + str(model))
        import onnx
        graph = onnx.load(str(model), load_external_data=False)
        for tensor in graph.graph.initializer:
            meta = {e.key: e.value for e in tensor.external_data}
            if 'location' in meta:
                rel = PurePosixPath(meta['location'].replace('\\', '/'))
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError('Unsafe external tensor path')
                dep = (model.parent / Path(*rel.parts)).resolve(strict=True)
                if dep not in covered:
                    raise ValueError('External tensor file not covered by manifest')
    return root


def context_files(model_path):
    import onnx
    model_path = Path(model_path).resolve(strict=True)
    graph = onnx.load(str(model_path), load_external_data=False)
    files = []
    for node in graph.graph.node:
        if node.op_type != 'EPContext': continue
        for attr in node.attribute:
            if attr.name != 'ep_cache_context': continue
            try: name = attr.s.decode('utf-8')
            except UnicodeDecodeError as exc: raise ValueError('Embedded context not supported') from exc
            rel = PurePosixPath(name.replace('\\', '/'))
            if rel.is_absolute() or '..' in rel.parts or not rel.parts:
                raise ValueError('Unsafe EPContext file path')
            path = (model_path.parent / Path(*rel.parts)).resolve(strict=True)
            if not path.is_relative_to(model_path.parent) or not path.is_file() or path.stat().st_size == 0:
                raise ValueError('Missing or empty EPContext binary')
            files.append(path)
    if not files: raise ValueError('No external EPContext binary references')
    return files


def smoke_model(path):
    import numpy as np
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), 'scale')
    zero = numpy_helper.from_array(np.array(128, dtype=np.uint8), 'zero')
    nodes = [helper.make_node('QuantizeLinear', ['x', 'scale', 'zero'], ['xq']),
        helper.make_node('DequantizeLinear', ['xq', 'scale', 'zero'], ['xd']),
        helper.make_node('Relu', ['xd'], ['r']),
        helper.make_node('QuantizeLinear', ['r', 'scale', 'zero'], ['yq']),
        helper.make_node('DequantizeLinear', ['yq', 'scale', 'zero'], ['y'])]
    graph = helper.make_graph(nodes, 'qnn_context_smoke',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4])], [scale, zero])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
    onnx.save(model, str(path))


def compile_context(models, out_dir, manifest=None, smoke=False, weight_sharing=False):
    import onnxruntime as ort
    import onnxruntime_qnn as qnn
    output = Path(out_dir).resolve()
    if output.exists(): raise ValueError('Output directory must not exist')
    if smoke and models: raise ValueError('Smoke and model inputs are mutually exclusive')
    if not smoke:
        if not models or not manifest: raise ValueError('Provide --model and --manifest, or --smoke')
        models = [Path(p).resolve(strict=True) for p in models]
        verify_manifest(manifest, models)
    if weight_sharing and (smoke or len(models) < 2): raise ValueError('Weight sharing requires multiple model inputs')
    output.mkdir(parents=True)
    if smoke:
        model = output / 'smoke.onnx'; smoke_model(model); models = [model]
    ep_name = qnn.get_ep_name()
    ort.register_execution_provider_library(ep_name, qnn.get_library_path())
    sessions = []
    try:
        devices = [d for d in ort.get_ep_devices() if d.ep_name == ep_name and d.device.type == ort.OrtHardwareDeviceType.NPU]
        if not devices: raise RuntimeError('No QNN NPU device detected')
        compiled = []
        for index, model in enumerate(models):
            target = output / f'{index:02d}_{model.stem}_ctx.onnx'
            options = ort.SessionOptions()
            for key, value in {'session.disable_cpu_ep_fallback': '1', 'ep.context_enable': '1',
                'ep.context_embed_mode': '0', 'ep.context_file_path': str(target),
                'ep.share_ep_contexts': '1' if weight_sharing else '0',
                'ep.stop_share_ep_contexts': '1' if weight_sharing and index == len(models)-1 else '0'}.items():
                options.add_session_config_entry(key, value)
            ep_options = {'backend_path': qnn.get_qnn_htp_path()}
            if weight_sharing: ep_options['enable_htp_weight_sharing'] = '1'
            options.add_provider_for_devices(devices, ep_options)
            sessions.append(ort.InferenceSession(str(model), sess_options=options))
            if not target.exists(): raise RuntimeError('Context ONNX not generated')
            compiled.append(target)
        # Shared binary can be emitted only by the last session.
        for path in compiled: context_files(path)
        sessions.clear()
        report = {'status': 'compiled_not_inference_validated', 'weight_sharing': weight_sharing,
                  'inputs': [{'path': str(p), 'sha256': sha(p)} for p in models], 'artifacts': [],
                  'cpu_fallback_disabled': True, 'device_type': 'NPU', 'device_count': len(devices),
                  'backend_path': str(qnn.get_qnn_htp_path()),
                  'versions': {name: version(name) for name in ('onnxruntime', 'onnxruntime-qnn', 'onnx')},
                  'source_script_sha256': sha(__file__),
                  'input_manifest_sha256': sha(manifest) if manifest else None}
        if smoke:
            import numpy as np
            options = ort.SessionOptions(); options.enable_profiling = True
            options.profile_file_prefix = str(output / 'htp-profile')
            options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
            options.add_provider_for_devices(devices, {'backend_path': qnn.get_qnn_htp_path(),
                'profiling_level': 'basic', 'profiling_file_path': str(output / 'qnn-htp-profile.csv')})
            session = ort.InferenceSession(str(compiled[0]), sess_options=options)
            actual = session.run(None, {'x': np.array([[-1, 0, 1, 2]], dtype=np.float32)})[0]
            np.testing.assert_allclose(actual, [[0, 0, 1, 2]], atol=.11)
            report['profile'] = session.end_profiling()
            events = json.loads(Path(report['profile']).read_text(encoding='utf-8'))
            counts = Counter(event.get('args', {}).get('provider') for event in events if event.get('args', {}).get('provider'))
            report['kernel_events_by_provider'] = dict(counts)
            if not counts.get(ep_name) or counts.get('CPUExecutionProvider'):
                raise RuntimeError('Compiled smoke lacks exclusive QNN kernel evidence')
            report['expected'] = [[0, 0, 1, 2]]
            report['actual'] = actual.tolist()
            del session
            report['status'] = 'smoke_inference_passed'
        for p in compiled:
            for file in [p, *context_files(p)]:
                if not any(e['path'] == file.name for e in report['artifacts']):
                    report['artifacts'].append({'path': file.name, 'bytes': file.stat().st_size, 'sha256': sha(file)})
        (output / 'compile-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        return report
    finally:
        sessions.clear()
        ort.unregister_execution_provider_library(ep_name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', action='append', default=[]); p.add_argument('--manifest')
    p.add_argument('--out-dir', required=True); p.add_argument('--smoke', action='store_true')
    p.add_argument('--weight-sharing', action='store_true')
    a = p.parse_args()
    try: print(json.dumps(compile_context(a.model, a.out_dir, a.manifest, a.smoke, a.weight_sharing), indent=2))
    except Exception as exc: p.exit(1, f'QNN compile failed: {type(exc).__name__}: {exc}\n')


if __name__ == '__main__': main()
