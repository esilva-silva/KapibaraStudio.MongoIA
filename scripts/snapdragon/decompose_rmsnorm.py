"""Experimental FP16 RMSNorm decomposition; never modifies the source package."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def transform_model(model):
    import onnx
    from onnx import TensorProto, helper
    opset = next((x.version for x in model.opset_import if x.domain in ('', 'ai.onnx')), None)
    if opset is None or opset < 13:
        raise ValueError('ONNX opset >= 13 is required')
    initializers = {x.name: x for x in model.graph.initializer}
    existing = {name for node in model.graph.node for name in list(node.input) + list(node.output)}
    existing.update(initializers)
    new_nodes = []
    count = 0
    for node in model.graph.node:
        simple = node.op_type == 'SimplifiedLayerNormalization' and node.domain in ('', 'ai.onnx')
        skip = node.op_type == 'SkipSimplifiedLayerNormalization' and node.domain == 'com.microsoft'
        if not (simple or skip):
            new_nodes.append(node)
            continue
        if simple and (len(node.input) != 2 or len(node.output) != 1 or not node.output[0]):
            raise ValueError('Only the single-output RMSNorm variant is supported')
        if skip and (len(node.input) != 3 or not node.output or not node.output[0]
                     or len(node.output) not in (1, 4)
                     or (len(node.output) == 4 and (node.output[1] or node.output[2] or not node.output[3]))):
            raise ValueError('Skip RMSNorm requires X, skip, scale and Y, optionally with residual')
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if set(attrs) - {'axis', 'epsilon', 'stash_type'} or attrs.get('axis', -1) != -1 or attrs.get('stash_type', 1) != 1:
            raise ValueError('Only axis=-1 and stash_type=1 are supported')
        scale_name = node.input[2] if skip else node.input[1]
        scale = initializers.get(scale_name)
        if scale is None or scale.data_type != TensorProto.FLOAT16:
            raise ValueError('RMSNorm scale must be an FP16 initializer')
        count += 1
        prefix = f'__rmsnorm_decomposed_{count}_'
        def name(suffix):
            value = prefix + suffix
            if value in existing:
                raise ValueError('Name collision: ' + value)
            existing.add(value)
            return value
        x = name('x32'); square = name('square'); mean = name('mean'); eps = name('epsilon')
        added = name('added'); root = name('sqrt'); div = name('normalized'); norm16 = name('norm16')
        model.graph.initializer.append(helper.make_tensor(eps, TensorProto.FLOAT, [], [attrs.get('epsilon', 1e-5)]))
        nodes = [helper.make_node('Cast', [node.input[0]], [x], to=TensorProto.FLOAT)]
        if skip:
            skip32 = name('skip32'); summed = name('summed')
            nodes.extend([helper.make_node('Cast', [node.input[1]], [skip32], to=TensorProto.FLOAT),
                          helper.make_node('Add', [x, skip32], [summed])])
            if len(node.output) == 4:
                nodes.append(helper.make_node('Cast', [summed], [node.output[3]], to=TensorProto.FLOAT16))
            x = summed
        nodes.extend([helper.make_node('Mul', [x, x], [square]),
                 helper.make_node('ReduceMean', [square], [mean], axes=[-1], keepdims=1) if opset < 18 else None,
                 helper.make_node('Add', [mean, eps], [added]), helper.make_node('Sqrt', [added], [root]),
                 helper.make_node('Div', [x, root], [div]),
                 helper.make_node('Cast', [div], [norm16], to=TensorProto.FLOAT16),
                 helper.make_node('Mul', [norm16, scale_name], [node.output[0]])])
        if skip:
            # CPU skip operator multiplies in FP32 and rounds once at the output.
            nodes = nodes[:-2]
            scale32 = name('scale32'); scaled = name('scaled')
            nodes.extend([helper.make_node('Cast', [scale_name], [scale32], to=TensorProto.FLOAT),
                          helper.make_node('Mul', [div, scale32], [scaled]),
                          helper.make_node('Cast', [scaled], [node.output[0]], to=TensorProto.FLOAT16)])
        if opset >= 18:
            axes = name('axes')
            model.graph.initializer.append(helper.make_tensor(axes, TensorProto.INT64, [1], [-1]))
            index = next(i for i, candidate in enumerate(nodes) if candidate is None)
            nodes[index] = helper.make_node('ReduceMean', [square, axes], [mean], keepdims=1)
        new_nodes.extend(nodes)
    if not count:
        raise ValueError('No supported SimplifiedLayerNormalization nodes found')
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return count


def build(source, output, model_name):
    import onnx
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output must not exist or overlap source')
    model_path = (source / model_name).resolve(strict=True)
    if not model_path.is_relative_to(source) or not model_path.is_file():
        raise ValueError('Model path must be inside source')
    parent_manifest_path = source / 'export_manifest.json'
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding='utf-8')) if parent_manifest_path.is_file() else None
    parent_manifest_hash = sha256(parent_manifest_path) if parent_manifest else None
    model = onnx.load(str(model_path), load_external_data=False)
    count = transform_model(model)
    output.mkdir(parents=True)
    for item in source.rglob('*'):
        if item.is_symlink():
            raise ValueError('Source symlinks are not supported')
        rel = item.relative_to(source)
        dest = output / rel
        if item.is_dir():
            dest.mkdir(exist_ok=True)
        elif item != model_path:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Copies make both packages independently mutable, even for external weights.
            shutil.copy2(item, dest)
            if sha256(item) != sha256(dest):
                raise IOError('Copy hash mismatch: ' + str(rel))
    output_model = output / model_path.relative_to(source)
    onnx.save_model(model, str(output_model))
    config_path = output / 'genai_config.json'
    if not config_path.is_file():
        raise ValueError('genai_config.json is required')
    genai = json.loads(config_path.read_text(encoding='utf-8'))
    decoder = genai.get('model', {}).get('decoder')
    if not isinstance(decoder, dict):
        raise ValueError('genai_config.json must contain model.decoder')
    decoder.setdefault('session_options', {})['graph_optimization_level'] = 'ORT_DISABLE_ALL'
    def set_pipeline_options(value):
        if isinstance(value, list):
            for item in value:
                set_pipeline_options(item)
        elif isinstance(value, dict):
            if 'filename' in value:
                value.setdefault('session_options', {})['graph_optimization_level'] = 'ORT_DISABLE_ALL'
            for child in value.values():
                if isinstance(child, (dict, list)):
                    set_pipeline_options(child)
    if 'pipeline' in decoder:
        set_pipeline_options(decoder['pipeline'])
    config_path.write_text(json.dumps(genai, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output / 'RMSNORM_TRANSFORM.md').write_text(
        '# Experimental RMSNorm graph transformation\n\n'
        'This changes only the graph: no new training was performed and FP16 weights are unchanged. '
        'The transformation is experimental and full 200-prompt validation is pending. '
        'Graph optimization must remain disabled to prevent refusion. '
        'The reported DirectML issue is an analogy, not proof of the cause on this device: '
        'https://github.com/microsoft/onnxruntime/issues/29739\n', encoding='utf-8')
    kapi_manifest_path = output / 'kapicoder_manifest.json'
    if kapi_manifest_path.is_file():
        kapi = json.loads(kapi_manifest_path.read_text(encoding='utf-8'))
        for entry in kapi.get('files', []):
            path = (output / entry['path']).resolve()
            if not path.is_relative_to(output) or not path.is_file():
                raise ValueError('Manifest resource missing or outside output: ' + entry['path'])
            entry['sha256'] = sha256(path)
            entry['bytes'] = path.stat().st_size
        note_path = output / 'RMSNORM_TRANSFORM.md'
        if not any(entry.get('path') == note_path.name for entry in kapi.get('files', [])):
            kapi.setdefault('files', []).append({'path': note_path.name, 'sha256': sha256(note_path), 'bytes': note_path.stat().st_size})
        kapi['graph_decomposition'] = {'experimental': True, 'training': 'none; graph-only transformation',
            'graph_optimization_level': 'ORT_DISABLE_ALL',
            'parent_export_manifest_sha256': parent_manifest_hash, 'source_model_sha256': sha256(model_path)}
        kapi_manifest_path.write_text(json.dumps(kapi, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    files = [{'path': p.relative_to(output).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'export_manifest.json']
    manifest = {'schema': 'experimental-rmsnorm-decomposition-v1', 'source': str(source),
        'source_model_sha256': sha256(model_path), 'parent_export_manifest_sha256': parent_manifest_hash,
        'parent_source_provenance': parent_manifest.get('source') if parent_manifest else None,
        'changed_nodes': count, 'training': 'none; experimental graph-only transformation',
        'graph_optimization_level': 'ORT_DISABLE_ALL',
        'note': 'Adreno cause unproven; ensure graph optimizations do not refusion the nodes and validate inference.',
        'files': files}
    (output / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', default='model.onnx')
    args = p.parse_args()
    try:
        print(json.dumps(build(args.source, args.out, args.model), indent=2))
    except (ValueError, OSError) as exc:
        p.exit(1, f'Decomposition failed: {exc}\n')


if __name__ == '__main__':
    main()
