"""Experimental external-view split for transposed FP16 token embedding."""
import argparse
import copy
import json
import shutil
from pathlib import Path, PurePosixPath
from decompose_rmsnorm import sha256

LIMIT = 1 << 27


def transform(model, root, weight_name='lm_head.MatMul.weight', chunk_size=768):
    from onnx import TensorProto, helper
    weights = [t for t in model.graph.initializer if t.name == weight_name]
    if len(weights) != 1:
        raise ValueError('Expected exactly one embedding initializer')
    w = weights[0]
    if w.data_type != TensorProto.FLOAT16 or len(w.dims) != 2 or w.data_location != TensorProto.EXTERNAL:
        raise ValueError('Expected external FP16 [hidden,vocab] initializer')
    hidden, vocab = w.dims
    if hidden <= 0 or vocab <= 0 or chunk_size <= 0:
        raise ValueError('Invalid shape or chunk size')
    chunk_size = min(chunk_size, (LIMIT - 1) // vocab)
    if chunk_size <= 0 or chunk_size >= hidden:
        raise ValueError('Embedding cannot be split into multiple safe chunks')
    meta = {x.key: x.value for x in w.external_data}
    if set(meta) - {'location', 'offset', 'length'} or not {'location', 'offset', 'length'} <= set(meta):
        raise ValueError('Unsupported external tensor metadata')
    location = PurePosixPath(meta['location'].replace('\\', '/'))
    if location.is_absolute() or '..' in location.parts:
        raise ValueError('External tensor path escapes package')
    data = (Path(root) / Path(*location.parts)).resolve(strict=True)
    if not data.is_relative_to(Path(root).resolve()) or not data.is_file():
        raise ValueError('Invalid external tensor file')
    offset, length = int(meta['offset']), int(meta['length'])
    if offset < 0 or length != hidden * vocab * 2 or offset + length > data.stat().st_size:
        raise ValueError('Invalid external tensor range')
    transposes = [n for n in model.graph.node if n.op_type == 'Transpose' and list(n.input) == [weight_name]]
    if len(transposes) != 1:
        raise ValueError('Expected one embedding transpose')
    trans = transposes[0]
    attrs = {a.name: helper.get_attribute_value(a) for a in trans.attribute}
    if attrs != {'perm': [1, 0]} or len(trans.output) != 1:
        raise ValueError('Unexpected transpose permutation')
    uses = [(n, i) for n in model.graph.node for i, v in enumerate(n.input) if v == trans.output[0]]
    if len(uses) != 1 or uses[0][0].op_type != 'Gather' or uses[0][1] != 0:
        raise ValueError('Expected a single Gather consuming transposed weight')
    gather = uses[0][0]
    attrs = {a.name: helper.get_attribute_value(a) for a in gather.attribute}
    if attrs.get('axis', 0) != 0 or set(attrs) - {'axis'} or len(gather.input) != 2 or len(gather.output) != 1:
        raise ValueError('Expected standard axis-0 Gather')
    existing = {v for n in model.graph.node for v in [*n.input, *n.output]} | {t.name for t in model.graph.initializer}
    names = []
    for i, start in enumerate(range(0, hidden, chunk_size)):
        size = min(chunk_size, hidden - start)
        name = f'__split_embedding_weight_{i}'
        outputs = (f'__split_embedding_transposed_{i}', f'__split_embedding_gathered_{i}')
        if any(n in existing for n in (name, *outputs)):
            raise ValueError('Generated tensor name collision')
        tensor = copy.deepcopy(w); tensor.name = name; tensor.dims[0] = size
        for attr in tensor.external_data:
            if attr.key == 'offset': attr.value = str(offset + start * vocab * 2)
            if attr.key == 'length': attr.value = str(size * vocab * 2)
        model.graph.initializer.append(tensor)
        names.append((name, *outputs))
    nodes = []
    for n in model.graph.node:
        if n == trans:
            nodes.extend(helper.make_node('Transpose', [name], [out], perm=[1, 0]) for name, out, _ in names)
        elif n == gather:
            nodes.extend(helper.make_node('Gather', [out, n.input[1]], [result], axis=0) for _, out, result in names)
            nodes.append(helper.make_node('Concat', [result for _, _, result in names], list(n.output), axis=-1))
        else:
            nodes.append(n)
    del model.graph.node[:]; model.graph.node.extend(nodes)
    return {'weight': weight_name, 'hidden': hidden, 'vocab': vocab, 'chunk_size': chunk_size,
            'chunk_elements': [min(chunk_size, hidden - i) * vocab for i in range(0, hidden, chunk_size)],
            'external_file': str(location)}


def build(source, output, chunk_size=768):
    import onnx
    source = Path(source).resolve(strict=True); output = Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output must be new and separate')
    model_path = source / 'model.onnx'
    parent_path = source / 'export_manifest.json'
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    entries = parent.get('files')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Parent manifest files missing')
    seen = set()
    for entry in entries:
        relative = PurePosixPath(entry['path'].replace('\\', '/'))
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.as_posix() in seen:
            raise ValueError('Unsafe or duplicate parent manifest path')
        seen.add(relative.as_posix())
        path = (source / Path(*relative.parts)).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError('Parent manifest resource missing or outside source')
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError('Parent manifest integrity mismatch: ' + entry['path'])
    if 'model.onnx' not in seen:
        raise ValueError('Parent manifest does not cover model.onnx')
    parent_hash = sha256(parent_path)
    model = onnx.load(str(model_path), load_external_data=False)
    details = transform(model, source, chunk_size=chunk_size)
    output.mkdir(parents=True)
    for item in source.rglob('*'):
        if item.is_symlink(): raise ValueError('Symlinks are unsupported')
        target = output / item.relative_to(source)
        if item.is_dir(): target.mkdir(exist_ok=True)
        elif item != model_path:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            if sha256(item) != sha256(target): raise ValueError('Copy hash mismatch')
    onnx.save_model(model, str(output / 'model.onnx'))
    config_path = output / 'genai_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config['model']['decoder'].setdefault('session_options', {})['graph_optimization_level'] = 'ORT_DISABLE_ALL'
    config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    note = output / 'SPLIT_EMBEDDING.md'
    note.write_text('Experimental graph-only split of embedding lookup; FP16 weights unchanged, no training. Keep ORT_DISABLE_ALL. Requires independent model and GPU validation.\n', encoding='utf-8')
    kapi_path = output / 'kapicoder_manifest.json'
    if kapi_path.exists():
        kapi = json.loads(kapi_path.read_text(encoding='utf-8'))
        entries = kapi.setdefault('files', [])
        if not any(e['path'] == note.name for e in entries): entries.append({'path': note.name})
        for entry in entries:
            p = (output / entry['path']).resolve()
            if not p.is_relative_to(output) or not p.is_file(): raise ValueError('Invalid internal manifest path')
            entry.update(sha256=sha256(p), bytes=p.stat().st_size)
        kapi['split_embedding'] = {'experimental': True, 'details': details, 'parent_export_manifest_sha256': parent_hash}
        kapi_path.write_text(json.dumps(kapi, indent=2) + '\n', encoding='utf-8')
    files = [{'path': p.relative_to(output).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'export_manifest.json']
    manifest = {'schema': 'experimental-split-embedding-v1', 'parent_export_manifest_sha256': parent_hash,
        'parent_source_provenance': parent.get('parent_source_provenance') or parent.get('source'),
        'source_model_sha256': sha256(model_path), 'graph_optimization_level': 'ORT_DISABLE_ALL',
        'experimental': True, 'training': 'none; graph-only transformation', 'details': details, 'files': files}
    (output / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path); p.add_argument('--out', required=True, type=Path)
    p.add_argument('--chunk-size', type=int, default=768)
    a = p.parse_args()
    try: print(json.dumps(build(a.source, a.out, a.chunk_size), indent=2))
    except (ValueError, OSError, KeyError) as e: p.exit(1, f'Split failed: {e}\n')

if __name__ == '__main__': main()

