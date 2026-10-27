"""Prepare a self-contained, unqualified QNN transfer without modifying the source."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from export_gpu import TOKEN_FILES


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def relative_name(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('invalid relative resource')
    p = PurePosixPath(value)
    if p.is_absolute() or PureWindowsPath(value).drive or any(x in ('.', '..') for x in value.split('/')):
        raise ValueError('unsafe relative resource: ' + value)
    return p


def source_file(root, name):
    relative = relative_name(name)
    result = root.joinpath(*relative.parts).resolve(strict=True)
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError('resource outside source or not a file: ' + name)
    return result


def prepare(source, destination, provenance, reports, source_snapshot, preserve_tokenizer_source=False):
    import onnx
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    if not source.is_dir() or source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('destination must be separate from source')
    if destination.exists():
        raise FileExistsError('destination already exists')
    original = source_file(source, 'model_config.json').read_bytes()
    model = json.loads(original)
    if str(model.get('type', '')).lower() != 'compositemodel':
        raise ValueError('expected CompositeModel')
    config = model['config']
    components = config['model_components']
    if config['model_component_names'] != ['embeddings', 'context', 'iterator', 'lm_head'] or len(components) != 4:
        raise ValueError('unexpected pipeline')
    files = set()
    def normalize_files(attributes):
        if 'additional_files' not in attributes: return
        refs = attributes['additional_files']
        if not isinstance(refs, list): raise ValueError('invalid additional_files')
        mapped = []
        for ref in refs:
            old = Path(ref).resolve(strict=True)
            if not old.is_file(): raise ValueError('invalid original resource')
            name = old.name
            candidate = source_file(source, name)
            if digest(old) != digest(candidate): raise ValueError('resource differs: ' + name)
            if name not in mapped: mapped.append(name)
            files.add(name)
        attributes['additional_files'] = mapped
    normalize_files(config.get('model_attributes', {}))
    config['model_path'] = '.'
    for component, expected in zip(components, config['model_component_names']):
        c = component['config']
        if str(component.get('type','')).lower() != 'onnxmodel' or c.get('onnx_file_name') != expected + '.onnx':
            raise ValueError('unexpected ONNX component')
        name = c['onnx_file_name']
        current = source_file(source, name)
        old = (Path(c['model_path']) / name).resolve(strict=True)
        if not old.is_file() or digest(old) != digest(current): raise ValueError('component differs: ' + name)
        files.add(name)
        graph = onnx.load(str(current), load_external_data=False)
        for tensor in graph.graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                info = {entry.key: entry.value for entry in tensor.external_data}
                location = str(relative_name(info['location']))
                source_file(source, location)
                files.add(location)
        normalize_files(c.get('model_attributes', {}))
        c['model_path'] = '.'
    for name in ('genai_config.json', 'config.json', 'tokenizer.json', 'tokenizer_config.json'):
        source_file(source, name)
        files.add(name)
    evidence = [(Path(p).resolve(strict=True), Path(p).name) for p in reports]
    if len({name for _, name in evidence}) != len(evidence): raise ValueError('duplicate report names')
    provenance_path = Path(provenance).resolve(strict=True)
    provenance_data = json.loads(provenance_path.read_text(encoding='utf-8'))
    if not provenance_data.get('source', {}).get('revision'): raise ValueError('missing source revision')
    snapshot = Path(source_snapshot).resolve(strict=True)
    if not snapshot.is_dir() or destination == snapshot or destination.is_relative_to(snapshot) or snapshot.is_relative_to(destination):
        raise ValueError('destination must be separate from snapshot')
    attribution = {}
    for name in ('LICENSE', 'NOTICE.md', 'REPLAY_ATTRIBUTION.md'):
        matches = [item for item in provenance_data['source']['files'] if item.get('path') == name]
        if len(matches) != 1: raise ValueError('missing or duplicate attribution provenance: ' + name)
        path = source_file(snapshot, name)
        record = matches[0]
        if path.stat().st_size != record.get('size') or digest(path) != record.get('sha256'):
            raise ValueError('attribution differs from pinned provenance: ' + name)
        attribution[name] = (path, record['size'], record['sha256'])
    tokenizer = {}
    if preserve_tokenizer_source:
        required = ('tokenizer.json', 'tokenizer_config.json')
        for name in TOKEN_FILES:
            if name not in required and not (snapshot/name).exists(): continue
            matches = [item for item in provenance_data['source']['files'] if item.get('path') == name]
            if len(matches) != 1: raise ValueError('missing or duplicate tokenizer provenance: ' + name)
            path = source_file(snapshot, name)
            record = matches[0]
            if path.stat().st_size != record.get('size') or digest(path) != record.get('sha256'):
                raise ValueError('tokenizer differs from pinned provenance: ' + name)
            tokenizer[name] = (path, record['size'], record['sha256'])
        # Olive resource lists must carry every available tokenizer auxiliary after rebasing.
        for attributes in [config.setdefault('model_attributes', {})] + [c['config'].setdefault('model_attributes', {}) for c in components]:
            if attributes is None: raise ValueError('component model_attributes must be an object')
            refs = attributes.setdefault('additional_files', [])
            for name in tokenizer:
                if name not in refs: refs.append(name)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name in sorted(files):
            target = destination.joinpath(*relative_name(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file(source, name), target)
        for name, (path, size, sha256) in attribution.items():
            if (destination/name).exists(): raise ValueError('attribution conflicts with package resource: ' + name)
            shutil.copyfile(path, destination/name)
            if (destination/name).stat().st_size != size or digest(destination/name) != sha256:
                raise ValueError('copied attribution differs from pinned provenance: ' + name)
        for name, (path, size, sha256) in tokenizer.items():
            shutil.copyfile(path, destination/name)
            if (destination/name).stat().st_size != size or digest(destination/name) != sha256:
                raise ValueError('copied tokenizer differs from pinned provenance: ' + name)
        (destination/'model_config.source.json').write_bytes(original)
        (destination/'model_config.json').write_text(json.dumps(model, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        audit = destination/'audit'; audit.mkdir()
        shutil.copyfile(provenance_path, audit/'prepare-manifest.json')
        for path, name in evidence:
            if name == 'prepare-manifest.json': raise ValueError('report name conflicts with provenance')
            shutil.copyfile(path, audit/name)
        (destination/'README.md').write_text('Pacote experimental não homologado; nenhuma inferência HTP foi realizada. Apenas context e iterator são destinados a compilação HTP; embeddings e lm_head permanecem em CPU. Antes de compilar, use make_qnn_recipe.rebase_composite para resolver recursos relativos neste diretório.\n', encoding='utf-8')
        entries = []
        for path in sorted(destination.rglob('*')):
            if path.is_file(): entries.append({'path': path.relative_to(destination).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)})
        manifest = {'schema':'kapi-qnn-transfer-v1', 'qualified':False, 'htp_inference':False,
                    'source':provenance_data['source'], 'calibration_sha256':provenance_data.get('calibration_sha256'), 'files':entries}
        (destination/'export_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    except BaseException:
        shutil.rmtree(destination)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--destination', required=True)
    p.add_argument('--provenance', required=True); p.add_argument('--report', action='append', default=[])
    p.add_argument('--source-snapshot', required=True)
    p.add_argument('--preserve-tokenizer-source', action='store_true')
    a = p.parse_args(); prepare(a.source, a.destination, a.provenance, a.report, a.source_snapshot, a.preserve_tokenizer_source)


if __name__ == '__main__': main()
