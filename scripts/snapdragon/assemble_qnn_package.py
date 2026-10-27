"""Assemble an experimental mixed CPU/QNN GenAI package; no inference is run."""
import argparse
import json
import shutil
from pathlib import Path, PurePosixPath
from compile_qnn_context import context_files, sha
from verify_manifest import verify
from export_gpu import source_provenance
from download_qnn_prepared import file_inventory

TOKEN_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
               'generation_config.json', 'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja', 'LICENSE', 'NOTICE', 'NOTICE.md')


def safe_file(root, name):
    rel = PurePosixPath(str(name).replace('\\', '/'))
    if rel.is_absolute() or not rel.parts or '..' in rel.parts:
        raise ValueError('Unsafe resource path')
    path = (root / Path(*rel.parts)).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file(): raise ValueError('Resource outside root')
    return path


def external_files(model, root):
    import onnx
    graph = onnx.load(str(model), load_external_data=False)
    files = set()
    for tensor in graph.graph.initializer:
        metadata = {e.key: e.value for e in tensor.external_data}
        if 'location' in metadata:
            loc = PurePosixPath(metadata['location'].replace('\\', '/'))
            if loc.is_absolute() or '..' in loc.parts: raise ValueError('Unsafe external data path')
            path = (model.parent / Path(*loc.parts)).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file(): raise ValueError('External data outside prepared package')
            files.add(path)
    return files


def validate_source_identity(expected, actual):
    if (expected.get('revision') != actual.get('revision') or expected.get('repo_id') != actual.get('repo_id')
            or file_inventory(expected.get('files')) != file_inventory(actual.get('files'))):
        raise ValueError('Tokenizer snapshot source identity or hashes do not match prepared model')


def assemble(prepared, compiled, source_tokenizer, out_dir):
    prepared = Path(prepared).resolve(strict=True); compiled = Path(compiled).resolve(strict=True)
    source_tokenizer = Path(source_tokenizer).resolve(strict=True); output = Path(out_dir).resolve()
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in (prepared, compiled, source_tokenizer)):
        raise ValueError('Output must be new and disjoint from inputs')
    verification = verify(prepared, 'export_manifest.json')
    if not verification['passed']: raise ValueError('Prepared manifest verification failed')
    source_prov = source_provenance(source_tokenizer)
    prepared_manifest = json.loads((prepared / 'export_manifest.json').read_text(encoding='utf-8'))
    expected_source = prepared_manifest.get('source') or prepared_manifest.get('parent_source_provenance') or {}
    validate_source_identity(expected_source, source_prov)
    if not (source_tokenizer / 'tokenizer.json').is_file(): raise ValueError('Original tokenizer.json missing')
    report_path = compiled / 'compile-report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('status') != 'compiled_not_inference_validated' or report.get('cpu_fallback_disabled') is not True or report.get('device_type') != 'NPU':
        raise ValueError('Compile report is not a successful NPU compilation')
    artifacts = {}
    for entry in report['artifacts']:
        path = safe_file(compiled, entry['path'])
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']: raise ValueError('Compiled artifact mismatch')
        if entry['path'] in artifacts: raise ValueError('Duplicate compiled artifact')
        artifacts[entry['path']] = path
    model_config = json.loads((prepared / 'model_config.json').read_text(encoding='utf-8'))
    if model_config.get('type', '').lower() != 'compositemodel': raise ValueError('Expected Olive CompositeModel')
    config = json.loads((prepared / 'genai_config.json').read_text(encoding='utf-8'))
    decoder = config['model']['decoder']
    if config['model'].get('context_length') != 2048: raise ValueError('Expected context length 2048')
    stages = decoder.get('pipeline')
    if not isinstance(stages, list) or len(stages) != 1 or not isinstance(stages[0], dict): raise ValueError('Expected grouped pipeline')
    group = stages[0]
    if set(group) != {'embeddings', 'context', 'iterator', 'lm_head'}: raise ValueError('Unexpected pipeline components')
    if (decoder.get('session_options') or {}).get('provider_options'): raise ValueError('Decoder default must be CPU')
    compiled_inputs = report['inputs']
    if len(compiled_inputs) != 2: raise ValueError('Expected exactly context and iterator compile inputs')
    copy_map = {}
    for name in ('embeddings', 'lm_head'):
        stage = group[name]
        if (stage.get('session_options') or {}).get('provider_options'): raise ValueError('CPU stage has provider options')
        path = safe_file(prepared, stage['filename'])
        copy_map[path.relative_to(prepared)] = path
        for dep in external_files(path, prepared): copy_map[dep.relative_to(prepared)] = dep
    for index, name in enumerate(('context', 'iterator')):
        stage = group[name]; source_path = safe_file(prepared, stage['filename'])
        if sha(source_path) != compiled_inputs[index]['sha256']: raise ValueError('Compiled input does not match prepared stage')
        filename = f'{index:02d}_{source_path.stem}_ctx.onnx'
        if filename not in artifacts: raise ValueError('Expected compiled wrapper not found: ' + filename)
        wrapper = artifacts[filename]
        copy_map[Path(filename)] = wrapper
        for binary in context_files(wrapper):
            rel = binary.relative_to(compiled).as_posix()
            if rel not in artifacts: raise ValueError('Context binary absent from compile report')
            copy_map[Path(rel)] = binary
        stage['filename'] = filename
        options = stage.setdefault('session_options', {})
        options['disable_cpu_ep_fallback'] = '1'
        options['provider_options'] = [{'qnn': {'backend_type': 'htp'}}]
    output.mkdir(parents=True)
    for rel, src in copy_map.items():
        dst = output / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        if sha(src) != sha(dst): raise IOError('Copy integrity mismatch: ' + str(rel))
    for name in TOKEN_FILES:
        source = source_tokenizer / name
        if source.is_file():
            shutil.copy2(source, output / name)
            if sha(source) != sha(output / name): raise IOError('Tokenizer copy mismatch')
    (output / 'genai_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    files = [{'path': p.relative_to(output).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'export_manifest.json']
    manifest = {'schema':'kapi-qnn-mixed-package-v1', 'status':'compiled_not_inference_validated',
        'prepared_manifest_sha256':sha(prepared/'export_manifest.json'), 'compile_report_sha256':sha(report_path),
        'derivation_parent_manifest_sha256':prepared_manifest.get('parent_manifest_sha256'),
        'derivation_metadata_dimensions_changed':prepared_manifest.get('metadata_dimensions_changed'),
        'derivation_window':prepared_manifest.get('window'),
        'source_provenance':source_prov, 'cpu_stages':['embeddings','lm_head'], 'qnn_stages':['context','iterator'],
        'files':files}
    (output/'export_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('prepared','compiled','source-tokenizer','out-dir'): p.add_argument('--'+arg,required=True)
    a=p.parse_args()
    try: print(json.dumps(assemble(a.prepared,a.compiled,a.source_tokenizer,a.out_dir),indent=2))
    except Exception as exc: p.exit(1,f'Assembly failed: {type(exc).__name__}: {exc}\n')

if __name__ == '__main__': main()
