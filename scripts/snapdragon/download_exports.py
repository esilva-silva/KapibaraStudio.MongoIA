"""Download pinned ONNX exports and verify package and source identity."""
import argparse
import json
from pathlib import Path, PurePosixPath
from download_models import file_hash
from verify_manifest import verify


def safe_subfolder(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Invalid subfolder')
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError('Unsafe subfolder')
    return path.as_posix()


def verify_export(package, source):
    package, source = Path(package).resolve(), Path(source).resolve()
    result = verify(package, 'export_manifest.json')
    if not result['passed']:
        raise RuntimeError(f'Export integrity mismatch: {result}')
    manifest = json.loads((package / 'export_manifest.json').read_text(encoding='utf-8-sig'))
    entries = manifest['files']
    listed = {entry['path'] for entry in entries}
    actual = {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    if actual != listed | {'export_manifest.json'}:
        raise ValueError('Export manifest does not cover all package files')
    provenance = manifest.get('source')
    if not isinstance(provenance, dict) or not isinstance(provenance.get('files_sha256'), dict):
        provenance = manifest.get('parent_source_provenance')
    if not isinstance(provenance, dict) or not isinstance(provenance.get('files_sha256'), dict):
        raise ValueError('Export has no source provenance')
    hashes = provenance['files_sha256']
    if not {'model.safetensors', 'config.json', 'tokenizer.json'} <= hashes.keys():
        raise ValueError('Export has incomplete source provenance')
    for path, expected in hashes.items():
        local = (source / path).resolve()
        if not local.is_relative_to(source) or not local.is_file() or file_hash(local) != expected:
            raise RuntimeError(f'Export does not match the pinned source: {path}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--catalog', choices=('published_exports', 'experimental_exports'), default='published_exports')
    parser.add_argument('--models', nargs='+', choices=('nano', 'small'))
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    root = args.root or Path(cfg['models_root'])
    catalog = cfg[args.catalog]
    names = args.models or list(catalog)
    missing = [name for name in names if name not in catalog]
    if missing: parser.error(f'Models unavailable in {args.catalog}: {missing}')
    subfolders = {name: safe_subfolder(catalog[name].get('subfolder', 'dml-fp16')) for name in names}
    from huggingface_hub import snapshot_download
    for name in names:
        spec = catalog[name]; source_spec = cfg['models'][name]; subfolder = subfolders[name]
        source = root / source_spec['repo_id'].split('/')[-1] / source_spec['revision']
        destination = root / spec['repo_id'].split('/')[-1] / spec['revision']
        snapshot_download(repo_id=spec['repo_id'], revision=spec['revision'], local_dir=destination,
                          allow_patterns=[subfolder + '/*', 'README.md'], max_workers=2)
        result = verify_export(destination / Path(*PurePosixPath(subfolder).parts), source)
        result.update(repo_id=spec['repo_id'], revision=spec['revision'], source_repo=source_spec['repo_id'],
                      source_revision=source_spec['revision'], source_identity_verified=True,
                      catalog=args.catalog, subfolder=subfolder)
        (destination / 'download_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result), flush=True)


if __name__ == '__main__': main()

