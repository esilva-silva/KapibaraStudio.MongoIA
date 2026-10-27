"""Verify manifested dataset ZIPs and unpack into new, isolated directories."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name):
    if (not isinstance(name, str) or not name or '\\' in name
            or PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive
            or any(part in ('', '.', '..') or ':' in part for part in name.split('/'))):
        raise ValueError('Unsafe archive path')
    return name


def unpack(manifest_path, destination):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('schema') != 'kapicoder-training-archive-v1':
        raise ValueError('Unknown dataset archive schema')
    archive_info = manifest['zip']
    archive_name = safe_name(archive_info['file'])
    if '/' in archive_name:
        raise ValueError('Archive must be next to manifest')
    archive = manifest_path.parent / archive_name
    if archive.stat().st_size != archive_info['bytes'] or digest(archive) != archive_info['sha256']:
        raise ValueError('Archive size or SHA-256 mismatch')
    expected = {}
    for entry in manifest['files']:
        name = safe_name(entry['path'])
        if name in expected or type(entry['bytes']) is not int or entry['bytes'] < 0:
            raise ValueError('Duplicate or invalid manifest entry')
        expected[name] = entry
    if not expected:
        raise ValueError('Empty dataset manifest')
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError('Destination must be new')
    verified = []
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        names = [safe_name(item.filename) for item in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('Archive inventory differs from manifest')
        for item in members:
            entry = expected[item.filename]
            if (item.is_dir() or stat.S_ISLNK(item.external_attr >> 16)
                    or item.file_size != entry['bytes']):
                raise ValueError('Archive member type or size mismatch')
        destination.mkdir(parents=True, exist_ok=False)
        for item in members:
            target = destination.joinpath(*PurePosixPath(item.filename).parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError('Archive target escaped destination')
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with source.open(item) as reader, target.open('xb') as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b''):
                    h.update(chunk)
                    writer.write(chunk)
            if h.hexdigest() != expected[item.filename]['sha256']:
                raise ValueError('Dataset member SHA-256 mismatch: ' + item.filename)
            verified.append({'path': item.filename, 'bytes': item.file_size, 'sha256': h.hexdigest()})
    return {'model': manifest['model'], 'status': 'all_files_verified_and_extracted',
            'source_manifest': str(manifest_path.resolve()), 'source_manifest_sha256': digest(manifest_path),
            'archive': str(archive.resolve()), 'archive_sha256': archive_info['sha256'],
            'directory': str(destination), 'train_examples_from_manifest': manifest['train_examples'],
            'published_model_revision': manifest['published_model_revision'],
            'provenance': manifest['provenance'], 'scope': manifest['scope'],
            'verified_files': verified, 'total_uncompressed_bytes': sum(x['bytes'] for x in verified)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Report must be new')
    result = {'schema': 'kapi-training-dataset-extraction-v1', 'status': 'verifying', 'models': []}
    try:
        for model in ('Nano', 'Small'):
            path = args.directory / f'KapiCoder-Mongo-{model}-training-validation-v3.manifest.json'
            result['models'].append(unpack(path, args.directory / model.lower()))
        result['status'] = 'verified'
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'status': result['status'], 'models': len(result['models']), 'error': result.get('error')}))
    return 0 if result['status'] == 'verified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
