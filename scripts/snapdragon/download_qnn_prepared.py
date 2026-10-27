"""Download an immutable QNN prepared package and verify its pinned provenance."""
import argparse
import json
from pathlib import Path
import re

from download_exports import safe_subfolder
from export_gpu import source_provenance, sha256, TOKEN_FILES
from verify_manifest import verify


def pin(value, length):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-fA-F]{' + str(length) + '}', value):
        raise ValueError(f'Expected immutable {length}-character hexadecimal pin')
    return value.lower()


def file_inventory(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError('Nonempty provenance files list required')
    normalized = []
    seen = set()
    for entry in entries:
        entry = dict(entry)
        path = safe_subfolder(entry['path'])
        if path in seen:
            raise ValueError('Duplicate provenance path')
        seen.add(path)
        entry['path'] = path
        size = entry.pop('size', entry.get('bytes'))
        if 'bytes' in entry and entry['bytes'] != size:
            raise ValueError('Conflicting byte counts')
        if type(size) is not int or size < 0:
            raise ValueError('Invalid byte count')
        entry['bytes'] = size
        entry['sha256'] = pin(entry['sha256'], 64)
        normalized.append(entry)
    return sorted(normalized, key=lambda entry: entry['path'])


def verify_prepared(package, source, expected_manifest_sha256, require_original_tokenizer=False):
    package, source = Path(package).resolve(), Path(source).resolve()
    expected = pin(expected_manifest_sha256, 64)
    manifest_path = package / 'export_manifest.json'
    if sha256(manifest_path) != expected:
        raise ValueError('Prepared manifest SHA-256 mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    entries = file_inventory(manifest['files'])
    actual = {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    if actual != {e['path'] for e in entries} | {'export_manifest.json'}:
        raise ValueError('Prepared package inventory mismatch')
    result = verify(package, 'export_manifest.json')
    if not result['passed']:
        raise ValueError('Prepared package integrity mismatch')
    prepared_source = (manifest.get('source') or manifest.get('parent_source_provenance')
                       or manifest.get('source_provenance'))
    if not isinstance(prepared_source, dict):
        raise ValueError('Prepared source provenance missing')
    file_inventory(prepared_source['files'])
    raw = source_provenance(source)
    if (prepared_source.get('repo_id') != raw.get('repo_id') or
            prepared_source.get('revision') != raw.get('revision') or
            file_inventory(prepared_source['files']) != file_inventory(raw['files'])):
        raise ValueError('Prepared source identity/provenance mismatch')
    if require_original_tokenizer:
        required = {'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'}
        for name in TOKEN_FILES:
            original, copied = source / name, package / name
            if name in required and not original.is_file():
                raise ValueError('Required original tokenizer file missing: ' + name)
            if original.is_file() and (not copied.is_file() or sha256(original) != sha256(copied)):
                raise ValueError('Original tokenizer mismatch: ' + name)
    result.update(manifest_sha256=expected, source_identity_verified=True,
                  original_tokenizer_required=require_original_tokenizer,
                  prompt_ids_validated=False)
    return result


def download(repo_id, revision, subfolder, root, source, manifest_sha256, require_original_tokenizer=False):
    revision = pin(revision, 40)
    manifest_sha256 = pin(manifest_sha256, 64)
    subfolder = safe_subfolder(subfolder)
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', repo_id) or repo_id.split('/')[-1] in ('.', '..'):
        raise ValueError('Expected owner/repository identifier')
    destination = Path(root).resolve() / repo_id.split('/')[-1] / revision
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=destination,
                      allow_patterns=[subfolder + '/**'], max_workers=2)
    result = verify_prepared(destination / subfolder, source, manifest_sha256, require_original_tokenizer)
    result.update(repo_id=repo_id, revision=revision, subfolder=subfolder)
    (destination / 'download_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo-id', 'revision', 'source', 'manifest-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--subfolder', default='qnn-prepared')
    parser.add_argument('--root', type=Path, default=Path('C:/dev/models'))
    parser.add_argument('--require-original-tokenizer', action='store_true')
    args = parser.parse_args()
    try:
        result = download(args.repo_id, args.revision, args.subfolder, args.root, args.source,
                          args.manifest_sha256, args.require_original_tokenizer)
    except Exception:
        # Do not echo transport exceptions, which may contain authenticated URLs.
        parser.exit(1, 'QNN download or verification failed; no verified result was written.\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
