"""Publish a reviewed experimental plan and verify every file at the immutable commit."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

BASE = 'b34d90bb97067ae888d51c391898a27c1707fafe'
SUBFOLDER = 'dml-fp16-rmsnorm'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(plan):
    root = Path(plan['prepared_package']).resolve(strict=True)
    manifest = root / 'export_manifest.json'
    if sha(manifest) != plan['manifest_sha256']:
        raise ValueError('Local export manifest differs from reviewed plan')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    expected = {}
    for entry in data['files']:
        rel = PurePosixPath(entry['path'])
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:
            raise ValueError('Unsafe manifest path')
        path = (root / Path(*rel.parts)).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != entry['sha256']:
            raise ValueError('Local hash mismatch: ' + entry['path'])
        if entry['path'] in expected:
            raise ValueError('Duplicate manifest path')
        expected[entry['path']] = entry['sha256']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(expected) | {'export_manifest.json'}:
        raise ValueError('Local package contains unexpected or missing files')
    expected['export_manifest.json'] = plan['manifest_sha256']
    return root, expected


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def publish(plan_path, report_path, cache_dir):
    from huggingface_hub import HfApi, hf_hub_download
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    branch = plan['revision']
    if not branch.startswith('experimental/') or '..' in branch:
        raise ValueError('Only experimental/ branches are allowed')
    root, expected = inventory(plan)
    report_path = Path(report_path)
    cache = Path(cache_dir).resolve()
    if not cache.is_relative_to(Path('C:/dev').resolve()):
        raise ValueError('Verification cache must be under C:/dev')
    base = plan.get('base_commit', BASE)
    subfolder = plan.get('path_in_repo', SUBFOLDER)
    if not re.fullmatch(r'[0-9a-fA-F]{40}', base):
        raise ValueError('Invalid base commit')
    rel = PurePosixPath(subfolder)
    if not subfolder or '\\' in subfolder or rel.is_absolute() or '..' in rel.parts or not rel.parts or any(c in subfolder for c in ':'):
        raise ValueError('Unsafe path_in_repo')
    base = base.lower()
    report = {'repo_id': plan['repo_id'], 'branch': branch, 'base_commit': base,
              'path_in_repo': subfolder, 'manifest_sha256': plan['manifest_sha256'], 'verified': {}, 'status': 'preparing'}
    api = HfApi()
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding='utf-8'))
        for key in ('repo_id', 'branch', 'base_commit', 'path_in_repo', 'manifest_sha256'):
            if previous.get(key) != report[key]:
                raise ValueError('Existing report does not match plan: ' + key)
        report = previous
    commit = report.get('commit')
    if commit:
        if not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise ValueError('Invalid recorded immutable commit')
    else:
        api.create_branch(repo_id=plan['repo_id'], branch=branch, revision=base, exist_ok=True)
        head = api.repo_info(plan['repo_id'], revision=branch).sha
        if head != base:
            raise ValueError('Existing branch has advanced; refusing overwrite without a saved publication report')
        report.update(status='uploading', parent_commit=head)
        write_report(report_path, report)
        info = api.upload_folder(repo_id=plan['repo_id'], folder_path=root, path_in_repo=subfolder,
            revision=branch, parent_commit=head,
            commit_message=plan.get('commit_message', 'Add experimental Snapdragon RMSNorm FP16 package'))
        commit = info.oid
        if not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise ValueError('Upload did not return an immutable commit SHA')
        report.update(status='verifying', commit=commit)
        write_report(report_path, report)
    remote_files = set(api.list_repo_files(plan['repo_id'], revision=commit))
    for name, expected_hash in expected.items():
        remote_name = f'{subfolder}/{name}'
        if remote_name not in remote_files:
            raise ValueError('Remote file missing: ' + remote_name)
        path = hf_hub_download(plan['repo_id'], filename=remote_name, revision=commit, cache_dir=cache)
        actual_hash = sha(path)
        if actual_hash != expected_hash:
            raise ValueError('Remote hash mismatch: ' + remote_name)
        report.setdefault('verified', {})[name] = actual_hash
        write_report(report_path, report)
    report['status'] = 'verified'
    write_report(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = publish(args.plan, args.report, args.cache_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        parser.exit(1, f'Publication failed ({type(exc).__name__}); see existing report and retry if safe.\n')


if __name__ == '__main__':
    main()
