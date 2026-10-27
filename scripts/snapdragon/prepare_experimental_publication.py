"""Prepare a validated experimental package and offline publication plan; no upload."""
import argparse
import hashlib
import json
import re
from pathlib import PurePosixPath
import shutil
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(source, output, comparison_path, repo, branch, base_commit='b34d90bb97067ae888d51c391898a27c1707fafe', subfolder='dml-fp16-rmsnorm'):
    if not re.fullmatch(r'[0-9a-fA-F]{40}', base_commit):
        raise ValueError('base commit must be a 40-character SHA')
    relative = PurePosixPath(subfolder)
    if not subfolder or '\\' in subfolder or relative.is_absolute() or '..' in relative.parts or not relative.parts or any(c in subfolder for c in ':'):
        raise ValueError('subfolder must be a safe relative path')
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    comparison_path = Path(comparison_path).resolve(strict=True)
    if not branch or branch.lower() in ('main', 'master') or not all(c.isalnum() or c in '-_./' for c in branch) or '..' in branch:
        raise ValueError('A valid non-main experimental branch is required')
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output must be new and separate from source')
    result = json.loads(comparison_path.read_text(encoding='utf-8'))
    if not isinstance(result.get('candidate'), dict) or not isinstance(result.get('baseline'), dict):
        raise ValueError('Comparison must contain candidate and baseline')
    evidence = result['candidate'].get('provider_evidence') or {}
    if (result.get('proxy') is not True or result.get('qualified') is not False
            or result['candidate'].get('n') != 200 or result['baseline'].get('n') != 200
            or evidence.get('verified') is not True or evidence.get('requested') != 'DmlExecutionProvider'):
        raise ValueError('Comparison requires 200 samples per side and verified DML proxy evidence')
    for item in source.rglob('*'):
        if item.is_symlink():
            raise ValueError('Source symlinks are not supported')
    parent = json.loads((source / 'export_manifest.json').read_text(encoding='utf-8'))
    for entry in parent['files']:
        path = (source / entry['path']).resolve()
        if not path.is_relative_to(source) or not path.is_file() or sha(path) != entry['sha256']:
            raise ValueError('Source manifest verification failed: ' + entry['path'])
    shutil.copytree(source, output)
    report_name = 'validation-comparison.json'
    shutil.copy2(comparison_path, output / report_name)
    report_sha = sha(output / report_name)
    validation = {'scope': 'proxy, not IDE-qualified', 'qualified': result.get('qualified'), 'proxy': result.get('proxy'),
        'n': result['candidate'].get('n'), 'execution_protocol': result['candidate'].get('execution_protocol'), 'candidate_trimmed_exact_match': result['candidate'].get('trimmed_exact_match'),
        'baseline_trimmed_exact_match': result['baseline'].get('trimmed_exact_match'),
        'provider_evidence': result['candidate'].get('provider_evidence'), 'limitations': result.get('limitations', []),
        'comparison_file': report_name, 'comparison_sha256': report_sha}
    protocol = result['candidate'].get('execution_protocol') or {}
    protocol_note = '' if protocol.get('single_session', True) else (
        f"Validated with a fresh process every {protocol.get('batch_size')} requests. "
        'Use scripts/snapdragon/run_batched_gpu.py; single-session stability is not established. ')
    note = output / 'RMSNORM_TRANSFORM.md'
    note.write_text('# Experimental RMSNorm graph transformation\n\n'
        'Graph-only transformation; no new training and FP16 weights unchanged. Graph optimization must remain disabled to prevent refusion. '
        'The DirectML issue is an analogy, not proof of the cause on this device: https://github.com/microsoft/onnxruntime/issues/29739\n\n'
        f"Proxy comparison: {validation['n']} prompts; candidate exact match {validation['candidate_trimmed_exact_match']}, "
        f"BF16 baseline {validation['baseline_trimmed_exact_match']}. This is not IDE qualification. "
        + protocol_note + 'See validation-comparison.json for provider evidence, CPU participation and limitations.\n', encoding='utf-8')
    split_note = output / 'SPLIT_EMBEDDING.md'
    if split_note.exists():
        split_note.write_text(split_note.read_text(encoding='utf-8').rstrip() +
            '\n\nProxy comparison completed: see [validation-comparison.json](validation-comparison.json). '
            + protocol_note + 'This evidence does not constitute IDE qualification.\n', encoding='utf-8')
    kapi_path = output / 'kapicoder_manifest.json'
    if kapi_path.exists():
        kapi = json.loads(kapi_path.read_text(encoding='utf-8'))
        entries = kapi.setdefault('files', [])
        for name in (report_name, note.name):
            if not any(e['path'] == name for e in entries):
                entries.append({'path': name})
        for entry in entries:
            path = (output / entry['path']).resolve()
            if not path.is_relative_to(output) or not path.is_file():
                raise ValueError('Invalid internal manifest path')
            entry.update(sha256=sha(path), bytes=path.stat().st_size)
        kapi['validation'] = validation
        kapi_path.write_text(json.dumps(kapi, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    parent['validation'] = validation
    parent['validation_parent_export_manifest_sha256'] = sha(source / 'export_manifest.json')
    parent['files'] = [{'path': p.relative_to(output).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                       for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'export_manifest.json']
    (output / 'export_manifest.json').write_text(json.dumps(parent, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    plan = {'status': 'not_published', 'repo_id': repo, 'revision': branch, 'source_package': str(source),
        'prepared_package': str(output), 'validation_sha256': report_sha, 'base_commit': base_commit.lower(), 'path_in_repo': subfolder,
        'upload': 'Create or use the non-main branch, then upload the complete prepared directory to that branch.',
        'verification': 'Resolve the resulting immutable commit, list repository files at that commit and download/hash each file against export_manifest.json; also hash the remote export_manifest.json against this local copy.',
        'manifest_sha256': sha(output / 'export_manifest.json')}
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--comparison', required=True, type=Path)
    p.add_argument('--repo', required=True)
    p.add_argument('--branch', required=True)
    p.add_argument('--base-commit', default='b34d90bb97067ae888d51c391898a27c1707fafe')
    p.add_argument('--subfolder', default='dml-fp16-rmsnorm')
    p.add_argument('--plan', required=True, type=Path)
    a = p.parse_args()
    try:
        plan = prepare(a.source, a.out, a.comparison, a.repo, a.branch, a.base_commit, a.subfolder)
        a.plan.parent.mkdir(parents=True, exist_ok=True)
        a.plan.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps(plan, indent=2, ensure_ascii=False))
    except (ValueError, OSError, KeyError) as exc:
        p.exit(1, f'Preparation failed: {exc}\n')


if __name__ == '__main__':
    main()
