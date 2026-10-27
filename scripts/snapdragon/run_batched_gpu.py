"""Run an ordered corpus in sequential fresh DirectML runner processes."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from verify_manifest import verify


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(part)
    return h.hexdigest()


def parse_corpus(path):
    data = Path(path).read_bytes()
    lines = [line for line in data.splitlines(keepends=True) if line.strip()]
    rows = [json.loads(line) for line in lines]
    ids = [row['id'] for row in rows]
    if not rows or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Corpus must contain unique nonempty string IDs')
    return lines, ids


def validate_child(report, expected_ids, corpus_hash, binary_hash, exit_code):
    evidence = report.get('provider_evidence') or {}
    if exit_code or report.get('status') != 'completed': raise ValueError('Child process failed')
    if report.get('corpus_sha256') != corpus_hash: raise ValueError('Child corpus hash mismatch')
    if report.get('runner_binary_sha256') != binary_hash: raise ValueError('Child runner binary hash mismatch')
    rows = report.get('items') or []
    if [r.get('id') for r in rows] != expected_ids: raise ValueError('Child IDs/order mismatch')
    if any(r.get('error') for r in rows): raise ValueError('Child item error')
    if evidence.get('requested') != 'DmlExecutionProvider' or evidence.get('verified') is not True:
        raise ValueError('DML execution not verified')
    return rows


def run(runner, model, corpus, out, work_dir, batch_size=25):
    runner, model, corpus, out, work = map(lambda p: Path(p).resolve(), (runner, model, corpus, out, work_dir))
    if batch_size < 1 or work.exists() or out.exists(): raise ValueError('Batch size must be positive; work-dir and output must be new')
    started = time.perf_counter()
    if runner.suffix.lower() != '.exe': raise ValueError('Use the native .exe apphost so its bundled ORT is selected')
    managed = runner.with_suffix('.dll')
    if not managed.is_file(): raise ValueError('Managed runner DLL missing beside apphost')
    lines, ids = parse_corpus(corpus)
    checked = verify(model)
    if not checked['passed']: raise ValueError('Model manifest verification failed')
    binary_hash = sha(managed)
    launcher_hash = sha(runner)
    manifest_hash = sha(model / checked['manifest'])
    config_hash = sha(model / 'genai_config.json')
    work.mkdir(parents=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'running', 'schema': 'kapi-batched-gpu-v1', 'executor': 'memoryboundedreloads',
        'single_session': False, 'model': str(model), 'corpus_sha256': sha(corpus), 'runner_binary_sha256': binary_hash,
        'runner_launcher_sha256': launcher_hash, 'model_manifest_sha256': manifest_hash, 'model_config_sha256': config_hash, 'batch_size': batch_size,
        'total': len(ids), 'items': [], 'batches': [], 'model_loads_count': 0, 'model_load_seconds': 0,
        'memory_after': {'peak_child_rss_bytes': 0, 'scope': 'max child process peak RSS; not VRAM'}}
    counts = {}; profiles = []
    def save():
        result['wall_elapsed_s'] = time.perf_counter() - started
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    save()
    try:
        for start in range(0, len(lines), batch_size):
            index = start // batch_size
            if sha(managed) != binary_hash or sha(runner) != launcher_hash or sha(model / checked['manifest']) != manifest_hash or sha(model / 'genai_config.json') != config_hash:
                raise ValueError('Runner or model metadata changed during evaluation')
            batch = work / f'batch-{index:03d}.jsonl'
            batch.write_bytes(b''.join(line if line.endswith((b'\n', b'\r')) else line + b'\n' for line in lines[start:start + batch_size]))
            child_path = work / f'batch-{index:03d}.report.json'
            cmd = [str(runner), '--model', str(model), '--corpus', str(batch), '--out', str(child_path), '--backend', 'dml']
            console_path = work / f'batch-{index:03d}.console.log'
            with console_path.open('w', encoding='utf-8') as log:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                for line in process.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
                process.wait()
            info = {'index': index, 'corpus_path': str(batch), 'corpus_sha256': sha(batch), 'report_path': str(child_path), 'exit_code': process.returncode, 'console_path': str(console_path), 'console_sha256': sha(console_path)}
            result['batches'].append(info)
            if not child_path.is_file(): raise ValueError('Child report missing')
            info['report_sha256'] = sha(child_path)
            report = json.loads(child_path.read_text(encoding='utf-8'))
            rows = validate_child(report, ids[start:start + batch_size], info['corpus_sha256'], binary_hash, process.returncode)
            if result.get('runtime_file_versions') is not None and result['runtime_file_versions'] != report.get('runtime_file_versions'):
                raise ValueError('Native runtime versions changed between batches')
            result['runtime_file_versions'] = report.get('runtime_file_versions')
            result['timing_convention'] = report.get('timing_convention')
            info['workspace_commit_at_run'] = report.get('workspace_commit_at_run')
            result['items'].extend(rows)
            result['model_loads_count'] += 1
            result['model_load_seconds'] += report.get('model_load_elapsed_s', 0)
            result['memory_after']['peak_child_rss_bytes'] = max(result['memory_after']['peak_child_rss_bytes'], (report.get('memory_after') or {}).get('peak_rss_bytes', 0))
            evidence = report['provider_evidence']
            for provider, count in evidence.get('kernel_events_by_provider', {}).items(): counts[provider] = counts.get(provider, 0) + count
            profiles.extend(evidence.get('profile_files') or [])
            save()
        if [row['id'] for row in result['items']] != ids: raise ValueError('Aggregate IDs/order mismatch')
        result['status'] = 'completed'
    except Exception as exc:
        result['status'] = 'failed'; result['error'] = f'{type(exc).__name__}: {exc}'
    result['provider_evidence'] = {'requested': 'DmlExecutionProvider', 'verified': result['status'] == 'completed',
        'all_child_verified': result['status'] == 'completed', 'effective': list(counts), 'kernel_events_by_provider': counts,
        'cpu_kernel_events': counts.get('CPUExecutionProvider', 0), 'profile_files': profiles}
    save()
    return 0 if result['status'] == 'completed' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('runner', 'model', 'corpus', 'out', 'work-dir'): p.add_argument('--' + arg, required=True)
    p.add_argument('--batch-size', type=int, default=25)
    a = p.parse_args()
    try: return run(a.runner, a.model, a.corpus, a.out, a.work_dir, a.batch_size)
    except Exception as exc:
        print(f'Batch runner failed before start: {exc}', file=sys.stderr); return 1

if __name__ == '__main__': raise SystemExit(main())
