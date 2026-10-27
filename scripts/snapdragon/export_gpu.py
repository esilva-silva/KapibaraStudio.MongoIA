"""Offline FP16 DirectML export from a pinned Hugging Face snapshot."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

TOKEN_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
               'generation_config.json', 'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def source_provenance(source):
    source = Path(source).resolve()
    for name in ('snapshot_manifest.json', 'source_manifest.json'):
        if (source / name).exists():
            data = json.loads((source / name).read_text(encoding='utf-8'))
            revision = data.get('revision') or data.get('resolved_revision')
            if revision and re.fullmatch(r'[0-9a-fA-F]{40}', revision):
                files = data.get('files', [])
                if not files:
                    raise ValueError('Source manifest must contain verified files')
                for entry in files:
                    path = (source / entry['path']).resolve()
                    if not path.is_relative_to(source) or not path.is_file():
                        raise ValueError('Invalid source manifest path')
                    if path.stat().st_size != entry.get('size', entry.get('bytes')) or sha256(path) != entry['sha256']:
                        raise ValueError(f'Source integrity mismatch: {entry["path"]}')
                covered = {entry['path'] for entry in files}
                required = {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file() and (p.suffix == '.safetensors' or p.name in ('config.json', 'tokenizer.json'))}
                if not required <= covered:
                    raise ValueError('Source manifest does not cover all weights/config/tokenizer')
                return {**data, 'revision': revision, 'path': str(source)}
    raise ValueError('Source must have a pinned 40-character commit in snapshot_manifest.json or snapshot directory name')

def export(source, out, python=sys.executable):
    source, out = Path(source).resolve(), Path(out).resolve()
    prov = source_provenance(source)
    if not (source / 'config.json').is_file() or not (source / 'tokenizer.json').is_file():
        raise ValueError('Source snapshot needs config.json and original tokenizer.json')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be empty; refusing to overwrite an existing package')
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='kapi-export-', dir=out.parent) as cache:
        cmd = [str(python), '-m', 'onnxruntime_genai.models.builder', '-i', str(source),
               '-o', str(out), '-p', 'fp16', '-e', 'dml', '-c', cache]
        subprocess.run(cmd, check=True, env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    cfg_path = out / 'genai_config.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    cfg.setdefault('search', {}).update(do_sample=False, max_length=2048)
    cfg_path.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    for name in (*TOKEN_FILES, 'LICENSE', 'NOTICE', 'NOTICE.md', 'source_manifest.json'):
        if (source / name).is_file():
            shutil.copyfile(source / name, out / name)
    files = [{'path': p.relative_to(out).as_posix(), 'sha256': sha256(p), 'bytes': p.stat().st_size}
             for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'export_manifest.json']
    manifest = {'schema': 'kapi-snapdragon-export-v1', 'source': prov,
                'build': {'command': cmd, 'precision': 'fp16', 'requested_provider': 'DmlExecutionProvider'},
                'context_tokens': 2048, 'max_new_tokens': 32, 'files': files}
    (out / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--python', default=sys.executable)
    a = p.parse_args()
    try:
        export(a.source, a.out, a.python)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        p.exit(1, f'Export failed (no fallback): {exc}\n')

if __name__ == '__main__':
    main()
