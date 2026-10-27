"""Derive StaticLLM window 32 from an existing SplitModel, on CPU only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def file_manifest(root):
    return [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(root.rglob('*')) if p.is_file()]


def derive(footprint, recipe, output, cache, report):
    footprint = Path(footprint).resolve(strict=True); recipe = Path(recipe).resolve(strict=True)
    output = Path(output).resolve(); cache = Path(cache).resolve(); report = Path(report).resolve()
    if output.exists() or cache.exists() or report.exists(): raise FileExistsError('output, cache and report must be new')
    if output == cache or output.is_relative_to(cache) or cache.is_relative_to(output): raise ValueError('output and cache must be separate')
    graph = json.loads(footprint.read_text(encoding='utf-8'))
    nodes = [(key, node) for key, node in graph.items() if node.get('from_pass') == 'splitmodel']
    if len(nodes) != 1: raise ValueError('expected exactly one SplitModel result')
    model_id, node = nodes[0]
    model = copy.deepcopy(node['model_config_data'])
    if model.get('type','').lower() != 'compositemodel': raise ValueError('expected CompositeModel')
    source = Path(model['config']['model_path']).resolve(strict=True)
    if not source.is_dir(): raise ValueError('split source must be a directory')
    for p in (output, cache, report):
        if p == source or p.is_relative_to(source) or source.is_relative_to(p): raise ValueError('destination overlaps split source')
    config = json.loads(recipe.read_text(encoding='utf-8'))
    if config.get('target') != 'qnn_system' or config['systems']['qnn_system']['accelerators'][0]['execution_providers'] != ['QNNExecutionProvider']:
        raise ValueError('QNN target required')
    if config['passes']['st']['type'].lower() != 'staticllm': raise ValueError('StaticLLM pass required')
    if model['config']['model_attributes']['max_position_embeddings'] != 2048: raise ValueError('expected context 2048')
    before = file_manifest(source)
    summary = {'status':'starting','qualified':False,'htp_inference':False,'scope':'StaticLLM only, CPU; no GPTQ or calibration',
               'source_model_id':model_id,'source_parent_model_id':node.get('parent_model_id'),'source_dir':str(source),
               'source_footprint':str(footprint),'source_footprint_sha256':digest(footprint), 'recipe_sha256':digest(recipe),
               'context_length':2048,'past_tokens':2016,'window_size':32,'source_before':before}
    started = time.monotonic()
    report.parent.mkdir(parents=True, exist_ok=True)
    try:
        if os.environ.get('HIP_VISIBLE_DEVICES') != '-1': raise RuntimeError('HIP_VISIBLE_DEVICES=-1 must be set before process start')
        import torch, onnxruntime as ort
        if torch.cuda.is_available() or torch.cuda.device_count(): raise RuntimeError('GPU is visible')
        if 'QNNExecutionProvider' not in ort.get_available_providers(): raise RuntimeError('QNN plugin not registered')
        summary['providers'] = ort.get_available_providers(); summary['gpu_visible_count'] = 0
        # Copy (not hardlink) to protect the original split and its shared files from Olive mutations.
        cache.mkdir(parents=True, exist_ok=False)
        copied = cache/'source-split'; shutil.copytree(source, copied, copy_function=shutil.copyfile)
        if file_manifest(copied) != before: raise RuntimeError('split copy differs from source')
        old = source
        def rewrite(obj):
            if isinstance(obj, dict):
                for key, value in list(obj.items()):
                    if key == 'model_path' and value is not None:
                        path = Path(value).resolve(strict=True)
                        if path != old: raise ValueError('unexpected component model_path')
                        obj[key] = str(copied)
                    elif key == 'additional_files' and value is not None:
                        mapped = []
                        for item in value:
                            path = Path(item).resolve(strict=True)
                            if not path.is_relative_to(old) or not path.is_file(): raise ValueError('resource outside split')
                            mapped.append(str(copied/path.relative_to(old)))
                        obj[key] = mapped
                    else: rewrite(value)
            elif isinstance(obj, list):
                for value in obj: rewrite(value)
        rewrite(model)
        config['input_model'] = model; config['data_configs'] = []
        st = dict(config['passes']['st']); st['context_length'] = 32
        config['passes'] = {'st': st}; config['output_dir'] = str(output); config['cache_dir'] = str(cache/'olive')
        config['no_artifacts'] = False
        (cache/'derive-config.json').write_text(json.dumps(config, indent=2)+'\n',encoding='utf-8')
        torch.compiler.set_stance('force_eager')
        from olive.workflows import run as olive_run
        run_start = time.monotonic(); result = olive_run(config, tempdir=str(cache/'tmp'))
        summary['staticllm_seconds'] = round(time.monotonic()-run_start,3)
        if not result or not (output/'model_config.json').is_file(): raise RuntimeError('StaticLLM produced no output model config')
        genai = json.loads((output/'genai_config.json').read_text(encoding='utf-8'))
        decoder = genai['model']['decoder']
        if genai['model']['context_length'] != 2048 or decoder['sliding_window']['window_size'] != 32:
            raise RuntimeError('unexpected generated context or window')
        summary['output_files'] = file_manifest(output)
        summary['status'] = 'derived_validation_pending'
    except Exception as error:
        summary.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    summary['source_after'] = file_manifest(source)
    summary['source_unchanged'] = summary['source_before'] == summary['source_after']
    if not summary['source_unchanged']: summary.update(status='failed',error='source files changed')
    summary['wall_seconds'] = round(time.monotonic()-started,3)
    report.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('footprint','recipe','output','cache','report'): p.add_argument('--'+arg, required=True)
    a=p.parse_args(); result=derive(a.footprint,a.recipe,a.output,a.cache,a.report)
    print(json.dumps(result,indent=2)); raise SystemExit(0 if result['status']=='derived_validation_pending' else 1)


if __name__=='__main__': main()
