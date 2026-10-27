"""Prepare CPU/QNN resume from a completed GPTQ checkpoint; does not run inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def inventory(root):
    result={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('checkpoint must not contain symlinks: '+str(path))
        if path.is_file(): result[path.relative_to(root).as_posix()]={'bytes':path.stat().st_size,'sha256':sha(path)}
    return result


def prepare(recipe_path, model_config_path, status_path, derived, output, cache, result_recipe, report):
    recipe_path=Path(recipe_path).resolve(strict=True); model_config_path=Path(model_config_path).resolve(strict=True)
    status_path=Path(status_path).resolve(strict=True)
    derived=Path(derived).resolve(); output=Path(output).resolve(); cache=Path(cache).resolve()
    result_recipe=Path(result_recipe).resolve(); report=Path(report).resolve()
    targets=[derived, output, cache, result_recipe, report]
    if len(set(targets))!=len(targets) or any(p.exists() for p in targets): raise ValueError('all destinations must be new and distinct')
    for i, p in enumerate(targets):
        for q in targets[i+1:]:
            if p.is_relative_to(q) or q.is_relative_to(p): raise ValueError('destinations must not overlap')
    status=json.loads(status_path.read_text(encoding='utf-8'))
    if status.get('status')!='workflow_returned_artifact_validation_pending' or not any(
        s.get('pass')=='GptqModel' and s.get('status')=='completed' for s in status.get('stages',[])):
        raise ValueError('GPTQ job has not completed')
    model=json.loads(model_config_path.read_text(encoding='utf-8'))
    if str(model.get('type','')).lower()!='hfmodel' or not isinstance(model.get('config'),dict): raise ValueError('expected serialized HfModel')
    source=Path(model['config']['model_path']).resolve(strict=True)
    if not source.is_dir() or source.parent != model_config_path.parent or source.name!='model': raise ValueError('checkpoint must be model/ inside GPTQ output')
    for p in targets:
        if p==source or p.is_relative_to(source) or source.is_relative_to(p) or p==model_config_path.parent or p.is_relative_to(model_config_path.parent):
            raise ValueError('destinations overlap GPTQ output')
    before=inventory(source)
    if 'config.json' not in before or not any(name.endswith('.safetensors') for name in before): raise ValueError('checkpoint missing config or weights')
    original=json.loads((source/'config.json').read_text(encoding='utf-8'))
    if 'max_position_embeddings' not in original: raise ValueError('missing context metadata')
    recipe=json.loads(recipe_path.read_text(encoding='utf-8'))
    if recipe.get('target')!='qnn_system' or recipe['systems']['qnn_system']['accelerators'][0]['execution_providers']!=['QNNExecutionProvider']:
        raise ValueError('QNN target required')
    passes=recipe['passes']; required=('mb','mq','gs','kv_shape','sq','sp','st')
    if any(k not in passes for k in required) or passes['sq']['calibration_providers']!=['CPUExecutionProvider']:
        raise ValueError('required CPU calibration and passes missing')
    recipe['passes']={key:copy.deepcopy(passes[key]) for key in required}; recipe['passes']['st']['context_length']=32
    recipe['input_model']=model; model['config']['model_path']=str(derived)
    if model['config'].get('load_kwargs') is None: model['config']['load_kwargs']={}
    recipe['output_dir']=str(output); recipe['cache_dir']=str(cache)
    derived.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,derived,copy_function=shutil.copyfile)
    updated=copy.deepcopy(original); updated['max_position_embeddings']=2048
    if 'seq_length' in updated: updated['seq_length']=2048
    (derived/'config.json').write_text(json.dumps(updated,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    after=inventory(source); copied=inventory(derived)
    if before!=after or {k:v for k,v in before.items() if k!='config.json'} != {k:v for k,v in copied.items() if k!='config.json'}:
        raise ValueError('source changed or checkpoint copy differs')
    result_recipe.parent.mkdir(parents=True,exist_ok=True); report.parent.mkdir(parents=True,exist_ok=True)
    result_recipe.write_text(json.dumps(recipe,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    summary={'schema':'kapi-qnn-resume-v1','status':'prepared_not_executed','qualified':False,'context':2048,'window':32,
        'gptq_status':str(status_path),'gptq_status_sha256':sha(status_path),'model_config':str(model_config_path),
        'model_config_sha256':sha(model_config_path),'input_recipe_sha256':sha(recipe_path),'output_recipe':str(result_recipe),
        'output_recipe_sha256':sha(result_recipe),'source':str(source),'derived':str(derived),'source_before':before,
        'source_after':after,'derived_files':copied,'source_unchanged':True,'non_config_files_identical':True}
    report.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('recipe','model-config','status','derived','output','cache','result-recipe','report'): p.add_argument('--'+arg,required=True)
    a=p.parse_args(); result=prepare(a.recipe,a.model_config,a.status,a.derived,a.output,a.cache,a.result_recipe,a.report)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
