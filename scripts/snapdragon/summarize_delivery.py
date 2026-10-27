"""Summarize local Snapdragon delivery evidence without changing its status."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(root, name):
    path=root/'reports'/'snapdragon'/name
    return json.loads(path.read_text(encoding='utf-8-sig')), {'path':str(path),'sha256':sha(path)}


def validate_comparisons(comparisons, expected=200):
    hashes=set()
    for name, obj in comparisons.items():
        if obj.get('qualified') is not False or obj.get('proxy') is not True:
            raise ValueError(name+': expected unqualified proxy evidence')
        if obj['candidate'].get('n') != expected or obj['baseline'].get('n') != expected:
            raise ValueError(name+': incomplete comparison')
        corpus=obj.get('corpus_sha256')
        if not isinstance(corpus,str) or not re.fullmatch(r'[0-9a-f]{64}',corpus): raise ValueError(name+': invalid corpus hash')
        hashes.add(corpus)
    if len(hashes)!=1:raise ValueError('Comparison corpus hashes do not match')
    return hashes.pop()


def successful_run(run, corpus, provider, expected_ids=None):
    items=run.get('items') or []
    return (run.get('status')=='completed' and run.get('corpus_sha256')==corpus and len(items)==200
        and len({x.get('id') for x in items})==200 and not any(x.get('error') for x in items)
        and (run.get('provider_evidence') or {}).get('requested')==provider
        and (run.get('provider_evidence') or {}).get('verified') is True
        and (expected_ids is None or [x.get('id') for x in items] == expected_ids)
        and all(type(x.get('generated_tokens')) is int and 0 <= x['generated_tokens'] <= 32 for x in items))


def boundary_ok(report, boundary_hash=None):
    rows=report.get('items') or []
    return (report.get('status')=='completed' and report.get('context_tokens')==2048 and report.get('max_new_tokens')==32 and report.get('greedy') is True and (boundary_hash is None or report.get('corpus_sha256')==boundary_hash) and len(rows)==1 and rows[0].get('prompt_tokens')==2016
        and rows[0].get('generated_tokens')==32 and not rows[0].get('error')
        and (report.get('provider_evidence') or {}).get('verified') is True)


def generation_protocol_ok(run):
    if run.get('executor') == 'memoryboundedreloads':
        batches = run.get('batches') or []
        if not batches: return False
        for batch in batches:
            report_path=Path(batch['report_path']); corpus_path=Path(batch['corpus_path'])
            if sha(report_path)!=batch['report_sha256'] or sha(corpus_path)!=batch['corpus_sha256']: return False
            child=json.loads(report_path.read_text(encoding='utf-8-sig'))
            if not generation_protocol_ok(child): return False
        return True
    return run.get('context_tokens')==2048 and run.get('max_new_tokens')==32 and run.get('greedy') is True


def read_dataset(root, config):
    directory=root/'data/calib/snapdragon-20261026'; manifest_path=directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    rows={}; audit={'manifest':{'path':str(manifest_path),'sha256':sha(manifest_path)}}
    for name,count in [('calibration',config['samples']['calibration_count']),('evaluation',config['samples']['evaluation_count'])]:
        entry=manifest['files'][name]; path=directory/entry['file']; digest=sha(path)
        data=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if digest!=entry['sha256'] or len(data)!=count or entry['count']!=count:
            raise ValueError(name+' dataset hash/count mismatch')
        rows[name]=data; audit[name]={'path':str(path),'sha256':digest,'count':len(data)}
    for key in ('group_id','split_key','schema_id','prompt_sha256'):
        if {x[key] for x in rows['calibration']} & {x[key] for x in rows['evaluation']}:
            raise ValueError('Calibration/evaluation overlap: '+key)
    ids=[x['id'] for x in rows['evaluation']]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate evaluation IDs')
    audit['disjoint_keys']=['group_id','split_key','schema_id','prompt_sha256']
    return audit,ids


def window32_files_ready(root):
    directory = root / 'reports/snapdragon'
    names = ('nano-npu-compiled-window32-200.json', 'nano-npu-window32-vs-bf16.json',
             'nano-npu-window32-200-stage-summary.json', 'nano-npu-window32-boundary-2048.json')
    return all((directory / name).is_file() for name in names)


def assess_window32(comparison, run, stage, boundary, corpus, ids, run_sha, boundary_hash):
    try:
        if validate_comparisons({'window32': comparison}) != corpus:
            return False
        if not successful_run(run, corpus, 'QNNExecutionProvider', ids) or not generation_protocol_ok(run):
            return False
        if stage.get('status') != 'verified' or stage.get('report_sha256') != run_sha:
            return False
        if not boundary_ok(boundary, boundary_hash):
            return False
        if (boundary.get('provider_evidence') or {}).get('requested') != 'QNNExecutionProvider':
            return False
        return True
    except (KeyError, ValueError, TypeError):
        return False


def private_release_ok(config, transfer, source_verification, smoke, evaluated_run):
    spec = config.get('npu_exports', {}).get('nano') or {}
    try:
        local_manifest = Path(spec['local_path']) / 'export_manifest.json'
        release_hash = sha(local_manifest)
        downloaded = Path(source_verification['directory']).resolve()
        smoke_rows = smoke.get('items') or []
        if not (spec.get('private') is True and transfer.get('private') is True
                and transfer.get('status') == 'verified' and transfer.get('repo_id') == spec['repo_id']
                and transfer.get('commit') == spec['revision'] and transfer.get('subfolder') == spec['subfolder']
                and transfer.get('manifest_sha256') == spec['manifest_sha256'] == release_hash
                and source_verification.get('passed') is True and source_verification.get('source_identity_verified') is True
                and source_verification.get('original_tokenizer_required') is True
                and source_verification.get('manifest_sha256') == release_hash
                and smoke.get('status') == 'completed' and Path(smoke['model']).resolve() == downloaded
                and (smoke.get('provider_evidence') or {}).get('requested') == 'QNNExecutionProvider'
                and (smoke.get('provider_evidence') or {}).get('verified') is True
                and len(smoke_rows) == 1 and smoke_rows[0].get('generated_tokens') == 32 and not smoke_rows[0].get('error')):
            return False
        release = json.loads(local_manifest.read_text(encoding='utf-8-sig'))
        evaluated = json.loads((Path(evaluated_run['model']) / 'export_manifest.json').read_text(encoding='utf-8-sig'))
        original = {x['path']: (x['sha256'], x['bytes']) for x in evaluated['files']}
        published = {x['path']: (x['sha256'], x['bytes']) for x in release['files']}
        return bool(original) and all(published.get(k) == v for k, v in original.items())
    except (KeyError, TypeError, OSError, ValueError):
        return False


def summarize(root):
    root=Path(root).resolve(strict=True)
    config_path=root/'configs/snapdragon.json';config=json.loads(config_path.read_text(encoding='utf-8-sig'))
    comparison_names={'nano_gpu':'nano-rmsnorm-split-vs-bf16-detailed.json',
        'small_gpu':'small-rmsnorm-split-batched-vs-bf16.json','nano_npu':'nano-npu-window64-vs-bf16.json'}
    comparisons={}; sources={'config':{'path':str(config_path),'sha256':sha(config_path)}}
    for key,name in comparison_names.items(): comparisons[key],sources[key+'_comparison']=read(root,name)
    corpus=validate_comparisons(comparisons,config['samples']['evaluation_count'])
    dataset,eval_ids=read_dataset(root,config)
    if corpus != dataset['evaluation']['sha256']: raise ValueError('Comparison hash does not match evaluation file')
    boundary_path=root/'tmp/snapdragon-boundary-2016.jsonl'
    boundary_hash=sha(boundary_path) if boundary_path.is_file() else None
    dataset['boundary']={'path':str(boundary_path),'sha256':boundary_hash,'exists':boundary_path.is_file()}
    file_names={'nano_gpu_run':'nano-gpu-rmsnorm-split-200.json',
        'small_gpu_run':'small-gpu-rmsnorm-split-batched-exe-200.json',
        'nano_npu_run':'nano-npu-compiled-window64-200.json',
        'nano_gpu_boundary':'nano-gpu-boundary-2048.json','small_gpu_boundary':'small-gpu-boundary-2048.json',
        'nano_gpu_publication':'nano-split-publication-result.json','small_gpu_publication':'small-split-publication-result.json',
        'small_gpu_single_failure':'small-gpu-rmsnorm-split-200.json',
        'nano_npu_stage_profile':'nano-npu-window64-200-stage-summary.json'}
    data={}
    for key,name in file_names.items(): data[key],sources[key]=read(root,name)
    delivery={}
    for name in ('nano','small'):
        run=data[name+'_gpu_run']; compare=comparisons[name+'_gpu']; pub=data[name+'_gpu_publication']
        passed=successful_run(run,corpus,'DmlExecutionProvider',eval_ids) and generation_protocol_ok(run) and boundary_ok(data[name+'_gpu_boundary'],boundary_hash) and pub.get('status')=='verified' and pub.get('commit')==config['experimental_exports'][name]['revision']
        if name=='small':passed=passed and (run.get('provider_evidence') or {}).get('all_child_verified') is True
        delivery[name+'_gpu']={'status':'validated_200' if passed else 'not_validated',
            'baseline':compare['baseline'],'candidate':compare['candidate'],
            'boundary_2016_plus_32':boundary_ok(data[name+'_gpu_boundary'],boundary_hash),
            'publication':{'status':pub.get('status'),'repo_id':pub.get('repo_id'),'commit':pub.get('commit'),'path_in_repo':pub.get('path_in_repo')}}
    stage=data['nano_npu_stage_profile']
    npu_ok=successful_run(data['nano_npu_run'],corpus,'QNNExecutionProvider',eval_ids) and generation_protocol_ok(data['nano_npu_run']) and stage.get('status')=='verified' and stage.get('report_sha256')==sources['nano_npu_run']['sha256']
    delivery['nano_npu']={'status':'validated_200_boundary_pending' if npu_ok else 'not_validated',
        'baseline':comparisons['nano_npu']['baseline'],'candidate':comparisons['nano_npu']['candidate'],
        'stage_profile_status':stage.get('status'),'prefill_window':64,'cache_tokens':2048,'boundary_2016_plus_32':'pending'}
    if window32_files_ready(root):
        names={'nano_npu32_comparison':'nano-npu-window32-vs-bf16.json',
               'nano_npu32_run':'nano-npu-compiled-window32-200.json',
               'nano_npu32_stage_profile':'nano-npu-window32-200-stage-summary.json',
               'nano_npu32_boundary':'nano-npu-window32-boundary-2048.json'}
        try:
            newer={}
            for key,name in names.items(): newer[key],sources[key]=read(root,name)
            passed=assess_window32(newer['nano_npu32_comparison'],newer['nano_npu32_run'],
                newer['nano_npu32_stage_profile'],newer['nano_npu32_boundary'],corpus,eval_ids,
                sources['nano_npu32_run']['sha256'],boundary_hash)
            comparison=newer['nano_npu32_comparison']
            for key,name in {'nano_npu_private_transfer':'nano-npu-private-transfer-result.json',
                             'nano_npu_private_source':'nano-npu-private-source-verification.json',
                             'nano_npu_private_smoke':'nano-npu-private-download-smoke.json'}.items():
                newer[key],sources[key]=read(root,name)
            release_ok=private_release_ok(config,newer['nano_npu_private_transfer'],newer['nano_npu_private_source'],
                                          newer['nano_npu_private_smoke'],newer['nano_npu32_run'])
            passed=passed and release_ok
            delivery['nano_npu']={'status':'validated_200_boundary_passed' if passed else 'not_validated',
                'baseline':comparison.get('baseline'),'candidate':comparison.get('candidate'),
                'stage_profile_status':newer['nano_npu32_stage_profile'].get('status'),
                'prefill_window':32,'cache_tokens':2048,
                'boundary_2016_plus_32':boundary_ok(newer['nano_npu32_boundary'],boundary_hash),
                'package':{'path':'C:/dev/models/KapiCoder-Mongo-Nano-QNN-Compiled/mixed-htp-window32-v1',
                           'publication':'private_verified' if release_ok else 'not_validated',
                           'release':config.get('npu_exports', {}).get('nano'),
                           'evaluated_package':newer['nano_npu32_run'].get('model'),
                           'release_identity_verified':release_ok}}
            compile_path=root/'reports/snapdragon/nano-qnn-htp-compile-window32.json'
            if compile_path.is_file():
                sources['nano_npu32_compile']={'path':str(compile_path),'sha256':sha(compile_path)}
        except (OSError,ValueError,KeyError,TypeError) as exc:
            delivery['nano_npu']={'status':'not_validated','prefill_window':32,
                'error':'Invalid complete window32 evidence: '+str(exc), 'publication':'not_published'}
    delivery['small_npu']={'status':'pending_remote_preparation'}
    failure=data['small_gpu_single_failure']
    failed_rows=[row for row in failure.get('items',[]) if row.get('error')]
    delivery['small_gpu']['single_session_attempt']={'status':failure.get('status'),'failed_items':len(failed_rows),
        'memory_after':failure.get('memory_after'),'note':'Separate failed single-session attempt; successful result used fresh batches.'}
    return {'schema':'kapi-snapdragon-delivery-summary-v1','completion':False,
        'scope':'proxy evaluation; not IDE qualified', 'corpus_sha256':corpus, 'dataset_audit':dataset,
        'samples':{'calibration_expected':config['samples']['calibration_count'],'evaluation_expected':config['samples']['evaluation_count']},
        'raw_models':config['models'],'published_exports':config.get('published_exports',{}),
        'experimental_exports':config.get('experimental_exports',{}),'npu_exports':config.get('npu_exports',{}),'delivery':delivery,'sources':sources}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Output must not exist')
    try:
        data=summarize(a.root);a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(json.dumps({k:v['status'] for k,v in data['delivery'].items()},indent=2))
    except Exception as exc:p.exit(1,f'Summary failed: {type(exc).__name__}: {exc}\n')

if __name__=='__main__':main()
