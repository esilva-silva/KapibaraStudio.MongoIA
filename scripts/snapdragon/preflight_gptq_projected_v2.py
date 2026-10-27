"""Real CPU-only preflight of projected GPTQ DataConfig; no model/GPU/pass execution."""
from __future__ import annotations
import argparse, copy, hashlib, importlib.metadata, inspect, json, logging, os, sys, tempfile, traceback, types
from datetime import datetime,timezone
from pathlib import Path
from preflight_fim_exact_tail_v1 import configure_temp,sha,require,json_file,ROOT,BASE,EXPECTED_STREAM

def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def canonical_sha(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def activation_equivalence():
    old=BASE/'activation'; new=BASE/'activation-v2-projected'
    old_pre=json_file(old/'preflight.json');new_pre=json_file(new/'preflight.json')
    keys=('items','policy','count','dataset_source_sha256','tokenizer_sha256','calibration_text_sha256')
    for key in keys: require(old_pre[key]==new_pre[key],f'Activation preflight differs: {key}')
    old_recipe=json_file(old/'recipe-fim-activation-v1.json');new_recipe=json_file(new/'recipe-fim-activation-v1.json')
    def act_config(recipe): return next(c for c in recipe['data_configs'] if c['name']==recipe['passes']['sq']['data_config'])
    old_config=act_config(old_recipe);new_config=act_config(new_recipe)
    require(old_config['pre_process_data_config']==new_config['pre_process_data_config'],'Activation preprocessing changed')
    require(old_config['dataloader_config']==new_config['dataloader_config'],'Activation DataLoader changed')
    old_data=Path(old_config['load_dataset_config']['data_files']);new_data=Path(new_config['load_dataset_config']['data_files'])
    require(sha(old_data)==sha(new_data)==new_pre['calibration_text_sha256'],'Activation input bytes changed')
    prior_path=ROOT/'reports/snapdragon/nano-fim-exact-tail-activation-reader-real-preflight-rx7800xt-v3.json'
    prior=json_file(prior_path)
    require(prior['status']=='passed' and prior['inference_count']==2,'Prior activation proof missing')
    require(prior['guard_validation']['record']['fim_preflight']['matched']==256 and prior['guard_validation']['record']['fim_preflight']['all_match'] is True,'Prior actual reader guard proof missing')
    require(prior['recipe_sha256']==sha(old/'recipe-fim-activation-v1.json'),'Prior tested activation recipe no longer matches')
    return {'equivalent':True,'same_items_policy_count_source_tokenizer':True,'same_preprocess_and_dataloader':True,
        'old_preflight':str(old/'preflight.json'),'old_preflight_sha256':sha(old/'preflight.json'),
        'new_preflight':str(new/'preflight.json'),'new_preflight_sha256':sha(new/'preflight.json'),
        'items_canonical_sha256':canonical_sha(new_pre['items']),'policy':new_pre['policy'],'count':new_pre['count'],
        'calibration_jsonl_sha256':sha(new_data),'new_recipe_sha256':sha(new/'recipe-fim-activation-v1.json'),
        'prior_real_reader_and_two_inference_report':str(prior_path),'prior_report_sha256':sha(prior_path),
        'new_inference_count':0,'scope':'Exact data/preprocessing equivalence to the already tested reader contract; no model/inference rerun and no statement about future quantized-weight quality'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,default=BASE/'gptq-only-v2.json')
    p.add_argument('--out',type=Path,default=ROOT/'reports/snapdragon/nano-fim-exact-tail-gptq-real-preflight-rx7800xt-v2.json')
    a=p.parse_args()
    require(a.out.resolve().is_relative_to((ROOT/'reports').resolve()),'Output must be under worktree reports')
    require(not a.out.exists(),'Refusing to overwrite existing report')
    temp=Path(tempfile.mkdtemp(prefix='kapi-gptq-projected-real-preflight-'));configure_temp(temp)
    os.environ['HF_DATASETS_IN_MEMORY_MAX_SIZE']='67108864'
    report={'status':'starting','created_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),
        'recipe':str(a.recipe),'recipe_sha256':sha(a.recipe),'isolated_temp':str(temp),
        'scope':'Real DataConfig/Olive plus installed GPTQ data preparation on CPU only; no model/GPU/quantization/activation-calibration/inference',
        'inference_count':0,'model_loaded':False,'GPU_execution':False,'quantization_run':False}
    try:
        import torch,olive
        from olive.data.config import DataConfig
        from olive.passes.pytorch.train_utils import get_calibration_dataset
        from audit_gptq_calibration_tokens import installed_function
        from run_olive_fwht import validate_calibration_batches
        require(importlib.metadata.version('transformers')=='4.57.6','Use original probe environment')
        recipe=json_file(a.recipe); require(set(recipe['passes'])=={'cs','g'},'Expected GPTQ-only recipe')
        raw_config=next(c for c in recipe['data_configs'] if c['name']==recipe['passes']['g']['data_config'])
        reserved=Path(raw_config['pre_process_data_config']['audit_path'])
        require(not reserved.exists(),'Reserved future-job runtime ledger already exists')
        ledger_path=reserved.with_name('preflight-token-ledger.json');ledger=json_file(ledger_path)
        projection_path=Path(raw_config['load_dataset_config']['data_files'])
        projection=rows(projection_path)
        originals_path=ROOT/'data/calib/snapdragon-20261026/calibration.jsonl'; originals=rows(originals_path)
        require(sha(originals_path)=='179736eed969a8a961af6d3fc410a3eead24306234d65ccf245c20cb16978b60','Frozen corpus SHA differs')
        require(len(projection)==len(originals)==256,'Expected256 rows')
        fields=('id','prompt','prompt_text','prompt_ids')
        require(all(set(r)==set(fields) and all(r[k]==o[k] for k in fields) for r,o in zip(projection,originals)),'Projection changed calibration fields/order')
        runtime=temp/'preflight-only-runtime-token-ledger.json';require(not runtime.exists(),'Temporary ledger must be new')
        cloned=copy.deepcopy(raw_config);cloned['pre_process_data_config']['audit_path']=str(runtime)
        expected=copy.deepcopy(raw_config);expected['pre_process_data_config']['audit_path']=str(runtime)
        require(cloned==expected,'Unexpected DataConfig modifications')
        # Only audit_path is changed. HF caches are isolated via environment, not recipe kwargs.
        data_config=DataConfig(**cloned)
        batches=get_calibration_dataset(None,data_config)
        before=validate_calibration_batches(batches,ledger)
        actual_ledger=json_file(runtime)
        require(actual_ledger['batches']==ledger['batches'],'Actual/provided batch ledgers differ')
        require(len(actual_ledger['prompts'])==256 and all(x['omitted_prompt_tokens']==0 for x in actual_ledger['prompts']),'Not all256 prompt token ranges covered')
        site=Path(olive.__file__).resolve().parent.parent
        ns={'torch':torch,'LongTensor':torch.LongTensor,'Tensor':torch.Tensor,'log':logging.getLogger('gptq_preflight')}
        _,collate=installed_function(site/'gptqmodel/utils/data.py','collate_data',ns)
        prepare,prepare_source=installed_function(site/'gptqmodel/models/base.py','prepare_dataset',ns,'BaseGPTQModel')
        prepared=prepare(types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=151643),support_batch_quantize=True),batches)
        after=validate_calibration_batches(prepared,ledger)
        require(before==after,'Installed GPTQ prepare/collate changed any tokens/masks/order')
        require(before['batch_count']==30 and before['token_count']==61273 and before['stream_token_ids_sha256']==EXPECTED_STREAM,'Wrong GPTQ token stream')
        require([x['tokens'] for x in before['batches']]==[2048]*29+[1881],'Wrong batch/tail lengths')
        require(not reserved.exists(),'Preflight consumed the reserved job ledger path')
        require(sha(a.recipe)==report['recipe_sha256'],'Recipe changed during preflight')
        report.update(status='passed',versions={n:importlib.metadata.version(n) for n in ('transformers','olive-ai','gptqmodel','datasets','torch','tokenizers')},
            projection={'path':str(projection_path),'sha256':sha(projection_path),'rows':256,'preserved_fields':list(fields),'original_sha256':sha(originals_path),'same_fields_and_order':True},
            only_manual_DataConfig_override={'field':'pre_process_data_config.audit_path','original_reserved_path':str(reserved),'preflight_path':str(runtime)},
            reserved_future_job_ledger={'existed_before':False,'exists_after':False,'untouched':True},
            resolved_dataloader={'type':data_config.dataloader_config.type,'params':data_config.dataloader_params},
            actual_runtime_preprocessor_ledger={'path':str(runtime),'sha256':sha(runtime)},
            validation_before_gptq_prepare=before,validation_after_gptq_prepare=after,
            installed_gptq_data_functions={'prepare_dataset':prepare_source,'collate_data':collate},
            source_files={str(f):sha(f) for f in (ledger_path,ROOT/'scripts/snapdragon/gptq_join_tail_v1.py',ROOT/'scripts/snapdragon/run_olive_fwht.py',Path(inspect.getsourcefile(DataConfig)),Path(inspect.getsourcefile(get_calibration_dataset)))},
            activation_equivalence=activation_equivalence(),recipe_unchanged_after_preflight=True)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'report':str(a.out.resolve()),'status':report['status'],'error':report.get('error'),'temp':str(temp)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
