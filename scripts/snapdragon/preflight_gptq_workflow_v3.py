"""Real workflow-resolved GPTQ data preflight; metadata descriptor only, no weights/GPU."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.metadata,inspect,json,logging,os,sys,tempfile,traceback,types
from datetime import datetime,timezone
from pathlib import Path
from preflight_fim_exact_tail_v1 import configure_temp,sha,require,json_file,ROOT,BASE,EXPECTED_STREAM
EXPECTED_HELPER='a4573a9b3b43723d877be8d2daffb31eee5e2819810e2b753ee98ee9a8c98948'
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def equivalent_activation(recipe):
    old=BASE/'activation'
    new_config=next(c for c in recipe['data_configs'] if c['name']=='wikitext2_train_act')
    new=Path(new_config['load_dataset_config']['data_files']).parent
    old_pre=json_file(old/'preflight.json');new_pre=json_file(new/'preflight.json')
    for key in ('items','policy','count','dataset_source_sha256','tokenizer_sha256','calibration_text_sha256'):
        require(old_pre[key]==new_pre[key],f'Activation equivalence failed: {key}')
    old_recipe=json_file(old/'recipe-fim-activation-v1.json')
    old_config=next(c for c in old_recipe['data_configs'] if c['name']==old_recipe['passes']['sq']['data_config'])
    new_recipe=json_file(new/'recipe-fim-activation-v1.json')
    active=next(c for c in new_recipe['data_configs'] if c['name']==new_recipe['passes']['sq']['data_config'])
    require(old_config['pre_process_data_config']==active['pre_process_data_config']==new_config['pre_process_data_config'],'Activation preprocessing changed')
    require(old_config['dataloader_config']==active['dataloader_config']==new_config['dataloader_config'],'Activation DataLoader changed')
    old_data=Path(old_config['load_dataset_config']['data_files']);new_data=Path(active['load_dataset_config']['data_files'])
    require(sha(old_data)==sha(new_data)==new_pre['calibration_text_sha256'],'Activation JSONL bytes differ')
    prior_path=ROOT/'reports/snapdragon/nano-fim-exact-tail-activation-reader-real-preflight-rx7800xt-v3.json'
    prior=json_file(prior_path)
    require(prior['status']=='passed' and prior['inference_count']==2,'Prior real activation reader proof missing')
    require(prior['guard_validation']['record']['fim_preflight']['matched']==256 and prior['guard_validation']['record']['fim_preflight']['all_match'] is True,'Prior guard proof missing')
    return {'equivalent':True,'new_activation_path':str(new),'preflight_sha256':sha(new/'preflight.json'),
        'items_canonical_sha256':canonical_sha(new_pre['items']),'count':new_pre['count'],'policy':new_pre['policy'],
        'same_items_ids_masks_policy_and_input_bytes':True,'calibration_jsonl_sha256':sha(new_data),
        'new_recipe_sha256':sha(new/'recipe-fim-activation-v1.json'),'prior_real_reader_and_two_inference_report':str(prior_path),
        'prior_report_sha256':sha(prior_path),'new_inference_count':0,'scope':'Data/reader-contract equivalence only; no new model or quality qualification'}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,default=BASE/'gptq-only-v3.json')
    p.add_argument('--out',type=Path,default=ROOT/'reports/snapdragon/nano-fim-exact-tail-gptq-real-preflight-rx7800xt-v3.json')
    a=p.parse_args()
    require(a.out.resolve().is_relative_to((ROOT/'reports').resolve()),'Report must be under worktree reports')
    require(not a.out.exists(),'Refusing to overwrite prior evidence')
    raw_script=Path(__file__).read_bytes();require(not raw_script.startswith(b'\xef\xbb\xbf') and b'\r\n' not in raw_script,'Execute UTF8 LF script')
    temp=Path(tempfile.mkdtemp(prefix='kapi-gptq-workflow-preflight-v3-'));configure_temp(temp)
    os.environ['HF_DATASETS_IN_MEMORY_MAX_SIZE']='67108864'
    report={'status':'starting','created_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),
        'recipe':str(a.recipe),'recipe_sha256':sha(a.recipe),'isolated_temp':str(temp),'model_weights_loaded':False,
        'GPU_execution':False,'quantization_run':False,'inference_count':0,
        'scope':'Installed RunConfig shared-parameter injection, actual metadata-only Hf descriptor, DataConfig/create_dataloader/get_calibration_dataset and installed GPTQ prepare/collate; no pass execution'}
    try:
        import torch,olive
        from olive.workflows.run.config import RunConfig
        from olive.model import HfModelHandler
        from olive.passes.pytorch.train_utils import get_calibration_dataset
        from audit_gptq_calibration_tokens import installed_function
        from run_olive_fwht import validate_calibration_batches
        require(importlib.metadata.version('transformers')=='4.57.6','Use probe Transformers4.57.6')
        recipe=json_file(a.recipe);require(set(recipe['passes'])=={'cs','g'},'Expected GPTQ-only recipe')
        custom=next(c for c in recipe['data_configs'] if c['name']==recipe['passes']['g']['data_config'])
        helper=Path(custom['user_script']);require(sha(helper)==EXPECTED_HELPER,'Helper bytes differ from regenerated v3 pin')
        helper_bytes=helper.read_bytes();require(not helper_bytes.startswith(b'\xef\xbb\xbf') and b'\r\n' not in helper_bytes,'Helper must be UTF8 LF')
        reserved=Path(custom['pre_process_data_config']['audit_path']);require(not reserved.exists(),'Reserved job ledger already exists')
        ledger_path=reserved.with_name('preflight-token-ledger.json');ledger=json_file(ledger_path)
        require(ledger.get('helper_sha256')==EXPECTED_HELPER,'Generated ledger pins stale helper')
        projection_path=Path(custom['load_dataset_config']['data_files']);projection=rows(projection_path)
        original_path=ROOT/'data/calib/snapdragon-20261026/calibration.jsonl';originals=rows(original_path)
        require(sha(original_path)=='179736eed969a8a961af6d3fc410a3eead24306234d65ccf245c20cb16978b60','Frozen source SHA differs')
        fields=('id','prompt','prompt_text','prompt_ids')
        require(len(projection)==len(originals)==256 and all(set(r)==set(fields) and all(r[k]==o[k] for k in fields) for r,o in zip(projection,originals)),'Projection changed actual calibration inputs/order')
        model_load_calls=[]
        def forbidden_load(*args,**kwargs):
            model_load_calls.append(True)
            raise RuntimeError('Weight/model loading is forbidden in this CPU data preflight')
        HfModelHandler.load_model=forbidden_load
        checks=[];descriptor_info=None
        for label in ('create_dataloader','get_calibration_dataset'):
            cloned=copy.deepcopy(recipe);fresh=temp/(label+'-preflight-only-ledger.json')
            selected=next(c for c in cloned['data_configs'] if c['name']==recipe['passes']['g']['data_config'])
            selected['pre_process_data_config']['audit_path']=str(fresh)
            # Full installed workflow parser is the exact source of model_name/task injection.
            workflow=RunConfig.parse_file_or_obj(cloned)
            descriptor=workflow.input_model.create_model()
            require(isinstance(descriptor,HfModelHandler) and descriptor.model is None,'Expected metadata-only Hf descriptor')
            data_config=next(c for c in workflow.data_configs if c.name==custom['name'])
            params=data_config.pre_process_params
            require('task' in params and params['task']==descriptor.task,'Workflow shared task missing/different from actual descriptor')
            require(Path(params['model_name'])==Path(descriptor.model_name_or_path),'Workflow model_name differs from descriptor')
            extra_names=sorted(set(params)-set(custom['pre_process_data_config'])-{'type'})
            declared_names=set(custom['pre_process_data_config'])-{'type'}
            injected_names=sorted(set(params)-declared_names)
            require('task' in injected_names,'The originally failing task injection was not exercised')
            if label=='create_dataloader':
                batches=[batch[0] for batch in data_config.to_data_container().create_dataloader()]
            else:
                batches=get_calibration_dataset(descriptor,data_config)
            observed=validate_calibration_batches(batches,ledger)
            actual=json_file(fresh)
            require(actual['batches']==ledger['batches'],'Actual and generated ledgers differ')
            require(actual.get('forwarded_parameter_names')==injected_names or sorted(actual.get('forwarded_parameter_names',[]))==injected_names,'Runtime helper extra-parameter names do not match workflow injection')
            require(descriptor.model is None and not model_load_calls,'A model was loaded during data preparation')
            checks.append({'entrypoint':label,'matched':observed,'resolved_preprocess_parameter_names':sorted(params),
                'workflow_injected_extra_parameter_names':injected_names,'runtime_helper_extra_parameter_names':actual.get('forwarded_parameter_names'),
                'extra_parameter_values_recorded':False,'resolved_dataloader_type':data_config.dataloader_config.type,
                'resolved_dataloader_parameter_names':sorted(data_config.dataloader_params),'actual_runtime_ledger':str(fresh),'actual_runtime_ledger_sha256':sha(fresh)})
            descriptor_info={'class':type(descriptor).__name__,'created_by':'Installed ModelConfig.create_model',
                'task_equals_workflow_injected_task':True,'model_name_equals_workflow_model_name':True,'model_is_none':True,'load_model_calls':len(model_load_calls)}
        before=checks[-1]['matched'];require(checks[0]['matched']==before,'Real create_dataloader/get_calibration_dataset differ')
        site=Path(olive.__file__).resolve().parent.parent
        ns={'torch':torch,'LongTensor':torch.LongTensor,'Tensor':torch.Tensor,'log':logging.getLogger('gptq_preflight')}
        _,collate=installed_function(site/'gptqmodel/utils/data.py','collate_data',ns)
        prepare,prepare_source=installed_function(site/'gptqmodel/models/base.py','prepare_dataset',ns,'BaseGPTQModel')
        prepared=prepare(types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=151643),support_batch_quantize=True),batches)
        after=validate_calibration_batches(prepared,ledger)
        require(before==after,'Installed prepare/collate changed tokens/masks/order')
        require(before['batch_count']==30 and before['token_count']==61273 and before['stream_token_ids_sha256']==EXPECTED_STREAM,'Wrong stream')
        require([b['tokens'] for b in before['batches']]==[2048]*29+[1881],'Wrong full/tail lengths')
        require(not reserved.exists(),'Reserved future-job ledger was consumed')
        require(sha(a.recipe)==report['recipe_sha256'] and sha(helper)==EXPECTED_HELPER,'Recipe/helper changed during preflight')
        report.update(status='passed',versions={n:importlib.metadata.version(n) for n in ('transformers','olive-ai','gptqmodel','datasets','torch','tokenizers')},
            real_workflow_entrypoint_checks=checks,model_descriptor=descriptor_info,
            validation_before_gptq_prepare=before,validation_after_gptq_prepare=after,
            installed_gptq_data_functions={'prepare_dataset':prepare_source,'collate_data':collate},
            only_manual_recipe_overrides='Each separate real entrypoint invocation clones only pre_process_data_config.audit_path to its own fresh Temp file; HF caches isolated by environment',
            reserved_future_job_ledger={'path':str(reserved),'existed_before':False,'exists_after':False,'untouched':True},
            projection={'path':str(projection_path),'sha256':sha(projection_path),'same_original_fields_and_order':True,'original_sha256':sha(original_path)},
            source_files={str(f):sha(f) for f in (helper,ledger_path,ROOT/'scripts/snapdragon/run_olive_fwht.py',Path(inspect.getsourcefile(RunConfig)),Path(inspect.getsourcefile(get_calibration_dataset)),Path(inspect.getsourcefile(HfModelHandler)))},
            activation_equivalence=equivalent_activation(recipe),recipe_and_helper_unchanged=True)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise
    finally:
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps({'report':str(a.out.resolve()),'status':report['status'],'error':report.get('error'),'temp':str(temp)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
