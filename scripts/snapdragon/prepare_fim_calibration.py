"""Prepare a new, FIM-preserving Olive activation-calibration policy and preflight."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def prepare(recipe_path, calibration_path, tokenizer_path, out_dir):
    from tokenizers import Tokenizer
    recipe_path=Path(recipe_path).resolve(strict=True);calibration_path=Path(calibration_path).resolve(strict=True)
    tokenizer_path=Path(tokenizer_path).resolve(strict=True);out=Path(out_dir).resolve()
    if out.exists():raise ValueError('Output directory must be new')
    config=json.loads(recipe_path.read_text(encoding='utf-8-sig'))
    rows=[json.loads(line) for line in calibration_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(rows)!=256:raise ValueError('Exactly 256 calibration rows required')
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    items=[];ids_seen=set()
    for row in rows:
        if row.get('prompt') != row.get('prompt_text') or not isinstance(row.get('prompt'),str):
            raise ValueError('prompt and prompt_text must be identical strings')
        if row.get('id') in ids_seen or not row.get('id'):raise ValueError('Duplicate or missing row ID')
        ids_seen.add(row['id'])
        encoding=tokenizer.encode(row['prompt'],add_special_tokens=False)
        ids=encoding.ids; mask=encoding.attention_mask
        expected=row.get('prompt_ids')
        if not ids or ids!=expected or mask!=[1]*len(ids) or len(ids)>2048:
            raise ValueError('Original prompt IDs mismatch or exceed 2048: '+str(row.get('id')))
        items.append({'id':row['id'],'tokens':len(ids),'input_ids':ids,'attention_mask':mask})
    result=copy.deepcopy(config)
    candidates=[x for x in result.get('data_configs',[]) if x.get('name')=='wikitext2_train_act']
    if len(candidates)!=1:raise ValueError('Expected one activation data config')
    data=candidates[0];pre=data['pre_process_data_config']
    pre.update(strategy='line-by-line',add_special_tokens=False,pad_to_max_len=False,
               max_seq_len=2048,max_samples=256,text_cols=['text'])
    data['dataloader_config']={'batch_size':1}
    out.mkdir(parents=True)
    local=out/'calibration.jsonl'
    local.write_text(''.join(json.dumps({'text':r['prompt']},ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    data['load_dataset_config']={'data_name':'json','data_files':str(local),'split':'train'}
    result['output_dir']=str(out/'olive-output');result['cache_dir']=str(out/'olive-cache')
    recipe_out=out/'recipe-fim-activation-v1.json'
    recipe_out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    report={'schema':'kapi-fim-activation-preflight-v1','status':'preflight_only_not_olive_or_inference',
        'policy':{'strategy':'line-by-line','add_special_tokens':False,'pad_to_max_len':False,
                  'batch_size':1,'max_seq_len':2048,'max_samples':256},
        'recipe_source_sha256':sha(recipe_path),'dataset_source_sha256':sha(calibration_path),
        'tokenizer_sha256':sha(tokenizer_path),'derived_recipe_sha256':sha(recipe_out),
        'calibration_text_sha256':sha(local),'count':len(items),
        'min_tokens':min(x['tokens'] for x in items),'max_tokens':max(x['tokens'] for x in items),
        'items':items,
        'limitations':['Standalone tokenizer preflight; installed Olive/Transformers dataloader and CPU calibration must still be validated against these IDs and masks.',
                       'Variable-length calibration is untested with the model and ORT CPU in this environment.']}
    (out/'preflight.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recipe','calibration','tokenizer','out-dir'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    try:
        report=prepare(a.recipe,a.calibration,a.tokenizer,a.out_dir)
        print(json.dumps({k:report[k] for k in ('status','count','min_tokens','max_tokens')},indent=2))
    except Exception as exc:p.exit(1,f'Preflight failed: {type(exc).__name__}: {exc}\n')

if __name__=='__main__':main()
