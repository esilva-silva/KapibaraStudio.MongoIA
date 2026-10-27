"""Derive a shorter static prefill window only when paired ONNX graphs prove equivalence."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from verify_manifest import verify


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def retarget_graph(context, iterator, window):
    if type(window) is not int or window <= 1 or window >= 64:
        raise ValueError('Window must be between 2 and 63')
    normalized=copy.deepcopy(context); changes=[]
    for group in ('input','output','value_info'):
        left=getattr(normalized.graph,group);right=getattr(iterator.graph,group)
        if len(left)!=len(right):raise ValueError('Graph metadata count mismatch')
        for index,(x,y) in enumerate(zip(left,right)):
            if x.name!=y.name:raise ValueError('Graph metadata name mismatch')
            xd=x.type.tensor_type.shape.dim;yd=y.type.tensor_type.shape.dim
            if len(xd)!=len(yd):raise ValueError('Graph rank mismatch')
            for axis,(a,b) in enumerate(zip(xd,yd)):
                if a.SerializeToString()!=b.SerializeToString():
                    if a.WhichOneof('value')!='dim_value' or b.WhichOneof('value')!='dim_value' or a.dim_value!=64 or b.dim_value!=1:
                        raise ValueError('Unexpected graph dimension difference')
                    a.dim_value=1;changes.append((group,index,axis))
    if not changes or normalized.SerializeToString()!=iterator.SerializeToString():
        raise ValueError('Context and iterator differ beyond sequence dimension metadata')
    result=copy.deepcopy(context)
    for group,index,axis in changes:
        getattr(result.graph,group)[index].type.tensor_type.shape.dim[axis].dim_value=window
    return result,changes


def derive(prepared, out_dir, window):
    import onnx
    prepared=Path(prepared).resolve(strict=True);output=Path(out_dir).resolve()
    if output.exists() or output.is_relative_to(prepared) or prepared.is_relative_to(output):
        raise ValueError('Output must be new and separate')
    check=verify(prepared,'export_manifest.json')
    if not check['passed']:raise ValueError('Parent manifest verification failed')
    parent_path=prepared/'export_manifest.json';parent=json.loads(parent_path.read_text(encoding='utf-8'))
    paths={e['path'] for e in parent['files']}
    actual={p.relative_to(prepared).as_posix() for p in prepared.rglob('*') if p.is_file()}
    if actual != paths | {'export_manifest.json'}:raise ValueError('Parent manifest must cover all files')
    if any(p.is_symlink() for p in prepared.rglob('*')):raise ValueError('Parent symlinks are unsupported')
    config=json.loads((prepared/'genai_config.json').read_text(encoding='utf-8'))
    decoder=config['model']['decoder'];sliding=decoder['sliding_window']
    if sliding.get('window_size')!=64:raise ValueError('Expected original prefill window 64')
    group=decoder['pipeline'][0];context_name=group['context']['filename'];iterator_name=group['iterator']['filename']
    for name in (context_name,iterator_name):
        p=(prepared/name).resolve(strict=True)
        if not p.is_relative_to(prepared) or p.relative_to(prepared).as_posix() not in paths:raise ValueError('Invalid graph path')
    context=onnx.load(str(prepared/context_name),load_external_data=False)
    iterator=onnx.load(str(prepared/iterator_name),load_external_data=False)
    result,changes=retarget_graph(context,iterator,window)
    sliding['window_size']=window
    output.mkdir(parents=True)
    for src in prepared.rglob('*'):
        rel=src.relative_to(prepared);dest=output/rel
        if src.is_dir():dest.mkdir(exist_ok=True)
        elif rel.as_posix() not in (context_name,'genai_config.json','export_manifest.json'):
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
            if sha(src)!=sha(dest):raise ValueError('Copy hash mismatch: '+str(rel))
    onnx.save_model(result,str(output/context_name))
    (output/'genai_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    note=output/'WINDOW_DERIVATION.md'
    note.write_text(f'Experimental local graph metadata derivation from prefill window 64 to {window}. Nodes and initializers are unchanged. Copied benchmark and audit files are parent evidence and do not validate this window. No inference validation has been performed.\n',encoding='utf-8')
    files=[{'path':p.relative_to(output).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(output.rglob('*')) if p.is_file() and p.name!='export_manifest.json']
    manifest={'schema':'kapi-qnn-window-derivation-v1','status':'not_inference_validated','window':window,
        'parent_manifest_sha256':sha(parent_path),'parent_source_provenance':parent.get('source') or parent.get('parent_source_provenance'),
        'parent_evidence_note':'Copied audit and benchmark artifacts refer to parent window 64 only',
        'metadata_dimensions_changed':len(changes),'files':files}
    (output/'export_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--window',required=True,type=int)
    a=p.parse_args()
    try:print(json.dumps(derive(a.prepared,a.out_dir,a.window),indent=2))
    except Exception as exc:p.exit(1,f'Derivation failed: {type(exc).__name__}: {exc}\n')

if __name__=='__main__':main()
