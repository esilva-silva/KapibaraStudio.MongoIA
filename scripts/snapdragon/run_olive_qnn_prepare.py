"""Run the remaining QNN preparation with real plugin discovery and explicit CPU calibration."""
import argparse,json,time,traceback,importlib.metadata
from pathlib import Path

def run(config_path,report_path):
    started=time.monotonic();report={'status':'starting','qualified':False,'scope':'QNN-target graph preparation, CPU activation calibration; no HTP inference or compilation'}
    try:
        import torch,onnxruntime as ort
        if torch.cuda.is_available() or torch.cuda.device_count():raise RuntimeError('This phase requires GPUs hidden; no implicit GPU selection')
        config=json.loads(config_path.read_text(encoding='utf-8'))
        if config.get('target')!='qnn_system':raise ValueError('QNN target must be preserved')
        if config['systems']['qnn_system']['accelerators'][0]['execution_providers']!=['QNNExecutionProvider']:raise ValueError('Explicit QNN accelerator required')
        if config['passes']['sq']['calibration_providers']!=['CPUExecutionProvider']:raise ValueError('Explicit CPU calibration required')
        if 'QNNExecutionProvider' not in ort.get_available_providers():raise RuntimeError('QNN plugin must be genuinely registered in this environment')
        report['providers']=ort.get_available_providers();report['gpu_visible_count']=torch.cuda.device_count()
        report['versions']={name:importlib.metadata.version(name) for name in ('olive-ai','onnxruntime','onnxruntime-qnn','onnxruntime-genai','onnx','transformers','optimum')}
        torch.compiler.set_stance('force_eager')
        report['execution_mode']='CPU preparation; Torch eager'
        from olive_genai18_compat import install
        report['modelbuilder_compatibility']=install()
        report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        import functools
        from olive.passes import Pass
        original_run=Pass.run
        report['stages']=[]
        @functools.wraps(original_run)
        def timed_pass(current,*args,**kwargs):
            start=time.monotonic();event={'pass':type(current).__name__,'status':'running'};report['stages'].append(event)
            report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            try:
                result=original_run(current,*args,**kwargs);event.update(status='completed',seconds=time.monotonic()-start)
                report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');return result
            except Exception as error:
                event.update(status='failed',seconds=time.monotonic()-start,error=f'{type(error).__name__}: {error}')
                report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');raise
        Pass.run=timed_pass
        from olive.workflows import run as olive_run
        result=olive_run(config,tempdir=str(config_path.parent/'qnn-tmp'))
        report['workflow_result_nonempty']=bool(result)
        final_config=Path(config['output_dir'])/'model_config.json'
        report['final_model_config_exists']=final_config.is_file()
        if not result or not final_config.is_file():raise RuntimeError('Olive produced no final model_config; inspect pass failures in the log')
        report.update(status='workflow_returned_artifact_validation_pending',final_model_config=str(final_config))
    except Exception as error:report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    report['wall_seconds']=time.monotonic()-started;report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    report=run(args.config.resolve(),args.report.resolve());print(json.dumps(report,indent=2));raise SystemExit(1 if report['status']=='failed' else 0)
