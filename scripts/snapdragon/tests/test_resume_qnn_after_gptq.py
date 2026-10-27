import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from resume_qnn_after_gptq import prepare


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.job=self.root/'gptq'; self.source=self.job/'model'; self.source.mkdir(parents=True)
        (self.source/'config.json').write_text(json.dumps({'max_position_embeddings':32768,'seq_length':32768,'name':'fake'}))
        (self.source/'weights.safetensors').write_bytes(b'synthetic weights')
        self.model=self.job/'model_config.json'; self.model.write_text(json.dumps({'type':'hfmodel','config':{'model_path':str(self.source),'load_kwargs':None}}))
        self.status=self.root/'status.json'; self.status.write_text(json.dumps({'status':'workflow_returned_artifact_validation_pending','stages':[{'pass':'GptqModel','status':'completed'}]}))
        passes={x:{'type':x} for x in ('cs','g','mb','mq','gs','kv_shape','sp','st')}; passes['sq']={'calibration_providers':['CPUExecutionProvider']}
        self.recipe=self.root/'prepare.json'; self.recipe.write_text(json.dumps({'target':'qnn_system','systems':{'qnn_system':{'accelerators':[{'execution_providers':['QNNExecutionProvider']}]}},'passes':passes}))
        self.args=(self.recipe,self.model,self.status,self.root/'derived',self.root/'output',self.root/'cache',self.root/'resume.json',self.root/'report.json')
    def tearDown(self): self.tmp.cleanup()

    def test_resume_copies_and_only_changes_context(self):
        before={p.name:p.read_bytes() for p in self.source.iterdir()}
        report=prepare(*self.args); config=json.loads((self.root/'derived/config.json').read_text()); recipe=json.loads((self.root/'resume.json').read_text())
        self.assertEqual(config,{'max_position_embeddings':2048,'seq_length':2048,'name':'fake'})
        self.assertEqual((self.root/'derived/weights.safetensors').read_bytes(),before['weights.safetensors'])
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.source.iterdir()})
        self.assertTrue(report['source_unchanged'] and report['non_config_files_identical'])
        self.assertEqual(list(recipe['passes']),['mb','mq','gs','kv_shape','sq','sp','st'])
        self.assertEqual(recipe['passes']['st']['context_length'],32)
        self.assertEqual(recipe['input_model']['config']['load_kwargs'],{})
        self.assertEqual(recipe['input_model']['config']['model_path'],str(self.root/'derived'))
        self.assertEqual(recipe['target'],'qnn_system')

    def test_refuses_incomplete_job_and_existing_or_overlapping_target(self):
        self.status.write_text(json.dumps({'status':'running'}))
        with self.assertRaises(ValueError): prepare(*self.args)
        self.assertFalse((self.root/'derived').exists())
        self.status.write_text(json.dumps({'status':'workflow_returned_artifact_validation_pending','stages':[{'pass':'GptqModel','status':'completed'}]}))
        with self.assertRaises(ValueError): prepare(*self.args[:3],self.source/'derived',*self.args[4:])
        (self.root/'derived').mkdir()
        with self.assertRaises(ValueError): prepare(*self.args)


if __name__=='__main__': unittest.main()
