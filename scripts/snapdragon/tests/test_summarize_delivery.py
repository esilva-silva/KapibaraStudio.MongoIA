import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
scripts=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('delivery',scripts/'summarize_delivery.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class SummaryTests(unittest.TestCase):
    def test_completion_requires_small_and_private_artifact_identity(self):
        delivery = {}
        for name in ('nano', 'small'):
            delivery[name + '_gpu'] = {'status': 'validated_200', 'boundary_2016_plus_32': True}
            delivery[name + '_npu'] = {'status': 'validated_200_boundary_passed', 'package': {'release_identity_verified': True}}
        self.assertTrue(mod.delivery_complete(delivery))
        delivery['small_npu'] = {'status': 'pending_remote_preparation'}
        self.assertFalse(mod.delivery_complete(delivery))
        delivery['small_npu'] = {'status': 'validated_200_boundary_passed', 'package': {'release_identity_verified': False}}
        self.assertFalse(mod.delivery_complete(delivery))

    def test_missing_small_pending_and_invalid_presence_not_validated(self):
        with tempfile.TemporaryDirectory(dir=scripts) as directory:
            root = Path(directory)
            report_dir = root / 'reports/snapdragon'; report_dir.mkdir(parents=True)
            result, _ = mod.summarize_npu32(root, {}, 'small', 'a' * 64, [], None)
            self.assertEqual(result['status'], 'pending_remote_preparation')
            (report_dir / 'small-npu-window32-vs-bf16.json').write_text('{invalid')
            result, _ = mod.summarize_npu32(root, {}, 'small', 'a' * 64, [], None)
            self.assertEqual(result['status'], 'not_validated')

    def test_comparison_hash_mismatch_rejected(self):
        def item(h):return {'qualified':False,'proxy':True,'corpus_sha256':h*64,'baseline':{'n':200},'candidate':{'n':200}}
        with self.assertRaises(ValueError):mod.validate_comparisons({'a':item('a'),'b':item('b')})
    def test_row_failure_prevents_success(self):
        report={'status':'completed','corpus_sha256':'a','provider_evidence':{'requested':'DmlExecutionProvider','verified':True},
                'items':[{'id':str(i), 'generated_tokens':32} for i in range(200)]}
        self.assertTrue(mod.successful_run(report,'a','DmlExecutionProvider'))
        report['items'][175]['error']='device removed'
        self.assertFalse(mod.successful_run(report,'a','DmlExecutionProvider'))
    def test_reordered_ids_rejected(self):
        ids=[str(i) for i in range(200)]
        report={'status':'completed','corpus_sha256':'a','provider_evidence':{'requested':'DmlExecutionProvider','verified':True},
                'items':[{'id':x,'generated_tokens':1} for x in reversed(ids)]}
        self.assertFalse(mod.successful_run(report,'a','DmlExecutionProvider',ids))

    def test_window32_rejects_stale_stage_or_unverified_boundary(self):
        ids=[str(i) for i in range(200)]; corpus='a'*64
        comparison={'qualified':False,'proxy':True,'corpus_sha256':corpus,'baseline':{'n':200},'candidate':{'n':200}}
        run={'status':'completed','corpus_sha256':corpus,'context_tokens':2048,'max_new_tokens':32,'greedy':True,
             'provider_evidence':{'requested':'QNNExecutionProvider','verified':True},
             'items':[{'id':i,'generated_tokens':32} for i in ids]}
        stage={'status':'verified','report_sha256':'fresh'}
        boundary={'status':'completed','context_tokens':2048,'max_new_tokens':32,'greedy':True,
                  'provider_evidence':{'requested':'QNNExecutionProvider','verified':True},
                  'items':[{'prompt_tokens':2016,'generated_tokens':32}]}
        self.assertTrue(mod.assess_window32(comparison,run,stage,boundary,corpus,ids,'fresh',None))
        self.assertFalse(mod.assess_window32(comparison,run,stage,boundary,corpus,ids,'stale',None))
        boundary['provider_evidence']['verified']=False
        self.assertFalse(mod.assess_window32(comparison,run,stage,boundary,corpus,ids,'fresh',None))

    def test_private_release_missing_identity_rejected(self):
        self.assertFalse(mod.private_release_ok({'npu_exports': {'nano': {}}}, {}, {}, {}, {}))

    def test_activation_helper_status_is_not_completion(self):
        trace={'status':'completed','activation_type':'QUInt16','calibration_providers':['CPUExecutionProvider']}
        self.assertFalse(mod.activation_trace_ok(trace,256,64))
        trace.update(calibration_status='input_complete',calibration_inputs_seen=254)
        self.assertFalse(mod.activation_trace_ok(trace,256,64))
        trace['calibration_inputs_seen']=256
        self.assertTrue(mod.activation_trace_ok(trace,256,64))
        trace['first_input_shapes']={'past_key_values.0.key':{'shape':[1,2,0,128]}}
        self.assertFalse(mod.activation_trace_ok(trace,256,64))

if __name__=='__main__':unittest.main()
