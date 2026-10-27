import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
scripts=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('delivery',scripts/'summarize_delivery.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class SummaryTests(unittest.TestCase):
    def test_explicit_evidence_set_never_falls_back_to_legacy(self):
        with tempfile.TemporaryDirectory(dir=scripts) as directory:
            root = Path(directory); reports = root / 'reports/snapdragon'; reports.mkdir(parents=True)
            keys = ('comparison', 'run', 'stage', 'boundary', 'compile', 'transfer', 'source', 'smoke')
            mapping = {key: 'reports/snapdragon/new-' + key + '.json' for key in keys}
            config = {'npu_exports': {'nano': {'evidence_files': mapping}}}
            # Invalid legacy file must not be selected for the explicit new revision.
            (reports / 'nano-npu-window32-vs-bf16.json').write_text('{legacy-invalid')
            result, _ = mod.summarize_npu32(root, config, 'nano', 'a' * 64, [], None)
            self.assertEqual(result['status'], 'pending_evidence')
            self.assertEqual(len(result['missing']), 8)
            (reports / 'new-run.json').write_text('{invalid-new')
            result, _ = mod.summarize_npu32(root, config, 'nano', 'a' * 64, [], None)
            self.assertEqual(result['status'], 'not_validated')
            mapping['run'] = 'reports/snapdragon/../../escape.json'
            result, _ = mod.summarize_npu32(root, config, 'nano', 'a' * 64, [], None)
            self.assertEqual(result['status'], 'not_validated')
            del mapping['run']
            result, _ = mod.summarize_npu32(root, config, 'nano', 'a' * 64, [], None)
            self.assertEqual(result['status'], 'not_validated')

    def test_evaluated_path_is_confined_to_models_root(self):
        with tempfile.TemporaryDirectory(dir=scripts) as directory:
            root = Path(directory).resolve()
            config = {'models_root': str(root), 'npu_exports': {'small': {'evaluated_path': 'corrected-v3'}}}
            self.assertEqual(mod.evaluated_model_path(config, 'small'), root / 'corrected-v3')
            config['npu_exports']['small']['evaluated_path'] = '../escape'
            with self.assertRaises(ValueError): mod.evaluated_model_path(config, 'small')

    def test_required_policy_preserves_history_and_cannot_pass_by_label(self):
        with tempfile.TemporaryDirectory(dir=scripts) as directory:
            root = Path(directory)
            delivery = {name + '_npu': {'status': 'validated_200_boundary_passed', 'candidate': {'n': 200}}
                        for name in ('nano', 'small')}
            delivery['nano_gpu'] = {'status': 'validated_200'}
            config = {'required_npu_calibration_policy': 'fim-exact-tail-v1',
                      'npu_exports': {'nano': {'calibration_policy': 'legacy'}, 'small': {'calibration_policy': 'fim-exact-tail-v1'}}}
            mod.apply_calibration_policy_gate(delivery, config, root)
            self.assertEqual(delivery['nano_npu']['status'], 'calibration_revision_pending')
            self.assertEqual(delivery['small_npu']['status'], 'calibration_revision_pending')
            self.assertEqual(delivery['nano_npu']['historical_status'], 'validated_200_boundary_passed')
            self.assertEqual(delivery['nano_npu']['candidate']['n'], 200)
            self.assertEqual(delivery['nano_gpu']['status'], 'validated_200')
            self.assertFalse(mod.delivery_complete(delivery))

    def test_exact_policy_requires_activation_and_effective_gptq_completion(self):
        with tempfile.TemporaryDirectory(dir=scripts) as directory:
            root = Path(directory)
            trace = {'calibration_status': 'input_complete', 'fim_preflight': {'all_match': True, 'fim_complete': True,
                'matched': 256, 'policy': {'add_special_tokens': False, 'pad_to_max_len': False, 'batch_size': 1}}}
            receipt = {'policy': 'kapi_gptq_join_tail_v1', 'all_prompt_tokens_covered': True, 'selected_rows': 256,
                'joined_tokens': 61273, 'consumed_tokens': 61273, 'add_special_tokens': False, 'padding': 'none',
                'batch_size': 1, 'shuffle': False, 'batches': [{'tokens': n} for n in [2048] * 29 + [1881]],
                'stream_token_ids_sha256': '07f7dec2ffed64e72f7d015bfd86bbce5d16f0297538c6fffd079fe9183da362'}
            job = {'status': 'completed', 'effective_gptq_calibration': {'batches': 30, 'min_tokens': 1881, 'max_tokens': 2048}}
            observed = {'matched': True, 'batch_count': 30, 'token_count': 61273,
                        'stream_token_ids_sha256': receipt['stream_token_ids_sha256'], 'batches': receipt['batches']}
            job['runtime_calibration_audit'] = {'quantize_entry': observed, 'after_prepare_dataset': observed}
            spec = {'calibration_policy': 'fim-exact-tail-v1'}
            for key, report in [('activation_calibration_trace', trace), ('gptq_runtime_receipt', receipt), ('gptq_completion_report', job)]:
                spec[key] = key + '.json'; (root / spec[key]).write_text(json.dumps(report))
            self.assertTrue(mod.calibration_policy_evidence(spec, 'fim-exact-tail-v1', root)[0])
            job.update(status='workflow_returned_artifact_validation_pending', workflow_result_nonempty=True,
                       stages=[{'pass': 'GptqModel', 'status': 'completed'}])
            (root / spec['gptq_completion_report']).write_text(json.dumps(job))
            self.assertTrue(mod.calibration_policy_evidence(spec, 'fim-exact-tail-v1', root)[0])
            job['status'] = 'running'
            (root / spec['gptq_completion_report']).write_text(json.dumps(job))
            self.assertFalse(mod.calibration_policy_evidence(spec, 'fim-exact-tail-v1', root)[0])
            job['status'] = 'workflow_returned_artifact_validation_pending'
            del job['runtime_calibration_audit']['after_prepare_dataset']
            (root / spec['gptq_completion_report']).write_text(json.dumps(job))
            self.assertFalse(mod.calibration_policy_evidence(spec, 'fim-exact-tail-v1', root)[0])
            job['runtime_calibration_audit']['after_prepare_dataset'] = observed
            job['effective_gptq_calibration']['batches'] = 29
            (root / spec['gptq_completion_report']).write_text(json.dumps(job))
            self.assertFalse(mod.calibration_policy_evidence(spec, 'fim-exact-tail-v1', root)[0])
            trace['fim_preflight']['policy']['add_special_tokens'] = True
            (root / spec['activation_calibration_trace']).write_text(json.dumps(trace))
            self.assertFalse(mod.calibration_policy_evidence(spec, 'fim-exact-tail-v1', root)[0])

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
