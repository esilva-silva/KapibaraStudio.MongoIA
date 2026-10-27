import importlib.util
from pathlib import Path
import unittest
scripts=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('delivery',scripts/'summarize_delivery.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class SummaryTests(unittest.TestCase):
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

if __name__=='__main__':unittest.main()
