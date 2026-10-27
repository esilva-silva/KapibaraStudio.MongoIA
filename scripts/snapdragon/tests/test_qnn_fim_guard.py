import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
scripts=Path(__file__).resolve().parents[1]
tree=ast.parse((scripts/'qnn_cpu_sitecustomize.py').read_text(encoding='utf-8'))
selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('_fim_digest','FimGuard','GuardedBatches','validate_fim_stride')]
namespace={'hashlib':hashlib,'json':json,'Path':Path}
exec(compile(ast.Module(body=selected,type_ignores=[]),'guard','exec'),namespace)
Guard=namespace['FimGuard']
class Fake:
    def __init__(self,rows):self.rows=rows
    def tolist(self):return self.rows
class GuardTests(unittest.TestCase):
    def fixture(self,root):
        path=root/'pre.json';path.write_text(json.dumps({'schema':'kapi-fim-activation-preflight-v1','count':256,
          'policy':{'add_special_tokens':False,'pad_to_max_len':False,'batch_size':1},
          'dataset_source_sha256':'a','tokenizer_sha256':'b',
          'items':[{'id':str(i),'input_ids':[i,4],'attention_mask':[1,1]} for i in range(256)]}))
        return path
    def test_strided_calibration_rejected(self):
        self.assertIsNone(namespace['validate_fim_stride'](None))
        self.assertEqual(namespace['validate_fim_stride'](0),0)
        for value in (1,128,-1,False,'0'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                namespace['validate_fim_stride'](value)

    def test_exact_ids_masks_and_completion(self):
        with tempfile.TemporaryDirectory(dir=scripts) as d:
            record={};guard=Guard(self.fixture(Path(d)),record,lambda:None)
            for i in range(256):guard.accept({'input_ids':Fake([[i,4]]),'attention_mask':Fake([[1,1]])})
            guard.finish();self.assertTrue(record['fim_preflight']['all_match']);self.assertEqual(record['fim_preflight']['matched'],256)
    def test_mismatch_fails_before_yield(self):
        with tempfile.TemporaryDirectory(dir=scripts) as d:
            record={};guard=Guard(self.fixture(Path(d)),record,lambda:None)
            with self.assertRaises(ValueError):guard.accept({'input_ids':Fake([[0,4]]),'attention_mask':Fake([[1,0]])})
            self.assertEqual(record['calibration_status'],'failed')
    def test_partial_rejected(self):
        with tempfile.TemporaryDirectory(dir=scripts) as d:
            record={};guard=Guard(self.fixture(Path(d)),record,lambda:None)
            with self.assertRaises(ValueError):guard.finish()
            self.assertEqual(record['calibration_status'],'failed')
if __name__=='__main__':unittest.main()
