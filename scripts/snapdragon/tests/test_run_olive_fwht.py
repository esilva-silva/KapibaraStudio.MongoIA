import hashlib
import importlib.util
from pathlib import Path
import struct
import unittest
scripts=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fwht_runner',scripts/'run_olive_fwht.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Fake:
    def __init__(self,data):self.data=data
    def tolist(self):return self.data

def fixture():
    rows=[];blocks=[];stream=hashlib.sha256()
    for index,length in enumerate([2048]*29+[1881]):
        ids=[index]*length;enc=b''.join(struct.pack('<q',x) for x in ids);stream.update(enc)
        rows.append({'input_ids':Fake([ids]),'attention_mask':Fake([[1]*length])})
        blocks.append({'index':index,'tokens':length,'token_ids_sha256':hashlib.sha256(enc).hexdigest()})
    return rows,{'policy':'kapi_gptq_join_tail_v1','selected_rows':256,'all_prompt_tokens_covered':True,
      'batch_size':1,'padding':'none','add_special_tokens':False,'shuffle':False,'batches':blocks,
      'consumed_tokens':61273,'stream_token_ids_sha256':stream.hexdigest()}
class AuditTests(unittest.TestCase):
    def test_exact_batches_and_tail(self):
        rows,ledger=fixture();result=m.validate_calibration_batches(rows,ledger)
        self.assertTrue(result['matched']);self.assertEqual(result['token_count'],61273)
    def test_hash_mask_and_tail_mismatch_fail(self):
        rows,ledger=fixture();rows[0]['input_ids'].data[0][0]=99
        with self.assertRaises(ValueError):m.validate_calibration_batches(rows,ledger)
        rows,ledger=fixture();rows[1]['attention_mask'].data[0][0]=0
        with self.assertRaises(ValueError):m.validate_calibration_batches(rows,ledger)
        rows,ledger=fixture();rows[-1]['input_ids'].data[0].pop()
        with self.assertRaises(ValueError):m.validate_calibration_batches(rows,ledger)
if __name__=='__main__':unittest.main()
