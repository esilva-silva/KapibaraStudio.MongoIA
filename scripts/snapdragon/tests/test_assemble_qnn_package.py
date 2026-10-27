import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
scripts=Path(__file__).resolve().parents[1];sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('assemble',scripts/'assemble_qnn_package.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class AssemblyTests(unittest.TestCase):
    def test_source_identity_requires_exact_hashes(self):
        expected={'repo_id':'owner/repo','revision':'a'*40,'files':[{'path':'tokenizer.json','size':4,'sha256':'b'*64}]}
        module.validate_source_identity(expected,expected)
        altered={'repo_id':'owner/repo','revision':'a'*40,'files':[{'path':'tokenizer.json','size':4,'sha256':'c'*64}]}
        with self.assertRaises(ValueError): module.validate_source_identity(expected,altered)
        with self.assertRaises(ValueError): module.validate_source_identity({},expected)

    def test_resource_must_be_inside_root(self):
        with tempfile.TemporaryDirectory(dir=scripts) as tmp:
            root=Path(tmp);(root/'ok').write_text('ok')
            self.assertEqual(module.safe_file(root.resolve(),'ok'),(root/'ok').resolve())
            with self.assertRaises(ValueError):module.safe_file(root.resolve(),'../escape')
    def test_missing_compile_report_rejected_before_output(self):
        with tempfile.TemporaryDirectory(dir=scripts) as tmp:
            root=Path(tmp)
            for name in ('prepared','compiled','source'): (root/name).mkdir()
            with self.assertRaises(Exception): module.assemble(root/'prepared',root/'compiled',root/'source',root/'out')
            self.assertFalse((root/'out').exists())
if __name__=='__main__':unittest.main()
