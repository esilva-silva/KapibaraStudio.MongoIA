import importlib.util
from pathlib import Path
import unittest
s=importlib.util.spec_from_file_location('olive_genai18_compat',Path(__file__).resolve().parents[1]/'olive_genai18_compat.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class CompatTests(unittest.TestCase):
 def test_quantization_options_keep_numeric_and_operator_policy(self):
  actual=m.cli_options({'int4_block_size':32,'int4_accuracy_level':4,'int4_op_types_to_quantize':['Gather'],'int4_is_symmetric':True})
  self.assertEqual(actual,['block_size=32','accuracy_level=4','op_types_to_quantize=Gather','is_symmetric=true'])
 def test_operator_and_node_lists_have_native_separators(self):
  self.assertEqual(m.cli_options({'int4_op_types_to_quantize':['Gather','MatMul'],'int4_nodes_to_exclude':['a','b']}),['op_types_to_quantize=Gather/MatMul','nodes_to_exclude=a,b'])
 def test_nonquantization_values_are_preserved(self):
  self.assertEqual(m.cli_options({'filename':'model.onnx','exclude_lm_head':False,'unset':None}),['filename=model.onnx','exclude_lm_head=false'])
 def test_unknown_nested_value_is_not_silently_reinterpreted(self):
  with self.assertRaises(TypeError):m.cli_options({'unknown':{'a':1}})
