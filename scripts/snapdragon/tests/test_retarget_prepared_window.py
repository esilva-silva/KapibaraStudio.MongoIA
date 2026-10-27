import importlib.util
from pathlib import Path
import sys
import unittest
import onnx
from onnx import helper,TensorProto
scripts=Path(__file__).resolve().parents[1];sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('retarget',scripts/'retarget_prepared_window.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def graph(seq, axis=1):
    n=helper.make_node('Identity',['x'],['y'])
    g=helper.make_graph([n],'g',[helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,seq,64])],[helper.make_tensor_value_info('y',TensorProto.FLOAT,[1,seq,64])])
    return helper.make_model(g)

class RetargetTest(unittest.TestCase):
    def test_only_differing_sequence_metadata_changes(self):
        c=graph(64);i=graph(1); result,changes=mod.retarget_graph(c,i,32)
        self.assertEqual(len(changes),2)
        self.assertEqual([x.dim_value for x in result.graph.input[0].type.tensor_type.shape.dim],[1,32,64])
        self.assertEqual(result.graph.node[0].SerializeToString(),c.graph.node[0].SerializeToString())
    def test_changed_node_attribute_rejected(self):
        c=graph(64);i=graph(1);i.graph.node[0].name='changed'
        with self.assertRaises(ValueError):mod.retarget_graph(c,i,32)
    def test_baked_constant_difference_rejected(self):
        c=graph(64);i=graph(1)
        c.graph.initializer.append(helper.make_tensor('baked',TensorProto.INT64,[1],[64]))
        i.graph.initializer.append(helper.make_tensor('baked',TensorProto.INT64,[1],[1]))
        with self.assertRaises(ValueError):mod.retarget_graph(c,i,32)
if __name__=='__main__':unittest.main()
