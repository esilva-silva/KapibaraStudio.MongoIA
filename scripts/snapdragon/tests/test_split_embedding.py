import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import onnx
from onnx import TensorProto, helper

scripts = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('split_embedding', scripts / 'split_embedding.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

class SplitTest(unittest.TestCase):
    def test_external_views_match_embedding_lookup(self):
        with tempfile.TemporaryDirectory(dir=scripts) as directory:
            root = Path(directory)
            weights = np.arange(30, dtype=np.float16).reshape(5, 6)
            (root / 'weights.bin').write_bytes(b'padding!' + weights.tobytes())
            w = onnx.TensorProto(); w.name = 'lm_head.MatMul.weight'; w.data_type = TensorProto.FLOAT16
            w.dims.extend([5, 6]); w.data_location = TensorProto.EXTERNAL
            for key, value in [('location', 'weights.bin'), ('offset', '8'), ('length', str(weights.nbytes))]:
                e = w.external_data.add(); e.key = key; e.value = value
            nodes = [helper.make_node('Transpose', [w.name], ['t'], perm=[1, 0]),
                     helper.make_node('Gather', ['t', 'ids'], ['embedding'], axis=0)]
            model = helper.make_model(helper.make_graph(nodes, 'g', [], [], [w]))
            info = mod.transform(model, root, chunk_size=2)
            self.assertTrue(all(n < mod.LIMIT for n in info['chunk_elements']))
            views = []
            raw = (root / 'weights.bin').read_bytes()
            for tensor in list(model.graph.initializer)[1:]:
                meta = {e.key: e.value for e in tensor.external_data}
                views.append(np.frombuffer(raw, dtype=np.float16, count=int(meta['length']) // 2,
                    offset=int(meta['offset'])).reshape(tuple(tensor.dims)))
            ids = np.array([[0, 3, 5]])
            np.testing.assert_array_equal(np.concatenate([v.T[ids] for v in views], axis=-1), weights.T[ids])
            self.assertEqual([n.op_type for n in model.graph.node], ['Transpose'] * 3 + ['Gather'] * 3 + ['Concat'])

    def test_build_rejects_tampered_parent_before_creating_output(self):
        import json
        with tempfile.TemporaryDirectory(dir=scripts) as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            (source / 'model.onnx').write_bytes(b'tampered')
            (source / 'export_manifest.json').write_text(json.dumps({'files': [
                {'path': 'model.onnx', 'bytes': 8, 'sha256': '0' * 64}]}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                mod.build(source, root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_rejects_ambiguous_transpose(self):
        with tempfile.TemporaryDirectory(dir=scripts) as directory:
            root=Path(directory); (root / 'w').write_bytes(bytes(24))
            w=onnx.TensorProto(); w.name='lm_head.MatMul.weight'; w.data_type=TensorProto.FLOAT16
            w.dims.extend([3,4]); w.data_location=TensorProto.EXTERNAL
            for k,v in [('location','w'),('offset','0'),('length','24')]:
                x=w.external_data.add(); x.key=k; x.value=v
            model=helper.make_model(helper.make_graph([helper.make_node('Transpose',[w.name],['x'],perm=[0,1])], 'g', [], [], [w]))
            with self.assertRaises(ValueError): mod.transform(model, root, chunk_size=2)

if __name__ == '__main__': unittest.main()
