import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import onnx
from onnx import helper, TensorProto
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_qnn_transfer import prepare


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name); self.source = self.root/'source'; self.source.mkdir()
        self.old = self.root/'old'; self.old.mkdir()
        for name in ['genai_config.json','config.json','generation_config.json','tokenizer.json','tokenizer_config.json']:
            (self.source/name).write_text('{}'); (self.old/name).write_text('{}')
        components = []
        for name in ['embeddings','context','iterator','lm_head']:
            graph = helper.make_graph([helper.make_node('Identity',['x'],['y'])], name,
                [helper.make_tensor_value_info('x',TensorProto.FLOAT,[1])], [helper.make_tensor_value_info('y',TensorProto.FLOAT,[1])])
            onnx.save(helper.make_model(graph), self.source/(name+'.onnx'))
            components.append({'type':'onnxmodel','config':{'model_path':str(self.source),'onnx_file_name':name+'.onnx',
                'model_attributes':{'additional_files':[str(self.old/'config.json')]}}})
        model = {'type':'compositemodel','config':{'model_path':str(self.old), 'model_component_names':['embeddings','context','iterator','lm_head'],
            'model_components':components,'model_attributes':{'additional_files':[str(self.old/'config.json'), str(self.old/'generation_config.json')]}}}
        (self.source/'model_config.json').write_text(json.dumps(model))
        self.snapshot = self.root/'snapshot'; self.snapshot.mkdir()
        attribution = []
        for name in ('LICENSE', 'NOTICE.md', 'REPLAY_ATTRIBUTION.md'):
            data = ('original '+name).encode(); (self.snapshot/name).write_bytes(data)
            attribution.append({'path':name,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        self.provenance = self.root/'provenance.json'; self.provenance.write_text(json.dumps({'source':{'revision':'a'*40,'files':attribution},'calibration_sha256':'b'*64}))

    def tearDown(self): self.temp.cleanup()

    def test_roundtrip_preserves_source_and_normalizes(self):
        before = (self.source/'model_config.json').read_bytes()
        output = self.root/'out'; prepare(self.source, output, self.provenance, [], self.snapshot)
        result = json.loads((output/'model_config.json').read_text())['config']
        self.assertEqual(result['model_path'], '.')
        self.assertEqual(result['model_components'][0]['config']['model_path'], '.')
        self.assertEqual(result['model_attributes']['additional_files'], ['config.json', 'generation_config.json'])
        self.assertEqual((output/'model_config.source.json').read_bytes(), before)
        self.assertEqual((self.source/'model_config.json').read_bytes(), before)
        manifest = json.loads((output/'export_manifest.json').read_text())
        self.assertFalse(manifest['qualified']); self.assertEqual(manifest['schema'], 'kapi-qnn-transfer-v1')
        for name in ('LICENSE', 'NOTICE.md', 'REPLAY_ATTRIBUTION.md'):
            self.assertEqual((output/name).read_bytes(), (self.snapshot/name).read_bytes())
            self.assertIn(name, [item['path'] for item in manifest['files']])

    def test_invalid_path_and_existing_target_rejected(self):
        with self.assertRaises(ValueError): prepare(self.source, self.source/'nested', self.provenance, [], self.snapshot)
        with self.assertRaises(FileExistsError): prepare(self.source, self.old, self.provenance, [], self.snapshot)
        model=json.loads((self.source/'model_config.json').read_text()); model['config']['model_components'][0]['config']['onnx_file_name']='../escape.onnx'
        (self.source/'model_config.json').write_text(json.dumps(model))
        with self.assertRaises(ValueError): prepare(self.source, self.root/'out', self.provenance, [], self.snapshot)

    def test_missing_or_different_resource_rejected_without_output(self):
        (self.old/'config.json').write_text('changed')
        with self.assertRaises(ValueError): prepare(self.source, self.root/'out', self.provenance, [], self.snapshot)
        self.assertFalse((self.root/'out').exists())
        (self.old/'config.json').unlink()
        with self.assertRaises(FileNotFoundError): prepare(self.source, self.root/'out', self.provenance, [], self.snapshot)

    def test_attribution_must_match_pinned_provenance(self):
        (self.snapshot/'LICENSE').write_text('altered')
        with self.assertRaises(ValueError): prepare(self.source, self.root/'out', self.provenance, [], self.snapshot)
        self.assertFalse((self.root/'out').exists())
        (self.snapshot/'LICENSE').unlink()
        with self.assertRaises(FileNotFoundError): prepare(self.source, self.root/'out', self.provenance, [], self.snapshot)

    def test_preserves_original_tokenizer_and_adds_auxiliary_resources(self):
        provenance = json.loads(self.provenance.read_text())
        for name in ('tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json', 'chat_template.jinja'):
            data = ('pinned tokenizer '+name).encode(); (self.snapshot/name).write_bytes(data)
            provenance['source']['files'].append({'path': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        (self.snapshot/'config.json').write_text('original model config')
        (self.source/'config.json').write_text('derived 2048 config')
        (self.old/'config.json').write_text('derived 2048 config')
        self.provenance.write_text(json.dumps(provenance))
        output = self.root/'out'; prepare(self.source, output, self.provenance, [], self.snapshot, True)
        model = json.loads((output/'model_config.json').read_text())['config']
        for name in ('tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json', 'chat_template.jinja'):
            self.assertEqual((output/name).read_bytes(), (self.snapshot/name).read_bytes())
            for attributes in [model['model_attributes']] + [item['config']['model_attributes'] for item in model['model_components']]:
                self.assertEqual(attributes['additional_files'].count(name), 1)
        self.assertEqual((output/'config.json').read_text(), 'derived 2048 config')
        self.assertEqual((output/'generation_config.json').read_text(), '{}')

    def test_tokenizer_hash_mismatch_refuses_before_output(self):
        provenance = json.loads(self.provenance.read_text())
        for name in ('tokenizer.json', 'tokenizer_config.json'):
            data = name.encode(); (self.snapshot/name).write_bytes(data)
            provenance['source']['files'].append({'path':name,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        self.provenance.write_text(json.dumps(provenance))
        (self.snapshot/'tokenizer.json').write_bytes(b'bad')
        with self.assertRaises(ValueError): prepare(self.source, self.root/'out', self.provenance, [], self.snapshot, True)
        self.assertFalse((self.root/'out').exists())


if __name__ == '__main__': unittest.main()
