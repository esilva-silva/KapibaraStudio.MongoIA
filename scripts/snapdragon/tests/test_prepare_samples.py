"""Bounded corpus generation tests; unittest also works without pytest installed."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "prepare_samples.py"
spec = importlib.util.spec_from_file_location("prepare_samples", MODULE)
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)


class CharacterTokenizer:
    """Exact reversible test codec; production always requires HF tokenizers."""
    marker_ids = (1, 2, 3)

    def __init__(self, path):
        self.path = Path(path)

    def encode(self, text):
        return [ord(c) + 4 for c in text]

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def decode(self, ids):
        markers = dict(zip(self.marker_ids, ps.rc.QWEN_FIM))
        return "".join(markers[i] if i in markers else chr(i - 4) for i in ids)


class PreparationTests(unittest.TestCase):
    def test_real_generators_reproducible_disjoint_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "samples.yaml"
            cfg.write_text("samples:\n  calibration_count: 12\n  evaluation_count: 10\n  initial_snippets: 64\n  max_snippets: 256\n  max_prompt_tokens: 128\nmodels:\n  nano:\n    revision: frozen-revision\n", encoding="utf-8")
            tok = root / "tokenizer.json"
            tok.write_text("test codec", encoding="utf-8")
            with patch.object(ps.rc, "Tokenizer", CharacterTokenizer):
                a = ps.prepare(cfg, tok, root / "a")
                b = ps.prepare(cfg, tok, root / "b")
            self.assertEqual(a, b)
            for name, n in (("calibration", 12), ("evaluation", 10)):
                self.assertEqual(a["files"][name]["count"], n)
                self.assertEqual((root / "a" / f"{name}.jsonl").read_bytes(), (root / "b" / f"{name}.jsonl").read_bytes())
                rows = [ps.json.loads(line) for line in (root / "a" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertTrue(all(len(r["prompt_ids"]) <= 128 for r in rows))
                self.assertTrue(all(r["prompt_ids"][0] == 1 and r["prompt_ids"][-1] == 3 for r in rows))
            self.assertEqual(a["models"]["nano"]["revision"], "frozen-revision")
            self.assertEqual(a["validation"]["exec"], "not_measured")

    def test_overlap_is_rejected(self):
        row = {"group_id": "same", "split_key": "same", "schema_id": "s", "prompt_sha256": "h"}
        with self.assertRaisesRegex(ValueError, "overlap"):
            ps.validate([row], [copy.deepcopy(row)], 1024)

    def test_missing_role_and_bounded_exhaustion_fail(self):
        cfg = ps.gs.load_config()
        with self.assertRaisesRegex(ValueError, "No schemas"):
            ps.generate_role(cfg, [], CharacterTokenizer("unused"), "eval", 10, 1)
        schemas = ps.gs.load_schemas(ps.ROOT / cfg["schemas_dir"])
        with patch.object(ps.gs, "generate", return_value=[]), patch.object(ps.gf, "generate_fim", return_value=[]):
            with self.assertRaisesRegex(ValueError, "bounded pool exhausted"):
                ps.generate_role(cfg, schemas, CharacterTokenizer("unused"), "train", 10, 1, 1, 2)

    def test_oversized_configuration_fails_before_tokenizer_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "bad.yaml"
            config.write_text("samples:\n  max_snippets: 3000000\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside supported limits"):
                ps.prepare(config, Path(tmp) / "missing.json", Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
