"""CPU-only audits of tiny real safetensors checkpoints and local tokenizer fixtures."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Checkpoint tensor fixtures require the remote Torch environment; notebook tools are intentionally Torch-free")

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers

path = Path(__file__).resolve().parents[1] / "inspect_gptq_checkpoint.py"
spec = importlib.util.spec_from_file_location("inspect_gptq_checkpoint", path)
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)


class CheckpointInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact, self.source, self.samples = [self.root / name for name in ("artifact", "source", "samples")]
        for folder in (self.artifact, self.source, self.samples):
            folder.mkdir()
        self.config = {"model_type": "qwen2", "quantization_config": {"quant_method": "gptq", "bits": 4,
            "checkpoint_format": "gptq", "lm_head": True, "dynamic": {"+:.*lm_head*": {"bits": 8}}}}
        (self.artifact / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        (self.source / "source_manifest.json").write_text(json.dumps({"repo_id": "fixture/source", "revision": "a" * 40}), encoding="utf-8")
        self.save_tensors()
        for folder in (self.artifact, self.source):
            self.save_tokenizer(folder)
            (folder / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        for split, count in (("calibration", 256), ("evaluation", 200)):
            rows = [{"id": f"{split}-{i}", "prompt": "hello world", "prompt_ids": [1, 2]} for i in range(count)]
            (self.samples / (split + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def save_tensors(self, value=0.125):
        save_file({"layer.qweight": torch.tensor([[1, -2]], dtype=torch.int32, device="cpu"),
                   "layer.scales": torch.tensor([value, 0.25], dtype=torch.float32, device="cpu"),
                   "lm_head.scales": torch.tensor([0.5], dtype=torch.bfloat16, device="cpu")}, str(self.artifact / "model.safetensors"))

    @staticmethod
    def save_tokenizer(folder, swapped=False):
        vocab = {"[UNK]": 0, "hello": 2 if swapped else 1, "world": 1 if swapped else 2}
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(folder / "tokenizer.json"))

    def audit(self):
        return inspector.inspect(self.artifact, self.source, self.samples)

    def test_finite_real_checkpoint_counts_dtypes_cpu_and456_ids(self):
        report = self.audit()
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["tensors"]["tensor_count"], 3)
        self.assertEqual(report["tensors"]["tensor_counts"], {"floating": 2, "integer": 1})
        self.assertEqual(report["tensors"]["element_counts"], {"floating": 3, "integer": 2})
        self.assertEqual(report["tensors"]["dtype_tensor_counts"], {"float32": 1, "int32": 1, "bfloat16": 1})
        self.assertEqual(report["tensors"]["device"], "cpu")
        self.assertEqual(report["tensors"]["hip_visible_devices"], "-1")
        self.assertEqual(report["tokenizer"]["prompts_checked"], 456)
        self.assertTrue(report["stable_artifact_files"])
        self.assertFalse(report["qualified"])
        self.assertTrue(all(entry["size"] > 0 and len(entry["sha256"]) == 64 for entry in report["files"]))

    def test_one_nan_rejects_checkpoint_and_identifies_tensor(self):
        self.save_tensors(float("nan"))
        report = self.audit()
        self.assertFalse(report["passed"])
        self.assertEqual(report["tensors"]["non_finite_tensors"], [{"tensor": "layer.scales", "file": "model.safetensors", "dtype": "float32", "shape": [2], "non_finite_elements": 1}])
        self.assertTrue(report["tokenizer"]["passed"])

    def test_tokenizer_id_swap_fails_despite_same_prompt_text_and_tensor_values(self):
        self.save_tokenizer(self.artifact, swapped=True)
        report = self.audit()
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["tokenizer"]["id_mismatches"]), 456)
        self.assertEqual(report["tokenizer"]["stored_id_mismatches"], [])
        self.assertTrue(report["tensors"]["passed"])

    def test_stored_prompt_ids_are_checked_independently_of_tokenizer_parity(self):
        path = self.samples / "calibration.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["prompt_ids"] = [1, 0]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        report = self.audit()
        self.assertFalse(report["passed"])
        self.assertEqual(report["tokenizer"]["id_mismatches"], [])
        self.assertEqual(report["tokenizer"]["stored_id_mismatches"], [{"split": "calibration", "id": "calibration-0"}])

    def test_metadata_bits_mismatch_is_not_presented_as_valid_gptq4(self):
        self.config["quantization_config"]["bits"] = 8
        (self.artifact / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        report = self.audit()
        self.assertFalse(report["passed"])
        self.assertFalse(report["config"]["passed"])
        self.assertTrue(report["config"]["declarations_only"])

    def test_indexed_two_shard_checkpoint_is_complete(self):
        (self.artifact / "model.safetensors").unlink()
        save_file({"layer.qweight": torch.tensor([1, 2], dtype=torch.int32, device="cpu")}, str(self.artifact / "part1.safetensors"))
        save_file({"layer.scales": torch.tensor([0.1], dtype=torch.float16, device="cpu")}, str(self.artifact / "part2.safetensors"))
        (self.artifact / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer.qweight": "part1.safetensors", "layer.scales": "part2.safetensors"}}), encoding="utf-8")
        report = self.audit()
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["tensors"]["tensor_count"], 2)

    def test_index_cannot_reference_file_outside_artifact(self):
        (self.artifact / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer.qweight": "../outside.safetensors"}}), encoding="utf-8")
        report = self.audit()
        self.assertFalse(report["passed"])
        self.assertTrue(any("Unsafe checkpoint-relative path" in value for value in report["failures"]))

    def test_report_inside_artifact_or_source_is_refused_before_audit(self):
        for folder in (self.artifact, self.source):
            out = folder / "report.json"
            args = ["--artifact", str(self.artifact), "--source", str(self.source), "--samples", str(self.samples), "--out", str(out)]
            with self.subTest(folder=folder.name), patch.object(inspector, "inspect") as audit:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    inspector.main(args)
                self.assertEqual(error.exception.code, 2)
                audit.assert_not_called()
                self.assertFalse(out.exists())

    def test_artifact_source_alias_and_nested_paths_are_refused(self):
        for artifact, source in ((self.source, self.source), (self.source / "nested", self.source)):
            with self.subTest(artifact=artifact), self.assertRaisesRegex(ValueError, "non-overlapping"):
                inspector.validate_paths(artifact, source, self.root / "out.json")

    def test_changed_artifact_during_audit_is_detected(self):
        original = inspector.tokenizer_audit
        def change(*args):
            result = original(*args)
            with (self.artifact / "tokenizer_config.json").open("a", encoding="utf-8") as stream:
                stream.write(" ")
            return result
        with patch.object(inspector, "tokenizer_audit", side_effect=change):
            report = self.audit()
        self.assertFalse(report["passed"])
        self.assertFalse(report["stable_artifact_files"])


if __name__ == "__main__":
    unittest.main()
