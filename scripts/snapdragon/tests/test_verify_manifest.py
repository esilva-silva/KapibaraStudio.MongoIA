import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import pytest

spec = importlib.util.spec_from_file_location("verify_manifest", Path(__file__).parents[1] / "verify_manifest.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_transfer_corruption_and_missing_file():
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        (root / "model").write_bytes(b"weights")
        files = [{"path": "model", "size": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}]
        (root / "source_manifest.json").write_text(json.dumps({"files": files}))
        assert module.verify(root)["passed"]
        (root / "model").write_bytes(b"changed")
        assert module.verify(root)["failures"][0]["reason"] == "sha256"
        (root / "model").unlink()
        assert module.verify(root)["failures"][0]["reason"] == "missing"


def test_manifest_cannot_escape_package():
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        (root / "source_manifest.json").write_text(json.dumps({"files": [{"path": "../other", "sha256": "x"}]}))
        with pytest.raises(ValueError, match="Invalid"):
            module.verify(root)
