import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location("compare_results", Path(__file__).parents[1] / "compare_results.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Tokenizer:
    def encode(self, text, **kwargs):
        return SimpleNamespace(ids=list(text.encode()))


def test_failed_or_partial_reports_never_qualify():
    corpus = [{"id": "x", "target": "abc"}]
    with pytest.raises(ValueError, match="complete"):
        m.summarize({"status": "completed", "items": []}, corpus, Tokenizer())
    with pytest.raises(ValueError, match="failed"):
        m.summarize({"status": "failed", "items": [{"id": "x"}]}, corpus, Tokenizer())


def test_eos_excluded_and_apt_compares_reference():
    result = m.summarize({"status": "completed", "items": [{"id": "x", "generated_text": "abc<|endoftext|>"}]}, [{"id": "x", "target": "abc"}], Tokenizer())
    assert result["accepted_prefix_tokens"] == 3
    assert result["trimmed_exact_match"] == 1
