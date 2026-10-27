"""Falhas de Git e arvore suja nunca autorizam publicar um lock aparentemente limpo."""
import subprocess
import sys
import pytest
import write_model_lock as lock


def test_git_failure_is_not_an_empty_clean_tree(monkeypatch):
    def fail(*args, **kwargs):
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(128, args[0])
    monkeypatch.setattr(lock.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        lock.git("status", "--porcelain")


def test_require_clean_aborts_before_reading_or_overwriting_models(tmp_path, monkeypatch):
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["write_model_lock.py", "--require-clean"])
    monkeypatch.setattr(lock, "git", lambda *args: " M scripts/student/train_student.py")
    original = tmp_path / "model.lock.json"
    original.write_text("sentinel", encoding="utf-8")
    with pytest.raises(SystemExit, match="sujos"):
        lock.main()
    assert original.read_text(encoding="utf-8") == "sentinel"


@pytest.mark.parametrize("repo", ["KapiCoder-Mongo-Nano", "KapiCoder-Mongo-Small"])
def test_require_clean_includes_training_configs_even_when_only_v3_config_changed(tmp_path, monkeypatch, repo):
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["write_model_lock.py", "--require-clean"])
    target = tmp_path / "Repos" / repo / "model.lock.json"
    target.parent.mkdir(parents=True)
    target.write_text("sentinel", encoding="utf-8")
    status_calls = []

    def git_status(*args):
        status_calls.append(args)
        assert args[:2] == ("status", "--porcelain")
        assert "--untracked-files=all" in args
        assert ":(glob)Repos/*/configs/**" in args
        return f" M Repos/{repo}/configs/sft_v3.yaml"

    monkeypatch.setattr(lock, "git", git_status)
    with pytest.raises(SystemExit, match="sujos"):
        lock.main()
    assert len(status_calls) == 1
    assert target.read_text(encoding="utf-8") == "sentinel"


def test_require_clean_git_failure_aborts_before_reading_or_overwriting_models(tmp_path, monkeypatch):
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["write_model_lock.py", "--require-clean"])

    def fail(*args):
        raise subprocess.CalledProcessError(128, ["git", *args])

    monkeypatch.setattr(lock, "git", fail)
    with pytest.raises(subprocess.CalledProcessError):
        lock.main()
    assert not (tmp_path / "Repos").exists()
