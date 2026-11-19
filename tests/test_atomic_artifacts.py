from pathlib import Path

import pytest

from src import artifacts


def test_reader_lock_retries_then_publishes_complete_file(tmp_path, monkeypatch):
    target = tmp_path / "data.csv"
    target.write_bytes(b"previous")
    real_replace = artifacts.os.replace
    attempts = []

    def replace(source, destination):
        attempts.append(source)
        assert target.read_bytes() == b"previous"
        if len(attempts) < 3:
            raise PermissionError("File is temporarily open by a reader")
        real_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", replace)
    monkeypatch.setattr(artifacts.time, "sleep", lambda seconds: None)
    artifacts.atomic_write_bytes(target, b"complete replacement")
    assert len(attempts) == 3
    assert target.read_bytes() == b"complete replacement"
    assert not list(tmp_path.glob("*.tmp"))


def test_persistent_write_failure_keeps_previous_artifact(tmp_path, monkeypatch):
    target = tmp_path / "active.json"
    target.write_bytes(b"previous active bundle")

    def locked(*args):
        raise PermissionError("Still locked")

    monkeypatch.setattr(artifacts.os, "replace", locked)
    monkeypatch.setattr(artifacts.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        artifacts.atomic_write_bytes(target, b"replacement")
    assert target.read_bytes() == b"previous active bundle"
    assert not list(tmp_path.glob("*.tmp"))
