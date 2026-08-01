from pathlib import Path

import pytest

from direct_s2st.io import ExistingOutputError, atomic_write_json, read_jsonl


def test_atomic_write_refuses_implicit_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    atomic_write_json(path, {"value": 1})
    with pytest.raises(ExistingOutputError):
        atomic_write_json(path, {"value": 2})
    assert atomic_write_json(path, {"value": 1}, resume=True) is False


def test_jsonl_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2"):
        list(read_jsonl(path))
