import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_extract", Path(__file__).resolve().parents[2] / "tools" / "extract.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_kwiver_failure_and_rate(tmp_path):
    with patch.object(m.sys, 'argv', ['extract', '-o', str(tmp_path), '-r', '2']), patch.object(m.subprocess, 'call', return_value=7) as call:
        assert m.main() == 1
    assert call.call_args.args[0][-2:] == ['-frate', '2']


def test_directory_inputs_are_files_only(tmp_path):
    (tmp_path / 'folder').mkdir()
    (tmp_path / 'video.mp4').touch()
    assert m.list_files_in_dir(str(tmp_path)) == [str(tmp_path / 'video.mp4')]
