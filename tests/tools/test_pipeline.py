import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_pipeline", Path(__file__).resolve().parents[2] / "tools" / "pipeline.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_repeated_include_is_not_a_cycle(tmp_path):
    p = tmp_path / 'main.pipe'
    common = tmp_path / 'common.pipe'
    common.write_text('config global\n  value = 1\n')
    p.write_text('include common.pipe\ninclude common.pipe\n')
    assert not m.load(p).errors
    common.write_text('include main.pipe\n')
    assert any('cycle' in e for e in m.load(p).errors)


def test_repeated_setting_preserves_comment(tmp_path):
    p = tmp_path / 'main.pipe'
    p.write_text('config global\n  value = 1 # comment\n')
    assert m.main(['set', str(p), '-s', 'global:value=long', '-s', 'global:value=2']) == 0
    assert p.read_text() == 'config global\n  value = 2 # comment\n'
