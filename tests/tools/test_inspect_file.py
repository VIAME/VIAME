import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_inspect_file", Path(__file__).resolve().parents[2] / "tools" / "inspect_file.py")
m = importlib.util.module_from_spec(spec)
import sys, types
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
with patch.dict(sys.modules, {'viame': types.ModuleType('viame'), 'viame.core': types.SimpleNamespace(model_wrap=types.SimpleNamespace())}):
    spec.loader.exec_module(m)

def test_malformed_json_does_not_abort_other_reports(tmp_path):
    broken, valid = tmp_path / 'broken.json', tmp_path / 'valid.json'
    broken.write_text('{"images":null,"annotations":1}')
    valid.write_text('{}')
    assert m.inspect_path(str(broken)).integrity.startswith('corrupt')
    assert m.inspect_path(str(valid)).integrity.startswith('ok')


def test_unreadable_input_becomes_report():
    with patch.object(m, '_inspect_path', side_effect=OSError('permission denied')):
        report = m.inspect_path('input')
    assert 'permission denied' in report.integrity
