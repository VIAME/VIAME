import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_database", Path(__file__).resolve().parents[2] / "tools" / "database.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_stop_is_scoped():
    with patch.object(m, '_execute_cmd') as call:
        assert m.stop(database_dir='custom')
    assert call.call_args_list == [(('pg_ctl', ['-D', 'custom/SQL', '-m', 'fast', 'stop']),)]


def test_declining_reset_has_no_side_effects(tmp_path):
    with patch.object(m, 'query_yes_no', return_value=False), patch.object(m, 'stop') as stop:
        assert m.init(database_dir=str(tmp_path)) == [False, True]
        stop.assert_not_called()


def test_failed_stop_prevents_reset(tmp_path):
    (tmp_path / 'SQL').mkdir()
    marker = tmp_path / 'keep'
    marker.write_text('data')
    with patch.object(m, 'stop', return_value=False):
        assert m.init(database_dir=str(tmp_path), prompt=False) == [False, False]
    assert marker.read_text() == 'data'
