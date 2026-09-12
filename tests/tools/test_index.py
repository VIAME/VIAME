import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_index", Path(__file__).resolve().parents[2] / "tools" / "index.py")
m = importlib.util.module_from_spec(spec)
import sys, types
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
with patch.dict(sys.modules, {'viame': types.ModuleType('viame'), 'viame.core': types.SimpleNamespace(index_descriptors=types.SimpleNamespace(list_index_bundles=lambda path: []))}):
    spec.loader.exec_module(m)

def test_backend_switch_refused(tmp_path):
    (tmp_path / 'SQL').mkdir()
    args = m.argparse.Namespace(database=str(tmp_path), backend='files')
    with patch.object(m.index_descriptors, 'list_index_bundles', return_value=[]), pytest.raises(SystemExit, match='backend changes'):
        m.resolve_backend(args)


def test_read_only_operation_does_not_initialize(tmp_path):
    with patch.object(m.database, 'init') as init, pytest.raises(SystemExit, match='No PostgreSQL'):
        m.ensure_postgres(str(tmp_path))
    init.assert_not_called()


def test_ingest_failure_exposes_current_pipeline_error(tmp_path, capsys):
    import os
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'current.txt').write_text('DEBUG loading plugins\nCaught unhandled std::exception: RuntimeError: No CUDA GPUs are available\nAt:\n  model.py(89)\n')
    stale = logs / 'old.txt'
    stale.write_text('ERROR: stale failure\n')
    os.utime(stale, (1, 1))
    m.report_ingest_errors(str(tmp_path), 2)
    output = capsys.readouterr().err
    assert 'RuntimeError: No CUDA GPUs are available' in output
    assert 'model.py(89)' in output
    assert 'stale failure' not in output
    assert 'DEBUG loading plugins' not in output


def test_missing_ingest_logs_do_not_mask_original_error(tmp_path):
    m.report_ingest_errors(str(tmp_path), 0)
