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
