import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_run_bulk", Path(__file__).resolve().parents[2] / "tools" / "run.py")
m = importlib.util.module_from_spec(spec)
import sys, types
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
with patch.dict(sys.modules, {'viame': types.ModuleType('viame'), 'viame.core': types.SimpleNamespace(model_wrap=types.SimpleNamespace())}):
    spec.loader.exec_module(m)

def test_worker_failures_are_aggregated():
    visited = []
    def process(entry, gpu, cpu):
        visited.append(entry)
        if entry == 'exception':
            raise RuntimeError('failed')
        if entry == 'exit':
            raise SystemExit(1)
        return 7 if entry == 'status' else 0
    failures = m.run_jobs(['ok', 'exception', 'exit', 'status'], process, [(0, 0), (0, 1)])
    assert len(visited) == 4
    assert {entry for entry, _ in failures} == {'exception', 'exit', 'status'}


def test_main_errors_fail():
    with pytest.raises(SystemExit) as exc:
        m.exit_with_error('bad input')
    assert exc.value.code == 1
