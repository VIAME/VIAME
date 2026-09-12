"""Device selection for indexing descriptors, without requiring a CUDA machine."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    'feature_utilities', Path(__file__).resolve().parents[3] / 'plugins/pytorch/utilities.py')
utilities = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utilities)


@pytest.mark.parametrize('requested', [None, [0], [2]])
def test_no_cuda_uses_cpu(monkeypatch, requested):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        device=lambda value: value,
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)))
    with pytest.warns(UserWarning, match='using CPU'):
        assert utilities.get_gpu_device(requested) == ('cpu', False)


@pytest.mark.parametrize('requested, expected', [(None, 'cuda:0'), ([1], 'cuda:1'), ([], 'cpu')])
def test_available_gpu_and_explicit_cpu(monkeypatch, requested, expected):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        device=lambda value: value,
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)))
    assert utilities.get_gpu_device(requested) == (expected, expected != 'cpu')
