"""CPU checks for isolated GPU probing and automatic transfer selection."""
import ast
from pathlib import Path
from types import SimpleNamespace
import warnings

import pytest


ROOT = Path(__file__).resolve().parents[3] / 'plugins/pytorch/netharn'


def load_function(filename, name, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('body, expected', [
    ('print("[]")', []),
    ('print(\'[[0, 1, "copy"]]\')', [[0, 1, 'copy']]),
    ('raise RuntimeError("driver failure")', 'exited'),
    ('import time; time.sleep(30)', 'timed out'),
    ('print("invalid result")', 'invalid'),
    ('print("{}")', 'invalid'),
])
def test_probe_subprocess(tmp_path, body, expected):
    worker = tmp_path / 'probe.py'
    worker.write_text(body)
    probe = load_function('host_parallel.py', 'probe_peer_copies', {
        '__file__': str(worker),
    })
    result = probe([0, 1], timeout=0.5)
    if isinstance(expected, list):
        assert result == expected
    else:
        assert expected in result[0][2]


@pytest.mark.parametrize('mode, failures, host, raises', [
    ('auto', [], False, False),
    ('auto', [(0, 1, 'timed out')], True, False),
    ('auto', [(0, 1, 'copy')], True, False),
    ('host', [], True, False),
    ('peer', [], False, False),
    ('require', [(0, 1, 'timed out')], False, True),
    ('single', [(0, 1, 'timed out')], False, False),
])
def test_parallel_policy(mode, failures, host, raises):
    calls = []
    def probe(ids):
        calls.append(ids)
        return failures
    plan = load_function('device.py', '_plan_parallel', {
        'probe_peer_copies': probe, 'warnings': warnings,
    })
    xpu = SimpleNamespace(_device_ids=[0, 1], _main_device_id=0, p2p=mode)
    if raises:
        with pytest.raises(RuntimeError):
            plan(xpu)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ids, staged = plan(xpu)
        assert staged == host
        assert ids == ([0] if mode == 'single' and failures else [0, 1])
    assert len(calls) == (0 if mode in ('host', 'peer') else 1)


@pytest.mark.parametrize('failures, expected', [([], 'peer'), ([(0, 1, 'copy')], 'host')])
def test_startup_selection_is_reused_at_mount(failures, expected):
    calls = []
    def probe(ids):
        calls.append(ids)
        return failures
    plan = load_function('device.py', '_plan_parallel', {
        'probe_peer_copies': probe, 'warnings': warnings,
    })
    prepare = load_function('device.py', 'prepare_parallel', {})
    xpu = SimpleNamespace(_device_ids=[0, 1], _main_device_id=0, p2p='auto')
    xpu._plan_parallel = lambda: plan(xpu)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prepare(xpu)
    assert xpu.p2p == expected
    plan(xpu)
    assert calls == [[0, 1]]


@pytest.mark.parametrize('ids', [None, [0]])
def test_no_probe_for_cpu_or_single_gpu(ids):
    def unexpected_probe(ids):
        pytest.fail('CPU and single GPU must not probe peer transfers')
    plan = load_function('device.py', '_plan_parallel', {
        'probe_peer_copies': unexpected_probe,
    })
    assert plan(SimpleNamespace(_device_ids=ids, p2p='auto')) == (ids, False)
