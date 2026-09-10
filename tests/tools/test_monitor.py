import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_monitor", Path(__file__).resolve().parents[2] / "tools" / "monitor.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_epoch_ignores_configured_maximum():
    assert m.nh_epoch('Maximum epoch: 100\nepoch=5\nepoch=6') == 6
    assert m.nh_epoch('epoch=100\nrestart\nepoch=2') == 2


def test_pid_reuse_is_not_killed(tmp_path):
    (tmp_path / 'monitor.pid').write_text('42')
    (tmp_path / 'monitor.identity.json').write_text('{"pid":42,"identity":"old"}')
    args = m.argparse.Namespace(output_dir=str(tmp_path))
    with patch.object(m, 'pid_running', return_value=True), patch.object(m, 'process_identity', return_value='new'), patch.object(m.os, 'kill') as kill:
        assert m.cmd_stop(args, []) == 1
        kill.assert_not_called()


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_invalid_poll_interval(value, tmp_path):
    with pytest.raises(SystemExit):
        m.main(['start', '-o', str(tmp_path), '-l', 'log', '--poll-seconds', value])
