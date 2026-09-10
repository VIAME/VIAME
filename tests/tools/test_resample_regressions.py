import pytest
from .tool_regression_helpers import tool_env, run_viame


@pytest.mark.parametrize('rate', ['nan', 'inf', '-inf'])
def test_nonfinite_rates_rejected(tool_env, tmp_path, rate):
    p = tmp_path / 'tracks.csv'
    p.write_text('42,a.png,0,0,0,10,10,1,-1,fish,1\n')
    out = tmp_path / 'out.csv'
    r = run_viame(tool_env, 'resample', '-i', str(p), '-o', str(out), '--input-rate', '1', '--output-rate', rate)
    assert r.returncode != 0
    assert not out.exists()
