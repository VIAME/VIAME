import json
from .tool_test_helpers import tool_env, run_viame


def test_partial_extraction_is_explicit(tool_env, tmp_path):
    p = tmp_path / 'partial.pipe'
    p.write_text('process reader\n  :: video_input\nprocess broken\n  :: no_such_process\n')
    output = tmp_path / 'output.json'
    r = run_viame(tool_env, 'configs', '-i', str(p), '-o', str(output))
    assert r.returncode == 0, r.stderr
    assert json.loads(output.read_text())['_errors']
    output.write_text('original')
    r = run_viame(tool_env, 'configs', '-i', str(p), '-o', str(output), '--strict')
    assert r.returncode != 0
    assert output.read_text() == 'original'
