import json
import os
import signal
import subprocess

import pytest
from .tool_test_helpers import tool_env, run_viame


@pytest.mark.skipif(os.name != 'posix', reason='Requires POSIX file-size limits')
def test_write_failure_keeps_original(tool_env, tmp_path):
    import resource
    p = tmp_path / 'tracks.json'
    text = json.dumps({'version': 2, 'tracks': {'1': {'id': 1, 'begin': 0, 'end': 0,
        'confidencePairs': [['fish', 1]], 'features': [{'frame': 0, 'bounds': [0, 0, 10, 10]}]}},
        'preserved': 'x' * 10000})
    p.write_text(text)
    def limit():
        resource.setrlimit(resource.RLIMIT_FSIZE, (512, 512))
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    result = subprocess.run(['viame', 'json', '-i', str(p), '--increase-fid'],
                            env=tool_env, capture_output=True, text=True,
                            preexec_fn=limit, timeout=60)
    assert result.returncode != 0
    assert p.read_text() == text
    assert not list(tmp_path.glob('.viame-write-*'))
