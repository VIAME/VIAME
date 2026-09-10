from .tool_regression_helpers import tool_env, run_viame


def test_boolean_flag_before_pipeline(tool_env):
    r = run_viame(tool_env, 'run', '--help', 'not-needed.pipe')
    assert r.returncode == 0
    assert 'supplemental configuration' in r.stdout
    assert 'two modes' not in r.stdout


def test_staged_settings_equals_and_relative_paths(tool_env, tmp_path):
    # Dumping the pipe exercises stage parsing without executing processes.
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'asset.txt').write_text('asset')
    p = source / 'stages.pipe'
    p.write_text('pipeline stage 1:\nconfig global\n  relativepath asset = asset.txt\n  value = original\n')
    source.chmod(0o555)
    try:
        r = run_viame(tool_env, 'run', str(p), '--setting=global:value=updated', '--dump-pipe')
        assert r.returncode == 0, r.stderr
        assert str(source / 'asset.txt') in r.stdout
        assert not list(source.glob('.viame_stage*'))
    finally:
        source.chmod(0o755)
