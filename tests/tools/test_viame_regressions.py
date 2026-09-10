from .tool_regression_helpers import tool_env, run_viame


def test_unknown_help_is_normal_error(tool_env):
    r = run_viame(tool_env, 'help', 'no-such-applet')
    assert r.returncode > 0
    assert 'not found' in r.stderr


def test_global_help(tool_env):
    r = run_viame(tool_env, '--help')
    assert r.returncode == 0
    assert 'Available applets' in r.stdout
