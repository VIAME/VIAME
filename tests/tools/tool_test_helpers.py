"""Run applet regressions against an install or an explicitly selected build."""
import os
from pathlib import Path

import pytest

from .test_viame_applets import find_viame_install, get_sourced_env, run_viame


@pytest.fixture
def tool_env():
    install = find_viame_install()
    if install is None:
        pytest.skip("No VIAME install found")
    env = get_sourced_env(install)
    build = os.environ.get('VIAME_TEST_BUILD')
    if build:
        root = Path(build).resolve()
        env['PATH'] = str(root / 'bin') + os.pathsep + env['PATH']
        env['KWIVER_PLUGIN_PATH'] = str(root / 'lib/viame/applets') + os.pathsep + env.get('KWIVER_PLUGIN_PATH', '')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env
