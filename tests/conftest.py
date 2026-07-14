# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Pytest configuration for VIAME tests.

Tests run against the installed VIAME modules (via PYTHONPATH set by ctest).
No source tree path manipulation is needed since installed modules are used.
"""

import sys
from pathlib import Path

import pytest

# Make the shared helpers importable regardless of which subtree's ctest
# invocation set PYTHONPATH (this conftest is loaded for every tests/ subtree).
sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

from viame_env import find_viame_install, get_viame_source


@pytest.fixture(scope="session")
def viame_source() -> Path:
    """The VIAME source directory."""
    return get_viame_source()


@pytest.fixture(scope="session")
def viame_install() -> Path:
    """The VIAME install directory, skipping the test if it cannot be found."""
    install = find_viame_install()
    if install is None:
        pytest.skip("VIAME install directory not found. Set VIAME_INSTALL environment variable.")
    return install
