# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for hello_world_pipeline example scripts.
"""

import pytest
from test_utilities import get_script_path, assert_script_runs_successfully

CATEGORY = "hello_world_pipeline"


class TestRunExample:
    """Tests for run_example script."""

    def test_run_example(self):
        """Test that run_example runs without error and produces output."""
        script = get_script_path(CATEGORY, "run_example.sh")
        assert_script_runs_successfully(script)


class TestRunPythonExample:
    """Tests for run_python_example script."""

    def test_run_python_example(self):
        """Test that run_python_example runs without error and produces output."""
        script = get_script_path(CATEGORY, "run_python_example.sh")
        assert_script_runs_successfully(script)
