# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for building_from_source example scripts.
"""

import pytest
from test_utilities import get_script_path, assert_script_runs_successfully

CATEGORY = "building_from_source"


class TestExample:
    """Tests for example script."""

    def test_example(self):
        """Test that example runs without error and produces output."""
        script = get_script_path(CATEGORY, "example.sh")
        assert_script_runs_successfully(script)
