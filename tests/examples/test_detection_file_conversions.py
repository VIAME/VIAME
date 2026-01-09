# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for detection_file_conversions example scripts.
"""

import pytest
from conftest import get_script_path, assert_script_runs_successfully

CATEGORY = "detection_file_conversions"


class TestBulkConvertUsingPipe:
    """Tests for bulk_convert_using_pipe script."""

    def test_bulk_convert_using_pipe(self):
        """Test that bulk_convert_using_pipe runs without error and produces output."""
        script = get_script_path(CATEGORY, "bulk_convert_using_pipe.sh")
        assert_script_runs_successfully(script)
