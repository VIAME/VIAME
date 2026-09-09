# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for detection_file_conversions example scripts.
"""

import pytest
from test_utilities import get_script_path, assert_script_runs_successfully

CATEGORY = "detection_file_conversions"


class TestBulkConvert:
    """Tests for the bulk_convert script (imagery found alongside)."""

    def test_bulk_convert(self):
        """Test that bulk_convert runs without error and produces output."""
        script = get_script_path(CATEGORY, "bulk_convert.sh")
        assert_script_runs_successfully(script)


class TestBulkConvertGtOnly:
    """Tests for bulk_convert_gt_only script."""

    def test_bulk_convert_gt_only(self):
        """Test that bulk_convert_gt_only runs without error and produces output."""
        script = get_script_path(CATEGORY, "bulk_convert_gt_only.sh")
        assert_script_runs_successfully(script)
