# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for detection_file_conversions example scripts.
"""

import pytest
from test_utilities import get_script_path, assert_script_runs_successfully

CATEGORY = "detection_file_conversions"


class TestBulkConvertGtPlusData:
    """Tests for bulk_convert_gt_plus_data script."""

    def test_bulk_convert_gt_plus_data(self):
        """Test that bulk_convert_gt_plus_data runs without error and produces output."""
        script = get_script_path(CATEGORY, "bulk_convert_gt_plus_data.sh")
        assert_script_runs_successfully(script)


class TestBulkConvertGtOnly:
    """Tests for bulk_convert_gt_only script."""

    def test_bulk_convert_gt_only(self):
        """Test that bulk_convert_gt_only runs without error and produces output."""
        script = get_script_path(CATEGORY, "bulk_convert_gt_only.sh")
        assert_script_runs_successfully(script)
