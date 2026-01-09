# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for image_enhancement example scripts.
"""

import pytest
from conftest import get_script_path, assert_script_runs_successfully

CATEGORY = "image_enhancement"


class TestDebayerAndEnhance:
    """Tests for debayer_and_enhance script."""

    def test_debayer_and_enhance(self):
        """Test that debayer_and_enhance runs without error and produces output."""
        script = get_script_path(CATEGORY, "debayer_and_enhance.sh")
        assert_script_runs_successfully(script)


class TestEnhance:
    """Tests for enhance script."""

    def test_enhance(self):
        """Test that enhance runs without error and produces output."""
        script = get_script_path(CATEGORY, "enhance.sh")
        assert_script_runs_successfully(script)
