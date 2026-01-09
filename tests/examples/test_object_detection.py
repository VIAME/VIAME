# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for object_detection example scripts.
"""

import pytest
from conftest import get_script_path, assert_script_runs_successfully

CATEGORY = "object_detection"


class TestGenericProposalsExample:
    """Tests for run_generic_proposals script."""

    def test_run_generic_proposals(self):
        """Test that run_generic_proposals runs without error and produces output."""
        script = get_script_path(CATEGORY, "run_generic_proposals.sh")
        assert_script_runs_successfully(script)


class TestFishDetectorExample:
    """Tests for run_fish_without_motion script."""

    def test_run_fish_without_motion(self):
        """Test that run_fish_without_motion runs without error and produces output."""
        script = get_script_path(CATEGORY, "run_fish_without_motion.sh")
        assert_script_runs_successfully(script)
