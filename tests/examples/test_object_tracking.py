# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for object_tracking example scripts.
"""

import pytest
from conftest import get_script_path, assert_script_runs_successfully

CATEGORY = "object_tracking"


class TestBulkRunUserInitTracking:
    """Tests for bulk_run_user_init_tracking script."""

    def test_bulk_run_user_init_tracking(self):
        """Test that bulk_run_user_init_tracking runs without error and produces output."""
        script = get_script_path(CATEGORY, "bulk_run_user_init_tracking.sh")
        assert_script_runs_successfully(script)


class TestComputeDepthMaps:
    """Tests for compute_depth_maps script."""

    def test_compute_depth_maps(self):
        """Test that compute_depth_maps runs without error and produces output."""
        script = get_script_path(CATEGORY, "compute_depth_maps.sh")
        assert_script_runs_successfully(script)


class TestRunFishTracker:
    """Tests for run_fish_tracker script."""

    def test_run_fish_tracker(self):
        """Test that run_fish_tracker runs without error and produces output."""
        script = get_script_path(CATEGORY, "run_fish_tracker.sh")
        assert_script_runs_successfully(script)


class TestRunGenericTracker:
    """Tests for run_generic_tracker script."""

    def test_run_generic_tracker(self):
        """Test that run_generic_tracker runs without error and produces output."""
        script = get_script_path(CATEGORY, "run_generic_tracker.sh")
        assert_script_runs_successfully(script)


class TestRunSimpleTracker:
    """Tests for run_simple_tracker script."""

    def test_run_simple_tracker(self):
        """Test that run_simple_tracker runs without error and produces output."""
        script = get_script_path(CATEGORY, "run_simple_tracker.sh")
        assert_script_runs_successfully(script)


class TestRunTrackViewer:
    """Tests for run_track_viewer script."""

    def test_run_track_viewer(self):
        """Test that run_track_viewer runs without error and produces output."""
        script = get_script_path(CATEGORY, "run_track_viewer.sh")
        assert_script_runs_successfully(script)


class TestRunUserInitTracker:
    """Tests for run_user_init_tracker script."""

    def test_run_user_init_tracker(self):
        """Test that run_user_init_tracker runs without error and produces output."""
        script = get_script_path(CATEGORY, "run_user_init_tracker.sh")
        assert_script_runs_successfully(script)
