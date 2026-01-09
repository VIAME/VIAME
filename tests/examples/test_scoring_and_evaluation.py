# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for scoring_and_evaluation example scripts.
"""

import pytest
from conftest import get_script_path, assert_script_runs_successfully

CATEGORY = "scoring_and_evaluation"


class TestDetectionPrcsAndConfMatAcrossAll:
    """Tests for detection_prcs_and_conf_mat_across_all script."""

    def test_detection_prcs_and_conf_mat_across_all(self):
        """Test that detection_prcs_and_conf_mat_across_all runs without error and produces output."""
        script = get_script_path(CATEGORY, "detection_prcs_and_conf_mat_across_all.sh")
        assert_script_runs_successfully(script)


class TestDetectionPrcsAndConfMatPerCategory:
    """Tests for detection_prcs_and_conf_mat_per_category script."""

    def test_detection_prcs_and_conf_mat_per_category(self):
        """Test that detection_prcs_and_conf_mat_per_category runs without error and produces output."""
        script = get_script_path(CATEGORY, "detection_prcs_and_conf_mat_per_category.sh")
        assert_script_runs_successfully(script)


class TestDetectionRocsAcrossAll:
    """Tests for detection_rocs_across_all script."""

    def test_detection_rocs_across_all(self):
        """Test that detection_rocs_across_all runs without error and produces output."""
        script = get_script_path(CATEGORY, "detection_rocs_across_all.sh")
        assert_script_runs_successfully(script)


class TestDetectionRocsPerCategory:
    """Tests for detection_rocs_per_category script."""

    def test_detection_rocs_per_category(self):
        """Test that detection_rocs_per_category runs without error and produces output."""
        script = get_script_path(CATEGORY, "detection_rocs_per_category.sh")
        assert_script_runs_successfully(script)


class TestTrackKwantStatsAcrossAll:
    """Tests for track_kwant_stats_across_all script."""

    def test_track_kwant_stats_across_all(self):
        """Test that track_kwant_stats_across_all runs without error and produces output."""
        script = get_script_path(CATEGORY, "track_kwant_stats_across_all.sh")
        assert_script_runs_successfully(script)


class TestTrackKwantStatsPerCategory:
    """Tests for track_kwant_stats_per_category script."""

    def test_track_kwant_stats_per_category(self):
        """Test that track_kwant_stats_per_category runs without error and produces output."""
        script = get_script_path(CATEGORY, "track_kwant_stats_per_category.sh")
        assert_script_runs_successfully(script)


class TestTrackMotStatsAcrossAll:
    """Tests for track_mot_stats_across_all script."""

    def test_track_mot_stats_across_all(self):
        """Test that track_mot_stats_across_all runs without error and produces output."""
        script = get_script_path(CATEGORY, "track_mot_stats_across_all.sh")
        assert_script_runs_successfully(script)


class TestTrackMotStatsPerCategory:
    """Tests for track_mot_stats_per_category script."""

    def test_track_mot_stats_per_category(self):
        """Test that track_mot_stats_per_category runs without error and produces output."""
        script = get_script_path(CATEGORY, "track_mot_stats_per_category.sh")
        assert_script_runs_successfully(script)
