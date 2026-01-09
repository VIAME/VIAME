# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for size_measurement example scripts.
"""

import pytest
from test_utilities import get_script_path, assert_script_runs_successfully

CATEGORY = "size_measurement"


class TestCalibrateCameras:
    """Tests for calibrate_cameras script."""

    def test_calibrate_cameras(self):
        """Test that calibrate_cameras runs without error and produces output."""
        script = get_script_path(CATEGORY, "calibrate_cameras.sh")
        assert_script_runs_successfully(script)


class TestComputeDepthMaps:
    """Tests for compute_depth_maps script."""

    def test_compute_depth_maps(self):
        """Test that compute_depth_maps runs without error and produces output."""
        script = get_script_path(CATEGORY, "compute_depth_maps.sh")
        assert_script_runs_successfully(script)


class TestGmmStandaloneTool:
    """Tests for gmm_standalone_tool script."""

    def test_gmm_standalone_tool(self):
        """Test that gmm_standalone_tool runs without error and produces output."""
        script = get_script_path(CATEGORY, "gmm_standalone_tool.sh")
        assert_script_runs_successfully(script)


class TestMeasureOverManualAnnotations:
    """Tests for measure_over_manual_annotations script."""

    def test_measure_over_manual_annotations(self):
        """Test that measure_over_manual_annotations runs without error and produces output."""
        script = get_script_path(CATEGORY, "measure_over_manual_annotations.sh")
        assert_script_runs_successfully(script)


class TestMeasureViaDefaultFish:
    """Tests for measure_via_default_fish script."""

    def test_measure_via_default_fish(self):
        """Test that measure_via_default_fish runs without error and produces output."""
        script = get_script_path(CATEGORY, "measure_via_default_fish.sh")
        assert_script_runs_successfully(script)


class TestMeasureViaGmmOrientedBoxes:
    """Tests for measure_via_gmm_oriented_boxes script."""

    def test_measure_via_gmm_oriented_boxes(self):
        """Test that measure_via_gmm_oriented_boxes runs without error and produces output."""
        script = get_script_path(CATEGORY, "measure_via_gmm_oriented_boxes.sh")
        assert_script_runs_successfully(script)
