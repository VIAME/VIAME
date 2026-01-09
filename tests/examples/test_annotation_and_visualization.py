# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for annotation_and_visualization example scripts.
"""

import pytest
from conftest import get_script_path, assert_script_runs_successfully

CATEGORY = "annotation_and_visualization"


class TestDrawDetectionsOnFrames:
    """Tests for draw_detections_on_frames script."""

    def test_draw_detections_on_frames(self):
        """Test that draw_detections_on_frames runs without error and produces output."""
        script = get_script_path(CATEGORY, "draw_detections_on_frames.sh")
        assert_script_runs_successfully(script)


class TestExtractChipsFromDetections:
    """Tests for extract_chips_from_detections script."""

    def test_extract_chips_from_detections(self):
        """Test that extract_chips_from_detections runs without error and produces output."""
        script = get_script_path(CATEGORY, "extract_chips_from_detections.sh")
        assert_script_runs_successfully(script)


class TestExtractFrames:
    """Tests for extract_frames script."""

    def test_extract_frames(self):
        """Test that extract_frames runs without error and produces output."""
        script = get_script_path(CATEGORY, "extract_frames.sh")
        assert_script_runs_successfully(script)


class TestExtractFramesWithDetsOnly:
    """Tests for extract_frames_with_dets_only script."""

    def test_extract_frames_with_dets_only(self):
        """Test that extract_frames_with_dets_only runs without error and produces output."""
        script = get_script_path(CATEGORY, "extract_frames_with_dets_only.sh")
        assert_script_runs_successfully(script)


class TestSimplePipelineDisplay:
    """Tests for simple_pipeline_display script."""

    def test_simple_pipeline_display(self):
        """Test that simple_pipeline_display runs without error and produces output."""
        script = get_script_path(CATEGORY, "simple_pipeline_display.sh")
        assert_script_runs_successfully(script)
