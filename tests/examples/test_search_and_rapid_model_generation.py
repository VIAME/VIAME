# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for search_and_rapid_model_generation example scripts.
"""

import pytest
from conftest import get_script_path, assert_script_runs_successfully

CATEGORY = "search_and_rapid_model_generation"


class TestCreateIndexAroundDetections:
    """Tests for create_index.around_detections script."""

    def test_create_index_around_detections(self):
        """Test that create_index.around_detections runs without error and produces output."""
        script = get_script_path(CATEGORY, "create_index.around_detections.sh")
        assert_script_runs_successfully(script)


class TestCreateIndexAroundExistingDetections:
    """Tests for create_index.around_existing_detections script."""

    def test_create_index_around_existing_detections(self):
        """Test that create_index.around_existing_detections runs without error and produces output."""
        script = get_script_path(CATEGORY, "create_index.around_existing_detections.sh")
        assert_script_runs_successfully(script)


class TestCreateIndexDetectionAndTracking:
    """Tests for create_index.detection_and_tracking script."""

    def test_create_index_detection_and_tracking(self):
        """Test that create_index.detection_and_tracking runs without error and produces output."""
        script = get_script_path(CATEGORY, "create_index.detection_and_tracking.sh")
        assert_script_runs_successfully(script)


class TestCreateIndexFullFrameOnly:
    """Tests for create_index.full_frame_only script."""

    def test_create_index_full_frame_only(self):
        """Test that create_index.full_frame_only runs without error and produces output."""
        script = get_script_path(CATEGORY, "create_index.full_frame_only.sh")
        assert_script_runs_successfully(script)


class TestGenerateDetectionsUsingSvmModel:
    """Tests for generate_detections_using_svm_model script."""

    def test_generate_detections_using_svm_model(self):
        """Test that generate_detections_using_svm_model runs without error and produces output."""
        script = get_script_path(CATEGORY, "generate_detections_using_svm_model.sh")
        assert_script_runs_successfully(script)


class TestPerformCliQuery:
    """Tests for perform_cli_query script."""

    def test_perform_cli_query(self):
        """Test that perform_cli_query runs without error and produces output."""
        script = get_script_path(CATEGORY, "perform_cli_query.sh")
        assert_script_runs_successfully(script)


class TestProcessDatabaseUsingSvmModel:
    """Tests for process_database_using_svm_model script."""

    def test_process_database_using_svm_model(self):
        """Test that process_database_using_svm_model runs without error and produces output."""
        script = get_script_path(CATEGORY, "process_database_using_svm_model.sh")
        assert_script_runs_successfully(script)


class TestProcessFullFramesUsingSvmModel:
    """Tests for process_full_frames_using_svm_model script."""

    def test_process_full_frames_using_svm_model(self):
        """Test that process_full_frames_using_svm_model runs without error and produces output."""
        script = get_script_path(CATEGORY, "process_full_frames_using_svm_model.sh")
        assert_script_runs_successfully(script)
