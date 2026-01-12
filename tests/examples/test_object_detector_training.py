# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for object_detector_training example scripts.
"""

import pytest
from test_utilities import get_script_path, assert_script_runs_successfully

CATEGORY = "object_detector_training"

# Timeout for training scripts that we just want to verify start correctly
TRAINING_TIMEOUT = 15  # seconds


class TestContinueTrainingCfrnn:
    """Tests for continue_training_cfrnn script."""

    def test_continue_training_cfrnn(self):
        """Test that continue_training_cfrnn runs without error and produces output."""
        script = get_script_path(CATEGORY, "continue_training_cfrnn.sh")
        assert_script_runs_successfully(script)


class TestRunTrainedModel:
    """Tests for run_trained_model script."""

    def test_run_trained_model(self):
        """Test that run_trained_model runs without error and produces output."""
        script = get_script_path(CATEGORY, "run_trained_model.sh")
        assert_script_runs_successfully(script)


class TestTrainNetharnCfrnnFromHabcamCsv:
    """Tests for train_netharn_cfrnn_habcam_csv script."""

    def test_train_netharn_cfrnn_from_habcam_csv(self):
        """Test that train_netharn_cfrnn_habcam_csv runs without error and produces output."""
        script = get_script_path(CATEGORY, "train_netharn_cfrnn_habcam_csv.sh")
        assert_script_runs_successfully(script)


class TestTrainNetharnCfrnnFromViameCsv:
    """Tests for train_netharn_cfrnn script."""

    def test_train_netharn_cfrnn_from_viame_csv(self):
        """Test that train_netharn_cfrnn runs without error and produces output."""
        script = get_script_path(CATEGORY, "train_netharn_cfrnn.sh")
        assert_script_runs_successfully(script)


class TestTrainMaskRcnnFromViameCsv:
    """Tests for train_mask_rcnn_from_viame_csv script."""

    def test_train_mask_rcnn_from_viame_csv(self):
        """Test that train_mask_rcnn_from_viame_csv runs without error and produces output."""
        script = get_script_path(CATEGORY, "train_mask_rcnn_from_viame_csv.sh")
        assert_script_runs_successfully(script)


class TestTrainMotionCfrnnFromViameCsv:
    """Tests for train_motion_cfrnn_from_viame_csv script."""

    def test_train_motion_cfrnn_from_viame_csv(self):
        """Test that train_motion_cfrnn_from_viame_csv runs without error and produces output."""
        script = get_script_path(CATEGORY, "train_motion_cfrnn_from_viame_csv.sh")
        assert_script_runs_successfully(script)


class TestTrainSvmOverFishDetsFromViameCsv:
    """Tests for train_svm_over_fish_dets_from_viame_csv script."""

    def test_train_svm_over_fish_dets_from_viame_csv(self):
        """Test that train_svm_over_fish_dets_from_viame_csv runs without error and produces output."""
        script = get_script_path(CATEGORY, "train_svm_over_fish_dets_from_viame_csv.sh")
        assert_script_runs_successfully(script)


class TestTrainSvmOverGenericDetsFromViameCsv:
    """Tests for train_svm_over_generic_dets_from_viame_csv script."""

    def test_train_svm_over_generic_dets_from_viame_csv(self):
        """Test that train_svm_over_generic_dets_from_viame_csv runs without error and produces output."""
        script = get_script_path(CATEGORY, "train_svm_over_generic_dets_from_viame_csv.sh")
        assert_script_runs_successfully(script)


class TestTrainYoloFromHabcamCsv:
    """Tests for train_yolo_from_habcam_csv script."""

    def test_train_yolo_from_habcam_csv(self):
        """Test that train_yolo_from_habcam_csv starts successfully.

        Training scripts are long-running, so we use a short timeout and consider
        reaching the timeout as success (the script started and is running).
        """
        script = get_script_path(CATEGORY, "train_yolo_from_habcam_csv.sh")
        assert_script_runs_successfully(script, timeout=TRAINING_TIMEOUT, timeout_is_success=True)


class TestTrainYoloFromKw18:
    """Tests for train_yolo_from_kw18 script."""

    def test_train_yolo_from_kw18(self):
        """Test that train_yolo_from_kw18 starts successfully.

        Training scripts are long-running, so we use a short timeout and consider
        reaching the timeout as success (the script started and is running).
        """
        script = get_script_path(CATEGORY, "train_yolo_from_kw18.sh")
        assert_script_runs_successfully(script, timeout=TRAINING_TIMEOUT, timeout_is_success=True)


class TestTrainYoloFromViameCsv:
    """Tests for train_yolo_from_viame_csv script."""

    def test_train_yolo_from_viame_csv(self):
        """Test that train_yolo_from_viame_csv starts successfully.

        Training scripts are long-running, so we use a short timeout and consider
        reaching the timeout as success (the script started and is running).
        """
        script = get_script_path(CATEGORY, "train_yolo_from_viame_csv.sh")
        assert_script_runs_successfully(script, timeout=TRAINING_TIMEOUT, timeout_is_success=True)
