# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for frame_level_classification example scripts.
"""

import pytest
from test_utilities import get_script_path, assert_script_runs_successfully

CATEGORY = "frame_level_classification"


class TestContinueTrainingDeepClassifier:
    """Tests for continue_training_deep_classifier script."""

    def test_continue_training_deep_classifier(self):
        """Test that continue_training_deep_classifier runs without error and produces output."""
        script = get_script_path(CATEGORY, "continue_training_deep_classifier.sh")
        assert_script_runs_successfully(script)


class TestRunTrainedModel:
    """Tests for run_trained_model script."""

    def test_run_trained_model(self):
        """Test that run_trained_model runs without error and produces output."""
        script = get_script_path(CATEGORY, "run_trained_model.sh")
        assert_script_runs_successfully(script)


class TestTrainDeepFrameClassifier:
    """Tests for train_deep_frame_classifier script."""

    def test_train_deep_frame_classifier(self):
        """Test that train_deep_frame_classifier runs without error and produces output."""
        script = get_script_path(CATEGORY, "train_deep_frame_classifier.sh")
        assert_script_runs_successfully(script)


class TestTrainSvmFrameClassifier:
    """Tests for train_svm_frame_classifier script."""

    def test_train_svm_frame_classifier(self):
        """Test that train_svm_frame_classifier runs without error and produces output."""
        script = get_script_path(CATEGORY, "train_svm_frame_classifier.sh")
        assert_script_runs_successfully(script)
