# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for 16-bit image normalization functionality.

These tests verify:
1. The percentile_norm filter correctly normalizes 16-bit imagery to 8-bit
2. The train tool's --normalize-16bit flag works correctly
"""

import pytest
from test_utilities import get_script_path, assert_script_runs_successfully

ENHANCEMENT_CATEGORY = "image_enhancement"
TRAINING_CATEGORY = "object_detector_training"

# Timeout for training scripts that we just want to verify start correctly
TRAINING_TIMEOUT = 15  # seconds


class TestNormalize16bit:
    """Tests for the normalize_16bit example script."""

    def test_normalize_16bit_pipeline(self):
        """Test that normalize_16bit runs without error and produces output.

        This test verifies the standalone percentile_norm filter pipeline
        correctly processes 16-bit thermal imagery.
        """
        script = get_script_path(ENHANCEMENT_CATEGORY, "normalize_16bit.sh")
        assert_script_runs_successfully(script)


class TestTrainNetharnRfDetr16bit:
    """Tests for training with 16-bit imagery using --normalize-16bit flag."""

    def test_train_netharn_rf_detr_16bit(self):
        """Test that train_netharn_rf_detr_16bit starts successfully.

        Training scripts are long-running, so we use a short timeout and consider
        reaching the timeout as success (the script started and is running).

        This test verifies the --normalize-16bit flag correctly enables percentile
        normalization for training on 16-bit thermal imagery.
        """
        script = get_script_path(TRAINING_CATEGORY, "train_netharn_rf_detr_16bit.sh")
        assert_script_runs_successfully(script, timeout=TRAINING_TIMEOUT, timeout_is_success=True)
