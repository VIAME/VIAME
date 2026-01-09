# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for object_tracker_training example scripts.
"""

import pytest
from test_utilities import get_script_path, assert_script_runs_successfully

CATEGORY = "object_tracker_training"


class TestTrainStTrackerViameCsv:
    """Tests for train_st_tracker_viame_csv script."""

    def test_train_st_tracker_viame_csv(self):
        """Test that train_st_tracker_viame_csv runs without error and produces output."""
        script = get_script_path(CATEGORY, "train_st_tracker_viame_csv.sh")
        assert_script_runs_successfully(script)
