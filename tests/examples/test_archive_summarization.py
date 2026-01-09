# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for archive_summarization example scripts.
"""

import pytest
from test_utilities import get_script_path, assert_script_runs_successfully

CATEGORY = "archive_summarization"


class TestSummarizeAndIndexVideos:
    """Tests for summarize_and_index_videos script."""

    def test_summarize_and_index_videos(self):
        """Test that summarize_and_index_videos runs without error and produces output."""
        script = get_script_path(CATEGORY, "summarize_and_index_videos.sh")
        assert_script_runs_successfully(script)


class TestSummarizeVideos:
    """Tests for summarize_videos script."""

    def test_summarize_videos(self):
        """Test that summarize_videos runs without error and produces output."""
        script = get_script_path(CATEGORY, "summarize_videos.sh")
        assert_script_runs_successfully(script)
