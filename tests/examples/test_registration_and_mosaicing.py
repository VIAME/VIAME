# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for registration_and_mosaicing example scripts.
"""

import pytest
from conftest import get_script_path, assert_script_runs_successfully

CATEGORY = "registration_and_mosaicing"


class TestGenerateMosaicForList:
    """Tests for generate_mosaic_for_list script."""

    def test_generate_mosaic_for_list(self):
        """Test that generate_mosaic_for_list runs without error and produces output."""
        script = get_script_path(CATEGORY, "generate_mosaic_for_list.sh")
        assert_script_runs_successfully(script)


class TestGenerateMosaicsForFolder:
    """Tests for generate_mosaics_for_folder script."""

    def test_generate_mosaics_for_folder(self):
        """Test that generate_mosaics_for_folder runs without error and produces output."""
        script = get_script_path(CATEGORY, "generate_mosaics_for_folder.sh")
        assert_script_runs_successfully(script)


class TestGenerateMosaicsXcameraOnly:
    """Tests for generate_mosaics_xcamera_only script."""

    def test_generate_mosaics_xcamera_only(self):
        """Test that generate_mosaics_xcamera_only runs without error and produces output."""
        script = get_script_path(CATEGORY, "generate_mosaics_xcamera_only.sh")
        assert_script_runs_successfully(script)


class TestGenerateTransformFromPoints:
    """Tests for generate_transform_from_points script."""

    def test_generate_transform_from_points(self):
        """Test that generate_transform_from_points runs without error and produces output."""
        script = get_script_path(CATEGORY, "generate_transform_from_points.sh")
        assert_script_runs_successfully(script)


class TestRegisterEoIrPerFrameItk:
    """Tests for register_eo_ir_per_frame_itk script."""

    def test_register_eo_ir_per_frame_itk(self):
        """Test that register_eo_ir_per_frame_itk runs without error and produces output."""
        script = get_script_path(CATEGORY, "register_eo_ir_per_frame_itk.sh")
        assert_script_runs_successfully(script)


class TestRegisterEoIrPerFrameOcv:
    """Tests for register_eo_ir_per_frame_ocv script."""

    def test_register_eo_ir_per_frame_ocv(self):
        """Test that register_eo_ir_per_frame_ocv runs without error and produces output."""
        script = get_script_path(CATEGORY, "register_eo_ir_per_frame_ocv.sh")
        assert_script_runs_successfully(script)
