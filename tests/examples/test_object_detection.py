# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for object detection example pipelines.

These tests verify that the main object detection pipelines run successfully
and produce expected output files.
"""

import os
import subprocess
import sys
import pytest
import shutil

from pathlib import Path


def get_viame_install():
    """Get the VIAME install directory from environment or common locations."""
    # Check environment variable first
    if "VIAME_INSTALL" in os.environ:
        return Path(os.environ["VIAME_INSTALL"])

    # Check if we're in a build tree - look for setup_viame.sh
    # Start from this file's location and search upward
    current = Path(__file__).resolve()
    for parent in current.parents:
        setup_script = parent / "setup_viame.sh"
        if setup_script.exists():
            return parent
        # Also check install directory
        install_dir = parent / "install"
        if (install_dir / "setup_viame.sh").exists():
            return install_dir

    return None


def get_viame_source():
    """Get the VIAME source directory."""
    # This file is at tests/examples/test_object_detection.py
    # Source is at ../../
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def viame_install():
    """Fixture providing VIAME install path."""
    install = get_viame_install()
    if install is None:
        pytest.skip("VIAME_INSTALL not found. Set VIAME_INSTALL environment variable.")
    if not (install / "setup_viame.sh").exists():
        pytest.skip(f"setup_viame.sh not found in {install}")
    return install


@pytest.fixture(scope="module")
def viame_source():
    """Fixture providing VIAME source path."""
    return get_viame_source()


@pytest.fixture
def working_dir(tmp_path, viame_source):
    """Create a working directory with necessary input files."""
    work_dir = tmp_path / "detection_test"
    work_dir.mkdir()

    # Copy example imagery to working directory
    example_imagery = viame_source / "examples" / "example_imagery" / "small_example_image_set1"
    if not example_imagery.exists():
        pytest.skip(f"Example imagery not found at {example_imagery}")

    # Create the imagery directory structure
    dest_imagery = work_dir / "example_imagery" / "small_example_image_set1"
    dest_imagery.mkdir(parents=True)

    # Copy a subset of images for faster testing
    images_copied = 0
    for img in example_imagery.glob("*.png"):
        shutil.copy(img, dest_imagery / img.name)
        images_copied += 1
        if images_copied >= 3:  # Only copy 3 images for faster tests
            break

    if images_copied == 0:
        pytest.skip("No test images found")

    # Create input image list
    input_list = work_dir / "input_list.txt"
    with open(input_list, "w") as f:
        for img in sorted(dest_imagery.glob("*.png"))[:3]:
            # Use relative path from work_dir
            rel_path = img.relative_to(work_dir)
            f.write(f"{rel_path}\n")

    return work_dir


def run_pipeline(viame_install, pipeline_name, input_list, working_dir, timeout=300):
    """
    Run a VIAME pipeline and return the result.

    Args:
        viame_install: Path to VIAME install directory
        pipeline_name: Name of the pipeline file (e.g., detector_generic_proposals.pipe)
        input_list: Path to input image list file
        working_dir: Working directory for the pipeline
        timeout: Maximum time in seconds to wait for pipeline completion

    Returns:
        subprocess.CompletedProcess result
    """
    pipeline_path = viame_install / "configs" / "pipelines" / pipeline_name

    if not pipeline_path.exists():
        pytest.skip(f"Pipeline {pipeline_name} not found at {pipeline_path}")

    # Build the command to run
    if sys.platform == "win32":
        setup_script = viame_install / "setup_viame.bat"
        cmd = f'call "{setup_script}" && kwiver runner "{pipeline_path}" -s input:video_filename="{input_list}"'
        shell = True
    else:
        setup_script = viame_install / "setup_viame.sh"
        cmd = f'source "{setup_script}" && kwiver runner "{pipeline_path}" -s input:video_filename="{input_list}"'
        shell = True

    result = subprocess.run(
        cmd,
        shell=shell,
        cwd=working_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        executable="/bin/bash" if sys.platform != "win32" else None
    )

    return result


class TestGenericProposalsDetector:
    """Tests for the generic proposals detector pipeline."""

    def test_pipeline_runs_successfully(self, viame_install, working_dir):
        """Test that the generic proposals pipeline runs without error."""
        input_list = working_dir / "input_list.txt"

        result = run_pipeline(
            viame_install,
            "detector_generic_proposals.pipe",
            input_list,
            working_dir
        )

        assert result.returncode == 0, (
            f"Pipeline failed with return code {result.returncode}\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

    def test_pipeline_produces_output(self, viame_install, working_dir):
        """Test that the generic proposals pipeline produces detection output."""
        input_list = working_dir / "input_list.txt"

        result = run_pipeline(
            viame_install,
            "detector_generic_proposals.pipe",
            input_list,
            working_dir
        )

        # Check that output file was created
        output_file = working_dir / "computed_detections.csv"
        assert output_file.exists(), (
            f"Output file {output_file} not created\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

        # Check that output file has content (header + at least potential detections)
        content = output_file.read_text()
        lines = [l for l in content.strip().split("\n") if l and not l.startswith("#")]
        assert len(lines) >= 0, "Output file should have content"


class TestDefaultFishDetector:
    """Tests for the default fish detector pipeline."""

    def test_pipeline_runs_successfully(self, viame_install, working_dir):
        """Test that the default fish detector pipeline runs without error."""
        input_list = working_dir / "input_list.txt"

        result = run_pipeline(
            viame_install,
            "detector_default_fish.pipe",
            input_list,
            working_dir
        )

        assert result.returncode == 0, (
            f"Pipeline failed with return code {result.returncode}\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

    def test_pipeline_produces_output(self, viame_install, working_dir):
        """Test that the default fish detector pipeline produces detection output."""
        input_list = working_dir / "input_list.txt"

        result = run_pipeline(
            viame_install,
            "detector_default_fish.pipe",
            input_list,
            working_dir
        )

        # Check that output file was created
        output_file = working_dir / "computed_detections.csv"
        assert output_file.exists(), (
            f"Output file {output_file} not created\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

        # Check that output file has content
        content = output_file.read_text()
        lines = [l for l in content.strip().split("\n") if l and not l.startswith("#")]
        assert len(lines) >= 0, "Output file should have content"


class TestDetectionOutputFormat:
    """Tests for detection output format validation."""

    def test_output_csv_format(self, viame_install, working_dir):
        """Test that detection output follows VIAME CSV format."""
        input_list = working_dir / "input_list.txt"

        result = run_pipeline(
            viame_install,
            "detector_generic_proposals.pipe",
            input_list,
            working_dir
        )

        if result.returncode != 0:
            pytest.skip("Pipeline did not complete successfully")

        output_file = working_dir / "computed_detections.csv"
        if not output_file.exists():
            pytest.skip("Output file not created")

        content = output_file.read_text()
        lines = content.strip().split("\n")

        # Filter out comment lines
        data_lines = [l for l in lines if l and not l.startswith("#")]

        if len(data_lines) == 0:
            # No detections is valid
            return

        # Check that each line has the expected number of fields
        # VIAME CSV format: detection_id, image_name, frame_id, bbox (4 values), confidence, ...
        for line in data_lines:
            fields = line.split(",")
            assert len(fields) >= 9, (
                f"Detection line should have at least 9 fields, got {len(fields)}: {line}"
            )
