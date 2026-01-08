# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Tests for object detection example scripts.

These tests run the actual example scripts from the examples/object_detection
directory and verify they complete successfully with non-empty output.
"""

import os
import subprocess
import sys
import pytest

from pathlib import Path


def get_viame_source():
    """Get the VIAME source directory."""
    # This file is at tests/examples/test_object_detection.py
    # Source is at ../../
    return Path(__file__).resolve().parent.parent.parent


def get_examples_dir():
    """Get the examples/object_detection directory."""
    return get_viame_source() / "examples" / "object_detection"


@pytest.fixture(scope="module")
def examples_dir():
    """Fixture providing the object_detection examples directory."""
    examples = get_examples_dir()
    if not examples.exists():
        pytest.skip(f"Examples directory not found at {examples}")
    return examples


@pytest.fixture(scope="module")
def input_image_list(examples_dir):
    """Verify input image list exists."""
    input_list = examples_dir / "input_image_list_small_set.txt"
    if not input_list.exists():
        pytest.skip(f"Input image list not found at {input_list}")
    return input_list


@pytest.fixture(scope="module")
def example_imagery(examples_dir):
    """Verify example imagery exists."""
    # The input list references ../example_imagery/small_example_image_set1/
    imagery_dir = examples_dir.parent / "example_imagery" / "small_example_image_set1"
    if not imagery_dir.exists():
        pytest.skip(f"Example imagery not found at {imagery_dir}")

    # Check that there are actual images
    images = list(imagery_dir.glob("*.png"))
    if len(images) == 0:
        pytest.skip(f"No PNG images found in {imagery_dir}")

    return imagery_dir


def run_example_script(script_path, working_dir, timeout=300):
    """
    Run an example shell script and return the result.

    Args:
        script_path: Path to the shell script
        working_dir: Working directory for the script
        timeout: Maximum time in seconds to wait for completion

    Returns:
        subprocess.CompletedProcess result
    """
    if sys.platform == "win32":
        # Use .bat version on Windows
        bat_script = script_path.with_suffix(".bat")
        if not bat_script.exists():
            pytest.skip(f"Windows script not found at {bat_script}")
        cmd = str(bat_script)
        shell = True
        executable = None
    else:
        if not script_path.exists():
            pytest.skip(f"Script not found at {script_path}")
        cmd = f"bash {script_path}"
        shell = True
        executable = "/bin/bash"

    result = subprocess.run(
        cmd,
        shell=shell,
        cwd=working_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        executable=executable
    )

    return result


class TestGenericProposalsExample:
    """Tests for the run_generic_proposals example script."""

    def test_input_exists(self, examples_dir, input_image_list, example_imagery):
        """Verify all required input files exist."""
        script = examples_dir / "run_generic_proposals.sh"
        assert script.exists(), f"Script not found: {script}"
        assert input_image_list.exists(), f"Input list not found: {input_image_list}"
        assert example_imagery.exists(), f"Example imagery not found: {example_imagery}"

    def test_script_runs_successfully(self, examples_dir, input_image_list, example_imagery):
        """Test that run_generic_proposals.sh runs without error."""
        script = examples_dir / "run_generic_proposals.sh"

        result = run_example_script(script, examples_dir)

        assert result.returncode == 0, (
            f"Script failed with return code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_produces_nonempty_output(self, examples_dir, input_image_list, example_imagery):
        """Test that the script produces non-empty detection output."""
        script = examples_dir / "run_generic_proposals.sh"

        result = run_example_script(script, examples_dir)

        if result.returncode != 0:
            pytest.fail(
                f"Script failed with return code {result.returncode}\n"
                f"STDERR:\n{result.stderr}"
            )

        # Check for output file
        output_file = examples_dir / "computed_detections.csv"
        assert output_file.exists(), (
            f"Output file not created: {output_file}\n"
            f"STDOUT:\n{result.stdout}"
        )

        # Check that output is non-empty
        content = output_file.read_text()
        assert len(content.strip()) > 0, "Output file is empty"


class TestFishDetectorExample:
    """Tests for the run_fish_without_motion example script."""

    def test_input_exists(self, examples_dir, input_image_list, example_imagery):
        """Verify all required input files exist."""
        script = examples_dir / "run_fish_without_motion.sh"
        assert script.exists(), f"Script not found: {script}"
        assert input_image_list.exists(), f"Input list not found: {input_image_list}"
        assert example_imagery.exists(), f"Example imagery not found: {example_imagery}"

    def test_script_runs_successfully(self, examples_dir, input_image_list, example_imagery):
        """Test that run_fish_without_motion.sh runs without error."""
        script = examples_dir / "run_fish_without_motion.sh"

        result = run_example_script(script, examples_dir)

        assert result.returncode == 0, (
            f"Script failed with return code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_produces_nonempty_output(self, examples_dir, input_image_list, example_imagery):
        """Test that the script produces non-empty detection output."""
        script = examples_dir / "run_fish_without_motion.sh"

        result = run_example_script(script, examples_dir)

        if result.returncode != 0:
            pytest.fail(
                f"Script failed with return code {result.returncode}\n"
                f"STDERR:\n{result.stderr}"
            )

        # Check for output file
        output_file = examples_dir / "computed_detections.csv"
        assert output_file.exists(), (
            f"Output file not created: {output_file}\n"
            f"STDOUT:\n{result.stdout}"
        )

        # Check that output is non-empty
        content = output_file.read_text()
        assert len(content.strip()) > 0, "Output file is empty"
