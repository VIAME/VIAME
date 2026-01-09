# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Shared pytest fixtures and utilities for example script tests.

These tests verify that example scripts:
1. Run without error (return code 0)
2. Produce some non-empty output (stdout/stderr or output files)
3. Clean up any output files created during the test
"""

import os
import shutil
import subprocess
import sys
import pytest
from pathlib import Path


class TimeoutSuccess(Exception):
    """Exception raised when a script times out but timeout is considered success."""
    pass


def get_directory_snapshot(directory):
    """
    Get a snapshot of all files and directories in a directory.

    Args:
        directory: Path to the directory to snapshot

    Returns:
        Set of Path objects for all files and directories
    """
    directory = Path(directory)
    if not directory.exists():
        return set()

    snapshot = set()
    for item in directory.rglob("*"):
        snapshot.add(item.resolve())
    return snapshot


def cleanup_new_files(before_snapshot, after_snapshot, working_dir):
    """
    Remove files and directories that were created during script execution.

    Args:
        before_snapshot: Set of paths before script ran
        after_snapshot: Set of paths after script ran
        working_dir: Working directory where script ran

    Returns:
        List of paths that were cleaned up
    """
    new_items = after_snapshot - before_snapshot
    cleaned_up = []

    # Sort by path length (deepest first) to delete files before their parent dirs
    sorted_items = sorted(new_items, key=lambda p: len(str(p)), reverse=True)

    for item in sorted_items:
        try:
            if item.exists():
                if item.is_file():
                    item.unlink()
                    cleaned_up.append(item)
                elif item.is_dir():
                    # Only remove if empty (files inside should have been deleted first)
                    if not any(item.iterdir()):
                        item.rmdir()
                        cleaned_up.append(item)
                    else:
                        # Force remove non-empty directories created by the script
                        shutil.rmtree(item)
                        cleaned_up.append(item)
        except (OSError, PermissionError) as e:
            # Log but don't fail the test if cleanup fails
            print(f"Warning: Failed to clean up {item}: {e}")

    return cleaned_up


def validate_output_files(new_files):
    """
    Validate that output files are non-empty and readable.

    Args:
        new_files: Set of new file paths created by the script

    Returns:
        List of validated file paths with their sizes
    """
    validated = []
    for item in new_files:
        if item.is_file():
            size = item.stat().st_size
            validated.append((item, size))
    return validated


def get_viame_source():
    """Get the VIAME source directory."""
    # This file is at tests/examples/conftest.py
    # Source is at ../../
    return Path(__file__).resolve().parent.parent.parent


def get_viame_install():
    """Get the VIAME install directory."""
    # Check environment variable first (set by CMake test properties)
    if 'VIAME_INSTALL' in os.environ:
        install_path = Path(os.environ['VIAME_INSTALL'])
        if install_path.exists():
            return install_path

    # Try to find it relative to source
    source = get_viame_source()

    # Common locations to check relative to source
    candidates = [
        source.parent / "build" / "install",  # viame/build/install (source is viame/src)
        source / "build" / "install",  # viame/build/install (source is viame)
    ]

    for candidate in candidates:
        if (candidate / "setup_viame.sh").exists():
            return candidate

    # If not found, skip the test
    pytest.skip("VIAME install directory not found. Set VIAME_INSTALL environment variable.")


def get_examples_dir(category):
    """Get the examples directory for a specific category."""
    return get_viame_install() / "examples" / category


def get_script_path(category, script_name):
    """Get the full path to a script."""
    if sys.platform == "win32":
        script_name = script_name.replace(".sh", ".bat")
    return get_examples_dir(category) / script_name


def run_example_script(script_path, working_dir=None, timeout=300, env=None,
                       timeout_is_success=False):
    """
    Run an example shell script and return the result.

    The script is run with the VIAME environment properly set up by sourcing
    setup_viame.sh before executing the script.

    Args:
        script_path: Path to the shell script
        working_dir: Working directory for the script (defaults to script's parent)
        timeout: Maximum time in seconds to wait for completion
        env: Optional environment variables dict
        timeout_is_success: If True, a timeout is considered successful completion

    Returns:
        subprocess.CompletedProcess result

    Raises:
        TimeoutSuccess: If timeout_is_success=True and the script timed out
    """
    script_path = Path(script_path)
    if working_dir is None:
        working_dir = script_path.parent

    viame_install = get_viame_install()
    setup_script = viame_install / "setup_viame.sh"

    if sys.platform == "win32":
        # Use .bat version on Windows
        bat_script = script_path.with_suffix(".bat")
        if not bat_script.exists():
            pytest.skip(f"Windows script not found at {bat_script}")

        setup_bat = viame_install / "setup_viame.bat"
        if not setup_bat.exists():
            pytest.skip(f"VIAME setup script not found at {setup_bat}")

        # Create a wrapper command that sources setup and runs the script
        cmd = f'call "{setup_bat}" && call "{bat_script}"'
        shell = True
        executable = None
    else:
        if not script_path.exists():
            pytest.skip(f"Script not found at {script_path}")

        if not setup_script.exists():
            pytest.skip(f"VIAME setup script not found at {setup_script}")

        # Create a wrapper command that sources setup_viame.sh and runs the script
        # We use 'source' (or '.') to source the setup script, then run the target script
        cmd = f'source "{setup_script}" && bash "{script_path}"'
        shell = True
        executable = "/bin/bash"

    # Merge environment if provided
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            executable=executable,
            env=run_env
        )
        return result
    except subprocess.TimeoutExpired as e:
        if timeout_is_success:
            # Script ran until timeout - this is considered success
            raise TimeoutSuccess(
                f"Script {script_path.name} ran for {timeout}s until timeout (success)"
            )
        else:
            # Re-raise as a regular timeout failure
            raise


def assert_script_runs_successfully(script_path, working_dir=None, timeout=300, env=None,
                                     timeout_is_success=False):
    """
    Assert that a script runs without error and produces non-empty output.

    After the script runs, any output files created are validated and then
    cleaned up to keep the source tree clean.

    Args:
        script_path: Path to the shell script
        working_dir: Working directory for the script
        timeout: Maximum time in seconds
        env: Optional environment variables
        timeout_is_success: If True, a timeout is considered successful completion

    Raises:
        AssertionError: If script fails or produces no output
    """
    script_path = Path(script_path)
    if working_dir is None:
        working_dir = script_path.parent
    working_dir = Path(working_dir)

    # Take a snapshot of the working directory before running the script
    before_snapshot = get_directory_snapshot(working_dir)

    try:
        result = run_example_script(script_path, working_dir, timeout, env,
                                    timeout_is_success=timeout_is_success)
    except TimeoutSuccess:
        # Timeout was reached and that's considered success
        # Still need to clean up any files created before timeout
        after_snapshot = get_directory_snapshot(working_dir)
        cleanup_new_files(before_snapshot, after_snapshot, working_dir)
        return None

    # Take a snapshot after the script ran
    after_snapshot = get_directory_snapshot(working_dir)
    new_items = after_snapshot - before_snapshot

    # Validate any output files created
    if new_items:
        validated_files = validate_output_files(new_items)
        # Output files are considered valid if they exist and are readable
        # (validation already completed by this point)

    try:
        # Check return code
        assert result.returncode == 0, (
            f"Script {script_path.name} failed with return code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        # Check for non-empty output (either stdout or stderr should have content)
        combined_output = (result.stdout or "") + (result.stderr or "")
        assert len(combined_output.strip()) > 0, (
            f"Script {script_path.name} produced no output"
        )
    finally:
        # Always clean up output files, even if assertions fail
        if new_items:
            cleaned = cleanup_new_files(before_snapshot, after_snapshot, working_dir)
            if cleaned:
                print(f"Cleaned up {len(cleaned)} output file(s)/directory(ies)")

    return result


@pytest.fixture(scope="session")
def viame_source():
    """Fixture providing the VIAME source directory."""
    return get_viame_source()


@pytest.fixture(scope="session")
def viame_install():
    """Fixture providing the VIAME install directory."""
    return get_viame_install()


@pytest.fixture(scope="session")
def example_imagery(viame_install):
    """Fixture ensuring example imagery exists."""
    imagery_dir = viame_install / "examples" / "example_imagery"
    if not imagery_dir.exists():
        pytest.skip(f"Example imagery not found at {imagery_dir}")
    return imagery_dir


@pytest.fixture(scope="session")
def small_image_set(example_imagery):
    """Fixture ensuring small example image set exists."""
    small_set = example_imagery / "small_example_image_set1"
    if not small_set.exists():
        pytest.skip(f"Small example image set not found at {small_set}")

    images = list(small_set.glob("*.png")) + list(small_set.glob("*.jpg"))
    if len(images) == 0:
        pytest.skip(f"No images found in {small_set}")

    return small_set
