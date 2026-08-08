# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Shared helpers for locating the VIAME install and sourcing its environment.

Used by both the example-script tests (tests/examples) and the pipeline
tests (tests/pipelines) so the install-discovery / setup-sourcing logic
lives in exactly one place.
"""

import os
import subprocess
from pathlib import Path


def get_viame_source() -> Path:
    """Return the VIAME source directory (the dir containing tests/)."""
    # tests/common/viame_env.py -> parents[2] == source root
    return Path(__file__).resolve().parents[2]


def find_viame_install() -> Path | None:
    """Locate the VIAME install directory, or return None if not found.

    Checks $VIAME_INSTALL first (set by the ctest ENVIRONMENT property),
    then the common build/install locations relative to the source tree.
    """
    env_install = os.environ.get("VIAME_INSTALL")
    if env_install:
        install_path = Path(env_install)
        if install_path.exists():
            return install_path

    source = get_viame_source()
    candidates = [
        source.parent / "build" / "install",
        source / "build" / "install",
    ]
    for candidate in candidates:
        if (candidate / "setup_viame.sh").exists():
            return candidate

    return None


def setup_script(install: Path) -> Path:
    """Return the platform setup script inside an install directory."""
    return install / ("setup_viame.bat" if os.name == "nt" else "setup_viame.sh")


def get_sourced_env(install: Path | None = None) -> dict:
    """Return os.environ updated with the VIAME setup environment.

    Sources setup_viame.{sh,bat} in a subshell and captures the resulting
    environment so tools (e.g. kwiver) can be invoked directly.
    """
    if install is None:
        install = find_viame_install()
        if install is None:
            raise ValueError(
                "VIAME install directory not found. Set VIAME_INSTALL environment variable."
            )

    script = setup_script(install)
    if not script.exists():
        raise FileNotFoundError(f"Setup script not found: {script}")

    if os.name == "nt":
        command = f"call {script} && set"
        executable = None
    else:
        command = f"source {script} && env"
        executable = "/bin/bash"

    result = subprocess.check_output(
        command, shell=True, executable=executable, text=True
    )

    env = os.environ.copy()
    for line in result.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            env[key] = value
    return env
