"""
Scikit-build requires configuration to be passed to it's setup function.
Support for setup.cfg is forthcoming (as of skbuild v0.12
"""

import os
import subprocess
from pathlib import Path
from setuptools import find_packages

try:
    from skbuild import setup
except ImportError:
    # skbuild is only needed for wheel builds, not for egg_info
    from setuptools import setup

SCRIPT_DIR = Path(__file__).parent
PACKAGE_SRC = "python"
PACKAGE_NAME = "kwiver"

with open(SCRIPT_DIR / "VERSION.txt", "r") as f:
    VERSION_FROM_FILE = f.read().strip()


def _get_version():
    # CI_COMMIT_TAG is set by GitLab CI on tag pipelines.
    if os.environ.get("CI_COMMIT_TAG"):
        return VERSION_FROM_FILE

    # Dev builds: append .dev{commit_count} so wheels are clearly non-release.
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        commit_count = result.stdout.strip()
        return f"{VERSION_FROM_FILE}.dev{commit_count}"
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: could not determine commit count ({e}), using dev0")
        return f"{VERSION_FROM_FILE}.dev0"


VERSION = _get_version()


with open(SCRIPT_DIR / "README.rst", "r") as f:
    LONG_DESCRIPTION = f.read()

# Set environment variable for CMake to pick up
# Adding f"-DKWIVER_VERSION={VERSION} to cmake_args didn't work
os.environ["KWIVER_WHEEL_VERSION"] = VERSION

# NumPy 2.0+ requires pybind11 2.12+ but CI currently uses 2.10.3
# Python 3.8: Safe - NumPy 2.0 dropped Python 3.8 support, pip installs 1.24.4
# Python 3.9+: Requires constraint to prevent NumPy 2.0 which causes segfaults
numpy_requirement = "numpy>=1.13.0,<2.0"

setup(
    # Basic Metadata ###########################################################
    name=PACKAGE_NAME,
    version=VERSION,
    description="Python and C++ toolkit that pulls together computer vision algorithms "
    " into highly modular run time configurable systems",
    long_description=LONG_DESCRIPTION,
    author="Kitware, Inc.",
    author_email="kwiver-developers@kitware.com",
    url="https://github.com/Kitware/kwiver",
    license="BSD 3-Clause",
    license_files=["LICENSE"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    platforms=[
        "linux",
        "Unix",
    ],
    # Options ##################################################################
    zip_safe=False,
    include_package_data=True,
    python_requires=">=3.8",
    # Package Specification ####################################################
    package_dir={"": PACKAGE_SRC},
    packages=find_packages(
        where=PACKAGE_SRC,
        include=[f"{PACKAGE_NAME}*"],
        # xxx(python-arrows) bring back once arrows are adapted to new API
        exclude=[
            f"{PACKAGE_NAME}.arrows*",
            f"{PACKAGE_NAME}.sprokit*",
        ],
    ),
    # Requirements #############################################################
    install_requires=[numpy_requirement],
    # extras_require=[],
    tests_require=["pytest"],
    # Entry-Points #############################################################
    entry_points={
        "kwiver.python_plugins": [
            "say=kwiver.vital.test_interface.python_say",
            "they_say=kwiver.vital.test_interface.python_they_say",
        ],
        "console_scripts": [
            "dump_klv=kwiver.tools.dump_klv:run",
        ],
    },
    # Scikit-Build Stuff #######################################################
    cmake_minimum_required_version="3.15",  # matches primary CMakeLists.txt req
    # Where build libraries and such will be installed into in order to be
    # within the package module space.
    cmake_install_dir=f"./{PACKAGE_SRC}/{PACKAGE_NAME}",
    cmake_args=[
        "-DKWIVER_ENABLE_PYTHON=ON",
        "-DPYBIND11_PYTHON_VERSION=3",
        "-DKWIVER_INSTALL_SET_UP_SCRIPT=OFF",
    ],
)
