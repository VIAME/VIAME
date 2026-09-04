#!/bin/bash

# VIAME Docker Web Build Script

# debugging flag
set -x

# Source utility scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/build_common_functions.sh"

# Install required system dependencies
install_system_deps apt

# Install CMake
install_cmake

# Update VIAME sub git deps
update_git_submodules /viame
setup_build_directory /viame

# Add VIAME and CUDA paths to build
setup_basic_build_environment /viame/build/install /usr/local/cuda

# Configure VIAME using cache presets
cmake ../ \
  -C ../cmake/build_cmake_base.cmake \
  -C ../cmake/build_cmake_docker.cmake \
  -C ../cmake/build_cmake_web.cmake \
  -DCUDA_ARCHITECTURES:STRING="7.0 7.5 8.0 8.6 8.9 9.0 10.0 12.0"

# Download OCV aux files from local server copy
download_opencv_extras

# Perform multi-threaded build
run_build build_log.txt true

# Below be krakens
# (V) (°,,,°) (V)   (V) (°,,,°) (V)   (V) (°,,,°) (V)

# The old MMDet plugin used by 1-2 models is no longer installed here. Its
# prebuilt tarball is a cp310 binary build (mmcv_depr/_ext.cpython-310-*.so
# under lib/python3.10), which this image's Python 3.12 can neither import nor
# even see on its path since the ubuntu24.04 base switch, so it was only
# shipping dead weight. The cu11 and ifremer web images still run Python 3.10
# and keep it.

# Fix libsvm symlink issue
fix_libsvm_symlink install

# Run the CRITICAL tests before finalize_docker_install, which moves the install
# tree to /opt/noaa/viame and deletes /viame, taking the ctest infrastructure
# with it -- after that point there is nothing left in the image to test.
CRITICAL_TESTS_STATUS=0
run_critical_tests /viame/build /viame/build/install || CRITICAL_TESTS_STATUS=$?

# Finalize Docker install
finalize_docker_install /viame/build

# Record the verdict inside the image rather than failing the RUN outright.
# A non-zero exit here aborts the docker build and BuildKit keeps nothing, so a
# single failing test throws away the whole multi-hour build and leaves nothing
# to reproduce it against. The tarball builds already take the other approach --
# see rename_tarball_broken, which keeps the artifact and marks it VIAME-BROKEN.
# Do the same here and let the driver script decide what to do with the tag.
if [ "$CRITICAL_TESTS_STATUS" -ne 0 ]; then
  echo "CRITICAL_TESTS_FAILED" > /opt/noaa/viame/CRITICAL_TESTS_FAILED
  echo "================================================================"
  echo "CRITICAL tests FAILED -- image is marked broken"
  echo "  /opt/noaa/viame/CRITICAL_TESTS_FAILED is present in this image"
  echo "================================================================"
else
  echo "All CRITICAL tests passed"
fi
