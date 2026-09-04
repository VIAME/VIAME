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

# Must run before finalize_docker_install removes /viame and the ctest tree.
# Only worth running if there is a GPU here: docker build gets the daemon's
# default runtime, normally runc, so pipelines fall back to the CPU and the
# heavy ones time out on an image that is fine. measure_via_default_fish takes
# 52s on a GPU and does not finish in 540s without one. Gate from outside the
# build instead, where --gpus can be passed.
CRITICAL_TESTS_STATUS=0
if nvidia-smi -L >/dev/null 2>&1; then
  run_critical_tests /viame/build /viame/build/install || CRITICAL_TESTS_STATUS=$?
else
  echo "No GPU visible to the build, skipping CRITICAL tests"
  CRITICAL_TESTS_SKIPPED=1
fi

# Finalize Docker install
finalize_docker_install /viame/build

# Mark the image rather than fail the RUN, which would discard the whole build
if [ -n "${CRITICAL_TESTS_SKIPPED:-}" ]; then
  echo "CRITICAL_TESTS_SKIPPED" > /opt/noaa/viame/CRITICAL_TESTS_SKIPPED
  echo "CRITICAL tests skipped, validate this image with --gpus"
elif [ "$CRITICAL_TESTS_STATUS" -ne 0 ]; then
  echo "CRITICAL_TESTS_FAILED" > /opt/noaa/viame/CRITICAL_TESTS_FAILED
  echo "================================================================"
  echo "CRITICAL tests FAILED -- image is marked broken"
  echo "  /opt/noaa/viame/CRITICAL_TESTS_FAILED is present in this image"
  echo "================================================================"
else
  echo "All CRITICAL tests passed"
fi
