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

# Gate the image on the CRITICAL tests. This has to happen before
# finalize_docker_install, which moves the install tree to /opt/noaa/viame and
# deletes /viame, taking the ctest infrastructure with it -- after that point
# there is nothing left in the image to test against.
if ! run_critical_tests /viame/build /viame/build/install; then
  echo "CRITICAL tests failed, refusing to finalize the image"
  exit 1
fi

# Finalize Docker install
finalize_docker_install /viame/build
