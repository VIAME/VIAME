#!/bin/bash

# VIAME Docker Web CUDA 11 Build Script

# debugging flag
set -x

# Source utility scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/build_common_functions.sh"

# Install required system dependencies
install_deps_apt

# Install CMake
install_cmake

# Update VIAME sub git deps
update_git_submodules /viame
setup_build_directory /viame

# Add VIAME and CUDA paths to build
setup_basic_build_environment /viame/build/install /usr/local/cuda

# Configure VIAME using cache presets
cmake ../ \
  -C ../cmake/viame_cmake_base.cmake \
  -C ../cmake/viame_cmake_docker.cmake \
  -C ../cmake/viame_cmake_web.cmake \
  -DCUDA_ARCHITECTURES:STRING="7.0 7.5 8.0 8.6 8.9 9.0"

# Download OCV aux files from local server copy
download_opencv_extras

# Perform multi-threaded build
run_build build_log.txt true

# Below be krakens
# (V) (°,,,°) (V)   (V) (°,,,°) (V)   (V) (°,,,°) (V)

# Install old MMDet plugin useful for 1-2 models (this is typically
# packaged in add-ons but VIAME-web doesn't handle binary code in
# add-ons currently)
wget https://viame.kitware.com/api/v1/file/685cd1a5a2df48d3c1ae8604/download
tar -xvf download
cp -r lib install
rm -rf lib download

# Fix libsvm symlink issue
fix_libsvm_symlink install

# Finalize Docker install
finalize_docker_install /viame/build
