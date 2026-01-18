#!/bin/bash

# VIAME Docker Default Build Script

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
  -C ../cmake/build_cmake_base.cmake \
  -C ../cmake/build_cmake_docker.cmake \
  -DVIAME_ENABLE_PYTORCH-LEARN:BOOL=ON \
  -DVIAME_ENABLE_ONNX:BOOL=ON \
  -DVIAME_BUILD_TORCHVISION_FROM_SOURCE=ON \
  -DVIAME_ENABLE_PYTORCH-VISION:BOOL=ON \
  -DVIAME_ENABLE_PYTORCH-SIAMMASK:BOOL=ON \
  -DVIAME_ENABLE_PYTORCH-SAM2:BOOL=ON \
  -DVIAME_DOWNLOAD_MODELS:BOOL=ON \
  -DVIAME_DOWNLOAD_MODELS-GENERIC:BOOL=ON \
  -DVIAME_DOWNLOAD_MODELS-FISH:BOOL=ON \
  -DVIAME_DOWNLOAD_MODELS-PYSOT:BOOL=ON

# Download OCV aux files from local server copy
download_opencv_extras

# Perform multi-threaded build
run_build build_log.txt true

# Below be krakens
# (V) (°,,,°) (V)   (V) (°,,,°) (V)   (V) (°,,,°) (V)

# Fix libsvm symlink issue
fix_libsvm_symlink install

# Finalize Docker install
finalize_docker_install /viame/build
