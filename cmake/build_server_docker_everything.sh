#!/bin/bash

# VIAME Docker Everything Build Script
# Builds with all features enabled

# debugging flag
set -x

# Source utility scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/build_common_functions.sh"

# Fletch, VIAME, CMAKE system deps
install_deps_apt

# Install CMAKE
install_cmake

# Update VIAME sub git deps
update_git_submodules /viame
setup_build_directory /viame

# Add VIAME and CUDA paths to build
setup_basic_build_environment /viame/build/install /usr/local/cuda

# Configure VIAME using cache presets with additional features
cmake ../ \
  -C ../cmake/build_cmake_base.cmake \
  -C ../cmake/build_cmake_docker.cmake \
  -DVIAME_ENABLE_BURNOUT:BOOL=ON \
  -DVIAME_ENABLE_ITK:BOOL=ON \
  -DVIAME_ENABLE_KWANT:BOOL=ON \
  -DVIAME_ENABLE_LEARN:BOOL=ON \
  -DVIAME_ENABLE_SCALLOP_TK:BOOL=ON \
  -DVIAME_ENABLE_TENSORFLOW:BOOL=ON \
  -DVIAME_ENABLE_UW_PREDICTOR:BOOL=ON \
  -DVIAME_ENABLE_WEB_EXCLUDES:BOOL=ON \
  -DVIAME_ENABLE_PYTORCH-PYSOT:BOOL=ON \
  -DVIAME_DOWNLOAD_MODELS:BOOL=ON \
  -DVIAME_DOWNLOAD_MODELS-PYTORCH:BOOL=ON \
  -DVIAME_DOWNLOAD_MODELS-ARCTIC-SEAL:BOOL=ON \
  -DVIAME_DOWNLOAD_MODELS-HABCAM:BOOL=ON \
  -DVIAME_DOWNLOAD_MODELS-MOUSS:BOOL=ON

# Perform multi-threaded build
run_build build_log.txt true

# Below be krakens
# (V) (°,,,°) (V)   (V) (°,,,°) (V)   (V) (°,,,°) (V)

# Fix libsvm symlink issue
fix_libsvm_symlink install

# Finalize Docker install
finalize_docker_install /viame/build
