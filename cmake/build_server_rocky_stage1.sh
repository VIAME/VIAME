#!/bin/bash

# Stage 1: Build fletch and pytorch (foundational dependencies)
# This script builds the base dependencies that rarely change and take the longest to compile.
# The output is packaged as an artifact for Stage 2 to continue from.

set -x
set -e

# Source utility scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/build_common_functions.sh"

export VIAME_SOURCE_DIR=/viame

# Extract version from RELEASE_NOTES.md
extract_viame_version $VIAME_SOURCE_DIR
export VIAME_BUILD_DIR=$VIAME_SOURCE_DIR/build
export VIAME_INSTALL_DIR=$VIAME_BUILD_DIR/install

export CUDA_DIRECTORY=/usr/local/cuda-viame
export CUDNN_DIRECTORY=/usr

# Install system dependencies
install_system_deps yum

# Install more modern CMAKE and OpenSSL from source
install_openssl
install_cmake

# Patch CUDNN when required
patch_cudnn_headers

# Use GCC11 for build (Rocky 9 has GCC 11 by default, Rocky 8 needs toolset)
setup_gcc_toolset 11

# Hack for storing paths to CUDA libs for some libraries
rm /usr/local/cuda
rm /usr/local/cuda-12
mv /usr/local/cuda-12.6 $CUDA_DIRECTORY

# Update VIAME sub git sources
update_git_submodules $VIAME_SOURCE_DIR
setup_build_directory $VIAME_SOURCE_DIR

# Configure Paths [should be removed when no longer necessary by fletch]
setup_build_environment $VIAME_INSTALL_DIR "" "3.10"

# Configure VIAME using cache presets
echo "Beginning VIAME CMake configuration for Stage 1"

cmake ../ \
  -C ../cmake/build_cmake_base.cmake \
  -C ../cmake/build_cmake_desktop.cmake \
  -DCUDA_TOOLKIT_ROOT_DIR:PATH=$CUDA_DIRECTORY \
  -DCUDA_NVCC_EXECUTABLE:PATH=$CUDA_DIRECTORY/bin/nvcc \
  -DVIAME_ENABLE_PYTORCH-ULTRALYTICS:BOOL=OFF

# Build Stage 1 targets: fletch and pytorch
echo "Beginning Stage 1 build (fletch + pytorch), routing build info to build_log_stage1.txt"

# Build fletch first (all external dependencies)
echo "Building fletch..."
make -j$(nproc) fletch 2>&1 | tee build_log_stage1.txt

# Build pytorch (requires fletch)
echo "Building pytorch..."
make -j$(nproc) pytorch 2>&1 | tee -a build_log_stage1.txt

# Verify Stage 1 completed
if grep -q "Built target fletch" build_log_stage1.txt; then
  echo "Stage 1: fletch build succeeded"
else
  echo "Stage 1: fletch build FAILED"
  exit 1
fi

if grep -q "Built target pytorch" build_log_stage1.txt; then
  echo "Stage 1: pytorch build succeeded"
else
  echo "Stage 1: pytorch build FAILED"
  exit 1
fi

echo "Stage 1 build completed successfully"
