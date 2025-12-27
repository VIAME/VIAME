#!/bin/bash

# Stage 2: Build pytorch
# This script continues from the Stage 1 artifact which contains fletch builds.
# It builds pytorch and packages the result for Stage 3 to continue from.

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
rm -f /usr/local/cuda
rm -f /usr/local/cuda-12
mv /usr/local/cuda-12.6 $CUDA_DIRECTORY

# Update VIAME sub git sources (needed for source files not in Stage 1 artifact)
update_git_submodules $VIAME_SOURCE_DIR

# Build directory should already exist from Stage 1 artifact
cd $VIAME_BUILD_DIR

# Configure Paths [should be removed when no longer necessary by fletch]
setup_build_environment $VIAME_INSTALL_DIR "" "3.10"

# Build Stage 2 target: pytorch
echo "Beginning Stage 2 build (pytorch), routing build info to build_log_stage2.txt"

# Build pytorch (requires fletch from Stage 1)
echo "Building pytorch..."
make -j$(nproc) pytorch 2>&1 | tee build_log_stage2.txt

# Verify Stage 2 completed
if grep -q "Built target pytorch" build_log_stage2.txt; then
  echo "Stage 2: pytorch build succeeded"
else
  echo "Stage 2: pytorch build FAILED"
  exit 1
fi

echo "Stage 2 build completed successfully"
