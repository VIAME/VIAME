#!/bin/bash

# Stage 3: Build remaining components (kwiver, pytorch-libs, viame)
# This script continues from the Stage 2 artifact which contains fletch and pytorch builds.
# It builds all remaining components and creates the final release package.

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

# Update VIAME sub git sources (needed for source files not in Stage 2 artifact)
update_git_submodules $VIAME_SOURCE_DIR

# Build directory should already exist from Stage 2 artifact
cd $VIAME_BUILD_DIR

# Configure Paths [should be removed when no longer necessary by fletch]
setup_build_environment $VIAME_INSTALL_DIR "" "3.10"

# Enable DIVE for Stage 3 (download prebuilt binaries, don't build from source)
echo "Enabling DIVE for Stage 3..."
cmake ../ -DVIAME_ENABLE_DIVE:BOOL=ON -DVIAME_BUILD_DIVE_FROM_SOURCE:BOOL=OFF

# Build Stage 3: Everything else (kwiver, pytorch-libs, viame)
echo "Beginning Stage 3 build (kwiver, pytorch-libs, viame), routing build info to build_log_stage3.txt"

# Run full build and setup libraries - CMake will skip already-built targets from Stage 1 and 2
run_build_and_setup_libraries "$CUDA_DIRECTORY" > build_log_stage3.txt 2>&1

# Verify build success and create tarball
if verify_build_success build_log_stage3.txt; then
  create_install_tarball "$VIAME_VERSION" "Linux-64Bit"
  echo "Stage 3 build completed successfully - Final release package created"
else
  exit 1
fi
