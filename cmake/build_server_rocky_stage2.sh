#! /bin/bash

# Stage 2: Build pytorch
# This script continues from the Stage 1 artifact which contains fletch builds.
# It builds pytorch and packages the result for Stage 3 to continue from.

set -x
set -e

export VIAME_SOURCE_DIR=/viame

# Extract version from RELEASE_NOTES.md (first token of first line)
export VIAME_VERSION=$(head -n 1 $VIAME_SOURCE_DIR/RELEASE_NOTES.md | awk '{print $1}')
export VIAME_BUILD_DIR=$VIAME_SOURCE_DIR/build
export VIAME_INSTALL_DIR=$VIAME_BUILD_DIR/install

export CUDA_DIRECTORY=/usr/local/cuda-viame
export CUDNN_DIRECTORY=/usr

# Install system dependencies and use more recent compiler
$VIAME_SOURCE_DIR/cmake/build_server_deps_yum.sh

# Install more modern CMAKE and OpenSSL from source
./viame/cmake/build_server_linux_ssl.sh
./viame/cmake/build_server_linux_cmake.sh

# Patch CUDNN when required
./viame/cmake/build_server_patch_cudnn.sh

# Use GCC11 for build
yum install -y gcc-toolset-11
source /opt/rh/gcc-toolset-11/enable

# Hack for storing paths to CUDA libs for some libraries
rm -f /usr/local/cuda
rm -f /usr/local/cuda-12
mv /usr/local/cuda-12.6 $CUDA_DIRECTORY

# Update VIAME sub git sources (needed for source files not in Stage 1 artifact)
echo "Checking out VIAME submodules"

cd $VIAME_SOURCE_DIR
git config --global --add safe.directory $VIAME_SOURCE_DIR
git submodule update --init --recursive

# Build directory should already exist from Stage 1 artifact
cd build

# Configure Paths [should be removed when no longer necessary by fletch]
export PATH=$VIAME_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$VIAME_INSTALL_DIR/lib:$VIAME_INSTALL_DIR/lib/python3.10:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$VIAME_INSTALL_DIR/include/python3.10:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$VIAME_INSTALL_DIR/include/python3.10:$CPLUS_INCLUDE_PATH

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
