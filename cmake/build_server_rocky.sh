#!/bin/bash

# VIAME Rocky Linux Build Script

# Debugging, logging, and options
set -x

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

# Install system dependencies and use more recent compiler
install_system_deps yum

# Install more modern CMAKE and OpenSSL from source
install_openssl
install_cmake

# Install Node.js and yarn for DIVE desktop build
install_nodejs_and_yarn 18

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
echo "Beginning VIAME CMake configuration"

cmake ../ \
  -C ../cmake/build_cmake_base.cmake \
  -C ../cmake/build_cmake_desktop.cmake \
  -C ../cmake/build_cmake_linux.cmake \
  -DCUDA_TOOLKIT_ROOT_DIR:PATH=$CUDA_DIRECTORY \
  -DCUDA_NVCC_EXECUTABLE:PATH=$CUDA_DIRECTORY/bin/nvcc

# Build VIAME and setup libraries, pipe output to file
echo "Beginning core build, routing build info to build_log.txt"

run_build_and_setup_libraries "$CUDA_DIRECTORY" > build_log.txt 2>&1

# Verify build success and create tarball
if verify_build_success build_log.txt; then
  create_install_tarball "$VIAME_VERSION" "Linux-64Bit"
else
  exit 1
fi
