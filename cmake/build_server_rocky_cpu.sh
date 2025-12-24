#!/bin/bash

# VIAME Rocky Linux CPU-Only Build Script

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

# Install system dependencies and use more recent compiler
install_system_deps yum

# Install more modern CMAKE and OpenSSL from source
install_openssl
install_cmake

# Use GCC11 for build
yum install -y gcc-toolset-11
source /opt/rh/gcc-toolset-11/enable

# Update VIAME sub git sources
update_git_submodules $VIAME_SOURCE_DIR
setup_build_directory $VIAME_SOURCE_DIR

# Configure Paths [should be removed when no longer necessary by fletch]
setup_build_environment $VIAME_INSTALL_DIR "" "3.10"

# Configure VIAME
cmake ../ -DCMAKE_BUILD_TYPE:STRING=Release \
-DVIAME_BUILD_DEPENDENCIES:BOOL=ON \
-DVIAME_FIXUP_BUNDLE:BOOL=ON \
-DVIAME_ENABLE_BURNOUT:BOOL=OFF \
-DVIAME_ENABLE_CUDA:BOOL=OFF \
-DVIAME_ENABLE_CUDNN:BOOL=OFF \
-DVIAME_ENABLE_DARKNET:BOOL=ON \
-DVIAME_ENABLE_DIVE:BOOL=ON \
-DVIAME_ENABLE_DOCS:BOOL=OFF \
-DVIAME_ENABLE_FFMPEG:BOOL=ON \
-DVIAME_ENABLE_FFMPEG-X264:BOOL=ON \
-DVIAME_ENABLE_GDAL:BOOL=OFF \
-DVIAME_ENABLE_FLASK:BOOL=OFF \
-DVIAME_ENABLE_ITK:BOOL=OFF \
-DVIAME_ENABLE_KWANT:BOOL=ON \
-DVIAME_ENABLE_KWIVER:BOOL=ON \
-DVIAME_ENABLE_LEARN:BOOL=OFF \
-DVIAME_ENABLE_MATLAB:BOOL=OFF \
-DVIAME_ENABLE_ONNX:BOOL=ON \
-DVIAME_ENABLE_OPENCV:BOOL=ON \
-DVIAME_OPENCV_VERSION:STRING=3.4.0 \
-DVIAME_ENABLE_POSTGRESQL=ON \
-DVIAME_ENABLE_PYTHON:BOOL=ON \
-DVIAME_PYTHON_BUILD_FROM_SOURCE:BOOL=ON \
-DVIAME_PYTHON_VERSION:STRING=3.10.4 \
-DVIAME_ENABLE_PYTORCH:BOOL=ON \
-DVIAME_PYTORCH_BUILD_FROM_SOURCE:BOOL=ON \
-DVIAME_PYTORCH_DISABLE_NINJA=OFF \
-DVIAME_PYTORCH_VERSION:STRING=2.7.1 \
-DVIAME_ENABLE_PYTORCH-MMDET:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-NETHARN:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-PYSOT:BOOL=OFF \
-DVIAME_ENABLE_PYTORCH-SAM:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-ULTRALYTICS:BOOL=OFF \
-DVIAME_ENABLE_SCALLOP_TK:BOOL=OFF \
-DVIAME_ENABLE_SEAL:BOOL=OFF \
-DVIAME_ENABLE_SMQTK:BOOL=ON \
-DVIAME_ENABLE_TENSORFLOW:BOOL=OFF \
-DVIAME_ENABLE_UW_PREDICTOR:BOOL=OFF \
-DVIAME_ENABLE_VIVIA:BOOL=ON \
-DVIAME_ENABLE_VXL:BOOL=ON

# Download OCV aux files from local server copy
#./viame/cmake/build_server_linux_ocv_extra.sh

# Build VIAME and setup libraries, pipe output to file
run_build_and_setup_libraries > build_log.txt 2>&1

# Verify build success and create tarball
if verify_build_success build_log.txt; then
  create_install_tarball "CPU-$VIAME_VERSION" "Linux-64Bit"
else
  exit 1
fi
