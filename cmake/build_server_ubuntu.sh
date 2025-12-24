#!/bin/bash

# VIAME Ubuntu Build Script

# debugging flag
set -x

# Source utility scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/build_common_functions.sh"

# Extract version from RELEASE_NOTES.md
extract_viame_version /viame

# System Deps
install_system_deps apt

# Install CMAKE
install_cmake

# Update VIAME sub git deps
update_git_submodules /viame
setup_build_directory /viame

# Configure Paths [should be removed when no longer necessary by fletch]
setup_basic_build_environment /viame/build/install

source ./viame/cmake/linux_add_internal_py36_paths.sh

# Configure VIAME
cmake ../ -DCMAKE_BUILD_TYPE:STRING=Release \
-DVIAME_BUILD_DEPENDENCIES:BOOL=ON \
-DVIAME_FIXUP_BUNDLE:BOOL=ON \
-DVIAME_ENABLE_BURNOUT:BOOL=OFF \
-DVIAME_ENABLE_CUDA:BOOL=ON \
-DVIAME_ENABLE_CUDNN:BOOL=ON \
-DVIAME_ENABLE_DIVE:BOOL=ON \
-DVIAME_ENABLE_DOCS:BOOL=OFF \
-DVIAME_ENABLE_FFMPEG:BOOL=ON \
-DVIAME_ENABLE_FFMPEG-X264:BOOL=ON \
-DVIAME_ENABLE_GDAL:BOOL=ON \
-DVIAME_ENABLE_FLASK:BOOL=OFF \
-DVIAME_ENABLE_ITK:BOOL=OFF \
-DVIAME_ENABLE_KWANT:BOOL=ON \
-DVIAME_ENABLE_KWIVER:BOOL=ON \
-DVIAME_ENABLE_MATLAB:BOOL=OFF \
-DVIAME_ENABLE_OPENCV:BOOL=ON \
-DVIAME_OPENCV_VERSION:STRING=3.4.0 \
-DVIAME_ENABLE_PYTHON:BOOL=ON \
-DVIAME_PYTHON_BUILD_FROM_SOURCE:BOOL=ON \
-DVIAME_PYTHON_VERSION:STRING=3.10.4 \
-DVIAME_ENABLE_PYTORCH:BOOL=ON \
-DVIAME_PYTORCH_BUILD_FROM_SOURCE:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-MMDET:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-NETHARN:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-PYSOT:BOOL=ON \
-DVIAME_ENABLE_SCALLOP_TK:BOOL=OFF \
-DVIAME_ENABLE_SEAL:BOOL=OFF \
-DVIAME_ENABLE_SMQTK:BOOL=ON \
-DVIAME_ENABLE_TENSORFLOW:BOOL=OFF \
-DVIAME_ENABLE_UW_PREDICTOR:BOOL=OFF \
-DVIAME_ENABLE_VIVIA:BOOL=ON \
-DVIAME_ENABLE_VXL:BOOL=ON \
-DVIAME_ENABLE_DARKNET:BOOL=ON

# Build VIAME, pipe output to file
run_build build_log.txt

# Verify build success
if verify_build_success build_log.txt; then
  # Make tarball of install
  create_install_tarball "$VIAME_VERSION" "Ubuntu-64Bit"
else
  exit 1
fi
