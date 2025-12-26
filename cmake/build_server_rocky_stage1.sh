#! /bin/bash

# Stage 1: Build fletch (foundational dependencies)
# This script builds the base fletch dependencies that rarely change and take the longest to compile.
# The output is packaged as an artifact for Stage 2 (pytorch) to continue from.

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
rm /usr/local/cuda
rm /usr/local/cuda-12
mv /usr/local/cuda-12.6 $CUDA_DIRECTORY

# Update VIAME sub git sources
echo "Checking out VIAME submodules"

cd $VIAME_SOURCE_DIR
git config --global --add safe.directory $VIAME_SOURCE_DIR
git submodule update --init --recursive
mkdir build
cd build

# Configure Paths [should be removed when no longer necessary by fletch]
export PATH=$VIAME_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$VIAME_INSTALL_DIR/lib:$VIAME_INSTALL_DIR/lib/python3.10:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$VIAME_INSTALL_DIR/include/python3.10:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$VIAME_INSTALL_DIR/include/python3.10:$CPLUS_INCLUDE_PATH

# Configure VIAME
echo "Beginning VIAME CMake configuration for Stage 1"

cmake ../ -DCMAKE_BUILD_TYPE:STRING=Release \
-DCUDA_TOOLKIT_ROOT_DIR:PATH=$CUDA_DIRECTORY \
-DCUDA_NVCC_EXECUTABLE:PATH=$CUDA_DIRECTORY/bin/nvcc \
-DVIAME_BUILD_DEPENDENCIES:BOOL=ON \
-DVIAME_FIXUP_BUNDLE:BOOL=ON \
-DVIAME_ENABLE_BURNOUT:BOOL=OFF \
-DVIAME_ENABLE_CUDA:BOOL=ON \
-DVIAME_ENABLE_CUDNN:BOOL=ON \
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
-DVIAME_ENABLE_LEARN:BOOL=ON \
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
-DVIAME_ENABLE_PYTORCH-PYSOT:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-SAM:BOOL=ON \
-DVIAME_ENABLE_PYTORCH-ULTRALYTICS:BOOL=OFF \
-DVIAME_ENABLE_SCALLOP_TK:BOOL=OFF \
-DVIAME_ENABLE_SEAL:BOOL=OFF \
-DVIAME_ENABLE_SMQTK:BOOL=ON \
-DVIAME_ENABLE_TENSORFLOW:BOOL=OFF \
-DVIAME_ENABLE_UW_PREDICTOR:BOOL=OFF \
-DVIAME_ENABLE_VIVIA:BOOL=ON \
-DVIAME_ENABLE_VXL:BOOL=ON

# Build Stage 1 target: fletch
echo "Beginning Stage 1 build (fletch), routing build info to build_log_stage1.txt"

# Build fletch (all external dependencies)
echo "Building fletch..."
make -j$(nproc) fletch 2>&1 | tee build_log_stage1.txt

# Verify Stage 1 completed
if grep -q "Built target fletch" build_log_stage1.txt; then
  echo "Stage 1: fletch build succeeded"
else
  echo "Stage 1: fletch build FAILED"
  exit 1
fi

echo "Stage 1 build completed successfully"
